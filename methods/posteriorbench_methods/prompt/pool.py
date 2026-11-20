"""The pool file of candidate system prompts, and the order of a task's candidates in a replicate.

A pool file is a parquet table with the columns COLUMNS, and optionally `batch` and `scenario`
(README, "Run"). A replicate orders the candidates at random within each batch, batches in
increasing order; its pool of size K is the first K, so the pools of a replicate are nested.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from posteriorbench_methods.common import errors

COLUMNS = ("task_id", "candidate_index", "candidate_id", "system_prompt")
BATCH_COLUMN = "batch"
SCENARIO_COLUMN = "scenario"
# The columns of a file that `add_candidates` creates. In a file that exists, the columns it
# shares with these keep their types.
WRITTEN_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("candidate_index", pa.int64()),
        ("candidate_id", pa.string()),
        ("system_prompt", pa.string()),
        (SCENARIO_COLUMN, pa.string()),
        (BATCH_COLUMN, pa.int32()),
    ]
)


class PoolError(errors.MethodsError, ValueError):
    """A pool file with a missing column, no candidate for the task, indices that are not
    0..n-1, a batch that is not a non-negative integer, or fewer candidates than a pool size."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One candidate; `index` is its candidate_index, `batch` is 0 in a file without batches."""

    index: int
    candidate_id: str
    system_prompt: str
    batch: int = 0


@dataclasses.dataclass(frozen=True)
class Written:
    """A candidate not yet in a pool file, with the scenario it was written from."""

    system_prompt: str
    scenario: str


def pool_file(path: pathlib.Path | None) -> pathlib.Path:
    """The run's --method-data; raises PoolError if it was not given."""
    if path is None:
        raise PoolError("prompt selection needs the file of candidate prompts (--method-data)")
    return path


def load_candidates(path: pathlib.Path, task_id: str) -> list[Candidate]:
    """The candidates of `task_id` in index order. Raises PoolError if the file cannot be used,
    including when it has no candidate for the task."""
    candidates = _read_candidates(path, task_id)
    if not candidates:
        raise PoolError(f"{path} has no candidate for the task {task_id}")
    return candidates


def existing_candidates(path: pathlib.Path, task_id: str) -> list[Candidate]:
    """Like `load_candidates`, but empty if the file or the task's rows do not exist."""
    return _read_candidates(path, task_id) if path.exists() else []


def _read_candidates(path: pathlib.Path, task_id: str) -> list[Candidate]:
    """Like `load_candidates`, but empty if the task has no rows."""
    names = pq.read_schema(path).names
    missing = sorted(set(COLUMNS) - set(names))
    if missing:
        raise PoolError(
            f"{path} has no column {', '.join(missing)}; a pool file has the columns "
            f"{', '.join(COLUMNS)}"
        )
    columns = [*COLUMNS, BATCH_COLUMN] if BATCH_COLUMN in names else list(COLUMNS)
    table = pq.read_table(path, columns=columns, filters=[("task_id", "=", task_id)])
    rows: list[dict[str, Any]] = table.to_pylist()
    rows.sort(key=lambda row: int(row["candidate_index"]))
    if [int(row["candidate_index"]) for row in rows] != list(range(len(rows))):
        raise PoolError(f"the candidate indices of {task_id} in {path} are not 0..n-1")
    batches = [row.get(BATCH_COLUMN, 0) for row in rows]
    if any(isinstance(b, bool) or not isinstance(b, int) or b < 0 for b in batches):
        raise PoolError(f"the batches of {task_id} in {path} are not non-negative integers")
    return [
        Candidate(
            int(row["candidate_index"]), str(row["candidate_id"]), str(row["system_prompt"]), batch
        )
        for row, batch in zip(rows, batches, strict=True)
    ]


def candidate_name(system_prompt: str) -> str:
    """The `candidate_id` that `add_candidates` gives: derived from the text alone, so a text
    has the same name in every file."""
    return hashlib.sha1(system_prompt.encode()).hexdigest()[:12]


def add_candidates(
    path: pathlib.Path, task_id: str, written: Sequence[Written], *, new_batch: bool
) -> list[Candidate]:
    """Append `written` to the candidates of `task_id` in `path`; return them as stored.

    New candidates take the next indices, in order, and join the task's last batch, or with
    `new_batch` the batch after it, so a replicate keeps the earlier candidates where they were.
    A task without candidates starts at batch 0 either way.

    Rewrites the file atomically, creating it and its directory if needed. Rows of other tasks
    and other columns are kept (null in the new rows; old rows that lacked `scenario` and
    `batch` get null and 0). Writes nothing if `written` is empty."""
    existing = existing_candidates(path, task_id)
    if not written:
        return []
    last_batch = max((candidate.batch for candidate in existing), default=0)
    batch = last_batch + 1 if new_batch and existing else last_batch
    added = [
        Candidate(
            len(existing) + offset, candidate_name(new.system_prompt), new.system_prompt, batch
        )
        for offset, new in enumerate(written)
    ]
    schema = _schema_with_written_columns(pq.read_schema(path) if path.exists() else None)
    rows = [
        {
            "task_id": task_id,
            "candidate_index": candidate.index,
            "candidate_id": candidate.candidate_id,
            "system_prompt": candidate.system_prompt,
            SCENARIO_COLUMN: new.scenario,
            BATCH_COLUMN: candidate.batch,
        }
        for candidate, new in zip(added, written, strict=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with pq.ParquetWriter(partial, schema) as writer:
        if path.exists():
            source = pq.ParquetFile(path)
            for group in range(source.num_row_groups):
                writer.write_table(_with_written_columns(source.read_row_group(group), schema))
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    os.replace(partial, path)
    return added


def _schema_with_written_columns(existing: Any) -> Any:
    """The columns of `existing` (the file's schema, None if no file) with their types, then
    the WRITTEN_SCHEMA columns it lacks."""
    if existing is None:
        return WRITTEN_SCHEMA
    fields = [pa.field(field.name, field.type) for field in existing]
    fields += [field for field in WRITTEN_SCHEMA if field.name not in existing.names]
    return pa.schema(fields)


def _with_written_columns(table: Any, schema: Any) -> Any:
    """Existing rows `table` cast to `schema`; a missing `scenario` is null, a missing `batch` 0."""
    for field in schema:
        if field.name in table.column_names:
            continue
        if field.name == BATCH_COLUMN:
            column = pa.array([0] * table.num_rows, type=field.type)
        else:
            column = pa.nulls(table.num_rows, type=field.type)
        table = table.append_column(field, column)
    return table.select(schema.names).cast(schema)


def replicate_order(count: int, seed: int) -> list[int]:
    """A uniformly random permutation of range(count) from `np.random.default_rng(seed)`."""
    return [int(index) for index in np.random.default_rng(seed).permutation(count)]


def batched_order(batches: Sequence[int], seed_of_batch: Callable[[int], int]) -> list[int]:
    """A replicate's order of the candidates, as indices into `batches` (each candidate's
    batch): batches in increasing order, each in `replicate_order` with `seed_of_batch(batch)`."""
    order: list[int] = []
    for batch in sorted(set(batches)):
        members = [index for index, value in enumerate(batches) if value == batch]
        order += [members[at] for at in replicate_order(len(members), seed_of_batch(batch))]
    return order
