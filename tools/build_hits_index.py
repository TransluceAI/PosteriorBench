"""Build the hit files of one prompt from its corpus.

For one prompt, `build_and_write_hits_index` reads the prompt's corpus once and writes, for each
scored pair of that prompt, `hits/<prompt_id>/<behavior>.parquet`: the pair's hits in corpus order
(files `part-*` ascending, rows in file order) without its references, with the columns of
`posteriorbench.data.HITS_SCHEMA`

    row: int32                    the hit's row in the prompt's corpus
    token_ids: list<int32>        the response
    reference_duplicate: bool     the token ids equal some reference's

and the key-value metadata `task_id`, `hit_count`, `references_removed`, `reference_duplicates`,
`reference_duplicate_fraction`, `references_with_duplicate` and `builder`. zstd, row groups of
100,000 rows.

The references to leave out are read from the evaluation file, so this is data-building code and
never part of a method.

    python tools/build_hits_index.py --corpus DIR --data DIR --prompt-id wc_0001 --out DIR
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer

import posteriorbench
from posteriorbench import data

ROW_GROUP_SIZE = 100_000
# How a hit table is written: `pq.write_table(table, where, **WRITE_OPTIONS)`.
WRITE_OPTIONS: dict[str, Any] = {"compression": "zstd", "row_group_size": ROW_GROUP_SIZE}


class HitsIndexError(ValueError):
    """The corpus disagrees with the task table (hit counts, references)."""


@dataclasses.dataclass(frozen=True)
class HitsIndexReport:
    """What was written for one task, with the checks' numbers."""

    task_id: str
    hit_count: int
    rows: int
    references_removed: int
    reference_duplicates: int
    reference_duplicate_fraction: float
    references_with_duplicate: int
    path: str


@dataclasses.dataclass
class _TaskBuffer:
    """Hits of one task collected while streaming the corpus."""

    behavior: str
    reference_rows: set[int]
    reference_row_array: np.ndarray[Any, np.dtype[np.int64]]
    reference_token_ids: dict[tuple[int, ...], list[int]]
    rows: list[np.ndarray[Any, np.dtype[np.int64]]] = dataclasses.field(default_factory=list)
    token_ids: list[pa.Array] = dataclasses.field(default_factory=list)
    duplicates: list[bool] = dataclasses.field(default_factory=list)
    references_seen: set[int] = dataclasses.field(default_factory=set)
    duplicated_references: set[int] = dataclasses.field(default_factory=set)


def hit_table(
    task: data.Task,
    *,
    rows: Sequence[int] | np.ndarray[Any, Any],
    token_ids: Any,
    duplicates: Sequence[bool] | np.ndarray[Any, Any],
    references_removed: int,
    references_with_duplicate: int,
) -> Any:
    """The pyarrow table of one task's hit file (`data.HITS_SCHEMA`) with its key-value metadata.
    `rows`, `token_ids` (a pyarrow array of int32 lists) and `duplicates` list the task's hits in
    corpus order, without its references. Raises HitsIndexError unless they number
    hit_count - references_removed."""
    table = pa.table(
        {
            "row": pa.array(rows, type=pa.int32()),
            "token_ids": token_ids,
            "reference_duplicate": pa.array(duplicates, type=pa.bool_()),
        },
        schema=data.HITS_SCHEMA,
    )
    if table.num_rows != task.hit_count - references_removed:
        raise HitsIndexError(
            f"{task.task_id}: {table.num_rows + references_removed} hits in the corpus, the task "
            f"table says {task.hit_count}"
        )
    duplicate_count = int(np.count_nonzero(np.asarray(duplicates, dtype=bool)))
    metadata = {
        "task_id": task.task_id,
        "hit_count": str(task.hit_count),
        "references_removed": str(references_removed),
        "reference_duplicates": str(duplicate_count),
        "reference_duplicate_fraction": repr(
            duplicate_count / table.num_rows if table.num_rows else 0.0
        ),
        "references_with_duplicate": str(references_with_duplicate),
        "builder": f"posteriorbench {posteriorbench.__version__}",
    }
    return table.replace_schema_metadata(metadata)


def write_hit_table(table: Any, path: pathlib.Path) -> None:
    """Write a hit table to `path` (zstd, row groups of ROW_GROUP_SIZE rows), creating its
    directory. The file is written under a temporary name and renamed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, **WRITE_OPTIONS)
    os.replace(temporary, path)


def _add_batch(buffer: _TaskBuffer, batch: Any) -> None:
    """Keep the hits of one task in `batch` (a pyarrow RecordBatch of corpus rows), minus its
    reference rows, with duplicate flags."""
    labels = batch.column(buffer.behavior).to_numpy(zero_copy_only=False)
    # A null label is not a hit.
    hits = batch.filter(pa.array(np.asarray(labels == True, dtype=bool)))
    if hits.num_rows == 0:
        return
    rows = hits.column("row").to_numpy(zero_copy_only=False).astype(np.int64)
    is_reference = np.isin(rows, buffer.reference_row_array)
    buffer.references_seen.update(int(r) for r in rows[is_reference])
    keep = ~is_reference
    kept = hits.filter(pa.array(keep))
    token_ids = kept.column("token_ids").cast(pa.list_(pa.int32()))
    buffer.rows.append(rows[keep])
    buffer.token_ids.append(token_ids)
    for ids in token_ids.to_pylist():
        indices = buffer.reference_token_ids.get(tuple(ids))
        buffer.duplicates.append(indices is not None)
        if indices is not None:
            buffer.duplicated_references.update(indices)


def _write_task(buffer: _TaskBuffer, task: data.Task, out_dir: pathlib.Path) -> HitsIndexReport:
    missing = buffer.reference_rows - buffer.references_seen
    if missing:
        raise HitsIndexError(
            f"{task.task_id}: {len(missing)} reference rows are not hits in the corpus "
            f"(e.g. {sorted(missing)[:3]})"
        )
    rows = np.concatenate(buffer.rows) if buffer.rows else np.zeros(0, dtype=np.int64)
    token_ids = (
        pa.concat_arrays(buffer.token_ids)
        if buffer.token_ids
        else pa.array([], data.HITS_SCHEMA.field("token_ids").type)
    )
    table = hit_table(
        task,
        rows=rows,
        token_ids=token_ids,
        duplicates=buffer.duplicates,
        references_removed=len(buffer.reference_rows),
        references_with_duplicate=len(buffer.duplicated_references),
    )
    path = out_dir / data.hits_file(task.prompt_id, task.behavior)
    write_hit_table(table, path)
    duplicates = int(sum(buffer.duplicates))
    return HitsIndexReport(
        task_id=task.task_id,
        hit_count=task.hit_count,
        rows=table.num_rows,
        references_removed=len(buffer.reference_rows),
        reference_duplicates=duplicates,
        reference_duplicate_fraction=duplicates / table.num_rows if table.num_rows else 0.0,
        references_with_duplicate=len(buffer.duplicated_references),
        path=str(path),
    )


def build_and_write_hits_index(
    corpus_dir: str | os.PathLike[str],
    data_source: str | os.PathLike[str],
    prompt_id: str,
    out_dir: str | os.PathLike[str],
    *,
    batch_size: int = ROW_GROUP_SIZE,
) -> list[HitsIndexReport]:
    """Read `<corpus_dir>/corpus/<prompt_id>/part-*.parquet` once and write the hit file of every
    scored pair of that prompt under `out_dir`. `data_source` is a data source holding the task
    table and the evaluation file, from which the references are read.

    Checks, per pair: every reference row is a hit of the pair in the corpus; the hits left after
    removing the references number hit_count minus the number of references. Raises
    HitsIndexError otherwise (nothing is written for that pair or the later ones)."""
    buffers: dict[str, tuple[data.Task, _TaskBuffer]] = {}
    for task in data.load_tasks(data_source):
        if task.prompt_id != prompt_id or not task.scored:
            continue
        references = data.load_evaluation_data(data_source, task.task_id).references
        reference_token_ids: dict[tuple[int, ...], list[int]] = {}
        for index, reference in enumerate(references):
            reference_token_ids.setdefault(reference.token_ids, []).append(index)
        reference_rows = {reference.row for reference in references}
        buffers[task.behavior] = (
            task,
            _TaskBuffer(
                behavior=task.behavior,
                reference_rows=reference_rows,
                reference_row_array=np.array(sorted(reference_rows), dtype=np.int64),
                reference_token_ids=reference_token_ids,
            ),
        )
    if not buffers:
        raise HitsIndexError(f"no scored pair of prompt {prompt_id}")
    columns = ["row", "token_ids", *sorted(buffers)]
    for file in data.corpus_files(corpus_dir, prompt_id):
        handle = pq.ParquetFile(file)
        for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
            for _, buffer in buffers.values():
                _add_batch(buffer, batch)
    return [
        _write_task(buffer, task, pathlib.Path(out_dir))
        for _, (task, buffer) in sorted(buffers.items())
    ]


def main(
    corpus_dir: Annotated[
        pathlib.Path, typer.Option("--corpus", help="Directory holding corpus/<prompt_id>/")
    ],
    data_dir: Annotated[
        pathlib.Path,
        typer.Option(
            "--data",
            help="Dataset directory holding the task table and the evaluation file (the references "
            "to leave out are read from the evaluation file)",
        ),
    ],
    prompt_id: Annotated[str, typer.Option()],
    out: Annotated[pathlib.Path, typer.Option(help="Output dataset directory (hits/ is written)")],
) -> None:
    """Build the hit files of one prompt from its corpus and print one report per pair."""
    reports = build_and_write_hits_index(corpus_dir, data_dir, prompt_id, out)
    typer.echo(json.dumps([dataclasses.asdict(report) for report in reports], indent=1))


if __name__ == "__main__":
    typer.run(main)
