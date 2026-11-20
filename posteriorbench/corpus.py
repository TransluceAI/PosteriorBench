"""Construction-time access to a task's corpus hits, charged to the method's ledger.

The corpus holds 50 million responses of the original model per prompt; a task's hits among them
are exact samples of its behavior posterior. A method may build its sampler from any of them except
the task's references, and pays 1/p̂(S) inferences per hit it takes, the expected number of
original-model draws that rejection sampling spends per hit. `CorpusAccess` hands hits out and
charges them automatically.

It reads the task's hit file `hits/<prompt_id>/<behavior>.parquet`, never the corpus. The hit file
lists the task's hits in corpus order without its references; the corpus is in random order within
a prompt, so any prefix of the file is a uniform sample of hits. Hits whose token ids equal a
reference's (frequent responses that also occur among the references) are flagged in the file; by
default `CorpusAccess` leaves them out too, so a sampler built from corpus hits never contains a
reference.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from posteriorbench import data, errors
from posteriorbench.ledger import Category, Ledger


class CorpusError(errors.PosteriorBenchError):
    """A request for corpus hits that the task's hit file cannot satisfy, or a hit file that does
    not belong to the task."""


@dataclasses.dataclass(frozen=True)
class CorpusHit:
    """One corpus hit handed to a method.

    `position` is the hit's index in the access's hit sequence (the task's hits in corpus order,
    without its references and, unless the access keeps them, without reference duplicates). It is
    None only for a hit taken by row id (`CorpusAccess.take_rows`) that lies outside that
    sequence. `row` is the hit's row in the prompt's corpus."""

    position: int | None
    row: int
    token_ids: tuple[int, ...]


def _metadata(file: pq.ParquetFile) -> dict[str, str]:
    raw: dict[bytes, bytes] = file.schema_arrow.metadata or {}
    return {key.decode(): value.decode() for key, value in raw.items()}


class CorpusAccess:
    """Construction-time access to one task's corpus hits.

    Every distinct hit handed out is charged 1/p̂(S) inferences (and budget units) to `ledger`
    under `Category.CORPUS_HITS`. The task's references are never returned. With
    `exclude_reference_duplicates` (the default), hits whose token ids equal a reference's are not
    in the access's hit sequence either. Hits are returned in the requested order; a hit requested
    again is returned again and not charged again. If the ledger has a budget limit that a charge
    would exceed, `ledger.BudgetExceeded` is raised and nothing is handed out. Each access charges
    on its own, so keep one per task and ledger.

    The constructor reads the task's hit file (rows and duplicate flags; token ids are read on
    demand), downloading it first from a Hub source. Raises data.DataError if the source has no
    hit file for the task or the task is not a scored pair, and CorpusError if the file does not
    belong to the task."""

    def __init__(
        self,
        source: str | os.PathLike[str],
        task: data.Task,
        ledger: Ledger,
        *,
        exclude_reference_duplicates: bool = True,
        revision: str | None = None,
    ) -> None:
        task.require_scored()
        self.task = task
        self.ledger = ledger
        self.exclude_reference_duplicates = exclude_reference_duplicates
        path = data.resolve_file(
            source, data.hits_file(task.prompt_id, task.behavior), revision=revision
        )
        self._file = pq.ParquetFile(path)
        missing = [
            name for name in data.HITS_SCHEMA.names if name not in self._file.schema_arrow.names
        ]
        if missing:
            raise CorpusError(
                f"{path} has no column {missing}; this package reads dataset version "
                f"{data.DATASET_REVISION}"
            )
        self._check_metadata(_metadata(self._file))
        table = self._file.read(columns=["row", "reference_duplicate"])
        self._rows: np.ndarray[Any, np.dtype[np.int64]] = (
            table.column("row").to_numpy().astype(np.int64)
        )
        self._duplicate: np.ndarray[Any, np.dtype[np.bool_]] = np.asarray(
            table.column("reference_duplicate").to_numpy(zero_copy_only=False), dtype=bool
        )
        if exclude_reference_duplicates:
            self._usable = np.flatnonzero(~self._duplicate)
        else:
            self._usable = np.arange(len(self._rows))
        sizes = [
            self._file.metadata.row_group(i).num_rows for i in range(self._file.num_row_groups)
        ]
        self._group_starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._row_order: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self._taken: set[int] = set()
        self._duplicates_taken = 0
        # The token-id column of the last row group read: (group, pyarrow column).
        self._cached_group: tuple[int, Any] | None = None

    def _check_metadata(self, metadata: dict[str, str]) -> None:
        task = self.task
        if "task_id" in metadata and metadata["task_id"] != task.task_id:
            raise CorpusError(f"the hit file belongs to {metadata['task_id']}, not {task.task_id}")
        if "hit_count" in metadata and int(metadata["hit_count"]) != task.hit_count:
            raise CorpusError(
                f"{task.task_id}: the hit file was built for hit_count {metadata['hit_count']}, "
                f"the task table says {task.hit_count}"
            )
        rows = self._file.metadata.num_rows
        if "references_removed" in metadata:
            expected = task.hit_count - int(metadata["references_removed"])
            if rows != expected:
                raise CorpusError(
                    f"{task.task_id}: the hit file has {rows} rows, expected {expected}"
                )

    @property
    def available(self) -> int:
        """The number of hits in this access's hit sequence."""
        return len(self._usable)

    def cost_of(self, k: int) -> float:
        """What taking k new hits would charge, k / p̂(S) inferences; charges nothing."""
        if k < 0:
            raise ValueError("k must be non-negative")
        return k / self.task.probability

    @property
    def hits_taken(self) -> int:
        """Distinct hits handed out (and charged) so far."""
        return len(self._taken)

    @property
    def reference_duplicates_taken(self) -> int:
        """Distinct hits handed out so far whose token ids equal a reference's (possible only with
        `exclude_reference_duplicates=False` or through `take_rows`). For the run record."""
        return self._duplicates_taken

    def take(self, positions: Sequence[int]) -> list[CorpusHit]:
        """The hits at these positions of this access's hit sequence, in the requested order.
        Raises CorpusError for a position outside [0, available)."""
        file_indices: list[int] = []
        for position in positions:
            position = int(position)
            if not 0 <= position < len(self._usable):
                raise CorpusError(
                    f"{self.task.task_id}: position {position} outside [0, {len(self._usable)})"
                )
            file_indices.append(int(self._usable[position]))
        return self._hand_out(file_indices)

    def take_first(self, n: int) -> list[CorpusHit]:
        """The first n hits of this access's hit sequence (a uniform sample of hits)."""
        if not 0 <= n <= len(self._usable):
            raise CorpusError(f"{self.task.task_id}: cannot take {n} of {len(self._usable)} hits")
        return self.take(range(n))

    def take_rows(self, rows: Sequence[int]) -> list[CorpusHit]:
        """The hits with these corpus rows, in the requested order (for training sets stored by
        row). A reference duplicate may be taken this way even when the access excludes them from
        its sequence; it is counted in `reference_duplicates_taken`. Raises CorpusError unless each
        row is a hit of the task other than a reference."""
        if self._row_order is None:
            self._row_order = np.argsort(self._rows, kind="stable").astype(np.int64)
        sorted_rows = self._rows[self._row_order]
        file_indices: list[int] = []
        for row in rows:
            at = int(np.searchsorted(sorted_rows, int(row)))
            if at >= len(sorted_rows) or int(sorted_rows[at]) != int(row):
                raise CorpusError(
                    f"{self.task.task_id}: corpus row {int(row)} is not a hit of this task outside "
                    "its references"
                )
            file_indices.append(int(self._row_order[at]))
        return self._hand_out(file_indices)

    def _position_of(self, file_index: int) -> int | None:
        if not self.exclude_reference_duplicates:
            return file_index
        at = int(np.searchsorted(self._usable, file_index))
        if at < len(self._usable) and int(self._usable[at]) == file_index:
            return at
        return None

    def _hand_out(self, file_indices: list[int]) -> list[CorpusHit]:
        """Charge the not yet taken hits among `file_indices`, then return all of them."""
        new = sorted(set(file_indices) - self._taken)
        if new:
            self.ledger.charge(
                Category.CORPUS_HITS,
                inferences=len(new) / self.task.probability,
                note=f"{len(new)} corpus hits of {self.task.task_id}",
            )
            self._taken.update(new)
            self._duplicates_taken += int(np.count_nonzero(self._duplicate[new]))
        token_ids = self._token_ids(file_indices)
        return [
            CorpusHit(self._position_of(index), int(self._rows[index]), ids)
            for index, ids in zip(file_indices, token_ids, strict=True)
        ]

    def _group(self, group: int) -> Any:
        """The token-id column (a pyarrow ChunkedArray) of one row group, cached."""
        if self._cached_group is None or self._cached_group[0] != group:
            column = self._file.read_row_group(group, columns=["token_ids"]).column("token_ids")
            self._cached_group = (group, column)
        return self._cached_group[1]

    def _token_ids(self, file_indices: list[int]) -> list[tuple[int, ...]]:
        """Token ids of the hits at these file indices, read row group by row group."""
        result: list[tuple[int, ...] | None] = [None] * len(file_indices)
        groups = np.searchsorted(self._group_starts, file_indices, side="right") - 1
        by_group: dict[int, list[int]] = {}
        for slot, group in enumerate(groups.tolist()):
            by_group.setdefault(int(group), []).append(slot)
        for group, slots in sorted(by_group.items()):
            start = int(self._group_starts[group])
            local = pa.array([file_indices[slot] - start for slot in slots], type=pa.int64())
            values = self._group(group).take(local).to_pylist()
            for slot, ids in zip(slots, values, strict=True):
                result[slot] = tuple(int(t) for t in ids)
        assert all(ids is not None for ids in result)
        return [ids for ids in result if ids is not None]
