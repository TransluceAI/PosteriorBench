"""Writing a pool file and adding candidates to one."""

from __future__ import annotations

import pathlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from posteriorbench_methods.prompt import pool

TASK_ID = "wc_1853/goblin_mention"
OTHER_TASK = "wc_0346/em_dash"


def written(*texts: str) -> list[pool.Written]:
    return [pool.Written(text, f"the scenario of {text}") for text in texts]


def rows(path: pathlib.Path, task_id: str) -> list[dict[str, Any]]:
    """The rows of a task in a pool file, by candidate index, with every column."""
    table = pq.read_table(path, filters=[("task_id", "=", task_id)])
    return sorted(table.to_pylist(), key=lambda row: row["candidate_index"])


def test_a_new_file_holds_the_candidates_in_the_order_given(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pools" / "candidates.parquet"
    assert pool.existing_candidates(path, TASK_ID) == []
    added = pool.add_candidates(path, TASK_ID, written("say a", "zzz"), new_batch=False)
    assert added == pool.load_candidates(path, TASK_ID)
    assert [(c.index, c.system_prompt, c.batch) for c in added] == [(0, "say a", 0), (1, "zzz", 0)]
    assert pq.read_schema(path).names == [
        "task_id",
        "candidate_index",
        "candidate_id",
        "system_prompt",
        "scenario",
        "batch",
    ]
    assert [row["scenario"] for row in rows(path, TASK_ID)] == [
        "the scenario of say a",
        "the scenario of zzz",
    ]
    assert not list(path.parent.glob("*.partial"))


def test_a_candidate_is_named_by_its_text() -> None:
    assert pool.candidate_name("say a") == pool.candidate_name("say a")
    assert pool.candidate_name("say a") != pool.candidate_name("say a ")
    assert len(pool.candidate_name("中 —")) == 12


def test_candidates_join_the_last_batch_or_form_a_new_one(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "candidates.parquet"
    # A task without candidates starts at batch 0, with or without a new batch.
    pool.add_candidates(path, TASK_ID, written("a", "b"), new_batch=True)
    pool.add_candidates(path, TASK_ID, written("c"), new_batch=False)
    pool.add_candidates(path, TASK_ID, written("d", "e"), new_batch=True)
    added = pool.add_candidates(path, TASK_ID, written("f"), new_batch=False)
    assert [(c.index, c.batch) for c in added] == [(5, 1)]
    candidates = pool.load_candidates(path, TASK_ID)
    assert [c.system_prompt for c in candidates] == ["a", "b", "c", "d", "e", "f"]
    assert [c.batch for c in candidates] == [0, 0, 0, 1, 1, 1]
    # A new batch leaves the order of the candidates before it as it was.
    order = pool.batched_order([c.batch for c in candidates], {0: 7, 1: 8}.__getitem__)
    assert order[:3] == pool.replicate_order(3, seed=7)
    assert sorted(order[3:]) == [3, 4, 5]


def test_other_tasks_keep_their_candidates(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "candidates.parquet"
    pool.add_candidates(path, OTHER_TASK, written("x", "y"), new_batch=False)
    pool.add_candidates(path, TASK_ID, written("a"), new_batch=False)
    pool.add_candidates(path, OTHER_TASK, written("z"), new_batch=True)
    assert [c.system_prompt for c in pool.load_candidates(path, OTHER_TASK)] == ["x", "y", "z"]
    assert [c.batch for c in pool.load_candidates(path, OTHER_TASK)] == [0, 0, 1]
    assert [(c.index, c.batch) for c in pool.load_candidates(path, TASK_ID)] == [(0, 0)]


def test_nothing_is_written_without_a_candidate(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "candidates.parquet"
    assert pool.add_candidates(path, TASK_ID, [], new_batch=False) == []
    assert not path.exists()


def test_a_file_made_elsewhere_keeps_its_columns(tmp_path: pathlib.Path) -> None:
    """A file with the four columns that a run reads and one of its own, as a reader may have
    made it: it gets the columns `scenario` and `batch`, and keeps its own."""
    path = tmp_path / "candidates.parquet"
    table = pa.table(
        {
            "task_id": [TASK_ID, TASK_ID, OTHER_TASK],
            "candidate_index": pa.array([1, 0, 0], type=pa.int32()),
            "candidate_id": ["second", "first", "other"],
            "system_prompt": ["b", "a", "x"],
            "note": ["kept", "kept", "kept"],
        }
    )
    pq.write_table(table, path)
    added = pool.add_candidates(path, TASK_ID, written("c"), new_batch=True)
    assert [(c.index, c.batch) for c in added] == [(2, 1)]
    schema = pq.read_schema(path)
    assert schema.names == [*table.column_names, "scenario", "batch"]
    assert schema.field("candidate_index").type == pa.int32()
    assert [
        (row["candidate_id"], row["note"], row["scenario"], row["batch"])
        for row in rows(path, TASK_ID)
    ] == [
        ("first", "kept", None, 0),
        ("second", "kept", None, 0),
        (pool.candidate_name("c"), None, "the scenario of c", 1),
    ]
    assert rows(path, OTHER_TASK)[0]["batch"] == 0


def test_a_file_that_a_run_would_refuse_is_not_added_to(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "gap.parquet"
    gap = [
        {"task_id": TASK_ID, "candidate_index": index, "candidate_id": "c", "system_prompt": "s"}
        for index in (0, 2)
    ]
    pq.write_table(pa.Table.from_pylist(gap), path)
    before = path.read_bytes()
    with pytest.raises(pool.PoolError, match="not 0..n-1"):
        pool.add_candidates(path, TASK_ID, written("a"), new_batch=False)
    assert path.read_bytes() == before
