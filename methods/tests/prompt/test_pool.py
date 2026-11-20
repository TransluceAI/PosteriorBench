"""The pool file and the order of a replicate."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from posteriorbench_methods.prompt import pool

TASK_ID = "wc_1853/goblin_mention"


def test_candidates_are_read_in_index_order(
    make_pool_file: Callable[..., pathlib.Path], tmp_path: pathlib.Path
) -> None:
    candidates = pool.load_candidates(make_pool_file(("say a", "zzz", "abc")), TASK_ID)
    assert [(c.index, c.candidate_id, c.system_prompt) for c in candidates] == [
        (0, "c000", "say a"),
        (1, "c001", "zzz"),
        (2, "c002", "abc"),
    ]
    rows = [
        {
            "task_id": task,
            "candidate_index": index,
            "candidate_id": f"{task}-{index}",
            "system_prompt": f"prompt {index}",
            "scenario": "ignored",
        }
        for task in ("other/task", TASK_ID)
        for index in (2, 0, 1)
    ]
    path = tmp_path / "two_tasks.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert [c.candidate_id for c in pool.load_candidates(path, TASK_ID)] == [
        f"{TASK_ID}-{index}" for index in range(3)
    ]


def test_unusable_pool_files_are_refused(
    make_pool_file: Callable[..., pathlib.Path], tmp_path: pathlib.Path
) -> None:
    with pytest.raises(pool.PoolError, match="no candidate for the task"):
        pool.load_candidates(make_pool_file(("say a",)), "wc_0001/em_dash")
    gap = tmp_path / "gap.parquet"
    rows = [
        {"task_id": TASK_ID, "candidate_index": index, "candidate_id": "c", "system_prompt": "s"}
        for index in (0, 2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), gap)
    with pytest.raises(pool.PoolError, match="not 0..n-1"):
        pool.load_candidates(gap, TASK_ID)
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_id": TASK_ID, "system_prompt": "s"}]), missing)
    with pytest.raises(pool.PoolError, match="candidate_id"):
        pool.load_candidates(missing, TASK_ID)


def test_a_replicate_orders_all_candidates() -> None:
    order = pool.replicate_order(1050, seed=7)
    assert sorted(order) == list(range(1050))
    assert order == pool.replicate_order(1050, seed=7)
    assert order != pool.replicate_order(1050, seed=8)


def test_a_pool_file_can_give_the_batch_of_each_candidate(
    make_pool_file: Callable[..., pathlib.Path], tmp_path: pathlib.Path
) -> None:
    prompts = ("say a", "zzz", "abc")
    assert [c.batch for c in pool.load_candidates(make_pool_file(prompts), TASK_ID)] == [0, 0, 0]
    batched = make_pool_file(prompts, batches=(0, 0, 1))
    assert [c.batch for c in pool.load_candidates(batched, TASK_ID)] == [0, 0, 1]
    with pytest.raises(pool.PoolError, match="non-negative integers"):
        pool.load_candidates(make_pool_file(prompts, batches=(0, -1, 1)), TASK_ID)
    null_batch = tmp_path / "null_batch.parquet"
    rows = [
        {
            "task_id": TASK_ID,
            "candidate_index": index,
            "candidate_id": "c",
            "system_prompt": "s",
            "batch": batch,
        }
        for index, batch in enumerate((0, None))
    ]
    pq.write_table(pa.Table.from_pylist(rows), null_batch)
    with pytest.raises(pool.PoolError, match="non-negative integers"):
        pool.load_candidates(null_batch, TASK_ID)


def test_batches_follow_one_another_and_each_is_ordered_at_random() -> None:
    seeds = {0: 7, 1: 8, 3: 9}
    batches = [0] * 1050 + [1] * 500 + [3] * 300
    order = pool.batched_order(batches, seeds.__getitem__)
    assert sorted(order) == list(range(1850))
    # One batch is ordered as a file without batches is.
    assert pool.batched_order([0] * 1050, seeds.__getitem__) == pool.replicate_order(1050, seed=7)
    assert order[:1050] == pool.replicate_order(1050, seed=7)
    assert order[1050:1550] == [1050 + at for at in pool.replicate_order(500, seed=8)]
    assert order[1550:] == [1550 + at for at in pool.replicate_order(300, seed=9)]


def test_batches_need_not_be_contiguous_in_the_file() -> None:
    order = pool.batched_order([1, 0, 1, 0, 1], {0: 7, 1: 8}.__getitem__)
    assert sorted(order[:2]) == [1, 3]
    assert sorted(order[2:]) == [0, 2, 4]
