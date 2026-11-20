"""Task averaging (replicates first, then tasks with equal weights) and grouping."""

from __future__ import annotations

import math

import pytest

from posteriorbench import aggregate, results


def make_result(
    task: str,
    replicate: int,
    forward: float,
    reverse: float,
    rate: float,
    *,
    method: str | None = "m",
    setting: str | None = "K=16",
) -> results.EvaluationResult:
    return results.EvaluationResult(
        task_id=task, method=method, setting=setting, kind="sampler", sampler="s", seed=replicate,
        replicate=replicate, draws=1024, hits=None, native_hit_rate=rate,
        passed_threshold=rate >= 0.1, fallback=rate < 0.1, capped_forward_kl=forward,
        reverse_kl=reverse, cap=50.0, log_p_event=-5.0,
        construction_inferences=100.0 * (replicate + 1), budget_passes=None,
        inferences_per_response=1.0, cost_rule="ledger", ledger=None, reference_logp="shipped",
    )  # fmt: skip


RESULTS = [
    make_result("a", 0, 1.0, 10.0, 0.5),
    make_result("a", 1, 3.0, 20.0, 0.3),
    make_result("b", 0, 5.0, 30.0, 0.05),
    make_result("b", 1, 7.0, 40.0, 0.2),
    make_result("c", 0, 50.0, 1.0, 0.0),
    make_result("c", 1, 50.0, 3.0, 0.0),
]


def test_aggregate_averages_replicates_then_tasks() -> None:
    summary = aggregate.aggregate(RESULTS)
    assert (summary.tasks, summary.replicates_per_task, summary.passed_threshold) == (3, 2, 3)
    forward = summary.metrics["capped_forward_kl"]
    task_means = [2.0, 6.0, 50.0]
    assert forward.mean == pytest.approx(sum(task_means) / 3)
    variance = sum((m - forward.mean) ** 2 for m in task_means) / 2
    assert forward.standard_error == pytest.approx(math.sqrt(variance / 3))
    assert summary.metrics["reverse_kl"].mean == pytest.approx((15 + 35 + 2) / 3)
    assert summary.metrics["construction_inferences"].mean == pytest.approx(150.0)
    with pytest.raises(ValueError):
        aggregate.aggregate(RESULTS[:-1])
    assert aggregate.aggregate(RESULTS[:-1], require_equal_replicates=False).tasks == 3
    with pytest.raises(ValueError):
        aggregate.aggregate([])
    with pytest.raises(ValueError, match="repeated replicate"):
        aggregate.aggregate([RESULTS[0], RESULTS[0]])


def test_expected_replicates() -> None:
    assert aggregate.aggregate(RESULTS, expected_replicates={0, 1}).tasks == 3
    with pytest.raises(ValueError, match="expected"):
        aggregate.aggregate(RESULTS, expected_replicates={0, 1, 2})
    only_zero = [r for r in RESULTS if r.replicate == 0]
    assert aggregate.aggregate(only_zero, expected_replicates=[0]).replicates_per_task == 1


def test_group_results() -> None:
    mixed = RESULTS + [
        make_result("a", 0, 1.0, 1.0, 0.5, setting="K=32"),
        make_result("a", 0, 1.0, 1.0, 0.5, method=None, setting=None),
    ]
    groups = aggregate.group_results(mixed)
    assert list(groups) == [("m", "K=16"), ("m", "K=32"), (None, None)]
    assert len(groups[("m", "K=16")]) == 6
    assert list(aggregate.group_results(mixed, keys=("task_id",))) == [("a",), ("b",), ("c",)]
    with pytest.raises(ValueError):
        aggregate.group_results(mixed, keys=("nope",))


def test_aggregate_finds_missing_tasks_and_mixed_scoring() -> None:
    import dataclasses

    with pytest.raises(ValueError, match=r"tasks missing: \['d'\]"):
        aggregate.aggregate(RESULTS, expected_tasks={"a", "b", "c", "d"})
    with pytest.raises(ValueError, match=r"unexpected tasks: \['c'\]"):
        aggregate.aggregate(RESULTS, expected_tasks={"a", "b"})
    assert aggregate.aggregate(RESULTS, expected_tasks={"a", "b", "c"}).tasks == 3
    recomputed = dataclasses.replace(RESULTS[1], reference_logp="recomputed")
    with pytest.raises(ValueError, match="different reference log p"):
        aggregate.aggregate([RESULTS[0], recomputed])
    with pytest.raises(ValueError, match="different caps"):
        aggregate.aggregate([RESULTS[0], dataclasses.replace(RESULTS[1], cap=10.0)])
    with pytest.raises(ValueError, match="different draw counts"):
        aggregate.aggregate([RESULTS[0], dataclasses.replace(RESULTS[1], draws=512)])
    # A run without a sampler made no draws and used no reference log p: it mixes with either.
    no_sampler = dataclasses.replace(
        RESULTS[1], kind="no_sampler", sampler=None, draws=0, reference_logp=None
    )
    assert (
        aggregate.aggregate([recomputed, dataclasses.replace(no_sampler, task_id="b")]).tasks == 2
    )
    # A submitted sampler is compared whatever its name is.
    named = dataclasses.replace(recomputed, sampler="fallback", task_id="b")
    with pytest.raises(ValueError, match="different reference log p"):
        aggregate.aggregate([RESULTS[0], named])
