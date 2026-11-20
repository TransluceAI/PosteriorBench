"""Averaging results over replicates and tasks.

Each task's replicates are averaged first; the task means are then averaged with equal weights,
and the standard error is the sample standard deviation of the task means (ddof 1) over
sqrt(number of tasks); an infinite value makes the mean infinite and the standard error NaN. The
equal-weight task average of a KL score is itself a KL divergence between joint distributions over
(task, response). The construction cost is averaged in the same way.
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Collection, Sequence
from typing import Any

from posteriorbench import estimators, results

METRICS: tuple[str, ...] = (
    "capped_forward_kl",
    "reverse_kl",
    "native_hit_rate",
    "construction_inferences",
)
GROUP_KEYS: tuple[str, ...] = ("method", "setting")


@dataclasses.dataclass(frozen=True)
class MetricSummary:
    """Task-averaged mean of one metric and its standard error (None with a single task)."""

    mean: float
    standard_error: float | None


@dataclasses.dataclass(frozen=True)
class Summary:
    """Task-averaged scores of one method at one setting."""

    tasks: int
    replicates_per_task: int
    results: int
    passed_threshold: int
    metrics: dict[str, MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_comparable(evaluated: Sequence[results.EvaluationResult]) -> None:
    """Raise ValueError if the results were not scored the same way: one cap C; one choice of the
    reference log p ("shipped" or "recomputed") over the results of submitted samplers; one number
    of evaluation draws over the results that made draws."""
    caps = {r.cap for r in evaluated}
    reference_logp = {r.reference_logp for r in evaluated if r.kind == "sampler"}
    draws = {r.draws for r in evaluated if r.draws > 0}
    for name, values in (
        ("caps", caps),
        ("reference log p", reference_logp),
        ("draw counts", draws),
    ):
        if len(values) > 1:
            raise ValueError(f"results with different {name}: {sorted(map(str, values))}")


def aggregate(
    evaluated: Sequence[results.EvaluationResult],
    *,
    require_equal_replicates: bool = True,
    expected_replicates: Collection[int] | None = None,
    expected_tasks: Collection[str] | None = None,
) -> Summary:
    """Summarize the results of one method at one setting over tasks; see the module docstring.

    Raises ValueError if there are no results, the results were not scored the same way
    (`check_comparable`), a task repeats a replicate index, tasks have different numbers of
    replicates (unless `require_equal_replicates` is False), with `expected_replicates` (e.g.
    {0, 1, 2}) some task's replicate indices differ from it, or with `expected_tasks` (e.g. the
    benchmark's 21 tasks) a task is missing or unexpected."""
    if not evaluated:
        raise ValueError("no results")
    check_comparable(evaluated)
    by_task: dict[str, list[results.EvaluationResult]] = collections.defaultdict(list)
    for result in evaluated:
        by_task[result.task_id].append(result)
    if expected_tasks is not None:
        missing = sorted(set(expected_tasks) - set(by_task))
        extra = sorted(set(by_task) - set(expected_tasks))
        if missing or extra:
            raise ValueError(f"tasks missing: {missing}; unexpected tasks: {extra}")
    counts = {len(group) for group in by_task.values()}
    if require_equal_replicates and len(counts) != 1:
        raise ValueError(f"tasks have different numbers of replicates: {sorted(counts)}")
    expected = None if expected_replicates is None else {int(r) for r in expected_replicates}
    for task_id, group in by_task.items():
        replicates = {r.replicate for r in group}
        if len(replicates) != len(group):
            raise ValueError(f"{task_id}: repeated replicate index")
        if expected is not None and replicates != expected:
            raise ValueError(
                f"{task_id}: replicates {sorted(replicates)}, expected {sorted(expected)}"
            )
    metrics: dict[str, MetricSummary] = {}
    for metric in METRICS:
        task_means = [
            estimators.mean_and_standard_error([float(getattr(r, metric)) for r in group])[0]
            for _, group in sorted(by_task.items())
        ]
        mean, standard_error = estimators.mean_and_standard_error(task_means)
        metrics[metric] = MetricSummary(mean, standard_error)
    return Summary(
        tasks=len(by_task),
        replicates_per_task=min(counts),
        results=len(evaluated),
        passed_threshold=sum(r.passed_threshold for r in evaluated),
        metrics=metrics,
    )


def group_results(
    evaluated: Sequence[results.EvaluationResult], keys: Sequence[str] = GROUP_KEYS
) -> dict[tuple[Any, ...], list[results.EvaluationResult]]:
    """Results grouped by the values of `keys` (result fields; by default method and setting),
    in first-seen order. Raises ValueError for a key that is not a result field."""
    names = {f.name for f in dataclasses.fields(results.EvaluationResult)}
    unknown = [key for key in keys if key not in names]
    if unknown:
        raise ValueError(f"unknown result fields {unknown}")
    groups: dict[tuple[Any, ...], list[results.EvaluationResult]] = {}
    for result in evaluated:
        groups.setdefault(tuple(getattr(result, key) for key in keys), []).append(result)
    return groups
