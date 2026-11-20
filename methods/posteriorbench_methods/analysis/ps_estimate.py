"""Estimating p(S) with a sampler: the importance-sampling estimate from a run's evaluation draws.

From n draws y_1, ..., y_n of a sampler q, importance sampling estimates the event probability as

    p_IS(S) = (1 / n) sum_i 1[y_i in S] p(y_i) / q(y_i),

which is unbiased when q can generate every hit that p can. The log error of a run is
log p_IS(S) - log p̂(S), where p̂(S) is the task's estimate from the corpus. The estimate reads
the first `draws` evaluation draws of the submitted sampler (960 of the 1,024 in the paper).

A run with a hit among these draws is scored by its own estimate, whether or not it passed the
threshold. A run without one, or without a sampler, is scored by the fallback's estimate p(a),
whose log error log(p(a) / p̂(S)) is minus the fallback's reverse KL.

The reverse KL bounds how far an own estimate can fall. With H hits among the n draws and
w_i = log p(y_i) - log q(y_i) on the hits, log p_IS(S) = log(H / n) + log mean(e^{w_i}), and the
reverse KL estimated from the same hits is -mean(w_i) - log(H / n) + log p̂(S). By Jensen's
inequality log mean(e^w) >= mean(w), so log error >= -reverse KL
(`RunEstimate.reverse_kl_all_hits`).

An own estimate needs log q and log p of every hit among the draws (evaluation_hits.json). A
record without that file has them only for the hits of the reverse KL, at most 128; if these are
not all the hits, its estimate is unavailable. Replay is left out: its evaluation draws nothing.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from posteriorbench import estimators

from posteriorbench_methods.analysis import records
from posteriorbench_methods.common import run_record

DRAWS = 960  # the evaluation draws that the paper's estimate reads, of the benchmark's 1,024
NOT_COVERED = ("replay",)
Source = Literal["own", "fallback", "unavailable"]


@dataclasses.dataclass(frozen=True)
class RunEstimate:
    """The estimate that one run is scored by.

    source: "own" (importance sampling), "fallback" (p(a)), or "unavailable" (an own estimate
        whose scores the record does not hold).
    draws: 0 for a run that produced no sampler.
    log_error: None if unavailable.
    reverse_kl_all_hits: the reverse KL over all the hits among the same draws; None unless the
        estimate is the run's own."""

    record: records.Record
    source: Source
    draws: int
    hits: int
    log_error: float | None
    reverse_kl_all_hits: float | None = None

    @property
    def passed(self) -> bool:
        """Whether the run's sampler passed the hit-rate threshold."""
        return not self.record.result["fallback"]

    def to_json(self) -> dict[str, Any]:
        result = self.record.result
        return {
            **self.record.label,
            "construction_inferences": self.record.construction_inferences,
            "passed_threshold": self.passed,
            "native_hit_rate": records.number(result["native_hit_rate"]),
            "estimate": self.source,
            "draws": self.draws,
            "hits": self.hits,
            "log_error": self.log_error,
            "abs_log_error": None if self.log_error is None else abs(self.log_error),
            "reverse_kl_all_hits": self.reverse_kl_all_hits,
            "reverse_kl": records.number(result["reverse_kl"]),
            "capped_forward_kl": records.number(result["capped_forward_kl"]),
        }


def _log_mean_exp(values: Sequence[float]) -> float:
    """log of the mean of exp(values); -inf if every value is -inf."""
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(value - top) for value in values) / len(values))


def log_error_from_hits(
    logp: Sequence[float], logq: Sequence[float], *, draws: int, log_p_event: float
) -> tuple[float, float]:
    """(log error of the importance-sampling estimate, reverse KL over the same hits) from log p
    and log q of the hits among `draws` draws. Needs at least one hit."""
    weights = [p - q for p, q in zip(logp, logq, strict=True)]
    log_hit_rate = math.log(len(weights) / draws)
    log_error = log_hit_rate + _log_mean_exp(weights) - log_p_event
    reverse_kl = -math.fsum(weights) / len(weights) - log_hit_rate + log_p_event
    return log_error, reverse_kl


def _fallback_log_error(record: records.Record) -> float:
    """log(p(a) / p̂(S)), the log error of the fallback's estimate p(a)."""
    result = record.result
    if result["fallback"]:
        return -records.number(result["reverse_kl"])
    log_p_fallback = records.number(record.evaluation_details["log_p_fallback"])
    return log_p_fallback - records.number(result["log_p_event"])


def _hit_scores(record: records.Record, rows: Sequence[Mapping[str, Any]]) -> dict[int, Any]:
    """(log q, log p) by draw index, of every hit if the record has evaluation_hits.json, else of
    the hits that the draws file has them for."""
    path = record.directory / run_record.EVALUATION_HITS_FILE
    if path.exists():
        scores = run_record.HitScores.from_json(run_record.read_json(path))
        return dict(zip(scores.draw_index, zip(scores.logq, scores.logp), strict=True))
    return {
        int(row["draw_index"]): (records.number(row["logq"]), records.number(row["logp"]))
        for row in rows
        if row.get("logq") is not None and row.get("logp") is not None
    }


def estimate(record: records.Record, draws: int = DRAWS) -> RunEstimate:
    """The estimate that the run of `record` is scored by (module docstring). Raises
    records.RecordsError if the record has fewer than `draws` evaluation draws."""
    result = record.result
    if result["kind"] != "sampler":
        return RunEstimate(record, "fallback", 0, 0, _fallback_log_error(record))
    path = record.directory / run_record.DRAWS_DIR / f"{run_record.EVALUATION_DRAWS_NAME}.jsonl"
    if not path.exists():
        raise records.RecordsError(f"{record.directory} has no evaluation draws")
    rows = run_record.read_jsonl(path)
    if len(rows) < draws:
        raise records.RecordsError(
            f"{record.directory}: the sampler was evaluated on {len(rows)} draws, fewer than "
            f"the {draws} that the estimate reads"
        )
    hits = [index for index, row in enumerate(rows[:draws]) if row["hit"]]
    if not hits:
        return RunEstimate(record, "fallback", draws, 0, _fallback_log_error(record))
    scores = _hit_scores(record, rows)
    if any(index not in scores for index in hits):
        return RunEstimate(record, "unavailable", draws, len(hits), None)
    log_error, reverse_kl = log_error_from_hits(
        [scores[index][1] for index in hits],
        [scores[index][0] for index in hits],
        draws=draws,
        log_p_event=records.number(result["log_p_event"]),
    )
    return RunEstimate(record, "own", draws, len(hits), log_error, reverse_kl)


def average_over_tasks(
    values: Mapping[tuple[str, int], float], *, require_equal_replicates: bool = True
) -> dict[str, Any]:
    """The average of one value per (task, replicate) as `posteriorbench.aggregate` averages:
    over the replicates of a task, then over the tasks, with the standard error over the tasks
    (None with one task). Returns mean, standard_error, tasks and replicates_per_task. Raises
    ValueError for tasks with different numbers of replicates, if they are required equal."""
    by_task: dict[str, list[float]] = {}
    for (task_id, _), value in sorted(values.items()):
        by_task.setdefault(task_id, []).append(value)
    counts = {len(group) for group in by_task.values()}
    if require_equal_replicates and len(counts) != 1:
        numbers = ", ".join(str(count) for count in sorted(counts))
        raise ValueError(f"tasks have different numbers of replicates: {numbers}")
    task_means = [estimators.mean_and_standard_error(group)[0] for group in by_task.values()]
    mean, standard_error = estimators.mean_and_standard_error(task_means)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "tasks": len(by_task),
        "replicates_per_task": min(counts),
    }


def point_summary(
    method: str,
    setting: str,
    runs: Sequence[RunEstimate],
    *,
    require_equal_replicates: bool = True,
) -> dict[str, Any]:
    """One point: the average absolute log error and cost of its runs, and how many runs are
    scored by which estimate. The error is None if the estimate of a run is unavailable."""
    key = {(run.record.task_id, run.record.replicate): run for run in runs}
    cost = average_over_tasks(
        {run: found.record.construction_inferences for run, found in key.items()},
        require_equal_replicates=require_equal_replicates,
    )
    errors = {
        run: abs(found.log_error) for run, found in key.items() if found.log_error is not None
    }
    error = None
    if len(errors) == len(runs):
        averaged = average_over_tasks(errors, require_equal_replicates=require_equal_replicates)
        error = {name: averaged[name] for name in ("mean", "standard_error")}
    return {
        "method": method,
        "setting": setting,
        "tasks": cost["tasks"],
        "replicates_per_task": cost["replicates_per_task"],
        "runs": len(runs),
        "own_estimate_passed": sum(run.source == "own" and run.passed for run in runs),
        "own_estimate_failed": sum(run.source == "own" and not run.passed for run in runs),
        "fallback_estimate": sum(run.source == "fallback" for run in runs),
        "unavailable": sum(run.source == "unavailable" for run in runs),
        "construction_inferences": {name: cost[name] for name in ("mean", "standard_error")},
        "abs_log_error": error,
    }


def sampler_rows(runs: Sequence[RunEstimate]) -> list[dict[str, Any]]:
    """One row for each sampler that passed the threshold and has its own estimate: its signed
    log error, the reverse KL and the capped forward KL of its evaluation, its hit rate, and
    draws_needed = exp(capped forward KL) / hit rate. An evaluation that several records hold (a
    composed method that was not composed, budgets that submit one checkpoint) is one sampler:
    the row is that of the first record, and `records` counts them."""
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in runs:
        if run.source != "own" or not run.passed:
            continue
        values = run.to_json()
        evaluation = (
            run.record.task_id,
            run.record.replicate,
            values["log_error"],
            values["reverse_kl"],
            values["capped_forward_kl"],
        )
        if evaluation in rows:
            rows[evaluation]["records"] += 1
            continue
        forward, rate = values["capped_forward_kl"], values["native_hit_rate"]
        rows[evaluation] = {
            **{name: values[name] for name in ("method", "setting", "task_id", "replicate")},
            "records": 1,
            "log_error": values["log_error"],
            "reverse_kl": values["reverse_kl"],
            "reverse_kl_all_hits": values["reverse_kl_all_hits"],
            "capped_forward_kl": forward,
            "native_hit_rate": rate,
            "draws_needed": math.exp(forward) / rate if forward < 700 else math.inf,
        }
    return list(rows.values())


def _ranks(values: Sequence[float]) -> np.ndarray:
    """The ranks of `values` from 1, with the average rank for equal values: float [len]."""
    _, inverse, counts = np.unique(np.asarray(values), return_inverse=True, return_counts=True)
    last = np.cumsum(counts)  # the largest rank of each distinct value
    return (last - (counts - 1) / 2)[inverse]


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Spearman's rank correlation of two sequences; None if either has fewer than two distinct
    values."""
    if len(set(first)) < 2 or len(set(second)) < 2:
        return None
    return float(np.corrcoef(_ranks(first), _ranks(second))[0, 1])


def samplers_summary(rows: Sequence[Mapping[str, Any]], draws: int = DRAWS) -> dict[str, Any]:
    """Over the rows of `sampler_rows`: their number; Spearman's correlation of the absolute log
    error with the capped forward KL and with the reverse KL; and the samplers that need fewer
    draws than the estimate reads, with their median log error."""
    errors = [abs(row["log_error"]) for row in rows]
    enough = [row["log_error"] for row in rows if row["draws_needed"] < draws]
    return {
        "samplers": len(rows),
        "spearman_abs_log_error_capped_forward_kl": spearman(
            errors, [row["capped_forward_kl"] for row in rows]
        ),
        "spearman_abs_log_error_reverse_kl": spearman(errors, [row["reverse_kl"] for row in rows]),
        "need_fewer_draws": len(enough),
        "need_fewer_draws_median_log_error": float(np.median(enough)) if enough else None,
    }


def analyze(
    found: Sequence[records.Record],
    draws: int = DRAWS,
    *,
    require_equal_replicates: bool = True,
    with_runs: bool = False,
    with_samplers: bool = False,
) -> dict[str, Any]:
    """The analysis of `found`, as a JSON-ready dict:

    points: for each method and setting, `point_summary`.
    samplers_summary: `samplers_summary` of the samplers that passed.
    samplers (with `with_samplers`): `sampler_rows`.
    runs (with `with_runs`): the estimate of every run.
    unavailable: the number of runs whose estimate is unavailable.
    not_covered: the records that were left out, by method.

    Raises records.RecordsError if the records of a point cannot be averaged."""
    records.check_one_record_per_run(found)
    not_covered: dict[str, int] = {}
    by_point: dict[tuple[str, str], list[RunEstimate]] = {}
    for record in found:
        if record.method in NOT_COVERED or record.result["exact"]:
            not_covered[record.method] = not_covered.get(record.method, 0) + 1
            continue
        by_point.setdefault((record.method, record.setting), []).append(estimate(record, draws))
    points: list[dict[str, Any]] = []
    for (method, setting), runs in by_point.items():
        try:
            points.append(
                point_summary(
                    method, setting, runs, require_equal_replicates=require_equal_replicates
                )
            )
        except ValueError as error:
            raise records.RecordsError(f"{method} at {setting}: {error}") from error
    every_run = [run for runs in by_point.values() for run in runs]
    rows = sampler_rows(every_run)
    analysis: dict[str, Any] = {
        "draws": draws,
        "points": points,
        "samplers_summary": samplers_summary(rows, draws),
        "unavailable": sum(run.source == "unavailable" for run in every_run),
        "not_covered": not_covered,
    }
    if with_samplers:
        analysis["samplers"] = rows
    if with_runs:
        analysis["runs"] = [run.to_json() for run in every_run]
    return analysis
