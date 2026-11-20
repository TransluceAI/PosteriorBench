"""Compute per counted inference: what a run computes, for each inference that it is charged.

Two measures: FLOPs, in forward passes of the original model over one response (the largest of
`flops.CONVENTIONS`), and GPU time, in generations of one response by the original model
(`gpu_times`). A point's value is the sum of its runs' compute over the sum of their inference
counts, the ratio of the two averages over tasks and replicates; a method's value is the highest
among its points. Methods without a rule (`operations.trains`) are left out and counted.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

from posteriorbench_methods.analysis import flops, gpu_times, operations, records


@dataclasses.dataclass(frozen=True)
class RunCompute:
    """The compute of one run.

    inferences: the run's inference count.
    flops: its compute in forward passes of the original model over one response, under each
        convention of `flops.CONVENTIONS`.
    gpu_time: its GPU time in generations of one response by the original model; None without
        measured times."""

    record: records.Record
    done: operations.Operations
    inferences: float
    flops: dict[str, float]
    gpu_time: float | None

    def to_json(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            **self.record.label,
            "construction_inferences": self.inferences,
            "operations": dataclasses.asdict(self.done),
            "flops_per_inference": _ratio(max(self.flops.values()), self.inferences),
            "flops_per_inference_by_convention": {
                name: _ratio(value, self.inferences) for name, value in self.flops.items()
            },
        }
        if self.gpu_time is not None:
            value["gpu_time_per_inference"] = _ratio(self.gpu_time, self.inferences)
        return value


def _ratio(compute: float, inferences: float) -> float | None:
    """compute / inferences; None for a run or a point that is charged nothing."""
    return compute / inferences if inferences > 0 else None


def flops_of_run(
    record: records.Record,
    done: operations.Operations,
    shape: flops.Shape,
    *,
    prompt_tokens: int,
) -> dict[str, float]:
    """The compute of a run in inferences, under each convention. Every response is counted at
    the response limit; SFT's layers and GFlowNet's rank are the run's settings."""
    kind = operations.trains(record.method)
    assert kind is not None, "the operations of a record are those of a method with a rule"
    settings = record.settings
    compute: dict[str, float] = {}
    for name, convention in flops.CONVENTIONS.items():
        cost = flops.passes(
            shape,
            prompt_tokens=prompt_tokens,
            response_tokens=int(record.config["scorer"]["max_response_tokens"]),
            convention=convention,
            adapter_rank=int(settings.get("lora_rank", 1)),
            trained_last_layers=int(settings.get("trainable_last_layers", 1)),
        )
        factor = flops.factors(kind, cost)
        compute[name] = math.fsum(
            [
                done.draws_from_p,
                factor.sampler_draw * done.sampler_draws,
                done.scored,
                factor.training_forward * done.training_forwards,
                factor.gradient_response * done.gradient_responses,
            ]
        )
    return compute


def of_records(
    found: Sequence[records.Record],
    shape: flops.Shape,
    *,
    prompt_tokens: Callable[[str], int],
    times: gpu_times.Times | None,
) -> tuple[list[RunCompute], dict[str, int]]:
    """The compute of every run among `found` whose method has a rule, and the number of records
    of each method that has none.

    prompt_tokens: the length of a task's prompt, by task id.
    times: None computes the FLOPs alone."""
    records.check_one_record_per_run(found)
    runs: list[RunCompute] = []
    not_covered: dict[str, int] = {}
    for record in found:
        kind = operations.trains(record.method)
        if kind is None:
            not_covered[record.method] = not_covered.get(record.method, 0) + 1
            continue
        done = operations.of_record(record)
        gpu_time = None
        if times is not None:
            session = times.session_for(kind)
            gpu_time = gpu_times.run_seconds(kind, done, session) / session.unit
        runs.append(
            RunCompute(
                record=record,
                done=done,
                inferences=record.construction_inferences,
                flops=flops_of_run(
                    record, done, shape, prompt_tokens=prompt_tokens(record.task_id)
                ),
                gpu_time=gpu_time,
            )
        )
    return runs, not_covered


def point_summary(method: str, setting: str, runs: Sequence[RunCompute]) -> dict[str, Any]:
    """One point: the summed compute of its runs over their summed inference counts. The FLOPs
    are the largest over the conventions."""
    inferences = math.fsum(run.inferences for run in runs)
    by_convention = {
        name: _ratio(math.fsum(run.flops[name] for run in runs), inferences)
        for name in flops.CONVENTIONS
    }
    known = [value for value in by_convention.values() if value is not None]
    point: dict[str, Any] = {
        "method": method,
        "setting": setting,
        "runs": len(runs),
        "construction_inferences": inferences / len(runs),
        "flops_per_inference": max(known) if known else None,
        "flops_per_inference_by_convention": by_convention,
    }
    times = [run.gpu_time for run in runs if run.gpu_time is not None]
    if len(times) == len(runs):
        point["gpu_time_per_inference"] = _ratio(math.fsum(times), inferences)
    return point


def _largest(entries: Sequence[dict[str, Any]], key: str, name: str) -> dict[str, Any] | None:
    """The entry with the largest `key`, as {key: value, name: where}; None if none has one."""
    known = [entry for entry in entries if entry.get(key) is not None]
    if not known:
        return None
    top = max(known, key=lambda entry: entry[key])
    return {key: top[key], name: top[name]}


def summarize(runs: Sequence[RunCompute]) -> dict[str, list[dict[str, Any]]]:
    """{"methods": ..., "points": ...}: every point, and for each method the highest value
    among its points (with its setting) and among its single runs."""
    by_point: dict[tuple[str, str], list[RunCompute]] = {}
    for run in runs:
        by_point.setdefault((run.record.method, run.record.setting), []).append(run)
    points = [
        point_summary(method, setting, members) for (method, setting), members in by_point.items()
    ]
    methods: list[dict[str, Any]] = []
    for method in dict.fromkeys(point["method"] for point in points):
        of_method = [point for point in points if point["method"] == method]
        single = [run.to_json() for run in runs if run.record.method == method]
        summary: dict[str, Any] = {
            "method": method,
            "points": len(of_method),
            "runs": len(single),
        }
        for key in ("flops_per_inference", "gpu_time_per_inference"):
            top = _largest(of_method, key, "setting")
            if top is not None:
                summary[key] = top[key]
                summary[f"{key}_at"] = top["setting"]
            top_run = _largest(single, key, "task_id")
            if top_run is not None:
                summary[f"largest_run_{key}"] = top_run[key]
        methods.append(summary)
    return {"methods": methods, "points": points}
