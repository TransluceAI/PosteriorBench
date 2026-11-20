"""Compute per counted inference, on the records of small runs."""

from __future__ import annotations

import math
from typing import Any

import pytest

from posteriorbench_methods.analysis import compute, flops, gpu_times, records

PROMPT_TOKENS = 12


@pytest.fixture(scope="module")
def shape(tiny_hybrid_model: Any) -> flops.Shape:
    """The shape of the tiny model with the layers of the paper's model."""
    return flops.Shape.from_config(tiny_hybrid_model.config)


@pytest.fixture(scope="module")
def computed(
    finished: list[records.Record], shape: flops.Shape, tiny_session: gpu_times.Session
) -> tuple[list[compute.RunCompute], dict[str, int]]:
    return compute.of_records(
        finished,
        shape,
        prompt_tokens=lambda task_id: PROMPT_TOKENS,
        times=gpu_times.Times((tiny_session,)),
    )


def runs_of(
    computed: tuple[list[compute.RunCompute], Any], method: str
) -> list[compute.RunCompute]:
    return [run for run in computed[0] if run.record.method == method]


def test_the_methods_without_a_rule_are_counted_and_left_out(
    computed: tuple[list[compute.RunCompute], dict[str, int]], finished: list[records.Record]
) -> None:
    runs, not_covered = computed
    assert {run.record.method for run in runs} == {
        "sft", "replay", "gflownet", "ctl", "kl-dpg", "gdc",
    }  # fmt: skip
    assert set(not_covered) == {
        "sft+composition", "gflownet+composition", "ctl+composition", "kl-dpg+composition",
        "gdc+composition",
    }  # fmt: skip
    assert len(runs) + sum(not_covered.values()) == len(finished)


def test_the_flops_of_a_run_are_its_operations_at_the_cost_of_its_method(
    computed: tuple[list[compute.RunCompute], dict[str, int]], shape: flops.Shape
) -> None:
    # Replay draws from the original model and does nothing else: one inference each.
    (replay,) = [run for run in runs_of(computed, "replay") if run.record.ledger_totals]
    assert set(replay.flops.values()) == {replay.inferences}
    # All weights: every counted operation costs one inference, a backward pass two.
    for run in runs_of(computed, "kl-dpg") + runs_of(computed, "gdc"):
        expected = run.inferences + 2 * run.done.gradient_responses
        assert set(run.flops) == set(flops.CONVENTIONS)
        assert all(value == pytest.approx(expected) for value in run.flops.values())
    # CTL: the twist network at every generated token and in its updates.
    (ctl,) = runs_of(computed, "ctl")
    cost = flops.passes(shape, prompt_tokens=PROMPT_TOKENS, response_tokens=50)
    factor = flops.factors("twist", cost)
    expected = math.fsum(
        [
            factor.sampler_draw * ctl.done.sampler_draws,
            ctl.done.scored,
            factor.gradient_response * ctl.done.gradient_responses,
        ]
    )
    assert ctl.flops["every_position"] == pytest.approx(expected)
    assert ctl.flops["every_position"] > ctl.inferences
    # SFT with its setting of one trained layer, GFlowNet with its rank of 64.
    (sft,) = [run for run in runs_of(computed, "sft") if run.record.ledger_totals]
    one_layer = flops.factors(
        "last_layers",
        flops.passes(shape, prompt_tokens=PROMPT_TOKENS, response_tokens=50, trained_last_layers=1),
    )
    expected = sft.inferences + one_layer.gradient_response * sft.done.gradient_responses
    assert sft.flops["every_position"] == pytest.approx(expected)
    (gfn,) = runs_of(computed, "gflownet")
    assert gfn.flops["every_position"] > gfn.inferences + gfn.done.gradient_responses


def test_a_point_divides_the_summed_compute_by_the_summed_count(
    computed: tuple[list[compute.RunCompute], dict[str, int]],
) -> None:
    runs = runs_of(computed, "kl-dpg")
    assert len(runs) == 2
    point = compute.point_summary("kl-dpg", "both", runs)
    inferences = sum(run.inferences for run in runs)
    compute_sum = sum(run.inferences + 2 * run.done.gradient_responses for run in runs)
    assert point["runs"] == 2 and point["construction_inferences"] == inferences / 2
    assert point["flops_per_inference"] == pytest.approx(compute_sum / inferences)
    times = sum(run.gpu_time for run in runs if run.gpu_time is not None)
    assert point["gpu_time_per_inference"] == pytest.approx(times / inferences)


def test_a_method_has_its_largest_point(
    computed: tuple[list[compute.RunCompute], dict[str, int]],
) -> None:
    summary = compute.summarize(computed[0])
    points = {(point["method"], point["setting"]): point for point in summary["points"]}
    assert len(points) == len({(run.record.method, run.record.setting) for run in computed[0]})
    methods = {method["method"]: method for method in summary["methods"]}
    kl = methods["kl-dpg"]
    of_kl = [points[("kl-dpg", setting)] for setting in ("B=3000", "B=6000")]
    assert (kl["points"], kl["runs"]) == (2, 2)
    assert kl["flops_per_inference"] == max(point["flops_per_inference"] for point in of_kl)
    assert kl["flops_per_inference_at"] in ("B=3000", "B=6000")
    assert 1 < kl["flops_per_inference"] < 2 and kl["gpu_time_per_inference"] > 0
    assert methods["replay"]["flops_per_inference"] == 1
    # SFT's run without a sampler is charged its budget and computed less.
    assert points[("sft", "B=2048")]["flops_per_inference"] < 1
    assert methods["sft"]["flops_per_inference_at"] == "B=6000"


def test_a_point_holds_one_record_for_a_task_and_replicate(
    finished: list[records.Record], shape: flops.Shape
) -> None:
    """The records of `run` and of a rule of `reselect` have the same labels: together they
    would count one run several times."""
    with pytest.raises(records.RecordsError, match="more than one record of replicate 0"):
        compute.of_records([*finished, finished[0]], shape, prompt_tokens=lambda _: 12, times=None)


def test_without_times_there_are_the_flops_alone(
    finished: list[records.Record], shape: flops.Shape
) -> None:
    runs, _ = compute.of_records(finished, shape, prompt_tokens=lambda _: 12, times=None)
    assert all(run.gpu_time is None for run in runs)
    summary = compute.summarize(runs)
    assert all("gpu_time_per_inference" not in entry for entry in summary["points"])
    assert all("gpu_time_per_inference" not in run.to_json() for run in runs)
    assert all("flops_per_inference" in entry for entry in summary["methods"])
