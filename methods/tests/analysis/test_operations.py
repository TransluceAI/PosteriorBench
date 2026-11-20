"""What a run did, read from its record: on the records of small runs of every family that
trains, and on records written here for the methods that train nothing."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from posteriorbench import ledger

from posteriorbench_methods.analysis import operations, records
from posteriorbench_methods.common import run_record


def record_of(found: list[records.Record], method: str, setting: str) -> records.Record:
    (record,) = [r for r in found if (r.method, r.setting) == (method, setting)]
    return record


def written(
    directory: pathlib.Path, method: str, spend: ledger.Ledger, details: dict[str, Any]
) -> records.Record:
    """A record written here: a result with the totals of `spend`, and the details."""
    totals = spend.totals()
    result = {"method": method, "construction_inferences": totals.inferences}
    run_record.write_json(directory / "result.json", {**result, "ledger": totals.to_dict()})
    run_record.write_json(directory / "construction.json", {"details": details})
    (record,) = records.find([directory])
    return record


def test_the_methods_with_a_rule() -> None:
    assert operations.trains("sft") == "last_layers"
    assert operations.trains("gflownet") == "adapters"
    assert operations.trains("ctl") == "twist"
    assert {operations.trains(m) for m in ("kl-dpg", "js-dpg", "gdc")} == {"all_weights"}
    for method in ("replay", "prompt-ebjc", "prompt-hitrate", "mixture-eb16"):
        assert operations.trains(method) == "nothing"
    for method in ("sft+composition", "mixture-eb16+composition", "steering", "instruction"):
        assert operations.trains(method) is None


def test_the_counted_operations_add_up_to_the_inference_count(
    finished: list[records.Record],
) -> None:
    """For every run with a sampler, of every method with a rule."""
    covered = [r for r in finished if operations.trains(r.method) and r.ledger_totals]
    assert {r.method for r in covered} == {"sft", "replay", "gflownet", "ctl", "kl-dpg", "gdc"}
    for record in covered:
        assert operations.of_record(record).counted == record.construction_inferences


def test_the_responses_of_gradient_steps_of_each_method(finished: list[records.Record]) -> None:
    sft = record_of(finished, "sft", "B=6000")
    done = operations.of_record(sft)
    division = sft.details["division"]
    assert done.draws_from_p == division["draws"]
    assert done.gradient_responses == done.training_forwards == division["training_examples"]
    assert (done.update_batch, done.updates) == (
        division["batch"],
        division["checkpoint_updates"][-1],
    )
    # GFlowNet, two updates of two groups of 20 responses: the fresh ones are generated, the
    # replayed ones have a counted forward pass, and all of them have a backward pass.
    done = operations.of_record(record_of(finished, "gflownet", "u=2"))
    assert (done.gradient_responses, done.updates) == (80, 2)
    assert done.training_forwards < done.gradient_responses
    # CTL, two updates on 64 responses: no counted forward pass.
    done = operations.of_record(record_of(finished, "ctl", "u=2"))
    assert (done.gradient_responses, done.training_forwards, done.updates) == (128, 0, 0)
    # KL-DPG, 32 responses in an update: every gradient step has a counted forward pass.
    done = operations.of_record(record_of(finished, "kl-dpg", "B=6000"))
    assert done.gradient_responses == done.training_forwards == 32 * done.updates > 0
    # GDC trains on its hits, and every update takes a step.
    gdc = record_of(finished, "gdc", "B=6000")
    done = operations.of_record(gdc)
    assert done.gradient_responses == done.training_forwards > 0
    assert done.updates == gdc.details["candidates"][-1]["update"] > 0
    for method, setting in (("gflownet", "u=2"), ("ctl", "u=2"), ("gdc", "B=6000")):
        assert operations.of_record(record_of(finished, method, setting)).draws_from_p == 0


def test_a_run_without_a_sampler_did_its_draws_alone(finished: list[records.Record]) -> None:
    """SFT and Replay at a budget that pays for no hit: charged the budget, having drawn less."""
    for method in ("sft", "replay"):
        record = record_of(finished, method, "B=2048")
        assert record.result["cost_rule"] == "fallback_at_budget" and record.ledger_totals is None
        done = operations.of_record(record)
        assert done == operations.Operations(draws_from_p=record.details["division"]["draws"])
        assert done.counted < record.construction_inferences == 2048


def test_a_method_that_trains_nothing_draws_from_the_original_model(
    tmp_path: pathlib.Path,
) -> None:
    spend = ledger.Ledger()
    spend.charge(ledger.Category.GENERATION, inferences=64)
    spend.charge(ledger.Category.P_SCORING, inferences=9)
    spend.charge(ledger.Category.Q_SCORING, inferences=3)
    spend.charge(ledger.Category.THRESHOLD_TEST, inferences=164)
    done = operations.of_record(written(tmp_path, "prompt-ebjc", spend, {}))
    assert done == operations.Operations(draws_from_p=228, scored=12)
    assert done.counted == spend.inferences


def test_a_method_or_a_charge_without_a_rule_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(operations.NotCoveredError, match="sft\\+composition"):
        operations.of_record(written(tmp_path / "a", "sft+composition", ledger.Ledger(), {}))
    spend = ledger.Ledger()
    spend.charge(ledger.Category.SHARED, inferences=100)
    with pytest.raises(operations.NotCoveredError, match="shared construction"):
        operations.of_record(written(tmp_path / "b", "prompt-ebjc", spend, {}))
