"""SFT, Replay and SFT + composition end to end on the tiny model: the records, their costs, and
what happens without a hit or without a checkpoint that passes."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.common import run_context, run_record, training_run
from posteriorbench_methods.sft import run, settings

TASK_DIR = "wc_1853__goblin_mention"
SMALL, BUDGET = 2048, 6000  # SMALL does not pay for a hit
SETTINGS = settings.Settings(budgets=(SMALL, BUDGET), trainable_last_layers=1, learning_rate=1e-3)
RECORD_FILES = {"result.json", "construction.json", "config.json"}


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def record(out: pathlib.Path, method: str, budget: int = BUDGET) -> pathlib.Path:
    return out / method / f"B={budget}" / TASK_DIR / "rep0"


@pytest.fixture(scope="module")
def finished(
    make_context: Callable[..., run_context.RunContext], tmp_path_factory: pytest.TempPathFactory
) -> tuple[pathlib.Path, list[run_record.RunSummary]]:
    """The output root and the summaries of one run at both budgets."""
    out = tmp_path_factory.mktemp("sft_run")
    return out, run.run(make_context(out), SETTINGS)


def test_a_run_writes_three_records_at_each_budget(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    assert [(s.method, s.setting) for s in summaries] == [
        (method, f"B={budget}")
        for budget in (SMALL, BUDGET)
        for method in ("sft", "replay", "sft+composition")
    ]
    assert [json.loads(line) for line in (out / "summary.jsonl").read_text().splitlines()] == [
        summary.to_json() for summary in summaries
    ]
    for summary in summaries:
        directory = out / summary.record
        assert RECORD_FILES <= {path.name for path in directory.iterdir()}
        assert (directory / "result_recomputed.json").exists() != summary.fallback
        config = read(directory / "config.json")
        assert config["family"] == "sft" and config["method"] == summary.method
        assert config["settings"]["budgets"] == [SMALL, BUDGET]
        assert config["settings"]["learning_rate"] == 1e-3
    sft = record(out, "sft")
    assert {path.stem for path in (sft / "draws").iterdir()} == {
        "training_set",
        "selection",
        "threshold_tests",
        "evaluation",
    }
    for path in (sft / "draws").iterdir():
        assert all("token_ids" in row for row in run_record.read_jsonl(path))


def test_a_budget_without_a_hit_is_the_fallback_at_cost_b(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    for summary in summaries[:3]:
        assert summary.fallback and summary.kind == "no_sampler"
        assert summary.construction_inferences == SMALL
        construction = read(out / summary.record / "construction.json")
        assert construction["ledger"] is None
        if summary.method == "sft+composition":
            assert construction["status"] == "not_composed"
            assert construction["details"]["plain"]["status"] == "no_sampler"
            assert construction["details"]["composition"] == {"reason": "plain_not_passed"}
        else:
            assert construction["status"] == "no_sampler"
            assert construction["details"]["division"]["hits"] == 0
    assert not (out / "training_runs" / "sft" / TASK_DIR / "rep0" / f"B={SMALL}").exists()


def test_sft_costs_its_draws_training_selection_draws_and_tests(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    construction = read(record(out, "sft") / "construction.json")
    division = construction["details"]["division"]
    checkpoints = construction["details"]["candidates"]
    assert [row["update"] for row in checkpoints] == division["checkpoint_updates"]
    selection = sum(row["selection_draws"] + row["selection_hits"] for row in checkpoints)
    tests = construction["submission"]["test_draws"]
    sft = summaries[3]
    assert sft.construction_inferences == (
        division["draws"] + division["training_examples"] + selection + tests
    )
    own_log_q = sum(row["selection_draws"] for row in checkpoints)
    assert sft.budget_passes == (
        division["draws"] + 2 * division["training_examples"] + selection + own_log_q + tests
    )
    assert sft.budget_passes is not None and sft.budget_passes <= BUDGET
    assert construction["ledger"]["totals"]["inferences"] == sft.construction_inferences
    # The submitted checkpoint is the first of the ranking that passed its test.
    assert construction["status"] == "passed"
    passed = construction["submission"]["tests"][-1]
    assert construction["submitted"]["checkpoint_id"] == passed["candidate_id"]
    assert construction["submitted"]["update"] in division["checkpoint_updates"]
    assert construction["ranking"][passed["rank"] - 1]["candidate_id"] == passed["candidate_id"]


def test_replay_returns_the_training_set_and_costs_the_draws(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    replay = summaries[4]
    result = read(record(out, "replay") / "result.json")
    construction = read(record(out, "replay") / "construction.json")
    division = construction["details"]["division"]
    assert construction["status"] == "no_selection"
    assert replay.construction_inferences == replay.budget_passes == division["draws"]
    assert result["exact"] and replay.native_hit_rate == 1.0
    assert result["inferences_per_response"] == 0.0
    training_set = run_record.read_jsonl(record(out, "replay") / "draws" / "training_set.jsonl")
    assert len(training_set) == division["hits"]
    places = [row["draw_index"] for row in training_set]
    assert places == sorted(set(places)) and places[-1] < division["draws"]
    assert training_set == run_record.read_jsonl(
        record(out, "sft") / "draws" / "training_set.jsonl"
    )


def test_composition_adds_its_selection_draws_and_tests_to_the_cost(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    sft, composed = summaries[3], summaries[5]
    construction = read(record(out, "sft+composition") / "construction.json")
    outcome = construction["details"]["composition"]
    assert construction["status"] == "passed"
    assert construction["details"]["plain"] == {
        "status": "passed",
        "submitted": read(record(out, "sft") / "construction.json")["submitted"],
    }
    plain_settings = {
        s["setting"] for s in outcome["settings"] if s["alpha"] == 0 and s["switch"] >= 8
    }
    by_index = [s["setting"] for s in outcome["settings"]]
    tests = sum(
        t["draws"] * (1 if by_index[int(t["candidate_id"])] in plain_settings else 2)
        for t in construction["submission"]["tests"]
    )
    assert composed.construction_inferences == (
        sft.construction_inferences + 496 + sum(outcome["selection_hits"]) + tests
    )
    assert construction["submitted"]["setting"] == outcome["winner"]["setting"]
    selection = run_record.read_jsonl(
        record(out, "sft+composition") / "draws" / "composition_selection.jsonl"
    )
    assert len(selection) == 16 * 16
    if outcome["winner_is_plain"]:
        assert composed.reverse_kl == sft.reverse_kl
    else:
        assert read(record(out, "sft+composition") / "result.json")["inferences_per_response"] == 2


def test_a_checkpoint_that_fails_its_tests_is_not_composed(
    make_context: Callable[..., run_context.RunContext],
    rare_dataset: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    context = make_context(tmp_path, rare_dataset, keep_checkpoints="returned")
    chosen = settings.Settings(budgets=(BUDGET,), trainable_last_layers=1)
    sft, _, composed = run.run(context, chosen)
    construction = read(record(tmp_path, "sft") / "construction.json")
    checkpoints = construction["details"]["candidates"]
    # No checkpoint has more than 15% hits among its selection draws: all are ranked by hits and
    # tested, and the one with the highest hit fraction in its test is submitted.
    assert {entry["group"] for entry in construction["ranking"]} == {"by_hits"}
    hits = [entry["selection_hits"] for entry in construction["ranking"]]
    assert hits == sorted(hits, reverse=True)
    tests = construction["submission"]["tests"]
    assert len(tests) == len(checkpoints) <= 8 and not any(t["passed"] for t in tests)
    assert construction["status"] == "best_observed"
    best = max(tests, key=lambda t: t["hits"] / t["draws"])
    assert construction["submitted"]["checkpoint_id"] == best["candidate_id"]
    # With `--keep-checkpoints returned` only the submitted checkpoint's file is left.
    directory = context.training_run_dir("sft", budget=BUDGET)
    assert [path.stem for path in (directory / "checkpoints").iterdir()] == [best["candidate_id"]]
    # The composed record is SFT's evaluation at SFT's cost.
    record_composed = read(record(tmp_path, "sft+composition") / "construction.json")
    assert record_composed["status"] == "not_composed"
    assert record_composed["details"]["plain"]["status"] == "best_observed"
    assert record_composed["details"]["composition"] == {"reason": "plain_not_passed"}
    assert composed.construction_inferences == sft.construction_inferences
    assert (composed.reverse_kl, composed.native_hit_rate) == (sft.reverse_kl, sft.native_hit_rate)
    for name in ("result.json", "result_recomputed.json"):
        assert read(record(tmp_path, "sft+composition") / name)["method"] == "sft+composition"


def test_a_second_run_uses_the_training_run_on_disk(
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen = settings.Settings(budgets=(BUDGET,), trainable_last_layers=1, learning_rate=1e-3)
    first = run.run(make_context(tmp_path, compose=False), chosen)
    assert [summary.method for summary in first] == ["sft", "replay"]
    monkeypatch.setattr(training_run, "save_checkpoint", None)  # a second training would fail
    second = run.run(make_context(tmp_path, compose=False), chosen)
    assert second == first
