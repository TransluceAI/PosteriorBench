"""A whole run on the tiny model: training, submission, evaluation, composition, records."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.common import composition, run_context, run_record, training_run
from posteriorbench_methods.ctl import composition as ctl_composition
from posteriorbench_methods.ctl import run as ctl_run
from posteriorbench_methods.ctl import selection
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import training, twisted_model

Context = Callable[..., run_context.RunContext]
Quick = Callable[..., ctl_settings.Settings]
TASK_DIR = "wc_1853__goblin_mention"


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def record(out: pathlib.Path, method: str, setting: str) -> pathlib.Path:
    return out / method / setting / TASK_DIR / "rep0"


def checkpoint_files(context: run_context.RunContext) -> list[str]:
    return sorted(path.stem for path in (training.directory(context) / "checkpoints").iterdir())


def test_a_run_writes_one_record_at_each_update_cap_and_budget(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    summaries = ctl_run.run(context, quick(update_caps=(1, 2), budgets=(3500,)))
    assert [(s.method, s.setting) for s in summaries] == [
        ("ctl", "u=1"),
        ("ctl", "u=2"),
        ("ctl", "B=3500"),
    ]
    lines = run_record.read_jsonl(tmp_path / "summary.jsonl")
    assert [line["record"] for line in lines] == [s.record for s in summaries]
    index = training_run.read_index(training.directory(context))
    for summary in summaries:
        directory = tmp_path / summary.record
        built = read(directory / "construction.json")
        result = read(directory / "result.json")
        details = built["details"]
        assert read(directory / "result_recomputed.json")["reference_logp"] == "recomputed"
        assert read(directory / "config.json")["settings"]["responses_per_update"] == 64
        assert built["status"] in ("passed", "best_observed")
        # The candidates: the checkpoints of the first u updates, or those saved within B - 1,912.
        candidates = [training_run.Checkpoint.from_json(row) for row in details["candidates"]]
        if "update_cap" in details:
            assert candidates == [row for row in index if row.update <= details["update_cap"]]
            # The budget that the update cap stands for pays for exactly these.
            budget = details["equivalent_budget"]
            assert budget == 256 * details["update_cap"] + 96 * len(candidates) + 1912
        else:
            budget = details["budget"]
            assert candidates == [row for row in index if row.budget_spent <= budget - 1912]
            assert budget in candidates[-1].boundary_for
        assert len(built["ranking"]) == len(candidates)
        # The cost: the training up to the last candidate, and the threshold tests.
        tests = built["submission"]["test_draws"]
        assert result["construction_inferences"] == candidates[-1].ledger["inferences"] + tests
        assert result["budget_passes"] == candidates[-1].budget_spent + tests <= budget
        assert built["ledger"]["resumed_from"] == candidates[-1].ledger
        submitted = built["submitted"]
        assert submitted["candidate_id"] == submitted["checkpoint_id"]
        assert submitted["checkpoint_id"] == f"update_{submitted['update']:06d}"
        selection_rows = run_record.read_jsonl(directory / "draws" / "selection.jsonl")
        assert [row["candidate_id"] for row in selection_rows[::32]] == [
            row.checkpoint_id for row in candidates
        ]
        test_rows = run_record.read_jsonl(directory / "draws" / "threshold_tests.jsonl")
        assert len(test_rows) == tests
        for test in built["submission"]["tests"]:
            rows = [row for row in test_rows if row["rank"] == test["rank"]]
            assert [row["draw_index"] for row in rows] == list(range(test["draws"]))
            assert sum(row["hit"] for row in rows) == test["hits"]
            assert {row["candidate_id"] for row in rows} == {test["candidate_id"]}
            # The test draws in parts of 32, the part at draw i with the test's seed + i.
            first_seed = rows[0]["seed"]
            assert [row["seed"] - first_seed for row in rows] == [
                32 * (index // 32) for index in range(test["draws"])
            ]
        assert len(run_record.read_jsonl(directory / "draws" / "evaluation.jsonl")) == 64


def test_composition_adds_its_selection_draws_and_tests_to_the_cost(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, compose=True)
    summaries = ctl_run.run(context, quick(update_caps=(1,), budgets=(4300,)))
    assert [(s.method, s.setting) for s in summaries] == [
        ("ctl", "u=1"),
        ("ctl+composition", "u=1"),
        ("ctl", "B=4300"),
        ("ctl+composition", "B=4300"),
    ]
    limit = context.original.max_response_tokens
    composed_records = 0
    for setting in ("u=1", "B=4300"):
        plain = read(record(tmp_path, "ctl", setting) / "construction.json")
        plain_result = read(record(tmp_path, "ctl", setting) / "result.json")
        directory = record(tmp_path, "ctl+composition", setting)
        built = read(directory / "construction.json")
        result = read(directory / "result.json")
        assert built["details"]["plain"] == {
            "status": plain["status"],
            "submitted": plain["submitted"],
        }
        if plain["status"] != "passed":
            assert built["status"] == "not_composed" and built["submitted"] is None
            continue
        composed_records += 1
        outcome = built["details"]["composition"]
        settings = [
            composition.Setting(entry["alpha"], entry["switch"]) for entry in outcome["settings"]
        ]
        assert settings == list(composition.settings_grid(limit))
        test_inferences = sum(
            test["draws"] * settings[int(test["candidate_id"])].inferences_per_response(limit)
            for test in built["submission"]["tests"]
        )
        assert result["construction_inferences"] == (
            plain_result["construction_inferences"]
            + 496
            + sum(outcome["selection_hits"])
            + test_inferences
        )
        assert built["ledger"]["resumed_from"] == plain["ledger"]["totals"]
        winner = composition.Setting(outcome["winner"]["alpha"], outcome["winner"]["switch"])
        assert built["submitted"]["setting"] == winner.setting_id
        assert result["inferences_per_response"] == winner.inferences_per_response(limit)
        drawn = run_record.read_jsonl(directory / "draws" / "composition_selection.jsonl")
        assert [row["setting"] for row in drawn] == [
            setting.setting_id for setting in settings for _ in range(16)
        ]
        assert [sum(row["hit"] for row in drawn[i : i + 16]) for i in range(0, 256, 16)] == (
            outcome["selection_hits"]
        )
        tested = run_record.read_jsonl(directory / "draws" / "composition_threshold_tests.jsonl")
        assert len(tested) == built["submission"]["test_draws"]
    assert composed_records, "after seven updates the twisted model hits often: it passes"


def test_a_checkpoint_that_did_not_pass_is_not_composed(
    make_context: Context,
    quick: Quick,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed record then holds CTL's evaluation, at CTL's cost, in the layout of every
    record."""
    monkeypatch.setattr(composition, "should_compose", lambda status, **_: False)
    ctl_run.run(make_context(tmp_path, compose=True), quick(update_caps=(1,)))
    plain, composed = record(tmp_path, "ctl", "u=1"), record(tmp_path, "ctl+composition", "u=1")
    built = read(composed / "construction.json")
    assert built["status"] == "not_composed"
    assert list(built["details"]["composition"]) == ["reason"]
    assert built["ranking"] is None and built["submission"] is None
    assert built["ledger"]["totals"] == read(plain / "construction.json")["ledger"]["totals"]
    for name in ("result.json", "result_recomputed.json"):
        mine, theirs = read(composed / name), read(plain / name)
        assert "details" not in mine
        assert mine.pop("method") == "ctl+composition" and theirs.pop("method") == "ctl"
        assert mine == theirs
    assert (composed / "evaluation_details.json").exists()
    assert read(composed / "config.json")["scorer"] is not None
    assert run_record.read_jsonl(composed / "draws" / "evaluation.jsonl") == (
        run_record.read_jsonl(plain / "draws" / "evaluation.jsonl")
    )


def test_checkpoint_files_are_kept_as_the_option_says(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(1, 2, 4))
    all_files = ["update_000001", "update_000002", "update_000004"]
    kept_all = make_context(tmp_path / "all", keep_checkpoints="all")
    ctl_run.run(kept_all, settings)
    assert checkpoint_files(kept_all) == all_files
    # Three candidates: the threshold tests can reach each of them.
    testable = make_context(tmp_path / "testable", keep_checkpoints="testable")
    ctl_run.run(testable, settings)
    assert checkpoint_files(testable) == all_files
    returned = make_context(tmp_path / "returned", keep_checkpoints="returned")
    (summary,) = ctl_run.run(returned, quick(update_caps=(4,)))
    built = read(tmp_path / "returned" / summary.record / "construction.json")
    submitted = built["submitted"]["checkpoint_id"]
    assert checkpoint_files(returned) == [submitted]
    # A later run that tests a deleted checkpoint stops with an error that says what to do.
    deleted = next(name for name in all_files if name != submitted)
    with pytest.raises(training_run.TrainingRunError, match="--keep-checkpoints all"):
        training.load_checkpoint(returned, twisted_model.build(returned), deleted)


def test_a_threshold_test_stream_belongs_to_its_point() -> None:
    assert selection.Point(update_cap=4).stream_index == "u00004"
    assert selection.Point(budget=4).stream_index == 4
    assert selection.Point(update_cap=4).setting == "u=4"
    assert selection.Point(budget=2**20).setting == "B=1048576"
    with pytest.raises(ValueError):
        selection.Point()
    with pytest.raises(ValueError):
        selection.Point(update_cap=4, budget=2**20)


@pytest.mark.parametrize("alpha, switch", [(0.0, 8), (0.5, 4), (0.25, 2)])
def test_the_composed_sampler_scores_what_it_draws(
    alpha: float, switch: int, twisted: twisted_model.TwistedModel
) -> None:
    setting = composition.Setting(alpha, switch)
    source = ctl_composition.sampler(twisted, setting)
    draws = source.sample_draws(70, seed=11)
    responses = [draw.token_ids for draw in draws]
    assert source.logprobs(responses) == pytest.approx([draw.logq for draw in draws], abs=1e-4)
    assert source.sample(64, seed=11) == responses[:64]
    plain = twisted_model.TwistedSampler(twisted)
    if setting.is_plain(twisted.response_format.max_tokens):
        assert source.inferences_per_response == 1.0
        assert source.logprobs(responses) == pytest.approx(plain.logprobs(responses), abs=1e-5)
    else:
        assert source.inferences_per_response == 2.0
