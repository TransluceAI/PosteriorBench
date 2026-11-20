"""A whole run on the tiny model: training, submission, evaluation, composition, records."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

from posteriorbench_methods.common import composition, run_context, run_record, training_run
from posteriorbench_methods.gfn import cli
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import run as gfn_run
from posteriorbench_methods.gfn import selection
from posteriorbench_methods.gfn import settings as gfn_settings

Context = Callable[..., run_context.RunContext]
Quick = Callable[..., gfn_settings.Settings]
TASK_DIR = "wc_1853__goblin_mention"


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def record(out: pathlib.Path, method: str, setting: str) -> pathlib.Path:
    return out / method / setting / TASK_DIR / "rep0"


def test_a_run_writes_one_record_at_each_update_cap_and_budget(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(1, 4), budgets=(4096,))
    summaries = gfn_run.run(make_context(tmp_path), settings)
    assert [(s.method, s.setting) for s in summaries] == [
        ("gflownet", "u=1"),
        ("gflownet", "u=4"),
        ("gflownet", "B=4096"),
    ]
    lines = run_record.read_jsonl(tmp_path / "summary.jsonl")
    assert [line["record"] for line in lines] == [s.record for s in summaries]
    index = training_run.read_index(tmp_path / "training_runs" / "gflownet" / TASK_DIR / "rep0")
    for summary in summaries:
        directory = tmp_path / summary.record
        built = read(directory / "construction.json")
        result = read(directory / "result.json")
        assert read(directory / "result_recomputed.json")["reference_logp"] == "recomputed"
        assert read(directory / "config.json")["settings"]["groups_per_update"] == 2
        assert built["status"] in ("passed", "best_observed")
        # The candidates: the checkpoints of the first u updates, or those saved within B - 1,912.
        candidates = [training_run.Checkpoint.from_json(c) for c in built["details"]["candidates"]]
        assert all(row in index for row in candidates)
        if "update_cap" in built["details"]:
            assert [row.update for row in candidates] == [
                row.update for row in index if row.update <= built["details"]["update_cap"]
            ]
        else:
            assert candidates == [row for row in index if row.budget_spent <= 4096 - 1912]
            assert result["budget_passes"] <= 4096
        assert len(built["ranking"]) == len(candidates)
        # The cost: the training up to the last candidate, and the threshold tests.
        tests = built["submission"]["test_draws"]
        assert result["construction_inferences"] == candidates[-1].ledger["inferences"] + tests
        assert result["budget_passes"] == candidates[-1].budget_spent + tests
        assert built["ledger"]["resumed_from"] == candidates[-1].ledger
        submitted = built["submitted"]
        assert submitted["candidate_id"] == f"update_{submitted['update']:06d}"
        selection = run_record.read_jsonl(directory / "draws" / "selection.jsonl")
        assert [row["candidate_id"] for row in selection[::32]] == [
            row.checkpoint_id for row in candidates
        ]
        assert sum(row["hit"] for row in selection) == sum(c.selection_hits for c in candidates)
        test_rows = run_record.read_jsonl(directory / "draws" / "threshold_tests.jsonl")
        assert len(test_rows) == tests
        assert [row["rank"] for row in test_rows] == sorted(row["rank"] for row in test_rows)
        for test in built["submission"]["tests"]:
            rows = [row for row in test_rows if row["rank"] == test["rank"]]
            assert sum(row["hit"] for row in rows) == test["hits"] and len(rows) == test["draws"]
            assert {row["candidate_id"] for row in rows} == {test["candidate_id"]}
        evaluation = run_record.read_jsonl(directory / "draws" / "evaluation.jsonl")
        assert len(evaluation) == 64 and all(row["token_ids"] for row in evaluation)


def test_a_checkpoint_that_is_not_composed_repeats_the_plain_record(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    """Update cap 1: the only candidate is the original model, which is never composed."""
    summaries = gfn_run.run(make_context(tmp_path, compose=True), quick(update_caps=(1,)))
    assert [s.method for s in summaries] == ["gflownet", "gflownet+composition"]
    plain = read(record(tmp_path, "gflownet", "u=1") / "result.json")
    directory = record(tmp_path, "gflownet+composition", "u=1")
    result, built = read(directory / "result.json"), read(directory / "construction.json")
    assert built["status"] == "not_composed" and built["ranking"] is None
    # With one candidate the test may pass or not; the original model is not composed either way.
    assert built["details"]["composition"]["reason"] in ("original_model", "plain_not_passed")
    assert built["submitted"] is None
    assert built["details"]["plain"]["submitted"]["checkpoint_id"] == "update_000001"
    assert result["method"] == "gflownet+composition"
    same = ("construction_inferences", "budget_passes", "capped_forward_kl", "reverse_kl", "hits")
    assert {key: result[key] for key in same} == {key: plain[key] for key in same}
    assert run_record.read_jsonl(directory / "draws" / "evaluation.jsonl") == run_record.read_jsonl(
        record(tmp_path, "gflownet", "u=1") / "draws" / "evaluation.jsonl"
    )


def test_composition_adds_its_selection_draws_and_tests_to_the_cost(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The submitted checkpoint is composed here whether or not it passed its test."""
    monkeypatch.setattr(composition, "should_compose", lambda status, **_: True)
    context = make_context(tmp_path, compose=True)
    plain_summary, summary = gfn_run.run(context, quick(update_caps=(2,)))
    plain = read(tmp_path / plain_summary.record / "result.json")
    directory = tmp_path / summary.record
    result, built = read(directory / "result.json"), read(directory / "construction.json")
    assert result["method"] == "gflownet+composition" and result["setting"] == "u=2"
    assert built["details"]["update_cap"] == 2
    assert built["details"]["plain"]["status"] in ("passed", "best_observed")
    outcome = built["details"]["composition"]
    assert built["status"] == outcome["submission"]["status"]
    assert len(built["ranking"]) == 16
    limit = context.original.max_response_tokens
    by_index = {str(index): setting for index, setting in enumerate(outcome["settings"])}
    tests = 0
    for test in outcome["submission"]["tests"]:
        setting = by_index[test["candidate_id"]]
        is_plain = setting["alpha"] == 0 and setting["switch"] >= limit
        tests += test["draws"] * (1 if is_plain else 2)
    hits = sum(outcome["selection_hits"])
    assert (
        result["construction_inferences"] - plain["construction_inferences"] == 496 + hits + tests
    )
    assert result["budget_passes"] - plain["budget_passes"] == 496 + 256 + tests
    assert result["inferences_per_response"] == (1.0 if outcome["winner_is_plain"] else 2.0)
    assert built["submitted"]["setting"] == outcome["winner"]["setting"]
    selection_rows = run_record.read_jsonl(directory / "draws" / "composition_selection.jsonl")
    assert len(selection_rows) == 16 * 16 and sum(row["hit"] for row in selection_rows) == hits
    assert {row["setting"] for row in selection_rows} == {s["setting"] for s in outcome["settings"]}
    test_rows = run_record.read_jsonl(directory / "draws" / "composition_threshold_tests.jsonl")
    assert len(test_rows) == outcome["submission"]["test_draws"]


def test_a_composition_that_chooses_the_plain_setting_repeats_the_evaluation(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composition, "should_compose", lambda status, **_: True)
    choose = composition.choose_setting

    def choose_the_plain_setting(*args: Any, **kwargs: Any) -> Any:
        outcome = choose(*args, **kwargs)
        plain = next(setting for setting in outcome.settings if setting.is_plain(8))
        return dataclasses.replace(outcome, winner=plain, winner_is_plain=True)

    monkeypatch.setattr(composition, "choose_setting", choose_the_plain_setting)
    plain_summary, summary = gfn_run.run(
        make_context(tmp_path, compose=True), quick(update_caps=(2,))
    )
    plain = read(tmp_path / plain_summary.record / "result.json")
    result = read(tmp_path / summary.record / "result.json")
    assert result["method"] == "gflownet+composition"
    assert result["inferences_per_response"] == 1.0
    assert result["construction_inferences"] > plain["construction_inferences"]
    same = ("capped_forward_kl", "reverse_kl", "hits", "native_hit_rate")
    assert {key: result[key] for key in same} == {key: plain[key] for key in same}
    assert (tmp_path / summary.record / "draws" / "composition_selection.jsonl").exists()


def test_checkpoint_files_that_no_test_can_reach_are_deleted(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    """Twelve candidates, of which the threshold tests can reach eight."""
    settings = quick(update_caps=(12,), checkpoint_updates=tuple(range(1, 13)))
    kept: dict[str, set[str]] = {}
    for keep in ("all", "testable", "returned"):
        out = tmp_path / keep
        (summary,) = gfn_run.run(make_context(out, keep_checkpoints=keep), settings)
        directory = out / "training_runs" / "gflownet" / TASK_DIR / "rep0"
        kept[keep] = {path.stem for path in (directory / "checkpoints").iterdir()}
        built = read(out / summary.record / "construction.json")
        submitted = built["submitted"]["candidate_id"]
        if keep == "returned":
            assert kept[keep] == {submitted}
        if keep == "testable":
            assert kept[keep] == {entry["candidate_id"] for entry in built["ranking"][:8]}
            assert submitted in kept[keep]
    assert len(kept["testable"]) == 8 and len(kept["all"]) == 12
    # Loading a checkpoint whose file was deleted says what happened.
    context = make_context(tmp_path / "returned")
    directory = context.training_run_dir("gflownet")
    draws = selection.CheckpointDraws(
        directory, gfn_policy.build(context, settings), context.original.decode, bool
    )
    deleted = [
        row
        for row in training_run.read_index(directory)
        if row.checkpoint_id not in kept["returned"]
    ]
    with pytest.raises(training_run.TrainingRunError, match="was deleted"):
        draws.load(deleted[0])


def test_the_command_runs_gflownet(
    tiny_model: Any,
    tiny_tokenizer: Any,
    make_tiny_original_model: Callable[..., Any],
    make_tiny_dataset: Callable[..., pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    """`gfn run` loads the model by its path, so the task here has the benchmark's response
    limit of 50 tokens."""
    model_dir = tmp_path / "model"
    tiny_model.save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)
    dataset = make_tiny_dataset(make_tiny_original_model(max_response_tokens=50), draws=1500)
    arguments = ["run", "--task", "wc_1853/goblin_mention", "--out", str(tmp_path / "runs")]
    arguments += ["--data", str(dataset), "--model", str(model_dir), "--device", "cpu"]
    arguments += ["--evaluation-draws", "64", "--no-compose"]
    for setting in ("budgets=", "update_caps=1,2", "groups_per_update=2", "selection_draws=32"):
        arguments += ["--set", setting]
    result = CliRunner().invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    written = run_record.read_jsonl(tmp_path / "runs" / "summary.jsonl")
    assert [(row["method"], row["setting"]) for row in written] == [
        ("gflownet", "u=1"),
        ("gflownet", "u=2"),
    ]
    assert (record(tmp_path / "runs", "gflownet", "u=2") / "result.json").exists()
    assert not (tmp_path / "runs" / "gflownet+composition").exists()
