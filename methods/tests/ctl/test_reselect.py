"""Choosing the checkpoint of a finished CTL training run again: with a 1% threshold, and by
CTL's own rule, the final checkpoint."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import (
    rank_table,
    reselection,
    run_context,
    run_record,
    seeds,
    training_run,
)
from posteriorbench_methods.ctl import reselect
from posteriorbench_methods.ctl import run as ctl_run
from posteriorbench_methods.ctl import selection
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import training, twisted_model

Context = Callable[..., run_context.RunContext]
Quick = Callable[..., ctl_settings.Settings]


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def files(root: pathlib.Path) -> dict[str, bytes]:
    """Every file under `root`, by its path."""
    return {str(path): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_reselect_repeats_the_submission_rule_with_a_1pct_threshold(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(2,), budgets=(3500,))
    context = make_context(tmp_path)
    of_run = ctl_run.run(context, settings)
    before = files(tmp_path)
    summaries = reselect.reselect(context, settings)
    assert [(s.method, s.setting) for s in summaries] == [("ctl", "u=2"), ("ctl", "B=3500")]
    # The run's records and its training run are as they were.
    assert {path: data for path, data in files(tmp_path).items() if path in before} == before
    root = reselection.records_root(tmp_path, "threshold-1pct")
    assert run_record.read_jsonl(root / "summary.jsonl") == [s.to_json() for s in summaries]
    at_rank_1 = rank_table.threshold_test(1, threshold=0.01)
    for plain, summary in zip(of_run, summaries, strict=True):
        assert summary.record == plain.record
        directory = root / summary.record
        built = read(directory / "construction.json")
        details = built["details"]
        assert list(details)[0] == "threshold" and details.pop("threshold") == 0.01
        assert details == read(tmp_path / plain.record / "construction.json")["details"]
        candidates = details["candidates"]
        assert len(built["ranking"]) == len(candidates)
        for entry in built["ranking"]:
            above = entry["selection_hits"] / entry["selection_draws"] > 0.015
            assert entry["group"] == ("by_signal" if above else "by_hits")
        first = built["submission"]["tests"][0]
        assert (first["draws"], first["required_hits"]) == (
            at_rank_1.draws,
            at_rank_1.required_hits,
        )
        # The cost: the training up to the last candidate and the tests at 1%.
        tests = built["submission"]["test_draws"]
        result = read(directory / "result.json")
        assert built["ledger"]["resumed_from"] == candidates[-1]["ledger"]
        assert result["construction_inferences"] == candidates[-1]["ledger"]["inferences"] + tests
        assert result["budget_passes"] == candidates[-1]["budget_spent"] + tests
        assert (directory / "draws" / "selection.jsonl").read_bytes() == (
            tmp_path / plain.record / "draws" / "selection.jsonl"
        ).read_bytes()
        test_rows = run_record.read_jsonl(directory / "draws" / "threshold_tests.jsonl")
        assert len(test_rows) == tests
        # The tests at 1% draw from streams of their own, in parts of 32 like those of a run.
        by_run = run_record.read_jsonl(tmp_path / plain.record / "draws" / "threshold_tests.jsonl")
        assert {row["seed"] for row in test_rows}.isdisjoint(row["seed"] for row in by_run)
        rank_1 = [row for row in test_rows if row["rank"] == 1]
        assert [row["seed"] - rank_1[0]["seed"] for row in rank_1] == [
            32 * (index // 32) for index in range(at_rank_1.draws)
        ]
        assert built["submitted"]["checkpoint_id"] == built["submitted"]["candidate_id"]


def test_the_tests_at_1pct_are_not_held_to_the_budget(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    """A run's ledger has its budget as its limit, which the tests of a run stay within. One test
    at 1% draws 1,857 responses, more than the budget here has room for, so the ledger of the
    tests at 1% has no limit."""
    settings = quick(update_caps=(2,))
    context = make_context(tmp_path)
    ctl_run.run(context, settings)
    checkpoints, twisted = reselect.open_training_run(context, settings)
    (point,) = selection.points(settings)
    at_cap = selection.candidates_at(point, checkpoints, settings)
    spent = at_cap.checkpoints[-1].budget_spent
    candidates = dataclasses.replace(at_cap, budget=int(spent) + 1000)
    ranking = selection.rank(context, candidates.checkpoints, reselect.RULE_1PCT)
    with pytest.raises(ledger.BudgetExceeded):
        selection.submit(
            context, twisted, candidates, ranking, reselect.RULE_1PCT, stream="threshold-test-1pct"
        )
    result, spend, _ = selection.submit(
        context,
        twisted,
        candidates,
        ranking,
        reselect.RULE_1PCT,
        stream="threshold-test-1pct",
        within_budget=False,
    )
    assert spend.budget_limit is None
    assert spend.budget == spent + result.test_draws > candidates.budget


def test_the_own_rule_submits_the_final_checkpoint(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(1, 2), budgets=(3500,))
    context = make_context(tmp_path)
    of_run = ctl_run.run(context, settings)
    before = files(tmp_path)
    summaries = reselect.final_checkpoint(context, settings)
    assert [(s.method, s.setting) for s in summaries] == [(s.method, s.setting) for s in of_run]
    assert {path: data for path, data in files(tmp_path).items() if path in before} == before
    root = reselection.records_root(tmp_path, "final-checkpoint")
    for plain, summary in zip(of_run, summaries, strict=True):
        directory = root / summary.record
        built = read(directory / "construction.json")
        details = built["details"]
        assert list(details)[0] == "own_rule" and details.pop("own_rule") == "final-checkpoint"
        assert details == read(tmp_path / plain.record / "construction.json")["details"]
        last = details["candidates"][-1]
        assert built["submitted"] == {
            "checkpoint_id": last["checkpoint_id"],
            "update": last["update"],
        }
        # No ranking and no threshold test: the cost is the training up to the last candidate.
        assert built["status"] == "own_rule"
        assert built["ranking"] is None and built["submission"] is None
        result = read(directory / "result.json")
        assert result["construction_inferences"] == last["ledger"]["inferences"]
        assert result["budget_passes"] == last["budget_spent"]
        assert built["ledger"]["charges"] == []
        names = {path.name for path in (directory / "draws").iterdir()}
        assert names == {"evaluation.jsonl"}
        assert result["fallback"] == (result["native_hit_rate"] < 0.10)
    # At the update cap 2 the final checkpoint is the one saved after update 2; at the budget, the
    # checkpoint that closed it.
    submitted = [
        read(root / summary.record / "construction.json")["submitted"]["update"]
        for summary in summaries
    ]
    index = training_run.read_index(training.directory(context))
    closing = next(row for row in index if 3500 in row.boundary_for)
    assert submitted == [1, 2, closing.update]


def test_the_final_checkpoint_is_evaluated_with_its_own_weights(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    """The evaluation draws are those of a twisted model that holds the final checkpoint."""
    settings = quick(update_caps=(2,))
    context = make_context(tmp_path)
    ctl_run.run(context, settings)
    (summary,) = reselect.final_checkpoint(context, settings)
    root = reselection.records_root(tmp_path, "final-checkpoint")
    evaluation = run_record.read_jsonl(root / summary.record / "draws" / "evaluation.jsonl")
    twisted = twisted_model.build(context)
    training.load_checkpoint(context, twisted, "update_000002")
    sampler = twisted_model.TwistedSampler(twisted)
    result = context.bench.evaluate(
        context.task,
        sampler,
        context.original,
        ledger=ledger.Ledger(),
        seed=0,
        settings=context.options.evaluation_settings(),
        keep_details=True,
    )
    assert [row["token_ids"] for row in evaluation] == [
        list(ids) for ids in (result.details or {})["draw_token_ids"]
    ]


def test_the_random_streams_of_reselect(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tests at 1% draw from streams of their own, indexed by the point and the rank like
    those of a run. The own rule draws nothing but the evaluation. Both build a new twisted
    model, whose twist the checkpoints then replace."""
    context = make_context(tmp_path)
    settings = quick(update_caps=(2,), budgets=(3500,))
    ctl_run.run(context, settings)
    used: set[tuple[Any, ...]] = set()
    stream_seed = seeds.stream_seed

    def recording(
        owner: str, task_id: str, replicate: int | None, stream: str, *index: str | int
    ) -> int:
        used.add((owner, stream, *(type(part) for part in index)))
        return stream_seed(owner, task_id, replicate, stream, *index)

    monkeypatch.setattr(seeds, "stream_seed", recording)
    reselect.reselect(context, settings)
    assert used == {
        ("ctl", "initialization"),
        ("ctl", "threshold-test-1pct", str, int),
        ("ctl", "threshold-test-1pct", int, int),
    }
    used.clear()
    reselect.final_checkpoint(context, settings)
    assert used == {("ctl", "initialization")}


def test_reselect_needs_a_training_run_that_went_as_far_as_its_points(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    for rule in (reselect.reselect, reselect.final_checkpoint):
        with pytest.raises(reselection.NoTrainingRunError, match="holds no training run"):
            rule(context, quick(update_caps=(2,)))
    assert not list(tmp_path.iterdir())
    ctl_run.run(context, quick(update_caps=(2,)))
    before = files(tmp_path)
    for rule in (reselect.reselect, reselect.final_checkpoint):
        with pytest.raises(reselection.NoTrainingRunError, match="has not trained as far"):
            rule(context, quick(update_caps=(4,)))
        with pytest.raises(training_run.TrainingRunError, match="other settings"):
            rule(context, quick(update_caps=(2,), epsilon=1e-12))
    assert files(tmp_path) == before


def test_a_checkpoint_file_that_a_run_deleted_is_reported(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    settings = quick(update_caps=(2,))
    ctl_run.run(context, settings)
    for path in (training.directory(context) / "checkpoints").iterdir():
        path.unlink()
    for rule in (reselect.reselect, reselect.final_checkpoint):
        with pytest.raises(training_run.TrainingRunError, match="--keep-checkpoints all"):
            rule(context, settings)
