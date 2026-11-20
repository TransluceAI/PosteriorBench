"""The re-selection of a finished run's checkpoint with a 1% threshold."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.common import rank_table, reselection, run_context, training_run
from posteriorbench_methods.sft import reselect, run, settings

BUDGET = 6000
SETTINGS = settings.Settings(budgets=(BUDGET,), trainable_last_layers=1)


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def test_reselect_lowers_the_threshold_for_the_same_training(
    make_context: Callable[..., run_context.RunContext],
    rare_dataset: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    context = make_context(tmp_path, rare_dataset, keep_checkpoints="all", compose=False)
    sft, _ = run.run(context, SETTINGS)
    (again,) = reselect.reselect(context, SETTINGS)
    assert again.record == f"sft/B={BUDGET}/wc_1853__goblin_mention/rep0"
    plain = read(tmp_path / sft.record / "construction.json")
    record = tmp_path / "reselected" / "threshold-1pct" / again.record
    assert record.parent.parent.parent.parent == reselection.records_root(
        tmp_path, reselection.THRESHOLD_1PCT
    )
    construction = read(record / "construction.json")
    assert construction["details"]["threshold"] == 0.01
    # About 5% of a checkpoint's draws are hits: it fails the tests and the evaluation at 10%,
    # and passes both at 1%.
    assert plain["status"] == "best_observed" and sft.fallback
    assert construction["status"] == "passed" and not again.fallback
    assert {entry["group"] for entry in construction["ranking"]} == {"by_signal"}
    first = construction["submission"]["tests"][0]
    at_rank_1 = rank_table.threshold_test(1, threshold=0.01)
    assert (first["draws"], first["required_hits"]) == (at_rank_1.draws, at_rank_1.required_hits)
    # The cost: the run until its training ended, and the tests at 1%.
    after_training = plain["details"]["candidates"][-1]["ledger"]["inferences"]
    assert again.construction_inferences == (
        after_training + construction["submission"]["test_draws"]
    )
    assert construction["ledger"]["resumed_from"]["inferences"] == after_training
    assert construction["details"]["candidates"] == plain["details"]["candidates"]
    # The record holds the selection draws that it ranked, those of the run.
    assert (record / "draws" / "selection.jsonl").read_text() == (
        tmp_path / sft.record / "draws" / "selection.jsonl"
    ).read_text()
    # The record of the run at 10% is untouched.
    assert read(tmp_path / sft.record / "construction.json") == plain


def test_reselect_needs_a_finished_run_with_the_same_settings(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, compose=False)
    with pytest.raises(reselection.NoTrainingRunError):
        reselect.reselect(context, SETTINGS)
    assert not list(tmp_path.iterdir())
    run.run(context, SETTINGS)
    other = settings.Settings(budgets=(BUDGET,), trainable_last_layers=1, training_share=0.125)
    with pytest.raises(training_run.TrainingRunError):
        reselect.reselect(context, other)
    # A budget that pays for no hit has no training run, whether or not `run` was there.
    small = settings.Settings(budgets=(2048,), trainable_last_layers=1)
    run.run(context, small)
    with pytest.raises(reselection.NoTrainingRunError, match="pays for no hit"):
        reselect.reselect(context, small)
