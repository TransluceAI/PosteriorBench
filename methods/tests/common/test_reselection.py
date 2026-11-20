"""`common.reselection`: the 1% variant of a method's rule, reading a finished training run, and
the records of a checkpoint that was chosen again."""

from __future__ import annotations

import dataclasses
import fractions
import json
import math
import pathlib
from collections.abc import Callable
from typing import Any

import posteriorbench
import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import (
    rank_table,
    reselection,
    run_context,
    run_record,
    submission,
    training_run,
)

TASK_ID = "wc_1853/goblin_mention"
ToyContext = Callable[[pathlib.Path], run_context.RunContext]


def checkpoint(update: int, spent: float, hits: int = 0) -> training_run.Checkpoint:
    """A checkpoint saved after `update` updates, at a budget count of `spent` and half as many
    inferences."""
    training = ledger.Ledger()
    training.charge(ledger.Category.GENERATION, inferences=spent / 2, budget=spent)
    return training_run.Checkpoint(
        checkpoint_id=training_run.checkpoint_id(update),
        update=update,
        budget_spent=spent,
        ledger=training.totals().to_dict(),
        sampler_id=f"sampler{update}",
        selection_draws=128,
        selection_hits=hits,
    )


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def test_the_1pct_rule_divides_the_threshold_by_ten_in_the_ranking_and_in_the_tests() -> None:
    of_run = submission.Rule(max_tests=8, representative="earliest")
    rule = submission.at_one_percent(of_run)
    assert (rule.threshold, rule.signal_hit_fraction) == (0.01, 0.015)
    assert dataclasses.replace(rule, threshold=0.10, signal_hit_fraction=0.15) == of_run
    assert reselection.EVALUATION_THRESHOLD_1PCT == fractions.Fraction(1, 100)
    # Two hits among 128 selection draws are above 1.5%, and 16 hits are not above 15%.
    candidates = [
        submission.Candidate("few", 0, "s0", 128, 2, score=1.0, tie_score=1.0),
        submission.Candidate("more", 1, "s1", 128, 16, score=0.0, tie_score=0.0),
    ]
    assert [entry.group for entry in submission.rank_candidates(candidates, rule)] == [
        "by_signal",
        "by_signal",
    ]
    by_hits = submission.rank_candidates(candidates, of_run)
    assert [(entry.candidate_id, entry.group) for entry in by_hits] == [
        ("more", "by_hits"),
        ("few", "by_hits"),
    ]
    # The tests at 1% draw 1,857 to 3,269 responses at the eight ranks that are tested.
    result = submission.submit(
        submission.rank_candidates(candidates, rule), lambda candidate, test: 0, rule
    )
    assert [test.draws for test in result.tests] == [
        rank_table.threshold_test(rank, threshold=0.01).draws for rank in (1, 2)
    ]
    assert result.tests[0].draws == 1857 and result.tests[0].required_hits == 28


def test_the_records_of_a_rule_have_an_output_root_of_their_own(tmp_path: pathlib.Path) -> None:
    root = reselection.records_root(tmp_path, reselection.FINAL_CHECKPOINT)
    assert root == tmp_path / "reselected" / "final-checkpoint"
    assert reselection.records_root(tmp_path, reselection.THRESHOLD_1PCT) == (
        tmp_path / "reselected" / "threshold-1pct"
    )


def test_reading_a_training_run_trains_and_writes_nothing(tmp_path: pathlib.Path) -> None:
    settings = {"task_id": TASK_ID, "replicate": 0, "learning_rate": 1e-4}
    request = training_run.Request(budgets=(4096,), reserve=1912)
    with pytest.raises(reselection.NoTrainingRunError, match="holds no training run"):
        reselection.read_training_run(tmp_path, settings, request)
    assert not list(tmp_path.iterdir())
    # A training run that stopped before the budget's candidates were final.
    training_run.check_settings(tmp_path, settings)
    saved = [checkpoint(1, 500.0), checkpoint(2, 1000.0)]
    training_run.write_index(tmp_path, saved)
    training_run.write_done(tmp_path, training_run.Done("stopped", 2, 1500.0))
    with pytest.raises(reselection.NoTrainingRunError, match="has not trained as far"):
        reselection.read_training_run(tmp_path, settings, request)
    assert training_run.read_done(tmp_path) is not None
    # One that was interrupted has no done.json.
    (tmp_path / training_run.DONE).unlink()
    with pytest.raises(reselection.NoTrainingRunError):
        reselection.read_training_run(tmp_path, settings, request)
    training_run.write_done(tmp_path, training_run.Done("stopped", 2, 2500.0))
    assert reselection.read_training_run(tmp_path, settings, request) == saved
    with pytest.raises(training_run.TrainingRunError, match="other settings"):
        reselection.read_training_run(tmp_path, settings | {"learning_rate": 1e-3}, request)
    assert read(tmp_path / training_run.SETTINGS) == settings


def test_the_spend_after_a_checkpoint_has_no_limit() -> None:
    spend = reselection.spend_after(checkpoint(4, 3000.0))
    assert (spend.inferences, spend.budget, spend.budget_limit) == (1500.0, 3000.0, None)
    assert spend.resumed_from is not None and spend.charges() == ()


def test_the_final_checkpoint_is_the_last_candidate() -> None:
    candidates = [checkpoint(1, 500.0), checkpoint(2, 1000.0), checkpoint(4, 2000.0)]
    assert reselection.final_checkpoint(candidates) is candidates[-1]


def test_a_record_at_the_1pct_threshold_is_evaluated_with_that_threshold(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    """A sampler that hits on about 5% of its draws is scored as the fallback at 10% and passes
    at 1%."""
    context = toy_context(tmp_path)
    sampler = posteriorbench.FiniteSampler(
        [([1, 0], 0.05), ([2, 0], 0.95)], inferences_per_response=1
    )
    candidates = [checkpoint(1, 500.0, hits=4), checkpoint(2, 1000.0, hits=6)]
    rule = submission.at_one_percent(submission.Rule(max_tests=8))
    ranking = submission.rank_candidates(
        [
            submission.Candidate(
                row.checkpoint_id,
                order,
                row.sampler_id,
                128,
                row.selection_hits,
                score=float(order),
            )
            for order, row in enumerate(candidates)
        ],
        rule,
    )
    result = submission.submit(ranking, lambda candidate, test: test.required_hits, rule)
    spend = reselection.spend_after(candidates[-1])
    spend.charge(ledger.Category.THRESHOLD_TEST, inferences=result.test_draws)
    recorded = reselection.record_threshold_1pct(
        context,
        sampler,
        method="toy",
        setting="B=4096",
        settings={"budgets": [4096]},
        spend=spend,
        ranking=ranking,
        submission=result,
        checkpoint=candidates[1],
        details={"budget": 4096, "candidates": [row.to_json() for row in candidates]},
        selection_rows=[],
        test_rows=[],
    )
    root = tmp_path / "reselected" / "threshold-1pct"
    assert recorded.directory == run_record.run_dir(root, "toy", "B=4096", TASK_ID, 2)
    assert run_record.read_jsonl(root / "summary.jsonl") == [recorded.summary.to_json()]
    assert not (tmp_path / "summary.jsonl").exists()
    construction = read(recorded.directory / "construction.json")
    assert list(construction["details"]) == ["threshold", "budget", "candidates"]
    assert construction["details"]["threshold"] == 0.01
    assert construction["status"] == "passed"
    assert construction["submitted"]["checkpoint_id"] == "update_000002"
    assert construction["submitted"]["update"] == 2
    assert construction["ledger"]["totals"]["inferences"] == 500.0 + 1857
    shipped = recorded.shipped
    assert 0.01 <= shipped.native_hit_rate < 0.10 and not shipped.fallback
    assert math.isfinite(shipped.reverse_kl)
    # The same sampler under the benchmark's threshold is scored as the fallback.
    at_10pct = context.evaluate_and_record(
        sampler, method="toy", setting="B=4096", settings={}, spend=spend
    )
    assert at_10pct.shipped.fallback and at_10pct.shipped.hits == shipped.hits


def test_a_record_of_an_own_rule_has_no_test_and_the_benchmarks_threshold(
    toy_context: ToyContext, tmp_path: pathlib.Path
) -> None:
    context = toy_context(tmp_path)
    sampler = posteriorbench.FiniteSampler(
        [([1, 0], 0.05), ([2, 0], 0.95)], inferences_per_response=1
    )
    candidates = [checkpoint(1, 500.0), checkpoint(2, 1000.0), checkpoint(4, 2000.0)]
    recorded = reselection.record_own_rule(
        context,
        sampler,
        rule=reselection.FINAL_CHECKPOINT,
        method="toy",
        setting="u=4",
        settings={"update_caps": [4]},
        checkpoint=candidates[-1],
        candidates=candidates,
        details={"update_cap": 4, "candidates": [row.to_json() for row in candidates]},
    )
    root = tmp_path / "reselected" / "final-checkpoint"
    assert recorded.directory == run_record.run_dir(root, "toy", "u=4", TASK_ID, 2)
    construction = read(recorded.directory / "construction.json")
    assert construction["status"] == "own_rule"
    assert construction["ranking"] is None and construction["submission"] is None
    assert construction["submitted"] == {"checkpoint_id": "update_000004", "update": 4}
    assert list(construction["details"]) == ["own_rule", "update_cap", "candidates"]
    assert construction["details"]["own_rule"] == "final-checkpoint"
    # The cost: the training until the last candidate was saved, and no threshold test.
    assert construction["ledger"]["totals"] == candidates[-1].ledger
    assert construction["ledger"]["charges"] == []
    assert recorded.shipped.construction_inferences == 1000.0
    assert recorded.shipped.budget_passes == 2000.0
    # About 5% of the draws are hits: below the benchmark's threshold, so the fallback is scored.
    assert recorded.shipped.fallback
    assert {path.name for path in (recorded.directory / "draws").iterdir()} == {"evaluation.jsonl"}
