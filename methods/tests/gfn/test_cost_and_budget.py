"""What GFlowNet pays for, and when its budgets are closed."""

from __future__ import annotations

import numpy as np
import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import (
    charges,
    composition,
    rank_table,
    signals,
    submission,
    training_run,
)
from posteriorbench_methods.gfn import budget_rule, cost


def totals(specs: list[charges.ChargeSpec]) -> tuple[float, float]:
    """(inference count, budget count) of the charges."""
    spend = ledger.Ledger()
    charges.apply(spend, specs)
    return spend.inferences, spend.budget


def test_training_charges() -> None:
    assert totals(cost.fresh_responses(1)) == (2, 5)
    assert totals(cost.replayed_responses(1)) == (1, 2)
    assert totals(cost.checkpoint_selection_draws(128, 17)) == (128 + 17, 384)
    assert totals(cost.fresh_responses(375) + cost.replayed_responses(125)) == (875, 2125)


def test_composition_adds_p_on_the_selection_misses_to_the_budget_count() -> None:
    rng = np.random.default_rng(0)

    def draw(setting: composition.Setting) -> signals.SelectionDraws:
        hit = rng.random(16) < 0.3
        return signals.SelectionDraws(hit, rng.normal(-5, 1, 16), rng.normal(-7, 1, 16))

    def test(
        setting: composition.Setting,
        candidate: submission.RankedCandidate,
        threshold_test: rank_table.ThresholdTest,
    ) -> int:
        return threshold_test.required_hits - (candidate.rank < 3)  # the third setting passes

    outcome = composition.choose_setting(draw, test)
    hits = outcome.total_selection_hits
    tests = sum(
        t.draws * outcome.setting_of(t).inferences_per_response() for t in outcome.submission.tests
    )
    inferences, budget = totals(cost.composition_charges(outcome, response_limit=50))
    assert len(outcome.submission.tests) == 3
    assert inferences == 496 + hits + tests
    assert budget == 496 + 256 + tests


def test_the_paper_update_and_checkpoint_take_at_most_2884_passes() -> None:
    assert budget_rule.checkpoint_passes(128) == 384
    assert budget_rule.update_bound(500, 128) == 2884
    assert training_run.candidate_limit(1_048_576) == 1_048_576 - 1_912
    with pytest.raises(training_run.BudgetError):
        training_run.candidate_limit(1_912)


def test_a_budget_is_due_when_one_more_update_could_pass_its_limit() -> None:
    budgets = [2_097_152, 1_048_576]
    limit = training_run.candidate_limit(1_048_576)
    assert budget_rule.due(0.0, 2884, budgets, checkpoint=384) == []
    assert budget_rule.due(limit - 2884, 2884, budgets, checkpoint=384) == []
    assert budget_rule.due(limit - 2883, 2884, budgets, checkpoint=384) == [1_048_576]
    assert budget_rule.due(limit, 2884, budgets, checkpoint=0) == [1_048_576]
    late = training_run.candidate_limit(2_097_152) - 100
    assert budget_rule.due(late, 2884, [2_097_152], checkpoint=0) == [2_097_152]


def test_a_budget_that_can_no_longer_be_closed_is_an_error() -> None:
    limit = training_run.candidate_limit(1_048_576)
    with pytest.raises(training_run.BudgetError):
        budget_rule.due(limit + 1, 2884, [1_048_576], checkpoint=0)
    with pytest.raises(training_run.BudgetError):  # the checkpoint that closes it does not fit
        budget_rule.due(limit - 383, 2884, [1_048_576], checkpoint=384)
