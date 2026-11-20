"""What CTL pays for, and the rule that closes a training budget."""

from __future__ import annotations

import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import charges, training_run
from posteriorbench_methods.ctl import budget_rule, cost
from posteriorbench_methods.ctl import settings as ctl_settings

PASSES = budget_rule.Passes(update=4000.0, checkpoint=384.0)


def spent(specs: list[charges.ChargeSpec]) -> tuple[float, float]:
    """(inference count, budget count) of the charges."""
    target = ledger.Ledger()
    charges.apply(target, specs)
    return target.inferences, target.budget


def test_an_update_counts_generation_and_p_and_charges_the_twist_network_to_the_budget() -> None:
    target = ledger.Ledger()
    charges.apply(target, cost.update(1000))
    assert (target.inferences, target.budget) == (2000, 4000)
    by_category = target.totals().by_category
    assert by_category["generation"] == (1000, 1000)
    assert by_category["p_scoring"] == (1000, 1000)
    assert by_category["budget_only"] == (0, 2000)


@pytest.mark.parametrize("hits", [0, 5, 128])
def test_a_checkpoint_counts_its_draws_and_p_on_its_hits(hits: int) -> None:
    assert spent(cost.checkpoint_selection_draws(128, hits)) == (128 + hits, 384)


def test_a_checkpoint_has_at_most_as_many_hits_as_draws() -> None:
    with pytest.raises(ValueError):
        cost.checkpoint_selection_draws(128, 129)


def test_the_passes_of_the_paper_settings() -> None:
    assert ctl_settings.Settings().passes == PASSES
    assert budget_rule.Passes.of(responses_per_update=64, selection_draws=32) == (
        budget_rule.Passes(update=256.0, checkpoint=96.0)
    )


def test_a_budget_is_closed_when_one_more_update_would_overrun_its_limit() -> None:
    budget = 2**20
    assert training_run.candidate_limit(budget) == budget - 1912
    # After 259 updates and 12 checkpoints, one more update and a checkpoint still fit.
    assert budget_rule.due(259 * 4000 + 12 * 384, [budget], PASSES, saved=False) == []
    # After 260 updates they do not: the twist is saved as the checkpoint that closes the budget.
    after_260 = 260 * 4000 + 12 * 384
    assert budget_rule.due(after_260, [budget], PASSES, saved=False) == [budget]
    assert after_260 + PASSES.checkpoint == 1_044_992 <= training_run.candidate_limit(budget)
    # Budgets close in increasing order, and several can close at once.
    assert budget_rule.due(8000, [11_912, 10_912, 2**20], PASSES, saved=True) == [10_912, 11_912]


def test_a_budget_that_can_no_longer_be_closed_is_an_error() -> None:
    with pytest.raises(training_run.BudgetError):
        training_run.candidate_limit(1912)
    with pytest.raises(training_run.BudgetError):  # the limit is behind the budget count
        budget_rule.due(9001, [10_912], PASSES, saved=True)
    with pytest.raises(training_run.BudgetError):  # the closing checkpoint would overrun it
        budget_rule.due(8800, [10_912], PASSES, saved=False)
    assert budget_rule.due(8800, [10_912], PASSES, saved=True) == [10_912]


def test_training_under_the_paper_budgets_ends_after_260_521_and_1044_updates() -> None:
    on_schedule = ctl_settings.Settings().saves_checkpoint_after
    assert budget_rule.last_update((2**20,), PASSES, on_schedule) == 260
    assert budget_rule.last_update((2**20, 2**21), PASSES, on_schedule) == 521
    assert budget_rule.last_update((2**20, 2**21, 2**22), PASSES, on_schedule) == 1044


def test_small_budgets_end_training_after_three_updates() -> None:
    """Updates 1 and 2 have a checkpoint (4,384 passes each), which closes 10,912 after update
    2. Update 3 has none on the schedule, and one is saved to close 15,912."""
    on_schedule = ctl_settings.Settings().saves_checkpoint_after
    assert budget_rule.last_update((10_912, 15_912), PASSES, on_schedule) == 3
    assert budget_rule.due(8768, [10_912, 15_912], PASSES, saved=True) == [10_912]
    assert budget_rule.due(12_768, [15_912], PASSES, saved=False) == [15_912]


def test_the_next_checkpoint_comes_after_a_checkpoint_or_after_an_update_and_a_checkpoint() -> None:
    assert budget_rule.next_checkpoint_lower_bound(1000, PASSES, saved=False) == 1384
    assert budget_rule.next_checkpoint_lower_bound(1000, PASSES, saved=True) == 5384


def test_an_update_cap_stands_for_the_budget_that_pays_for_its_candidates() -> None:
    # Update cap 4: the checkpoints of updates 1, 2 and 4.
    assert budget_rule.update_cap_budget(4, 3, PASSES) == 4 * 4000 + 3 * 384 + 1912 == 19_064
