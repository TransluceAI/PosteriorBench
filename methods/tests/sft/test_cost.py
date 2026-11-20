"""What SFT pays for each event: the inference count and the budget count."""

from __future__ import annotations

from posteriorbench import ledger

from posteriorbench_methods.common import charges
from posteriorbench_methods.sft import cost


def totals(specs: list[charges.ChargeSpec]) -> ledger.LedgerTotals:
    spend = ledger.Ledger()
    charges.apply(spend, specs)
    return spend.totals()


def test_a_draw_from_the_original_model_costs_one_inference() -> None:
    spent = totals(cost.draws_from_p(2022))
    assert (spent.inferences, spent.budget) == (2022, 2022)
    assert spent.by_category["generation"] == (2022, 2022)


def test_an_update_counts_its_forward_pass_and_charges_its_backward_pass_to_the_budget() -> None:
    spent = totals(cost.training(32))
    assert (spent.inferences, spent.budget) == (32, 64)
    assert spent.by_category["training_forward"] == (32, 32)
    assert spent.by_category["budget_only"] == (0, 32)


def test_selection_draws_pay_for_p_on_the_hits_only() -> None:
    spent = totals(cost.checkpoint_selection_draws(128, 21))
    assert (spent.inferences, spent.budget) == (128 + 21, 2 * 128 + 21)
    assert spent.by_category["p_scoring"] == (21, 21)
    assert totals(cost.checkpoint_selection_draws(128, 0)).by_category["p_scoring"] == (0, 0)


def test_the_readme_example() -> None:
    """A run with 3,000 draws and 40 hits that trains on 1,000 responses, saves 6 checkpoints
    with 30 selection hits in all and passes the test at rank 1."""
    spend = ledger.Ledger()
    charges.apply(spend, cost.draws_from_p(3000))
    charges.apply(spend, cost.training(1000))
    for hits in (2, 3, 4, 6, 7, 8):
        charges.apply(spend, cost.checkpoint_selection_draws(128, hits))
    charges.apply(spend, charges.threshold_test(164))
    assert spend.inferences == 3000 + 1000 + 6 * 128 + 30 + 164 == 4962
    assert spend.budget == 3000 + 2000 + 6 * 256 + 30 + 164 == 6730
