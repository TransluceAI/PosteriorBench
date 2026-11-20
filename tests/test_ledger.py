"""The construction-cost ledger: charge rules, totals, budget limit, resume."""

from __future__ import annotations

import math

import pytest

from posteriorbench import ledger

Category = ledger.Category


def test_budget_defaults_to_inferences() -> None:
    spend = ledger.Ledger()
    spend.charge(Category.GENERATION, inferences=128, note="selection draws")
    spend.charge(Category.P_SCORING, inferences=7)
    spend.charge(Category.BUDGET_ONLY, inferences=0, budget=128, note="TF q of own draws")
    assert (spend.inferences, spend.budget) == (135.0, 263.0)
    totals = spend.totals()
    assert totals.by_category["generation"] == (128.0, 128.0)
    assert totals.by_category["budget_only"] == (0.0, 128.0)
    assert totals.by_category["corpus_hits"] == (0.0, 0.0)
    assert set(totals.by_category) == {c.value for c in Category}
    assert [c.note for c in spend.charges()] == ["selection draws", "", "TF q of own draws"]


def test_inferences_and_budget_of_a_training_update() -> None:
    """An update that draws 1,000 responses, computes p on each and runs a forward and a backward
    pass that the inference count leaves out costs 2,000 inferences and 4,000 budget."""
    spend = ledger.Ledger()
    spend.charge(Category.GENERATION, inferences=1000)
    spend.charge(Category.P_SCORING, inferences=1000)
    spend.charge(Category.BUDGET_ONLY, inferences=0, budget=2000)
    assert (spend.inferences, spend.budget) == (2000.0, 4000.0)


def test_charge_rules() -> None:
    spend = ledger.Ledger()
    with pytest.raises(ValueError, match="BUDGET_ONLY"):
        spend.charge(Category.BUDGET_ONLY, inferences=1, budget=1)
    with pytest.raises(ValueError, match="UNCOUNTED"):
        spend.charge(Category.UNCOUNTED, inferences=0, budget=1)
    with pytest.raises(ValueError, match="UNCOUNTED"):
        spend.charge(Category.UNCOUNTED, inferences=1)
    for bad in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            spend.charge(Category.GENERATION, inferences=bad)
        with pytest.raises(ValueError):
            spend.charge(Category.GENERATION, inferences=1, budget=bad)
    with pytest.raises(ValueError, match="every inference is a pass"):
        spend.charge(Category.GENERATION, inferences=2, budget=1)
    with pytest.raises(ValueError):
        spend.charge("not a category", inferences=1)  # pyright: ignore[reportArgumentType]
    spend.charge(Category.UNCOUNTED, inferences=0, budget=0, note="LLM prompt writer")
    assert (spend.inferences, spend.budget) == (0.0, 0.0)
    assert len(spend.charges()) == 1  # nothing else was recorded


def test_budget_limit() -> None:
    spend = ledger.Ledger(budget_limit=100)
    spend.charge(Category.GENERATION, inferences=60)
    assert spend.remaining_budget() == 40
    with pytest.raises(ledger.BudgetExceeded):
        spend.charge(Category.BUDGET_ONLY, inferences=0, budget=41)
    assert spend.budget == 60 and len(spend.charges()) == 1  # the refused charge is not recorded
    spend.charge(Category.BUDGET_ONLY, inferences=0, budget=40)
    assert spend.remaining_budget() == 0
    with pytest.raises(ValueError):
        ledger.Ledger().remaining_budget()
    with pytest.raises(ValueError):
        ledger.Ledger(budget_limit=-1)


def test_totals_are_exact_sums_independent_of_order() -> None:
    amounts = [1e16, 1.0, 1.0, 0.1, 0.2, 0.3]
    forward, backward = ledger.Ledger(), ledger.Ledger()
    for amount in amounts:
        forward.charge(Category.GENERATION, inferences=amount)
    for amount in reversed(amounts):
        backward.charge(Category.GENERATION, inferences=amount)
    assert forward.inferences == backward.inferences == math.fsum(amounts)
    naive = 0.0
    for amount in amounts:
        naive += amount
    assert forward.inferences != naive  # a naive left-to-right sum loses the small charges
    assert forward.totals() == backward.totals()


def test_totals_round_trip() -> None:
    spend = ledger.Ledger()
    spend.charge(Category.CORPUS_HITS, inferences=1234.5)
    spend.charge(Category.THRESHOLD_TEST, inferences=164)
    totals = spend.totals()
    assert ledger.LedgerTotals.from_dict(totals.to_dict()) == totals
    with pytest.raises(ValueError):
        ledger.LedgerTotals.from_dict({**totals.to_dict(), "by_category": {"bogus": [1, 1]}})
    zero = ledger.LedgerTotals.zero()
    assert zero.inferences == zero.budget == 0.0 and zero == ledger.Ledger().totals()


def test_resume_from_saved_totals() -> None:
    """A ledger resumed from the totals saved with a checkpoint continues from them and lists
    only its own charges."""
    trajectory = ledger.Ledger()
    trajectory.charge(Category.GENERATION, inferences=1000)
    trajectory.charge(Category.BUDGET_ONLY, inferences=0, budget=3000)
    anchor = trajectory.totals()
    cut = ledger.Ledger.resume(anchor, budget_limit=5000)
    cut.charge(Category.THRESHOLD_TEST, inferences=164)
    assert (cut.inferences, cut.budget) == (1164.0, 4164.0)
    assert cut.resumed_from == anchor
    assert [c.category for c in cut.charges()] == [Category.THRESHOLD_TEST]  # only new charges
    totals = cut.totals()
    assert totals.by_category["generation"] == (1000.0, 1000.0)
    assert totals.by_category["threshold_test"] == (164.0, 164.0)
    assert totals.inferences == 1164.0
    with pytest.raises(ledger.BudgetExceeded):
        cut.charge(Category.THRESHOLD_TEST, inferences=900)
    with pytest.raises(ledger.BudgetExceeded):
        ledger.Ledger.resume(anchor, budget_limit=3999)
