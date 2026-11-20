"""`charges`: the shared charge specs and `apply`."""

from __future__ import annotations

import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import charges


def totals(specs: list[charges.ChargeSpec]) -> tuple[float, float]:
    target = ledger.Ledger()
    charges.apply(target, specs)
    return target.inferences, target.budget


def test_selection_draws_and_threshold_test() -> None:
    assert totals(charges.selection_draws(128, 7)) == (135.0, 135.0)
    assert totals(charges.selection_draws(16, 3, inferences_per_response=2.0)) == (35.0, 35.0)
    assert totals(charges.threshold_test(164)) == (164.0, 164.0)
    assert totals(charges.threshold_test(198, inferences_per_response=2.0)) == (396.0, 396.0)


def test_composition_selection_draws() -> None:
    assert totals(charges.composition_selection_draws(0)) == (496.0, 496.0)
    assert totals(charges.composition_selection_draws(40)) == (536.0, 536.0)
    assert totals(charges.composition_selection_draws(40, components=3)) == (616.0, 616.0)


def test_budget_only_and_apply_skips_empty_charges() -> None:
    target = ledger.Ledger()
    charges.apply(target, [charges.budget_only(256, "backward"), charges.budget_only(0, "none")])
    assert (target.inferences, target.budget) == (0.0, 256.0)
    assert len(target.charges()) == 1


def test_budget_limit_is_the_ledgers() -> None:
    target = ledger.Ledger(budget_limit=100)
    with pytest.raises(ledger.BudgetExceeded):
        charges.apply(target, charges.threshold_test(164))


@pytest.mark.parametrize(
    "make",
    [
        lambda: charges.selection_draws(16, 17),
        lambda: charges.selection_draws(-1, 0),
        lambda: charges.threshold_test(10, inferences_per_response=3.0),
        lambda: charges.composition_selection_draws(257),
        lambda: charges.composition_selection_draws(3, components=0),
        lambda: charges.counted(ledger.Category.BUDGET_ONLY, 1, "x"),
    ],
)
def test_impossible_charges_raise(make: object) -> None:
    with pytest.raises(charges.ChargeError):
        make()  # pyright: ignore[reportCallIssue]
