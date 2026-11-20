"""The charges of the prompt family."""

from __future__ import annotations

from posteriorbench import ledger

from posteriorbench_methods.common import charges
from posteriorbench_methods.prompt import cost


def spent(specs: list[charges.ChargeSpec]) -> ledger.LedgerTotals:
    target = ledger.Ledger()
    charges.apply(target, specs)
    return target.totals()


def test_selection_draws_pay_for_p_only_on_the_hits_a_method_reads() -> None:
    reads_p = spent(cost.selection_draws(16, 16, p_on_hits=30))
    assert (reads_p.inferences, reads_p.budget) == (16 * 16 + 30, 16 * 16 + 30)
    assert reads_p.by_category["generation"][0] == 256
    assert reads_p.by_category["p_scoring"][0] == 30
    hit_rate = spent(cost.selection_draws(16, 16, p_on_hits=0))
    assert hit_rate.inferences == 256 and hit_rate.by_category["p_scoring"][0] == 0


def test_the_worked_examples_of_the_readme() -> None:
    """EB-JC at K = 16 with 82 selection hits and one test of 164 draws, the hit rate on the same
    pool, and the CE method with 40 hits in its bank and 300 cross-scored pairs."""
    test = charges.threshold_test(164)
    assert spent(cost.selection_draws(16, 16, p_on_hits=82) + test).inferences == 502
    assert spent(cost.selection_draws(16, 16, p_on_hits=0) + test).inferences == 420
    ce = spent(cost.selection_draws(16, 16, p_on_hits=0) + cost.ce_bank(40, 300) + test)
    assert ce.inferences == 256 + 40 + 300 + 164
    assert (ce.by_category["p_scoring"][0], ce.by_category["q_scoring"][0]) == (40, 300)
