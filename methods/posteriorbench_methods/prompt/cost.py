"""Charges of the prompt family; every charge counts in both totals of the ledger.

A selection draw costs one inference, with its q. p is paid where a method reads it: on every
selection hit for JC, EB-JC, IWAE and the mixtures, on the bank's hits for the CE method, nowhere
for the hit rate. The writer LLM is not charged.
"""

from __future__ import annotations

from posteriorbench_methods.common import charges


def selection_draws(pool_size: int, draws: int, *, p_on_hits: int) -> list[charges.ChargeSpec]:
    """`draws` responses of each of `pool_size` candidates, and p on `p_on_hits` hits: all hits
    for the signals that read p, none for the hit rate and the CE method (`ce_bank`)."""
    return charges.selection_draws(
        pool_size * draws, p_on_hits, note=f"selection draws of {pool_size} candidates"
    )


def ce_bank(hits: int, cross_scored_pairs: int) -> list[charges.ChargeSpec]:
    """The CE method's scoring: p on each hit in the bank, q on each of
    `ce.Bank.cross_scored_pairs`."""
    return [
        charges.counted(charges.Category.P_SCORING, hits, f"CE bank: p on {hits} hits"),
        charges.counted(
            charges.Category.Q_SCORING,
            cross_scored_pairs,
            f"CE bank: q of the scored candidates on {cross_scored_pairs} hits of other candidates",
        ),
    ]
