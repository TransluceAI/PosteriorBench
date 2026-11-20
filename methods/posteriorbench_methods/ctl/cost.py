"""What CTL pays for.

    event                                           inference count   budget count
    an update on N responses                        2 N               4 N
    the m selection draws of a checkpoint (h hits)  m + h             3 m
    a threshold test of n draws                     n                 n

An update generates its N responses and computes p of every one of them, hits and misses, because
its weights read p of both. The passes of the twist network over the responses, forward and
backward, are charged to the budget alone: the inference count counts passes of the model, and the
network is small next to it. A checkpoint generates its selection draws and gets log p of the hits
among them. Its log p of the misses, which no selection signal reads, and the teacher-forced pass
that computes log q of the draws are charged to the budget alone. The budget count therefore does
not depend on the responses, which `budget_rule` relies on.
"""

from __future__ import annotations

from posteriorbench_methods.common import charges

Category = charges.Category


def update(responses: int) -> list[charges.ChargeSpec]:
    """One update on `responses` responses of the twisted model."""
    return [
        charges.counted(Category.GENERATION, responses, f"update: {responses} responses"),
        charges.counted(Category.P_SCORING, responses, f"update: p of {responses} responses"),
        charges.budget_only(
            2 * responses, f"update: the twist network on {responses} responses, both ways"
        ),
    ]


def checkpoint_selection_draws(draws: int, hits: int) -> list[charges.ChargeSpec]:
    """The selection draws of one checkpoint, `hits` of them hits."""
    return [
        *charges.selection_draws(draws, hits),
        charges.budget_only(draws - hits, f"selection draws: p on {draws - hits} misses"),
        charges.budget_only(draws, f"selection draws: log q of the {draws} draws"),
    ]
