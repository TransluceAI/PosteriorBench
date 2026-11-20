"""What SFT and Replay pay for.

    event                                           inference count   budget count
    a draw from the original model                  1                 1
    an update on b training responses               b                 2 b
    the m selection draws of a checkpoint (h hits)  m + h             2 m + h
    a threshold test of n draws                     n                 n

The backward pass of an update, and the second pass that gives log q of a checkpoint's selection
draws, are charged to the budget alone. Replay pays for the draws only. A run on the whole corpus
pays, in place of the first row, what the benchmark charges for corpus hits: 1 / p̂(S) draws for
each hit it takes (`corpus_draws`).
"""

from __future__ import annotations

from posteriorbench_methods.common import charges


def draws_from_p(draws: int) -> list[charges.ChargeSpec]:
    """The run's draws from the original model, hits and misses."""
    return [
        charges.counted(
            charges.Category.GENERATION, draws, f"draws from the original model: {draws} draws"
        )
    ]


def corpus_draws(hits: int, inferences: float) -> list[charges.ChargeSpec]:
    """`hits` corpus hits, charged `inferences` (1 / p̂(S) draws each)."""
    return [
        charges.counted(
            charges.Category.GENERATION,
            inferences,
            f"draws from the original model: the corpus, {hits} hits at 1 / p(S) draws each",
        )
    ]


def training(responses: int) -> list[charges.ChargeSpec]:
    """Updates over `responses` training responses in all."""
    return [
        charges.counted(
            charges.Category.TRAINING_FORWARD,
            responses,
            f"training, forward: {responses} responses",
        ),
        charges.budget_only(responses, f"training, backward: {responses} responses"),
    ]


def checkpoint_selection_draws(draws: int, hits: int) -> list[charges.ChargeSpec]:
    """The selection draws of one checkpoint, `hits` of them hits."""
    return [
        *charges.selection_draws(draws, hits),
        charges.budget_only(draws, f"selection draws: log q of the {draws} draws"),
    ]
