"""What GFlowNet pays for.

    event                                           inference count   budget count
    a fresh training response                       2                 5
    a replayed training response                    1                 2
    the m selection draws of a checkpoint (h hits)  m + h             3 m
    a threshold test of n draws                     n                 n

A fresh response is generated and scored by p, which the reward needs for hits and misses. The
forward pass of its loss is counted with the generation, because the original code computes the
loss while it generates; the budget count adds q of the response by teacher forcing and the
forward and backward passes of the loss. A replayed response pays the forward pass of its loss,
and its backward pass to the budget alone. p on the misses among a checkpoint's selection draws,
and q of the draws, are charged to the budget alone; composition adds p on the misses among its
selection draws to the budget alone.

An update of 375 fresh and 125 replayed responses costs 875 inferences and 2,125 passes.
"""

from __future__ import annotations

from posteriorbench_methods.common import charges, composition

Category = charges.Category


def fresh_responses(count: int) -> list[charges.ChargeSpec]:
    """`count` fresh training responses: generated, scored by p and by q, and trained on."""
    return [
        charges.counted(Category.GENERATION, count, f"{count} fresh training responses"),
        charges.counted(Category.P_SCORING, count, f"p of {count} fresh training responses"),
        charges.budget_only(
            3 * count, f"q, loss forward and backward of {count} fresh training responses"
        ),
    ]


def replayed_responses(count: int) -> list[charges.ChargeSpec]:
    """`count` training responses replayed from the buffer."""
    return [
        charges.counted(
            Category.TRAINING_FORWARD, count, f"loss forward of {count} replayed responses"
        ),
        charges.budget_only(count, f"loss backward of {count} replayed responses"),
    ]


def checkpoint_selection_draws(draws: int, hits: int) -> list[charges.ChargeSpec]:
    """The selection draws of one checkpoint: `draws` responses with `hits` hits among them."""
    return [
        *charges.selection_draws(draws, hits),
        charges.budget_only(draws - hits, f"selection draws: p on {draws - hits} misses"),
        charges.budget_only(draws, f"selection draws: q of {draws} draws"),
    ]


def composition_charges(
    outcome: composition.CompositionOutcome, *, response_limit: int
) -> list[charges.ChargeSpec]:
    """What composition adds to the cost of the checkpoint it composes: the selection draws of
    the 16 settings and their threshold tests."""
    misses = charges.COMPOSITION_SELECTION_DRAWS - outcome.total_selection_hits
    return [
        *composition.composition_charges(outcome, response_limit=response_limit),
        charges.budget_only(misses, f"composition: p on {misses} selection misses"),
    ]
