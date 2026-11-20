"""What KL-DPG, JS-DPG and GDC pay for.

    event                                           inference count   budget count
    an f-DPG update on N draws (h hits)             2 N + h           4 N (KL-DPG), 5 N (JS-DPG)
    the same before the first hit of the training   N                 3 N
    the m selection draws of a checkpoint (h hits)  m + h             3 m
    a GDC update on N draws (h hits)                N + 2 h           N + 4 h
    a GDC fit check on D draws (h hits)             D + 2 h           D + 3 h
    a threshold test of n draws                     n                 n
    the selection draws of composition (h hits)     496 + h           1,264

The inference count holds generation, p where a method uses it, which is on hits, and the forward
pass of a gradient step; every other pass is charged to the budget alone.

An f-DPG update computes p of all its draws, runs a forward and a backward pass over them, and in
JS-DPG a further pass for its coefficients. Before the first hit of the training there is no
gradient step, and the forward pass is charged to the budget alone. A checkpoint computes p of
all its selection draws and gets their log q from a pass of its own. GDC computes p, r and pi
on hits only: an update pays p and the forward pass of pi as inferences, and a fit check pays p
and pi (pi did not generate the draws). Composition computes p of all its 256 selection draws and
their log q from the checkpoint and the original model (`composition_charges`).
"""

from __future__ import annotations

from typing import Literal

from posteriorbench_methods.common import charges, composition

Category = charges.Category


def _check_hits(hits: int, draws: int) -> None:
    if not 0 <= hits <= draws:
        raise charges.ChargeError(f"{hits} hits among {draws} draws")


def fdpg_update(
    draws: int, hits: int, *, objective: Literal["kl", "js"], has_gradient: bool
) -> list[charges.ChargeSpec]:
    """One update of KL-DPG or JS-DPG on `draws` new responses of the policy, `hits` of them
    hits. `has_gradient` is False for an update before the first hit of the training, which
    takes no gradient step."""
    _check_hits(hits, draws)
    specs = [
        charges.counted(Category.GENERATION, draws, f"update: {draws} draws"),
        charges.counted(Category.P_SCORING, hits, f"update: p on {hits} hits"),
        charges.budget_only(draws - hits, f"update: p on {draws - hits} misses"),
    ]
    if not has_gradient:
        return [*specs, charges.budget_only(draws, "update: forward pass, no gradient step")]
    specs += [
        charges.counted(Category.TRAINING_FORWARD, draws, "update: forward pass"),
        charges.budget_only(draws, "update: backward pass"),
    ]
    if objective == "js":
        specs.append(charges.budget_only(draws, "update: pass for JS-DPG's coefficients"))
    return specs


def checkpoint_selection_draws(draws: int, hits: int) -> list[charges.ChargeSpec]:
    """The selection draws of one checkpoint, `hits` of them hits."""
    return [
        *charges.selection_draws(draws, hits),
        charges.budget_only(draws - hits, f"selection draws: p on {draws - hits} misses"),
        charges.budget_only(draws, f"selection draws: log q of the {draws} draws"),
    ]


def gdc_update(draws: int, hits: int) -> list[charges.ChargeSpec]:
    """One update of GDC on `draws` new responses of r, `hits` of them hits."""
    _check_hits(hits, draws)
    return [
        charges.counted(Category.GENERATION, draws, f"update: {draws} draws from r"),
        charges.counted(Category.P_SCORING, hits, f"update: p on {hits} hits"),
        charges.counted(Category.TRAINING_FORWARD, hits, f"update: forward pass on {hits} hits"),
        charges.budget_only(2 * hits, f"update: r and the backward pass on {hits} hits"),
    ]


def gdc_fit_check(draws: int, hits: int) -> list[charges.ChargeSpec]:
    """One fit check of GDC on `draws` new responses of r, `hits` of them hits."""
    _check_hits(hits, draws)
    return [
        charges.counted(Category.GENERATION, draws, f"fit check: {draws} draws from r"),
        charges.counted(Category.P_SCORING, hits, f"fit check: p on {hits} hits"),
        charges.counted(Category.Q_SCORING, hits, f"fit check: pi on {hits} hits"),
        charges.budget_only(hits, f"fit check: r on {hits} hits"),
    ]


def composition_charges(
    outcome: composition.CompositionOutcome, *, response_limit: int
) -> list[charges.ChargeSpec]:
    """What composition adds: the shared charges (`composition.composition_charges`) and, to the
    budget alone, p on the selection misses and the two passes that give the composed log q of
    the selection draws."""
    draws = charges.COMPOSITION_SELECTION_DRAWS
    misses = draws - outcome.total_selection_hits
    return [
        *composition.composition_charges(outcome, response_limit=response_limit),
        charges.budget_only(misses, f"composition: p on {misses} selection misses"),
        charges.budget_only(2 * draws, "composition: log q of the selection draws, two models"),
    ]
