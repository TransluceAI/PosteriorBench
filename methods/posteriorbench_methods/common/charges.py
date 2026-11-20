"""Construction charges, and the charges that every family makes.

A family turns each event of its method (a training update, a candidate's selection draws, ...)
into a list of `ChargeSpec`s and records them with `apply`. The ledger (`posteriorbench.ledger`)
keeps two totals: the inference count, which is the cost the benchmark reports, and the budget
count, which also counts the passes the inference count leaves out (`budget_only`). A draw costs 1
inference, or 2 for a composed sampler, which runs the sampler and the original model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

from posteriorbench import ledger, sampler

Category = ledger.Category
COMPOSITION_SELECTION_DRAWS = 256  # 16 settings x 16 draws
# Generating them: 15 composed settings x 16 draws x 2 inferences + the plain setting's 16 draws.
COMPOSITION_SELECTION_GENERATION = 496


class ChargeError(ValueError):
    """A charge with impossible counts (negative draws, more hits than draws)."""


@dataclasses.dataclass(frozen=True)
class ChargeSpec:
    """A charge still to be recorded: `inferences` and `budget` are what it adds to the two
    totals."""

    category: ledger.Category
    inferences: float
    budget: float
    note: str


def passes(charges: Iterable[ChargeSpec]) -> float:
    """What `charges` add to the budget count."""
    return sum(spec.budget for spec in charges)


def apply(target: ledger.Ledger, charges: Iterable[ChargeSpec]) -> None:
    """Record every charge in `target`, in order, skipping a charge of zero to both totals. Raises
    the ledger's `BudgetExceeded` if the budget count would pass the ledger's limit."""
    for spec in charges:
        if spec.inferences == 0 and spec.budget == 0:
            continue
        target.charge(spec.category, inferences=spec.inferences, budget=spec.budget, note=spec.note)


def _check_count(name: str, count: int) -> None:
    if isinstance(count, bool) or count < 0:
        raise ChargeError(f"{name} must be a non-negative integer, got {count!r}")


def _check_hits(hits: int, draws: int) -> None:
    _check_count("draws", draws)
    _check_count("hits", hits)
    if hits > draws:
        raise ChargeError(f"{hits} hits among {draws} draws")


def _check_inferences_per_response(value: float) -> None:
    if not (math.isfinite(value) and 0 < value <= sampler.MAX_INFERENCES_PER_RESPONSE):
        raise ChargeError(f"inferences per response must lie in (0, 2], got {value}")


def counted(category: ledger.Category, amount: float, note: str) -> ChargeSpec:
    """A charge of `amount` to both totals."""
    if category in (Category.BUDGET_ONLY, Category.UNCOUNTED):
        raise ChargeError(f"{category.value} charges are not counted as inferences")
    return ChargeSpec(category, float(amount), float(amount), note)


def budget_only(passes: float, note: str) -> ChargeSpec:
    """A charge to the budget count alone, for passes the inference count leaves out: backward
    passes, q of a sampler's own draws, p on misses."""
    return ChargeSpec(Category.BUDGET_ONLY, 0.0, float(passes), note)


def selection_draws(
    draws: int, hits: int, *, inferences_per_response: float = 1.0, note: str = "selection draws"
) -> list[ChargeSpec]:
    """The selection draws of one candidate: generating `draws` responses, and p on the hits among
    them (a selection signal reads p only on hits)."""
    _check_hits(hits, draws)
    _check_inferences_per_response(inferences_per_response)
    return [
        counted(Category.GENERATION, draws * inferences_per_response, f"{note}: {draws} draws"),
        counted(Category.P_SCORING, hits, f"{note}: p on {hits} hits"),
    ]


def threshold_test(
    draws: int, *, inferences_per_response: float = 1.0, note: str = "threshold test"
) -> list[ChargeSpec]:
    """The draws of one threshold test."""
    _check_count("draws", draws)
    _check_inferences_per_response(inferences_per_response)
    amount = draws * inferences_per_response
    return [counted(Category.THRESHOLD_TEST, amount, f"{note}: {draws} draws")]


def composition_selection_draws(hits: int, *, components: int = 1) -> list[ChargeSpec]:
    """The selection draws of the 16 composition settings: 496 inferences to generate them, p on
    the hits and, for a mixture of `components` samplers, the q of the other components on every
    hit (the q of a mixture needs every component). `composition.composition_charges` adds the
    threshold tests."""
    _check_hits(hits, COMPOSITION_SELECTION_DRAWS)
    if components < 1:
        raise ChargeError(f"a composed sampler has at least one component, got {components}")
    charges = [
        counted(
            Category.GENERATION,
            COMPOSITION_SELECTION_GENERATION,
            "composition: 16 selection draws at each of 16 settings",
        ),
        counted(Category.P_SCORING, hits, f"composition: p on {hits} selection hits"),
    ]
    if components > 1:
        charges.append(
            counted(
                Category.Q_SCORING,
                hits * (components - 1),
                f"composition: the other {components - 1} components' q on the selection hits",
            )
        )
    return charges
