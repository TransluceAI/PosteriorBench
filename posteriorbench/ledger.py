"""The construction-cost ledger: the benchmark's counting rules.

A method builds its sampler while charging what it computes to a `Ledger`. The ledger keeps two
counts:

- the inference count, the benchmark's cost axis. One inference is generating one response with
  the original model or a sampler built from it, computing p(y) or q(y) of one response, or the
  forward pass of a gradient step on one response. A sampler gets q(y) of a response it generates
  itself for free. Calls to other models (an LLM that writes candidate prompts, a coherence judge)
  and evaluation are not counted. Costs shared by several runs (for example one behavior's
  steering vectors) are charged in full to every task and replicate.
- the training-budget count B: every pass over one response, including the passes the inference
  count leaves out (backward passes, for example). A run spends at most B; every counted inference
  is also a pass, so the budget count is never below the inference count.

The package cannot meter compute done outside it. It charges corpus hits itself
(`corpus.CorpusAccess` charges 1/p̂(S) per hit taken); every other cost a method charges through
`Ledger.charge`, under the category that names the work. p(y) is paid only on the responses where
the method uses it. A threshold-test draw costs what its candidate spends to generate one response
(one for a plain sampler, two for a composed one, which runs both q and the original model).

Totals are exact sums (`math.fsum` over the history), so they do not depend on the order of the
charges.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any

from posteriorbench import errors


class Category(enum.StrEnum):
    """What a charge pays for. Inference and budget amounts are given per charge; the categories
    keep the history readable and let the totals be broken down."""

    # Corpus hits handed out by `corpus.CorpusAccess` (1/p̂(S) each, inference and budget). Only
    # `CorpusAccess` charges this category.
    CORPUS_HITS = "corpus_hits"
    # Responses generated during construction, such as training draws and selection draws; a
    # response of a sampler that runs two models costs two.
    GENERATION = "generation"
    # Threshold-test draws, each charged its candidate's `inferences_per_response`.
    THRESHOLD_TEST = "threshold_test"
    # p(y) of responses where the method uses it.
    P_SCORING = "p_scoring"
    # q(y) of responses the sampler did not generate.
    Q_SCORING = "q_scoring"
    # The forward pass of a gradient step on one response.
    TRAINING_FORWARD = "training_forward"
    # Passes the inference count leaves out (backward passes, a separate teacher-forcing pass over
    # the sampler's own draws, a small head's pass, ...): inferences must be 0.
    BUDGET_ONLY = "budget_only"
    # Shared construction (steering vectors, a refusal direction), charged in full to every run.
    SHARED = "shared"
    # Recorded for transparency and never counted (other-model calls): inferences = budget = 0.
    UNCOUNTED = "uncounted"


@dataclasses.dataclass(frozen=True)
class Charge:
    """One entry of a ledger's history."""

    category: Category
    inferences: float
    budget: float
    note: str


@dataclasses.dataclass(frozen=True)
class LedgerTotals:
    """A ledger's totals: the inference count (the benchmark's cost axis), the training budget
    count B, and both per category (category value -> (inferences, budget), every category
    listed)."""

    inferences: float
    budget: float
    by_category: dict[str, tuple[float, float]]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable copy."""
        return {
            "inferences": self.inferences,
            "budget": self.budget,
            "by_category": {key: [value[0], value[1]] for key, value in self.by_category.items()},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LedgerTotals:
        """The inverse of `to_dict`. Raises ValueError for an unknown category or a malformed
        entry."""
        by_category: dict[str, tuple[float, float]] = {}
        for key, amounts in dict(value["by_category"]).items():
            Category(key)  # raises ValueError for an unknown category
            inferences, budget = amounts
            by_category[str(key)] = (float(inferences), float(budget))
        return cls(float(value["inferences"]), float(value["budget"]), by_category)

    @classmethod
    def zero(cls) -> LedgerTotals:
        """Totals of a ledger with no charges."""
        return cls(0.0, 0.0, {category.value: (0.0, 0.0) for category in Category})


class BudgetExceeded(errors.PosteriorBenchError):
    """A charge would take the ledger's budget count beyond its limit."""


class Ledger:
    """Construction-cost accounting for one run (one method, task, setting and replicate).

    `budget_limit` (optional) is the run's training budget B: a charge that would take the budget
    count beyond it raises BudgetExceeded and is not recorded."""

    def __init__(self, *, budget_limit: float | None = None) -> None:
        if budget_limit is not None and not (math.isfinite(budget_limit) and budget_limit >= 0):
            raise ValueError(f"budget_limit must be finite and non-negative, got {budget_limit}")
        self.budget_limit = budget_limit
        self._charges: list[Charge] = []
        self._base: LedgerTotals | None = None

    @classmethod
    def resume(cls, totals: LedgerTotals, *, budget_limit: float | None = None) -> Ledger:
        """A ledger that starts from saved totals, for example at a checkpoint of a training run
        that is evaluated at several budgets. `charges()` then lists only the charges made after
        resuming; `resumed_from` keeps the starting totals. Raises BudgetExceeded if the saved
        budget count already exceeds `budget_limit`."""
        ledger = cls(budget_limit=budget_limit)
        for key in totals.by_category:
            Category(key)
        if budget_limit is not None and totals.budget > budget_limit:
            raise BudgetExceeded(
                f"the saved budget count {totals.budget} exceeds the limit {budget_limit}"
            )
        ledger._base = LedgerTotals(totals.inferences, totals.budget, dict(totals.by_category))
        return ledger

    @property
    def resumed_from(self) -> LedgerTotals | None:
        """The totals this ledger was resumed from (`resume`), or None."""
        return self._base

    def charge(
        self, category: Category, *, inferences: float, budget: float | None = None, note: str = ""
    ) -> None:
        """Record a charge. `budget` defaults to `inferences`.

        Raises ValueError for a negative or non-finite amount, a budget below the inferences (every
        counted inference is a pass), a BUDGET_ONLY charge with inferences, or an UNCOUNTED charge
        with any amount; raises BudgetExceeded (recording nothing) if the budget count would exceed
        `budget_limit`."""
        category = Category(category)
        if budget is None:
            budget = inferences
        for name, amount in (("inferences", inferences), ("budget", budget)):
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {amount}")
        if category is Category.BUDGET_ONLY and inferences != 0:
            raise ValueError("a BUDGET_ONLY charge has no inferences; use another category")
        if category is Category.UNCOUNTED and (inferences != 0 or budget != 0):
            raise ValueError("an UNCOUNTED charge records work that is never counted: use 0, 0")
        if budget < inferences:
            raise ValueError(
                f"budget {budget} is below inferences {inferences}: every inference is a pass"
            )
        if self.budget_limit is not None:
            new_budget = math.fsum([self.budget, budget])
            if new_budget > self.budget_limit:
                raise BudgetExceeded(
                    f"charging {budget} ({category.value}: {note!r}) takes the budget count to "
                    f"{new_budget} > {self.budget_limit}"
                )
        self._charges.append(Charge(category, float(inferences), float(budget), note))

    def _sum(self, attribute: str, category: Category | None = None) -> float:
        values = [
            getattr(charge, attribute)
            for charge in self._charges
            if category is None or charge.category is category
        ]
        if self._base is not None:
            if category is None:
                values.append(getattr(self._base, attribute))
            else:
                base = self._base.by_category.get(category.value, (0.0, 0.0))
                values.append(base[0] if attribute == "inferences" else base[1])
        return math.fsum(values)

    @property
    def inferences(self) -> float:
        """The inference count so far."""
        return self._sum("inferences")

    @property
    def budget(self) -> float:
        """The training-budget count so far."""
        return self._sum("budget")

    def remaining_budget(self) -> float:
        """budget_limit minus the budget count. Raises ValueError if the ledger has no limit."""
        if self.budget_limit is None:
            raise ValueError("this ledger has no budget limit")
        return self.budget_limit - self.budget

    def totals(self) -> LedgerTotals:
        """The current totals, overall and per category."""
        return LedgerTotals(
            inferences=self.inferences,
            budget=self.budget,
            by_category={
                category.value: (
                    self._sum("inferences", category),
                    self._sum("budget", category),
                )
                for category in Category
            },
        )

    def charges(self) -> tuple[Charge, ...]:
        """The charges made through this ledger, in order (after `resume`, only the new ones)."""
        return tuple(self._charges)
