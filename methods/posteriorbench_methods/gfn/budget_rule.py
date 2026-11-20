"""When a training budget is closed, and when training ends.

The candidates at a budget B are the checkpoints saved while the budget count was at most the
limit of B, B less the passes set aside for its threshold tests (`training_run.candidate_limit`).

Before every update, training looks one update ahead. An update and the selection draws of a
checkpoint take at most `update_bound` passes. If that many passes could take the budget count
beyond the limit of a budget, the budget is closed now (`due`): the policy as it is becomes a
checkpoint, unless it is one already, and it is the last candidate at that budget that training
is sure to produce. An update after it may still cost little enough for one more checkpoint within
the limit, which is then a candidate too. Training ends when its largest budget is closed.
"""

from __future__ import annotations

from collections.abc import Sequence

from posteriorbench_methods.common import charges, training_run
from posteriorbench_methods.gfn import cost


def checkpoint_passes(selection_draws: int) -> float:
    """The passes that one checkpoint takes: its selection draws with their p and q."""
    return charges.passes(cost.checkpoint_selection_draws(selection_draws, 0))


def update_bound(responses: int, selection_draws: int) -> float:
    """The most passes that one update of `responses` responses and a checkpoint after it can
    take: every response fresh."""
    return charges.passes(cost.fresh_responses(responses)) + checkpoint_passes(selection_draws)


def due(spent: float, bound: float, open_budgets: Sequence[int], *, checkpoint: float) -> list[int]:
    """The budgets among `open_budgets` that must be closed before another update: those whose
    limit lies below `spent` + `bound`, where `spent` is the budget count and `bound` the most
    that an update and a checkpoint can take. `checkpoint` is what saving the current policy as
    a checkpoint would still take: 0 if it is one already.

    Raises BudgetError if a budget can no longer be closed within its limit."""
    budgets: list[int] = []
    for budget in sorted(open_budgets):
        limit = training_run.candidate_limit(budget)
        if spent + checkpoint > limit:
            raise training_run.BudgetError(
                f"budget {budget} cannot be closed: the budget count is {spent}, a checkpoint "
                f"takes {checkpoint} and the limit is {limit}"
            )
        if spent + bound > limit:
            budgets.append(budget)
    return budgets
