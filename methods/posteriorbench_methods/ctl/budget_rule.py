"""When a training budget is closed, and when training ends.

The candidates at a budget B are the checkpoints saved while the budget count was at most the
limit of B, B less the passes set aside for its threshold tests (`training_run.candidate_limit`).

Before every update, training looks one update ahead (`due`). If an update and a checkpoint after
it would take the budget count beyond the limit of a budget, the budget is closed now: the twist
as it is becomes a checkpoint, unless it is one already, and that checkpoint is the last candidate
at the budget. Training ends when its largest budget is closed.

Every update of CTL takes the same number of passes, and so does every checkpoint (`Passes`). The
rule can therefore be followed before training: `last_update` gives the number of updates of a
training run, and `update_cap_budget` the budget that an update cap stands for.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from posteriorbench_methods.common import charges, submission, training_run
from posteriorbench_methods.ctl import cost


@dataclasses.dataclass(frozen=True)
class Passes:
    """What the two steps of training add to the budget count: 4 N passes for an update on N
    responses, and 3 m for a checkpoint with m selection draws."""

    update: float
    checkpoint: float

    @classmethod
    def of(cls, *, responses_per_update: int, selection_draws: int) -> Passes:
        """The passes that `cost` charges for an update and for a checkpoint."""
        return cls(
            update=charges.passes(cost.update(responses_per_update)),
            checkpoint=charges.passes(cost.checkpoint_selection_draws(selection_draws, 0)),
        )


def due(spent: float, open_budgets: Sequence[int], passes: Passes, *, saved: bool) -> list[int]:
    """The budgets among `open_budgets` that must be closed before another update: those whose
    limit an update and a checkpoint after it would take the budget count beyond. `spent` is the
    budget count, and `saved` says whether the twist as it is has been saved as a checkpoint.

    Raises BudgetError if such a budget can no longer be closed within its limit."""
    closing: list[int] = []
    for budget in sorted(open_budgets):
        limit = training_run.candidate_limit(budget)
        if spent + passes.update + passes.checkpoint <= limit:
            continue
        after_checkpoint = spent if saved else spent + passes.checkpoint
        if after_checkpoint > limit:
            raise training_run.BudgetError(
                f"budget {budget} cannot be closed: its limit is {limit}, and the "
                f"budget count with the checkpoint that would close it is {after_checkpoint}"
            )
        closing.append(budget)
    return closing


def next_checkpoint_lower_bound(spent: float, passes: Passes, *, saved: bool) -> float:
    """The lowest budget count at which training could save its next checkpoint: after a
    checkpoint of the twist as it is, or, if that is saved already, after one more update."""
    return spent + passes.checkpoint + (passes.update if saved else 0.0)


def last_update(
    training_budgets: Sequence[int], passes: Passes, on_schedule: Callable[[int], bool]
) -> int:
    """The number of updates of a training run under `training_budgets`, found by following the
    rule with the known passes of an update and of a checkpoint. `on_schedule(u)` says whether the
    checkpoint schedule saves a checkpoint after update u.

    Raises BudgetError if one of the budgets cannot be closed."""
    spent, update, saved = 0.0, 0, False
    open_budgets = list(training_budgets)
    while True:
        closing = due(spent, open_budgets, passes, saved=saved)
        if closing and not saved:
            spent += passes.checkpoint
        open_budgets = [budget for budget in open_budgets if budget not in closing]
        if not open_budgets:
            return update
        update += 1
        spent += passes.update
        saved = on_schedule(update)
        if saved:
            spent += passes.checkpoint


def update_cap_budget(update_cap: int, candidates: int, passes: Passes) -> int:
    """The budget that an update cap stands for: the smallest budget at which a run has the same
    candidates, the `candidates` checkpoints saved in the first `update_cap` updates. It pays for
    those updates, for the checkpoints and for the threshold tests."""
    training = update_cap * passes.update + candidates * passes.checkpoint
    return int(training) + submission.THRESHOLD_TEST_RESERVE
