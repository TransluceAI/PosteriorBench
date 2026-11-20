"""The settings of CTL."""

from __future__ import annotations

import dataclasses

from posteriorbench_methods.ctl import budget_rule


@dataclasses.dataclass(frozen=True)
class Settings:
    """The settings of CTL and CTL + composition. The defaults are the paper's. Training is Adam
    at a constant `learning_rate`.

    budgets: in passes of a model over one response; each is one of `training_budgets`.
    update_caps: a run at cap u submits one of the checkpoints saved in the first u updates; each
        cap is on the checkpoint schedule and within training.
    training_budgets: training closes each with a checkpoint (`budget_rule`) and ends at the
        largest. `budgets` and `update_caps` only decide how far a run trains, so runs that
        differ in them share one training run.
    epsilon: the posterior weights of an update give a miss the factor epsilon and a hit
        1 + epsilon (`twist.positive_weights`), so they stay defined in an update without a hit.
    checkpoint_updates, checkpoint_interval: a checkpoint is saved after each update of
        `checkpoint_updates`, then after every multiple of `checkpoint_interval`."""

    budgets: tuple[int, ...] = (2**20, 2**21, 2**22)
    update_caps: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    training_budgets: tuple[int, ...] = (2**20, 2**21, 2**22)
    responses_per_update: int = 1000
    learning_rate: float = 1e-4
    epsilon: float = 1e-16
    selection_draws: int = 128
    checkpoint_updates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    checkpoint_interval: int = 32

    def __post_init__(self) -> None:
        if not self.budgets and not self.update_caps:
            raise ValueError("a run needs at least one budget or update cap")
        if not self.training_budgets:
            raise ValueError("a training run needs at least one training budget")
        if self.responses_per_update < 1 or self.selection_draws < 1:
            raise ValueError("responses_per_update and selection_draws are at least 1")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon are positive")
        if not self.checkpoint_updates or min(self.checkpoint_updates) < 1:
            raise ValueError("checkpoint_updates holds at least one update, and none below 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval is at least 1")
        unknown = sorted(set(self.budgets) - set(self.training_budgets))
        if unknown:
            raise ValueError(
                f"budgets {unknown} are not among the training budgets {self.training_budgets}"
            )
        off_schedule = [u for u in self.update_caps if not self.saves_checkpoint_after(u)]
        if off_schedule:
            raise ValueError(
                f"update caps {off_schedule} are not on the checkpoint schedule: "
                f"{self.checkpoint_updates}, then every {self.checkpoint_interval}th update"
            )
        last = budget_rule.last_update(
            self.training_budgets, self.passes, self.saves_checkpoint_after
        )
        beyond = [u for u in self.update_caps if u > last]
        if beyond:
            raise ValueError(
                f"update caps {beyond} lie beyond the last update of training, {last}, under the "
                f"training budgets {self.training_budgets}"
            )

    @property
    def passes(self) -> budget_rule.Passes:
        """What an update and a checkpoint add to the budget count."""
        return budget_rule.Passes.of(
            responses_per_update=self.responses_per_update, selection_draws=self.selection_draws
        )

    def saves_checkpoint_after(self, update: int) -> bool:
        """Whether the schedule saves a checkpoint after `update` updates. The checkpoints that
        close a budget come on top."""
        if update in self.checkpoint_updates:
            return True
        return update > max(self.checkpoint_updates) and update % self.checkpoint_interval == 0
