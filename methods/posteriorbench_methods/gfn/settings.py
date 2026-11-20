"""The settings of GFlowNet fine-tuning."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Settings:
    """The settings of GFlowNet and GFlowNet + composition. The defaults are the paper's.

    budgets: in passes of a model over one response; each is one of `training_budgets`.
    update_caps: a run at cap u submits one of the checkpoints saved in the first u updates; each
        cap is on the checkpoint schedule.
    training_budgets: training closes each with a checkpoint (`budget_rule`) and ends at the
        largest. `budgets` and `update_caps` only decide how far a run trains, so runs that
        differ in them share one training run.
    epsilon: the reward of a miss is epsilon p(y); that of a hit is p(y).
    warmup_updates: update u has learning rate learning_rate min((u - 1) / warmup_updates, 1), so
        the first update has learning rate 0.
    replay_probability, tempered_probability, temperature_low, temperature_high: how the groups
        of an update are drawn (`training.plan_groups`).
    buffer_similarity: two responses of n and m tokens are near-duplicates if their edit distance
        is below buffer_similarity (n + m); the buffer keeps the one with the higher reward.
    checkpoint_updates, checkpoint_interval: a checkpoint is saved after each update of
        `checkpoint_updates`, then after every multiple of `checkpoint_interval`.
    checkpoint_budget_counts: a checkpoint is also saved after the first update at which the
        budget count reaches each of these.
    generation_batch_size, forward_batch_size: the most responses in one generating and one
        teacher-forced pass; they change memory use and the rounding of sums, not the method."""

    budgets: tuple[int, ...] = (2**20, 2**21, 2**22)
    update_caps: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    training_budgets: tuple[int, ...] = (2**20, 2**21, 2**22)
    epsilon: float = 1e-8
    lora_rank: int = 64
    lora_alpha: int = 16
    learning_rate: float = 1e-4
    warmup_updates: int = 20
    groups_per_update: int = 25
    responses_per_group: int = 20
    replay_probability: float = 0.25
    tempered_probability: float = 0.666
    temperature_low: float = 0.5
    temperature_high: float = 2.0
    buffer_size: int = 50
    buffer_similarity: float = 0.25
    selection_draws: int = 128
    checkpoint_updates: tuple[int, ...] = (1, 2, 4, 8, 16, 20, 32, 64, 128)
    checkpoint_interval: int = 32
    checkpoint_budget_counts: tuple[int, ...] = (2**10, 2**12, 2**14, 2**16, 2**18, 2**20)
    generation_batch_size: int = 500
    forward_batch_size: int = 20

    def __post_init__(self) -> None:
        if not self.budgets and not self.update_caps:
            raise ValueError("a run needs at least one budget or update cap")
        if not self.training_budgets or min(self.training_budgets) < 1:
            raise ValueError(f"training_budgets are positive, got {self.training_budgets}")
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
        if not 0 < self.epsilon < 1:
            raise ValueError(f"epsilon must lie in (0, 1), got {self.epsilon}")
        counts = {
            "lora_rank": self.lora_rank,
            "warmup_updates": self.warmup_updates,
            "groups_per_update": self.groups_per_update,
            "responses_per_group": self.responses_per_group,
            "buffer_size": self.buffer_size,
            "selection_draws": self.selection_draws,
            "checkpoint_interval": self.checkpoint_interval,
            "forward_batch_size": self.forward_batch_size,
        }
        too_small = [name for name, value in counts.items() if value < 1]
        if too_small:
            raise ValueError(f"{', '.join(too_small)} must be at least 1")
        if self.learning_rate <= 0 or self.lora_alpha <= 0 or self.buffer_similarity < 0:
            raise ValueError(
                "learning_rate and lora_alpha are positive and buffer_similarity is not negative"
            )
        if not (0 <= self.replay_probability <= 1 and 0 <= self.tempered_probability <= 1):
            raise ValueError("replay_probability and tempered_probability lie in [0, 1]")
        if not 0 < self.temperature_low <= self.temperature_high:
            raise ValueError("temperatures are positive, with temperature_low <= temperature_high")
        if not self.checkpoint_updates or min(self.checkpoint_updates) < 1:
            raise ValueError("checkpoint_updates holds at least one update, and none below 1")
        if list(self.checkpoint_budget_counts) != sorted(set(self.checkpoint_budget_counts)):
            raise ValueError("checkpoint_budget_counts must increase")
        if self.generation_batch_size < self.responses_per_group:
            raise ValueError("generation_batch_size must hold one group of responses")

    def saves_checkpoint_after(self, update: int) -> bool:
        """Whether the schedule saves a checkpoint after `update` updates. The checkpoints at
        `checkpoint_budget_counts`, and those that close a budget, come on top."""
        if update in self.checkpoint_updates:
            return True
        return update > max(self.checkpoint_updates) and update % self.checkpoint_interval == 0
