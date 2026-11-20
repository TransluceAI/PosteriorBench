"""The settings of SFT."""

from __future__ import annotations

import dataclasses
from typing import Literal

TrainingSet = Literal["draws", "corpus"]


@dataclasses.dataclass(frozen=True)
class Settings:
    """The settings of SFT, Replay and SFT + composition. The defaults are the paper's. Training
    is AdamW at a constant `learning_rate`, with the gradient norm clipped to `gradient_clip`.

    budgets: in passes of a model over one response. SFT trains separately at each.
    training_set: "draws", the hits among the draws that a budget pays for
        (`budget_rule.divide`); or "corpus", every corpus hit outside the references, with one
        budget that follows from the size of the corpus (`budget_rule.divide_corpus`) and
        `budgets` not read.
    training_share: in (0, 1); the rest of a budget pays for draws from the original model.
    trainable_last_layers: counted from the last decoder layer; all other weights stay frozen.
    update_tokens: the most tokens in one forward pass of an update, counted as the responses in
        the pass times the longest of them with its prompt. A batch that holds more is computed
        in parts whose gradients are added, so the update is that of the whole batch. The default
        holds 5 responses of 2,048 tokens; the benchmark's batches are far below it."""

    budgets: tuple[int, ...] = tuple(2**exponent for exponent in range(12, 23))
    training_set: TrainingSet = "draws"
    training_share: float = 0.25
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    trainable_last_layers: int = 4
    batch_size: int = 32
    update_tokens: int = 11104
    selection_draws: int = 128

    def __post_init__(self) -> None:
        if any(budget < 1 for budget in self.budgets):
            raise ValueError(f"budgets are positive, got {self.budgets}")
        if not 0 < self.training_share < 1:
            raise ValueError(f"training_share must lie in (0, 1), got {self.training_share}")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError(
                "learning_rate and gradient_clip are positive and weight_decay is not negative"
            )
        if self.trainable_last_layers < 1 or self.batch_size < 1 or self.selection_draws < 1:
            raise ValueError("trainable_last_layers, batch_size and selection_draws are at least 1")
        if self.training_set not in ("draws", "corpus"):
            raise ValueError(f'training_set is "draws" or "corpus", got {self.training_set!r}')
        if self.update_tokens < 1:
            raise ValueError(f"update_tokens must be positive, got {self.update_tokens}")
