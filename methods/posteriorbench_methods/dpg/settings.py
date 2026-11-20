"""The settings of KL-DPG, JS-DPG and GDC."""

from __future__ import annotations

import dataclasses
from typing import Literal

from posteriorbench_methods.common import errors, submission

BUDGETS = tuple(2**exponent for exponent in range(12, 21))  # 4,096 to 1,048,576
TRAINING_BUDGET = 2**20


class MethodMismatchError(errors.MethodsError, ValueError):
    """The settings are those of another method than the one that a run was asked for."""


def _check_budgets(budgets: tuple[int, ...], training_budget: int, selection_draws: int) -> None:
    """Raise ValueError unless the budgets are distinct, each pays for its threshold tests and
    none exceeds the training budget, which pays for the first checkpoint."""
    reserve = submission.THRESHOLD_TEST_RESERVE
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError(f"a run needs at least one budget, and each budget once, got {budgets}")
    if min(budgets) <= reserve or max(budgets) > training_budget:
        raise ValueError(
            f"budgets lie above {reserve}, the passes set aside for the threshold tests, and "
            f"do not exceed training_budget = {training_budget}, got {budgets}"
        )
    if training_budget - reserve < 3 * selection_draws:
        raise ValueError(
            f"training_budget = {training_budget} does not pay for the selection draws of the "
            "first checkpoint"
        )


def _check_optimizer(
    learning_rate: float, warmup_updates: int, beta1: float, beta2: float, epsilon: float
) -> None:
    if learning_rate <= 0 or warmup_updates < 0 or epsilon <= 0:
        raise ValueError(
            "learning_rate and adam_epsilon are positive and warmup_updates is not negative"
        )
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("adam_beta1 and adam_beta2 lie in [0, 1)")


@dataclasses.dataclass(frozen=True)
class FdpgSettings:
    """The settings of KL-DPG and JS-DPG, which differ in `objective` alone. The defaults are
    the paper's. Training is Adam without weight decay or gradient clipping; its learning rate
    rises linearly from 0 over `warmup_updates`, so the first update has learning rate 0.

    objective: the divergence between the policy and the posterior that training minimizes:
        "kl", the forward KL (KL-DPG), or "js", the Jensen-Shannon divergence (JS-DPG).
    budgets: in passes of a model over one response, none above `training_budget`. A run trains
        only as far as its largest budget needs.
    training_budget: the policy is trained once, until another update might not fit in this
        budget, and the run at a budget B submits one of the checkpoints saved within B.
    baseline_decay: JS-DPG's baseline is a moving average that keeps this share of its old value
        each time it is updated, once per `baseline_group` responses of an update."""

    objective: Literal["kl", "js"] = "kl"
    budgets: tuple[int, ...] = BUDGETS
    training_budget: int = TRAINING_BUDGET
    responses_per_update: int = 256
    learning_rate: float = 1.41e-5
    warmup_updates: int = 100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    baseline_decay: float = 0.99
    baseline_group: int = 64
    selection_draws: int = 128

    def __post_init__(self) -> None:
        if self.objective not in ("kl", "js"):
            raise ValueError(f'objective is "kl" or "js", got {self.objective!r}')
        if self.responses_per_update < 1 or self.baseline_group < 1 or self.selection_draws < 1:
            raise ValueError(
                "responses_per_update, baseline_group and selection_draws are at least 1"
            )
        if not 0 <= self.baseline_decay < 1:
            raise ValueError(f"baseline_decay must lie in [0, 1), got {self.baseline_decay}")
        _check_optimizer(
            self.learning_rate,
            self.warmup_updates,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )
        _check_budgets(self.budgets, self.training_budget, self.selection_draws)

    @property
    def method(self) -> str:
        """The method that these settings run: "kl-dpg" or "js-dpg"."""
        return f"{self.objective}-dpg"


@dataclasses.dataclass(frozen=True)
class GdcSettings:
    """The settings of GDC. The defaults are the paper's; the budgets and the optimizer are as
    in `FdpgSettings`.

    responses_per_update: drawn from the sampling distribution r; a multiple of `loss_divisor`.
    loss_divisor: the loss of an update is minus the sum of (P / r) log pi over its responses,
        divided by this number. The original code averages over 256 responses at a time and
        adds up the averages of an update.
    fit_check_interval: a fit check follows every update whose number is a multiple of this."""

    budgets: tuple[int, ...] = BUDGETS
    training_budget: int = TRAINING_BUDGET
    responses_per_update: int = 2048
    loss_divisor: int = 256
    fit_check_interval: int = 10
    fit_check_draws: int = 5120
    learning_rate: float = 1.41e-5
    warmup_updates: int = 100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    selection_draws: int = 128

    def __post_init__(self) -> None:
        counts = (
            self.responses_per_update,
            self.loss_divisor,
            self.fit_check_interval,
            self.fit_check_draws,
            self.selection_draws,
        )
        if min(counts) < 1:
            raise ValueError(
                "responses_per_update, loss_divisor, fit_check_interval, fit_check_draws and "
                "selection_draws are at least 1"
            )
        if self.responses_per_update % self.loss_divisor:
            raise ValueError(
                f"responses_per_update = {self.responses_per_update} is not a multiple of "
                f"loss_divisor = {self.loss_divisor}"
            )
        _check_optimizer(
            self.learning_rate,
            self.warmup_updates,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )
        _check_budgets(self.budgets, self.training_budget, self.selection_draws)

    @property
    def method(self) -> str:
        """The method that these settings run."""
        return "gdc"


Settings = FdpgSettings | GdcSettings
