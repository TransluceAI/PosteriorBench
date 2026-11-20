"""What the training loops of KL-DPG, JS-DPG (`fdpg`) and GDC (`gdc`) share. A training run is
kept in a directory in the layout of `common.training_run`, trains only as far as the budgets of
its run need, and continues when a later run needs more of it.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from posteriorbench import ledger

from posteriorbench_methods.common import charges, run_context, submission, training_run
from posteriorbench_methods.dpg import cost, samplers
from posteriorbench_methods.dpg import settings as dpg_settings


@dataclasses.dataclass(frozen=True)
class Training:
    """The training run of one method on one task and replicate."""

    context: run_context.RunContext
    settings: dpg_settings.Settings

    @property
    def method(self) -> str:
        return self.settings.method

    @property
    def directory(self) -> pathlib.Path:
        return self.context.training_run_dir(self.method)

    @property
    def request(self) -> training_run.Request:
        """What the run needs from the training: the candidates at its budgets below the training
        budget, or the whole training if the training budget is one of its budgets."""
        settings = self.settings
        return training_run.Request(
            budgets=tuple(
                budget for budget in settings.budgets if budget < settings.training_budget
            ),
            reserve=submission.THRESHOLD_TEST_RESERVE,
            to_end=settings.training_budget in settings.budgets,
        )


@dataclasses.dataclass
class Normalizer:
    """Z, the running importance-sampling estimate of p(S): the mean of P(y) / s(y) over all the
    draws added so far, where P(y) = p(y) 1[y in S] and s is the distribution that drew y.

    draws: the number of draws, hits and misses.
    log_weight_sum: the log of the sum of p(y) / s(y) over the hits; None before the first hit."""

    draws: int = 0
    log_weight_sum: float | None = None

    def add(self, log_weights: Sequence[float], *, draws: int) -> None:
        """Add `draws` new draws. `log_weights` holds log p(y) - log s(y) of the hits among
        them."""
        self.draws += draws
        if len(log_weights) == 0:
            return
        added = float(np.logaddexp.reduce(np.asarray(log_weights, dtype=np.float64)))
        if not math.isfinite(added):
            raise FloatingPointError("the log weight of a hit is not finite")
        if self.log_weight_sum is None:
            self.log_weight_sum = added
        else:
            self.log_weight_sum = float(np.logaddexp(self.log_weight_sum, added))

    @property
    def log_z(self) -> float | None:
        """log Z; None while Z is zero, that is, before the first hit."""
        if self.log_weight_sum is None:
            return None
        return self.log_weight_sum - math.log(self.draws)


def learning_rate(updates_done: int, settings: dpg_settings.Settings) -> float:
    """The learning rate of the update that follows `updates_done` updates. It rises linearly
    from 0 over `warmup_updates` updates and then stays at `learning_rate`."""
    return settings.learning_rate * min(1.0, updates_done / max(1, settings.warmup_updates))


def new_optimizer(model: Any, settings: dpg_settings.Settings) -> torch.optim.Adam:
    """Adam over every weight of `model`. The loops set the learning rate before each update."""
    return torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_epsilon,
    )


def check_gradients(model: Any) -> None:
    """Raise FloatingPointError if a gradient of `model` is not finite."""
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("a gradient is not finite")


def seed_global_generators(seed: int) -> None:
    """Seed the global random generators of Python, numpy and torch. Responses are drawn from
    generators of their own (`common.hf_runtime`); the global generators give the order in which
    GDC reads the responses of an update, and whatever a model draws when it is loaded."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weights(model: Any) -> dict[str, torch.Tensor]:
    """Every tensor of the model's state dict, on the CPU. The three methods train all the
    weights, so a checkpoint is the whole model."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(
    training: Training,
    models: samplers.Models,
    spend: ledger.Ledger,
    checkpoints: list[training_run.Checkpoint],
    *,
    checkpoint_id: str,
    update: int,
    forced: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> training_run.Checkpoint:
    """Save the weights that `models.q` holds as a checkpoint of the training run, with its
    selection draws charged to `spend`. Writes the draws and the weights (weights that an earlier
    checkpoint has are stored once), appends the checkpoint to `checkpoints` and rewrites the
    index. `forced` marks a checkpoint off the method's schedule; `extra` holds the method's own
    fields."""
    context, settings, directory = training.context, training.settings, training.directory
    seed = context.stream_seed(training.method, "selection", checkpoint_id)
    rows = samplers.selection_draws(
        context, samplers.sampler(models), settings.selection_draws, seed=seed
    )
    hits = sum(row["hit"] for row in rows)
    charges.apply(spend, cost.checkpoint_selection_draws(settings.selection_draws, hits))
    training_run.write_selection_draws(directory, checkpoint_id, rows)
    sampler_id = training_run.save_checkpoint(
        directory, checkpoint_id, weights(models.q.model), earlier=checkpoints
    )
    checkpoint = training_run.Checkpoint(
        checkpoint_id=checkpoint_id,
        update=update,
        budget_spent=spend.budget,
        ledger=spend.totals().to_dict(),
        sampler_id=sampler_id,
        selection_draws=settings.selection_draws,
        selection_hits=hits,
        forced=forced,
        extra=dict(extra or {}),
    )
    checkpoints.append(checkpoint)
    training_run.write_index(directory, checkpoints)
    context.tracker.event(
        "checkpoint",
        checkpoint_id=checkpoint_id,
        update=update,
        selection_hits=hits,
        budget_spent=spend.budget,
    )
    return checkpoint


def training_run_settings(training: Training) -> dict[str, Any]:
    """What makes two training runs differ: the task, the replicate and the method's settings,
    without the budgets that a run asks for."""
    method = dataclasses.asdict(training.settings)
    del method["budgets"]
    return {
        "task_id": training.context.task.task_id,
        "replicate": training.context.replicate,
        **method,
    }


def check_directory(training: Training) -> list[training_run.Checkpoint] | None:
    """The checkpoints in the training run's directory if it holds every candidate that the
    run's budgets need; else None, after removing done.json so that training starts or
    continues. A directory with another task, replicate or settings raises
    training_run.TrainingRunError; an empty one records them (settings.json)."""
    directory = training.directory
    training_run.check_settings(directory, training_run_settings(training))
    checkpoints = training_run.read_index(directory)
    if training_run.serves(checkpoints, training_run.read_done(directory), training.request):
        return checkpoints
    training_run.clear_done(directory)
    return None


@dataclasses.dataclass
class Resumed:
    """Where an interrupted or stopped training run continues: the weights, the state of the
    optimizer and the method's counters (`state`), what the run had spent, and the checkpoints
    it had saved."""

    state: training_run.ResumeState
    spend: ledger.Ledger
    checkpoints: list[training_run.Checkpoint]


def load_resume(training: Training) -> Resumed | None:
    """What `save_resume` last saved in the training run's directory; None if training starts
    from the original model. Sets the global random generators to their saved state."""
    state = training_run.load_resume(training.directory)
    if state is None:
        return None
    training_run.restore_rng_state(state.rng)
    saved = training_run.read_index(training.directory)
    return Resumed(
        state=state,
        spend=ledger.Ledger.resume(ledger.LedgerTotals.from_dict(state.loop["ledger"])),
        checkpoints=[row for row in saved if row.update <= state.loop["updates"]],
    )


def save_resume(
    training: Training,
    *,
    tensors: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    spend: ledger.Ledger,
    updates: int,
    counters: Mapping[str, Any],
) -> None:
    """Write what training needs to continue after `updates` updates (resume.pt): the weights,
    the state of the optimizer and of the global random generators, what the run has spent, and
    the method's `counters`."""
    training_run.save_resume(
        training.directory,
        training_run.ResumeState(
            tensors=tensors,
            optimizer=optimizer.state_dict(),
            rng=training_run.capture_rng_state(),
            loop={"updates": updates, "ledger": spend.totals().to_dict(), **counters},
        ),
    )


def stop(
    training: Training,
    checkpoints: Sequence[training_run.Checkpoint],
    updates: int,
    next_checkpoint_at_least: float,
) -> None:
    """Record that training has stopped after `updates` updates because the run's budgets need
    no more (done.json). `next_checkpoint_at_least` is the least budget count at which a further
    checkpoint could be saved; it tells a later run whether it has to train on."""
    done = training_run.Done("stopped", updates, next_checkpoint_at_least)
    assert training_run.serves(checkpoints, done, training.request), "training stopped too early"
    training_run.write_done(training.directory, done)


def complete(
    training: Training, checkpoints: Sequence[training_run.Checkpoint], updates: int
) -> list[training_run.Checkpoint]:
    """Record that training has reached the end of its training budget after `updates` updates:
    the last checkpoint closes the training budget (index.jsonl), done.json is written and
    resume.pt deleted. Returns the checkpoints, the last one with its mark."""
    budget = training.settings.training_budget
    assert checkpoints[-1].budget_spent <= training_run.candidate_limit(
        budget
    ), "training went on too long"
    marked = list(checkpoints)
    if budget not in marked[-1].boundary_for:
        marked = training_run.mark_boundary(marked, budget)
        training_run.write_index(training.directory, marked)
    training_run.write_done(training.directory, training_run.Done("complete", updates, math.inf))
    training_run.delete_resume(training.directory)
    return marked
