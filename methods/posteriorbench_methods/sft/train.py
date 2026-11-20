"""SFT's training: maximum likelihood on the hits of a run's draws, with checkpoints.

Only the last decoder layers of a copy of the original model train. Training reads the training
set epoch after epoch, each in a new random order, in batches of `Division.batch` responses, until
it has read `Division.training_examples` responses. An update is one step of AdamW on minus the
mean log q(y) of its batch, with the gradient norm clipped. After each update of
`Division.checkpoint_updates` the run saves a checkpoint with its selection draws.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from posteriorbench import ledger

from posteriorbench_methods.common import charges, hf_runtime, run_context, training_run
from posteriorbench_methods.sft import budget_rule, cost, samplers
from posteriorbench_methods.sft import settings as sft_settings

_LAYER_OF_PARAMETER = re.compile(r"\.layers\.(\d+)\.")


@dataclasses.dataclass(frozen=True)
class SftRun:
    """One run of SFT: a task and replicate (`context`) at one budget. `corpus_cost` is what the
    corpus costs a run whose draws are the whole corpus (`cost.corpus_draws`); None for a run
    that pays for each of its draws."""

    context: run_context.RunContext
    settings: sft_settings.Settings
    division: budget_rule.Division
    models: samplers.Models
    corpus_cost: float | None = None

    @property
    def budget(self) -> int:
        return self.division.budget

    @property
    def budget_limit(self) -> int | None:
        """The most that the run may spend: its budget, or nothing for a run on a corpus, whose
        budget follows from its draws."""
        return None if self.corpus_cost is not None else self.budget

    def draws_charges(self) -> list[charges.ChargeSpec]:
        """What the run pays for its draws from the original model."""
        if self.corpus_cost is not None:
            return cost.corpus_draws(self.division.hits, self.corpus_cost)
        return cost.draws_from_p(self.division.draws)

    @property
    def directory(self) -> pathlib.Path:
        """The directory of the run's training run."""
        return self.context.training_run_dir("sft", budget=self.budget)


def freeze_all_but_last_layers(model: Any, layers: int) -> None:
    """Make the last `layers` decoder layers the only parameters of `model` that require a
    gradient (sets `requires_grad` of every parameter). Raises ValueError if the model has fewer
    layers, or if its parameters are not named "...layers.<index>...."."""
    count = int(model.config.num_hidden_layers)
    if not 1 <= layers <= count:
        raise ValueError(f"the model has {count} decoder layers, cannot train the last {layers}")
    trainable = 0
    for name, parameter in model.named_parameters():
        match = _LAYER_OF_PARAMETER.search(name)
        in_last_layers = match is not None and int(match.group(1)) >= count - layers
        parameter.requires_grad_(in_last_layers)
        trainable += in_last_layers
    if not trainable:
        raise ValueError("no parameter of the model is named as part of a decoder layer")


def open_run(
    context: run_context.RunContext,
    settings: sft_settings.Settings,
    division: budget_rule.Division,
    *,
    corpus_cost: float | None = None,
) -> SftRun:
    """A run at the budget of `division`, with a new copy of the original model whose last layers
    are trainable. `corpus_cost` is `SftRun.corpus_cost`."""
    model = context.load_model_copy().model
    freeze_all_but_last_layers(model, settings.trainable_last_layers)
    return SftRun(context, settings, division, samplers.Models.of(context, model), corpus_cost)


def load_checkpoint(run: SftRun, checkpoint: str) -> None:
    """Load the weights of a saved checkpoint into the run's model and put it in eval mode."""
    model = run.models.q.model
    tensors = training_run.load_checkpoint(run.directory, checkpoint)
    training_run.restore_trainable_tensors(model, tensors)
    model.eval()


def epoch_order(run: SftRun, epoch: int) -> list[int]:
    """The order in which epoch `epoch` (counted from 1) reads the training set: a permutation of
    its indices."""
    hits = run.division.hits
    seed = run.context.stream_seed("sft", "training-order", hits, epoch)
    return [int(index) for index in np.random.default_rng(seed).permutation(hits)]


def batches(run: SftRun, first_update: int = 1) -> Iterator[list[int]]:
    """The batches of the updates from `first_update` on, as indices into the training set.
    Training reads the epochs one after the other, so a batch can end one epoch and start the
    next."""
    division = run.division
    epoch = 0
    order: list[int] = []
    for start in range(
        (first_update - 1) * division.batch, division.training_examples, division.batch
    ):
        batch: list[int] = []
        for example in range(start, min(start + division.batch, division.training_examples)):
            if example // division.hits + 1 != epoch:
                epoch = example // division.hits + 1
                order = epoch_order(run, epoch)
            batch.append(order[example % division.hits])
        yield batch


def update_parts(
    responses: Sequence[Sequence[int]], *, prompt_tokens: int, token_limit: int
) -> list[list[Sequence[int]]]:
    """The batch of one update as the parts that pass through the model together, in the order
    of the batch. A part takes the next responses as long as its size in tokens, the number of
    its responses times the longest of them with the prompt, stays within `token_limit`; a
    response that is longer than the limit is a part of its own."""
    parts: list[list[Sequence[int]]] = []
    part: list[Sequence[int]] = []
    longest = 0
    for response in responses:
        length = prompt_tokens + len(response)
        if part and (len(part) + 1) * max(longest, length) > token_limit:
            parts.append(part)
            part, longest = [], 0
        part.append(response)
        longest = max(longest, length)
    if part:
        parts.append(part)
    return parts


def _update(
    run: SftRun,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    responses: Sequence[Sequence[int]],
) -> float:
    """One update on `responses`; returns its loss, minus the mean log q(y), whose gradient is
    added up over the parts of `update_parts`."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for part in update_parts(
        responses,
        prompt_tokens=len(run.models.q.prompt_ids),
        token_limit=run.settings.update_tokens,
    ):
        loss = -hf_runtime.sequence_logprobs_train(run.models.q, part).sum() / len(responses)
        if not torch.isfinite(loss):
            raise FloatingPointError("the training loss is not finite")
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(parameters, run.settings.gradient_clip)
    optimizer.step()
    return total


def _save_checkpoint(
    run: SftRun, spend: ledger.Ledger, update: int, training_loss: float
) -> training_run.Checkpoint:
    """Save the model's weights after `update` updates with their selection draws, charged to
    `spend`. `training_loss` is the mean loss of the updates since the last checkpoint."""
    model = run.models.q.model
    model.eval()
    checkpoint = training_run.checkpoint_id(update)
    sampler_id = training_run.save_checkpoint(
        run.directory, checkpoint, training_run.trainable_tensors(model)
    )
    draws = run.settings.selection_draws
    seed = run.context.stream_seed("sft", "selection", run.budget, run.division.hits, checkpoint)
    rows = samplers.selection_draws(run.context, samplers.sampler(run.models), draws, seed=seed)
    training_run.write_selection_draws(run.directory, checkpoint, rows)
    hits = sum(row["hit"] for row in rows)
    charges.apply(spend, cost.checkpoint_selection_draws(draws, hits))
    model.train()
    return training_run.Checkpoint(
        checkpoint_id=checkpoint,
        update=update,
        budget_spent=spend.budget,
        ledger=spend.totals().to_dict(),
        sampler_id=sampler_id,
        selection_draws=draws,
        selection_hits=hits,
        extra={
            "examples": min(update * run.division.batch, run.division.training_examples),
            "training_loss": training_loss,
        },
    )


def spend_after(checkpoint: training_run.Checkpoint, budget: int | None) -> ledger.Ledger:
    """A ledger that holds what a run had spent when it saved `checkpoint`, with `budget` as its
    limit (None for no limit)."""
    totals = ledger.LedgerTotals.from_dict(checkpoint.ledger)
    return ledger.Ledger.resume(totals, budget_limit=budget)


def training_run_settings(
    context: run_context.RunContext, settings: sft_settings.Settings, budget: int
) -> dict[str, Any]:
    """What makes two training runs differ: the task, the replicate, the budget and the method's
    settings, without the other budgets that a command asks for."""
    method = dataclasses.asdict(settings)
    del method["budgets"]
    return {
        "task_id": context.task.task_id,
        "replicate": context.replicate,
        "budget": budget,
        **method,
    }


def check_settings(run: SftRun) -> None:
    """`training_run.check_settings` with this run's `training_run_settings`: an empty directory
    records them, and one that holds other values raises TrainingRunError."""
    training_run.check_settings(
        run.directory, training_run_settings(run.context, run.settings, run.budget)
    )


def _start(
    run: SftRun, optimizer: torch.optim.Optimizer
) -> tuple[list[training_run.Checkpoint], ledger.Ledger]:
    """Where training starts: without a checkpoint, having paid for the run's draws; or after the
    last checkpoint of an interrupted training run, with the weights, optimizer and random state
    saved with it. Returns the checkpoints so far and what the run has spent."""
    resume = training_run.load_resume(run.directory)
    if resume is None:
        spend = ledger.Ledger(budget_limit=run.budget_limit)
        charges.apply(spend, run.draws_charges())
        return [], spend
    training_run.restore_trainable_tensors(run.models.q.model, resume.tensors)
    optimizer.load_state_dict(resume.optimizer)
    training_run.restore_rng_state(resume.rng)
    saved = training_run.read_index(run.directory)
    checkpoints = [row for row in saved if row.update <= resume.loop["updates"]]
    return checkpoints, spend_after(checkpoints[-1], run.budget_limit)


def train(
    run: SftRun, training_set: Sequence[Sequence[int]]
) -> tuple[list[training_run.Checkpoint], ledger.Ledger]:
    """Train the run's model on `training_set`, the hits of its draws, and save its checkpoints.
    Returns the checkpoints in the order saved, and a ledger (limit `SftRun.budget_limit`) of what
    the run has spent when training ends.

    Changes the model's weights and writes the training run to `run.directory`. A complete
    training run there is not trained again, an interrupted one continues after its last
    checkpoint, and one with other settings raises training_run.TrainingRunError."""
    division, directory = run.division, run.directory
    check_settings(run)
    checkpoints = training_run.read_index(directory)
    done = training_run.read_done(directory)
    if training_run.serves(checkpoints, done, training_run.Request(to_end=True)):
        return checkpoints, spend_after(checkpoints[-1], run.budget_limit)
    model = run.models.q.model
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=run.settings.learning_rate, weight_decay=run.settings.weight_decay
    )
    # The seed of what a forward pass in training mode draws (dropout, in a model that has it).
    torch.manual_seed(run.context.stream_seed("sft", "training-start", division.hits))
    checkpoints, spend = _start(run, optimizer)
    updates_done = checkpoints[-1].update if checkpoints else 0
    losses: list[float] = []  # of the updates since the last checkpoint
    trained = 0  # the responses that these updates read
    model.train()
    for update, batch in enumerate(batches(run, updates_done + 1), start=updates_done + 1):
        losses.append(_update(run, optimizer, parameters, [training_set[i] for i in batch]))
        trained += len(batch)
        if update not in division.checkpoint_updates:
            continue
        charges.apply(spend, cost.training(trained))
        checkpoint = _save_checkpoint(run, spend, update, math.fsum(losses) / len(losses))
        checkpoints.append(checkpoint)
        training_run.write_index(directory, checkpoints)
        training_run.save_resume(
            directory,
            training_run.ResumeState(
                tensors=training_run.trainable_tensors(model),
                optimizer=optimizer.state_dict(),
                rng=training_run.capture_rng_state(),
                loop={"updates": update},
            ),
        )
        run.context.tracker.event(
            "checkpoint",
            checkpoint_id=checkpoint.checkpoint_id,
            update=update,
            training_loss=checkpoint.extra["training_loss"],
            selection_hits=checkpoint.selection_hits,
            budget_spent=checkpoint.budget_spent,
            budget=run.budget,
        )
        losses, trained = [], 0
    assert checkpoints[-1].update == division.updates, "the last checkpoint ends training"
    training_run.write_done(
        directory,
        training_run.Done("complete", division.updates, next_checkpoint_lower_bound=math.inf),
    )
    training_run.delete_resume(directory)
    return checkpoints, spend
