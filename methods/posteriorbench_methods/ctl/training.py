"""The CTL training run of one task and replicate.

One update draws N responses from the twisted model, computes p of each of them, and takes one
step of Adam on the contrastive loss of the twist (`twist`). The weights of the loss use log q as
the twisted model generated the responses, and the gradient uses the hidden states kept from
their generation, so an update runs the model once per response.

Checkpoints are saved after the updates of the checkpoint schedule, each with its selection draws,
and one more closes each training budget (`budget_rule`). Every update and every checkpoint draws
from seeds of its own, so a training run that stops and is continued later trains the same updates
as one that runs through.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Sequence
from typing import Any

import torch
from posteriorbench import ledger

from posteriorbench_methods.common import charges, run_context, run_record, submission, training_run
from posteriorbench_methods.ctl import budget_rule, cost, p_scores
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import twist, twisted_model


def directory(context: run_context.RunContext) -> pathlib.Path:
    """The directory of the training run of the context's task and replicate."""
    return context.training_run_dir("ctl")


def load_checkpoint(
    context: run_context.RunContext, twisted: twisted_model.TwistedModel, checkpoint_id: str
) -> None:
    """Put the weights of a saved checkpoint into the twist network (changes the twisted model).
    Raises training_run.TrainingRunError if an earlier run deleted the checkpoint's file."""
    tensors = training_run.load_checkpoint(directory(context), checkpoint_id)
    twisted_model.load_twist_weights(twisted, tensors)


def selection_draws(
    context: run_context.RunContext, twisted: twisted_model.TwistedModel, count: int, *, seed: int
) -> list[dict[str, Any]]:
    """`count` selection draws of the twisted model as rows (`twisted_model.label`) with logp and
    logq, the twisted model's by teacher forcing."""
    responses = twisted_model.sample(twisted, count, seed).token_ids()
    token_logp = p_scores.token_logprobs(context.original, twisted.prompt_ids, responses)
    token_logq = twisted_model.token_logprobs(twisted, responses)
    return [
        {**row, "logq": math.fsum(q_values), "logp": math.fsum(p_values)}
        for row, q_values, p_values in zip(
            twisted_model.label(context, responses, seed=seed), token_logq, token_logp, strict=True
        )
    ]


def update_loss(
    head: twist.TwistHead,
    draws: twisted_model.Draws,
    token_logp: Sequence[Sequence[float]],
    hits: Sequence[bool],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The loss of one update on `draws`, N responses of the twisted model, with `token_logp`
    log p(y_t | y_<t) of their tokens. Returns the loss, a scalar with a gradient in the twist
    network, and the weights it holds fixed: omega^- [N, T] and omega^+ [N]."""
    device = draws.tokens.device
    lengths = torch.tensor(draws.lengths, device=device)
    logp = torch.zeros(draws.tokens.shape, device=device)
    for row, values in enumerate(token_logp):
        logp[row, : len(values)] = torch.tensor(values, device=device)
    # Like `draws.prefix_logq`, this stays at the value of the whole response after its end.
    prefix_logp = logp.cumsum(1)
    log_psi = twist.absorb(head.log_psi_of(draws.hidden, draws.tokens), lengths)
    negative = twist.negative_weights(prefix_logp, draws.prefix_logq, log_psi)
    positive = twist.positive_weights(
        prefix_logp[:, -1], draws.prefix_logq[:, -1], torch.tensor(hits, device=device), epsilon
    )
    return twist.contrastive_loss(log_psi, negative, positive), negative, positive


@dataclasses.dataclass
class Training:
    """A training run between two updates, and the steps that take it further."""

    context: run_context.RunContext
    settings: ctl_settings.Settings
    twisted: twisted_model.TwistedModel
    optimizer: torch.optim.Optimizer
    spend: ledger.Ledger
    checkpoints: list[training_run.Checkpoint]
    update: int = 0

    def run(self, request: training_run.Request) -> training_run.Done:
        """Train until the training run ends or `request` needs no more of it."""
        while True:
            self.close_due_budgets()
            closed = {budget for row in self.checkpoints for budget in row.boundary_for}
            if closed >= set(self.settings.training_budgets):
                return training_run.Done("complete", self.update, math.inf)
            # The run needs no more once its update caps are trained and no checkpoint that
            # training could still save would be a candidate at one of its budgets.
            bound = budget_rule.next_checkpoint_lower_bound(
                self.spend.budget, self.settings.passes, saved=self.is_saved()
            )
            if training_run.stop_reached(
                request, updates=self.update, next_checkpoint_lower_bound=bound
            ):
                return training_run.Done("stopped", self.update, bound)
            self.train_update()
            if self.settings.saves_checkpoint_after(self.update):
                self.save_checkpoint(forced=False)
                self.save_progress()

    def is_saved(self) -> bool:
        """Whether the twist as it is has been saved as a checkpoint."""
        return bool(self.checkpoints) and self.checkpoints[-1].update == self.update

    def close_due_budgets(self) -> None:
        """Close the training budgets that one more update would overrun (`budget_rule.due`):
        the twist as it is becomes a checkpoint, unless it is one already, and that checkpoint is
        recorded as the one that closes them."""
        closed = {budget for row in self.checkpoints for budget in row.boundary_for}
        closing = budget_rule.due(
            self.spend.budget,
            [budget for budget in self.settings.training_budgets if budget not in closed],
            self.settings.passes,
            saved=self.is_saved(),
        )
        if not closing:
            return
        if not self.is_saved():
            self.save_checkpoint(forced=True)
        for budget in closing:
            assert self.checkpoints[-1].budget_spent <= training_run.candidate_limit(budget)
            self.checkpoints = training_run.mark_boundary(self.checkpoints, budget)
        self.save_progress()

    def train_update(self) -> None:
        """One update of the twist: N responses of the twisted model, p of each, and one step
        of Adam on the contrastive loss. Writes the responses to the training run's directory."""
        context, twisted = self.context, self.twisted
        update = self.update + 1
        seed = context.stream_seed("ctl", "training", update)
        draws = twisted_model.sample(twisted, self.settings.responses_per_update, seed)
        rows = twisted_model.label(context, draws.token_ids(), seed=seed)
        token_logp = p_scores.token_logprobs(
            context.original, twisted.prompt_ids, [row["token_ids"] for row in rows]
        )
        hits = [row["hit"] for row in rows]
        self.optimizer.zero_grad(set_to_none=True)
        loss, negative, positive = update_loss(
            twisted.head, draws, token_logp, hits, self.settings.epsilon
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"the loss of update {update} is not finite")
        loss.backward()
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.float().square().sum())
                for parameter in twisted.head.parameters()
                if parameter.grad is not None
            )
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"the gradient of update {update} is not finite")
        self.optimizer.step()
        charges.apply(self.spend, cost.update(len(rows)))
        self.update = update
        run_record.write_jsonl(
            training_run.training_draws_path(directory(context), update),
            [
                {
                    **row,
                    "logq": math.fsum(q_values),
                    "logp": math.fsum(p_values),
                    "token_logq": q_values,
                    "token_logp": p_values,
                }
                for row, q_values, p_values in zip(
                    rows, draws.token_logq(), token_logp, strict=True
                )
            ],
        )
        # The effective sample size of the weights: N if they are equal, 1 if one response has
        # all the weight. The smallest over the prefix lengths, for omega^-.
        context.tracker.event(
            "update",
            update=update,
            loss=float(loss.detach()),
            gradient_norm=gradient_norm,
            hits=sum(hits),
            positive_ess=float(1 / positive.square().sum()),
            negative_ess_min=float((1 / negative.square().sum(0)).min()),
            budget_spent=self.spend.budget,
        )

    def save_checkpoint(self, *, forced: bool) -> None:
        """Save the twist as a checkpoint with its selection draws, and charge the draws.
        `forced` marks a checkpoint that is saved only to close a budget."""
        assert not self.is_saved(), "the twist is saved once after each update"
        context, twisted = self.context, self.twisted
        checkpoint_id = training_run.checkpoint_id(self.update)
        seed = context.stream_seed("ctl", "selection", self.update)
        rows = selection_draws(context, twisted, self.settings.selection_draws, seed=seed)
        hits = sum(row["hit"] for row in rows)
        charges.apply(self.spend, cost.checkpoint_selection_draws(len(rows), hits))
        training_run.write_selection_draws(directory(context), checkpoint_id, rows)
        sampler_id = training_run.save_checkpoint(
            directory(context), checkpoint_id, twisted_model.twist_weights(twisted)
        )
        self.checkpoints.append(
            training_run.Checkpoint(
                checkpoint_id=checkpoint_id,
                update=self.update,
                budget_spent=self.spend.budget,
                ledger=self.spend.totals().to_dict(),
                sampler_id=sampler_id,
                selection_draws=len(rows),
                selection_hits=hits,
                forced=forced,
            )
        )
        context.tracker.event(
            "checkpoint",
            checkpoint_id=checkpoint_id,
            update=self.update,
            selection_hits=hits,
            budget_spent=self.spend.budget,
            forced=forced,
        )

    def save_progress(self) -> None:
        """Write the index of the checkpoints and what training needs to continue from here."""
        training_run.write_index(directory(self.context), self.checkpoints)
        state = training_run.ResumeState(
            tensors=twisted_model.twist_weights(self.twisted),
            optimizer=self.optimizer.state_dict(),
            rng={},  # every draw of CTL comes from a generator with a seed of its own
            loop={
                "update": self.update,
                "checkpoints": [row.to_json() for row in self.checkpoints],
                "ledger": self.spend.totals().to_dict(),
            },
        )
        training_run.save_resume(directory(self.context), state)

    def restore(self, state: training_run.ResumeState) -> None:
        """Take up the training that `save_progress` saved (changes the twisted model)."""
        twisted_model.load_twist_weights(self.twisted, state.tensors)
        self.optimizer.load_state_dict(state.optimizer)
        self.update = int(state.loop["update"])
        self.checkpoints = [
            training_run.Checkpoint.from_json(row) for row in state.loop["checkpoints"]
        ]
        self.spend = ledger.Ledger.resume(
            ledger.LedgerTotals.from_dict(state.loop["ledger"]),
            budget_limit=self.spend.budget_limit,
        )


def training_run_settings(
    context: run_context.RunContext, settings: ctl_settings.Settings
) -> dict[str, Any]:
    """What makes two training runs differ: the task, the replicate and the method's settings,
    without the budgets and update caps that a run asks for."""
    method = dataclasses.asdict(settings)
    del method["budgets"], method["update_caps"]
    return {"task_id": context.task.task_id, "replicate": context.replicate, **method}


def request(settings: ctl_settings.Settings) -> training_run.Request:
    """What a run with `settings` needs from the training run: the candidates at its update caps
    and budgets."""
    return training_run.Request(
        update_caps=settings.update_caps,
        budgets=settings.budgets,
        reserve=submission.THRESHOLD_TEST_RESERVE,
    )


def train(
    context: run_context.RunContext,
    settings: ctl_settings.Settings,
    twisted: twisted_model.TwistedModel,
) -> list[training_run.Checkpoint]:
    """Train the twist of `twisted` as far as the budgets and update caps of `settings` need, and
    return the checkpoints of the training run.

    Changes the weights of the twist network and writes the training run directory of the task
    and replicate. One that holds every checkpoint the run needs is used as it is, one that holds
    fewer is continued, and one with other settings raises training_run.TrainingRunError."""
    run_directory = directory(context)
    training_run.check_settings(run_directory, training_run_settings(context, settings))
    needed = request(settings)
    checkpoints = training_run.read_index(run_directory)
    if training_run.serves(checkpoints, training_run.read_done(run_directory), needed):
        return checkpoints
    training_run.clear_done(run_directory)
    training = Training(
        context=context,
        settings=settings,
        twisted=twisted,
        optimizer=torch.optim.Adam(
            [value for value in twisted.head.parameters() if value.requires_grad],
            lr=settings.learning_rate,
        ),
        spend=ledger.Ledger(budget_limit=max(settings.training_budgets)),
        checkpoints=[],
    )
    saved = training_run.load_resume(run_directory)
    if saved is not None:
        training.restore(saved)
    done = training.run(needed)
    if done.status == "complete":
        training_run.delete_resume(run_directory)
    else:
        training.save_progress()
    training_run.write_done(run_directory, done)
    context.tracker.event("training", status=done.status, updates=done.updates)
    return training.checkpoints
