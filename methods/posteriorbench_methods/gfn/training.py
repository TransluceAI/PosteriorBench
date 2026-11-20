"""The GFlowNet training run of one task and replicate.

One update trains on `groups_per_update` groups of `responses_per_group` responses. A group is
either replayed from the buffer or drawn fresh from the policy at a sampling temperature of its
own (`plan_groups`); fresh responses are scored by the original model and offered to the buffer.
Every group adds the gradient of its subtrajectory-balance loss, and one AdamW step follows.

Checkpoints are saved on a schedule of updates and of budget counts, each with its selection
draws, and one more closes each training budget (`budget_rule`). Every update draws from seeds of
its own, so a training run that stops and is continued later trains the same updates as one that
runs through.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from posteriorbench import ledger as pb_ledger
from posteriorbench import rules

from posteriorbench_methods.common import (
    charges,
    hf_runtime,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.gfn import budget_rule, cost
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import replay, reward
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import subtb, teacher_forcing

METHOD = gfn_policy.METHOD


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    """What one group of an update trains on. `seed` seeds `rng` and, for a fresh group, the
    generator that draws its responses; `temperature` is None for a replayed group, which draws
    its responses from the buffer with `rng`."""

    index: int
    seed: int
    temperature: float | None
    rng: np.random.Generator

    @property
    def replay(self) -> bool:
        return self.temperature is None


def plan_groups(
    seeds_of_groups: Sequence[int], *, buffer_is_empty: bool, settings: gfn_settings.Settings
) -> list[GroupPlan]:
    """Decide for each group of an update, in order, whether it is replayed or fresh, and the
    temperature of a fresh group.

    A group is replayed with probability `replay_probability` if the buffer holds a response
    when its turn comes. The buffer does after the first fresh group of the update, because the
    first response offered to an empty buffer always enters. A fresh group is drawn at a
    temperature from U[temperature_low, temperature_high] with probability
    `tempered_probability`, else at temperature 1."""
    plans: list[GroupPlan] = []
    for index, seed in enumerate(seeds_of_groups):
        rng = np.random.default_rng(seed)
        temperature = None
        if buffer_is_empty or rng.random() >= settings.replay_probability:
            temperature = 1.0
            if rng.random() < settings.tempered_probability:
                temperature = float(
                    rng.uniform(settings.temperature_low, settings.temperature_high)
                )
            buffer_is_empty = False
        plans.append(GroupPlan(index, seed, temperature, rng))
    return plans


@dataclasses.dataclass
class Training:
    """A training run between two updates, and the steps that take it further.
    `budget_counts_reached` is how many of `settings.checkpoint_budget_counts` the budget count
    had reached after the training passes of the last update."""

    context: run_context.RunContext
    settings: gfn_settings.Settings
    policy: gfn_policy.Policy
    p: hf_runtime.CausalLM
    is_hit: Callable[[str], bool]
    directory: pathlib.Path
    optimizer: torch.optim.Optimizer
    buffer: replay.ReplayBuffer
    ledger: pb_ledger.Ledger
    checkpoints: list[training_run.Checkpoint]
    update: int = 0
    budget_counts_reached: int = 0

    def run(self, request: training_run.Request) -> training_run.Done:
        """Train until the training run ends or `request` needs no more of it."""
        final_budget = max(self.settings.training_budgets)
        while True:
            self.close_due_budgets()
            if any(final_budget in row.boundary_for for row in self.checkpoints):
                return training_run.Done("complete", self.update, math.inf)
            # No later checkpoint is saved at a lower budget count than the present one.
            if training_run.stop_reached(
                request, updates=self.update, next_checkpoint_lower_bound=self.ledger.budget
            ):
                if not self.is_saved():  # else the progress was saved with the checkpoint
                    self.save_progress()
                return training_run.Done("stopped", self.update, self.ledger.budget)
            self.train_update()
            self.save_checkpoint_if_due()

    def is_saved(self) -> bool:
        """Whether the policy as it is has been saved as a checkpoint."""
        return bool(self.checkpoints) and self.checkpoints[-1].update == self.update

    def close_due_budgets(self) -> None:
        """Close the training budgets that one more update could overrun (`budget_rule`): the
        policy as it is becomes a checkpoint, unless it is one already, and that checkpoint is
        recorded as the one that closes them."""
        settings = self.settings
        closed = {budget for row in self.checkpoints for budget in row.boundary_for}
        saved = self.is_saved()
        due = budget_rule.due(
            self.ledger.budget,
            budget_rule.update_bound(
                settings.groups_per_update * settings.responses_per_group, settings.selection_draws
            ),
            [budget for budget in settings.training_budgets if budget not in closed],
            checkpoint=0.0 if saved else budget_rule.checkpoint_passes(settings.selection_draws),
        )
        if not due:
            return
        if not saved:
            self.save_checkpoint(forced=True)
        for budget in due:
            assert self.checkpoints[-1].budget_spent <= training_run.candidate_limit(budget)
            self.checkpoints = training_run.mark_boundary(self.checkpoints, budget)
        self.save_progress()

    def train_update(self) -> None:
        """One update: plan the groups, draw the fresh ones, add the gradient of every group in
        turn, and take one AdamW step."""
        settings, policy = self.settings, self.policy
        update = self.update + 1
        learning_rate = settings.learning_rate * min((update - 1) / settings.warmup_updates, 1.0)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        plans = plan_groups(
            [
                self.context.stream_seed(METHOD, "training-group", update, group)
                for group in range(settings.groups_per_update)
            ],
            buffer_is_empty=len(self.buffer) == 0,
            settings=settings,
        )
        # The fresh groups are drawn first, together, so that several share a forward pass.
        fresh_groups = [
            hf_runtime.DrawGroup(settings.responses_per_group, plan.seed, plan.temperature)
            for plan in plans
            if plan.temperature is not None
        ]
        fresh_draws = iter(policy.draw(fresh_groups))
        losses: list[float] = []
        rows: list[dict[str, Any]] = []
        for plan in plans:
            draws = None if plan.replay else next(fresh_draws)
            loss, responses, logq = self.train_group(plan, draws)
            losses.append(loss)
            rows.extend(_training_rows(plan, responses, logq, first=len(rows)))
        parameters = [p for p in policy.model.parameters() if p.requires_grad]
        # The norm is only measured: nothing is clipped.
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("the gradient of the GFlowNet loss is not finite")
        self.optimizer.step()
        self.update = update
        run_record.write_jsonl(training_run.training_draws_path(self.directory, update), rows)
        self.context.tracker.event(
            "update",
            update=update,
            learning_rate=learning_rate,
            loss=float(np.mean(losses)),
            gradient_norm=float(gradient_norm),
            fresh_groups=len(fresh_groups),
            buffer_size=len(self.buffer),
            budget_spent=self.ledger.budget,
            inferences=self.ledger.inferences,
        )

    def train_group(
        self, plan: GroupPlan, draws: Sequence[hf_runtime.Draw] | None
    ) -> tuple[float, list[reward.Response], list[float]]:
        """Add the gradient of one group and charge what the group costs. `draws` are the
        responses of a fresh group, which are scored by the original model and by the policy and
        offered to the buffer; a replayed group, without draws, takes its responses from the
        buffer. Returns the group's loss, its responses, and log q of each under the policy."""
        if draws is None:
            responses = self.buffer.sample(plan.rng, self.settings.responses_per_group)
            loss, logq = self.add_gradient(responses)
            charges.apply(self.ledger, cost.replayed_responses(len(responses)))
            return loss, responses, logq
        responses = reward.score(
            self.p,
            self.policy.stop_token_ids,
            [draw.token_ids for draw in draws],
            decode=self.context.original.decode,
            is_hit=self.is_hit,
        )
        logq = self.policy.logq([response.token_ids for response in responses])
        charges.apply(self.ledger, cost.fresh_responses(len(responses)))
        for response in responses:
            self.buffer.add(response)
        loss, _ = self.add_gradient(responses)
        return loss, responses, logq

    def add_gradient(self, responses: Sequence[reward.Response]) -> tuple[float, list[float]]:
        """Add the gradient of one group's loss to the gradients of the policy. The loss is the
        sum of the subtrajectory-balance terms of the group's responses, divided by their number
        and by the groups of an update. Returns the mean term, and log q of each response under
        the policy."""
        settings, policy = self.settings, self.policy
        stops = policy.stop_token_ids
        terms = sum(
            subtb.term_count(
                len(r.token_ids), terminated=r.token_ids[-1] in stops, stop_tokens=len(stops)
            )
            for r in responses
        )
        policy.model.train()
        total, counted = 0.0, 0
        logq: list[float] = []
        for lo in range(0, len(responses), policy.forward_batch_size):
            batch = responses[lo : lo + policy.forward_batch_size]
            scores = teacher_forcing.token_scores(policy.lm, stops, [r.token_ids for r in batch])
            batch_sum = scores[0][0].new_zeros(())
            for response, (token_logq, stop_logq) in zip(batch, scores, strict=True):
                response_sum, count = subtb.subtb_terms(
                    token_logq,
                    stop_logq,
                    token_logq.new_tensor(response.token_logp),
                    token_logq.new_tensor(response.stop_logp),
                    response.prefix_hits,
                    epsilon=settings.epsilon,
                    terminated=response.token_ids[-1] in stops,
                )
                batch_sum = batch_sum + response_sum
                counted += count
                logq.append(float(token_logq.detach().double().sum()))
            loss = batch_sum / max(terms, 1) / settings.groups_per_update
            if not torch.isfinite(loss):
                raise FloatingPointError("the GFlowNet loss is not finite")
            loss.backward()
            total += float(batch_sum.detach())
        assert counted == terms
        return total / max(terms, 1), logq

    def save_checkpoint_if_due(self) -> None:
        """After an update: save a checkpoint if the update is on the schedule, or if the budget
        count has reached another of `checkpoint_budget_counts`."""
        counts = self.settings.checkpoint_budget_counts
        reached = sum(1 for count in counts if count <= self.ledger.budget)
        due = reached > self.budget_counts_reached
        self.budget_counts_reached = reached
        if due or self.settings.saves_checkpoint_after(self.update):
            self.save_checkpoint(forced=False)
            self.save_progress()

    def save_checkpoint(self, *, forced: bool) -> None:
        """Save the policy as a checkpoint with its selection draws. `forced` marks a checkpoint
        that is saved only to close a budget."""
        assert not self.checkpoints or self.checkpoints[-1].update < self.update
        rows = self.draw_selection_draws()
        hits = sum(bool(row["hit"]) for row in rows)
        charges.apply(self.ledger, cost.checkpoint_selection_draws(len(rows), hits))
        checkpoint_id = training_run.checkpoint_id(self.update)
        sampler_id = training_run.save_checkpoint(
            self.directory, checkpoint_id, training_run.trainable_tensors(self.policy.model)
        )
        training_run.write_selection_draws(self.directory, checkpoint_id, rows)
        self.checkpoints.append(
            training_run.Checkpoint(
                checkpoint_id=checkpoint_id,
                update=self.update,
                budget_spent=self.ledger.budget,
                ledger=self.ledger.totals().to_dict(),
                sampler_id=sampler_id,
                selection_draws=len(rows),
                selection_hits=hits,
                forced=forced,
            )
        )
        self.context.tracker.event(
            "checkpoint",
            checkpoint_id=checkpoint_id,
            update=self.update,
            selection_hits=hits,
            budget_spent=self.ledger.budget,
            forced=forced,
        )

    def draw_selection_draws(self) -> list[dict[str, Any]]:
        """The selection draws of the policy as it is, with p and q of each. They are drawn in
        parts, each from a stream of its own, and each part is scored on its own."""
        policy, decode = self.policy, self.context.original.decode
        count = self.settings.selection_draws
        groups = [
            hf_runtime.DrawGroup(
                min(gfn_policy.PART_SIZE, count - lo),
                self.context.stream_seed(METHOD, "selection", self.update, lo),
            )
            for lo in range(0, count, gfn_policy.PART_SIZE)
        ]
        rows: list[dict[str, Any]] = []
        for group, draws in zip(groups, policy.draw(groups), strict=True):
            responses = [draw.token_ids for draw in draws]
            logp = reward.log_p(self.p, policy.stop_token_ids, responses)
            logq = policy.logq(responses)
            for token_ids, p_value, q_value in zip(responses, logp, logq, strict=True):
                text = decode(token_ids)
                rows.append(
                    {
                        "draw_index": len(rows),
                        "token_ids": token_ids,
                        "text": text,
                        "hit": self.is_hit(text),
                        "logq": q_value,
                        "logp": p_value,
                        "seed": group.seed,
                    }
                )
        return rows

    def save_progress(self) -> None:
        """Write the index of the checkpoints and what training needs to continue from here."""
        training_run.write_index(self.directory, self.checkpoints)
        loop = {
            "update": self.update,
            "budget_counts_reached": self.budget_counts_reached,
            "buffer": [dataclasses.asdict(response) for response in self.buffer.responses()],
            "checkpoints": [row.to_json() for row in self.checkpoints],
            "ledger": self.ledger.totals().to_dict(),
        }
        state = training_run.ResumeState(
            tensors=training_run.trainable_tensors(self.policy.model),
            optimizer=self.optimizer.state_dict(),
            rng=training_run.capture_rng_state(),
            loop=loop,
        )
        training_run.save_resume(self.directory, state)

    def restore(self, state: training_run.ResumeState) -> None:
        """Take up the training that `save_progress` saved (changes the policy's weights)."""
        training_run.restore_trainable_tensors(self.policy.model, state.tensors)
        self.optimizer.load_state_dict(state.optimizer)
        training_run.restore_rng_state(state.rng)
        loop = state.loop
        self.update = int(loop["update"])
        self.budget_counts_reached = int(loop["budget_counts_reached"])
        self.buffer = _buffer(
            self.settings, self.policy, [reward.Response(**saved) for saved in loop["buffer"]]
        )
        self.checkpoints = [training_run.Checkpoint.from_json(row) for row in loop["checkpoints"]]
        self.ledger = pb_ledger.Ledger.resume(pb_ledger.LedgerTotals.from_dict(loop["ledger"]))


def _training_rows(
    plan: GroupPlan, responses: Sequence[reward.Response], logq: Sequence[float], *, first: int
) -> list[dict[str, Any]]:
    """The rows of one group in the file of an update's training draws. `logq` is log q of each
    response under the policy before the update's step."""
    return [
        {
            "draw_index": first + offset,
            "group": plan.index,
            "replayed": plan.replay,
            "token_ids": response.token_ids,
            "text": response.text,
            "hit": response.hit,
            "logq": q_value,
            "logp": response.logp,
            "seed": plan.seed,
            "temperature": plan.temperature,
        }
        for offset, (response, q_value) in enumerate(zip(responses, logq, strict=True))
    ]


def _buffer(
    settings: gfn_settings.Settings,
    policy: gfn_policy.Policy,
    responses: Sequence[reward.Response] = (),
) -> replay.ReplayBuffer:
    return replay.ReplayBuffer(
        size=settings.buffer_size,
        similarity=settings.buffer_similarity,
        epsilon=settings.epsilon,
        stop_token_ids=policy.stop_token_ids,
        held=responses,
    )


def is_original_model(checkpoint: training_run.Checkpoint) -> bool:
    """Whether the policy of a checkpoint is the original model. The first update has learning
    rate 0, so until the second update the adapters keep their initial value, whose second
    factor is zero."""
    return checkpoint.update <= 1


def training_run_settings(
    context: run_context.RunContext, settings: gfn_settings.Settings
) -> dict[str, Any]:
    """What makes two training runs differ: the task, the replicate and the method's settings,
    without the budgets and update caps that a run asks for."""
    method = dataclasses.asdict(settings)
    del method["budgets"], method["update_caps"]
    return {"task_id": context.task.task_id, "replicate": context.replicate, **method}


def request(settings: gfn_settings.Settings) -> training_run.Request:
    """What a run with `settings` needs from the training run: the candidates at its update caps
    and budgets, and the whole training if its largest training budget is one of its budgets."""
    return training_run.Request(
        update_caps=settings.update_caps,
        budgets=settings.budgets,
        reserve=submission.THRESHOLD_TEST_RESERVE,
        to_end=max(settings.training_budgets) in settings.budgets,
    )


def train(
    context: run_context.RunContext, settings: gfn_settings.Settings, policy: gfn_policy.Policy
) -> list[training_run.Checkpoint]:
    """Train `policy` as far as the budgets and update caps of `settings` need, and return the
    checkpoints of the training run.

    Changes the policy's weights (left in eval mode) and writes the training run directory of the
    task and replicate. One that holds every checkpoint the run needs is used as it is, one that
    holds fewer is continued, and one with other settings raises training_run.TrainingRunError."""
    directory = context.training_run_dir(METHOD)
    training_run.check_settings(directory, training_run_settings(context, settings))
    needed = request(settings)
    checkpoints = training_run.read_index(directory)
    if training_run.serves(checkpoints, training_run.read_done(directory), needed):
        return checkpoints
    training_run.clear_done(directory)
    training = Training(
        context=context,
        settings=settings,
        policy=policy,
        p=hf_runtime.CausalLM(context.original.model, tuple(context.task.prompt_token_ids)),
        is_hit=rules.parse_rule(context.task.rule),
        directory=directory,
        optimizer=torch.optim.AdamW(
            [p for p in policy.model.parameters() if p.requires_grad], lr=settings.learning_rate
        ),
        buffer=_buffer(settings, policy),
        ledger=pb_ledger.Ledger(),
        checkpoints=[],
    )
    saved = training_run.load_resume(directory)
    if saved is not None:
        training.restore(saved)
    done = training.run(needed)
    training_run.write_index(directory, training.checkpoints)
    training_run.write_done(directory, done)
    if done.status == "complete":
        training_run.delete_resume(directory)
    policy.model.eval()
    context.tracker.event("training", status=done.status, updates=done.updates)
    return training.checkpoints
