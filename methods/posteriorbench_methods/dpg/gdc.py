"""GDC: generation with distributional control (Khalifa et al., 2021).

GDC trains a policy pi toward the posterior p_S = P / p(S), where P(y) = p(y) 1[y in S], by
importance-weighted maximum likelihood on samples of a sampling distribution r. Both start as
copies of the original model p. One update

1. draws N responses y_1, ..., y_N from r;
2. takes one step of Adam on minus sum_i (P(y_i) / r(y_i)) log pi(y_i), divided by `loss_divisor`.

After every `fit_check_interval` updates, a fit check draws new responses from r and estimates on
them the forward KL from the posterior to r and to pi, with the same weights P / r
(`forward_kl_estimates`). If the estimate of pi is the lower one, r becomes a copy of pi.

A miss has weight zero, so p, r and pi are computed on the hits only, and an update without a hit
is a step of Adam on a zero gradient.

Either model could be GDC's output, so both are candidates for submission: the original model,
r whenever a fit check replaces it, pi at every fit check that keeps r, and pi after the last
update. When training ends is `budget_rule`.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Literal

import numpy as np
import torch
from posteriorbench import ledger

from posteriorbench_methods.common import charges, hf_runtime, run_record, training_run
from posteriorbench_methods.dpg import budget_rule, cost, samplers
from posteriorbench_methods.dpg import settings as dpg_settings
from posteriorbench_methods.dpg import training


def importance_weights(logp: np.ndarray, logr: np.ndarray) -> np.ndarray:
    """P(y) / r(y) of hits, from their log p and log r; float arrays [hits]. The weights are
    neither normalized nor clipped. Raises FloatingPointError if one is not finite."""
    weights = np.exp(logp - logr)
    if not np.isfinite(weights).all():
        raise FloatingPointError("an importance weight of GDC is not finite")
    return weights


def forward_kl_estimates(
    logp: np.ndarray, logr: np.ndarray, logpi: np.ndarray, *, draws: int, log_z: float
) -> tuple[float, float]:
    """The estimates of KL(p_S || r) and KL(p_S || pi) on `draws` responses of r.

    logp, logr, logpi: log p(y), log r(y) and log pi(y) of the hits among the responses; float
        arrays [hits].
    log_z: log Z, the estimate of log p(S).

    With w = P / (Z r), the estimate for a model m is the sum over the hits of
    w (log p - log m), divided by `draws`, minus log Z."""
    scaled = importance_weights(logp, logr) * math.exp(-log_z)
    kl_r = float(np.sum(scaled * (logp - logr)) / draws) - log_z
    kl_pi = float(np.sum(scaled * (logp - logpi)) / draws) - log_z
    return kl_r, kl_pi


@dataclasses.dataclass(frozen=True)
class FitCheck:
    """One fit check: `hits` hits among `draws` new responses of r, log Z over the draws of all
    fit checks so far, and the two forward-KL estimates. They are None while Z is zero."""

    draws: int
    hits: int
    log_z: float | None
    forward_kl_r: float | None
    forward_kl_pi: float | None

    @property
    def policy_is_closer(self) -> bool:
        """Whether r is replaced by pi: the estimate of pi is below that of r."""
        if self.forward_kl_r is None or self.forward_kl_pi is None:
            return False
        return self.forward_kl_pi < self.forward_kl_r


@dataclasses.dataclass
class _State:
    """One training run of GDC while it trains. `pi` and `r` are the policy and the sampling
    distribution, each under the task's prompt; `normalizer` is Z over the draws of the fit checks
    so far."""

    run: training.Training
    settings: dpg_settings.GdcSettings
    pi: samplers.Models
    r: samplers.Models
    optimizer: torch.optim.Adam
    updates: int = 0
    normalizer: training.Normalizer = dataclasses.field(default_factory=training.Normalizer)
    spend: ledger.Ledger = dataclasses.field(default_factory=ledger.Ledger)
    checkpoints: list[training_run.Checkpoint] = dataclasses.field(default_factory=lambda: [])


def checkpoint_id(model: Literal["r", "pi"], update: int) -> str:
    """The id of the candidate that is `model` after `update` updates of the policy:
    "r_update_000010", "pi_update_000020"."""
    return f"{model}_{training_run.checkpoint_id(update)}"


def _save_candidate(
    state: _State, model: Literal["r", "pi"], fit_check: FitCheck | None = None
) -> None:
    """Save r or pi as it is as a candidate. `fit_check` is the fit check that it follows."""
    training.save_checkpoint(
        state.run,
        state.r if model == "r" else state.pi,
        state.spend,
        state.checkpoints,
        checkpoint_id=checkpoint_id(model, state.updates),
        update=state.updates,
        extra={
            "model": model,
            "fit_check": None if fit_check is None else dataclasses.asdict(fit_check),
        },
    )


def _save_resume(state: _State) -> None:
    """Save what training needs to continue from here: both models."""
    tensors = {
        f"{name}.{key}": value
        for name, models in (("pi", state.pi), ("r", state.r))
        for key, value in training.weights(models.q.model).items()
    }
    training.save_resume(
        state.run,
        tensors=tensors,
        optimizer=state.optimizer,
        spend=state.spend,
        updates=state.updates,
        counters={"normalizer": dataclasses.asdict(state.normalizer)},
    )


def _start(state: _State) -> None:
    """Start training: at the original model, which is saved as the first candidate; or, if the
    directory holds a training run that stopped or was interrupted, where that run last saved
    its state."""
    resumed = training.load_resume(state.run)
    if resumed is None:
        _save_candidate(state, "r")
        _save_resume(state)
        return
    saved = resumed.state
    for name, models in (("pi", state.pi), ("r", state.r)):
        prefix = f"{name}."
        models.q.model.load_state_dict(
            {key[len(prefix) :]: v for key, v in saved.tensors.items() if key.startswith(prefix)}
        )
    state.optimizer.load_state_dict(saved.optimizer)
    state.updates = saved.loop["updates"]
    state.normalizer = training.Normalizer(**saved.loop["normalizer"])
    state.spend, state.checkpoints = resumed.spend, resumed.checkpoints


def _draws_from_r(state: _State, count: int, *, seed: int) -> list[samplers.Row]:
    """`count` new responses of r as draw rows (`samplers.training_draws`). A hit also has logp
    and logr, its log-probabilities under the original model and, by teacher forcing, under r.
    They are None on a miss, like logpi, which the caller fills in."""
    context = state.run.context
    rows = samplers.training_draws(context, samplers.sampler(state.r), count, seed=seed)
    hits = [row for row in rows if row["hit"]]
    responses = [row["token_ids"] for row in hits]
    logp = samplers.log_p(context, responses)
    logr = hf_runtime.sequence_logprobs(state.r.q, responses, batch_size=samplers.SCORING_BATCH)
    for row in rows:
        row.update(logp=None, logr=None, logpi=None)
    for row, p_value, r_value in zip(hits, logp, logr, strict=True):
        row.update(logp=p_value, logr=r_value)
    return rows


def set_zero_gradients(model: Any) -> None:
    """Give every trainable weight of `model` a gradient of zero, the gradient of an update
    without a hit, so that Adam still takes its step."""
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)


def check_every_weight_has_a_gradient(model: Any) -> None:
    """Raise RuntimeError if a backward pass left a trainable weight of `model` without a
    gradient. Zero gradients for an update without a hit (`set_zero_gradients`) are the same step
    of Adam only if an update with hits gives every weight a gradient."""
    without = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if without:
        raise RuntimeError(f"weights without a gradient after a backward pass: {without[:5]}")


def _update(state: _State) -> None:
    """One update of the policy (module docstring), charged to `state.spend`. Writes the
    update's draws to the training run's directory, with log p, log r and log pi of the hits."""
    context, settings, model = state.run.context, state.settings, state.pi.q.model
    update = state.updates + 1
    rows = _draws_from_r(
        state, settings.responses_per_update, seed=context.stream_seed("gdc", "training", update)
    )
    hits = [row for row in rows if row["hit"]]
    weights = importance_weights(
        np.array([row["logp"] for row in hits], dtype=np.float64),
        np.array([row["logr"] for row in hits], dtype=np.float64),
    )
    weight_of = dict(zip((row["draw_index"] for row in hits), weights.tolist(), strict=True))
    rate = training.learning_rate(state.updates, settings)
    for group in state.optimizer.param_groups:
        group["lr"] = rate
    state.optimizer.zero_grad(set_to_none=True)
    # The update reads its hits in a random order, from Python's global generator, which
    # `training.seed_global_generators` seeds when training starts.
    order = list(range(len(rows)))
    random.shuffle(order)
    read = [rows[index] for index in order if rows[index]["hit"]]
    for start in range(0, len(read), samplers.SCORING_BATCH):
        batch = read[start : start + samplers.SCORING_BATCH]
        logpi = hf_runtime.sequence_logprobs_train(state.pi.q, [row["token_ids"] for row in batch])
        batch_weights = torch.as_tensor(
            [weight_of[row["draw_index"]] for row in batch], device=logpi.device, dtype=logpi.dtype
        )
        loss = -(batch_weights * logpi).sum() / settings.loss_divisor
        if not torch.isfinite(loss):
            raise FloatingPointError("the loss of an update is not finite")
        loss.backward()
        for row, value in zip(batch, logpi.detach().cpu().tolist(), strict=True):
            row["logpi"] = value
    if read:
        check_every_weight_has_a_gradient(model)
    else:
        set_zero_gradients(model)
    training.check_gradients(model)
    state.optimizer.step()
    run_record.write_jsonl(training_run.training_draws_path(state.run.directory, update), rows)
    charges.apply(state.spend, cost.gdc_update(len(rows), len(hits)))
    state.updates = update
    context.tracker.event(
        "update",
        update=update,
        hits=len(hits),
        learning_rate=rate,
        budget_spent=state.spend.budget,
        inferences=state.spend.inferences,
    )


def _fit_check(state: _State) -> FitCheck:
    """The fit check after the update just done, charged to `state.spend`: `fit_check_draws` new
    responses of r, drawn in parts of at most `responses_per_update`, with log p, log r and
    log pi of their hits. Adds the draws to Z. Writes them to the training run's directory."""
    context, settings, update = state.run.context, state.settings, state.updates
    rows: list[samplers.Row] = []
    starts = range(0, settings.fit_check_draws, settings.responses_per_update)
    for part, start in enumerate(starts):
        count = min(settings.responses_per_update, settings.fit_check_draws - start)
        seed = context.stream_seed("gdc", "fit-check", update, part)
        drawn = _draws_from_r(state, count, seed=seed)
        part_hits = [row for row in drawn if row["hit"]]
        logpi = hf_runtime.sequence_logprobs(
            state.pi.q, [row["token_ids"] for row in part_hits], batch_size=samplers.SCORING_BATCH
        )
        for row, value in zip(part_hits, logpi, strict=True):
            row["logpi"] = value
        rows += [{**row, "part": part} for row in drawn]
    hits = [row for row in rows if row["hit"]]
    logp, logr, logpi = (
        np.array([row[key] for row in hits], dtype=np.float64) for key in ("logp", "logr", "logpi")
    )
    state.normalizer.add((logp - logr).tolist(), draws=len(rows))
    log_z = state.normalizer.log_z
    kl_r: float | None = None
    kl_pi: float | None = None
    if log_z is not None:
        kl_r, kl_pi = forward_kl_estimates(logp, logr, logpi, draws=len(rows), log_z=log_z)
    charges.apply(state.spend, cost.gdc_fit_check(len(rows), len(hits)))
    run_record.write_jsonl(state.run.directory / "training" / f"fit_check_{update:06d}.jsonl", rows)
    check = FitCheck(len(rows), len(hits), log_z, kl_r, kl_pi)
    context.tracker.event("fit_check", update=update, **dataclasses.asdict(check))
    return check


def train(run: training.Training) -> tuple[list[training_run.Checkpoint], samplers.Models]:
    """Train the policy of GDC as far as the run's budgets need. Returns the candidates in the
    order saved, and the policy's model under the task's prompt for the caller to load them
    into. Writes the training run to its directory; a training run there that stopped or was
    interrupted continues from its last saved state, random state included."""
    context, settings = run.context, run.settings
    assert isinstance(settings, dpg_settings.GdcSettings)
    training.seed_global_generators(context.stream_seed("gdc", "initialization"))
    pi_model = context.load_model_copy().model
    r_model = context.load_model_copy().model
    pi_model.eval()
    r_model.eval()
    for parameter in r_model.parameters():
        parameter.requires_grad_(False)
    pi = samplers.Models.of(context, pi_model)
    optimizer = training.new_optimizer(pi_model, settings)
    state = _State(run, settings, pi, pi.with_model(r_model), optimizer)
    _start(state)
    while not budget_rule.gdc_training_ends(state.spend.budget, state.updates, settings):
        at_least = budget_rule.gdc_next_checkpoint_at_least(
            state.spend.budget, state.updates, settings
        )
        if training_run.stop_reached(
            run.request, updates=state.updates, next_checkpoint_lower_bound=at_least
        ):
            if state.checkpoints[-1].update != state.updates:
                _save_resume(state)
            training.stop(run, state.checkpoints, state.updates, at_least)
            return state.checkpoints, pi
        _update(state)
        if budget_rule.gdc_fit_check_due(state.updates, settings):
            fit_check = _fit_check(state)
            if fit_check.policy_is_closer:
                r_model.load_state_dict(pi_model.state_dict())
                _save_candidate(state, "r", fit_check)
            else:
                _save_candidate(state, "pi", fit_check)
            _save_resume(state)
    # The policy after the last update is a candidate too, and the last candidate closes the
    # training budget.
    if state.checkpoints[-1].update != state.updates:
        _save_candidate(state, "pi")
    return training.complete(run, state.checkpoints, state.updates), pi
