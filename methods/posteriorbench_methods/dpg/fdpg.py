"""KL-DPG and JS-DPG: f-DPG (Go et al., 2023) with the forward KL and the Jensen-Shannon
divergence.

f-DPG fine-tunes a policy pi, a copy of the original model p, toward the posterior
p_S = P / p(S), where P(y) = p(y) 1[y in S], by minimizing an f-divergence between the two on the
policy's own samples. One update

1. draws N responses y_1, ..., y_N from the policy;
2. adds them to Z, the mean of P(y) / pi(y) over all the training draws so far, each with the
   policy that drew it (`training.Normalizer`). While Z is zero, that is, until a hit has been
   drawn, the update takes no gradient step;
3. gives each response a coefficient a_i, from eta_i = P(y_i) / (Z pi(y_i)), the estimate of
   p_S(y_i) / pi(y_i):

       KL-DPG   a_i = 1 - eta_i
       JS-DPG   a_i = log 2 - log(1 + eta_i) - beta

   where beta is a moving average of log 2 - log(1 + eta) (`kl_coefficients`, `js_coefficients`);
4. takes one step of Adam on sum_i a_i log pi(y_i) / N, with the a_i held fixed.

Training starts at the original model; when it saves checkpoints and when it ends is
`budget_rule`.

A response has two values of log pi: the one that generation recorded when the response was drawn,
and the one from the forward pass of the update. They differ by floating-point rounding. Z uses the
first, since it is the distribution that the response was in fact drawn from. The coefficients are
multiplied by rho = pi_forward(y) / pi_drawn(y), which is 1 up to that rounding, so that the
gradient estimate is exact for that distribution.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from posteriorbench import ledger

from posteriorbench_methods.common import charges, hf_runtime, run_record, training_run
from posteriorbench_methods.dpg import budget_rule, cost, samplers
from posteriorbench_methods.dpg import settings as dpg_settings
from posteriorbench_methods.dpg import training


def kl_coefficients(
    logp: np.ndarray, logq_drawn: np.ndarray, logq: np.ndarray, hit: np.ndarray, log_z: float
) -> np.ndarray:
    """KL-DPG's coefficients rho (1 - eta) of a batch of responses.

    logp, logq_drawn, logq: log p(y), the log pi(y) recorded when y was drawn, and log pi(y) from
        the forward pass; float arrays [responses].
    hit: whether y is a hit; bool array [responses].
    log_z: log Z, finite.

    Raises FloatingPointError if a coefficient is not finite."""
    rho = np.exp(logq - logq_drawn)
    coefficients = rho - np.where(hit, np.exp(logp - log_z - logq_drawn), 0.0)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("a coefficient of KL-DPG is not finite")
    return coefficients


def js_coefficients(
    logp: np.ndarray,
    logq_drawn: np.ndarray,
    logq: np.ndarray,
    hit: np.ndarray,
    log_z: float,
    *,
    baseline: float,
    decay: float,
) -> tuple[np.ndarray, float]:
    """JS-DPG's coefficients rho (log 2 - log(1 + eta) - beta) of a group of responses, and the
    new baseline beta.

    The arrays and `log_z` are those of `kl_coefficients`. The baseline is updated before it is
    subtracted: beta = decay * baseline + (1 - decay) * mean of rho (log 2 - log(1 + eta)) over the
    group.

    Raises FloatingPointError if a coefficient is not finite."""
    log_eta = np.where(hit, logp - log_z - logq, -np.inf)
    rho = np.exp(logq - logq_drawn)
    weighted = rho * (math.log(2) - np.logaddexp(0.0, log_eta))
    beta = decay * baseline + (1 - decay) * float(weighted.mean())
    coefficients = weighted - rho * beta
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("a coefficient of JS-DPG is not finite")
    return coefficients, beta


@dataclasses.dataclass
class _State:
    """One training run of KL-DPG or JS-DPG while it trains. `normalizer` is Z over the training
    draws so far, and `baseline` JS-DPG's moving average beta."""

    run: training.Training
    settings: dpg_settings.FdpgSettings
    models: samplers.Models
    optimizer: torch.optim.Adam
    updates: int = 0
    normalizer: training.Normalizer = dataclasses.field(default_factory=training.Normalizer)
    baseline: float = 0.0
    spend: ledger.Ledger = dataclasses.field(default_factory=ledger.Ledger)
    checkpoints: list[training_run.Checkpoint] = dataclasses.field(default_factory=lambda: [])


def _save_checkpoint(state: _State, *, forced: bool = False) -> None:
    """Save the policy as it is as a checkpoint."""
    training.save_checkpoint(
        state.run,
        state.models,
        state.spend,
        state.checkpoints,
        checkpoint_id=training_run.checkpoint_id(state.updates),
        update=state.updates,
        forced=forced,
        extra={"log_z": state.normalizer.log_z},
    )


def _save_resume(state: _State) -> None:
    """Save what training needs to continue from here."""
    training.save_resume(
        state.run,
        tensors=training.weights(state.models.q.model),
        optimizer=state.optimizer,
        spend=state.spend,
        updates=state.updates,
        counters={
            "normalizer": dataclasses.asdict(state.normalizer),
            "baseline": state.baseline,
        },
    )


def _start(state: _State) -> None:
    """Start training: at the original model, which is saved as the first checkpoint; or, if the
    directory holds a training run that stopped or was interrupted, where that run last saved
    its state."""
    resumed = training.load_resume(state.run)
    if resumed is None:
        _save_checkpoint(state)
        _save_resume(state)
        return
    saved = resumed.state
    state.models.q.model.load_state_dict(saved.tensors)
    state.optimizer.load_state_dict(saved.optimizer)
    state.updates = saved.loop["updates"]
    state.normalizer = training.Normalizer(**saved.loop["normalizer"])
    state.baseline = saved.loop["baseline"]
    state.spend, state.checkpoints = resumed.spend, resumed.checkpoints


def _accumulate_gradient(state: _State, rows: list[samplers.Row], log_z: float) -> list[float]:
    """Add the gradient of sum_i a_i log pi(y_i) / N to the gradients of the policy's weights, in
    batches of `samplers.SCORING_BATCH` responses. Returns log pi(y_i) from the forward passes.

    KL-DPG computes the coefficients of a batch from the log pi of its forward pass. JS-DPG
    updates its baseline once per group of `baseline_group` responses, across these batches, so it
    takes the log pi for its coefficients from a pass of its own over the update's responses."""
    settings, q = state.settings, state.models.q
    responses = [row["token_ids"] for row in rows]
    logp = np.array([row["logp"] for row in rows], dtype=np.float64)
    logq_drawn = np.array([row["logq_generation"] for row in rows], dtype=np.float64)
    hit = np.array([row["hit"] for row in rows], dtype=bool)
    js: list[float] = []
    if settings.objective == "js":
        separate = np.array(
            hf_runtime.sequence_logprobs(q, responses, batch_size=samplers.SCORING_BATCH),
            dtype=np.float64,
        )
        for start in range(0, len(rows), settings.baseline_group):
            group = slice(start, start + settings.baseline_group)
            coefficients, state.baseline = js_coefficients(
                logp[group],
                logq_drawn[group],
                separate[group],
                hit[group],
                log_z,
                baseline=state.baseline,
                decay=settings.baseline_decay,
            )
            js += coefficients.tolist()
    logq: list[float] = []
    for start in range(0, len(rows), samplers.SCORING_BATCH):
        batch = slice(start, start + samplers.SCORING_BATCH)
        values = hf_runtime.sequence_logprobs_train(q, responses[batch])  # [batch], float32
        batch_logq = values.detach().cpu().tolist()
        if settings.objective == "js":
            coefficients = np.asarray(js[batch])
        else:
            coefficients = kl_coefficients(
                logp[batch], logq_drawn[batch], np.asarray(batch_logq), hit[batch], log_z
            )
        weights = torch.as_tensor(coefficients, device=values.device, dtype=values.dtype)
        loss = (weights * values).sum() / len(rows)
        if not torch.isfinite(loss):
            raise FloatingPointError("the loss of an update is not finite")
        loss.backward()
        logq += batch_logq
    return logq


def _update(state: _State) -> None:
    """One update of the policy (module docstring), charged to `state.spend`. Writes the
    update's draws to the training run's directory, with their log p and both values of
    log pi."""
    context, settings, q = state.run.context, state.settings, state.models.q
    update = state.updates + 1
    rows = samplers.training_draws(
        context,
        samplers.sampler(state.models),
        settings.responses_per_update,
        seed=context.stream_seed(state.run.method, "training", update),
    )
    responses = [row["token_ids"] for row in rows]
    for row, logp in zip(rows, samplers.log_p(context, responses), strict=True):
        row["logp"] = logp
    hits = [row for row in rows if row["hit"]]
    state.normalizer.add([row["logp"] - row["logq_generation"] for row in hits], draws=len(rows))
    log_z = state.normalizer.log_z
    rate = training.learning_rate(state.updates, settings)
    for group in state.optimizer.param_groups:
        group["lr"] = rate
    state.optimizer.zero_grad(set_to_none=True)
    if log_z is None:
        # No hit yet: a forward pass for the record of the draws, and no gradient step.
        logq = hf_runtime.sequence_logprobs(q, responses, batch_size=samplers.SCORING_BATCH)
    else:
        logq = _accumulate_gradient(state, rows, log_z)
        training.check_gradients(q.model)
        state.optimizer.step()
    for row, value in zip(rows, logq, strict=True):
        row["logq"] = value
    run_record.write_jsonl(training_run.training_draws_path(state.run.directory, update), rows)
    charges.apply(
        state.spend,
        cost.fdpg_update(
            len(rows), len(hits), objective=settings.objective, has_gradient=log_z is not None
        ),
    )
    state.updates = update
    context.tracker.event(
        "update",
        update=update,
        hits=len(hits),
        log_z=log_z,
        learning_rate=rate,
        budget_spent=state.spend.budget,
        inferences=state.spend.inferences,
    )


def train(run: training.Training) -> tuple[list[training_run.Checkpoint], samplers.Models]:
    """Train the policy of KL-DPG or JS-DPG as far as the run's budgets need. Returns the
    checkpoints in the order saved, and the policy's model under the task's prompt for the
    caller to load them into. Writes the training run to its directory; a training run there
    that stopped or was interrupted continues from its last saved state, random state
    included."""
    context, settings = run.context, run.settings
    assert isinstance(settings, dpg_settings.FdpgSettings)
    training.seed_global_generators(context.stream_seed("dpg", "initialization"))
    model = context.load_model_copy().model
    model.eval()
    state = _State(
        run, settings, samplers.Models.of(context, model), training.new_optimizer(model, settings)
    )
    _start(state)
    while not budget_rule.fdpg_training_ends(state.spend.budget, settings):
        at_least = budget_rule.fdpg_next_checkpoint_at_least(
            state.spend.budget, state.updates, settings
        )
        if training_run.stop_reached(
            run.request, updates=state.updates, next_checkpoint_lower_bound=at_least
        ):
            if state.checkpoints[-1].update != state.updates:
                _save_resume(state)
            training.stop(run, state.checkpoints, state.updates, at_least)
            return state.checkpoints, state.models
        _update(state)
        if budget_rule.fdpg_checkpoint_due(state.updates):
            _save_checkpoint(state)
            _save_resume(state)
    # The policy as it is closes the training budget.
    if state.checkpoints[-1].update != state.updates:
        _save_checkpoint(state, forced=True)
    return training.complete(run, state.checkpoints, state.updates), state.models
