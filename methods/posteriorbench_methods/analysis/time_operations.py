"""Timing, on your GPU, the operations that `gpu_times` prices a run from.

Each operation is timed with the functions, batch sizes and float32 weights with which its method
runs it here. Every timed response has the full response limit: generation has no stop token, and
the other operations read random tokens, since a pass's time does not depend on which tokens it
reads. SFT's updates are timed as SFT makes them, on a fresh copy of the model. The other gradient
steps use a plain sum as loss and a learning rate of zero, so that the copy they share keeps the
original weights. CTL's update is timed from tensors already on the device; moving log p there,
as `ctl.training` does first, is not part of the time.
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from posteriorbench_methods.analysis import flops, gpu_times
from posteriorbench_methods.common import hf_runtime
from posteriorbench_methods.ctl import settings as ctl_settings
from posteriorbench_methods.ctl import twist, twisted_model
from posteriorbench_methods.dpg import settings as dpg_settings
from posteriorbench_methods.dpg import training as dpg_training
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import teacher_forcing
from posteriorbench_methods.sft import settings as sft_settings
from posteriorbench_methods.sft import train as sft_train

NO_STOP_TOKEN = -1  # no token that a model draws: generation runs to the response limit
Rows = list[list[int]]


@dataclasses.dataclass
class _Timer:
    """Times calls and collects their measurements.

    tokens: the length of every response.
    allowed: the token ids that random responses are drawn from; no stop token among them."""

    prompt: tuple[int, ...]
    tokens: int
    allowed: torch.Tensor
    repetitions: int
    warmup: int
    generator: torch.Generator
    measurements: list[gpu_times.Measurement] = dataclasses.field(default_factory=lambda: [])

    @property
    def generation_format(self) -> hf_runtime.ResponseFormat:
        """The response format of timed generation: no response ends before the limit."""
        return hf_runtime.ResponseFormat((NO_STOP_TOKEN,), self.tokens)

    def responses(self, count: int) -> Rows:
        """`count` random responses of the full length."""
        index = torch.randint(len(self.allowed), (count, self.tokens), generator=self.generator)
        return self.allowed[index].tolist()

    def time(
        self,
        operation: gpu_times.Operation,
        trains: flops.Trains,
        batch: int,
        call: Callable[[], Any],
    ) -> None:
        """Keep the median time of `repetitions` calls after the warm-up as the measurement of
        (operation, trains, batch)."""
        times: list[float] = []
        for repetition in range(self.warmup + self.repetitions):
            _wait_for_gpu()
            start = time.perf_counter()
            call()
            _wait_for_gpu()
            if repetition >= self.warmup:
                times.append(time.perf_counter() - start)
        self.measurements.append(
            gpu_times.Measurement(operation, trains, batch, statistics.median(times), len(times))
        )


def _wait_for_gpu() -> None:
    """Wait for the GPU, so that a time is that of the work and not of its launch."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _batches(operation: gpu_times.Operation, trains: flops.Trains) -> list[int]:
    """The batch sizes at which `operation` by `trains` is to be timed."""
    return [key[2] for key in gpu_times.every_key() if key[:2] == (operation, trains)]


def time_original_model(timer: _Timer, model: Any) -> None:
    """Generation and the teacher-forced forward pass of the original model."""
    lm = hf_runtime.CausalLM(model, timer.prompt)
    generation = hf_runtime.Generation(timer.generation_format)
    model.eval()
    for batch in _batches("generation", "nothing"):
        group = [hf_runtime.DrawGroup(batch, seed=batch)]
        timer.time(
            "generation",
            "nothing",
            batch,
            lambda group=group: hf_runtime.sample_groups(
                hf_runtime.TwoModels(lm), group, generation
            ),
        )
    for batch in _batches("forward", "nothing"):
        rows = timer.responses(batch)
        timer.time(
            "forward",
            "nothing",
            batch,
            lambda rows=rows, batch=batch: hf_runtime.sequence_logprobs(lm, rows, batch_size=batch),
        )


def time_all_weights(timer: _Timer, model: Any) -> None:
    """The gradient step of KL-DPG, JS-DPG and GDC on one batch: the forward pass with its
    graph, the same with the backward pass, and a step of Adam. Leaves every weight of `model`
    with a gradient."""
    lm = hf_runtime.CausalLM(model, timer.prompt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = dpg_training.new_optimizer(model, dpg_settings.FdpgSettings())  # learning rate 0
    (batch,) = _batches("training_forward", "all_weights")
    rows = timer.responses(batch)

    def loss() -> torch.Tensor:
        return hf_runtime.sequence_logprobs_train(lm, rows).sum() / batch

    timer.time("training_forward", "all_weights", batch, loss)
    timer.time("training_forward_backward", "all_weights", batch, lambda: loss().backward())
    timer.time("optimizer_step", "all_weights", 0, optimizer.step)
    optimizer.zero_grad(set_to_none=True)


def sft_update(
    lm: hf_runtime.CausalLM,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    rows: Rows,
    gradient_clip: float,
) -> float:
    """One update of SFT on `rows`, which pass through the model in one forward pass as the
    benchmark's batches do. Returns the loss. It must stay step for step equal to the update in
    `sft.train`."""
    optimizer.zero_grad(set_to_none=True)
    loss = -hf_runtime.sequence_logprobs_train(lm, rows).sum() / len(rows)
    if not torch.isfinite(loss):
        raise FloatingPointError("the training loss is not finite")
    loss.backward()
    value = float(loss.detach())
    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()
    return value


def time_original_model_and_all_weights(timer: _Timer, model: Any) -> None:
    """The measurements that share one copy of the model; the gradient step on all weights has
    a learning rate of zero, so it can come after those of the original model."""
    time_original_model(timer, model)
    time_all_weights(timer, model)


def time_last_layers(timer: _Timer, model: Any) -> None:
    """A whole update of SFT at each of its batch sizes, set up as `sft.train` sets up a run.
    Changes the weights of `model`, which must be a copy that no other measurement uses."""
    settings = sft_settings.Settings()
    lm = hf_runtime.CausalLM(model, timer.prompt)
    sft_train.freeze_all_but_last_layers(model, settings.trainable_last_layers)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    model.train()
    for batch in _batches("update", "last_layers"):
        rows = timer.responses(batch)
        timer.time(
            "update",
            "last_layers",
            batch,
            lambda rows=rows: sft_update(lm, optimizer, parameters, rows, settings.gradient_clip),
        )
    model.eval()


def time_adapters(timer: _Timer, model: Any) -> None:
    """GFlowNet's policy: generation, the forward pass of its loss over one batch, the same with
    the backward pass, and a step of AdamW. Puts adapters on `model`."""
    settings = gfn_settings.Settings()
    gfn_policy.attach_adapters(model, rank=settings.lora_rank, alpha=settings.lora_alpha)
    gfn_policy.disable_dropout(model)
    lm = hf_runtime.CausalLM(model, timer.prompt)
    generation = hf_runtime.Generation(
        timer.generation_format, batch_size=settings.generation_batch_size, renormalize=False
    )
    policy = gfn_policy.Policy(model, lm, generation, settings.forward_batch_size)
    for batch in _batches("generation", "adapters"):
        group = [hf_runtime.DrawGroup(batch, seed=batch)]
        timer.time("generation", "adapters", batch, lambda group=group: policy.draw(group))
    (batch,) = _batches("training_forward", "adapters")
    rows = timer.responses(batch)
    # Two columns of stop-token log-probabilities, as the loss reads for the model's two stop
    # tokens; which tokens they are does not change the time.
    stop_columns = [int(token) for token in timer.allowed[:2]]
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=0.0
    )
    model.train()

    def loss() -> torch.Tensor:
        scores = teacher_forcing.token_scores(lm, stop_columns, rows)
        return torch.stack([token.sum() + stop.sum() for token, stop in scores]).sum()

    timer.time("training_forward", "adapters", batch, loss)
    timer.time("training_forward_backward", "adapters", batch, lambda: loss().backward())
    timer.time("optimizer_step", "adapters", 0, optimizer.step)
    optimizer.zero_grad(set_to_none=True)
    model.eval()


def twist_loss(
    head: twist.TwistHead,
    draws: twisted_model.Draws,
    lengths: torch.Tensor,
    token_logp: torch.Tensor,
    hits: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """The loss of an update of CTL on `draws`, N responses of the twisted model, from tensors
    on the device. It must stay equal to `ctl.training.update_loss`.

    lengths: [N] the number of tokens of each response.
    token_logp: [N, T] float32, log p(y_t | y_<t) of every token, 0 after the end of a response.
    hits: [N] bool."""
    prefix_logp = token_logp.cumsum(1)
    log_psi = twist.absorb(head.log_psi_of(draws.hidden, draws.tokens), lengths)
    negative = twist.negative_weights(prefix_logp, draws.prefix_logq, log_psi)
    positive = twist.positive_weights(prefix_logp[:, -1], draws.prefix_logq[:, -1], hits, epsilon)
    return twist.contrastive_loss(log_psi, negative, positive)


def time_twist(timer: _Timer, model: Any) -> None:
    """CTL's twisted model: generation, and a whole update of the twist network on random
    hidden states, log-probabilities and hits of one update's responses. The update includes the
    gradient norm, which an update of CTL checks. Puts a twist head on `model`."""
    settings = ctl_settings.Settings()
    twisted_model.attach_twist_head(model, seed=0)
    twisted = twisted_model.TwistedModel(model, timer.prompt, timer.generation_format)
    for batch in _batches("generation", "twist"):
        timer.time(
            "generation",
            "twist",
            batch,
            lambda batch=batch: twisted_model.sample(twisted, batch, seed=batch),
        )
    head, device = twisted.head, twisted.device
    optimizer = torch.optim.Adam(
        [value for value in head.parameters() if value.requires_grad], lr=0.0
    )
    for batch in _batches("update", "twist"):
        shape = (batch, timer.tokens)
        generator = timer.generator
        draws = twisted_model.Draws(
            tokens=torch.tensor(timer.responses(batch), device=device),
            hidden=torch.randn((*shape, head.fc1.in_features), generator=generator).to(device),
            prefix_logq=(-3 * torch.rand(shape, generator=generator)).cumsum(1).to(device),
            lengths=[timer.tokens] * batch,
        )
        lengths = torch.tensor(draws.lengths, device=device)
        token_logp = (-3 * torch.rand(shape, generator=generator)).to(device)
        hits = (torch.rand(batch, generator=generator) < 0.05).to(device)

        def update(
            draws: twisted_model.Draws = draws,
            lengths: torch.Tensor = lengths,
            token_logp: torch.Tensor = token_logp,
            hits: torch.Tensor = hits,
        ) -> None:
            optimizer.zero_grad(set_to_none=True)
            twist_loss(head, draws, lengths, token_logp, hits, settings.epsilon).backward()
            sum(
                float(value.grad.float().square().sum())
                for value in head.parameters()
                if value.grad is not None
            )
            optimizer.step()

        timer.time("update", "twist", batch, update)


def measure(
    load_model: Callable[[], Any],
    prompt_ids: Sequence[int],
    *,
    response_tokens: int,
    stop_token_ids: Sequence[int],
    repetitions: int = 6,
    warmup: int = 2,
    conditions: Mapping[str, Any] | None = None,
) -> gpu_times.Session:
    """Time every operation that `gpu_times` reads, on the device of the models.

    load_model: returns a new float32 copy of the original model on its device. It is called four
        times, since SFT's updates, the adapters and the twist head each change their own copy.
    response_tokens: the length of every timed response, the benchmark's response limit.
    stop_token_ids: left out of the random responses.
    conditions: what the session records about how it was made (GPU, versions, lengths)."""
    if repetitions < 1 or warmup < 0:
        raise ValueError("a measurement needs at least one repetition and no negative warm-up")
    model = load_model()
    stop_tokens = {int(token) for token in stop_token_ids}
    timer = _Timer(
        prompt=tuple(int(token) for token in prompt_ids),
        tokens=response_tokens,
        allowed=torch.tensor(
            [token for token in range(int(model.config.vocab_size)) if token not in stop_tokens]
        ),
        repetitions=repetitions,
        warmup=warmup,
        generator=torch.Generator().manual_seed(0),
    )
    time_last_layers(timer, model)
    for time_copy in (time_original_model_and_all_weights, time_adapters, time_twist):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_model()
        time_copy(timer, model)
    session = gpu_times.Session(dict(conditions or {}), tuple(timer.measurements))
    assert session.has(gpu_times.every_key()), "every operation that pricing reads is timed"
    return session
