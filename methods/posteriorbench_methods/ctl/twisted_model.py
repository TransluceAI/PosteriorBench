"""The twisted model: the frozen original model with a twist head, its draws and their log q.

The twisted model q draws the token a after the prefix y_<t with probability

    q(a | y_<t) = p(a | y_<t) psi_t(y_<t a) / Z_t(y_<t),

the softmax of the original model's logits plus the output of the twist network
(`twist.TwistHead`). `sample` keeps the hidden state that predicted each token, which is all that
an update reads of the model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from posteriorbench import rules
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import hf_runtime, run_context
from posteriorbench_methods.ctl import twist

GENERATION_BATCH = 64  # responses generated in one forward pass, from one random generator
TEACHER_FORCING_BATCH = 8  # responses in one teacher-forced pass


@dataclasses.dataclass(frozen=True, eq=False)
class TwistedModel:
    """A copy of the original model whose output layer is a `twist.TwistHead`, under one prompt.

    The weights of the twist network are the sampler: loading a checkpoint into the head changes
    what every function of this module draws and scores. `model.model` is the transformers model
    without its output layer, and `model.lm_head` the twist head."""

    model: Any
    prompt_ids: tuple[int, ...]
    response_format: hf_runtime.ResponseFormat

    @property
    def head(self) -> twist.TwistHead:
        return self.model.lm_head

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def attach_twist_head(model: Any, seed: int) -> None:
    """Freeze every weight of `model` and replace its output layer by a new `twist.TwistHead`
    around it, whose network is initialized from `seed` (changes the model, and leaves it in
    eval mode)."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.lm_head = twist.TwistHead(model.lm_head, seed).to(next(model.parameters()).device)
    model.eval()


def build(context: run_context.RunContext) -> TwistedModel:
    """The twisted model of a run before training: a new copy of the original model with a new
    twist head, under the task's prompt."""
    model = context.load_model_copy().model
    attach_twist_head(model, context.stream_seed("ctl", "initialization"))
    return TwistedModel(
        model=model,
        prompt_ids=tuple(context.task.prompt_token_ids),
        response_format=hf_runtime.ResponseFormat.of(context.original),
    )


def twist_weights(twisted: TwistedModel) -> dict[str, torch.Tensor]:
    """Copies, on the CPU, of the tensors of the twist network: what a checkpoint holds. The
    output layer of the original model, which the head also holds, is not among them."""
    return {
        name: value.detach().cpu().clone()
        for name, value in twisted.head.state_dict().items()
        if not name.startswith("inner.")
    }


def load_twist_weights(twisted: TwistedModel, tensors: Mapping[str, torch.Tensor]) -> None:
    """Put the tensors of `twist_weights` into the twist network (changes the twisted model).
    Raises ValueError if they are not the tensors of the network."""
    missing, unexpected = twisted.head.load_state_dict(dict(tensors), strict=False)
    if unexpected or any(not name.startswith("inner.") for name in missing):
        raise ValueError(
            "these are not the tensors of the twist network: "
            f"missing {list(missing)}, unexpected {list(unexpected)}"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class Draws:
    """Responses drawn from the twisted model, with what an update reads of them. N responses;
    T is the response limit and H the hidden size.

    tokens: [N, T] long. tokens[i, t] is token t of response i, and 0 after its end.
    hidden: [N, T, H] float32. hidden[i, t] is the model's last hidden state after y_{i,<t},
        which predicted token t; 0 after the end.
    prefix_logq: [N, T] float32. prefix_logq[i, t] = log q(y_{i,<=t}) as generated; after the
        end of the response it stays at log q(y_i).
    lengths: the number of tokens of each response, its stop token included."""

    tokens: torch.Tensor
    hidden: torch.Tensor
    prefix_logq: torch.Tensor
    lengths: list[int]

    def token_ids(self) -> list[list[int]]:
        """The responses as lists of token ids."""
        return [self.tokens[row, :length].tolist() for row, length in enumerate(self.lengths)]

    def token_logq(self) -> list[list[float]]:
        """log q(y_t | y_<t) of every token of each response: the steps of `prefix_logq`."""
        steps = torch.cat(
            [self.prefix_logq[:, :1], self.prefix_logq[:, 1:] - self.prefix_logq[:, :-1]], dim=1
        )
        return [steps[row, :length].double().tolist() for row, length in enumerate(self.lengths)]


def _sample_batch(twisted: TwistedModel, rows: int, seed: int) -> Draws:
    """One batch of `sample`: `rows` responses drawn from a torch generator with seed `seed`."""
    head, trunk, device = twisted.head, twisted.model.model, twisted.device
    limit = twisted.response_format.max_tokens
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt = torch.tensor([list(twisted.prompt_ids)] * rows, dtype=torch.long, device=device)
    tokens = torch.zeros((rows, limit), dtype=torch.long, device=device)
    hidden = torch.zeros((rows, limit, head.fc1.in_features), device=device)
    # Of the sampled token at each position: log p, log psi, and log Z of its prefix.
    token_logp = torch.zeros((rows, limit), device=device)
    token_logpsi = torch.zeros((rows, limit), device=device)
    token_logz = torch.zeros((rows, limit), device=device)
    lengths = torch.zeros(rows, dtype=torch.long, device=device)
    alive = torch.ones(rows, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = trunk(input_ids=prompt, use_cache=True)
        for position in range(limit):
            state = out.last_hidden_state[:, -1].float()
            original_logits = head.inner(state).float()
            log_psi = head.log_psi(state)
            # log Z as a difference of two log-sum-exps over the same logits. It is exactly 0
            # where the twist is 0. logsumexp(log_softmax(logits) + log psi) is the same number
            # only up to the rounding of the softmax, about 2e-5 per token over 248,320 logits,
            # which would enter log q of every prefix and through it the weights of an update.
            log_normalizer = torch.logsumexp(original_logits, dim=-1)
            log_p = original_logits - log_normalizer.unsqueeze(1)
            log_z = torch.logsumexp(original_logits + log_psi, dim=-1) - log_normalizer
            log_q = log_p + log_psi - log_z.unsqueeze(1)
            chosen = torch.multinomial(log_q.exp(), 1, generator=generator).squeeze(1)
            # A response that has ended keeps running in the batch; nothing of it is recorded.
            recorded = alive.float()
            hidden[:, position] = state * recorded.unsqueeze(1)
            tokens[:, position] = chosen * alive.long()
            token_logp[:, position] = log_p.gather(1, chosen.unsqueeze(1)).squeeze(1) * recorded
            token_logpsi[:, position] = log_psi.gather(1, chosen.unsqueeze(1)).squeeze(1) * recorded
            token_logz[:, position] = log_z * recorded
            lengths += alive.long()
            for stop_id in twisted.response_format.stop_token_ids:
                alive = alive & (chosen != stop_id)
            if not bool(alive.any()) or position == limit - 1:
                break
            out = trunk(
                input_ids=chosen.unsqueeze(1), past_key_values=out.past_key_values, use_cache=True
            )
    return Draws(
        tokens=tokens,
        hidden=hidden,
        prefix_logq=token_logp.cumsum(1) + token_logpsi.cumsum(1) - token_logz.cumsum(1),
        lengths=lengths.tolist(),
    )


def sample(twisted: TwistedModel, count: int, seed: int) -> Draws:
    """`count` responses of the twisted model, at temperature 1 over the whole output layer.

    The responses are generated in batches of GENERATION_BATCH, and the batch that starts at draw
    i draws from a torch generator with seed `seed + i`, with one `torch.multinomial` call per
    position. The model runs with its cache, and the hidden state that predicted each token is
    kept, so an update of the twist needs no second pass of the model."""
    batches = [
        _sample_batch(twisted, min(GENERATION_BATCH, count - first), seed + first)
        for first in range(0, count, GENERATION_BATCH)
    ]
    return Draws(
        tokens=torch.cat([batch.tokens for batch in batches]),
        hidden=torch.cat([batch.hidden for batch in batches]),
        prefix_logq=torch.cat([batch.prefix_logq for batch in batches]),
        lengths=[length for batch in batches for length in batch.lengths],
    )


def label(
    context: run_context.RunContext, responses: Sequence[Sequence[int]], *, seed: int
) -> list[dict[str, Any]]:
    """One row for each response, as in a draws file: draw_index (its place in `responses`),
    token_ids, text, hit (the task's rule on the text) and seed (of the call that drew the
    responses)."""
    is_hit = rules.parse_rule(context.task.rule)
    rows: list[dict[str, Any]] = []
    for index, token_ids in enumerate(responses):
        text = context.original.decode(token_ids)
        rows.append(
            {
                "draw_index": index,
                "token_ids": list(token_ids),
                "text": text,
                "hit": bool(is_hit(text)),
                "seed": seed,
            }
        )
    return rows


def _teacher_forced_states(
    twisted: TwistedModel, responses: Sequence[Sequence[int]]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each response, the hidden states [length, H] (float32) that predict its tokens, and
    its tokens [length]. The frozen model runs over batches of TEACHER_FORCING_BATCH responses,
    padded on the right with token 0 and attention 0."""
    prompt = list(twisted.prompt_ids)
    device = twisted.device
    states: list[tuple[torch.Tensor, torch.Tensor]] = []
    for first in range(0, len(responses), TEACHER_FORCING_BATCH):
        batch = [list(response) for response in responses[first : first + TEACHER_FORCING_BATCH]]
        width = len(prompt) + max(len(response) for response in batch)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for row, response in enumerate(batch):
            input_ids[row, : len(prompt) + len(response)] = torch.tensor(
                prompt + response, device=device
            )
            attention[row, : len(prompt) + len(response)] = 1
        out = twisted.model.model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        for row, response in enumerate(batch):
            first_state = len(prompt) - 1
            state = out.last_hidden_state[row, first_state : first_state + len(response)].float()
            states.append((state, torch.tensor(response, device=device)))
    return states


def _logprobs_of(logits: torch.Tensor, tokens: torch.Tensor) -> list[float]:
    """The log-softmax of logits [length, V] at tokens [length]."""
    values = logits.gather(1, tokens[:, None]).squeeze(1) - torch.logsumexp(logits, dim=1)
    return values.double().tolist()


def token_logprobs(twisted: TwistedModel, responses: Sequence[Sequence[int]]) -> list[list[float]]:
    """log q(y_t | prompt, y_<t) of every token of each response, by teacher forcing."""
    with torch.no_grad():
        return [
            _logprobs_of(twisted.head(state), tokens)
            for state, tokens in _teacher_forced_states(twisted, responses)
        ]


def token_logprobs_with_p(
    twisted: TwistedModel, responses: Sequence[Sequence[int]]
) -> tuple[list[list[float]], list[list[float]]]:
    """(log q, log p) of every token of each response, both from one teacher-forced pass of the
    frozen model: log p from its own output layer, log q from that layer plus the twist."""
    q_tokens: list[list[float]] = []
    p_tokens: list[list[float]] = []
    with torch.no_grad():
        for state, tokens in _teacher_forced_states(twisted, responses):
            original_logits = twisted.head.inner(state).float()
            q_tokens.append(_logprobs_of(original_logits + twisted.head.log_psi(state), tokens))
            p_tokens.append(_logprobs_of(original_logits, tokens))
    return q_tokens, p_tokens


class TwistedSampler(pb_sampler.Sampler):
    """The twisted model as a sampler that the benchmark evaluates. It holds the model, not a
    copy: it is the sampler of whichever checkpoint the twist head holds. A response costs one
    inference, a pass of the model; the twist network is not counted."""

    name = "ctl"

    def __init__(self, twisted: TwistedModel) -> None:
        self.twisted = twisted

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        return sample(self.twisted, n, seed).token_ids()

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        if not responses:
            return []
        return [math.fsum(values) for values in token_logprobs(self.twisted, responses)]
