"""Generation, teacher forcing and composed sampling with Hugging Face causal language models.

A response is the list of token ids that a model generates after a prompt, at temperature 1 over
the whole vocabulary: at most `ResponseFormat.max_tokens` tokens, ending after the first stop token,
which is part of the response. Log-probabilities are natural logarithms, computed in float32 from
the model's logits.

A sampler reads the next-token probabilities of up to two models: q, the sampler that a method
builds, and p, the original model. Under a composition setting (`composition.Setting`: alpha, L)
token t of a response, counted from 0, is drawn from (1 - alpha) q + alpha p while t < L and from p
from token L on. Without a setting, or under the plain setting (alpha = 0, L >= max_tokens), the
sampler is q itself and p is never run. `ModelSampler` is such a sampler for the benchmark.

What determines the draws. Responses are drawn in groups (`DrawGroup`), each group from
`torch.Generator(device).manual_seed(group.seed)`, one `torch.multinomial` call per token
position. A group's responses therefore depend on its seed and on the values of `Generation`, and
through the rounding of a batched forward pass also on the other groups in its batch. With the
same weights, the same seeds, the same `Generation` and the same grouping, two runs on one machine
draw the same responses.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import torch
from posteriorbench import protocol
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import composition


class GenerationError(ValueError):
    """Arguments that generation or scoring cannot use: an empty prompt, an impossible setting, a
    group larger than the batch size."""


@dataclasses.dataclass(frozen=True)
class ResponseFormat:
    """How a response ends: after the first of `stop_token_ids`, which stays in the response, or
    after `max_tokens` tokens. The first stop id pads the rows of a batch that have ended."""

    stop_token_ids: tuple[int, ...]
    max_tokens: int

    def __post_init__(self) -> None:
        if not self.stop_token_ids:
            raise GenerationError("a response format needs at least one stop token")
        if self.max_tokens < 1:
            raise GenerationError(f"max_tokens must be positive, got {self.max_tokens}")

    @classmethod
    def of(cls, scorer: protocol.Scorer) -> ResponseFormat:
        """The format of the package's scorer: its stop tokens, in its order, and its limit."""
        return cls(tuple(int(t) for t in scorer.stop_token_ids), int(scorer.max_response_tokens))


def _check_setting(alpha: float, switch: int) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise GenerationError(f"alpha must lie in [0, 1], got {alpha}")
    if switch < 0:
        raise GenerationError(f"the switch position must be non-negative, got {switch}")


def _alpha_switch(setting: composition.Setting | None, max_tokens: int) -> tuple[float, int]:
    """(alpha, L) of a setting; no setting is q alone (alpha = 0, L = max_tokens)."""
    if setting is None:
        return 0.0, max_tokens
    _check_setting(setting.alpha, setting.switch)
    return float(setting.alpha), int(setting.switch)


ForwardContext = Callable[[int], contextlib.AbstractContextManager[Any]]
"""Called with the position, in the prompt and response together, of the first token of a forward
pass (0 for a pass that starts with the prompt). Returns the context in which the pass runs, a
steering hook for example."""


@dataclasses.dataclass(frozen=True, eq=False)
class CausalLM:
    """A Hugging Face causal language model under one prompt.

    `model` is called as model(input_ids=, attention_mask=, past_key_values=, use_cache=,
    return_dict=) and returns `.logits` [rows, positions, vocab] and `.past_key_values`. `context`,
    if given, wraps every forward pass of generation and of teacher forcing."""

    model: Any
    prompt_ids: tuple[int, ...]
    context: ForwardContext | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise GenerationError("empty prompt")

    def forward_context(self, start: int) -> contextlib.AbstractContextManager[Any]:
        """The context of a forward pass whose first token is at position `start`."""
        return contextlib.nullcontext() if self.context is None else self.context(start)


class LogitSource(Protocol):
    """The next-token logits of q and p while one batch is generated (a cache for each model)."""

    def logits(
        self, response: torch.Tensor, attention: torch.Tensor, *, q: bool, p: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """float32 next-token logits [rows, vocab] of q (if `q`) and of p (if `p`) after each
        model's prompt and `response` [rows, t], the tokens sampled so far, with the rows that
        have ended padded. `attention` [rows, t] is 1 for a token that was appended while its row
        was still generating, else 0. Called once per position, with growing t; each model is fed
        only the tokens it has not seen."""
        ...


class NextTokenModels(Protocol):
    """q and p as generation reads them. `start(rows)` opens new caches for one batch."""

    @property
    def device(self) -> torch.device: ...

    def start(self, rows: int) -> LogitSource: ...


class _CachedForward:
    """The cache of one model during a batch: the first call runs the prompt and the response so
    far; a later call feeds the response tokens appended since."""

    def __init__(self, lm: CausalLM, rows: int, device: torch.device) -> None:
        self.lm = lm
        self.prompt = torch.tensor([list(lm.prompt_ids)] * rows, dtype=torch.long, device=device)
        self.cache: Any = None
        self.fed = 0

    def logits(self, response: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        length = response.shape[1]
        prompt_attention = torch.ones_like(self.prompt)
        mask = torch.cat((prompt_attention, attention), dim=1) if length else prompt_attention
        if self.cache is None:
            ids = torch.cat((self.prompt, response), dim=1) if length else self.prompt
        else:
            assert length > self.fed, "a model was asked twice at the same position"
            ids = response[:, self.fed :]
        start = mask.shape[1] - ids.shape[1]
        with torch.inference_mode(), self.lm.forward_context(start):
            out = self.lm.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
            )
        if out.past_key_values is None:
            raise RuntimeError("the model returned no recurrent/KV cache")
        self.cache = out.past_key_values
        self.fed = length
        return out.logits[:, -1].float()


class _TwoModelsSource:
    def __init__(self, q: CausalLM, p: CausalLM | None, rows: int, device: torch.device) -> None:
        self.q = _CachedForward(q, rows, device)
        self.p = None if p is None else _CachedForward(p, rows, device)

    def logits(
        self, response: torch.Tensor, attention: torch.Tensor, *, q: bool, p: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        p_logits = None
        if p:
            if self.p is None:
                raise GenerationError("this setting reads p, but no model was given for p")
            p_logits = self.p.logits(response, attention)
        q_logits = self.q.logits(response, attention) if q else None
        if self.p is not None and self.p.cache is not None and self.p.cache is self.q.cache:
            raise RuntimeError("q and p must not share a recurrent/KV cache")
        return q_logits, p_logits


@dataclasses.dataclass(frozen=True, eq=False)
class TwoModels:
    """q and p as two models, each with a cache of its own. The same weights may serve both (the
    original model under two prompts, for example). p is None for a sampler that is never composed.
    At each position p runs before q."""

    q: CausalLM
    p: CausalLM | None = None

    @property
    def device(self) -> torch.device:
        return next(self.q.model.parameters()).device

    def start(self, rows: int) -> LogitSource:
        return _TwoModelsSource(self.q, self.p, rows, self.device)


TrunkHeads = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
"""Maps the float32 last hidden state [rows, hidden] to float32 (p logits, q logits) [rows, vocab]."""


class _SharedTrunkSource:
    def __init__(self, models: SharedTrunk, rows: int) -> None:
        self.models = models
        self.prompt = torch.tensor(
            [list(models.prompt_ids)] * rows, dtype=torch.long, device=models.device
        )
        self.cache: Any = None
        self.fed = 0

    def logits(
        self, response: torch.Tensor, attention: torch.Tensor, *, q: bool, p: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        length = response.shape[1]
        prompt_attention = torch.ones_like(self.prompt)
        mask = torch.cat((prompt_attention, attention), dim=1) if length else prompt_attention
        with torch.no_grad():
            if self.cache is None:
                ids = torch.cat((self.prompt, response), dim=1) if length else self.prompt
                out = self.models.trunk(input_ids=ids, attention_mask=mask, use_cache=True)
            else:
                assert length > self.fed, "the trunk was asked twice at the same position"
                out = self.models.trunk(
                    input_ids=response[:, self.fed :],
                    attention_mask=mask,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache = out.past_key_values
            self.fed = length
            hidden = out.last_hidden_state[:, -1].float()
            p_logits, q_logits = self.models.heads(hidden)
        return (q_logits if q else None), (p_logits if p else None)


@dataclasses.dataclass(frozen=True, eq=False)
class SharedTrunk:
    """q and p from one forward pass of a shared trunk (a model without its output layer that
    returns `.last_hidden_state` and `.past_key_values`) through `heads`: for a q that is the
    original model with another head on its hidden states."""

    trunk: Any
    prompt_ids: tuple[int, ...]
    heads: TrunkHeads

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise GenerationError("empty prompt")

    @property
    def device(self) -> torch.device:
        return next(self.trunk.parameters()).device

    def start(self, rows: int) -> LogitSource:
        return _SharedTrunkSource(self, rows)


@dataclasses.dataclass(frozen=True)
class Generation:
    """How responses are drawn. With the seeds, these values determine the draws exactly, so a
    method builds one `Generation` for each way it draws and passes it to every call.

    batch_size: the most rows in one forward pass (`pack`); None runs each group in its own pass.
    renormalize: True draws each token from the log-softmax of the logits, mixed with p's under a
        composition setting, and normalized once more, so that the recorded log-probability is
        that of the distribution torch draws from. False draws from softmax(logits / temperature)
        of q alone and records log-softmax(logits); only this allows a temperature other than 1,
        and it has no composed settings.
    p_from_start: when p runs, in a composed setting whose first tokens do not read it (alpha = 0
        before token L). True runs p from the first token, so its cache grows one token at a time.
        False starts p at token L with one pass over the prompt and the response so far. The
        distribution is the same; the rounding of the two differs."""

    response_format: ResponseFormat
    batch_size: int | None = None
    renormalize: bool = True
    p_from_start: bool = True


@dataclasses.dataclass(frozen=True)
class DrawGroup:
    """`count` responses drawn from one torch generator with seed `seed`. A `temperature` other
    than 1 needs `Generation.renormalize` to be False."""

    count: int
    seed: int
    temperature: float = 1.0


@dataclasses.dataclass(frozen=True)
class Draw:
    """One response. `token_logq[t]` is the log-probability of token t under the sampler at
    temperature 1 (the composed sampler, under a composition setting). `token_logr[t]` is its
    log-probability under the distribution it was drawn from, which is the sampler at the group's
    temperature; the two are equal at temperature 1. `finish_reason` is "stop" if the last token
    is a stop token, else "length"."""

    token_ids: list[int]
    token_logq: list[float]
    token_logr: list[float]
    finish_reason: str

    @property
    def logq(self) -> float:
        return math.fsum(self.token_logq)


PartSeed = Callable[[int, int], int]
"""(the seed of a `sample(n, seed)` call, the index of a part's first draw) -> the part's seed."""


def parts(n: int, seed: int, *, part_size: int, part_seed: PartSeed) -> list[DrawGroup]:
    """n draws as groups of `part_size` draws (the last may be smaller). The part whose first draw
    has index `lo` has the seed `part_seed(seed, lo)`."""
    if n < 0 or part_size < 1:
        raise GenerationError("n must be non-negative and part_size positive")
    return [DrawGroup(min(part_size, n - lo), part_seed(seed, lo)) for lo in range(0, n, part_size)]


def pack(groups: Sequence[DrawGroup], batch_size: int | None) -> list[list[DrawGroup]]:
    """The groups in batches of at most `batch_size` rows: each batch takes as many of the next
    groups as fit, and a group is never split. None puts every group in a batch of its own."""
    if batch_size is None:
        return [[group] for group in groups]
    batches: list[list[DrawGroup]] = []
    pending = list(groups)
    while pending:
        batch: list[DrawGroup] = []
        rows = 0
        while pending and rows + pending[0].count <= batch_size:
            group = pending.pop(0)
            batch.append(group)
            rows += group.count
        if not batch:
            raise GenerationError(
                f"a group of {pending[0].count} draws exceeds the batch size {batch_size}"
            )
        batches.append(batch)
    return batches


def _draw_batch(
    models: NextTokenModels,
    groups: Sequence[DrawGroup],
    *,
    fmt: ResponseFormat,
    alpha: float,
    switch: int,
    renormalize: bool,
    p_from_start: bool,
) -> list[Draw]:
    """One batch: the rows of the groups side by side, each group with its own generator."""
    device = models.device
    ranges: list[tuple[int, int]] = []
    rows = 0
    for group in groups:
        ranges.append((rows, rows + group.count))
        rows += group.count
    generators = [torch.Generator(device=device).manual_seed(group.seed) for group in groups]
    temperatures = None
    if not renormalize:
        temperatures = torch.cat(
            [
                torch.full((group.count,), float(group.temperature), device=device)
                for group in groups
            ]
        )
    plain = alpha == 0.0 and switch >= fmt.max_tokens
    stop_ids = list(fmt.stop_token_ids)
    source = models.start(rows)
    response = torch.zeros((rows, 0), dtype=torch.long, device=device)
    attention = torch.zeros((rows, 0), dtype=torch.long, device=device)
    live = torch.ones(rows, dtype=torch.bool, device=device)
    tokens: list[list[int]] = [[] for _ in range(rows)]
    token_q: list[list[float]] = [[] for _ in range(rows)]
    token_r: list[list[float]] = [[] for _ in range(rows)]
    for position in range(fmt.max_tokens):
        uses_q = position < switch and alpha < 1
        uses_p = alpha > 0 or position >= switch
        runs_p = uses_p or (p_from_start and not plain)
        q_logits, p_logits = source.logits(response, attention, q=uses_q, p=runs_p)
        if renormalize:
            p_lp = None if p_logits is None else torch.log_softmax(p_logits, dim=-1)
            if uses_q:
                assert q_logits is not None
                q_lp = torch.log_softmax(q_logits, dim=-1)
                if alpha == 0:
                    proposal = q_lp
                else:
                    assert p_lp is not None
                    proposal = torch.logaddexp(p_lp + math.log(alpha), q_lp + math.log1p(-alpha))
            else:
                assert p_lp is not None
                proposal = p_lp
            # Normalized here, so that the recorded log-probability is that of the distribution
            # torch.multinomial draws from.
            proposal = proposal - torch.logsumexp(proposal, dim=-1, keepdim=True)
            law_lp = sample_lp = proposal
        else:
            assert q_logits is not None and temperatures is not None
            law_lp = q_logits.log_softmax(-1)
            sample_lp = (q_logits / temperatures[:, None]).log_softmax(-1)
        chosen = torch.cat(
            [
                torch.multinomial(sample_lp[lo:hi].exp(), 1, generator=generator).squeeze(1)
                for (lo, hi), generator in zip(ranges, generators, strict=True)
            ]
        )
        q_values = law_lp.gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
        r_values = (
            q_values
            if sample_lp is law_lp
            else sample_lp.gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
        )
        token_values = chosen.cpu().tolist()
        for index, alive in enumerate(live.cpu().tolist()):
            if alive:
                tokens[index].append(int(token_values[index]))
                token_q[index].append(float(q_values[index]))
                token_r[index].append(float(r_values[index]))
        ended = torch.zeros_like(live)
        for stop_id in stop_ids:
            ended |= chosen == stop_id
        live &= ~ended
        if not live.any():
            break
        chosen = torch.where(live, chosen, stop_ids[0])
        response = torch.cat((response, chosen[:, None]), dim=1)
        attention = torch.cat((attention, live.long()[:, None]), dim=1)
    return [
        Draw(ids, qs, rs, "stop" if ids[-1] in stop_ids else "length")
        for ids, qs, rs in zip(tokens, token_q, token_r, strict=True)
    ]


def sample_groups(
    models: NextTokenModels,
    groups: Sequence[DrawGroup],
    generation: Generation,
    *,
    setting: composition.Setting | None = None,
) -> list[list[Draw]]:
    """The draws of each group, in the order of the groups, from q composed with p under `setting`
    (from q itself without one). Raises GenerationError for a setting or group that `models` or
    `generation` cannot draw (`Generation.renormalize`, `pack`)."""
    fmt = generation.response_format
    alpha, switch = _alpha_switch(setting, fmt.max_tokens)
    for group in groups:
        if group.count < 1 or group.temperature <= 0:
            raise GenerationError("a group needs a positive count and temperature")
        if generation.renormalize and group.temperature != 1.0:
            raise GenerationError("temperatures other than 1 need renormalize=False")
    if not generation.renormalize and not (alpha == 0.0 and switch >= fmt.max_tokens):
        raise GenerationError("renormalize=False has no composed settings")
    result: list[list[Draw]] = []
    for batch in pack(groups, generation.batch_size):
        draws = _draw_batch(
            models,
            batch,
            fmt=fmt,
            alpha=alpha,
            switch=switch,
            renormalize=generation.renormalize,
            p_from_start=generation.p_from_start,
        )
        start = 0
        for group in batch:
            result.append(draws[start : start + group.count])
            start += group.count
    return result


def _padded_inputs(
    prompt_ids: Sequence[int], responses: Sequence[Sequence[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Input ids and attention [rows, prompt + longest response], padded on the right with id 0
    and attention 0, and the length of each row."""
    inputs = [list(prompt_ids) + [int(t) for t in r] for r in responses]
    width = max(len(x) for x in inputs)
    input_ids = torch.zeros((len(inputs), width), dtype=torch.long)
    attention = torch.zeros((len(inputs), width), dtype=torch.long)
    for row, values in enumerate(inputs):
        input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        attention[row, : len(values)] = 1
    return input_ids.to(device), attention.to(device), [len(x) for x in inputs]


def response_token_logprobs(lm: CausalLM, responses: Sequence[Sequence[int]]) -> list[torch.Tensor]:
    """log q(y_t | prompt, y_<t) of every token of each response: one forward pass over the batch,
    padded on the right, a float32 log-softmax over every position, then the positions of each
    response. Returns one tensor [len(response)] for each response, with gradients if they are
    enabled. The model's train or eval mode is left as it is."""
    if not responses or any(len(r) == 0 for r in responses):
        raise GenerationError("cannot score an empty batch or an empty response")
    device = next(lm.model.parameters()).device
    input_ids, attention, lengths = _padded_inputs(lm.prompt_ids, responses, device)
    with lm.forward_context(0):
        logits = lm.model(input_ids=input_ids, attention_mask=attention).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    start = len(lm.prompt_ids)
    out: list[torch.Tensor] = []
    for row, end in enumerate(lengths):
        targets = input_ids[row, start:end]
        values = logprobs[row, start - 1 : end - 1, :]
        out.append(values.gather(1, targets.unsqueeze(1)).squeeze(1))
    return out


def sequence_logprobs_train(lm: CausalLM, responses: Sequence[Sequence[int]]) -> torch.Tensor:
    """log q(y) of each response as a float32 tensor [len(responses)]: the sum, in float32, of its
    per-token values. One forward pass, with gradients if they are enabled."""
    return torch.stack([values.sum() for values in response_token_logprobs(lm, responses)])


def sequence_logprobs(
    lm: CausalLM, responses: Sequence[Sequence[int]], *, batch_size: int
) -> list[float]:
    """log q(y) of each response, without gradients: `sequence_logprobs_train` on batches of
    `batch_size` responses, taken in the order given."""
    out: list[float] = []
    with torch.no_grad():
        for lo in range(0, len(responses), batch_size):
            values = sequence_logprobs_train(lm, responses[lo : lo + batch_size])
            out.extend(float(v) for v in values.cpu().tolist())
    return out


def token_logprobs(
    lm: CausalLM, responses: Sequence[Sequence[int]], *, batch_size: int
) -> list[list[float]]:
    """log q(y_t | prompt, y_<t) of every token of each response, without gradients, on batches of
    `batch_size` responses, taken in the order given."""
    out: list[list[float]] = []
    with torch.inference_mode():
        for lo in range(0, len(responses), batch_size):
            for values in response_token_logprobs(lm, responses[lo : lo + batch_size]):
                out.append(values.tolist())
    return out


def _token_array(tokens: Sequence[Sequence[float]], width: int) -> np.ndarray:
    array = np.zeros((len(tokens), width), dtype=np.float64)
    for row, values in enumerate(tokens):
        array[row, : len(values)] = values
    return array


def composed_token_logprobs(
    q_tokens: np.ndarray, p_tokens: np.ndarray, alpha: float, switch: int
) -> np.ndarray:
    """The per-token log-probabilities of responses under the setting (alpha, L), from their
    per-token log q and log p: float64 arrays [responses, positions] in, the same shape out. A
    position before L has log((1 - alpha) q + alpha p), a position from L on has log p."""
    if q_tokens.shape != p_tokens.shape or q_tokens.ndim != 2:
        raise GenerationError(
            "q and p token arrays must have the same [responses, positions] shape"
        )
    _check_setting(alpha, switch)
    if alpha == 1:
        return p_tokens.copy()
    if alpha == 0:
        mixed = q_tokens.copy()
    else:
        mixed = np.logaddexp(p_tokens + math.log(alpha), q_tokens + math.log1p(-alpha))
    mixed[:, switch:] = p_tokens[:, switch:]
    return mixed


def composed_logprobs(
    q_tokens: Sequence[Sequence[float]] | None,
    p_tokens: Sequence[Sequence[float]] | None,
    *,
    setting: composition.Setting,
) -> list[float]:
    """The log-probability of each response under the sampler composed with `setting`, from the
    per-token log q and log p of the responses: `composed_token_logprobs`, summed over the tokens
    of each response with math.fsum. `p_tokens` may be None for a setting that reads no p on
    these responses (the plain setting), and `q_tokens` for one that never reads q (alpha = 1 or
    L = 0)."""
    _check_setting(setting.alpha, setting.switch)
    alpha, switch = float(setting.alpha), int(setting.switch)
    given = q_tokens if q_tokens is not None else p_tokens
    if given is None:
        raise GenerationError("per-token log q or log p is required")
    if q_tokens is not None and p_tokens is not None:
        if [len(r) for r in q_tokens] != [len(r) for r in p_tokens]:
            raise GenerationError("per-token log q and log p differ in length")
    lengths = [len(r) for r in given]
    if not lengths:
        return []
    width = max(lengths)
    reads_q = alpha < 1 and switch > 0
    reads_p = alpha > 0 or switch < width
    if (reads_q and q_tokens is None) or (reads_p and p_tokens is None):
        raise GenerationError("this setting needs both per-token log q and log p")
    q_array = (
        _token_array(q_tokens, width) if q_tokens is not None else np.zeros((len(lengths), width))
    )
    p_array = (
        _token_array(p_tokens, width) if p_tokens is not None else np.zeros((len(lengths), width))
    )
    mixed = composed_token_logprobs(q_array, p_array, alpha, switch)
    return [math.fsum(mixed[row, :length].tolist()) for row, length in enumerate(lengths)]


LogprobsFn = Callable[[Sequence[Sequence[int]]], list[float]]


class ModelSampler(pb_sampler.Sampler):
    """q, or q composed with p under `setting`, as a sampler that the benchmark evaluates.

    `sample(n, seed)` draws the groups `parts(n, seed, part_size=, part_seed=)` with
    `sample_groups` as `generation` says. `logprobs` is `logprobs_fn`, which computes log q of
    given responses (for a composed sampler usually `composed_logprobs` of the per-token values of
    the two models).

    The sampler holds the models, not copies: it changes when other weights are loaded into
    them."""

    def __init__(
        self,
        models: NextTokenModels,
        logprobs_fn: LogprobsFn,
        generation: Generation,
        *,
        part_size: int,
        part_seed: PartSeed,
        setting: composition.Setting | None = None,
        name: str = "sampler",
    ) -> None:
        self.models = models
        self.logprobs_fn = logprobs_fn
        self.generation = generation
        self.part_size = part_size
        self.part_seed = part_seed
        self.setting = setting
        self.name = name
        _alpha_switch(setting, generation.response_format.max_tokens)

    @property
    def is_plain(self) -> bool:
        limit = self.generation.response_format.max_tokens
        return self.setting is None or self.setting.is_plain(limit)

    @property
    def inferences_per_response(self) -> float:
        if self.setting is None:
            return 1.0
        limit = self.generation.response_format.max_tokens
        return float(self.setting.inferences_per_response(limit))

    def sample_draws(self, n: int, *, seed: int) -> list[Draw]:
        """The draws of `sample`, with their per-token log q."""
        groups = parts(n, seed, part_size=self.part_size, part_seed=self.part_seed)
        out: list[Draw] = []
        for group_draws in sample_groups(
            self.models, groups, self.generation, setting=self.setting
        ):
            out.extend(group_draws)
        return out

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        return [draw.token_ids for draw in self.sample_draws(n, seed=seed)]

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        if not responses:
            return []
        return self.logprobs_fn(responses)
