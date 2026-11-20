"""The sampler interface a method submits for one task, and samplers with finite known support.

A sampler q generates responses and computes log q(y) for any response. A response is what
generation returns (`invalid_response_reason`): 1 to 50 generated token ids, cut after the first
stop token or at the 50-token limit, so a response shorter than the limit ends with a stop token.
A sampler declares the inferences it spends to generate one response, at most two; the limit
covers generation only, so a mixture of samplers costs one. What a method spent to build the
sampler is not a property of the sampler: it is the method's `ledger.Ledger`, passed to evaluation
next to the sampler.

Random streams. `Sampler.sample` receives an integer seed. An evaluation with seed s draws with
seed 2**31 + s (`evaluation_stream_seed`), so evaluation uses the seeds [2**31, 2**32). A method
that calls `Sampler.sample` while it builds its sampler passes a seed in [0, 2**31) (for example
`construction_stream_seed`), so the evaluation draws are always new. Seeds stay below 2**32
because torch's CPU generator keeps only the low 32 bits of a seed.
"""

from __future__ import annotations

import abc
import collections
import math
from collections.abc import Collection, Sequence

import numpy as np

from posteriorbench import errors

MAX_INFERENCES_PER_RESPONSE = 2.0
EVALUATION_SEED_OFFSET = 2**31


class SamplerError(errors.PosteriorBenchError):
    """A sampler breaks the benchmark's rules (inference limit, malformed responses or
    probabilities)."""


def invalid_response_reason(
    token_ids: Sequence[int], *, stop_token_ids: Collection[int], max_response_tokens: int
) -> str | None:
    """Why `token_ids` is not a complete response of the stop-or-limit generation process, or None.

    Generation stops after the first stop token or after `max_response_tokens` tokens. A complete
    response therefore has 1 to `max_response_tokens` tokens, has a stop token only as its last
    token, and ends with a stop token if it is shorter than the limit; a response that reaches the
    limit may end without one. Its likelihood is that of this whole outcome, including the final
    stop token. A shorter sequence without a stop token is a prefix, whose likelihood is not the
    probability of any response, so it is invalid; it is never completed by appending a stop
    token."""
    length = len(token_ids)
    if not 1 <= length <= max_response_tokens:
        return f"a response has 1..{max_response_tokens} tokens, got {length}"
    if any(int(t) in stop_token_ids for t in token_ids[:-1]):
        return "a stop token can only be the last token of a response"
    if length < max_response_tokens and int(token_ids[-1]) not in stop_token_ids:
        return (
            f"a response shorter than {max_response_tokens} tokens must end with a stop token "
            f"{sorted(stop_token_ids)} (generation stopped); this one has {length} tokens and ends "
            f"with {int(token_ids[-1])}, so it is a prefix of a response"
        )
    return None


def evaluation_stream_seed(seed: int) -> int:
    """The `Sampler.sample` seed of the evaluation draws with evaluation seed `seed`: 2**31 + seed.
    Raises ValueError unless 0 <= seed < 2**31."""
    if not 0 <= seed < EVALUATION_SEED_OFFSET:
        raise ValueError(f"evaluation seeds lie in [0, 2**31), got {seed}")
    return EVALUATION_SEED_OFFSET + seed


def construction_stream_seed(seed: int, index: int) -> int:
    """A `Sampler.sample` seed for construction: seed + index, which must lie in [0, 2**31), below
    every evaluation seed. Raises ValueError outside that range."""
    if seed < 0 or index < 0 or seed + index >= EVALUATION_SEED_OFFSET:
        raise ValueError(
            f"construction streams need seed >= 0, index >= 0 and seed + index < 2**31, "
            f"got seed {seed} and index {index}"
        )
    return seed + index


def validate_inferences_per_response(value: float) -> None:
    """Raise SamplerError unless 0 <= value <= 2. There is no lower bound above zero: a sampler
    that returns stored responses runs no model per response."""
    if not (math.isfinite(value) and 0 <= value <= MAX_INFERENCES_PER_RESPONSE):
        raise SamplerError(
            f"inferences_per_response must lie in [0, {MAX_INFERENCES_PER_RESPONSE}], got {value}"
        )


def merge_support(
    support: Sequence[tuple[Sequence[int], float]],
) -> list[tuple[tuple[int, ...], float]]:
    """A finite support as distinct (response, probability) pairs, in first-seen order. Repeated
    responses are merged (their probabilities add up). Raises SamplerError unless the support is a
    probability distribution: non-empty, with positive finite probabilities that sum to 1 (within
    1e-9)."""
    if not support:
        raise SamplerError("empty support")
    masses: dict[tuple[int, ...], float] = collections.defaultdict(float)
    for response, probability in support:
        if not (math.isfinite(probability) and probability > 0):
            raise SamplerError("support probabilities must be positive and finite")
        masses[tuple(int(t) for t in response)] += probability
    total = math.fsum(masses.values())
    if abs(total - 1) > 1e-9:
        raise SamplerError(f"support probabilities sum to {total}, not 1")
    return list(masses.items())


class Sampler(abc.ABC):
    """A submitted sampler q for one task."""

    name: str = "sampler"

    @property
    def inferences_per_response(self) -> float:
        """Inferences this sampler spends to generate one response: 1 for a plain sampler (the
        default), 2 for a composed one, 0 for one that returns stored responses. A mixture of
        samplers costs 1."""
        return 1.0

    @abc.abstractmethod
    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        """n independent complete responses (generated token ids): the same responses for the same
        seed on one machine and configuration, and different ones for different seeds below 2**32.
        """

    @abc.abstractmethod
    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        """log q(y) of each response; -inf where q cannot produce it. Evaluation calls it with
        all references and with up to 128 hits at once, so batch it."""

    def logprob(self, token_ids: Sequence[int]) -> float:
        """log q(y) of one response (calls `logprobs`)."""
        (value,) = self.logprobs([token_ids])
        return value

    def finite_support(self) -> list[tuple[tuple[int, ...], float]] | None:
        """For a sampler whose distribution is a known finite set of responses, its (response,
        probability) pairs; None for every other sampler (the default).

        Evaluation uses the support only if every response in it is a hit, and then scores the
        sampler exactly over that set (`scoring.score_finite_support`). A support that contains a
        miss is ignored and the sampler is evaluated from draws like any other, so declaring a
        support never changes the threshold test."""
        return None


class FiniteSampler(Sampler):
    """A sampler with finite, known support: each response with a fixed probability. Repeated
    responses are merged (`merge_support`).

    `inferences_per_response` is required: 0 for a sampler that returns stored responses without
    running a model, otherwise what generating one response costs."""

    def __init__(
        self,
        support: Sequence[tuple[Sequence[int], float]],
        *,
        inferences_per_response: float,
        name: str = "finite",
    ) -> None:
        merged = merge_support(support)
        validate_inferences_per_response(inferences_per_response)
        total = math.fsum(mass for _, mass in merged)
        self._responses = [response for response, _ in merged]
        self._probabilities = np.array([mass for _, mass in merged]) / total
        self._log_probability = {
            r: math.log(float(p)) for r, p in zip(self._responses, self._probabilities, strict=True)
        }
        self._inferences_per_response = float(inferences_per_response)
        self.name = name

    @property
    def inferences_per_response(self) -> float:
        return self._inferences_per_response

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self._responses), size=n, replace=True, p=self._probabilities)
        return [list(self._responses[int(i)]) for i in picks]

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        return [self._log_probability.get(tuple(int(t) for t in r), -math.inf) for r in responses]

    def finite_support(self) -> list[tuple[tuple[int, ...], float]] | None:
        return [(r, float(p)) for r, p in zip(self._responses, self._probabilities, strict=True)]


def fallback_sampler(fallback_token_ids: Sequence[int]) -> FiniteSampler:
    """The fallback as a sampler: it always returns the fixed fallback response. It runs no model,
    so it costs no inferences per response and nothing to build."""
    return FiniteSampler([(fallback_token_ids, 1.0)], inferences_per_response=0.0, name="fallback")
