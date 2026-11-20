"""The reward of GFlowNet on a task, and what subtrajectory balance reads of it.

The reward of a response y is R(y) = p(y) if y is a hit and epsilon p(y) if it is a miss, where p
is the original model. A policy that generates each response with probability proportional to R
hits almost always, and its hits follow the posterior.

Subtrajectory balance needs the reward of the response that ends after each prefix of y
(`subtb`). A `Response` therefore carries, for every prefix, the log-probability under p of each
token and of each stop token, and whether the text of the prefix is a hit.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Sequence

import torch

from posteriorbench_methods.common import hf_runtime
from posteriorbench_methods.gfn import teacher_forcing

BATCH_SIZE = 20  # responses in one forward pass of the original model


@dataclasses.dataclass(frozen=True)
class Response:
    """A response with the terms of its reward. L is its number of tokens, N its number of
    content tokens (L without a final stop token) and K the number of stop tokens.

    token_logp: [L] the log-probability under p of each token after the tokens before it.
    stop_logp: [L][K] the log-probability under p of each stop token at the same positions.
    prefix_hits: [N + 1] whether the text of the first i content tokens is a hit, for i = 0..N.
    logp: log p(y), the sum of `token_logp`."""

    token_ids: list[int]
    text: str
    hit: bool
    logp: float
    token_logp: list[float]
    stop_logp: list[list[float]]
    prefix_hits: list[bool]

    def log_reward(self, epsilon: float) -> float:
        """log R(y): log p(y) for a hit, log p(y) + log epsilon for a miss."""
        return self.logp + (0.0 if self.hit else math.log(epsilon))


def _scores(
    p: hf_runtime.CausalLM, stop_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
) -> list[tuple[list[float], list[list[float]]]]:
    """(token log p [L], stop log p [L][K]) of each response, in batches of BATCH_SIZE."""
    scores: list[tuple[list[float], list[list[float]]]] = []
    with torch.inference_mode():
        for lo in range(0, len(responses), BATCH_SIZE):
            batch = responses[lo : lo + BATCH_SIZE]
            for token, stop in teacher_forcing.token_scores(p, stop_token_ids, batch):
                scores.append((token.double().tolist(), stop.double().tolist()))
    return scores


def log_p(
    p: hf_runtime.CausalLM, stop_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
) -> list[float]:
    """log p(y) of each response."""
    return [math.fsum(token) for token, _ in _scores(p, stop_token_ids, responses)]


def score(
    p: hf_runtime.CausalLM,
    stop_token_ids: Sequence[int],
    responses: Sequence[Sequence[int]],
    *,
    decode: Callable[[Sequence[int]], str],
    is_hit: Callable[[str], bool],
) -> list[Response]:
    """Each response with the terms of its reward. `decode` gives the text of token ids as the
    benchmark labels it, and `is_hit` is the task's rule on a text. Raises FloatingPointError if
    p gives a token a log-probability that is not finite."""
    scored: list[Response] = []
    for response, (token_logp, stop_logp) in zip(
        responses, _scores(p, stop_token_ids, responses), strict=True
    ):
        token_ids = [int(token) for token in response]
        if not all(map(math.isfinite, itertools.chain(token_logp, *stop_logp))):
            raise FloatingPointError("the original model gave a log-probability that is not finite")
        content = token_ids[:-1] if token_ids[-1] in stop_token_ids else token_ids
        text = decode(token_ids)
        scored.append(
            Response(
                token_ids=token_ids,
                text=text,
                hit=is_hit(text),
                logp=math.fsum(token_logp),
                token_logp=token_logp,
                stop_logp=stop_logp,
                prefix_hits=[is_hit(decode(content[:i])) for i in range(len(content) + 1)],
            )
        )
    return scored
