"""The sampler interface: the per-response cost cap, complete responses, finite-support samplers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from posteriorbench import sampler

STOP = 2


def test_per_response_cost_has_no_lower_bound() -> None:
    """The paper caps a sampler at two inferences per response and sets no minimum; a sampler
    that returns stored responses (replay, the fallback) runs no model per response."""
    for allowed in (0.0, 1.0, 2.0):
        sampler.validate_inferences_per_response(allowed)
    for refused in (-0.5, 2.5, math.nan, math.inf):
        with pytest.raises(sampler.SamplerError):
            sampler.validate_inferences_per_response(refused)
    with pytest.raises(sampler.SamplerError):
        sampler.FiniteSampler([([1, STOP], 1.0)], inferences_per_response=3.0)
    assert sampler.fallback_sampler([1, STOP]).inferences_per_response == 0.0


def test_default_sampler_costs_one_inference_per_response() -> None:
    class Minimal(sampler.Sampler):
        def sample(self, n: int, *, seed: int) -> list[list[int]]:
            return [[1, STOP]] * n

        def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
            return [0.0 for _ in responses]

    minimal = Minimal()
    assert minimal.inferences_per_response == 1.0 and minimal.finite_support() is None
    assert minimal.logprob([1, STOP]) == 0.0


def test_finite_sampler() -> None:
    bank = [[1, STOP], [3, STOP], [1, STOP], [4, STOP]]
    replay = sampler.FiniteSampler([(r, 0.25) for r in bank], inferences_per_response=0.0)
    support = dict(replay.finite_support() or [])
    assert support == {(1, STOP): 0.5, (3, STOP): 0.25, (4, STOP): 0.25}  # repeats merged
    assert replay.logprobs([[1, STOP], [9, STOP]]) == [math.log(0.5), -math.inf]
    assert replay.sample(50, seed=3) == replay.sample(50, seed=3)
    assert replay.sample(50, seed=3) != replay.sample(50, seed=4)
    with pytest.raises(sampler.SamplerError):
        sampler.FiniteSampler([], inferences_per_response=0.0)
    with pytest.raises(sampler.SamplerError, match="sum to"):
        sampler.FiniteSampler([([1, STOP], 0.5)], inferences_per_response=0.0)
    with pytest.raises(sampler.SamplerError):
        sampler.FiniteSampler([([1, STOP], -1.0), ([2, STOP], 2.0)], inferences_per_response=0.0)


def test_complete_responses() -> None:
    def reason(tokens: list[int]) -> str | None:
        return sampler.invalid_response_reason(tokens, stop_token_ids={STOP}, max_response_tokens=4)

    for valid in ([STOP], [5, STOP], [5, 6, 7, 8], [5, 6, 7, STOP]):
        assert reason(valid) is None
    assert reason([]) is not None  # empty
    assert reason([5, 6, 7, 8, 9]) is not None  # beyond the cap
    assert reason([STOP, 5]) is not None  # a stop token inside
    assert "prefix" in (reason([5, 6]) or "")  # short and unterminated: a prefix
