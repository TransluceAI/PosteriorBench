"""log p of the twisted model's draws, and the rule for tokens that the tokenizer does not
define."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from posteriorbench import data, original_model

from posteriorbench_methods.ctl import p_scores, twisted_model


class TenTokenScorer:
    """A stand-in for the original model: a tokenizer of ten tokens, the stop token 2, and
    log p = -0.5 for every token. It records the responses it is asked to score."""

    tokenizer = range(10)
    stop_token_ids = (2,)

    def __init__(self) -> None:
        self.calls: list[list[list[int]]] = []

    def token_logprobs(
        self, prompt_token_ids: Sequence[int], responses: Sequence[Sequence[int]]
    ) -> list[list[float]]:
        self.calls.append([list(response) for response in responses])
        return [[-0.5] * len(response) for response in responses]


def test_log_p_is_the_original_models(
    original: original_model.OriginalModel,
    tiny_task: data.Task,
    twisted: twisted_model.TwistedModel,
) -> None:
    responses = twisted_model.sample(twisted, 70, 5).token_ids()
    values = p_scores.token_logprobs(original, tiny_task.prompt_token_ids, responses)
    expected = original.token_logprobs(tiny_task.prompt_token_ids, responses)
    assert [len(v) for v in values] == [len(r) for r in responses]
    for mine, theirs in zip(values, expected, strict=True):
        assert mine == pytest.approx(theirs, abs=1e-5)


def test_p_is_zero_from_the_first_undefined_token() -> None:
    scorer = TenTokenScorer()
    values = p_scores.token_logprobs(
        cast(original_model.OriginalModel, scorer), [1], [[4, 5, 2], [4, 12, 5, 2], [11, 4, 2]]
    )
    undefined = p_scores.UNDEFINED_TOKEN_LOGP
    assert values == [[-0.5, -0.5, -0.5], [-0.5, undefined, undefined, undefined], [undefined] * 3]
    # One call: the first response whole, the defined start of the second closed with a stop
    # token, and nothing of the third.
    assert scorer.calls == [[[4, 5, 2], [4, 2]]]


def test_responses_are_scored_in_batches_of_64() -> None:
    scorer = TenTokenScorer()
    p_scores.token_logprobs(cast(original_model.OriginalModel, scorer), [1], [[4, 2]] * 70)
    assert [len(call) for call in scorer.calls] == [64, 6]
