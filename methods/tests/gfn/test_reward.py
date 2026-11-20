"""The reward's terms of a response."""

from __future__ import annotations

import math
from typing import Any

import pytest
from posteriorbench import original_model, rules

from posteriorbench_methods.common import hf_runtime
from posteriorbench_methods.gfn import reward


def test_scored_responses_carry_p_of_every_prefix(
    tiny_original_model: original_model.OriginalModel, tiny_task: Any
) -> None:
    prompt = tuple(tiny_task.prompt_token_ids)
    stops = tiny_original_model.stop_token_ids
    responses = tiny_original_model.generate(list(prompt), 30, seed=5)
    is_hit = rules.parse_rule(tiny_task.rule)
    p = hf_runtime.CausalLM(tiny_original_model.model, prompt)
    scored = reward.score(p, stops, responses, decode=tiny_original_model.decode, is_hit=is_hit)
    expected = tiny_original_model.token_logprobs(list(prompt), responses)
    assert {r.hit for r in scored} == {True, False}
    for response, item, values in zip(responses, scored, expected, strict=True):
        assert item.token_ids == list(response)
        assert item.text == tiny_original_model.decode(response)
        assert item.token_logp == pytest.approx(values, abs=1e-5)
        assert item.logp == math.fsum(item.token_logp)
        assert len(item.stop_logp) == len(response) and len(item.stop_logp[0]) == len(stops)
        content = len(response) - int(response[-1] in stops)
        assert len(item.prefix_hits) == content + 1
        assert item.prefix_hits[0] is False  # the empty text has no behavior
        assert item.prefix_hits[-1] == item.hit == is_hit(item.text)
        expected_reward = item.logp + (0.0 if item.hit else math.log(1e-8))
        assert item.log_reward(1e-8) == expected_reward
    assert reward.log_p(p, stops, responses) == [item.logp for item in scored]
