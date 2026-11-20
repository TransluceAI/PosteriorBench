"""The twisted model on the tiny model: its draws, their log q, and its view as a sampler."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from posteriorbench import data, original_model
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.ctl import twisted_model

Twisted = twisted_model.TwistedModel


def test_attaching_a_twist_head_freezes_the_model(twisted: Twisted) -> None:
    trainable = [name for name, value in twisted.model.named_parameters() if value.requires_grad]
    assert sorted(trainable) == sorted(
        f"lm_head.{layer}.{part}" for layer in ("fc1", "fc2", "out") for part in ("weight", "bias")
    )
    assert not twisted.model.training


def test_draws_are_whole_responses_with_their_log_q(twisted: Twisted) -> None:
    draws = twisted_model.sample(twisted, 70, 123)
    limit = twisted.response_format.max_tokens
    assert draws.tokens.shape == (70, limit) and draws.hidden.shape == (70, limit, 32)
    responses = draws.token_ids()
    for response, length in zip(responses, draws.lengths, strict=True):
        assert len(response) == length
        assert (
            pb_sampler.invalid_response_reason(
                response,
                stop_token_ids=twisted.response_format.stop_token_ids,
                max_response_tokens=limit,
            )
            is None
        )
    # log q of generation is log q by teacher forcing.
    forced = twisted_model.token_logprobs(twisted, responses)
    for generated, scored in zip(draws.token_logq(), forced, strict=True):
        assert generated == pytest.approx(scored, abs=1e-4)
    # After the end of a response nothing is recorded, and log q of the prefix stays.
    for row, length in enumerate(draws.lengths):
        assert not draws.tokens[row, length:].any() and not draws.hidden[row, length:].any()
        assert (draws.prefix_logq[row, length - 1 :] == draws.prefix_logq[row, length - 1]).all()


def test_each_batch_of_64_draws_from_its_own_generator(twisted: Twisted) -> None:
    responses = twisted_model.sample(twisted, 70, 123).token_ids()
    assert twisted_model.sample(twisted, 70, 123).token_ids() == responses
    assert twisted_model.sample(twisted, 64, 123).token_ids() == responses[:64]
    assert twisted_model.sample(twisted, 6, 123 + 64).token_ids() == responses[64:]


def test_a_zero_twist_is_the_original_model(
    twisted: Twisted, original: original_model.OriginalModel, tiny_task: data.Task
) -> None:
    with torch.no_grad():
        twisted.head.out.weight.zero_()
    draws = twisted_model.sample(twisted, 20, 9)
    responses = draws.token_ids()
    logp = original.token_logprobs(tiny_task.prompt_token_ids, responses)
    for generated, scored in zip(draws.token_logq(), logp, strict=True):
        assert generated == pytest.approx(scored, abs=1e-5)


def test_one_pass_gives_log_q_and_log_p(
    twisted: Twisted, original: original_model.OriginalModel, tiny_task: data.Task
) -> None:
    responses = twisted_model.sample(twisted, 20, 5).token_ids()
    q_tokens, p_tokens = twisted_model.token_logprobs_with_p(twisted, responses)
    assert q_tokens == twisted_model.token_logprobs(twisted, responses)
    logp = original.token_logprobs(tiny_task.prompt_token_ids, responses)
    for mine, theirs in zip(p_tokens, logp, strict=True):
        assert mine == pytest.approx(theirs, abs=1e-5)


def test_the_sampler_is_the_twisted_model_with_the_weights_it_holds(
    twisted: Twisted, tiny_model: Any
) -> None:
    sampler = twisted_model.TwistedSampler(twisted)
    responses = sampler.sample(70, seed=123)
    assert responses == twisted_model.sample(twisted, 70, 123).token_ids()
    logq = sampler.logprobs(responses[:5])
    assert logq == [
        math.fsum(values) for values in twisted_model.token_logprobs(twisted, responses[:5])
    ]
    assert sampler.logprobs([]) == [] and sampler.inferences_per_response == 1.0
    # Other weights in the twist network: another sampler.
    with torch.no_grad():
        twisted.head.out.weight.mul_(2)
    assert sampler.logprobs(responses[:5]) != logq


def test_rows_carry_the_text_and_the_hit(
    twisted: Twisted, make_context: Any, tmp_path: Any
) -> None:
    context = make_context(tmp_path)
    responses = twisted_model.sample(twisted, 10, 3).token_ids()
    rows = twisted_model.label(context, responses, seed=3)
    assert [row["draw_index"] for row in rows] == list(range(10))
    for row, response in zip(rows, responses, strict=True):
        assert row["token_ids"] == response and row["seed"] == 3
        assert row["text"] == context.original.decode(response)
        assert row["hit"] == ("a" in row["text"])
