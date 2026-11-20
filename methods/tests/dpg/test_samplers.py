"""The samplers of the family and their draws, on the tiny model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from posteriorbench_methods.common import composition, run_context, signals
from posteriorbench_methods.dpg import samplers


@pytest.fixture(scope="module")
def context(
    make_context: Callable[..., run_context.RunContext], tmp_path_factory: pytest.TempPathFactory
) -> run_context.RunContext:
    return make_context(tmp_path_factory.mktemp("dpg_samplers"), "kl-dpg")


@pytest.fixture(scope="module")
def models(context: run_context.RunContext, make_tiny_model: Callable[..., Any]) -> samplers.Models:
    """A model with other weights than the original model as q."""
    return samplers.Models.of(context, make_tiny_model(seed=1))


def test_generation_records_the_log_q_of_teacher_forcing(
    context: run_context.RunContext, models: samplers.Models
) -> None:
    limit = models.generation.response_format.max_tokens
    sources = [samplers.sampler(models)]
    sources += [
        samplers.composed_sampler(models, composition.Setting(alpha, switch), name="composed")
        for alpha, switch in ((0.5, 4), (0.0, 4), (0.75, limit), (0.0, limit))
    ]
    for source in sources:
        rows = samplers.training_draws(context, source, 24, seed=3)
        assert all(len(row["token_ids"]) <= limit for row in rows)
        logq = source.logprobs([row["token_ids"] for row in rows])
        assert np.allclose(logq, [row["logq_generation"] for row in rows], atol=1e-4)
        assert rows == samplers.training_draws(context, source, 24, seed=3)


def test_the_plain_setting_is_the_sampler_itself(models: samplers.Models) -> None:
    limit = models.generation.response_format.max_tokens
    plain = samplers.composed_sampler(models, composition.Setting(0.0, limit), name="composed")
    mixed = samplers.composed_sampler(models, composition.Setting(0.5, limit), name="composed")
    assert plain.sample(20, seed=5) == samplers.sampler(models).sample(20, seed=5)
    assert (plain.inferences_per_response, mixed.inferences_per_response) == (1.0, 2.0)
    assert samplers.sampler(models).inferences_per_response == 1.0


def test_a_threshold_test_draws_in_parts_of_32_with_seeds_of_their_own(
    models: samplers.Models,
) -> None:
    """The part that starts at draw 32 k is the first part of a call with the seed + 32 k."""
    in_parts = samplers.sampler(models, generation_part=32)
    expected = [
        response
        for start, count in ((0, 32), (32, 32), (64, 6))
        for response in in_parts.sample(count, seed=1000 + start)
    ]
    assert samplers.test_sampler(models).sample(70, seed=1000) == expected


def test_selection_draws_have_log_p_and_log_q_of_every_draw(
    context: run_context.RunContext, models: samplers.Models
) -> None:
    rows = samplers.selection_draws(context, samplers.sampler(models), 40, seed=11)
    assert [row["draw_index"] for row in rows] == list(range(40))
    assert all(
        set(row) == {"draw_index", "token_ids", "hit", "seed", "logq", "logp"} for row in rows
    )
    assert 0 < sum(row["hit"] for row in rows) < 40
    responses = [row["token_ids"] for row in rows]
    prompt = context.task.prompt_token_ids
    assert [row["logp"] for row in rows] == pytest.approx(
        context.original.logprobs(prompt, responses)
    )
    assert samplers.log_p(context, responses) == [row["logp"] for row in rows]
    arrays = signals.SelectionDraws.from_rows(rows)
    assert arrays.hit.dtype == bool and arrays.logq.shape == arrays.logp.shape == (40,)
    # A draw of q that the original model would have made is scored alike by both.
    same = samplers.Models.of(context, context.original.model)
    for row in samplers.selection_draws(context, samplers.sampler(same), 8, seed=11):
        assert row["logq"] == pytest.approx(row["logp"], abs=1e-4)


def test_a_sampler_follows_the_weights_of_its_model(
    context: run_context.RunContext, make_tiny_model: Callable[..., Any]
) -> None:
    model = make_tiny_model()
    source = samplers.sampler(samplers.Models.of(context, model))
    before = source.sample(16, seed=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(1.5)
    assert source.sample(16, seed=2) != before


def test_the_sampler_that_gdc_returns_has_smaller_passes(models: samplers.Models) -> None:
    assert samplers.returned_sampler(models, "gdc").part_size == 32
    assert samplers.returned_sampler(models, "kl-dpg").part_size == 128
    assert samplers.returned_sampler(models, "js-dpg").name == "js-dpg"
    responses = samplers.sampler(models).sample(20, seed=4)
    assert samplers.returned_sampler(models, "gdc").logprobs(responses) == pytest.approx(
        samplers.returned_sampler(models, "kl-dpg").logprobs(responses), abs=1e-4
    )
