"""The samplers of Steering on the tiny models: what they draw, their log q, and their cost."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
from posteriorbench import original_model
from posteriorbench import sampler as pb_sampler

from posteriorbench_methods.common import composition, hf_runtime, run_context
from posteriorbench_methods.steering import candidates, samplers, vectors

STEERED = candidates.Candidate("with_ablation", 0, 2.0)
COMPOSED = composition.Setting(0.5, 16)
PLAIN = composition.Setting(0.0, 50)


@pytest.fixture(scope="module")
def steered_model(
    make_context: Callable[..., run_context.RunContext],
    vectors_directory: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> samplers.SteeredModel:
    context = make_context(tmp_path_factory.mktemp("samplers"))
    return samplers.SteeredModel.of(
        context, vectors.load(vectors_directory, "goblin_mention", (0, 1))
    )


def draws_of(source: hf_runtime.ModelSampler, count: int, seed: int) -> list[hf_runtime.Draw]:
    (drawn,) = hf_runtime.sample_groups(
        source.models,
        [hf_runtime.DrawGroup(count, seed)],
        source.generation,
        setting=source.setting,
    )
    return drawn


@pytest.mark.parametrize("composed", [None, PLAIN, COMPOSED])
def test_log_q_of_a_response_is_the_log_q_it_was_drawn_with(
    steered_model: samplers.SteeredModel, composed: composition.Setting | None
) -> None:
    """Generation reads the model one token at a time from a cache; `logprobs` reads it in one
    pass over the whole response. Both run inside the same hook."""
    source = samplers.sampler(steered_model, STEERED, composed)
    drawn = draws_of(source, 12, seed=5)
    scored = source.logprobs([one.token_ids for one in drawn])
    assert scored == pytest.approx([one.logq for one in drawn], abs=1e-4)


def test_the_same_holds_on_the_24_layer_model(
    tiny_hybrid_model: Any,
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
) -> None:
    scorer = make_tiny_original_model(model=tiny_hybrid_model, max_response_tokens=50)
    vector = torch.randn(32, generator=torch.Generator().manual_seed(1)) * 0.02
    steered = samplers.SteeredModel(
        model=tiny_hybrid_model,
        prompt=tuple(scorer.render_prompt("hi")),
        tensors={"with_ablation": {12: vector}},
        generation=hf_runtime.Generation(hf_runtime.ResponseFormat.of(scorer), p_from_start=False),
    )
    candidate = candidates.Candidate("with_ablation", 12, 3.0)
    for composed in (None, COMPOSED):
        source = samplers.sampler(steered, candidate, composed)
        drawn = draws_of(source, 6, seed=11)
        scored = source.logprobs([one.token_ids for one in drawn])
        assert scored == pytest.approx([one.logq for one in drawn], abs=1e-4)


def test_steering_changes_the_responses_and_the_original_model_candidate_does_not(
    steered_model: samplers.SteeredModel, steering_original: original_model.OriginalModel
) -> None:
    original = samplers.sampler(steered_model, candidates.ORIGINAL_MODEL)
    drawn = draws_of(original, 16, seed=3)
    responses = [one.token_ids for one in drawn]
    # The candidate without a vector is the original model: its log q is the package's log p.
    assert original.logprobs(responses) == pytest.approx(
        steering_original.logprobs(steered_model.prompt, responses), abs=1e-4
    )
    steered = samplers.sampler(steered_model, STEERED)
    assert [one.token_ids for one in draws_of(steered, 16, seed=3)] != responses
    assert steered.logprobs(responses) != pytest.approx(original.logprobs(responses), abs=1e-2)


def test_a_composed_draw_costs_two_inferences(steered_model: samplers.SteeredModel) -> None:
    plain = samplers.sampler(steered_model, STEERED)
    assert plain.inferences_per_response == 1.0 and plain.name == "steering"
    under_plain_setting = samplers.sampler(steered_model, STEERED, PLAIN)
    assert under_plain_setting.inferences_per_response == 1.0 and under_plain_setting.is_plain
    composed = samplers.sampler(steered_model, STEERED, COMPOSED)
    assert composed.inferences_per_response == 2.0 and composed.name == "steering+composition"
    original = samplers.sampler(steered_model, candidates.ORIGINAL_MODEL).models
    assert isinstance(original, hf_runtime.TwoModels) and original.p is None


def test_sample_gives_complete_responses_that_depend_on_the_seed(
    steered_model: samplers.SteeredModel, steering_original: original_model.OriginalModel
) -> None:
    source = samplers.sampler(steered_model, STEERED, COMPOSED)
    responses = source.sample(130, seed=2**31 + 3)  # two parts
    assert len(responses) == 130
    assert responses == source.sample(130, seed=2**31 + 3)
    assert responses[:5] != source.sample(5, seed=2**31 + 4)
    for response in responses:
        reason = pb_sampler.invalid_response_reason(
            response, stop_token_ids=steering_original.stop_token_ids, max_response_tokens=50
        )
        assert reason is None


def test_vectors_that_do_not_fit_the_model_are_refused(
    make_context: Callable[..., run_context.RunContext],
    write_vectors: Callable[..., pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    context = make_context(tmp_path / "out")
    beyond = vectors.load(write_vectors(tmp_path / "a", layers=(0, 7)), "goblin_mention", (0,))
    with pytest.raises(vectors.VectorError, match="does not fit"):
        samplers.SteeredModel.of(context, beyond)
    fitting = vectors.load(write_vectors(tmp_path / "b"), "goblin_mention", (0,))
    longer = vectors.BehaviorVectors(
        "goblin_mention", {"with_ablation": {0: torch.zeros(33)}}, fitting.cost
    )
    with pytest.raises(vectors.VectorError, match="does not fit"):
        samplers.SteeredModel.of(context, longer)
