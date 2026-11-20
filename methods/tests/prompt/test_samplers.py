"""The samplers of the prompt family on the tiny model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from posteriorbench import model_samplers, original_model, sampler

from posteriorbench_methods.common import composition, hf_runtime
from posteriorbench_methods.prompt import samplers


@pytest.fixture(scope="module")
def factory(scorer: original_model.OriginalModel) -> samplers.Factory:
    return samplers.Factory(scorer, scorer.render_prompt("hi"))


def spec(scorer: original_model.OriginalModel, *system_prompts: str) -> samplers.Spec:
    prompts = tuple(tuple(scorer.render_prompt("hi", system=s)) for s in system_prompts)
    weights = (
        (0.5, 0.3, 0.2)[: len(prompts)] if len(prompts) == 3 else (1 / len(prompts),) * len(prompts)
    )
    return samplers.Spec(prompts, weights)


def test_one_prompt_is_the_package_prompt_sampler(
    scorer: original_model.OriginalModel, factory: samplers.Factory
) -> None:
    one = spec(scorer, "be brief")
    built = factory.build(one, name="x")
    reference = model_samplers.PromptSampler(scorer, one.prompts[0])
    assert isinstance(built, model_samplers.PromptSampler) and built.name == "x"
    assert built.sample(5, seed=7) == reference.sample(5, seed=7)
    # The plain setting is the sampler itself: one inference per response, the same draws.
    plain = factory.build(one.composed(composition.Setting(0.0, 50)), name="x")
    assert plain.inferences_per_response == 1.0
    assert plain.sample(5, seed=7) == reference.sample(5, seed=7)


@pytest.mark.parametrize("alpha, switch", [(0.25, 4), (0.5, 50), (0.75, 16), (0.0, 4)])
def test_a_composed_sampler_scores_its_own_draws_as_it_generated_them(
    alpha: float, switch: int, scorer: original_model.OriginalModel, factory: samplers.Factory
) -> None:
    composed = factory.build(
        spec(scorer, "say a").composed(composition.Setting(alpha, switch)), name="composed"
    )
    assert isinstance(composed, hf_runtime.ModelSampler)
    assert composed.inferences_per_response == 2.0
    draws = composed.sample_draws(12, seed=3)
    teacher_forced = composed.logprobs([draw.token_ids for draw in draws])
    for draw, value in zip(draws, teacher_forced, strict=True):
        assert math.isclose(draw.logq, value, abs_tol=1e-4)


def test_a_composed_sampler_draws_in_parts_with_their_own_seeds(
    scorer: original_model.OriginalModel, factory: samplers.Factory
) -> None:
    composed = factory.build(
        spec(scorer, "be brief").composed(composition.Setting(0.5, 16)), name="composed"
    )
    drawn = composed.sample(70, seed=1)
    assert len(drawn) == 70 and drawn == composed.sample(70, seed=1)
    assert drawn != composed.sample(70, seed=2)
    assert drawn[: samplers.COMPOSED_PART_SIZE] == composed.sample(64, seed=1)


def test_a_mixture_picks_components_by_weight_and_scores_the_mixture(
    scorer: original_model.OriginalModel, factory: samplers.Factory
) -> None:
    mixed = spec(scorer, "say a", "zzz", "abc")
    built = factory.build(mixed, name="mixture")
    assert isinstance(built, samplers.MixtureSampler) and built.inferences_per_response == 1.0
    responses = built.sample(40, seed=4)
    assert len(responses) == 40 and responses == built.sample(40, seed=4)
    for response, value in zip(responses[:5], built.logprobs(responses[:5]), strict=True):
        parts = [scorer.logprobs(prompt, [response])[0] for prompt in mixed.prompts]
        expected = math.log(sum(w * math.exp(v) for w, v in zip(mixed.weights, parts, strict=True)))
        assert math.isclose(value, expected, abs_tol=1e-5)


class Constant(sampler.Sampler):
    """A sampler that always returns one response."""

    def __init__(self, response: list[int]) -> None:
        self.response = response

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        return [self.response] * n

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        return [0.0 if list(r) == self.response else -math.inf for r in responses]


def test_mixture_frequencies_and_density() -> None:
    mixture = samplers.MixtureSampler(
        [Constant([1]), Constant([2]), Constant([3])], [0.5, 0.3, 0.2]
    )
    drawn = mixture.sample(4000, seed=4)
    for response, weight in (([1], 0.5), ([2], 0.3), ([3], 0.2)):
        assert abs(drawn.count(response) / 4000 - weight) < 0.03
    assert mixture.logprobs([[1], [3], [9]]) == pytest.approx(
        [math.log(0.5), math.log(0.2), -math.inf]
    )
    assert samplers.mixture_logq([0.25, 0.75], [-2.0, -3.0]) == pytest.approx(
        math.log(0.25 * math.exp(-2.0) + 0.75 * math.exp(-3.0))
    )


def test_a_composed_mixture_composes_every_component_with_one_setting(
    scorer: original_model.OriginalModel, factory: samplers.Factory
) -> None:
    setting = composition.Setting(0.25, 4)
    mixed = spec(scorer, "say a", "zzz")
    built = factory.build(mixed.composed(setting), name="composed-mixture")
    assert built.inferences_per_response == 2.0
    responses = built.sample(10, seed=2)
    for response, value in zip(responses, built.logprobs(responses), strict=True):
        parts = [
            factory.build(samplers.Spec((prompt,)).composed(setting), name="x").logprobs(
                [response]
            )[0]
            for prompt in mixed.prompts
        ]
        expected = math.log(0.5 * math.exp(parts[0]) + 0.5 * math.exp(parts[1]))
        assert math.isclose(value, expected, abs_tol=1e-5)


def test_sampler_ids(scorer: original_model.OriginalModel) -> None:
    one, same, other = spec(scorer, "be brief"), spec(scorer, "be brief"), spec(scorer, "zzz")
    assert one.sampler_id == same.sampler_id != other.sampler_id
    plain = one.composed(composition.Setting(0.0, 50))
    composed = one.composed(composition.Setting(0.25, 4))
    assert len({one.sampler_id, plain.sampler_id, composed.sampler_id}) == 3
    mixed = spec(scorer, "be brief", "zzz")
    reweighted = samplers.Spec(mixed.prompts, (0.4, 0.6))
    assert mixed.sampler_id != reweighted.sampler_id


def test_impossible_samplers_are_refused(scorer: original_model.OriginalModel) -> None:
    prompt = tuple(scorer.render_prompt("hi", system="zzz"))
    with pytest.raises(sampler.SamplerError):
        samplers.Spec((prompt, prompt), (1.0,))
    with pytest.raises(sampler.SamplerError):
        samplers.Spec((prompt, prompt), (0.5, 0.6))
    with pytest.raises(sampler.SamplerError):
        samplers.Spec((prompt, ()), (0.5, 0.5))
    with pytest.raises(sampler.SamplerError):
        samplers.Spec((prompt,)).composed(composition.Setting(0.5, 4)).composed(
            composition.Setting(0.5, 4)
        )
