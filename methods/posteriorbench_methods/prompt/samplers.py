"""The samplers of the prompt family: q(y) = p(y | c, x) for a candidate system prompt c before
the task's user prompt x, a fixed-weight mixture of such prompts, and either one composed with
the original model (a composed mixture uses one setting for all components). `Spec` names a
sampler without a model; `Factory` builds it on the original model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Sequence
from typing import Any

from posteriorbench import model_samplers, original_model, sampler

from posteriorbench_methods.common import composition, hf_runtime, seeds

# A composed sampler generates the draws of a `sample` call in parts of 64, each with its own
# seed; in smaller parts for a task whose model needs smaller forward passes.
COMPOSED_PART_SIZE = 64


def check_weights(weights: Sequence[float], *, components: int) -> None:
    """Raise SamplerError unless `weights` has one positive weight per component, summing to 1."""
    if components < 1 or len(weights) != components:
        raise sampler.SamplerError("a sampler has one weight for each of its components")
    positive = all(weight > 0 for weight in weights)
    if not positive or not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-10):
        raise sampler.SamplerError("the weights of a mixture are positive and sum to 1")


@dataclasses.dataclass(frozen=True)
class Spec:
    """A sampler of this family. `prompts` holds each component's rendered token ids (system
    prompt, then the task's user prompt); `setting` is None if the sampler is not composed."""

    prompts: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...] = (1.0,)
    setting: composition.Setting | None = None

    def __post_init__(self) -> None:
        if any(not prompt for prompt in self.prompts):
            raise sampler.SamplerError("a prompt has at least one token")
        check_weights(self.weights, components=len(self.prompts))

    @property
    def sampler_id(self) -> str:
        """The sha256 of the prompts, weights and setting. A single uncomposed prompt is hashed by
        its token ids alone, so candidates that render to the same tokens are one sampler."""
        if len(self.prompts) == 1 and self.setting is None:
            description: list[Any] = list(self.prompts[0])
        else:
            setting = None if self.setting is None else self.setting.setting_id
            description = [[list(prompt) for prompt in self.prompts], list(self.weights), setting]
        encoded = json.dumps(description, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def composed(self, setting: composition.Setting) -> Spec:
        """This sampler composed with the original model under `setting`."""
        if self.setting is not None:
            raise sampler.SamplerError("the sampler is composed already")
        return dataclasses.replace(self, setting=setting)


def mixture_logq(weights: Sequence[float], component_logq: Sequence[float]) -> float:
    """log q_mix(y) = log sum_i w_i q_i(y), from the log q_i(y) of the components."""
    terms = [math.log(weight) + logq for weight, logq in zip(weights, component_logq, strict=True)]
    largest = max(terms)
    if largest == -math.inf:
        return -math.inf
    return largest + math.log(math.fsum(math.exp(term - largest) for term in terms))


class MixtureSampler(sampler.Sampler):
    """q_mix(y) = sum_i w_i q_i(y): each response picks component i with probability w_i and is
    drawn from q_i. A draw costs what a draw of a component costs."""

    def __init__(
        self,
        components: Sequence[sampler.Sampler],
        weights: Sequence[float],
        *,
        name: str = "mixture",
    ) -> None:
        check_weights(weights, components=len(components))
        self.components = list(components)
        self.weights = [float(weight) for weight in weights]
        self.name = name

    @property
    def inferences_per_response(self) -> float:
        return max(component.inferences_per_response for component in self.components)

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        """n responses. The components of the n responses are drawn first, independently by
        weight; each component then draws its responses in one call, with a seed of its own."""
        picker = random.Random(seeds.substream_seed(seed, "components"))
        picked = picker.choices(range(len(self.weights)), weights=self.weights, k=n)
        responses: list[list[int]] = [[] for _ in range(n)]
        for component in sorted(set(picked)):
            positions = [position for position, c in enumerate(picked) if c == component]
            drawn = self.components[component].sample(
                len(positions), seed=seeds.substream_seed(seed, "component", component)
            )
            for position, response in zip(positions, drawn, strict=True):
                responses[position] = response
        return responses

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        if not responses:
            return []
        by_component = [component.logprobs(responses) for component in self.components]
        return [
            mixture_logq(self.weights, [values[index] for values in by_component])
            for index in range(len(responses))
        ]


class Factory:
    """Builds the samplers of one task on the original model. `task_prompt` is the token ids
    under which the original model is p; `composed_part_size` is the responses a composed sampler
    generates per forward pass."""

    def __init__(
        self,
        original: original_model.OriginalModel,
        task_prompt: Sequence[int],
        *,
        composed_part_size: int = COMPOSED_PART_SIZE,
    ) -> None:
        self.original = original
        self.task_prompt = tuple(int(token) for token in task_prompt)
        self.composed_part_size = composed_part_size
        # A composed sampler runs the original model under two prompts, each with its own cache;
        # the cache of p starts at the first token that reads p.
        self.generation = hf_runtime.Generation(
            hf_runtime.ResponseFormat.of(original), p_from_start=False
        )

    @property
    def response_limit(self) -> int:
        """The most tokens in a response: the largest L of the composition settings."""
        return self.generation.response_format.max_tokens

    def build(self, spec: Spec, *, name: str) -> sampler.Sampler:
        """The sampler that `spec` describes, with `name` as its name in a result."""
        components = [self._component(prompt, spec.setting, name) for prompt in spec.prompts]
        if len(components) == 1:
            return components[0]
        return MixtureSampler(components, spec.weights, name=name)

    def _component(
        self, prompt: tuple[int, ...], setting: composition.Setting | None, name: str
    ) -> sampler.Sampler:
        """The original model under `prompt`, composed with the original model under the task's
        prompt if `setting` is a composed setting."""
        if setting is None or setting.is_plain(self.response_limit):
            return model_samplers.PromptSampler(self.original, prompt, name=name)
        models = hf_runtime.TwoModels(
            q=hf_runtime.CausalLM(self.original.model, prompt),
            p=hf_runtime.CausalLM(self.original.model, self.task_prompt),
        )

        def logprobs(responses: Sequence[Sequence[int]]) -> list[float]:
            return hf_runtime.composed_logprobs(
                self.original.token_logprobs(prompt, responses),
                self.original.token_logprobs(self.task_prompt, responses),
                setting=setting,
            )

        part_size = self.composed_part_size
        return hf_runtime.ModelSampler(
            models,
            logprobs,
            self.generation,
            part_size=part_size,
            part_seed=lambda seed, lo: seeds.substream_seed(seed, "chunk", lo // part_size),
            setting=setting,
            name=name,
        )
