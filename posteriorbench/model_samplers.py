"""Reference samplers built from the original model itself: the original model as a sampler, the
original model under another prompt (an added system prompt), and the Instruction sampler.

Each costs one inference per response, and building them costs nothing; a method that selects
among system prompts charges that selection to its own ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from posteriorbench import data, instructions, protocol, sampler


class GenerativeModel(protocol.Scorer, Protocol):
    """What a prompt sampler needs from the original model: a `protocol.Scorer` that also
    generates responses and renders chat prompts (`original_model.OriginalModel` implements
    it)."""

    def generate(self, prompt_token_ids: Sequence[int], n: int, *, seed: int) -> list[list[int]]:
        """n complete responses to the prompt, sampled at temperature 1 with no truncation."""
        ...

    def render_prompt(self, user: str, system: str | None = None) -> list[int]:
        """Token ids of a chat prompt with thinking off."""
        ...


class PromptSampler(sampler.Sampler):
    """q(y) = p(y | sampler prompt): the original model run on `prompt_token_ids`, which may differ
    from the task's prompt. One inference per response."""

    def __init__(
        self, model: GenerativeModel, prompt_token_ids: Sequence[int], *, name: str = "prompt"
    ) -> None:
        if not prompt_token_ids:
            raise sampler.SamplerError("a prompt sampler needs a non-empty prompt")
        self.model = model
        self.prompt_token_ids = [int(t) for t in prompt_token_ids]
        self.name = name

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        return self.model.generate(self.prompt_token_ids, n, seed=seed)

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        return self.model.logprobs(self.prompt_token_ids, responses)


def original_model_sampler(model: GenerativeModel, task: data.Task) -> PromptSampler:
    """q = p, built at no cost. Like any sampler it passes or fails on its native hit rate q̂(S) in
    the evaluation draws: it passes when p(S) is well above 10%, fails when p(S) is well below,
    and near 10% it can go either way. When it passes and its references are scored by the same
    model, forward = log(q̂/p̂) and reverse = log(p̂/q̂), both close to zero."""
    return PromptSampler(model, task.prompt_token_ids, name="original_model")


def system_prompt_sampler(
    model: GenerativeModel, task: data.Task, system_prompt: str, *, name: str = "system_prompt"
) -> PromptSampler:
    """The original model with `system_prompt` added before the task's user turn, rendered with the
    chat template (thinking off). A method that chose the system prompt charges that choice to its
    own ledger."""
    prompt = model.render_prompt(task.prompt, system=system_prompt)
    return PromptSampler(model, prompt, name=name)


def instruction_sampler(model: GenerativeModel, task: data.Task) -> PromptSampler:
    """The Instruction reference sampler: the original model with the behavior's handwritten
    instruction (`instructions.INSTRUCTIONS`) as its system prompt. No selection and no test, so
    its construction cost is 0 inferences; one inference per response. Raises
    instructions.InstructionError for a behavior without an instruction."""
    text = instructions.instruction_for(task.behavior)
    return system_prompt_sampler(model, task, text, name="instruction")
