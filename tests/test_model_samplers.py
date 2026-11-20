"""Reference samplers: the original model, a system-prompt sampler and the Instruction sampler."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from posteriorbench import data, instructions, model_samplers, rules, sampler


def test_every_behavior_has_an_instruction() -> None:
    assert set(instructions.INSTRUCTIONS) == set(rules.LABELS)
    assert instructions.instruction_for("em_dash") == (
        "Use an em dash (—) in the very first sentence of your reply."
    )
    with pytest.raises(instructions.InstructionError):
        instructions.instruction_for("synthetic")


def test_instruction_sampler(tiny_model: Any, common_task: data.Task) -> None:
    candidate = model_samplers.instruction_sampler(tiny_model, common_task)
    text = instructions.INSTRUCTIONS[common_task.behavior]
    assert candidate.name == "instruction" and candidate.inferences_per_response == 1.0
    assert candidate.prompt_token_ids == tiny_model.render_prompt(common_task.prompt, system=text)
    assert candidate.prompt_token_ids[-len(common_task.prompt_token_ids) :] == list(
        common_task.prompt_token_ids
    )
    draws = candidate.sample(5, seed=1)
    assert draws == tiny_model.generate(candidate.prompt_token_ids, 5, seed=1)
    assert candidate.logprobs(draws) == tiny_model.logprobs(candidate.prompt_token_ids, draws)
    other = dataclasses.replace(common_task, behavior="synthetic")
    with pytest.raises(instructions.InstructionError):
        model_samplers.instruction_sampler(tiny_model, other)


def test_original_and_system_prompt_samplers(tiny_model: Any, common_task: data.Task) -> None:
    original = model_samplers.original_model_sampler(tiny_model, common_task)
    assert original.prompt_token_ids == list(common_task.prompt_token_ids)
    assert original.name == "original_model" and original.inferences_per_response == 1.0
    response = original.sample(1, seed=0)[0]
    assert (
        original.logprob(response) == tiny_model.logprobs(original.prompt_token_ids, [response])[0]
    )
    custom = model_samplers.system_prompt_sampler(tiny_model, common_task, "be odd", name="mine")
    assert custom.name == "mine"
    assert custom.prompt_token_ids == tiny_model.render_prompt("hi", system="be odd")
    with pytest.raises(sampler.SamplerError):
        model_samplers.PromptSampler(tiny_model, [])
