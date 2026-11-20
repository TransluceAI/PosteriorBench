"""The policy: its adapters, its teacher forcing, and how it draws."""

from __future__ import annotations

import copy
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
from posteriorbench import original_model

from posteriorbench_methods.common import hf_runtime, run_context, training_run
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import teacher_forcing


def test_adapters_are_the_only_trained_weights_and_start_at_zero(
    make_tiny_model: Callable[..., Any], tiny_model: Any
) -> None:
    model = make_tiny_model()
    wrapped = gfn_policy.attach_adapters(model, rank=4, alpha=16)
    assert len(wrapped) == 2 * 6  # two layers: four attention and two feed-forward projections
    trained = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trained) == 2 * len(wrapped)
    assert all(name.endswith(("lora_a", "lora_b")) for name in trained)
    ids = torch.tensor([[5, 6, 7, 8]])
    assert torch.equal(model(input_ids=ids).logits, tiny_model(input_ids=ids).logits)


def test_adapters_on_the_24_layers_of_the_benchmark_model(tiny_hybrid_model: Any) -> None:
    """18 layers with linear attention get four adapters each, 6 with full attention six."""
    model = copy.deepcopy(tiny_hybrid_model)
    wrapped = gfn_policy.attach_adapters(model, rank=2, alpha=16)
    assert len(wrapped) == 18 * 4 + 6 * 6 == 108
    assert "model.layers.0.linear_attn.in_proj_qkv" in wrapped
    assert "model.layers.3.self_attn.q_proj" in wrapped


def test_dropout_is_disabled() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.3))
    gfn_policy.disable_dropout(model)
    assert [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)] == [0.0]


def test_token_scores_are_the_models_log_probabilities(
    tiny_original_model: original_model.OriginalModel, tiny_task: Any
) -> None:
    prompt = tuple(tiny_task.prompt_token_ids)
    responses = tiny_original_model.generate(list(prompt), 6, seed=3)
    lm = hf_runtime.CausalLM(tiny_original_model.model, prompt)
    stops = tiny_original_model.stop_token_ids
    with torch.inference_mode():
        scores = teacher_forcing.token_scores(lm, stops, responses)
    expected = tiny_original_model.token_logprobs(list(prompt), responses)
    for response, (token, stop), values in zip(responses, scores, expected, strict=True):
        assert token.tolist() == pytest.approx(values, abs=1e-5)
        assert stop.shape == (len(response), len(stops))
        if response[-1] in stops:  # the final stop token's own log-probability, read twice
            assert float(stop[-1, stops.index(response[-1])]) == float(token[-1])
    sums = teacher_forcing.sequence_logprobs(lm, stops, responses, batch_size=4)
    assert sums == pytest.approx([sum(values) for values in expected], abs=1e-4)
    lists = teacher_forcing.token_logprobs(lm, stops, responses, batch_size=4)
    assert [len(values) for values in lists] == [len(response) for response in responses]


def test_a_policy_starts_as_the_original_model_and_depends_on_the_replicate(
    make_context: Callable[..., run_context.RunContext],
    quick: Callable[..., gfn_settings.Settings],
    tmp_path: pathlib.Path,
) -> None:
    def adapters(replicate: int) -> dict[str, torch.Tensor]:
        context = make_context(tmp_path, replicate=replicate)
        policy = gfn_policy.build(context, quick(update_caps=(1,)))
        assert not policy.model.training
        return training_run.trainable_tensors(policy.model)

    first, again, other = adapters(0), adapters(0), adapters(1)
    assert training_run.sampler_id(first) == training_run.sampler_id(again)
    assert training_run.sampler_id(first) != training_run.sampler_id(other)
    assert all(not tensor.any() for name, tensor in first.items() if name.endswith("lora_b"))


def test_the_sampler_draws_in_seeded_parts_and_scores_its_own_draws(
    make_context: Callable[..., run_context.RunContext],
    quick: Callable[..., gfn_settings.Settings],
    tmp_path: pathlib.Path,
) -> None:
    context = make_context(tmp_path)
    policy = gfn_policy.build(context, quick(update_caps=(1,)))
    sampler = policy.sampler()
    draws = sampler.sample_draws(40, seed=7)
    assert [draw.token_ids for draw in draws] == sampler.sample(40, seed=7)
    assert sampler.sample(40, seed=8) != sampler.sample(40, seed=7)
    for draw in draws:
        context.original.validate_response(draw.token_ids)
    # The second part of 32 draws has the seed 7 + 32.
    (tail,) = policy.draw([hf_runtime.DrawGroup(8, 7 + gfn_policy.PART_SIZE)])
    assert [draw.token_ids for draw in tail] == [draw.token_ids for draw in draws[32:]]
    logq = sampler.logprobs([draw.token_ids for draw in draws])
    assert logq == pytest.approx([draw.logq for draw in draws], abs=1e-4)
    assert sampler.inferences_per_response == 1.0


def test_a_generation_batch_must_hold_a_part(
    make_context: Callable[..., run_context.RunContext],
    quick: Callable[..., gfn_settings.Settings],
    tmp_path: pathlib.Path,
) -> None:
    settings = quick(update_caps=(1,), generation_batch_size=16)
    with pytest.raises(ValueError, match="generation_batch_size"):
        gfn_policy.build(make_context(tmp_path), settings)
