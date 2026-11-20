"""The FLOPs model: the sizes of the benchmark's model, what its passes cost, and the formulas
against a model that is built."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import transformers

from posteriorbench_methods.analysis import flops
from posteriorbench_methods.ctl import twisted_model
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.sft import train as sft_train

MEDIAN_PROMPT = 24  # tokens of the median prompt of the benchmark's 21 tasks
RESPONSE = 50  # the benchmark's response limit


def benchmark_model() -> flops.Shape:
    """The shape of Qwen3.5-0.8B, from the sizes of its configuration."""
    config = transformers.Qwen3_5TextConfig(
        vocab_size=248_320,
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        tie_word_embeddings=True,
    )
    return flops.Shape.from_config(config)


def trainable(model: Any) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def test_the_parameters_of_the_benchmark_model_and_of_what_each_method_trains() -> None:
    shape = benchmark_model()
    assert shape.layer_types.count(flops.FULL_ATTENTION) == 6
    assert shape.parameters() == 752_393_024
    assert shape.last_layers_parameters(4) == 83_018_720  # SFT
    assert shape.adapter_weights(64) == 30_277_632  # GFlowNet
    assert shape.twist_parameters() == 256_627_200  # CTL


def test_the_weights_that_a_token_passes_in_the_benchmark_model() -> None:
    shape = benchmark_model()
    layers = [shape.layer_weights(kind) for kind in shape.layer_types]
    assert shape.layer_weights(flops.LINEAR_ATTENTION) == 10_518_528 + 11_010_048
    assert shape.layer_weights(flops.FULL_ATTENTION) == 7_340_032 + 11_010_048
    assert sum(layers) == 497_614_848
    assert sum(layers[-4:]) == 82_935_808
    assert shape.output_weights == 254_279_680


def test_what_one_response_costs_each_method_in_inferences() -> None:
    cost = flops.passes(benchmark_model(), prompt_tokens=MEDIAN_PROMPT, response_tokens=RESPONSE)
    adapters = flops.factors("adapters", cost)
    assert adapters.sampler_draw == adapters.training_forward == pytest.approx(1.039, abs=5e-4)
    assert adapters.gradient_response == pytest.approx(1.094, abs=5e-4)
    twist = flops.factors("twist", cost)
    assert twist.sampler_draw == pytest.approx(1.224, abs=5e-4)
    assert (twist.training_forward, twist.gradient_response) == (0, pytest.approx(0.005, abs=5e-4))
    last_layers = flops.factors("last_layers", cost)
    assert (last_layers.sampler_draw, last_layers.training_forward) == (1, 1)
    assert last_layers.gradient_response == pytest.approx(0.542, abs=5e-4)
    assert flops.factors("nothing", cost) == flops.Factors(1.0, 1.0, 0.0)
    cached = flops.passes(
        benchmark_model(),
        prompt_tokens=MEDIAN_PROMPT,
        response_tokens=RESPONSE,
        convention=flops.CONVENTIONS["prompt_cached"],
    )
    assert flops.factors("twist", cached).sampler_draw == pytest.approx(1.33, abs=5e-3)


@pytest.mark.parametrize("convention", sorted(flops.CONVENTIONS))
def test_a_backward_pass_over_all_weights_costs_two_forward_passes(convention: str) -> None:
    """So a gradient step on a response that is also generated costs four inferences where the
    count has two: the compute of KL-DPG, JS-DPG and GDC stays below twice their count."""
    cost = flops.passes(
        benchmark_model(),
        prompt_tokens=MEDIAN_PROMPT,
        response_tokens=RESPONSE,
        convention=flops.CONVENTIONS[convention],
    )
    assert flops.factors("all_weights", cost) == flops.Factors(1.0, 1.0, 2.0)


def test_the_conventions_differ_in_what_a_forward_pass_counts() -> None:
    shape = benchmark_model()

    def forward(name: str) -> float:
        return flops.passes(
            shape,
            prompt_tokens=MEDIAN_PROMPT,
            response_tokens=RESPONSE,
            convention=flops.CONVENTIONS[name],
        ).forward

    every = forward("every_position")
    weights = 74 * 497_614_848 + 74 * 254_279_680
    assert forward("weights_only") == weights
    assert 0.02 < (every - weights) / every < 0.035  # the operations without weights
    assert every - forward("output_layer_where_read") == MEDIAN_PROMPT * 254_279_680
    assert forward("prompt_cached") < forward("output_layer_where_read")


def test_the_formulas_count_the_parameters_of_a_model_that_is_built(
    hybrid_copy: Callable[[], Any],
) -> None:
    """On a tiny model with the two kinds of layers of the benchmark's model: the parameters of
    the model, and those that each method's own code makes trainable."""
    model = hybrid_copy()
    shape = flops.Shape.from_config(model.config)
    assert shape.parameters() == sum(value.numel() for value in model.parameters())
    sft_train.freeze_all_but_last_layers(model, 4)
    assert shape.last_layers_parameters(4) == trainable(model)
    with_adapters = hybrid_copy()
    gfn_policy.attach_adapters(with_adapters, rank=8, alpha=16)
    assert shape.adapter_weights(8) == trainable(with_adapters)
    with_twist = hybrid_copy()
    twisted_model.attach_twist_head(with_twist, seed=0)
    assert shape.twist_parameters() == trainable(with_twist)


def test_a_configuration_of_another_architecture_is_an_error(tiny_model: Any) -> None:
    with pytest.raises(flops.ShapeError, match="Qwen3.5"):
        flops.Shape.from_config(tiny_model.config)
