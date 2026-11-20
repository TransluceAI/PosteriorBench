"""The steering hook on the tiny 24-layer model: what it adds, and where."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from posteriorbench_methods.steering import hook

LAYER, STRENGTH = 12, 3.0
IDS = torch.tensor([[1, 5, 9, 2, 7, 4], [1, 6, 6, 3, 8, 8]])


@pytest.fixture(scope="module")
def vector() -> torch.Tensor:
    return torch.randn(32, generator=torch.Generator().manual_seed(1)) * 0.02


def run(model: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The output of decoder layer `layer` [rows, positions, hidden size], as the next layer
    receives it, and the model's logits, for the rows of IDS."""
    seen: list[torch.Tensor] = []

    def keep(module: Any, inputs: Any, output: Any) -> None:
        seen.append(output[0] if isinstance(output, tuple) else output)

    handle = hook.decoder_layers(model)[layer].register_forward_hook(keep)
    try:
        with torch.no_grad():
            logits = model(input_ids=IDS).logits
    finally:
        handle.remove()
    return seen[0], logits


def test_the_vector_is_added_after_its_layer_at_every_position(
    tiny_hybrid_model: Any, vector: torch.Tensor
) -> None:
    plain, plain_logits = run(tiny_hybrid_model, LAYER)
    before, _ = run(tiny_hybrid_model, LAYER - 1)
    with hook.steer(tiny_hybrid_model, LAYER, vector, STRENGTH):
        steered, steered_logits = run(tiny_hybrid_model, LAYER)
        steered_before, _ = run(tiny_hybrid_model, LAYER - 1)
    assert plain.shape == (2, 6, 32)
    assert torch.allclose(steered - plain, (vector * STRENGTH).expand(2, 6, 32), atol=1e-6)
    assert torch.equal(steered_before, before)
    assert not torch.allclose(steered_logits, plain_logits, atol=1e-3)


def test_the_hook_is_removed_when_the_context_closes(
    tiny_hybrid_model: Any, vector: torch.Tensor
) -> None:
    _, plain_logits = run(tiny_hybrid_model, LAYER)
    with pytest.raises(RuntimeError, match="stop"):
        with hook.steer(tiny_hybrid_model, LAYER, vector, STRENGTH):
            raise RuntimeError("stop")
    assert torch.equal(run(tiny_hybrid_model, LAYER)[1], plain_logits)


def test_a_model_without_decoder_layers_is_refused() -> None:
    with pytest.raises(TypeError, match="model.model.layers"):
        hook.decoder_layers(torch.nn.Linear(2, 2))
