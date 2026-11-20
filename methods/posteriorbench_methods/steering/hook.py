"""The steering hook: a vector added to the residual stream after one layer.

Steering changes the residual stream h_b after decoder layer b to h_b + gamma v_b at every token
position, the prompt's included. `steer` does this to a transformers model for as long as its
context is open, so the same weights serve the steered sampler and the original model.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

import torch


def decoder_layers(model: Any) -> Any:
    """The decoder layers of a transformers text model, `model.model.layers`. Raises TypeError
    for a model that keeps its layers elsewhere."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a text model with its decoder layers at model.model.layers")
    return layers


@contextlib.contextmanager
def steer(
    model: Any, layer: int, vector: torch.Tensor, strength: float
) -> Generator[None, None, None]:
    """Steer `model` while the context is open: every forward pass adds `strength` x `vector` to
    the output of decoder layer `layer` (counted from 0), at every position of every row.

    `vector` has shape [hidden size]; it is moved to the device and dtype of the layer's output.
    The hook is removed when the context closes, also after an error."""
    assert vector.ndim == 1, "a steering vector has shape [hidden size]"

    def add_vector(module: Any, inputs: Any, output: Any) -> Any:
        # A layer returns its hidden states [rows, positions, hidden size], alone or first in a
        # tuple.
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + vector.to(device=hidden.device, dtype=hidden.dtype) * strength
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    handle = decoder_layers(model)[layer].register_forward_hook(add_vector)
    try:
        yield
    finally:
        handle.remove()
