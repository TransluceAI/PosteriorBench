"""Stand-in for the pipeline's ablation hooks: the same two functions, with the same arguments
and the same kinds of hook."""

from __future__ import annotations

from typing import Any

import torch


def _without_component(hidden: torch.Tensor, direction: torch.Tensor, intensity: float) -> Any:
    unit = (direction / direction.norm()).to(hidden)
    return hidden - intensity * (hidden @ unit).unsqueeze(-1) * unit


def get_direction_ablation_input_pre_hook(
    direction: torch.Tensor, ablation_intensity: float = 1.0
) -> Any:
    """A forward pre-hook hook_fn(module, input): `input` is the tuple of a layer's positional
    arguments, hidden states first. Returns the tuple with the hidden states ablated."""

    def hook_fn(module: Any, input: tuple[Any, ...]) -> tuple[Any, ...]:
        return (_without_component(input[0], direction, ablation_intensity), *input[1:])

    return hook_fn


def get_direction_ablation_output_hook(
    direction: torch.Tensor, ablation_intensity: float = 1.0
) -> Any:
    """A forward hook hook_fn(module, input, output): `output` is a module's hidden states,
    alone or first in a tuple. Returns it with the hidden states ablated."""

    def hook_fn(module: Any, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            return (_without_component(output[0], direction, ablation_intensity), *output[1:])
        return _without_component(output, direction, ablation_intensity)

    return hook_fn
