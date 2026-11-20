"""Stand-in for the pipeline's candidate directions."""

from __future__ import annotations

import os
from typing import Any

import torch


def _mean_layer_inputs(model_base: Any, instructions: list[str]) -> torch.Tensor:
    """The mean input of every decoder layer at the token positions after the instruction:
    [positions, layers, hidden size], float64."""
    positions = len(model_base.eoi_toks)
    blocks = list(model_base.model_block_modules)
    total = torch.zeros(
        (positions, len(blocks), model_base.model.config.hidden_size), dtype=torch.float64
    )

    def add_input_of(layer: int) -> Any:
        def pre_hook(module: Any, args: tuple[Any, ...]) -> None:
            total[:, layer] += args[0][:, -positions:].double().sum(0).cpu()

        return pre_hook

    handles = [
        block.register_forward_pre_hook(add_input_of(layer)) for layer, block in enumerate(blocks)
    ]
    try:
        prompts = model_base.tokenize_instructions_fn(instructions=instructions)
        model_base.model(
            input_ids=prompts.input_ids.to(model_base.model.device),
            attention_mask=prompts.attention_mask.to(model_base.model.device),
        )
    finally:
        for handle in handles:
            handle.remove()
    return total / len(instructions)


def generate_directions(
    model_base: Any,
    harmful_instructions: list[str],
    harmless_instructions: list[str],
    artifact_dir: str,
) -> torch.Tensor:
    """One candidate for each token position after the instruction and each layer:
    [positions, layers, hidden size]."""
    os.makedirs(artifact_dir, exist_ok=True)
    candidates = _mean_layer_inputs(model_base, harmful_instructions) - _mean_layer_inputs(
        model_base, harmless_instructions
    )
    torch.save(candidates, os.path.join(artifact_dir, "mean_diffs.pt"))
    return candidates
