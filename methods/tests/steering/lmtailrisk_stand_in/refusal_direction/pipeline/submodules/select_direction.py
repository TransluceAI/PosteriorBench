"""Stand-in for the pipeline's refusal scores and its choice among the candidates."""

from __future__ import annotations

import os
from typing import Any

import torch


def get_refusal_scores(
    model: Any, instructions: list[str], tokenize_instructions_fn: Any, refusal_toks: list[int]
) -> torch.Tensor:
    """One score for each instruction: the log odds that the model's next token is a refusal
    token, minus their mean over the instructions, so that a random model has scores of both
    signs. One forward pass for each instruction."""
    prompts = tokenize_instructions_fn(instructions=instructions)
    logits = model(
        input_ids=prompts.input_ids.to(model.device),
        attention_mask=prompts.attention_mask.to(model.device),
    ).logits[:, -1]
    refusal = logits.double().softmax(-1)[:, refusal_toks].sum(-1)
    log_odds = torch.log(refusal + 1e-8) - torch.log(1 - refusal + 1e-8)
    return log_odds - log_odds.mean()


def select_direction(
    model_base: Any,
    harmful_instructions: list[str],
    harmless_instructions: list[str],
    candidate_directions: torch.Tensor,
    artifact_dir: str,
) -> tuple[int, int, torch.Tensor]:
    """The candidate with the largest norm, after one forward pass for each validation
    instruction. Returns its token position, counted from the end, its layer and the candidate."""
    os.makedirs(artifact_dir, exist_ok=True)
    for instructions in (harmful_instructions, harmless_instructions):
        get_refusal_scores(
            model_base.model,
            instructions,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
    positions, layers, _ = candidate_directions.shape
    position, layer = divmod(int(candidate_directions.norm(dim=-1).argmax()), layers)
    return position - positions, layer, candidate_directions[position, layer]
