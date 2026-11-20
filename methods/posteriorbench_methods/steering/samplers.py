"""The samplers of Steering. A candidate's q runs the original model inside the steering hook
(`hook.steer`) and p runs it without, so one copy of the weights serves both.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from posteriorbench_methods.common import composition, hf_runtime, run_context, seeds
from posteriorbench_methods.steering import candidates, hook
from posteriorbench_methods.steering import vectors as steering_vectors

GENERATION_PART = 128  # responses that a sampler generates in one batch
SCORING_BATCH = 32  # responses in one forward pass that computes log q of given responses


@dataclasses.dataclass(frozen=True)
class SteeredModel:
    """The original model under the task's prompt, with the behavior's vectors
    (`tensors[vector][layer]`, shape [hidden size], on the model's device). `generation_rows` and
    `scoring_rows` are responses per forward pass when generating and when scoring; fewer than
    the constants above on tasks whose model or response limit needs it."""

    model: Any
    prompt: tuple[int, ...]
    tensors: Mapping[str, Mapping[int, torch.Tensor]]
    generation: hf_runtime.Generation
    generation_rows: int = GENERATION_PART
    scoring_rows: int = SCORING_BATCH

    @classmethod
    def of(
        cls, context: run_context.RunContext, vectors: steering_vectors.BehaviorVectors
    ) -> SteeredModel:
        """The run's original model with `vectors`. Raises VectorError if a vector does not fit
        the model's hidden size or layers."""
        model = context.original.model
        layers, hidden_size = len(hook.decoder_layers(model)), int(model.config.hidden_size)
        for name, by_layer in vectors.tensors.items():
            for layer, tensor in by_layer.items():
                if layer >= layers or tensor.shape[0] != hidden_size:
                    raise steering_vectors.VectorError(
                        f"the vector {name} at layer {layer} of {vectors.behavior} does not fit "
                        f"a model with {layers} layers and hidden size {hidden_size}"
                    )
        device = context.original.device
        return cls(
            model=model,
            prompt=tuple(context.task.prompt_token_ids),
            tensors={
                name: {layer: tensor.to(device) for layer, tensor in by_layer.items()}
                for name, by_layer in vectors.tensors.items()
            },
            # In a setting whose first tokens do not read p, p starts at the first token that
            # does, with one pass over the prompt and the response so far.
            generation=hf_runtime.Generation(
                hf_runtime.ResponseFormat.of(context.original), p_from_start=False
            ),
            generation_rows=context.generation_batch(GENERATION_PART),
            scoring_rows=context.scoring_batch(SCORING_BATCH),
        )

    @property
    def response_limit(self) -> int:
        return self.generation.response_format.max_tokens


def _summed_token_logprobs(
    q: hf_runtime.CausalLM, responses: Sequence[Sequence[int]], response_limit: int, batch: int
) -> list[float]:
    """log q of each response: its per-token values, summed in float64 over a zero-padded row of
    `response_limit` entries. The rounding of the sum depends on the row's length, so the row is
    always that long."""
    sums: list[float] = []
    for lo in range(0, len(responses), batch):
        part = responses[lo : lo + batch]
        rows = np.zeros((len(part), response_limit), dtype=np.float64)
        token_values = hf_runtime.token_logprobs(q, part, batch_size=batch)
        for row, values in enumerate(token_values):
            rows[row, : len(values)] = values
        sums.extend(rows.sum(1).tolist())
    return sums


def sampler(
    steered: SteeredModel,
    candidate: candidates.Candidate,
    composed: composition.Setting | None = None,
) -> hf_runtime.ModelSampler:
    """The sampler of `candidate`, composed with the original model under `composed` unless that
    is None or plain. log q is by teacher forcing: the summed per-token values, or for a composed
    sampler the per-token values of q and p mixed as the setting says."""
    model, prompt = steered.model, steered.prompt
    p = hf_runtime.CausalLM(model, prompt)
    if candidate.is_original_model:
        q, models = p, hf_runtime.TwoModels(p)
    else:
        assert candidate.vector is not None and candidate.layer is not None
        layer, strength = candidate.layer, candidate.strength
        vector = steered.tensors[candidate.vector][layer]

        def steered_pass(start: int) -> contextlib.AbstractContextManager[None]:
            return hook.steer(model, layer, vector, strength)

        q = hf_runtime.CausalLM(model, prompt, steered_pass)
        models = hf_runtime.TwoModels(q, p)
    limit = steered.response_limit
    part, batch = steered.generation_rows, steered.scoring_rows

    def part_seed(seed: int, first_draw: int) -> int:
        """The seed of the part of a `sample(n, seed)` call that starts at draw `first_draw`."""
        return seeds.substream_seed(seed, first_draw // part)

    if composed is None or composed.is_plain(limit):
        return hf_runtime.ModelSampler(
            models,
            lambda responses: _summed_token_logprobs(q, responses, limit, batch),
            steered.generation,
            part_size=part,
            part_seed=part_seed,
            setting=composed,
            name="steering",
        )

    def composed_logprobs(responses: Sequence[Sequence[int]]) -> list[float]:
        q_tokens = hf_runtime.token_logprobs(q, responses, batch_size=batch)
        p_tokens = hf_runtime.token_logprobs(p, responses, batch_size=batch)
        return hf_runtime.composed_logprobs(q_tokens, p_tokens, setting=composed)

    return hf_runtime.ModelSampler(
        models,
        composed_logprobs,
        steered.generation,
        part_size=part,
        part_seed=part_seed,
        setting=composed,
        name="steering+composition",
    )
