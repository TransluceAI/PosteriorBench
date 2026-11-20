"""The GFlowNet policy: a copy of the original model with low-rank adapters.

The adapters are the only weights that train. They sit on the attention projections and on the
up and down projections of the feed-forward network of every layer, with the second factor
initialized to zero, so the policy starts as the original model.

The policy draws responses in groups, each group from a generator of its own, several groups in
one forward pass; selection draws, threshold tests and evaluation draws come in parts of
`PART_SIZE`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import torch

from posteriorbench_methods.common import hf_runtime, run_context
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import teacher_forcing

METHOD = "gflownet"  # the method's name in records, and the owner of its random streams
PART_SIZE = 32
# The projections that get an adapter, by the kind of attention of a layer.
FULL_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LINEAR_ATTENTION_PROJECTIONS = ("in_proj_qkv", "out_proj")
FEED_FORWARD_PROJECTIONS = ("up_proj", "down_proj")


class LoRALinear(torch.nn.Module):
    """A frozen linear layer plus a trained low-rank update: inner(x) + (alpha / rank) B A x,
    with A [rank, in] initialized as a linear layer's weight and B [out, rank] as zero."""

    def __init__(self, inner: torch.nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        self.inner = inner
        self.lora_a = torch.nn.Parameter(inner.weight.new_empty(rank, inner.in_features))
        self.lora_b = torch.nn.Parameter(inner.weight.new_zeros(inner.out_features, rank))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inner(inputs) + ((inputs @ self.lora_a.T) @ self.lora_b.T) * self.scaling


def attach_adapters(model: Any, *, rank: int, alpha: float) -> list[str]:
    """Freeze every weight of `model` and wrap its attention and feed-forward projections in
    `LoRALinear` (changes the model; the adapters' first factors are drawn from torch's global
    generator). Returns the names of the wrapped modules."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped: list[str] = []
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            attention, names = "self_attn", FULL_ATTENTION_PROJECTIONS
        else:
            attention, names = "linear_attn", LINEAR_ATTENTION_PROJECTIONS
        targets = [(attention, name) for name in names]
        targets += [("mlp", name) for name in FEED_FORWARD_PROJECTIONS]
        for parent_name, name in targets:
            parent = getattr(layer, parent_name)
            inner = getattr(parent, name)
            if not isinstance(inner, torch.nn.Linear):
                raise TypeError(f"layers.{index}.{parent_name}.{name} is not a linear layer")
            setattr(parent, name, LoRALinear(inner, rank=rank, alpha=alpha))
            wrapped.append(f"model.layers.{index}.{parent_name}.{name}")
    return wrapped


# Adapted from GFNOrg/gfn-lm-tuning, next_sentence/train.py, lines 138-141 (MIT License,
# Copyright (c) 2023 GFNOrg; THIRD_PARTY_LICENSES/GFNOrg-gfn-lm-tuning-MIT.txt).
def disable_dropout(model: torch.nn.Module) -> None:
    """Set the probability of every dropout module of `model` to zero (changes the model)."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


@dataclasses.dataclass(frozen=True)
class Policy:
    """The policy under the task's prompt. Drawing and scoring put the model in eval mode, and
    loading a checkpoint into `model` changes what this object draws and scores. A token of a
    group at temperature T is drawn from softmax(logits / T); the recorded log q is that of
    temperature 1."""

    model: Any
    lm: hf_runtime.CausalLM
    generation: hf_runtime.Generation
    forward_batch_size: int

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.generation.response_format.stop_token_ids

    def draw(self, groups: Sequence[hf_runtime.DrawGroup]) -> list[list[hf_runtime.Draw]]:
        """The responses of each group, the groups packed into forward passes in order."""
        self.model.eval()
        return hf_runtime.sample_groups(hf_runtime.TwoModels(self.lm), groups, self.generation)

    def parts(self, count: int, seed: int) -> list[hf_runtime.DrawGroup]:
        """`count` draws at temperature 1 as groups of PART_SIZE; the part that starts at draw
        `lo` has the seed `seed` + `lo`."""
        return hf_runtime.parts(count, seed, part_size=PART_SIZE, part_seed=_offset_seed)

    def logq(self, responses: Sequence[Sequence[int]]) -> list[float]:
        """log q(y) of each response under the policy, by teacher forcing."""
        self.model.eval()
        return teacher_forcing.sequence_logprobs(
            self.lm, self.stop_token_ids, responses, batch_size=self.forward_batch_size
        )

    def sampler(self) -> hf_runtime.ModelSampler:
        """The policy as a sampler that the benchmark evaluates."""
        return hf_runtime.ModelSampler(
            hf_runtime.TwoModels(self.lm),
            self.logq,
            self.generation,
            part_size=PART_SIZE,
            part_seed=_offset_seed,
            name=METHOD,
        )


def _offset_seed(seed: int, lo: int) -> int:
    return seed + lo


def build(context: run_context.RunContext, settings: gfn_settings.Settings) -> Policy:
    """A new copy of the original model with adapters, as the policy of this run's task. Seeds
    torch's global generator with the run's initialization seed, from which the adapters are
    drawn. Raises ValueError if the generation batch cannot hold one part."""
    if settings.generation_batch_size < PART_SIZE:
        raise ValueError(f"generation_batch_size must be at least {PART_SIZE}")
    torch.manual_seed(context.stream_seed(METHOD, "initialization"))
    model = context.load_model_copy().model
    attach_adapters(model, rank=settings.lora_rank, alpha=settings.lora_alpha)
    disable_dropout(model)
    model.eval()
    generation = hf_runtime.Generation(
        hf_runtime.ResponseFormat.of(context.original),
        batch_size=settings.generation_batch_size,
        renormalize=False,
    )
    lm = hf_runtime.CausalLM(model, tuple(context.task.prompt_token_ids))
    return Policy(model, lm, generation, settings.forward_batch_size)
