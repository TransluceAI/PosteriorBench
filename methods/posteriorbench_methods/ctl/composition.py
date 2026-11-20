"""CTL + composition: the composed sampler of the submitted checkpoint and the draws of each
setting, for `common.composition`. The twisted model is the original model with another output
layer, so one pass of the frozen model gives the next-token probabilities of both.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from posteriorbench_methods.common import composition, hf_runtime, run_context, seeds
from posteriorbench_methods.ctl import p_scores, twisted_model


def _part_seed(seed: int, first_draw: int) -> int:
    return seeds.substream_seed(seed, first_draw)


def sampler(
    twisted: twisted_model.TwistedModel, setting: composition.Setting
) -> hf_runtime.ModelSampler:
    """The twisted model composed with the original model under `setting`, as a sampler that the
    benchmark evaluates. It holds the twisted model, not a copy. Its log q of a response mixes
    the per-token log q and log p of one teacher-forced pass."""
    head = twisted.head

    def heads(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_logits = head.inner(hidden).float()
        return original_logits, original_logits + head.log_psi(hidden)

    def logprobs(responses: Sequence[Sequence[int]]) -> list[float]:
        q_tokens, p_tokens = twisted_model.token_logprobs_with_p(twisted, responses)
        return hf_runtime.composed_logprobs(q_tokens, p_tokens, setting=setting)

    return hf_runtime.ModelSampler(
        hf_runtime.SharedTrunk(twisted.model.model, twisted.prompt_ids, heads),
        logprobs,
        hf_runtime.Generation(twisted.response_format),
        part_size=twisted_model.GENERATION_BATCH,
        part_seed=_part_seed,
        setting=setting,
        name="ctl+composition",
    )


def selection_draws(
    context: run_context.RunContext,
    twisted: twisted_model.TwistedModel,
    setting: composition.Setting,
    sampler_id: str,
) -> list[dict[str, Any]]:
    """The selection draws of the checkpoint that the twisted model holds, composed under
    `setting`, as rows with the composed log q and log p. `sampler_id`, the checkpoint's, enters
    the seed."""
    seed = context.stream_seed("ctl", "composition-selection", sampler_id, setting.setting_id)
    source = sampler(twisted, setting)
    responses = source.sample(composition.SELECTION_DRAWS, seed=seed)
    logq = source.logprobs(responses)
    logp = [
        math.fsum(values)
        for values in p_scores.token_logprobs(context.original, twisted.prompt_ids, responses)
    ]
    return [
        {**row, "logq": q_value, "logp": p_value}
        for row, q_value, p_value in zip(
            twisted_model.label(context, responses, seed=seed), logq, logp, strict=True
        )
    ]


def threshold_test(
    context: run_context.RunContext,
    twisted: twisted_model.TwistedModel,
    setting: composition.Setting,
    count: int,
    sampler_id: str,
) -> list[dict[str, Any]]:
    """`count` new responses of the same sampler for a threshold test, as rows."""
    seed = context.stream_seed("ctl", "composition-threshold-test", sampler_id, setting.setting_id)
    responses = sampler(twisted, setting).sample(count, seed=seed)
    return twisted_model.label(context, responses, seed=seed)
