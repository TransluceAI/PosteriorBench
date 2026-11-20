"""GFlowNet + composition: the composed sampler of the submitted checkpoint and the draws of each
setting, for `common.composition`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

from posteriorbench_methods.common import composition, hf_runtime, run_context, seeds
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import teacher_forcing

METHOD = f"{gfn_policy.METHOD}+composition"
PART_SIZE = 128  # the draws of a composed sampler come in parts of this many responses
SCORE_BATCH_SIZE = 32  # responses in one teacher-forced pass of q or of p


def _part_seed(seed: int, lo: int) -> int:
    return seeds.substream_seed(seed, lo)


@dataclasses.dataclass(frozen=True)
class ComposedPolicy:
    """The policy q and the original model p under the task's prompt, to be composed under any
    setting. q is the policy as it is: it changes when another checkpoint is loaded into it."""

    q: hf_runtime.CausalLM
    p: hf_runtime.CausalLM
    generation: hf_runtime.Generation

    @property
    def response_limit(self) -> int:
        return self.generation.response_format.max_tokens

    def logprobs(
        self, responses: Sequence[Sequence[int]], setting: composition.Setting
    ) -> tuple[list[float], list[float]]:
        """(log q under `setting`, log p) of each response, from the per-token values of both
        models."""
        stops = self.generation.response_format.stop_token_ids
        q_tokens = teacher_forcing.token_logprobs(
            self.q, stops, responses, batch_size=SCORE_BATCH_SIZE
        )
        p_tokens = teacher_forcing.token_logprobs(
            self.p, stops, responses, batch_size=SCORE_BATCH_SIZE
        )
        logq = hf_runtime.composed_logprobs(q_tokens, p_tokens, setting=setting)
        return logq, [math.fsum(tokens) for tokens in p_tokens]

    def sampler(self, setting: composition.Setting) -> hf_runtime.ModelSampler:
        """q composed with p under `setting`, as a sampler that the benchmark evaluates."""
        return hf_runtime.ModelSampler(
            hf_runtime.TwoModels(self.q, self.p),
            lambda responses: self.logprobs(responses, setting)[0],
            self.generation,
            part_size=PART_SIZE,
            part_seed=_part_seed,
            setting=setting,
            name=METHOD,
        )


def composed_policy(context: run_context.RunContext, policy: gfn_policy.Policy) -> ComposedPolicy:
    """`policy` and the run's original model, ready to be composed."""
    return ComposedPolicy(
        q=policy.lm,
        p=hf_runtime.CausalLM(context.original.model, policy.lm.prompt_ids),
        generation=hf_runtime.Generation(policy.generation.response_format),
    )


def selection_draws(
    context: run_context.RunContext,
    composed: ComposedPolicy,
    setting: composition.Setting,
    *,
    sampler_id: str,
    is_hit: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """The selection draws of the checkpoint that the policy holds, composed under `setting`, as
    rows with the composed log q and log p. `sampler_id`, the checkpoint's, enters the seed."""
    seed = context.stream_seed(
        gfn_policy.METHOD, "composition-selection", sampler_id, setting.setting_id
    )
    draws = composed.sampler(setting).sample_draws(composition.SELECTION_DRAWS, seed=seed)
    responses = [d.token_ids for d in draws]
    logq, logp = composed.logprobs(responses, setting)
    texts = [context.original.decode(response) for response in responses]
    return [
        {
            "draw_index": index,
            "token_ids": response,
            "text": texts[index],
            "hit": is_hit(texts[index]),
            "logq": logq[index],
            "logp": logp[index],
            "seed": seed,
        }
        for index, response in enumerate(responses)
    ]


def threshold_test(
    context: run_context.RunContext,
    composed: ComposedPolicy,
    setting: composition.Setting,
    count: int,
    *,
    sampler_id: str,
    is_hit: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """`count` new responses of the same sampler for a threshold test, as rows."""
    seed = context.stream_seed(
        gfn_policy.METHOD, "composition-threshold-test", sampler_id, setting.setting_id
    )
    draws = composed.sampler(setting).sample_draws(count, seed=seed)
    texts = [context.original.decode(d.token_ids) for d in draws]
    return [
        {
            "draw_index": index,
            "token_ids": d.token_ids,
            "text": texts[index],
            "hit": is_hit(texts[index]),
            "logq": d.logq,
            "seed": seed,
        }
        for index, d in enumerate(draws)
    ]
