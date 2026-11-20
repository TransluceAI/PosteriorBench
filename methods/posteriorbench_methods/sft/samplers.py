"""SFT's samplers and their draws as rows of a draws file. The sampler is the trained model q
under the task's prompt, with whichever checkpoint's weights it holds."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from posteriorbench_methods.common import composition, hf_runtime, run_context, seeds

GENERATION_PART = 128  # responses generated in one forward pass
SCORING_BATCH = 32  # responses in one forward pass that computes log q of given responses


@dataclasses.dataclass(frozen=True)
class Models:
    """The model that a run trains (q) and the original model (p), under the task's prompt.
    `generation_rows` and `scoring_rows` are the responses in one generating and one
    teacher-forced pass: the module's constants, or fewer where the task needs smaller passes."""

    q: hf_runtime.CausalLM
    p: hf_runtime.CausalLM
    generation: hf_runtime.Generation
    generation_rows: int = GENERATION_PART
    scoring_rows: int = SCORING_BATCH

    @classmethod
    def of(cls, context: run_context.RunContext, trained_model: Any) -> Models:
        """The models of a run. `trained_model` is the run's copy of the original model."""
        prompt = tuple(context.task.prompt_token_ids)
        return cls(
            q=hf_runtime.CausalLM(trained_model, prompt),
            p=hf_runtime.CausalLM(context.original.model, prompt),
            generation=hf_runtime.Generation(hf_runtime.ResponseFormat.of(context.original)),
            generation_rows=context.generation_batch(GENERATION_PART),
            scoring_rows=context.scoring_batch(SCORING_BATCH),
        )


def _part_seed(seed: int, first_draw: int) -> int:
    return seeds.substream_seed(seed, first_draw)


def sampler(models: Models, setting: composition.Setting | None = None) -> hf_runtime.ModelSampler:
    """q with the weights its model holds when the sampler is used; under `setting`, q composed
    with p. log q is computed by teacher forcing; a composed sampler mixes the per-token values
    of the two models, also under the plain setting, where it is q itself."""
    q, p = models.q, models.p
    part, batch = models.generation_rows, models.scoring_rows
    if setting is None:
        return hf_runtime.ModelSampler(
            hf_runtime.TwoModels(q),
            lambda responses: hf_runtime.sequence_logprobs(q, responses, batch_size=batch),
            models.generation,
            part_size=part,
            part_seed=_part_seed,
            name="sft",
        )
    limit = models.generation.response_format.max_tokens
    reads_p = not setting.is_plain(limit)

    def composed_logprobs(responses: Sequence[Sequence[int]]) -> list[float]:
        q_tokens = hf_runtime.token_logprobs(q, responses, batch_size=batch)
        p_tokens = hf_runtime.token_logprobs(p, responses, batch_size=batch) if reads_p else None
        return hf_runtime.composed_logprobs(q_tokens, p_tokens, setting=setting)

    return hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q, p),
        composed_logprobs,
        models.generation,
        part_size=part,
        part_seed=_part_seed,
        setting=setting,
        name="sft+composition",
    )


def draw(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[dict[str, Any]]:
    """`count` responses of `source` as draw rows (draw_index, token_ids, hit, seed), labeled by
    `RunContext.label`."""
    responses = source.sample(count, seed=seed)
    hits = context.label(responses)
    return [
        {"draw_index": index, "token_ids": token_ids, "hit": hit, "seed": seed}
        for index, (token_ids, hit) in enumerate(zip(responses, hits, strict=True))
    ]


def selection_draws(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[dict[str, Any]]:
    """`count` selection draws of `source`: the rows of `draw` with logq, the sampler's log q of
    the response, and logp, the original model's log p of a hit (None on a miss)."""
    rows = draw(context, source, count, seed=seed)
    logq = source.logprobs([row["token_ids"] for row in rows])
    hit_responses = [row["token_ids"] for row in rows if row["hit"]]
    logp = iter(
        context.original.logprobs(context.task.prompt_token_ids, hit_responses)
        if hit_responses
        else []
    )
    return [
        {**row, "logq": q, "logp": next(logp) if row["hit"] else None}
        for row, q in zip(rows, logq, strict=True)
    ]
