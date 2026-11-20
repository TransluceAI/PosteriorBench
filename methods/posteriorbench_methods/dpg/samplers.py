"""The samplers of KL-DPG, JS-DPG and GDC, and their draws as rows of a draws file. A sampler is a
trained copy of the original model under the task's prompt, with whichever checkpoint's weights it
holds.

The numbers of responses in one forward pass are fixed here. They do not change a method, but with
the seeds they determine a sampler's draws, and the rounding of a batched pass depends on them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from posteriorbench import protocol

from posteriorbench_methods.common import composition, hf_runtime, run_context, seeds

GENERATION_PART = 128  # responses generated in one forward pass
SCORING_BATCH = 32  # responses in one forward pass over given responses (log q, log p, training)
TEST_PART = 32  # responses that a threshold test generates in one forward pass
# The sampler that GDC returns generates and scores in smaller passes.
GDC_RETURNED_GENERATION_PART = 32
GDC_RETURNED_SCORING_BATCH = 8

Row = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Models:
    """A model that a method trains (q) and the original model (p), both under the task's prompt,
    and how responses are drawn from them."""

    q: hf_runtime.CausalLM
    p: hf_runtime.CausalLM
    generation: hf_runtime.Generation

    @classmethod
    def of(cls, context: run_context.RunContext, model: Any) -> Models:
        """The models of a run. `model` is a copy of the original model that the run trains."""
        prompt = tuple(context.task.prompt_token_ids)
        return cls(
            q=hf_runtime.CausalLM(model, prompt),
            p=hf_runtime.CausalLM(context.original.model, prompt),
            generation=hf_runtime.Generation(hf_runtime.ResponseFormat.of(context.original)),
        )

    def with_model(self, model: Any) -> Models:
        """The same with another trained model as q (GDC trains two)."""
        return dataclasses.replace(self, q=hf_runtime.CausalLM(model, self.q.prompt_ids))


def _part_seed(seed: int, first_draw: int) -> int:
    return seeds.substream_seed(seed, first_draw)


def _test_part_seed(seed: int, first_draw: int) -> int:
    """The seed of a part of a threshold test: that of the first part of a call with the seed
    `seed + first_draw`."""
    return seeds.substream_seed(seed + first_draw, 0)


def sampler(
    models: Models,
    *,
    generation_part: int = GENERATION_PART,
    scoring_batch: int = SCORING_BATCH,
    name: str = "policy",
) -> hf_runtime.ModelSampler:
    """q with the weights that its model holds when the sampler is used. log q of a response is
    computed by teacher forcing."""
    q = models.q
    return hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q),
        lambda responses: hf_runtime.sequence_logprobs(q, responses, batch_size=scoring_batch),
        models.generation,
        part_size=generation_part,
        part_seed=_part_seed,
        name=name,
    )


def returned_sampler(models: Models, method: str) -> hf_runtime.ModelSampler:
    """The sampler that `method` returns to the evaluation: q with the weights of the submitted
    checkpoint, which its model holds."""
    if method == "gdc":
        return sampler(
            models,
            generation_part=GDC_RETURNED_GENERATION_PART,
            scoring_batch=GDC_RETURNED_SCORING_BATCH,
            name=method,
        )
    return sampler(models, name=method)


def test_sampler(models: Models) -> hf_runtime.ModelSampler:
    """q as a threshold test draws from it: in parts of 32 responses, each with a seed of its
    own."""
    q = models.q
    return hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q),
        lambda responses: hf_runtime.sequence_logprobs(q, responses, batch_size=SCORING_BATCH),
        models.generation,
        part_size=TEST_PART,
        part_seed=_test_part_seed,
        name="threshold-test",
    )


def composed_sampler(
    models: Models, setting: composition.Setting, *, name: str
) -> hf_runtime.ModelSampler:
    """q composed with p under `setting`. log q of a response mixes the per-token values of the
    two models, also under the plain setting, where the sampler is q itself and p is not run."""
    q, p = models.q, models.p
    reads_p = not setting.is_plain(models.generation.response_format.max_tokens)

    def composed_logprobs(responses: Sequence[Sequence[int]]) -> list[float]:
        q_tokens = hf_runtime.token_logprobs(q, responses, batch_size=SCORING_BATCH)
        p_tokens = (
            hf_runtime.token_logprobs(p, responses, batch_size=SCORING_BATCH) if reads_p else None
        )
        return hf_runtime.composed_logprobs(q_tokens, p_tokens, setting=setting)

    return hf_runtime.ModelSampler(
        hf_runtime.TwoModels(q, p),
        composed_logprobs,
        models.generation,
        part_size=GENERATION_PART,
        part_seed=_part_seed,
        setting=setting,
        name=name,
    )


def log_p(context: run_context.RunContext, responses: Sequence[Sequence[int]]) -> list[float]:
    """log p of each response under the original model, in batches of `SCORING_BATCH` responses
    taken in the order given."""
    prompt = context.task.prompt_token_ids
    values: list[float] = []
    for start in range(0, len(responses), SCORING_BATCH):
        values += context.original.logprobs(prompt, responses[start : start + SCORING_BATCH])
    return values


def training_draws(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[Row]:
    """`count` responses of `source` as draw rows: draw_index, token_ids, hit, seed, and
    logq_generation, the log-probability with which the sampler generated the response (the sum
    of the per-token values that generation records)."""
    drawn = source.sample_draws(count, seed=seed)
    hits = protocol.label(context.task, context.original, [one.token_ids for one in drawn])
    return [
        {
            "draw_index": index,
            "token_ids": one.token_ids,
            "hit": hit,
            "seed": seed,
            "logq_generation": one.logq,
        }
        for index, (one, hit) in enumerate(zip(drawn, hits, strict=True))
    ]


def draw(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[Row]:
    """`count` responses of `source` as draw rows: draw_index, token_ids, hit, seed."""
    rows = training_draws(context, source, count, seed=seed)
    return [{key: row[key] for key in ("draw_index", "token_ids", "hit", "seed")} for row in rows]


def selection_draws(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[Row]:
    """`count` selection draws of `source`: the rows of `draw` with logp, the original model's
    log p of the response, and logq, the sampler's log q of it, for hits and misses."""
    rows = draw(context, source, count, seed=seed)
    responses = [row["token_ids"] for row in rows]
    logp = log_p(context, responses)
    logq = source.logprobs(responses)
    return [{**row, "logq": q, "logp": p} for row, q, p in zip(rows, logq, logp, strict=True)]
