"""The draws of a Steering run. The replicates of a task share their selection draws: their seed
omits the replicate, and they are kept in `<out>/cache/steering/<task>/` for later replicates to
read. Threshold tests are drawn anew in each replicate (`ThresholdTestStreams`).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

from posteriorbench_methods.common import errors, hf_runtime, run_context, run_record, seeds
from posteriorbench_methods.steering import samplers

CACHE_SOURCE_FILE = "source.json"
SELECTION = "selection"  # the stream of a candidate's selection draws
COMPOSITION_SELECTION = "composition-selection"  # the stream of a composition setting's


class CacheError(errors.MethodsError, RuntimeError):
    """The task's cache directory holds selection draws of another model, other vectors or
    another number of selection draws."""


def label(context: run_context.RunContext, responses: Sequence[Sequence[int]]) -> list[bool]:
    """`RunContext.label`, except that a response with a token beyond the tokenizer's vocabulary
    is a miss: the output layer can be larger than the vocabulary, and such a token has no text."""
    hits = context.label(responses)
    vocabulary = len(context.original.tokenizer)
    return [
        hit and max(response) < vocabulary for response, hit in zip(responses, hits, strict=True)
    ]


def draw(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[dict[str, Any]]:
    """`count` responses of `source` as draw rows, generated as one batch seeded with `seed`. On a
    task whose forward passes hold fewer (`RunContext.generation_batch`), they are generated in
    parts, each seeded from `seed` and the part's index."""
    part = context.generation_batch(count)
    groups = [hf_runtime.DrawGroup(count, seed)]
    if part < count:
        groups = hf_runtime.parts(
            count,
            seed,
            part_size=part,
            part_seed=lambda of_call, first: seeds.substream_seed(of_call, "part", first // part),
        )
    drawn = hf_runtime.sample_groups(
        source.models, groups, source.generation, setting=source.setting
    )
    responses = [one.token_ids for group in drawn for one in group]
    hits = label(context, responses)
    return [
        {"draw_index": index, "token_ids": token_ids, "hit": hit, "seed": seed}
        for index, (token_ids, hit) in enumerate(zip(responses, hits, strict=True))
    ]


def selection_draws(
    context: run_context.RunContext, source: hf_runtime.ModelSampler, count: int, *, seed: int
) -> list[dict[str, Any]]:
    """The rows of `draw` with the sampler's logq, and the original model's logp on hits (None
    on a miss)."""
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


def check_cache(context: run_context.RunContext, count: int) -> None:
    """Write `source.json` (model, method data, `count`) in a new cache directory; raise
    CacheError if an existing one was written by a run that differs in any of them."""
    method_data = context.options.method_data
    source = {
        "model": context.original.name,
        "model_revision": context.original.revision,
        "method_data": None if method_data is None else str(method_data.resolve()),
        "selection_draws": count,
    }
    path = context.cache_dir() / CACHE_SOURCE_FILE
    if not path.exists():
        run_record.write_json(path, source)
    elif run_record.read_json(path) != source:
        raise CacheError(
            f"{path.parent} holds the selection draws of a run with another model, other vectors "
            "or another number of selection draws; use another output root or delete it"
        )


def shared_selection_draws(
    context: run_context.RunContext,
    stream: str,
    key: str,
    source: hf_runtime.ModelSampler,
    count: int,
) -> list[dict[str, Any]]:
    """The `count` selection draws of sampler `key`, shared by all replicates: read from
    `<cache>/<stream>/<key>.jsonl`, or drawn and written there."""
    path: pathlib.Path = context.cache_dir() / stream / f"{key}.jsonl"
    if path.exists():
        return run_record.read_jsonl(path)
    seed = context.stream_seed("steering", stream, key, shared_by_replicates=True)
    rows = selection_draws(context, source, count, seed=seed)
    run_record.write_jsonl(path, rows)
    return rows


class ThresholdTestStreams:
    """The threshold-test draws of one run: one stream per sampler, drawn a batch at a time, and
    a test of n draws reads its first n. A sampler tested twice is tested on the same responses:
    Steering's submitted candidate is tested again as the plain setting of its composition."""

    def __init__(self, context: run_context.RunContext) -> None:
        self.context = context
        self._streams: dict[str, list[dict[str, Any]]] = {}

    def first(self, key: str, source: hf_runtime.ModelSampler, count: int) -> list[dict[str, Any]]:
        """The first `count` rows of the stream of sampler `key` (`source`), drawing what is
        missing."""
        stream = self._streams.setdefault(key, [])
        while len(stream) < count:
            part = len(stream) // samplers.GENERATION_PART
            seed = self.context.stream_seed("steering", "threshold-test", key, part)
            stream.extend(draw(self.context, source, samplers.GENERATION_PART, seed=seed))
        return stream[:count]
