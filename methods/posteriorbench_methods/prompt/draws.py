"""The draws the prompt family makes to choose a sampler.

Selection draws of a candidate (m responses, log q and log p on hits), one threshold-test stream
per sampler (the test at rank j reads its first n_j responses), and 16 selection draws per
composition setting. Every stream has its own seed, so a sampler's draws do not depend on which
method asks for them or in what order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from posteriorbench import sampler

from posteriorbench_methods.common import composition, run_context, signals
from posteriorbench_methods.prompt import pool, samplers

OWNER = "prompt"  # the methods of the family share their streams
# A threshold-test stream is generated in parts of 32 responses, each with its own seed.
TEST_PART_SIZE = 32


@dataclasses.dataclass(frozen=True)
class LabeledDraws:
    """The responses of one `sample` call, in the order drawn, with their labels. `values` are
    arrays [draws]; log q and log p are NaN on a miss."""

    token_ids: tuple[tuple[int, ...], ...]
    values: signals.SelectionDraws
    seed: int

    def rows(self, **labels: Any) -> list[dict[str, Any]]:
        """One draw-file row per draw, led by `labels`; logq and logp are None on a miss."""
        rows: list[dict[str, Any]] = []
        for index, token_ids in enumerate(self.token_ids):
            hit = bool(self.values.hit[index])
            rows.append(
                {
                    **labels,
                    "draw_index": index,
                    "token_ids": list(token_ids),
                    "hit": hit,
                    "logq": float(self.values.logq[index]) if hit else None,
                    "logp": float(self.values.logp[index]) if hit else None,
                    "seed": self.seed,
                }
            )
        return rows


class Draws:
    """The draws of one run. Threshold-test streams and composition selection draws are drawn on
    first use and kept, so methods that test or compose the same sampler read the same draws."""

    def __init__(self, context: run_context.RunContext, factory: samplers.Factory) -> None:
        self.context = context
        self.factory = factory
        self._test_streams: dict[str, list[dict[str, Any]]] = {}
        self._composition: dict[tuple[str, str, bool], LabeledDraws] = {}

    def _label(
        self, candidate: sampler.Sampler, responses: Sequence[Sequence[int]], seed: int
    ) -> LabeledDraws:
        """Hit flags of `responses`, and log q and log p computed on the hits only."""
        original = self.context.original
        hit = np.asarray(self.context.label(responses), dtype=bool)
        logq = np.full(len(responses), np.nan)
        logp = np.full(len(responses), np.nan)
        hit_responses = [responses[index] for index in np.flatnonzero(hit)]
        if hit_responses:
            logq[hit] = candidate.logprobs(hit_responses)
            logp[hit] = original.logprobs(self.factory.task_prompt, hit_responses)
        token_ids = tuple(tuple(int(token) for token in response) for response in responses)
        return LabeledDraws(token_ids, signals.SelectionDraws(hit, logq, logp), seed)

    def selection(self, candidate: pool.Candidate, spec: samplers.Spec, draws: int) -> LabeledDraws:
        """`draws` selection draws of a candidate. The seed depends only on the candidate's index
        in the pool file, so every pool of the replicate reads the same draws for it."""
        seed = self.context.stream_seed(OWNER, "selection", candidate.index)
        drawn = self.factory.build(spec, name=candidate.candidate_id)
        return self._label(drawn, drawn.sample(draws, seed=seed), seed)

    def threshold_test(self, spec: samplers.Spec, draws: int) -> list[dict[str, Any]]:
        """The first `draws` rows of the sampler's threshold-test stream. The stream grows in
        seeded parts of TEST_PART_SIZE, so a longer test extends a shorter one."""
        stream = self._test_streams.setdefault(spec.sampler_id, [])
        if len(stream) < draws:
            tested = self.factory.build(spec, name="threshold-test")
            while len(stream) < draws:
                part = len(stream) // TEST_PART_SIZE
                seed = self.context.stream_seed(OWNER, "threshold-test", spec.sampler_id, part)
                responses = tested.sample(TEST_PART_SIZE, seed=seed)
                labels = self.context.label(responses)
                for response, hit in zip(responses, labels, strict=True):
                    stream.append(
                        {
                            "draw_index": len(stream),
                            "token_ids": [int(token) for token in response],
                            "hit": bool(hit),
                            "seed": seed,
                        }
                    )
        return stream[:draws]

    def composition_selection(
        self, base: samplers.Spec, setting: composition.Setting, *, shared_by_replicates: bool
    ) -> LabeledDraws:
        """composition.SELECTION_DRAWS selection draws of `base` composed under `setting`. With
        `shared_by_replicates`, every replicate reads the same stream."""
        key = (base.sampler_id, setting.setting_id, shared_by_replicates)
        if key not in self._composition:
            seed = self.context.stream_seed(
                OWNER,
                "composition-selection",
                base.sampler_id,
                setting.setting_id,
                shared_by_replicates=shared_by_replicates,
            )
            composed = self.factory.build(base.composed(setting), name="composition-selection")
            responses = composed.sample(composition.SELECTION_DRAWS, seed=seed)
            self._composition[key] = self._label(composed, responses, seed)
        return self._composition[key]


def count_hits(rows: Sequence[dict[str, Any]]) -> int:
    """The number of hits among draw rows."""
    return sum(bool(row["hit"]) for row in rows)
