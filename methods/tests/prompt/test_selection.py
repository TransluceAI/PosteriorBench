"""Ranking and testing a pool, with selection draws and threshold tests that need no model."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from posteriorbench_methods.common import run_context, run_record, signals
from posteriorbench_methods.prompt import (
    draws,
    methods,
    pool,
    run,
    samplers,
    selection,
    selection_signals,
    settings,
    shared,
)

M = 16


def member(index: int, hits: int = 0, mean_weight: float = -10.0) -> selection.Member:
    """Candidate `index` with `hits` hits among its selection draws, each with log p - log q =
    `mean_weight`, and a prompt of its own."""
    hit = np.arange(M) < hits
    logq = np.where(hit, -5.0, np.nan)
    values = signals.SelectionDraws(hit, logq, logq + mean_weight)
    drawn = draws.LabeledDraws(tuple((index, d, 2) for d in range(M)), values, seed=0)
    spec = samplers.Spec(((1, 2, 10 + index),))
    candidate = pool.Candidate(index, f"c{index}", f"prompt {index}")
    return selection.Member(candidate, spec, drawn, selection_signals.statistics(values))


class FakeDraws:
    """Threshold tests without a model: every draw of a sampler in `passing` is a hit, and a test
    of another sampler returns no rows, which counts as no hit."""

    def __init__(self, passing: set[str]) -> None:
        self.passing = passing

    def threshold_test(self, spec: samplers.Spec, draws: int) -> list[dict[str, Any]]:
        if spec.sampler_id not in self.passing:
            return []
        return [{"draw_index": i, "token_ids": [5], "hit": True, "seed": 0} for i in range(draws)]


def session(context: run_context.RunContext, passing: set[str], **changes: Any) -> shared.Session:
    factory = samplers.Factory(context.original, context.task.prompt_token_ids)
    return shared.Session(
        context, settings.Settings(**changes), factory, cast(draws.Draws, FakeDraws(passing))
    )


def test_candidates_above_the_hit_fraction_come_first_by_score() -> None:
    members = [member(0, 2), member(1, 5, -12.0), member(2, 0), member(3, 3, -9.0), member(4, 1)]
    scores = selection_signals.scores(
        "jc", [m.statistics for m in members], signal_hit_fraction=0.15
    )
    ranking = selection.rank(members, scores)
    assert [(e.candidate_id, e.group) for e in ranking] == [
        ("3", "by_signal"),
        ("1", "by_signal"),
        ("0", "by_hits"),
        ("4", "by_hits"),
        ("2", "by_hits"),
    ]
    # Equal scores and equal hits go to the candidate that comes first in the pool.
    tied = [member(7, 4), member(3, 4), member(9, 1), member(5, 1)]
    ranking = selection.rank(tied, [s.statistics.jc for s in tied])
    assert [e.candidate_id for e in ranking] == ["7", "3", "9", "5"]


def test_the_candidates_of_a_batch_follow_those_of_the_batches_before_it(
    make_context: Callable[..., run_context.RunContext],
    make_pool_file: Callable[..., pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    """Batch 0 of a pool file is ordered as a file that holds only its candidates, so the pools
    within it are the same from either file; the later batches follow, each in an order of its
    own."""
    prompts = tuple(f"say {index}" for index in range(40))
    first = make_pool_file(prompts[:24])
    batched = make_pool_file(prompts, batches=(0,) * 24 + (1,) * 10 + (2,) * 6)

    def order(name: str, pool_file: pathlib.Path, largest: int, replicate: int = 0) -> list[int]:
        context = make_context(tmp_path / name, replicate=replicate, method_data=pool_file)
        return [c.index for c in selection.ordered_candidates(context, largest)]

    whole = order("batched", batched, 40)
    assert whole[:24] == order("first", first, 24)
    assert whole[:30] == order("smaller", batched, 30)
    assert sorted(whole[24:34]) == list(range(24, 34)) and whole[24:34] != list(range(24, 34))
    assert sorted(whole[34:]) == list(range(34, 40))
    assert order("replicate1", batched, 40, replicate=1) != whole


def test_a_pool_beyond_1024_candidates_is_ranked_and_tested(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """A pool may be as large as the pool file, and the tests go on until a candidate passes: here
    the candidate at rank 1,030 of 2,048, beyond the last rank of the shipped table."""
    members = [member(index) for index in range(2048)]
    passing = {members[1029].spec.sampler_id}
    ranking = selection.rank(members, [None] * len(members))
    assert [e.rank for e in ranking] == list(range(1, 2049))
    result, rows = selection.run_tests(
        session(make_context(tmp_path), passing), members, ranking, selection.RULE
    )
    assert result.status == "passed" and len(result.tests) == 1030
    assert result.submitted is not None and result.submitted.candidate_id == "1029"
    assert result.submitted.draws >= result.tests[1023].draws == 576
    assert len(rows) == result.submitted.draws
    assert all(row["rank"] == 1030 and row["candidate_id"] == "1029" for row in rows)


def test_a_mixture_collects_the_candidates_that_pass(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    members = [member(index, hits=4 + index % 3, mean_weight=-10.0 - index) for index in range(40)]
    passing = {members[index].spec.sampler_id for index in range(0, 40, 2)}
    scores = selection_signals.ebjc_scores(
        [m.statistics for m in members], signal_hit_fraction=0.15
    )
    ranking = selection.rank(members, scores)
    fake = session(make_context(tmp_path), passing)
    sixteen, _ = selection.run_tests(
        fake, members, ranking, dataclasses.replace(selection.RULE, passes_needed=16)
    )
    assert len(sixteen.passes) == 16 and all(int(t.candidate_id) % 2 == 0 for t in sixteen.passes)
    assert [t.rank for t in sixteen.passes] == sorted(t.rank for t in sixteen.passes)
    assert sixteen.tests[-1] == sixteen.passes[-1], "testing stops at the sixteenth pass"
    everything, _ = selection.run_tests(
        fake, members, ranking, dataclasses.replace(selection.RULE, passes_needed=64)
    )
    assert len(everything.passes) == 20 and len(everything.tests) == 40


def test_ce_without_a_hit_in_the_bank_has_no_sampler(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """One candidate of 1,024 has 3 hits and is to be scored; the bank holds 512 of the 16,384
    selection draws and usually none of the 3 hits. The run is then recorded as a run without a
    sampler, at the cost of its selection draws, and is not composed."""
    members = [member(0, hits=3), *(member(index) for index in range(1, 1024))]
    for replicate in range(50):
        fake = session(make_context(tmp_path / f"rep{replicate}", replicate=replicate), set())
        returned = selection.select_by_ce(fake, members)
        if returned.spec is None:
            break
    else:
        raise AssertionError("every replicate had a hit in the bank")
    assert returned.status == "no_sampler" and returned.recorded.shipped.fallback
    assert returned.recorded.shipped.construction_inferences == 16 * 1024
    construction = run_record.read_json(returned.recorded.directory / "construction.json")
    assert construction["status"] == "no_sampler" and construction["ranking"] is None
    assert construction["details"]["bank"]["hits"] == 0
    assert construction["details"]["bank"]["scored_candidates"] == ["0"]
    composed = run.compose(fake, returned)
    assert composed.shipped.fallback
    assert composed.shipped.construction_inferences == 16 * 1024
    record = run_record.read_json(composed.directory / "construction.json")
    assert record["status"] == "not_composed"
    assert record["details"]["plain"] == {"status": "no_sampler", "submitted": None}
    assert record["details"]["composition"] == {"reason": "plain_not_passed"}
    assert methods.parse(composed.summary.method) == (methods.Selection("ce"), True)
