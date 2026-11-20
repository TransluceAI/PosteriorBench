"""The submission rule: `Rule`, `rank_candidates`, `rank_from_selection_draws`, `submit`."""

from __future__ import annotations

import numpy as np
import pytest

from posteriorbench_methods.common import rank_table, signals, submission

TRAINING = submission.Rule(max_tests=submission.TRAINING_MAX_TESTS)
IN_ORDER = submission.Rule(max_tests=None, ties="order")


def candidate(
    name: str, order: int, hits: int, score: float | None = None, *, sampler: str | None = None,
    tie: float | None = None, draws: int = 16,
) -> submission.Candidate:  # fmt: skip
    return submission.Candidate(name, order, sampler or name, draws, hits, score, tie)


def test_groups_orders_and_duplicates() -> None:
    candidates = [
        candidate("a", 0, 0),
        candidate("b", 1, 5, 1.0),
        candidate("c", 2, 2, tie=0.5),
        candidate("d", 3, 4, 2.0),  # the highest score: first
        candidate("e", 4, 2, tie=1.0),
        candidate("f", 5, 5, 1.0, sampler="b"),  # the same sampler as b, ranked after it
        candidate("g", 6, 1, tie=9.0),
    ]
    ranking = submission.rank_candidates(candidates, TRAINING)
    assert [(e.rank, e.candidate_id, e.group) for e in ranking] == [
        (1, "d", "by_signal"),
        (2, "b", "by_signal"),
        (3, "e", "by_hits"),  # 2 hits, a higher tie score than c
        (4, "c", "by_hits"),
        (5, "g", "by_hits"),
        (6, "a", "by_hits"),  # no hit: last, in order
    ]
    assert ranking[1].aliases == ("f",)
    in_order = submission.rank_candidates(
        candidates, submission.Rule(max_tests=None, ties="order", min_hits=1)
    )
    assert [e.candidate_id for e in in_order] == ["d", "b", "c", "e", "g"]


def test_ties_between_equal_scores() -> None:
    candidates = [candidate("x", 0, 4, 1.0), candidate("y", 1, 6, 1.0)]
    assert [e.candidate_id for e in submission.rank_candidates(candidates, TRAINING)] == ["y", "x"]
    assert [e.candidate_id for e in submission.rank_candidates(candidates, IN_ORDER)] == ["x", "y"]


def test_invalid_candidates_raise() -> None:
    with pytest.raises(submission.SubmissionError, match="finite score"):
        submission.rank_candidates([candidate("a", 0, 8)], TRAINING)
    with pytest.raises(submission.SubmissionError, match="unique"):
        submission.rank_candidates([candidate("a", 0, 0), candidate("a", 1, 0)], TRAINING)
    with pytest.raises(submission.SubmissionError, match="tie score"):
        submission.rank_candidates([candidate("a", 0, 1)], TRAINING)


def test_invalid_rules_raise() -> None:
    for values in (
        {"max_tests": 0},
        {"max_tests": 8, "threshold": 0.05},
        {"max_tests": 8, "signal_hit_fraction": 1.0},
        {"max_tests": 8, "passes_needed": 0},
    ):
        with pytest.raises(submission.SubmissionError):
            submission.Rule(**values)  # pyright: ignore[reportArgumentType]
    one_percent = submission.Rule(max_tests=8, threshold=0.01, signal_hit_fraction=0.015)
    assert one_percent.threshold == 0.01


def random_draws(rng: np.random.Generator, m: int, rate: float) -> signals.SelectionDraws:
    hit = rng.random(m) < rate
    logq = rng.normal(-20.0, 3.0, m)
    logp = np.where(hit, rng.normal(-25.0, 3.0, m), np.nan)
    return signals.SelectionDraws(hit, logq, logp)


def test_rank_from_selection_draws_and_representatives() -> None:
    rng = np.random.default_rng(0)
    candidates = [
        submission.CandidateDraws(f"c{i}", i, f"sampler{i}", random_draws(rng, 128, 0.3))
        for i in range(6)
    ]
    # With distinct samplers both choices of the representative give the same ranking.
    best = submission.rank_from_selection_draws(candidates, TRAINING)
    earliest = submission.rank_from_selection_draws(
        candidates, submission.Rule(max_tests=8, representative="earliest")
    )
    assert [e.candidate_id for e in best] == [e.candidate_id for e in earliest]
    assert all(e.group == "by_signal" for e in best)
    # The score is EB-JC, by which the candidates are ranked, and the tie score is JC.
    scores = [e.score for e in best if e.score is not None]
    assert len(scores) == 6 and scores == sorted(scores, reverse=True)
    jc = {c.candidate_id: signals.jc_score(c.draws) for c in candidates}
    assert [e.tie_score for e in best] == [jc[e.candidate_id] for e in best]
    assert scores != [jc[e.candidate_id] for e in best]
    # Two candidates that are one sampler: one entry, the other its alias.
    shared = [
        *candidates,
        submission.CandidateDraws("c6", 6, "sampler0", random_draws(rng, 128, 0.3)),
    ]
    ranking = submission.rank_from_selection_draws(shared, TRAINING)
    assert len(ranking) == 6
    assert sum(len(e.aliases) for e in ranking) == 1
    unequal = [*candidates, submission.CandidateDraws("c7", 7, "s7", random_draws(rng, 16, 0.3))]
    with pytest.raises(submission.SubmissionError, match="unequal"):
        submission.rank_from_selection_draws(unequal, TRAINING)


def test_threshold_test_reserve_is_the_draws_of_the_first_tests() -> None:
    draws = [rank_table.threshold_test(rank).draws for rank in range(1, 9)]
    assert submission.TRAINING_MAX_TESTS == 8
    assert sum(draws) == submission.THRESHOLD_TEST_RESERVE == 1912


def ranking_of(n: int) -> list[submission.RankedCandidate]:
    return submission.rank_candidates(
        [candidate(f"c{i}", i, 8, float(n - i)) for i in range(n)], IN_ORDER
    )


def test_submit_first_pass_best_observed_and_passes() -> None:
    hits = {"c0": 20, "c1": 40, "c2": 50, "c3": 60}
    asked: list[tuple[str, int]] = []

    def test(entry: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest) -> int:
        asked.append((entry.candidate_id, threshold_test.draws))
        return hits[entry.candidate_id]

    result = submission.submit(ranking_of(4), test, TRAINING)
    assert result.status == "passed" and result.submitted is not None
    assert result.submitted.candidate_id == "c1"
    assert asked == [("c0", 164), ("c1", 198)]
    assert result.test_draws == 362
    hits = {"c0": 24, "c1": 25, "c2": 30, "c3": 35}
    asked.clear()
    failed = submission.submit(ranking_of(4), test, submission.Rule(max_tests=3))
    assert failed.status == "best_observed" and failed.submitted is not None
    assert [t.candidate_id for t in failed.tests] == ["c0", "c1", "c2"]
    assert failed.submitted.candidate_id == "c0"  # 24/164 is the highest hit fraction
    hits = {"c0": 40, "c1": 10, "c2": 50, "c3": 60}
    several = submission.submit(
        ranking_of(4), test, submission.Rule(max_tests=None, passes_needed=2)
    )
    assert [t.candidate_id for t in several.passes] == ["c0", "c2"]
    empty = submission.submit([], test, TRAINING)
    assert empty.status == "no_candidate" and empty.submitted is None
    assert set(result.to_json()) == {
        "status", "submitted", "passes", "tests", "test_draws", "ranking_length", "max_tests",
        "passes_needed",
    }  # fmt: skip


def test_submit_tests_at_the_rules_threshold() -> None:
    asked: list[int] = []

    def test(entry: submission.RankedCandidate, threshold_test: rank_table.ThresholdTest) -> int:
        asked.append(threshold_test.draws)
        return 0

    rule = submission.Rule(max_tests=2, threshold=0.01, signal_hit_fraction=0.015)
    submission.submit(ranking_of(3), test, rule)
    assert asked == [1857, 2315]


def test_submit_rejects_impossible_counts() -> None:
    with pytest.raises(submission.SubmissionError):
        submission.submit(ranking_of(2), lambda e, t: t.draws + 1, TRAINING)


def test_a_rule_at_one_percent_has_a_tenth_of_its_thresholds_and_at_most_16_tests() -> None:
    until_one_passes = submission.Rule(max_tests=None, min_hits=1, ties="order")
    at_1 = submission.at_one_percent(until_one_passes)
    assert (at_1.threshold, at_1.signal_hit_fraction, at_1.max_tests) == (0.01, 0.015, 16)
    assert (at_1.min_hits, at_1.ties) == (1, "order")
    # A rule with a number of tests of its own keeps it, and so does a mixture its passes.
    assert submission.at_one_percent(submission.Rule(max_tests=8)).max_tests == 8
    mixture = submission.at_one_percent(submission.Rule(max_tests=None, passes_needed=16))
    assert (mixture.max_tests, mixture.passes_needed) == (16, 16)
    with pytest.raises(submission.SubmissionError):
        submission.at_one_percent(at_1)
    # Twenty candidates that all fail: the tests stop after the sixteenth, and the best-observed
    # one is submitted.
    candidates = [
        submission.Candidate(f"c{i}", i, f"s{i}", 16, 4, score=float(-i)) for i in range(20)
    ]
    ranking = submission.rank_candidates(candidates, at_1)
    result = submission.submit(ranking, lambda candidate, test: 20 - candidate.rank, at_1)
    assert len(result.tests) == 16 and result.status == "best_observed"
    assert result.submitted is not None and result.submitted.candidate_id == "c0"
