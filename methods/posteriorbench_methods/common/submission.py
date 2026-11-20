"""The submission rule: how a method chooses the one sampler it submits among its candidates.

1. Every candidate has selection draws: m responses of the candidate, with h hits among them.
2. The candidates are ranked in two groups (`rank_candidates`; `rank_from_selection_draws` also
   computes the selection signal, EB-JC). A candidate whose hit fraction h / m is above 15% is
   ranked by the signal; the others follow in decreasing order of hits.
3. The candidates are tested against the hit-rate threshold in rank order (`submit`). The threshold
   test at rank j (`rank_table.threshold_test`) draws n_j new responses and passes with at least k_j
   hits. The first candidate that passes is submitted.
4. If none passes, the tested candidate with the highest hit fraction in its test is submitted, the
   best-observed candidate.

Candidates that are one sampler (the same weights, prompt or setting: equal `sampler_id`) are ranked
once. The one that comes first keeps its rank, and the later ones are listed as its `aliases`.

A method states its settings of the rule once, as a `Rule`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, Literal

from posteriorbench_methods.common import rank_table, signals

# A candidate whose hit fraction among its selection draws is above this is ranked by the signal.
SIGNAL_HIT_FRACTION = 0.15
# A method that trains under a budget tests at most its first 8 candidates, so that it can set the
# cost of its tests aside: the tests at ranks 1 to 8 draw 164 + 198 + ... + 282 = 1,912 responses.
TRAINING_MAX_TESTS = 8
THRESHOLD_TEST_RESERVE = 1912
# At the 1% threshold both thresholds of the rule, the tests' and the signal hit fraction, are a
# tenth of the benchmark's. A rule without its own `max_tests` then tests at most 16 candidates,
# the length of the table of 1% tests.
ONE_PERCENT = 0.01
ONE_PERCENT_SIGNAL_HIT_FRACTION = 0.015
ONE_PERCENT_MAX_TESTS = 16

Group = Literal["by_signal", "by_hits"]
Status = Literal["passed", "best_observed", "no_candidate"]
Ties = Literal["hits_and_tie_score", "order"]
Representative = Literal["best_jc", "earliest"]


class SubmissionError(ValueError):
    """Candidates, selection draws, rule settings or test results that the rule cannot use."""


@dataclasses.dataclass(frozen=True)
class Rule:
    """A method's settings of the submission rule.

    max_tests: the most candidates tested, in rank order; None for no limit.
    threshold: the hit-rate threshold of the tests, 0.10 (the benchmark's) or 0.01.
    signal_hit_fraction: candidates above this hit fraction are ranked by the signal, the others
        by hits after them.
    min_hits: a candidate ranked by hits with fewer selection hits is left out of the ranking.
    ties: "hits_and_tie_score": equal scores go to more selection hits, equal hits to the higher
        `tie_score`, and what remains to the lower `order`. "order": every tie goes to the lower
        `order`.
    representative: whose selection draws stand for a sampler of several candidates in the EB-JC
        fit. "best_jc": the candidate that a ranking by JC puts first; the others become its
        aliases. "earliest": the lowest `order`; every candidate of the sampler gets its score.
        With distinct samplers the two differ only in floating-point rounding (the fit reads the
        candidates in another order).
    passes_needed: testing stops when this many candidates have passed; a mixture collects its
        components with a value above 1."""

    max_tests: int | None
    threshold: float = 0.10
    signal_hit_fraction: float = SIGNAL_HIT_FRACTION
    min_hits: int = 0
    ties: Ties = "hits_and_tie_score"
    representative: Representative = "best_jc"
    passes_needed: int = 1

    def __post_init__(self) -> None:
        if self.max_tests is not None and self.max_tests < 1:
            raise SubmissionError(f"max_tests is positive or None, got {self.max_tests}")
        if self.threshold not in rank_table.THRESHOLDS:
            raise SubmissionError(
                f"threshold tests exist for the thresholds {rank_table.THRESHOLDS}, "
                f"not {self.threshold}"
            )
        fraction = self.signal_hit_fraction
        if not (math.isfinite(fraction) and 0.0 <= fraction < 1.0):
            raise SubmissionError(f"signal_hit_fraction must lie in [0, 1), got {fraction!r}")
        if self.passes_needed < 1:
            raise SubmissionError("at least one pass is needed")


def at_one_percent(rule: Rule) -> Rule:
    """`rule`, a method's settings at the benchmark's threshold, moved to the 1% threshold (the
    `ONE_PERCENT` constants); its other settings stay. Raises SubmissionError for a rule that is
    not at the benchmark's threshold."""
    if (rule.threshold, rule.signal_hit_fraction) != (0.10, SIGNAL_HIT_FRACTION):
        raise SubmissionError("the rule to take to 1% has the benchmark's threshold")
    return dataclasses.replace(
        rule,
        threshold=ONE_PERCENT,
        signal_hit_fraction=ONE_PERCENT_SIGNAL_HIT_FRACTION,
        max_tests=ONE_PERCENT_MAX_TESTS if rule.max_tests is None else rule.max_tests,
    )


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One candidate sampler as the ranking sees it.

    candidate_id: unique (a checkpoint id, a prompt's index, a setting id).
    order: unique; the order of saving, of the pool or of the grid.
    sampler_id: candidates with equal ids are one sampler and are ranked once.
    score: the selection signal (higher is better); finite for a candidate ranked by the signal.
    tie_score: orders equal hits under `Rule.ties` = "hits_and_tie_score" (JC for checkpoints)."""

    candidate_id: str
    order: int
    sampler_id: str
    selection_draws: int
    selection_hits: int
    score: float | None = None
    tie_score: float | None = None

    @property
    def hit_fraction(self) -> float:
        """selection_hits / selection_draws."""
        return self.selection_hits / self.selection_draws


@dataclasses.dataclass(frozen=True)
class RankedCandidate:
    """One entry of a ranking: a candidate with its rank (from 1) and its group ("by_signal" or
    "by_hits"). `aliases` are the ids of the other candidates that are the same sampler, in the
    order in which they would have been ranked."""

    rank: int
    group: Group
    candidate_id: str
    order: int
    sampler_id: str
    selection_draws: int
    selection_hits: int
    score: float | None
    tie_score: float | None
    aliases: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {**dataclasses.asdict(self), "aliases": list(self.aliases)}


def _check_candidates(candidates: Sequence[Candidate], signal_hit_fraction: float) -> None:
    ids = [c.candidate_id for c in candidates]
    orders = [c.order for c in candidates]
    if len(set(ids)) != len(ids) or len(set(orders)) != len(orders):
        raise SubmissionError("candidate ids and orders must be unique")
    for c in candidates:
        if c.selection_draws <= 0 or not 0 <= c.selection_hits <= c.selection_draws:
            raise SubmissionError(f"{c.candidate_id}: invalid selection counts")
        if c.hit_fraction > signal_hit_fraction and (c.score is None or not math.isfinite(c.score)):
            raise SubmissionError(
                f"{c.candidate_id}: a candidate ranked by the signal needs a finite score"
            )


def _rank(
    candidates: Sequence[Candidate],
    *,
    signal_hit_fraction: float,
    min_hits: int,
    hits_break_score_ties: bool,
    tie_scores_break_hit_ties: bool,
) -> list[RankedCandidate]:
    """Step 2 of the rule, with the two kinds of tie-breaking as separate switches."""
    _check_candidates(candidates, signal_hit_fraction)
    by_signal = [c for c in candidates if c.hit_fraction > signal_hit_fraction]
    by_hits = [
        c
        for c in candidates
        if not c.hit_fraction > signal_hit_fraction and c.selection_hits >= min_hits
    ]

    def signal_key(c: Candidate) -> tuple[float, int, int]:
        assert c.score is not None
        hits = -c.selection_hits if hits_break_score_ties else 0
        return (-c.score, hits, c.order)

    def hits_key(c: Candidate) -> tuple[int, float, int]:
        tie = 0.0
        if tie_scores_break_hit_ties and c.selection_hits > 0:
            if c.tie_score is None:
                raise SubmissionError(
                    f"{c.candidate_id}: a candidate ranked by hits needs a tie score"
                )
            tie = -c.tie_score
        return (-c.selection_hits, tie, c.order)

    ordered: list[tuple[Candidate, Group]] = []
    for c in sorted(by_signal, key=signal_key):
        ordered.append((c, "by_signal"))
    for c in sorted(by_hits, key=hits_key):
        ordered.append((c, "by_hits"))
    first: dict[str, tuple[Candidate, Group]] = {}
    aliases: dict[str, list[str]] = {}
    for c, group in ordered:
        if c.sampler_id in first:
            aliases[c.sampler_id].append(c.candidate_id)
            continue
        first[c.sampler_id] = (c, group)
        aliases[c.sampler_id] = []
    return [
        RankedCandidate(
            rank=rank,
            group=group,
            candidate_id=c.candidate_id,
            order=c.order,
            sampler_id=c.sampler_id,
            selection_draws=c.selection_draws,
            selection_hits=c.selection_hits,
            score=c.score,
            tie_score=c.tie_score,
            aliases=tuple(aliases[c.sampler_id]),
        )
        for rank, (c, group) in enumerate(first.values(), start=1)
    ]


def rank_candidates(candidates: Sequence[Candidate], rule: Rule) -> list[RankedCandidate]:
    """Step 2 of the rule for candidates that carry their scores. Ranks are 1..len(ranking).
    Raises SubmissionError for candidates that break the invariants of `Candidate`, or lack a
    score or tie score that `rule` reads."""
    informed = rule.ties == "hits_and_tie_score"
    return _rank(
        candidates,
        signal_hit_fraction=rule.signal_hit_fraction,
        min_hits=rule.min_hits,
        hits_break_score_ties=informed,
        tie_scores_break_hit_ties=informed,
    )


@dataclasses.dataclass(frozen=True)
class CandidateDraws:
    """A candidate (`candidate_id`, `order` and `sampler_id` as in `Candidate`) with its selection
    draws."""

    candidate_id: str
    order: int
    sampler_id: str
    draws: signals.SelectionDraws


def rank_from_selection_draws(
    candidates: Sequence[CandidateDraws], rule: Rule
) -> list[RankedCandidate]:
    """Step 2 of the rule from the candidates' selection draws: `rank_candidates` with EB-JC as
    the score and JC as the tie score. EB-JC is fitted with one set of selection draws for each
    sampler (`rule.representative`).

    Raises SubmissionError for inconsistent draws (`signals.SelectionDrawsError`), and under
    "earliest" for a sampler whose earliest candidate is ranked by hits while a later one would
    be ranked by the signal."""
    if not candidates:
        return []
    sizes = {c.draws.hit.size for c in candidates}
    if len(sizes) != 1:
        raise SubmissionError("candidates of one ranking have unequal numbers of selection draws")
    (draws,) = sizes
    jc = {c.candidate_id: signals.jc_score(c.draws) for c in candidates}
    base = [
        Candidate(
            candidate_id=c.candidate_id,
            order=c.order,
            sampler_id=c.sampler_id,
            selection_draws=draws,
            selection_hits=int(c.draws.hit.sum()),
            score=jc[c.candidate_id],
            tie_score=jc[c.candidate_id],
        )
        for c in candidates
    ]
    by_id = {c.candidate_id: c for c in candidates}
    above = {c.candidate_id: c.hit_fraction > rule.signal_hit_fraction for c in base}
    unranked: dict[str, tuple[str, ...]] = {}  # sampler -> its candidates left out ("best_jc")
    if rule.representative == "best_jc":
        jc_ranking = _rank(
            base,
            signal_hit_fraction=rule.signal_hit_fraction,
            min_hits=0,
            hits_break_score_ties=True,
            tie_scores_break_hit_ties=rule.ties == "hits_and_tie_score",
        )
        fitted = [entry.candidate_id for entry in jc_ranking if entry.group == "by_signal"]
        unranked = {
            entry.sampler_id: entry.aliases for entry in jc_ranking if entry.group == "by_signal"
        }
        keep = set(fitted)
        remaining = [c for c in base if c.sampler_id not in unranked or c.candidate_id in keep]
    else:
        earliest: dict[str, Candidate] = {}
        for c in sorted(base, key=lambda c: c.order):
            earliest.setdefault(c.sampler_id, c)
        fitted = [c.candidate_id for c in earliest.values() if above[c.candidate_id]]
        fitted_samplers = {by_id[i].sampler_id for i in fitted}
        for c in base:
            if above[c.candidate_id] and c.sampler_id not in fitted_samplers:
                raise SubmissionError(
                    f"{c.candidate_id}: ranked by the signal, but the earliest candidate of its "
                    "sampler is not"
                )
        remaining = list(base)
    try:
        scores = signals.score_candidates(
            [by_id[i].draws for i in fitted], signal_hit_fraction=rule.signal_hit_fraction
        )
    except signals.SelectionDrawsError as error:
        raise SubmissionError(str(error)) from error
    if not all(scores.by_signal):
        raise SubmissionError("the EB-JC fit disagrees with the candidates ranked by the signal")
    ebjc = {by_id[i].sampler_id: score for i, score in zip(fitted, scores.ebjc, strict=True)}
    scored = [
        dataclasses.replace(c, score=ebjc[c.sampler_id]) if above[c.candidate_id] else c
        for c in remaining
    ]
    ranking = rank_candidates(scored, rule)
    # Under "best_jc" the other candidates of a sampler ranked by the signal were left out above;
    # list them as the entry's aliases, before the aliases the ranking itself found.
    return [
        (
            dataclasses.replace(entry, aliases=unranked[entry.sampler_id] + entry.aliases)
            if entry.sampler_id in unranked
            else entry
        )
        for entry in ranking
    ]


@dataclasses.dataclass(frozen=True)
class TestOutcome:
    """The threshold test of one ranked candidate: `hits` hits among `draws` new responses; passed
    if hits >= required_hits."""

    rank: int
    candidate_id: str
    sampler_id: str
    group: Group
    draws: int
    required_hits: int
    hits: int
    passed: bool

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Submission:
    """What the rule submits, and the tests that decided it.

    status: "passed", "best_observed" (candidates were tested and none passed) or "no_candidate"
        (nothing was ranked).
    submitted: the test of the first candidate that passed, else of the best-observed one
        (`best_observed_test`); None without a test.
    passes, tests: the tests that passed (at most `passes_needed`) and every test, in rank
        order."""

    status: Status
    submitted: TestOutcome | None
    passes: tuple[TestOutcome, ...]
    tests: tuple[TestOutcome, ...]
    ranking_length: int
    max_tests: int | None
    passes_needed: int

    @property
    def test_draws(self) -> int:
        """The responses drawn by all the tests."""
        return sum(t.draws for t in self.tests)

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict, with `test_draws`."""
        return {
            "status": self.status,
            "submitted": None if self.submitted is None else self.submitted.to_json(),
            "passes": [t.to_json() for t in self.passes],
            "tests": [t.to_json() for t in self.tests],
            "test_draws": self.test_draws,
            "ranking_length": self.ranking_length,
            "max_tests": self.max_tests,
            "passes_needed": self.passes_needed,
        }


def best_observed_test(tests: Sequence[TestOutcome]) -> TestOutcome | None:
    """The test with the highest hit fraction; ties go to more hits, then to the lower rank."""
    if not tests:
        return None
    return max(tests, key=lambda t: (Fraction(t.hits, t.draws), t.hits, -t.rank))


def submit(
    ranking: Sequence[RankedCandidate],
    test: Callable[[RankedCandidate, rank_table.ThresholdTest], int],
    rule: Rule,
) -> Submission:
    """Steps 3 and 4 of the rule: test the candidates in rank order, at most the first
    `rule.max_tests`, until `rule.passes_needed` have passed or none is left.

    `test(candidate, threshold_test)` draws `threshold_test.draws` new responses from the
    candidate's sampler and returns the number of hits. The caller chooses the seeds, stores the
    draws and charges them (`Submission.test_draws`)."""
    if [entry.rank for entry in ranking] != list(range(1, len(ranking) + 1)):
        raise SubmissionError("the ranks of a ranking are 1..len(ranking)")
    limit = len(ranking) if rule.max_tests is None else min(rule.max_tests, len(ranking))
    tests: list[TestOutcome] = []
    passes: list[TestOutcome] = []
    for entry in ranking[:limit]:
        threshold_test = rank_table.threshold_test(entry.rank, threshold=rule.threshold)
        hits = test(entry, threshold_test)
        if isinstance(hits, bool) or not 0 <= hits <= threshold_test.draws:
            raise SubmissionError(
                f"rank {entry.rank}: {hits!r} hits among {threshold_test.draws} draws"
            )
        outcome = TestOutcome(
            rank=entry.rank,
            candidate_id=entry.candidate_id,
            sampler_id=entry.sampler_id,
            group=entry.group,
            draws=threshold_test.draws,
            required_hits=threshold_test.required_hits,
            hits=int(hits),
            passed=hits >= threshold_test.required_hits,
        )
        tests.append(outcome)
        if outcome.passed:
            passes.append(outcome)
            if len(passes) == rule.passes_needed:
                break
    if passes:
        status: Status = "passed"
        submitted: TestOutcome | None = passes[0]
    elif tests:
        status, submitted = "best_observed", best_observed_test(tests)
    else:
        status, submitted = "no_candidate", None
    return Submission(
        status=status,
        submitted=submitted,
        passes=tuple(passes),
        tests=tuple(tests),
        ranking_length=len(ranking),
        max_tests=rule.max_tests,
        passes_needed=rule.passes_needed,
    )
