"""The threshold test of the submission rule at every rank.

The test at rank j draws n_j new responses and passes with at least k_j hits. It is a one-sided
binomial test of q(S) <= threshold at level 0.05 / (j (j + 1)), with power at least 0.95 at
q(S) = 2 x threshold; among the tests with this level and power it has the fewest draws, and then
the fewest required hits. At the benchmark's threshold of 10% the test at rank 1 draws 164 responses
and needs 25 hits.

`threshold_test(rank)` gives the test at a rank. Ranks 1..1,024 at the 10% threshold and 1..16 at
the 1% threshold are read from the tables in `data/`; a later rank is computed here. The tables are
written by `scripts/make_rank_table.py` from `build_rank_table`.

How a test is computed: for each rank, every n from the previous rank's n upward is checked (the
smallest feasible n never decreases with the rank, since a test valid at a smaller level is valid at
a larger one), and the smallest k with a valid level is taken at the first n where it has the
power. A comparison is made in exact integer arithmetic wherever floating point is within 1e-9
(relative) of the boundary. The size and the power of a test are computed as exact rationals.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
LEVEL = Fraction(1, 20)  # the level 0.05 of all the tests of one ranking together
POWER = Fraction(19, 20)  # 0.95
FLOAT_MARGIN = 1e-9  # closer than this (in log space) to a boundary: decide exactly


def rank_level(rank: int) -> Fraction:
    """The level of the test at rank j: 0.05 / (j (j + 1))."""
    if rank < 1:
        raise ValueError("ranks start at 1")
    return LEVEL / (rank * (rank + 1))


def _log_upper_tails(draws: int, probability: float) -> list[float]:
    """[log P[Binomial(draws, probability) >= h] for h = 0..draws] in floating point (log pmf by
    its ratio recursion, then a running log-sum-exp from the top; relative error far below
    FLOAT_MARGIN for the table's sizes)."""
    k = np.arange(draws, dtype=np.float64)
    steps = np.log((draws - k) / (k + 1)) + math.log(probability) - math.log1p(-probability)
    log_pmf = np.concatenate(([draws * math.log1p(-probability)], steps)).cumsum()
    tails = np.logaddexp.accumulate(log_pmf[::-1])[::-1]
    return [min(float(t), 0.0) for t in tails]


def exact_upper_tail(draws: int, hits: int, probability: Fraction) -> Fraction:
    """P[Binomial(draws, probability) >= hits] as an exact rational (summing the shorter side)."""
    if not 0 <= hits <= draws + 1:
        raise ValueError(f"hits must lie in [0, draws + 1], got {hits}")
    a, q = probability.numerator, probability.denominator
    b = q - a

    def mass(lo: int, hi: int) -> int:
        """sum over k in [lo, hi) of C(draws, k) a^k b^(draws - k), terms by their ratio."""
        if lo >= hi:
            return 0
        term = math.comb(draws, lo) * a**lo * b ** (draws - lo)
        total = term
        for k in range(lo, hi - 1):
            term = term * (draws - k) * a // ((k + 1) * b)
            total += term
        return total

    if hits <= draws - hits:
        upper = q**draws - mass(0, hits)
    else:
        upper = mass(hits, draws + 1)
    return Fraction(upper, q**draws)


def _size_valid(
    draws: int, hits: int, threshold: Fraction, level: Fraction, log_tails: list[float]
) -> bool:
    """Whether P[Bin(draws, threshold) >= hits] <= level, exactly when floating point is close."""
    if hits > draws:
        return True
    log_level = math.log(level)
    if abs(log_tails[hits] - log_level) > FLOAT_MARGIN:
        return log_tails[hits] <= log_level
    return exact_upper_tail(draws, hits, threshold) <= level


def _powerful(draws: int, hits: int, alternative: Fraction, log_tails: list[float]) -> bool:
    """Whether P[Bin(draws, alternative) >= hits] >= 0.95, exactly when floating point is close."""
    log_power = math.log(POWER)
    if abs(log_tails[hits] - log_power) > FLOAT_MARGIN:
        return log_tails[hits] >= log_power
    return exact_upper_tail(draws, hits, alternative) >= POWER


def smallest_test(
    rank: int, *, threshold: Fraction, alternative: Fraction, start: int = 1
) -> tuple[int, int]:
    """(n_j, k_j) of the test at `rank`, searching n from `start` (a lower bound of n_j)."""
    level = rank_level(rank)
    draws = start
    while True:
        null = _log_upper_tails(draws, float(threshold))
        hits = next(
            (h for h in range(draws + 2) if _size_valid(draws, h, threshold, level, null)), None
        )
        if hits is not None and hits <= draws:
            if _powerful(draws, hits, alternative, _log_upper_tails(draws, float(alternative))):
                return draws, hits
        draws += 1


def build_rank_table(threshold: Fraction, alternative: Fraction, max_rank: int) -> dict[str, Any]:
    """The table of the threshold tests at ranks 1..max_rank, as written to `data/`: the
    threshold, the hit rate `alternative` at which the tests have their power, and one entry for
    each test with the fields of `ThresholdTest`."""
    tests: list[dict[str, Any]] = []
    start = 1
    for rank in range(1, max_rank + 1):
        draws, hits = smallest_test(rank, threshold=threshold, alternative=alternative, start=start)
        start = draws
        tests.append(
            {
                "rank": rank,
                "draws": draws,
                "required_hits": hits,
                "size": float(exact_upper_tail(draws, hits, threshold)),
                "power": float(exact_upper_tail(draws, hits, alternative)),
            }
        )
    return {
        "threshold": float(threshold),
        "alternative": float(alternative),
        "level": "0.05 / (j (j + 1)) at rank j",
        "minimum_power": float(POWER),
        "tests": tests,
    }


def render(table: dict[str, Any]) -> str:
    """The text of a rank-table file: JSON, with one line for each test."""
    head = {key: value for key, value in table.items() if key != "tests"}
    lines = json.dumps(head, indent=1).rstrip("}").rstrip("\n") + ',\n "tests": [\n'
    lines += ",\n".join(f"  {json.dumps(test)}" for test in table["tests"])
    return lines + "\n ]\n}\n"


# File name -> (threshold, alternative, last rank) of each table in `data/`.
TABLES: dict[str, tuple[Fraction, Fraction, int]] = {
    "rank_table.json": (Fraction(1, 10), Fraction(1, 5), 1024),
    "rank_table_1pct.json": (Fraction(1, 100), Fraction(1, 50), 16),
}
THRESHOLDS: tuple[float, ...] = tuple(float(null) for null, _, _ in TABLES.values())


class RankTableError(ValueError):
    """A threshold test was asked for at a rank below 1 or at a threshold without a table."""


@dataclasses.dataclass(frozen=True)
class ThresholdTest:
    """The threshold test at one rank: it draws `draws` new responses and passes with at least
    `required_hits` hits. `size` is the probability that it passes a sampler whose hit rate equals
    the threshold, `power` one whose hit rate is twice the threshold."""

    rank: int
    draws: int
    required_hits: int
    size: float
    power: float


@functools.lru_cache(maxsize=None)
def _table_tests(name: str) -> dict[int, ThresholdTest]:
    """The tests of the table `data/<name>`, by rank."""
    table = json.loads((DATA_DIR / name).read_text())
    return {int(test["rank"]): ThresholdTest(**test) for test in table["tests"]}


@functools.lru_cache(maxsize=None)
def threshold_test(rank: int, *, threshold: float = 0.10) -> ThresholdTest:
    """The threshold test at `rank` (1 is the first candidate tested) for the hit-rate threshold
    0.10 (the benchmark's) or 0.01. A rank within the table of `data/` is read from it; a later rank
    is computed, which takes a few milliseconds. Raises RankTableError for a rank below 1 or
    another threshold."""
    if rank < 1:
        raise RankTableError(f"ranks start at 1, got {rank}")
    for name, (null, alternative, last_rank) in TABLES.items():
        if float(null) == threshold:
            break
    else:
        raise RankTableError(
            f"threshold tests exist for the thresholds {THRESHOLDS}, not {threshold}"
        )
    tests = _table_tests(name)
    if rank <= last_rank:
        return tests[rank]
    draws, hits = smallest_test(
        rank, threshold=null, alternative=alternative, start=tests[last_rank].draws
    )
    return ThresholdTest(
        rank=rank,
        draws=draws,
        required_hits=hits,
        size=float(exact_upper_tail(draws, hits, null)),
        power=float(exact_upper_tail(draws, hits, alternative)),
    )
