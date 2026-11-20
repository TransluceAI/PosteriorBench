"""`rank_table`: the tables in `data/` are what the module computes, and `threshold_test` gives the
test at any rank."""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction

import pytest

from posteriorbench_methods.common import rank_table


@pytest.mark.parametrize("name", sorted(rank_table.TABLES))
def test_table_in_data_equals_the_computed_one(name: str) -> None:
    threshold, alternative, max_rank = rank_table.TABLES[name]
    computed = rank_table.build_rank_table(threshold, alternative, max_rank)
    text = (rank_table.DATA_DIR / name).read_text()
    assert text == rank_table.render(computed)
    assert json.loads(text) == computed


def test_exact_tail() -> None:
    assert rank_table.exact_upper_tail(3, 0, Fraction(1, 2)) == 1
    assert rank_table.exact_upper_tail(3, 2, Fraction(1, 2)) == Fraction(1, 2)


def test_threshold_test_within_the_tables() -> None:
    first = rank_table.threshold_test(1)
    assert (first.draws, first.required_hits) == (164, 25)
    assert first.size == pytest.approx(0.02210998, abs=1e-8)
    assert first.power == pytest.approx(0.95129454, abs=1e-8)
    assert [rank_table.threshold_test(rank).draws for rank in range(1, 9)] == [
        164, 198, 227, 243, 255, 266, 277, 282
    ]  # fmt: skip
    last = rank_table.threshold_test(1024)
    assert (last.draws, last.required_hits) == (576, 100)
    one_percent = rank_table.threshold_test(1, threshold=0.01)
    assert (one_percent.draws, one_percent.required_hits) == (1857, 28)
    for name, (threshold, _, max_rank) in rank_table.TABLES.items():
        tests = json.loads((rank_table.DATA_DIR / name).read_text())["tests"]
        assert [test["rank"] for test in tests] == list(range(1, max_rank + 1))
        for test in (tests[0], tests[max_rank // 2], tests[-1]):
            read = rank_table.threshold_test(test["rank"], threshold=float(threshold))
            assert dataclasses.asdict(read) == test


@pytest.mark.parametrize(
    ("threshold", "alternative", "ranks"),
    [
        (Fraction(1, 10), Fraction(1, 5), (1025, 1026, 3000, 131072)),
        (Fraction(1, 100), Fraction(1, 50), (17, 40)),
    ],
)
def test_threshold_test_beyond_the_tables(
    threshold: Fraction, alternative: Fraction, ranks: tuple[int, ...]
) -> None:
    """A rank after the last one of a table is computed: its test has the level and the power, and
    no test with fewer draws, or with the same draws and fewer hits, has both."""
    previous = rank_table.threshold_test(
        max(rank for _, _, rank in rank_table.TABLES.values() if rank < ranks[0]),
        threshold=float(threshold),
    )
    for rank in ranks:
        test = rank_table.threshold_test(rank, threshold=float(threshold))
        level = rank_table.rank_level(rank)
        n, k = test.draws, test.required_hits
        assert rank_table.exact_upper_tail(n, k, threshold) <= level
        assert rank_table.exact_upper_tail(n, k, alternative) >= rank_table.POWER
        assert rank_table.exact_upper_tail(n, k - 1, threshold) > level  # fewer hits: not valid
        # One draw fewer: the smallest valid number of hits has too little power.
        fewer = next(
            h for h in range(n + 1) if rank_table.exact_upper_tail(n - 1, h, threshold) <= level
        )
        assert rank_table.exact_upper_tail(n - 1, fewer, alternative) < rank_table.POWER
        assert test.draws >= previous.draws
        assert test.size == pytest.approx(float(rank_table.exact_upper_tail(n, k, threshold)))
        previous = test


def test_threshold_test_errors() -> None:
    with pytest.raises(rank_table.RankTableError):
        rank_table.threshold_test(0)
    with pytest.raises(rank_table.RankTableError):
        rank_table.threshold_test(1, threshold=0.05)
