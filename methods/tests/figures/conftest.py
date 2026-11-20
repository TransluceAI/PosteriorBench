"""Fixtures of the tests of `figures/`: the scripts there are not part of the installed package,
so their folder is put on the import path, and the paper's tables are read once."""

from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

FIGURES = pathlib.Path(__file__).resolve().parents[2] / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import paper_runs

TOLERANCE = 1e-9


@pytest.fixture(scope="session")
def main_runs() -> list[paper_runs.Row]:
    """The runs of the main comparison on the 21 tasks."""
    return paper_runs.read_table(paper_runs.DATA_DIR / "main_comparison_runs.csv.gz")


@pytest.fixture(scope="session")
def pair_runs() -> list[paper_runs.Row]:
    """The runs of the training-free methods on the 108 pairs."""
    return paper_runs.read_table(paper_runs.DATA_DIR / "training_free_108pairs_runs.csv.gz")


@pytest.fixture(scope="session")
def probabilities() -> dict[str, float]:
    return paper_runs.event_probabilities()


SamePoints = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, str]], Sequence[str], Sequence[str]], None
]


@pytest.fixture(scope="session")
def same_points() -> SamePoints:
    """`assert_same_points`, for the test modules (which cannot import this file)."""
    return assert_same_points


def assert_same_points(
    computed: Sequence[Mapping[str, Any]],
    stored: Sequence[Mapping[str, str]],
    keys: Sequence[str],
    columns: Sequence[str],
) -> None:
    """The computed points and the stored rows have the same keys, and every column of every
    point agrees within `TOLERANCE` (an empty stored cell is a missing standard error)."""
    by_key = {tuple(str(point[key]) for key in keys): point for point in computed}
    assert len(by_key) == len(computed), "the computed points repeat a key"
    assert set(by_key) == {tuple(row[key] for key in keys) for row in stored}
    for row in stored:
        point = by_key[tuple(row[key] for key in keys)]
        for column in columns:
            if row[column] == "":
                assert point[column] is None, (row, column)
            else:
                assert math.isclose(
                    float(point[column]), float(row[column]), rel_tol=TOLERANCE, abs_tol=TOLERANCE
                ), (tuple(row[key] for key in keys), column, point[column], row[column])
