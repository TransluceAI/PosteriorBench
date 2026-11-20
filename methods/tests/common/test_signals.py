"""The selection signals: JC, EB-JC and the statistics of a candidate's weights."""

from __future__ import annotations

import math

import numpy as np
import pytest

from posteriorbench_methods.common import signals


def draws(hit: list[bool], weights: list[float]) -> signals.SelectionDraws:
    """Selection draws whose hits have the log importance weights `weights`, in order; log p of a
    miss is NaN, as in stored draws."""
    logq = np.full(len(hit), -10.0)
    logp = np.full(len(hit), np.nan)
    logp[np.array(hit)] = -10.0 + np.array(weights)
    return signals.SelectionDraws(np.array(hit), logq, logp)


def test_jc_and_the_statistics_of_one_candidate() -> None:
    candidate = draws([True, True, False, False], [-2.0, -4.0])
    assert signals.jc_score(candidate) == pytest.approx(math.log(0.5) - 3.0)
    weights = signals.weight_statistics(candidate)
    assert (weights.hits, weights.mean, weights.variance) == (2, -3.0, 2.0)
    assert weights.iwae == pytest.approx(math.log((math.exp(-2.0) + math.exp(-4.0)) / 4))
    one_hit = signals.weight_statistics(draws([True, False], [-1.0]))
    assert (one_hit.mean, one_hit.variance) == (-1.0, None)
    without = draws([False, False], [])
    assert signals.jc_score(without) is None
    assert signals.weight_statistics(without) == signals.WeightStatistics(0, None, None, None)


def test_ebjc_shrinks_toward_the_mean_of_the_candidates() -> None:
    candidates = [
        draws([True, True, False, False], [-2.0, -4.0]),  # theta -3, variance 2
        draws([True, True, True, False], [-1.0, -1.0, -4.0]),  # theta -2, variance 3
        draws([False, False, False, False], []),  # not above the hit fraction: no score
    ]
    scores = signals.score_candidates(candidates, signal_hit_fraction=0.15)
    assert scores.by_signal == [True, True, False]
    assert scores.jc[:2] == pytest.approx([math.log(0.5) - 3.0, math.log(0.75) - 2.0])
    assert scores.mu == pytest.approx(-2.5)
    assert scores.kappa == pytest.approx(2.5 / 0.5)  # median variance / variance of theta
    shrunk = [(2 * -3.0 + 5 * -2.5) / 7, (3 * -2.0 + 5 * -2.5) / 8]
    assert scores.ebjc[:2] == pytest.approx([math.log(0.5) + shrunk[0], math.log(0.75) + shrunk[1]])
    assert scores.jc[2] is None and scores.ebjc[2] is None
    # One candidate above the hit fraction: its EB-JC is its JC, and nothing is fitted.
    alone = signals.score_candidates(candidates[:1], signal_hit_fraction=0.15)
    assert alone.ebjc == alone.jc and alone.mu is None and alone.kappa is None


def test_inconsistent_draws_raise() -> None:
    with pytest.raises(signals.SelectionDrawsError, match="unequal"):
        signals.score_candidates(
            [draws([True, False], [-1.0]), draws([True], [-1.0])], signal_hit_fraction=0.15
        )
    broken = draws([True, False], [-1.0])
    broken.logq[0] = -math.inf
    with pytest.raises(signals.SelectionDrawsError, match="non-finite"):
        signals.score_candidates([broken], signal_hit_fraction=0.15)


def test_selection_draws_from_rows() -> None:
    rows = [
        {"hit": True, "logq": -3.0, "logp": -5.0},
        {"hit": False, "logq": -2.0, "logp": None},
        {"hit": False},
    ]
    read = signals.SelectionDraws.from_rows(rows)
    assert read.hit.tolist() == [True, False, False]
    assert read.logq[:2].tolist() == [-3.0, -2.0] and math.isnan(read.logq[2])
    assert read.logp[0] == -5.0 and np.isnan(read.logp[1:]).all()
    assert signals.jc_score(read) == pytest.approx(math.log(1 / 3) - 2.0)
