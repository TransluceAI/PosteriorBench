"""The weights of a prompt mixture."""

from __future__ import annotations

import math

import numpy as np
import pytest

from posteriorbench_methods.common import signals
from posteriorbench_methods.prompt import mixture, selection_signals

M = 16


def candidate(hits: int, mean_weight: float | None, variance: float | None = 1.0):
    return selection_signals.Statistics(M, hits, mean_weight, variance, iwae=None)


def test_uniform_weights() -> None:
    components = [candidate(5, -1.0), candidate(3, -2.0), candidate(8, -4.0)]
    assert mixture.weights("uniform", components, mixture.Shrinkage(-2.0, 1.0)) == [1 / 3] * 3
    assert mixture.weights("eb", components[:1], mixture.Shrinkage(-2.0, 1.0)) == [1.0]


def test_eb_weights_are_proportional_to_exp_of_the_shrunk_mean_weight() -> None:
    shrinkage = mixture.Shrinkage(mu=-2.5, kappa=4.0)
    components = [candidate(5, -1.0), candidate(0, None, None), candidate(2, -3.0)]
    weights = mixture.weights("eb", components, shrinkage)
    shrunk = [(5 * -1.0 + 4.0 * -2.5) / 9, -2.5, (2 * -3.0 + 4.0 * -2.5) / 6]
    expected = np.exp(shrunk) / np.exp(shrunk).sum()
    assert weights == pytest.approx(expected.tolist())
    assert math.isclose(sum(weights), 1.0)
    # Without a fit (fewer than two candidates above the hit fraction) the weights are uniform.
    assert mixture.weights("eb", components, None) == [1 / 3] * 3


def test_the_fit_is_that_of_ebjc() -> None:
    rng = np.random.default_rng(4)
    hit = rng.random((9, M)) < 0.5
    draws = [
        signals.SelectionDraws(hit[i], rng.normal(-30, 5, M), rng.normal(-40, 5, M))
        for i in range(9)
    ]
    scored = [selection_signals.statistics(d) for d in draws]
    assert all(s.hit_fraction > 0.15 for s in scored)
    shrinkage = mixture.fit_shrinkage(scored)
    shared = signals.score_candidates(draws, signal_hit_fraction=0.15)
    assert shrinkage is not None and shared.mu is not None and shared.kappa is not None
    assert math.isclose(shrinkage.mu, shared.mu) and math.isclose(shrinkage.kappa, shared.kappa)
    assert mixture.fit_shrinkage(scored[:1]) is None
    same = [candidate(4, -2.0), candidate(6, -2.0)]
    fit = mixture.fit_shrinkage(same)
    assert fit is not None and fit.kappa == 1e12, "no variance between candidates: full shrinkage"
