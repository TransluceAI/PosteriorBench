"""The selection signals computed on a candidate's own selection draws."""

from __future__ import annotations

import math

import numpy as np

from posteriorbench_methods.common import signals
from posteriorbench_methods.prompt import selection_signals

M = 16
HIT_FRACTION = 0.15


def pool_draws(seed: int, candidates: int = 12) -> list[signals.SelectionDraws]:
    """Selection draws of a pool whose candidates have hit rates between 0 and 0.6; log q and
    log p are NaN on the misses, as the family stores them."""
    rng = np.random.default_rng(seed)
    hit = rng.random((candidates, M)) < rng.uniform(0.0, 0.6, size=(candidates, 1))
    logp = np.where(hit, rng.normal(-40, 5, size=(candidates, M)), np.nan)
    logq = np.where(hit, rng.normal(-30, 5, size=(candidates, M)), np.nan)
    return [signals.SelectionDraws(hit[i], logq[i], logp[i]) for i in range(candidates)]


def test_scores_follow_their_definitions() -> None:
    draws = pool_draws(0)
    statistics = [selection_signals.statistics(d) for d in draws]
    by_signal = {
        signal: selection_signals.scores(signal, statistics, signal_hit_fraction=HIT_FRACTION)
        for signal in ("jc", "hitrate", "iwae")
    }
    for index, (d, s) in enumerate(zip(draws, statistics, strict=True)):
        hits = int(d.hit.sum())
        assert (s.draws, s.hits, s.hit_fraction) == (M, hits, hits / M)
        assert by_signal["hitrate"][index] == hits / M
        if hits == 0:
            assert (s.mean_weight, s.iwae, s.jc) == (None, None, None)
            continue
        weights = d.logp[d.hit] - d.logq[d.hit]
        assert s.mean_weight is not None and math.isclose(s.mean_weight, weights.mean())
        jc, iwae = by_signal["jc"][index], by_signal["iwae"][index]
        assert jc is not None and math.isclose(jc, math.log(hits / M) + weights.mean())
        assert iwae is not None and math.isclose(iwae, math.log(np.exp(weights).sum() / M))
        if hits >= 2:
            assert s.weight_variance is not None
            assert math.isclose(s.weight_variance, weights.var(ddof=1))
        else:
            assert s.weight_variance is None


def test_ebjc_shrinks_toward_the_mean_of_the_scored_candidates() -> None:
    draws = pool_draws(1)
    statistics = [selection_signals.statistics(d) for d in draws]
    scores = selection_signals.scores("ebjc", statistics, signal_hit_fraction=HIT_FRACTION)
    scored = [s for s in statistics if s.hit_fraction > HIT_FRACTION]
    assert len(scored) >= 2
    means = np.array([s.mean_weight for s in scored])
    variances = np.array([s.weight_variance for s in scored])
    mu, kappa = means.mean(), np.median(variances) / means.var(ddof=1)
    for s, score in zip(statistics, scores, strict=True):
        if s.hit_fraction <= HIT_FRACTION:
            assert score is None
            continue
        assert s.mean_weight is not None and score is not None
        shrunk = (s.hits * s.mean_weight + kappa * mu) / (s.hits + kappa)
        assert math.isclose(score, math.log(s.hits / M) + shrunk)
    # The same scores as the signal that ranks checkpoints and settings in the other families.
    shared = signals.score_candidates(draws, signal_hit_fraction=HIT_FRACTION)
    assert scores == shared.ebjc


def test_a_candidate_scored_alone_has_its_jc_score() -> None:
    hit = np.zeros((3, M), dtype=bool)
    hit[1, :5] = True
    hit[2, :2] = True  # 2 of 16 is not above 15%
    draws = [
        signals.SelectionDraws(hit[i], np.full(M, -30.0), np.linspace(-41.0, -39.0, M))
        for i in range(3)
    ]
    statistics = [selection_signals.statistics(d) for d in draws]
    scores = selection_signals.ebjc_scores(statistics, signal_hit_fraction=HIT_FRACTION)
    assert scores == [None, statistics[1].jc, None]
