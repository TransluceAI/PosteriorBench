"""Selection signals computed from selection draws: JC and EB-JC.

A candidate's selection draws are m responses of the candidate with their hit flags, log q and
log p. The log importance weight of a hit is log p - log q; `theta` below is the mean of these
weights over a candidate's hits, and `k` the number of hits.

A score is higher for a better candidate. JC is log(k / m) + theta. EB-JC replaces theta by theta
shrunk toward mu, the mean of theta over the candidates being ranked: mu counts as kappa hits,
where kappa is the median within-candidate variance of the log weights divided by the variance of
theta across candidates. mu and kappa are fitted on the candidates whose hit fraction is above the
signal hit fraction (15%), and only those candidates get a score.

No signal reads p̂(S): it is the same for every candidate and would not change a ranking.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class SelectionDrawsError(ValueError):
    """Inconsistent selection draws: candidates with unequal numbers of draws, or a non-finite
    log q or log p on a hit."""


@dataclasses.dataclass
class SelectionDraws:
    """One candidate's selection draws, in the order drawn."""

    hit: np.ndarray  # [m] bool
    logq: np.ndarray  # [m] float
    logp: np.ndarray  # [m] float; read on hits only (NaN allowed on a miss)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> SelectionDraws:
        """The selection draws stored as rows with `hit`, `logq` and `logp`. A row may leave out
        a log q or log p that was not computed, or hold None for it: it is NaN here."""

        def values(key: str) -> np.ndarray:
            return np.array(
                [np.nan if row.get(key) is None else row[key] for row in rows], dtype=float
            )

        return cls(
            hit=np.array([row["hit"] for row in rows], dtype=bool),
            logq=values("logq"),
            logp=values("logp"),
        )


@dataclasses.dataclass(frozen=True)
class WeightStatistics:
    """The log importance weights, log p - log q, of the hits among one candidate's selection
    draws.

    hits: k, the number of hits.
    mean: theta, the mean of the weights; None without a hit.
    variance: their sample variance; None with fewer than two hits.
    iwae: the IWAE score, log((1 / m) sum over the hits of p / q); None without a hit."""

    hits: int
    mean: float | None
    variance: float | None
    iwae: float | None


def weight_statistics(draws: SelectionDraws) -> WeightStatistics:
    """The statistics of one candidate's weights. log q and log p are read on the hits only."""
    hit = draws.hit.astype(bool)
    k = int(np.count_nonzero(hit))
    if k == 0:
        return WeightStatistics(0, None, None, None)
    lw = (draws.logp - draws.logq)[hit]
    return WeightStatistics(
        hits=k,
        mean=float(lw.mean()),
        variance=float(lw.var(ddof=1)) if k >= 2 else None,
        iwae=float(np.logaddexp.reduce(lw)) - math.log(hit.size),
    )


def eb_scores(theta: np.ndarray, s2: np.ndarray, kk: np.ndarray, n_draws: int) -> np.ndarray:
    """The EB-JC score of the candidates of one ranking, from their theta, s2 and hits
    (index-aligned arrays; NaN where a candidate has none).

    mu is the mean of theta, kappa the median of s2 divided by the sample variance of theta (1e12
    if that variance is at most 1e-12); the score of candidate i is
    log(k_i / n_draws) + (k_i theta_i + kappa mu) / (k_i + kappa). NaN for a candidate without
    hits, and for every candidate if fewer than two have hits."""
    valid = ~np.isnan(theta)
    out = np.full(theta.shape, np.nan)
    if valid.sum() < 2:
        return out
    mu, kappa = _prior(theta[valid], s2)
    out[valid] = _shrunk(theta[valid], kk[valid], n_draws, mu, kappa)
    return out


def _prior(theta: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
    """mu and kappa from the theta of two or more candidates and the s2 of the candidates (NaN
    where a candidate has none)."""
    mu = float(np.mean(theta))
    between = float(np.var(theta, ddof=1))
    within = s2[~np.isnan(s2)]
    median_within = float(np.median(within)) if within.size else 0.0
    return mu, (median_within / between) if between > 1e-12 else 1e12


def _shrunk(theta: np.ndarray, kk: np.ndarray, n_draws: int, mu: float, kappa: float) -> np.ndarray:
    """log(k_i / n_draws) + (k_i theta_i + kappa mu) / (k_i + kappa) for each candidate."""
    kv = kk.astype(float)
    return np.log(kv / n_draws) + (kv * theta + kappa * mu) / (kv + kappa)


def jc_score(draws: SelectionDraws) -> float | None:
    """JC of one candidate: log(hits / m) plus the mean over its hits of (log p - log q). None
    without hits."""
    k = int(draws.hit.sum())
    if k == 0:
        return None
    m = draws.hit.size
    return math.log(k / m) + float(np.mean(draws.logp[draws.hit] - draws.logq[draws.hit]))


@dataclasses.dataclass(frozen=True)
class CandidateScores:
    """JC and EB-JC of the candidates of one ranking, index-aligned with their selection draws.

    by_signal: whether the candidate's hit fraction is above the signal hit fraction; `jc` and
        `ebjc` are None for the others.
    mu, kappa: the fitted prior mean of theta and the number of hits it counts as; None with fewer
        than two candidates above the hit fraction (then EB-JC equals JC)."""

    by_signal: list[bool]
    jc: list[float | None]
    ebjc: list[float | None]
    mu: float | None
    kappa: float | None


def score_candidates(
    draws: Sequence[SelectionDraws], *, signal_hit_fraction: float
) -> CandidateScores:
    """JC and EB-JC of every candidate from its selection draws. EB-JC is fitted over the
    candidates whose hit fraction is above `signal_hit_fraction`, so `draws` holds exactly the
    candidates of one ranking, one entry for each distinct sampler. With one candidate above the
    hit fraction its EB-JC is its JC. Raises SelectionDrawsError for inconsistent draws."""
    if not draws:
        return CandidateScores([], [], [], None, None)
    m = draws[0].hit.size
    if any(d.hit.size != m or d.logq.size != m or d.logp.size != m for d in draws):
        raise SelectionDrawsError("candidates of one ranking have unequal numbers of draws")
    hit = np.stack([d.hit.astype(bool) for d in draws])
    logq = np.stack([np.where(d.hit, d.logq, 0.0) for d in draws])
    logp = np.stack([np.where(d.hit, d.logp, 0.0) for d in draws])
    if not np.all(np.isfinite(logq[hit])) or not np.all(np.isfinite(logp[hit])):
        raise SelectionDrawsError("non-finite log q or log p on a hit")
    counts = hit.sum(axis=1)
    fractions = counts / m
    by_signal = fractions > signal_hit_fraction
    ids = np.flatnonzero(by_signal)
    theta = np.full(len(draws), np.nan)
    s2 = np.full(len(draws), np.nan)
    jc = np.full(len(draws), np.nan)
    ebjc = np.full(len(draws), np.nan)
    for i in ids:
        weights = weight_statistics(SelectionDraws(hit[i], logq[i], logp[i]))
        theta[i] = weights.mean
        s2[i] = weights.variance if weights.variance is not None else np.nan
        jc[i] = theta[i] + np.log(fractions[i])
    mu = kappa = None
    if len(ids) > 1:
        mu, kappa = _prior(theta[ids], s2[ids])
        ebjc[ids] = _shrunk(theta[ids], counts[ids], m, mu, kappa)
    elif len(ids) == 1:
        ebjc[ids] = jc[ids]
    return CandidateScores(
        by_signal=[bool(x) for x in by_signal],
        jc=[float(x) if math.isfinite(x) else None for x in jc],
        ebjc=[float(x) if math.isfinite(x) else None for x in ebjc],
        mu=mu,
        kappa=kappa,
    )
