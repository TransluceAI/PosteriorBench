"""The selection signals that score a candidate on its own selection draws: JC, EB-JC, the hit rate
and IWAE. (The CE method scores the candidates on a bank that they share; it is in `ce`.)

A candidate has m selection draws with h hits. The log importance weight of a hit y is
log p(y) - log q(y), and l is the mean of these weights over the candidate's hits. A score is higher
for a better candidate:

    JC        log(h / m) + l
    EB-JC     log(h / m) + (h l + kappa mu) / (h + kappa), with mu and kappa fitted over the
              candidates above the signal hit fraction, 15% (`signals.eb_scores`)
    hit rate  h / m
    IWAE      log((1 / m) sum over the hits of p(y) / q(y))
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from posteriorbench_methods.common import signals

OwnDrawsSignal = Literal["ebjc", "jc", "hitrate", "iwae"]
# The signals that read p, on every hit among a candidate's selection draws. The hit rate does not.
READ_P_ON_HITS: tuple[OwnDrawsSignal, ...] = ("ebjc", "jc", "iwae")


@dataclasses.dataclass(frozen=True)
class Statistics:
    """One candidate's selection draws in numbers: `draws` is m, `hits` h, `mean_weight` l.
    `mean_weight` and `iwae` are None without a hit; `weight_variance` (sample variance of the
    hits' log weights) is None with fewer than two."""

    draws: int
    hits: int
    mean_weight: float | None
    weight_variance: float | None
    iwae: float | None

    @property
    def hit_fraction(self) -> float:
        return self.hits / self.draws

    @property
    def jc(self) -> float | None:
        """The JC score; None without a hit."""
        if self.mean_weight is None:
            return None
        return self.mean_weight + math.log(self.hits / self.draws)


def statistics(draws: signals.SelectionDraws) -> Statistics:
    """The statistics of one candidate from its selection draws."""
    weights = signals.weight_statistics(draws)
    return Statistics(
        draws=int(draws.hit.size),
        hits=weights.hits,
        mean_weight=weights.mean,
        weight_variance=weights.variance,
        iwae=weights.iwae,
    )


def ebjc_scores(pool: Sequence[Statistics], *, signal_hit_fraction: float) -> list[float | None]:
    """EB-JC scores, with mu and kappa fitted over the candidates above `signal_hit_fraction`;
    None for the others. A candidate scored alone gets its JC score."""
    scored = [index for index, s in enumerate(pool) if s.hit_fraction > signal_hit_fraction]
    scores: list[float | None] = [None] * len(pool)
    if len(scored) == 1:
        scores[scored[0]] = pool[scored[0]].jc
    elif scored:
        variances = [pool[index].weight_variance for index in scored]
        fitted = signals.eb_scores(
            np.asarray([pool[index].mean_weight for index in scored], dtype=float),
            np.asarray([np.nan if v is None else v for v in variances], dtype=float),
            np.asarray([pool[index].hits for index in scored], dtype=np.int64),
            pool[0].draws,
        )
        for index, score in zip(scored, fitted.tolist(), strict=True):
            scores[index] = score
    return scores


def scores(
    signal: OwnDrawsSignal, pool: Sequence[Statistics], *, signal_hit_fraction: float
) -> list[float | None]:
    """Each candidate's score under `signal`; None without a hit, and for EB-JC also at or below
    `signal_hit_fraction`."""
    if signal == "ebjc":
        return ebjc_scores(pool, signal_hit_fraction=signal_hit_fraction)
    if signal == "jc":
        return [s.jc for s in pool]
    if signal == "iwae":
        return [s.iwae for s in pool]
    return [s.hit_fraction for s in pool]
