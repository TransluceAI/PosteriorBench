"""The weights of a prompt mixture.

A mixture takes the first s candidates of the EB-JC ranking that pass their threshold tests. Its
EB weights are w_i proportional to exp(l~_i), where l~_i is the mean log importance weight l_i of
component i shrunk toward mu as in EB-JC:

    l~_i = (h_i l_i + kappa mu) / (h_i + kappa)

for a component with h_i hits among its selection draws. mu is the mean of l over the candidates
of the pool whose hit fraction is above 15%, and kappa the median over these candidates of the
variance of a candidate's log importance weights, divided by the variance of l across them. With
these weights a component's share of the mixture's hits is about proportional to the exponential
of its EB-JC score, which aims at a low reverse KL.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Sequence

from posteriorbench_methods.prompt import methods, selection_signals


@dataclasses.dataclass(frozen=True)
class Shrinkage:
    """The fitted prior mean `mu` of the mean log importance weight, and the number of hits
    `kappa` that it counts as."""

    mu: float
    kappa: float

    def shrunk_mean_weight(self, candidate: selection_signals.Statistics) -> float:
        """l~ of a candidate; mu for a candidate without a hit."""
        if candidate.mean_weight is None:
            return self.mu
        return (candidate.hits * candidate.mean_weight + self.kappa * self.mu) / (
            candidate.hits + self.kappa
        )


def fit_shrinkage(scored: Sequence[selection_signals.Statistics]) -> Shrinkage | None:
    """mu and kappa from the candidates of a pool whose hit fraction is above 15% (`scored`).
    kappa is 1e12 if the variance of their mean weights is at most 1e-12. None with fewer than two
    such candidates, which leaves nothing to fit."""
    if len(scored) < 2:
        return None
    means = [candidate.mean_weight for candidate in scored if candidate.mean_weight is not None]
    assert len(means) == len(scored), "a candidate above the hit fraction has hits"
    mu = math.fsum(means) / len(means)
    between = math.fsum((mean - mu) ** 2 for mean in means) / (len(means) - 1)
    within = [c.weight_variance for c in scored if c.weight_variance is not None]
    median_within = statistics.median(within) if within else 0.0
    return Shrinkage(mu, median_within / between if between > 1e-12 else 1e12)


def weights(
    kind: methods.Weights,
    components: Sequence[selection_signals.Statistics],
    shrinkage: Shrinkage | None,
) -> list[float]:
    """The weights of a mixture of `components`: equal weights for "uniform"; for "eb", weights
    proportional to exp(l~), and equal weights if there is no fit (`shrinkage` None)."""
    count = len(components)
    if kind == "uniform" or shrinkage is None or count == 1:
        return [1.0 / count] * count
    shrunk = [shrinkage.shrunk_mean_weight(component) for component in components]
    largest = max(shrunk)
    unnormalized = [math.exp(value - largest) for value in shrunk]
    total = math.fsum(unnormalized)
    return [value / total for value in unnormalized]
