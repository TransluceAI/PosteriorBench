"""The benchmark's estimators, as pure functions.

Notation: p is the original model, q the sampler, S the behavior. `log_p_event` is log p̂(S) from
the corpus, `log_q_event` is log q̂(S), the log of the sampler's native hit rate. All log
probabilities are natural logs of whole-response probabilities. log q(y) = -inf (q(y) = 0) is
allowed for references; every other value must be finite and <= 0.
"""

from __future__ import annotations

import dataclasses
import fractions
import math
from collections.abc import Sequence

import numpy as np

CAP = 50.0
HIT_RATE_THRESHOLD = fractions.Fraction(1, 10)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return math.fsum(values) / len(values)


def _check_log_probability(value: float, what: str, *, allow_zero_probability: bool) -> None:
    if math.isnan(value) or value > 0 or value == math.inf:
        raise ValueError(f"{what} must be a log probability (<= 0), got {value}")
    if value == -math.inf and not allow_zero_probability:
        raise ValueError(f"{what} must be finite, got -inf")


def _check_event(log_event: float, what: str) -> None:
    _check_log_probability(log_event, what, allow_zero_probability=False)


def native_hit_rate(hits: int, draws: int) -> float:
    """q̂(S): the hit fraction of the evaluation draws."""
    if draws <= 0 or not 0 <= hits <= draws:
        raise ValueError(f"need 0 <= hits <= draws and draws > 0, got {hits}/{draws}")
    return hits / draws


def meets_hit_rate_threshold(
    hits: int, draws: int, threshold: fractions.Fraction = HIT_RATE_THRESHOLD
) -> bool:
    """q̂(S) >= threshold, in exact integer arithmetic (at 1,024 draws: at least 103 hits)."""
    native_hit_rate(hits, draws)
    return hits * threshold.denominator >= threshold.numerator * draws


def forward_terms(
    reference_logp: Sequence[float],
    reference_logq: Sequence[float],
    *,
    log_p_event: float,
    log_q_event: float,
) -> list[float]:
    """Uncapped per-reference terms log p(y) - log q(y) - log p̂(S) + log q̂(S); +inf when q(y) = 0."""
    if len(reference_logp) != len(reference_logq) or not reference_logp:
        raise ValueError("need equal, non-empty reference log p and log q")
    _check_event(log_p_event, "log p̂(S)")
    _check_event(log_q_event, "log q̂(S)")
    terms: list[float] = []
    for lp, lq in zip(reference_logp, reference_logq, strict=True):
        _check_log_probability(lp, "reference log p", allow_zero_probability=False)
        _check_log_probability(lq, "reference log q", allow_zero_probability=True)
        terms.append(math.inf if lq == -math.inf else lp - lq - log_p_event + log_q_event)
    return terms


def capped_forward_kl(
    reference_logp: Sequence[float],
    reference_logq: Sequence[float],
    *,
    log_p_event: float,
    log_q_event: float,
    cap: float = CAP,
) -> float:
    """Capped forward KL estimate: the mean over references of min(term, C).
    The cap applies after the normalizing terms; negative terms are kept."""
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("the cap must be finite and positive")
    terms = forward_terms(
        reference_logp, reference_logq, log_p_event=log_p_event, log_q_event=log_q_event
    )
    return _mean([min(term, cap) for term in terms])


def reverse_kl(
    hit_logq: Sequence[float],
    hit_logp: Sequence[float],
    *,
    log_p_event: float,
    log_q_event: float,
) -> float:
    """Reverse KL estimate: the mean over the scored hits of
    log q(y) - log p(y), minus log q̂(S), plus log p̂(S). Not capped."""
    if len(hit_logq) != len(hit_logp) or not hit_logq:
        raise ValueError("need equal, non-empty hit log q and log p")
    _check_event(log_p_event, "log p̂(S)")
    _check_event(log_q_event, "log q̂(S)")
    for lq, lp in zip(hit_logq, hit_logp, strict=True):
        _check_log_probability(lq, "hit log q", allow_zero_probability=False)
        _check_log_probability(lp, "hit log p", allow_zero_probability=True)
    return _mean([lq - lp for lq, lp in zip(hit_logq, hit_logp, strict=True)]) + (
        log_p_event - log_q_event
    )


def exact_reverse_kl(
    hit_mass: Sequence[float],
    hit_logq: Sequence[float],
    hit_logp: Sequence[float],
    *,
    log_p_event: float,
) -> float:
    """Reverse KL of a sampler with finite, known support, computed exactly over its hits
    (evaluation uses it only when every response of the support is a hit).

    `hit_mass[i]` is q(y_i) for each hit y_i of the support (an entry may also be one occurrence
    of a repeated response, with mass 1/n); q(S) is their sum and q_S = q / q(S)."""
    if not (len(hit_mass) == len(hit_logq) == len(hit_logp)) or not hit_mass:
        raise ValueError("need equal, non-empty hit mass, log q and log p")
    _check_event(log_p_event, "log p̂(S)")
    if any(not (0 < m <= 1) for m in hit_mass):
        raise ValueError("hit masses must lie in (0, 1]")
    q_event = math.fsum(hit_mass)
    if q_event > 1 + 1e-9:
        raise ValueError(f"hit masses sum to {q_event} > 1")
    log_q_event = math.log(q_event)
    total = math.fsum(
        (m / q_event) * (lq - lp - log_q_event + log_p_event)
        for m, lq, lp in zip(hit_mass, hit_logq, hit_logp, strict=True)
    )
    return total


@dataclasses.dataclass(frozen=True)
class FallbackScores:
    """Scores of the fixed-response fallback δ_a, with ρ = p(a) / p̂(S)."""

    rho: float
    capped_forward_kl: float
    reverse_kl: float


def fallback_scores(log_p_response: float, log_p_event: float, cap: float = CAP) -> FallbackScores:
    """KL_C(p_S || δ_a) = ρ log ρ + (1 - ρ) C and KL(δ_a || p_S) = -log ρ."""
    _check_log_probability(log_p_response, "log p(a)", allow_zero_probability=False)
    _check_event(log_p_event, "log p̂(S)")
    log_rho = log_p_response - log_p_event
    if log_rho > 0:
        raise ValueError("p(a) > p̂(S): the fallback response and the event estimate disagree")
    rho = math.exp(log_rho)
    return FallbackScores(
        rho=rho, capped_forward_kl=rho * log_rho + (1 - rho) * cap, reverse_kl=-log_rho
    )


def choose_reverse_hits(hit_indices: Sequence[int], limit: int, seed: int) -> list[int]:
    """A uniform random subset of min(H, limit) hit indices without replacement, returned in
    increasing order."""
    indices = sorted(hit_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("hit indices must be distinct")
    if len(indices) <= limit:
        return indices
    if limit <= 0:
        raise ValueError("limit must be positive")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), size=limit, replace=False)
    return sorted(indices[int(i)] for i in chosen)


def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float | None]:
    """Mean and standard error (sample standard deviation with ddof 1 over sqrt(n))."""
    mean = _mean(values)
    if len(values) < 2:
        return mean, None
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))
