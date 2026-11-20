"""Every estimator against hand-computed values."""

from __future__ import annotations

import fractions
import math

import pytest

from posteriorbench import estimators

LOG_P_EVENT = math.log(1e-3)
LOG_Q_EVENT = math.log(0.25)


def test_forward_terms_and_cap() -> None:
    logp = [-10.0, -12.0, -11.0, -5.0]
    logq = [-9.0, -math.inf, -14.0, -1.0]
    shift = -LOG_P_EVENT + LOG_Q_EVENT  # 6.907755 - 1.386294 = 5.521461
    expected = [-1.0 + shift, math.inf, 3.0 + shift, -4.0 + shift]
    terms = estimators.forward_terms(logp, logq, log_p_event=LOG_P_EVENT, log_q_event=LOG_Q_EVENT)
    assert terms == pytest.approx(expected)
    value = estimators.capped_forward_kl(
        logp, logq, log_p_event=LOG_P_EVENT, log_q_event=LOG_Q_EVENT
    )
    assert value == pytest.approx(
        (4.521460917862246 + 50 + 8.521460917862246 + 1.521460917862246) / 4
    )
    # The cap applies per reference after the normalizing terms; a negative term stays negative.
    capped = estimators.capped_forward_kl(
        logp, logq, log_p_event=LOG_P_EVENT, log_q_event=LOG_Q_EVENT, cap=5.0
    )
    assert capped == pytest.approx((4.521460917862246 + 5 + 5 + 1.521460917862246) / 4)
    negative = estimators.capped_forward_kl(
        [-10.0], [-1.0], log_p_event=LOG_P_EVENT, log_q_event=LOG_Q_EVENT
    )
    assert negative == pytest.approx(-9.0 + 5.521460917862246)


def test_forward_is_kl_at_q_equal_p() -> None:
    # q = p: every term is log q̂(S) - log p̂(S).
    logp = [-3.0, -7.5, -20.0]
    value = estimators.capped_forward_kl(
        logp, logp, log_p_event=math.log(0.2), log_q_event=math.log(0.21)
    )
    assert value == pytest.approx(math.log(0.21 / 0.2))


def test_reverse_kl() -> None:
    value = estimators.reverse_kl(
        [-3.0, -4.0], [-20.0, -22.0], log_p_event=LOG_P_EVENT, log_q_event=LOG_Q_EVENT
    )
    assert value == pytest.approx(17.5 + LOG_P_EVENT - LOG_Q_EVENT)
    assert value == pytest.approx(17.5 - 6.907755278982137 + 1.3862943611198906)
    # A hit the original model cannot produce gives an infinite reverse KL (no cap).
    assert (
        estimators.reverse_kl([-1.0], [-math.inf], log_p_event=LOG_P_EVENT, log_q_event=LOG_Q_EVENT)
        == math.inf
    )


def test_exact_reverse_kl() -> None:
    masses, logq, logp = [0.5, 0.25], [math.log(0.5), math.log(0.25)], [-10.0, -11.0]
    log_p_event = -8.0
    q_s = [2 / 3, 1 / 3]
    expected = q_s[0] * (math.log(q_s[0]) - (-10 + 8)) + q_s[1] * (math.log(q_s[1]) - (-11 + 8))
    assert estimators.exact_reverse_kl(
        masses, logq, logp, log_p_event=log_p_event
    ) == pytest.approx(expected)
    # Occurrence form (mass 1/n per occurrence, q = count/n) gives the same value.
    occurrences = estimators.exact_reverse_kl(
        [0.25, 0.25, 0.25], [math.log(0.5), math.log(0.5), math.log(0.25)], [-10.0, -10.0, -11.0],
        log_p_event=log_p_event,
    )  # fmt: skip
    assert occurrences == pytest.approx(expected)


def test_threshold_boundaries() -> None:
    assert not estimators.meets_hit_rate_threshold(102, 1024)
    assert estimators.meets_hit_rate_threshold(103, 1024)
    assert estimators.meets_hit_rate_threshold(100, 1000)
    assert not estimators.meets_hit_rate_threshold(99, 1000)
    assert estimators.meets_hit_rate_threshold(1, 100, fractions.Fraction(1, 100))
    with pytest.raises(ValueError):
        estimators.meets_hit_rate_threshold(5, 4)


def test_fallback_scores() -> None:
    scores = estimators.fallback_scores(math.log(1e-5), math.log(1e-3))
    assert scores.rho == pytest.approx(0.01)
    assert scores.capped_forward_kl == pytest.approx(0.01 * math.log(0.01) + 0.99 * 50)
    assert scores.capped_forward_kl == pytest.approx(49.45394829814)
    assert scores.reverse_kl == pytest.approx(math.log(100))
    tiny = estimators.fallback_scores(-200.0, math.log(0.01))
    assert tiny.capped_forward_kl == pytest.approx(50.0)
    with pytest.raises(ValueError):
        estimators.fallback_scores(math.log(0.1), math.log(0.01))  # p(a) > p̂(S)


def test_choose_reverse_hits() -> None:
    hits = list(range(0, 600, 2))
    chosen = estimators.choose_reverse_hits(hits, 128, seed=7)
    assert len(chosen) == 128 and len(set(chosen)) == 128 and set(chosen) <= set(hits)
    assert chosen == sorted(chosen)
    assert chosen == estimators.choose_reverse_hits(hits, 128, seed=7)
    assert chosen != estimators.choose_reverse_hits(hits, 128, seed=8)
    assert estimators.choose_reverse_hits(hits[:50], 128, seed=7) == hits[:50]


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        estimators.capped_forward_kl([-1.0], [-1.0, -2.0], log_p_event=-1.0, log_q_event=-1.0)
    with pytest.raises(ValueError):
        estimators.capped_forward_kl([0.5], [-1.0], log_p_event=-1.0, log_q_event=-1.0)
    with pytest.raises(ValueError):
        estimators.capped_forward_kl([-1.0], [math.nan], log_p_event=-1.0, log_q_event=-1.0)
    with pytest.raises(ValueError):
        estimators.reverse_kl([-math.inf], [-1.0], log_p_event=-1.0, log_q_event=-1.0)
    with pytest.raises(ValueError):
        estimators.capped_forward_kl([-1.0], [-1.0], log_p_event=-1.0, log_q_event=-1.0, cap=0)


def test_mean_and_standard_error() -> None:
    mean, se = estimators.mean_and_standard_error([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert se == pytest.approx(math.sqrt(5 / 3) / 2)
    assert estimators.mean_and_standard_error([3.0]) == (3.0, None)
