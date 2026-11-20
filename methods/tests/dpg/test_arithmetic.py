"""The arithmetic of the three methods on distributions small enough to enumerate: Z, the
coefficients of KL-DPG and JS-DPG, the weights of GDC and its fit check."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from posteriorbench_methods.dpg import fdpg, gdc, settings, training

OUTCOMES = 12


def toy(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A policy over 12 outcomes as trainable logits [12], the log-probabilities [12] of an
    original model p, and the hit flags [12] of a behavior S."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(OUTCOMES, generator=generator, dtype=torch.float64, requires_grad=True)
    logp = torch.randn(OUTCOMES, generator=generator, dtype=torch.float64).log_softmax(0)
    hit = torch.rand(OUTCOMES, generator=generator) < 0.4
    hit[0] = True
    return logits, logp, hit


def expected_gradient(logits: torch.Tensor, coefficients: np.ndarray) -> torch.Tensor:
    """The expectation over y ~ pi of a(y) grad log pi(y), the gradient that f-DPG estimates."""
    logq = logits.log_softmax(0)
    surrogate = (logq.exp().detach() * torch.as_tensor(coefficients) * logq).sum()
    (gradient,) = torch.autograd.grad(surrogate, logits)
    return gradient


@pytest.mark.parametrize("seed", range(5))
def test_kl_dpg_follows_the_gradient_of_the_forward_kl(seed: int) -> None:
    logits, logp, hit = toy(seed)
    logq = logits.log_softmax(0)
    log_z = float(logp[hit].logsumexp(0))  # log p(S)
    posterior = torch.where(hit, (logp - log_z).exp(), 0.0)
    forward_kl = (posterior[hit] * (logp[hit] - log_z - logq[hit])).sum()
    (gradient,) = torch.autograd.grad(forward_kl, logits)
    values = logq.detach().numpy()
    coefficients = fdpg.kl_coefficients(logp.numpy(), values, values, hit.numpy(), log_z)
    assert np.allclose(coefficients, 1 - (posterior / logq.exp()).detach().numpy())
    assert torch.allclose(expected_gradient(logits, coefficients), gradient)


@pytest.mark.parametrize("seed", range(5))
def test_js_dpg_follows_the_gradient_of_the_jensen_shannon_divergence(seed: int) -> None:
    """The divergence of Go et al.: sum of pi log(2 pi / (pi + p_S)) + p_S log(2 p_S / (pi +
    p_S)). The baseline does not change the expected gradient."""
    logits, logp, hit = toy(seed)
    logq = logits.log_softmax(0)
    q = logq.exp()
    log_z = float(logp[hit].logsumexp(0))
    posterior = torch.where(hit, (logp - log_z).exp(), 0.0)
    divergence = (q * (2 * q / (q + posterior)).log()).sum() + (
        posterior[hit] * (2 * posterior[hit] / (q[hit] + posterior[hit])).log()
    ).sum()
    (gradient,) = torch.autograd.grad(divergence, logits)
    values = logq.detach().numpy()
    for baseline in (0.0, 0.3):
        coefficients, _ = fdpg.js_coefficients(
            logp.numpy(), values, values, hit.numpy(), log_z, baseline=baseline, decay=0.99
        )
        assert torch.allclose(expected_gradient(logits, coefficients), gradient)


def test_js_dpg_updates_its_baseline_before_it_subtracts_it() -> None:
    logp = np.array([-3.0, -5.0, -4.0])
    logq = np.array([-2.0, -2.5, -3.0])
    hit = np.array([True, False, True])
    eta = np.where(hit, np.exp(logp - (-1.0) - logq), 0.0)
    derivative = math.log(2) - np.log1p(eta)
    coefficients, beta = fdpg.js_coefficients(logp, logq, logq, hit, -1.0, baseline=0.5, decay=0.9)
    assert beta == pytest.approx(0.9 * 0.5 + 0.1 * derivative.mean())
    assert np.allclose(coefficients, derivative - beta)


def test_the_coefficients_correct_for_the_rounding_between_the_two_values_of_log_pi() -> None:
    """With rho = pi_forward / pi_drawn: KL-DPG has rho (1 - eta), with eta from pi_forward."""
    logp, logq, logq_drawn = np.array([-3.0]), np.array([-2.0]), np.array([-2.1])
    hit, log_z = np.array([True]), -1.0
    rho, eta = math.exp(0.1), math.exp(-3.0 + 1.0 + 2.0)
    assert fdpg.kl_coefficients(logp, logq_drawn, logq, hit, log_z)[0] == pytest.approx(
        rho * (1 - eta)
    )
    coefficients, _ = fdpg.js_coefficients(
        logp, logq_drawn, logq, hit, log_z, baseline=0.0, decay=1.0
    )
    assert coefficients[0] == pytest.approx(rho * (math.log(2) - math.log1p(eta)))


@pytest.mark.filterwarnings("ignore:overflow")
def test_a_coefficient_that_is_not_finite_is_an_error() -> None:
    huge = np.array([800.0])
    with pytest.raises(FloatingPointError):
        fdpg.kl_coefficients(huge, np.array([0.0]), np.array([0.0]), np.array([True]), 0.0)


def test_z_is_the_mean_weight_over_all_draws_so_far() -> None:
    normalizer = training.Normalizer()
    normalizer.add([], draws=10)
    assert normalizer.log_z is None  # no hit yet: Z is zero
    normalizer.add([math.log(0.2), math.log(0.6)], draws=6)
    assert normalizer.log_z == pytest.approx(math.log(0.8 / 16))
    normalizer.add([math.log(0.4)], draws=4)
    assert normalizer.log_z == pytest.approx(math.log(1.2 / 20))
    assert (normalizer.draws, normalizer.log_weight_sum) == (20, pytest.approx(math.log(1.2)))


def test_the_learning_rate_rises_from_zero_over_the_warmup() -> None:
    paper = settings.FdpgSettings()
    rates = [training.learning_rate(updates, paper) for updates in (0, 1, 50, 100, 500)]
    assert rates == pytest.approx([0.0, 1.41e-7, 7.05e-6, 1.41e-5, 1.41e-5])
    no_warmup = settings.GdcSettings(warmup_updates=0)
    assert training.learning_rate(0, no_warmup) == 0.0
    assert training.learning_rate(1, no_warmup) == 1.41e-5


@pytest.mark.filterwarnings("ignore:overflow")
def test_the_weights_of_gdc_are_p_over_r() -> None:
    logp, logr = np.array([-4.0, -6.0]), np.array([-3.0, -6.5])
    assert np.allclose(gdc.importance_weights(logp, logr), np.exp([-1.0, 0.5]))
    with pytest.raises(FloatingPointError):
        gdc.importance_weights(np.array([800.0]), np.array([0.0]))


@pytest.mark.parametrize("seed", range(5))
def test_the_fit_check_estimates_the_forward_kl(seed: int) -> None:
    """With r uniform over the outcomes and each outcome drawn once, the draws are r exactly:
    Z is p(S) and the two estimates are the forward KL from the posterior to r and to pi."""
    logits, logp, hit = toy(seed)
    logpi = logits.detach().log_softmax(0)
    logr = torch.full((OUTCOMES,), -math.log(OUTCOMES), dtype=torch.float64)
    normalizer = training.Normalizer()
    normalizer.add((logp[hit] - logr[hit]).tolist(), draws=OUTCOMES)
    log_z = normalizer.log_z
    assert log_z is not None and log_z == pytest.approx(float(logp[hit].logsumexp(0)))
    posterior = (logp[hit] - log_z).exp()
    kl_r, kl_pi = gdc.forward_kl_estimates(
        logp[hit].numpy(), logr[hit].numpy(), logpi[hit].numpy(), draws=OUTCOMES, log_z=log_z
    )
    assert kl_r == pytest.approx(float((posterior * (logp[hit] - log_z - logr[hit])).sum()))
    assert kl_pi == pytest.approx(float((posterior * (logp[hit] - log_z - logpi[hit])).sum()))


def test_a_fit_check_replaces_r_only_if_the_policy_is_closer() -> None:
    assert gdc.FitCheck(96, 20, -1.5, forward_kl_r=1.2, forward_kl_pi=1.1).policy_is_closer
    assert not gdc.FitCheck(96, 20, -1.5, forward_kl_r=1.2, forward_kl_pi=1.2).policy_is_closer
    # While Z is zero there are no estimates, and r stays.
    assert not gdc.FitCheck(96, 0, None, None, None).policy_is_closer
