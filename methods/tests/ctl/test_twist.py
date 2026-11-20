"""The twist network, the weights of an update and its loss."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from posteriorbench_methods.ctl import twist


def test_the_twist_network_starts_from_its_seed_with_zero_biases(tiny_model: Any) -> None:
    first = twist.TwistHead(tiny_model.lm_head, 7)
    again = twist.TwistHead(tiny_model.lm_head, 7)
    other = twist.TwistHead(tiny_model.lm_head, 8)
    assert torch.equal(first.out.weight, again.out.weight)
    assert not torch.equal(first.out.weight, other.out.weight)
    for layer in (first.fc1, first.fc2, first.out):
        assert float(layer.bias.detach().abs().max()) == 0
    assert first.inner is tiny_model.lm_head


def test_the_logits_of_the_twisted_model_are_the_original_logits_plus_log_psi(
    tiny_model: Any,
) -> None:
    head = twist.TwistHead(tiny_model.lm_head, 7)
    hidden = torch.randn(3, 5, 32, generator=torch.Generator().manual_seed(0))
    tokens = torch.tensor([[1, 4, 9, 2, 0]] * 3)
    with torch.no_grad():
        log_psi = head.log_psi(hidden)
        assert torch.equal(head(hidden), tiny_model.lm_head(hidden).float() + log_psi)
        chosen = log_psi.gather(2, tokens[:, :, None]).squeeze(2)
        assert torch.allclose(head.log_psi_of(hidden, tokens), chosen, atol=1e-6)


def test_absorb_repeats_the_last_value_of_a_response() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    assert twist.absorb(values, torch.tensor([2, 4])).tolist() == [
        [1.0, 2.0, 2.0, 2.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    with pytest.raises(ValueError):
        twist.absorb(values, torch.tensor([0, 4]))
    with pytest.raises(ValueError):
        twist.absorb(values, torch.tensor([2, 5]))


def test_weights_are_normalized_over_the_responses() -> None:
    prefix_logp = torch.tensor([[-1.0, -2.0], [-1.5, -1.0], [-0.5, -3.0]])
    prefix_logq = torch.tensor([[-1.2, -2.1], [-1.4, -1.3], [-0.7, -2.0]])
    log_psi = torch.tensor([[0.1, 0.2], [0.3, -0.1], [0.0, 0.4]])
    negative = twist.negative_weights(prefix_logp, prefix_logq, log_psi)
    assert torch.allclose(negative.sum(0), torch.ones(2))
    assert torch.allclose(negative, torch.softmax(prefix_logp + log_psi - prefix_logq, dim=0))
    hits = torch.tensor([True, False, False])
    positive = twist.positive_weights(prefix_logp[:, -1], prefix_logq[:, -1], hits, 1e-16)
    assert float(positive[0]) == pytest.approx(1.0) and float(positive[1:].max()) < 1e-10


def test_without_a_hit_the_positive_weights_are_importance_weights_toward_p() -> None:
    logp = torch.tensor([-1.0, -2.0, -3.0])
    logq = torch.tensor([-1.5, -1.0, -3.5])
    positive = twist.positive_weights(logp, logq, torch.tensor([False] * 3), 1e-16)
    assert torch.allclose(positive, torch.softmax(logp - logq, dim=0))
    with pytest.raises(ValueError):
        twist.positive_weights(logp, logq, torch.tensor([False] * 3), 0.0)


def test_the_gradient_of_the_loss_is_the_weight_difference_averaged_over_positions() -> None:
    log_psi = torch.tensor([[0.1, 0.2], [0.3, -0.1], [0.0, 0.4]], requires_grad=True)
    negative = torch.tensor([[0.2, 0.5], [0.3, 0.1], [0.5, 0.4]])
    positive = torch.tensor([0.7, 0.2, 0.1])
    loss = twist.contrastive_loss(log_psi, negative, positive)
    expected = (negative * log_psi).sum(0).mean() - (positive * log_psi.mean(1)).sum()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    loss.backward()
    assert log_psi.grad is not None
    assert torch.allclose(log_psi.grad, (negative - positive[:, None]) / 2)
    with pytest.raises(ValueError):
        twist.contrastive_loss(log_psi, negative[:, :1], positive)
