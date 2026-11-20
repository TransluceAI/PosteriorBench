"""The twist of contrastive twist learning (CTL; Zhao et al., 2024), and the loss of its update.

A twist psi_t scores the prefixes of a response. The twisted model draws the token a after the
prefix y_<t with probability proportional to p(a | y_<t) psi_t(y_<t a), where p is the frozen
original model. If psi_t(y_<=t) were the probability under p that a response with this prefix ends
up a hit, the twisted model would sample the posterior exactly.

Here log psi_t(y_<t a), for every token a, is the output of a small network on the model's last
hidden state after y_<t (`TwistHead`). The logits of the twisted model are then the logits of the
original model plus the output of the network.

CTL fits the twist on N responses y_1, ..., y_N of the twisted model itself. It descends the mean
over the prefix lengths t of

    sum_i omega^-_{i,t} log psi_t(y_{i,<=t}) - sum_i omega^+_i log psi_t(y_{i,<=t})

with the weights held fixed (`contrastive_loss`). The weights omega^-_{i,t}, normalized over i
for each t, make the responses a sample of the prefixes that the twist defines, whose distribution
is proportional to p(y_<=t) psi_t(y_<=t) (`negative_weights`). The weights omega^+_i, normalized
over i, make them a sample of the posterior (`positive_weights`). A response that has ended keeps
its last prefix at every later t (`absorb`), so every response has T prefixes, where T is the
response limit.

Shapes: N responses, T positions, H the hidden size, V the size of the model's output layer.
"""

from __future__ import annotations

from typing import Any

import torch


class TwistHead(torch.nn.Module):
    """The output layer of the twisted model: the output layer of the original model, `inner`,
    plus the twist network, three linear layers with ReLUs between them (H -> H -> H -> V,
    float32).

    The network starts with Xavier-normal weights, drawn from a generator with `seed`, and zero
    biases. `inner` is used as it is; the twisted model keeps it frozen."""

    def __init__(self, inner: Any, seed: int) -> None:
        super().__init__()
        self.inner = inner
        vocabulary, hidden = inner.weight.shape
        self.fc1 = torch.nn.Linear(hidden, hidden, dtype=torch.float32)
        self.fc2 = torch.nn.Linear(hidden, hidden, dtype=torch.float32)
        self.out = torch.nn.Linear(hidden, vocabulary, dtype=torch.float32)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2, self.out):
                torch.nn.init.xavier_normal_(layer.weight, generator=generator)
                layer.bias.zero_()

    def _features(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(hidden.float()))))

    def log_psi(self, hidden: torch.Tensor) -> torch.Tensor:
        """log psi of every next token: hidden states [..., H] -> [..., V]."""
        return self.out(self._features(hidden))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """The logits of the twisted model: hidden states [..., H] -> [..., V] float32."""
        return self.inner(hidden).float() + self.log_psi(hidden)

    def log_psi_of(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """log psi of the given next tokens: hidden states [N, T, H] and tokens [N, T] ->
        [N, T], without the [N, T, V] output of `log_psi`.

        The gradient of the last layer adds up over the rows that `tokens` picks, in an order
        that torch fixes only under `torch.use_deterministic_algorithms(True)`: without it, two
        runs of the same update can differ in the last bits."""
        features = self._features(hidden)
        return (features * self.out.weight[tokens]).sum(-1) + self.out.bias[tokens]


def absorb(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """`values` [N, T, ...] with every position after the end of a response set to the value at
    its last token: position t of response i takes the value at min(t, lengths[i] - 1). Raises
    ValueError for a length outside 1..T."""
    if bool((lengths < 1).any()) or bool((lengths > values.shape[1]).any()):
        raise ValueError("the length of a response lies in 1..T")
    index = torch.arange(values.shape[1], device=values.device)[None, :]
    index = torch.minimum(index, lengths[:, None] - 1)
    shape = (*index.shape, *([1] * (values.ndim - 2)))
    return values.gather(1, index.reshape(shape).expand_as(values))


def negative_weights(
    prefix_logp: torch.Tensor, prefix_logq: torch.Tensor, log_psi: torch.Tensor
) -> torch.Tensor:
    """omega^- [N, T]: for each t, the softmax over the responses of
    log p(y_<=t) + log psi_t(y_<=t) - log q(y_<=t), all [N, T], where q is the twisted model that
    drew the responses. No gradient."""
    return torch.softmax((prefix_logp + log_psi - prefix_logq).detach(), dim=0)


def positive_weights(
    logp: torch.Tensor, logq: torch.Tensor, hits: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """omega^+ [N]: the softmax over the responses of log p(y) - log q(y) + log(1_S(y) + epsilon),
    from log p and log q of the whole responses [N] and their hit flags [N]. No gradient. Raises
    ValueError unless epsilon is positive: with epsilon = 0 an update without a hit has no
    weights."""
    if not epsilon > 0:
        raise ValueError("epsilon must be positive")
    return torch.softmax((logp - logq + torch.log(hits.to(logp.dtype) + epsilon)).detach(), dim=0)


def contrastive_loss(
    log_psi: torch.Tensor, negative_weight: torch.Tensor, positive_weight: torch.Tensor
) -> torch.Tensor:
    """The loss of an update, a scalar whose gradient is the mean over t of
    sum_i (omega^-_{i,t} - omega^+_i) grad log psi_t(y_{i,<=t}).

    log_psi: [N, T], log psi_t of every prefix of every response, absorbed past its end.
    negative_weight: [N, T]; positive_weight: [N]. Both are held fixed."""
    if log_psi.shape != negative_weight.shape or positive_weight.shape != log_psi.shape[:1]:
        raise ValueError("log psi and omega^- are [N, T], and omega^+ is [N]")
    negative = (negative_weight.detach() * log_psi).sum(0).mean()
    positive = (positive_weight.detach() * log_psi.mean(1)).sum()
    return negative - positive
