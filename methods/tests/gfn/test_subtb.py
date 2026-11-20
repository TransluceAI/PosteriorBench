"""The subtrajectory-balance loss of one response."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from posteriorbench_methods.gfn import subtb

EPSILON = 1e-8


def pairwise_loss(
    q: list[float],
    q_stop: list[list[float]],
    p: list[float],
    p_stop: list[list[float]],
    hits: list[bool],
    terminated: bool,
) -> tuple[float, int]:
    """The loss written out term by term: one squared difference for each pair of prefixes, and
    at each position one for each pair of stop tokens."""
    content = len(q) - int(terminated)
    log_factor = [0.0 if hit else math.log(EPSILON) for hit in hits]
    q_prefix, p_prefix = [0.0], [0.0]
    for t in range(content):
        q_prefix.append(q_prefix[-1] + q[t])
        p_prefix.append(p_prefix[-1] + p[t])
    endings = content + int(terminated)
    flows: list[float] = []
    stop_flows: list[list[float]] = []
    for i in range(endings):
        rewards = [p_prefix[i] + value + log_factor[i] for value in p_stop[i]]
        flows.append(float(np.logaddexp.reduce(rewards) - np.logaddexp.reduce(q_stop[i])))
        stop_flows.append([reward - value for reward, value in zip(rewards, q_stop[i])])
    if not terminated:
        flows.append(p_prefix[-1] + log_factor[-1])
    balance = [flow - prefix for flow, prefix in zip(flows, q_prefix, strict=True)]
    total, count = 0.0, 0
    for i in range(len(balance)):
        for j in range(i + 1, len(balance)):
            total += (balance[i] - balance[j]) ** 2
            count += 1
    for row in stop_flows:
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                total += (row[a] - row[b]) ** 2
                count += 1
    return total, count


@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize("stop_tokens", [1, 2, 3])
def test_terms_equal_the_loss_written_out(terminated: bool, stop_tokens: int) -> None:
    rng = np.random.default_rng(stop_tokens)
    length = 7
    q = (-rng.random(length) * 3).tolist()
    p = (-rng.random(length) * 3).tolist()
    q_stop = (-rng.random((length, stop_tokens)) * 5).tolist()
    p_stop = (-rng.random((length, stop_tokens)) * 5).tolist()
    hits = [bool(x) for x in rng.random(length - int(terminated) + 1) < 0.5]
    total, count = subtb.subtb_terms(
        torch.tensor(q, dtype=torch.float64),
        torch.tensor(q_stop, dtype=torch.float64),
        torch.tensor(p, dtype=torch.float64),
        torch.tensor(p_stop, dtype=torch.float64),
        hits,
        epsilon=EPSILON,
        terminated=terminated,
    )
    expected, expected_count = pairwise_loss(q, q_stop, p, p_stop, hits, terminated)
    assert float(total) == pytest.approx(expected, rel=1e-10)
    assert count == expected_count
    assert count == subtb.term_count(length, terminated=terminated, stop_tokens=stop_tokens)


@pytest.mark.parametrize("terminated", [True, False])
def test_a_policy_proportional_to_the_reward_has_no_loss(terminated: bool) -> None:
    """With every prefix a hit the reward is p, and the policy that equals p is balanced."""
    rng = np.random.default_rng(0)
    tokens = torch.tensor(-rng.random(6) * 3)
    stops = torch.tensor(-rng.random((6, 2)) * 5)
    hits = [True] * (6 - int(terminated) + 1)
    total, _ = subtb.subtb_terms(
        tokens, stops, tokens, stops, hits, epsilon=EPSILON, terminated=terminated
    )
    assert float(total) == pytest.approx(0.0, abs=1e-20)
    missed = [True] * (len(hits) - 1) + [False]
    total, _ = subtb.subtb_terms(
        tokens, stops, tokens, stops, missed, epsilon=EPSILON, terminated=terminated
    )
    assert float(total) > 1.0  # the same policy is far from balanced when the response is a miss


def test_scores_that_do_not_fit_are_refused() -> None:
    tokens, stops = torch.zeros(3), torch.zeros(3, 2)
    with pytest.raises(ValueError):
        subtb.subtb_terms(tokens, stops, tokens, stops, [True] * 3, epsilon=1e-8, terminated=False)
    with pytest.raises(ValueError):
        subtb.subtb_terms(tokens, stops, tokens, stops, [True] * 3, epsilon=1.0, terminated=True)
