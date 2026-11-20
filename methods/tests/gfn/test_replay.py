"""The replay buffer and its edit distance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from posteriorbench_methods.gfn import edit_distance, replay, reward

STOP = 9
EPSILON = 1e-8


def response(token_ids: list[int], text: str, logp: float, hit: bool = True) -> reward.Response:
    return reward.Response(token_ids, text, hit, logp, [], [], [])


def buffer(size: int = 50) -> replay.ReplayBuffer:
    return replay.ReplayBuffer(size=size, similarity=0.25, epsilon=EPSILON, stop_token_ids=[STOP])


def test_edit_distance_is_the_levenshtein_distance() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.integers(0, 5, rng.integers(0, 12)).tolist()
        b = rng.integers(0, 5, rng.integers(0, 12)).tolist()
        previous = list(range(len(b) + 1))
        for i, left in enumerate(a, 1):
            current = [i]
            for j, right in enumerate(b, 1):
                current.append(
                    min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
                )
            previous = current
        assert edit_distance.distance(a, b) == previous[-1]


def test_a_near_duplicate_stays_only_if_it_is_better() -> None:
    held = buffer()
    held.add(response([1, 2, 3, 4, STOP], "abcd", -5.0))
    held.add(response([1, 2, 3, 5, STOP], "abce", -1.0))  # one substitution away, better
    assert [r.text for r in held.responses()] == ["abce"]
    held.add(response([1, 2, 3, 6, STOP], "abcf", -3.0))  # near the held one, worse
    held.add(response([1, 2, 3, 7, STOP], "abcg", -1.0))  # near the held one, equal: it stays
    assert [r.text for r in held.responses()] == ["abce"]


def test_a_text_is_held_once() -> None:
    held = buffer()
    held.add(response([1, 2, 3, 4, STOP], "abcd", -5.0))
    held.add(response([6, 7, 8, 6, 7, 8], "abcd.", -1.0))  # the same text but for a period
    assert [r.logp for r in held.responses()] == [-5.0]


def test_a_miss_counts_epsilon_times_its_probability() -> None:
    hit, miss = response([1, 2, STOP], "ab", -30.0), response([3, 4, STOP], "cd", -20.0, False)
    assert miss.log_reward(EPSILON) == pytest.approx(-20.0 + math.log(EPSILON))
    held = buffer(size=1)
    held.add(miss)
    held.add(hit)
    assert held.responses() == [hit]


def test_a_full_buffer_keeps_the_highest_rewards() -> None:
    held = buffer(size=3)
    for index, logp in enumerate([-4.0, -2.0, -6.0, -1.0, -9.0]):
        # Responses of one distinct token each: no two are near-duplicates.
        held.add(response([index] * 4, f"text {index}", logp))
    assert sorted(r.logp for r in held.responses()) == [-4.0, -2.0, -1.0]
    assert len(held) == 3


def test_sampling_is_uniform_with_replacement_and_restarts_from_its_responses() -> None:
    held = buffer()
    for index in range(5):
        held.add(response([index] * 4, f"text {index}", -float(index)))
    drawn = held.sample(np.random.default_rng(0), 4000)
    counts = [sum(r.text == f"text {index}" for r in drawn) for index in range(5)]
    assert min(counts) > 700
    again = replay.ReplayBuffer(
        size=50, similarity=0.25, epsilon=EPSILON, stop_token_ids=[STOP], held=held.responses()
    )
    assert again.responses() == held.responses()
    assert again.sample(np.random.default_rng(3), 20) == held.sample(np.random.default_rng(3), 20)
    with pytest.raises(ValueError):
        buffer().sample(np.random.default_rng(0), 1)
