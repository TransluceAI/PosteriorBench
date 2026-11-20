"""The CE method: the bank, the score and the pairs that it pays for."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from posteriorbench_methods.prompt import ce

M = 16


@dataclasses.dataclass(frozen=True)
class PoolDraws:
    """The selection draws of a pool as `ce.build_bank` reads them, indexed [candidate][draw]."""

    token_ids: list[list[tuple[int, ...]]]
    hit: list[list[bool]]
    logp: list[list[float]]
    logq: list[list[float]]

    def bank(self, bank_size: int, seed: int) -> ce.Bank:
        chosen = ce.choose_bank(len(self.hit), M, bank_size, seed)
        return ce.build_bank(
            chosen, token_ids=self.token_ids, hit=self.hit, logp=self.logp, logq=self.logq
        )


def pool_draws(pool_size: int, seed: int = 2, hit_rate: float = 0.4) -> PoolDraws:
    """Selection draws of a pool. Candidates 0 and 1 draw the same responses; every other
    response is distinct."""
    rng = np.random.default_rng(seed)
    return PoolDraws(
        token_ids=[
            [(7, d, 2) if c in (0, 1) else (c, d, 2) for d in range(M)] for c in range(pool_size)
        ],
        hit=(rng.random((pool_size, M)) < hit_rate).tolist(),
        logp=rng.normal(-20, 2, (pool_size, M)).tolist(),
        logq=rng.normal(-15, 2, (pool_size, M)).tolist(),
    )


def test_the_bank_is_a_uniform_subset_of_the_selection_draws() -> None:
    whole = ce.choose_bank(4, M, 512, seed=5)
    assert sorted(whole) == [(c, d) for c in range(4) for d in range(M)]  # min(512, m K)
    small = ce.choose_bank(4, M, 10, seed=5)
    assert small == whole[:10]
    assert ce.choose_bank(4, M, 10, seed=6) != small
    assert len(ce.choose_bank(1024, M, 512, seed=5)) == 512


def test_hits_and_cross_scored_pairs() -> None:
    draws = pool_draws(6)
    bank = draws.bank(512, seed=1)
    hit = np.asarray(draws.hit)
    assert len(bank.hits) == int(hit.sum())
    assert all(hit[h.candidate, h.draw] for h in bank.hits)
    responses = {draws.token_ids[c][d] for c in range(6) for d in range(M) if hit[c, d]}
    assert set(bank.responses) == responses
    scored = [0, 1, 3]
    own = sum(len(drawn_by & set(scored)) for drawn_by in bank.drawn_by)
    assert bank.cross_scored_pairs(scored) == len(scored) * len(responses) - own
    # A response that candidates 0 and 1 both drew is paid for by neither of them.
    shared = [drawn_by for drawn_by in bank.drawn_by if drawn_by == frozenset({0, 1})]
    assert shared and bank.cross_scored_pairs([0, 1]) <= 2 * (len(responses) - len(shared))
    assert bank.cross_scored_pairs([]) == 0


def test_the_score_is_the_weighted_mean_log_q() -> None:
    bank = pool_draws(5, seed=3).bank(40, seed=0)
    rng = np.random.default_rng(0)
    candidate_logq = {c: rng.normal(-9, 1, len(bank.responses)).tolist() for c in (1, 4)}
    scores = ce.scores(bank, [1, 4], candidate_logq)
    weights = np.exp([h.logp - h.logq for h in bank.hits])
    for candidate, score in zip((1, 4), scores, strict=True):
        logq = np.array([candidate_logq[candidate][h.response] for h in bank.hits])
        assert math.isclose(score, float((weights * logq).sum() / weights.sum()))


def test_a_bank_without_hits_cannot_score() -> None:
    bank = pool_draws(3, hit_rate=0.0).bank(512, seed=1)
    assert bank.hits == () and bank.responses == ()
    with pytest.raises(ce.CEError, match="no hit"):
        ce.scores(bank, [0], {0: []})
