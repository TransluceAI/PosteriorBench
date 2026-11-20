"""The CE method: score the candidates on a bank of selection draws that they share.

The bank holds min(n, m K) of the m K selection draws of a pool's K candidates, chosen uniformly
without replacement. A candidate i whose hit fraction is above 15% is scored by

    sum_j w_j log q_i(y_j) / sum_j w_j,    w_j = p(y_j) / q_s(j)(y_j),

over the hits y_j in the bank, where q_s(j) is the candidate that drew y_j. The score estimates
E_{y ~ p_S}[log q_i(y)], which is higher for a candidate closer to the posterior.

The method pays one inference for p on each hit in the bank, and one for q_i on each pair of a
scored candidate i and a hit response in the bank that i did not draw. A response that is in the
bank several times is one response for the second count, and a candidate has the q of its own
draws from generating them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np


class CEError(ValueError):
    """The bank holds no hit, or a hit's log p or log q is not finite."""


@dataclasses.dataclass(frozen=True)
class BankHit:
    """Draw `draw` of pool position `candidate`. `response` indexes `Bank.responses`; `logq` is
    under the candidate that drew it."""

    candidate: int
    draw: int
    response: int
    logp: float
    logq: float


@dataclasses.dataclass(frozen=True)
class Bank:
    """The bank of one pool; candidates are positions in the pool.

    `draws` are (candidate, draw) pairs in the order chosen, `hits` the hits among them in the
    same order (a response drawn twice is there twice). `responses` are the distinct hit token
    ids in order of first appearance; `drawn_by[t]` is the candidates that drew response t."""

    draws: tuple[tuple[int, int], ...]
    hits: tuple[BankHit, ...]
    responses: tuple[tuple[int, ...], ...]
    drawn_by: tuple[frozenset[int], ...]

    def cross_scored_pairs(self, scored: Sequence[int]) -> int:
        """The pairs of a candidate of `scored` and a distinct hit response of the bank that the
        candidate did not draw: the q that the method pays for."""
        candidates = set(scored)
        own = sum(len(drawn_by & candidates) for drawn_by in self.drawn_by)
        return len(candidates) * len(self.responses) - own


def choose_bank(pool_size: int, draws: int, bank_size: int, seed: int) -> list[tuple[int, int]]:
    """min(bank_size, draws x pool_size) (candidate, draw) pairs, uniformly without replacement,
    in the order of `np.random.default_rng(seed).permutation`."""
    order = np.random.default_rng(seed).permutation(pool_size * draws)
    return [divmod(int(index), draws) for index in order[: min(bank_size, pool_size * draws)]]


def build_bank(
    chosen: Sequence[tuple[int, int]],
    *,
    token_ids: Sequence[Sequence[Sequence[int]]],
    hit: Sequence[Sequence[bool]],
    logp: Sequence[Sequence[float]],
    logq: Sequence[Sequence[float]],
) -> Bank:
    """The bank of the draws `chosen` (`choose_bank`). The other arguments are the pool's
    selection draws, indexed [candidate][draw]; `logp` and `logq` are read on hits only."""
    index_of: dict[tuple[int, ...], int] = {}
    drawn_by: list[set[int]] = []
    hits: list[BankHit] = []
    for candidate, draw in chosen:
        if not hit[candidate][draw]:
            continue
        response = tuple(int(token) for token in token_ids[candidate][draw])
        if response not in index_of:
            index_of[response] = len(index_of)
            drawn_by.append(set())
        drawn_by[index_of[response]].add(candidate)
        hits.append(
            BankHit(
                candidate,
                draw,
                response=index_of[response],
                logp=float(logp[candidate][draw]),
                logq=float(logq[candidate][draw]),
            )
        )
    return Bank(tuple(chosen), tuple(hits), tuple(index_of), tuple(frozenset(s) for s in drawn_by))


def scores(
    bank: Bank, scored: Sequence[int], candidate_logq: Mapping[int, Sequence[float]]
) -> list[float]:
    """The CE score of each pool position in `scored`, higher is better. `candidate_logq[i][t]`
    is log q_i of `bank.responses[t]`. Raises CEError if the bank cannot be scored."""
    if not bank.hits:
        raise CEError("no hit of the pool is in the bank of the CE method")
    log_weights = np.asarray([hit.logp - hit.logq for hit in bank.hits], dtype=float)
    # [scored candidates, hits in the bank]: log q_i of each hit, a repeated response repeated.
    logq = np.asarray(
        [[candidate_logq[candidate][hit.response] for hit in bank.hits] for candidate in scored],
        dtype=float,
    )
    if not (np.isfinite(log_weights).all() and np.isfinite(logq).all()):
        raise CEError("a log p or log q of a hit in the bank is not finite")
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    return [float(score) for score in logq @ weights]
