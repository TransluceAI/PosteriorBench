"""SFT's draws from the original model, taken from the corpus.

The corpus is a sample of the original model, so a run does not generate its draws. Each draw is a
hit with probability p̂(S), independently, and the hits are the task's corpus hits outside the
references, in a random order. The j-th hit is draw number g_1 + ... + g_j with
g_i ~ Geometric(p̂(S)), so n draws hold Binomial(n, p̂(S)) hits. A task and replicate have one
sequence of draws, so a smaller budget reads a prefix of a larger one's; replicates are
independent.

Only this module reads p̂(S) and the number of the task's corpus hits. The method sees where its
hits are among its draws (`arrival`) and the hits themselves (`take`).
"""

from __future__ import annotations

import numpy as np
from posteriorbench import corpus, ledger

from posteriorbench_methods.common import data_source, errors, run_context

ARRIVALS_PER_BLOCK = 4096


class CorpusExhausted(errors.MethodsError, RuntimeError):
    """A run drew more hits than the task has corpus hits outside its references."""


class DrawsFromP:
    """The draws from the original model of one task and replicate.

    `access` hands out the task's corpus hits. `arrival_seed` seeds the places of the hits among
    the draws, and `order_seed` the order in which the corpus hits arrive."""

    def __init__(self, access: corpus.CorpusAccess, *, arrival_seed: int, order_seed: int) -> None:
        self._access = access
        self._hit_probability = access.task.probability
        self._gaps = np.random.default_rng(arrival_seed)
        # order[j] is the position in the task's hit sequence of the hit that arrives j-th.
        self._order = np.random.default_rng(order_seed).permutation(access.available)
        self._arrivals = np.zeros(0, dtype=np.int64)  # the places of the hits generated so far

    def arrival(self, j: int) -> int:
        """The place of the j-th hit among the draws, both counted from 1."""
        if j < 1:
            raise ValueError("hits are counted from 1")
        while self._arrivals.size < j:
            # Whole blocks from one generator, so the places do not depend on how far earlier
            # calls looked.
            gaps = self._gaps.geometric(self._hit_probability, size=ARRIVALS_PER_BLOCK)
            start = int(self._arrivals[-1]) if self._arrivals.size else 0
            self._arrivals = np.concatenate(
                [self._arrivals, start + np.cumsum(gaps.astype(np.int64))]
            )
        return int(self._arrivals[j - 1])

    def take(self, hits: int) -> list[corpus.CorpusHit]:
        """The first `hits` hits of the draws, in the order drawn. Raises CorpusExhausted if the
        task has fewer corpus hits."""
        if hits > self._order.size:
            raise CorpusExhausted(
                f"{self._access.task.task_id}: {hits} hits drawn, the corpus holds "
                f"{self._order.size}"
            )
        return self._access.take([int(position) for position in self._order[:hits]])

    @property
    def corpus_draws(self) -> int:
        """The number of responses of the task's corpus."""
        return int(self._access.task.num_responses)

    def take_all(self) -> list[corpus.CorpusHit]:
        """Every hit of the task's corpus that a method may take, in corpus order: the hits
        outside the task's references, without those whose tokens equal a reference's."""
        return self._access.take_first(self._access.available)

    def corpus_cost(self, hits: int) -> float:
        """What the benchmark charges for `hits` corpus hits: 1 / p̂(S) draws for each."""
        return self._access.cost_of(hits)


def open_draws(context: run_context.RunContext) -> DrawsFromP:
    """The draws of the run's task and replicate, from the task's hit file. The corpus access
    gets a ledger of its own, which the run does not read: the package charges 1 / p̂(S) for a
    corpus hit, while SFT pays for each of its draws."""
    with data_source.reading(context.options.data):
        access = context.bench.corpus(context.task, ledger.Ledger())
    return DrawsFromP(
        access,
        arrival_seed=context.stream_seed("sft", "draw-arrivals"),
        order_seed=context.stream_seed("sft", "pool-order"),
    )
