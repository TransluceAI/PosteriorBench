"""How SFT divides a budget between draws from the original model and training.

SFT makes all of its draws before it trains, so a run decides in advance how many to make. With a
budget of B passes, a training share s, batches of at most b responses and m selection draws for
each checkpoint, a run whose draws hold k hits

- sets aside 1,912 passes for its threshold tests (the tests at ranks 1 to 8);
- trains on T(k) = max(ceil(s (B - 1,912) / 2), k) responses, at two passes each (forward and
  backward): the share s of the budget, but at least one epoch over its hits;
- sets aside 3 m passes for each of V(T) = #{j >= 0: b 2^j < T} + 2 checkpoints (generating the
  selection draws, their log q, and log p of the hits among them).

What a run with k hits pays besides its draws is therefore at most

    C(k) = 2 T(k) + 3 m V(T(k)) + 1,912,

and a draw costs one pass. Let k(D) be the number of hits among the first D draws. The run makes
the largest number of draws D with D + C(k(D)) <= B (`divide`). C never decreases with k, and
k(D) never decreases with D, so every smaller number of draws fits the budget too. The run is
charged what it actually uses (`cost`), which is at most what the rule sets aside, so it spends
at most B.

Training saves a checkpoint after b 2^j responses for every j with b 2^j < T, after its last
response, and after the first epoch if the run has fewer than b hits (`checkpoint_examples`).

The rule reads the budget, the settings and the places of the run's own hits among its draws. It
never reads p̂(S).

A run whose draws are the whole corpus of its task has no number of draws to decide
(`divide_corpus`). Its D draws are the share 1 - s of a budget B = D / (1 - s), and the share s
pays for training: s B / 2 responses, read as E whole epochs over the k hits.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

from posteriorbench_methods.common import submission
from posteriorbench_methods.sft import settings as sft_settings

# Upper bounds of the charges in `cost`: an update makes a forward and a backward pass over each
# response, and a selection draw is generated, scored by the checkpoint and, if it is a hit, by the
# original model.
PASSES_PER_TRAINING_RESPONSE = 2
PASSES_PER_SELECTION_DRAW = 3


def training_examples(budget: int, settings: sft_settings.Settings, hits: int) -> int:
    """T(k): the responses that training reads, repeats counted."""
    after_tests = budget - submission.THRESHOLD_TEST_RESERVE
    if after_tests <= 0:
        return hits
    paid = math.ceil(settings.training_share * after_tests / PASSES_PER_TRAINING_RESPONSE)
    return max(paid, hits)


def checkpoint_examples(total: int, hits: int, batch_size: int) -> list[int]:
    """The numbers of training responses after which a checkpoint is saved, in increasing
    order."""
    points = {total}
    if hits < batch_size and hits < total:
        points.add(hits)
    doubling = batch_size
    while doubling < total:
        points.add(doubling)
        doubling *= 2
    return sorted(points)


def checkpoints_reserved(total: int, batch_size: int) -> int:
    """V(T): the number of checkpoints that the rule sets passes aside for. It is at least
    len(checkpoint_examples(total, hits, batch_size)) for every number of hits."""
    doublings = 0
    doubling = batch_size
    while doubling < total:
        doublings += 1
        doubling *= 2
    return doublings + 2


def reserved_passes(budget: int, settings: sft_settings.Settings, hits: int) -> int:
    """C(k): what a run with `hits` hits sets aside for training, the selection draws of its
    checkpoints and its threshold tests."""
    total = training_examples(budget, settings, hits)
    checkpoints = checkpoints_reserved(total, settings.batch_size)
    return (
        PASSES_PER_TRAINING_RESPONSE * total
        + PASSES_PER_SELECTION_DRAW * settings.selection_draws * checkpoints
        + submission.THRESHOLD_TEST_RESERVE
    )


@dataclasses.dataclass(frozen=True)
class Division:
    """How one run divides its budget.

    draws: the draws from the original model (the paper's n).
    hits: the hits among them (k), the training set.
    training_examples: the responses that training reads, repeats counted (T = E k for E epochs).
    batch: the responses in one update, min(hits, batch_size); the last update may have fewer.
    checkpoint_updates: the updates after which a checkpoint is saved; the last one ends training.
        Empty for a run without a hit, which does not train."""

    budget: int
    draws: int
    hits: int
    training_examples: int
    batch: int
    checkpoint_updates: tuple[int, ...]

    @property
    def updates(self) -> int:
        """The number of updates of training."""
        return self.checkpoint_updates[-1]

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict, with the number of epochs for a run that trains."""
        value = dataclasses.asdict(self) | {"checkpoint_updates": list(self.checkpoint_updates)}
        if self.hits:
            value["epochs"] = self.training_examples / self.hits
        return value


def _division(budget: int, settings: sft_settings.Settings, draws: int, hits: int) -> Division:
    total = training_examples(budget, settings, hits)
    if hits == 0:
        return Division(budget, draws, 0, total, 0, ())
    batch = min(hits, settings.batch_size)
    updates = {
        math.ceil(examples / batch)
        for examples in checkpoint_examples(total, hits, settings.batch_size)
    }
    return Division(budget, draws, hits, total, batch, tuple(sorted(updates)))


def corpus_epochs(settings: sft_settings.Settings, corpus_draws: int, hits: int) -> int:
    """E: the epochs of a run whose draws are a corpus of `corpus_draws` responses with `hits`
    training responses. The training share s of the budget corpus_draws / (1 - s) pays for
    s corpus_draws / (2 (1 - s)) responses; E is the whole number of epochs nearest to that, and
    at least 1."""
    share = settings.training_share
    paid = share * corpus_draws / (PASSES_PER_TRAINING_RESPONSE * (1 - share))
    return max(1, math.floor(paid / hits + 0.5))


def corpus_checkpoint_epochs(epochs: int, hits: int, batch_size: int) -> list[int]:
    """The epochs after which a run on a corpus saves a checkpoint, in increasing order: the
    first, the whole number of epochs nearest to batch_size 2^j responses for every j with fewer
    responses than training reads, and the last."""
    points = {1, epochs}
    doubling = batch_size
    while doubling < epochs * hits:
        nearest = math.floor(doubling / hits + 0.5)
        if 1 <= nearest < epochs:
            points.add(nearest)
        doubling *= 2
    return sorted(points)


def divide_corpus(settings: sft_settings.Settings, *, corpus_draws: int, hits: int) -> Division:
    """The division of a run whose draws are the `corpus_draws` responses of its task's corpus
    and whose training set is its `hits` hits outside the references.

    The budget, ceil(corpus_draws / (1 - s)), names the run and is not a limit: nothing is set
    aside for selection draws and threshold tests. A batch can end one epoch and start the next;
    a checkpoint is saved after the update that completes each epoch of
    `corpus_checkpoint_epochs`. Raises ValueError for a corpus without a hit."""
    if hits < 1:
        raise ValueError("a run on a corpus needs at least one hit to train on")
    budget = math.ceil(corpus_draws / (1 - settings.training_share))
    epochs = corpus_epochs(settings, corpus_draws, hits)
    batch = min(hits, settings.batch_size)
    updates = {
        math.ceil(epoch * hits / batch)
        for epoch in corpus_checkpoint_epochs(epochs, hits, settings.batch_size)
    }
    return Division(budget, corpus_draws, hits, epochs * hits, batch, tuple(sorted(updates)))


def divide(budget: int, settings: sft_settings.Settings, arrival: Callable[[int], int]) -> Division:
    """The division of `budget`: the largest number of draws D with D + C(k(D)) <= budget.

    `arrival(j)` is the place of the run's j-th hit among its draws (both counted from 1); it
    increases with j. A budget too small for a single hit gives a division without hits."""
    hits = 0
    last_arrival = 0  # the place of hit number `hits`; 0 before the first hit
    while True:
        draws = budget - reserved_passes(budget, settings, hits)  # what `hits` hits leave for draws
        if draws < last_arrival:
            # The run cannot pay for the draws up to its last hit and for training on it: it stops
            # one draw before that hit.
            hits, draws = (hits - 1, last_arrival - 1) if hits else (0, 0)
            break
        next_arrival = arrival(hits + 1)
        assert next_arrival > last_arrival, "the places of the hits increase"
        if draws < next_arrival:
            break
        hits, last_arrival = hits + 1, next_arrival
    return _division(budget, settings, draws, hits)
