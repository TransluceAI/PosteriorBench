"""How SFT divides a budget: the largest number of draws that fits, the training share, the
checkpoints, and that the passes set aside cover the charges."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import charges, submission
from posteriorbench_methods.sft import budget_rule, cost, settings

PAPER = settings.Settings()


def arrivals(gaps: list[int]) -> Callable[[int], int]:
    """The places of the hits of a run whose j-th hit comes `gaps[j - 1]` draws after the one
    before; no hit after the listed ones."""
    places = np.cumsum(gaps).tolist()

    def arrival(j: int) -> int:
        if j <= len(places):
            return int(places[j - 1])
        return int(places[-1]) + (j - len(places)) * 10**9

    return arrival


def largest_draws_that_fit(
    budget: int, chosen: settings.Settings, arrival: Callable[[int], int]
) -> tuple[int, int]:
    """The rule by trying every number of draws D: (the largest D with D + C(k(D)) <= budget,
    k(D)), or (0, 0) if none fits."""
    best = (0, 0)
    hits = 0
    for draws in range(budget + 1):
        while arrival(hits + 1) <= draws:
            hits += 1
        if draws + budget_rule.reserved_passes(budget, chosen, hits) > budget:
            break
        best = (draws, hits)
    return best


@pytest.mark.parametrize("seed", range(40))
def test_divide_makes_the_largest_number_of_draws_that_fits(seed: int) -> None:
    rng = np.random.default_rng(seed)
    budget = int(rng.integers(1_500, 30_000))
    chosen = settings.Settings(
        training_share=float(rng.choice([1 / 32, 1 / 8, 1 / 4, 1 / 2])),
        batch_size=int(rng.choice([8, 32])),
        selection_draws=int(rng.choice([16, 128])),
    )
    hit_rate = float(rng.choice([0.002, 0.02, 0.2, 0.6]))
    arrival = arrivals(rng.geometric(hit_rate, size=20_000).tolist())
    division = budget_rule.divide(budget, chosen, arrival)
    assert (division.draws, division.hits) == largest_draws_that_fit(budget, chosen, arrival)
    if division.hits:
        # The hits are those among the draws, and the division fits the budget.
        assert arrival(division.hits) <= division.draws < arrival(division.hits + 1)
        reserved = budget_rule.reserved_passes(budget, chosen, division.hits)
        assert division.draws + reserved <= budget


def test_training_gets_its_share_and_at_least_one_epoch() -> None:
    budget = 2**20
    share = math.ceil(0.25 * (budget - submission.THRESHOLD_TEST_RESERVE) / 2)
    assert budget_rule.training_examples(budget, PAPER, 10) == share == 130_833
    assert budget_rule.training_examples(budget, PAPER, share + 7) == share + 7
    assert budget_rule.training_examples(1_000, PAPER, 3) == 3  # below the reserve for the tests


def test_a_budget_too_small_for_a_hit_gives_no_training() -> None:
    division = budget_rule.divide(4096, PAPER, arrivals([10, 10, 10]))
    assert (division.draws, division.hits, division.checkpoint_updates) == (0, 0, ())
    assert "epochs" not in division.to_json()


def test_checkpoints() -> None:
    # 100 hits: batches of 32; checkpoints after 32, 64, 128, 256 and 300 responses.
    assert budget_rule.checkpoint_examples(300, 100, 32) == [32, 64, 128, 256, 300]
    # 20 hits: batches of 20, and a checkpoint after the first epoch.
    assert budget_rule.checkpoint_examples(300, 20, 32) == [20, 32, 64, 128, 256, 300]
    division = budget_rule.divide(6000, PAPER, arrivals([40] * 20 + [10**9]))
    assert (division.draws, division.hits, division.batch) == (762, 19, 19)
    assert division.training_examples == 511
    assert division.checkpoint_updates == (1, 2, 4, 7, 14, 27)
    assert division.updates == 27 == math.ceil(511 / 19)
    assert division.to_json()["epochs"] == 511 / 19


def test_passes_are_set_aside_for_every_checkpoint() -> None:
    for total in [1, 2, 31, 32, 33, 64, 1000, 131_072, 262_145]:
        for hits in [1, 5, 31, 32, 40, 10_000]:
            if hits > total:
                continue
            points = budget_rule.checkpoint_examples(total, hits, 32)
            assert points == sorted(set(points)) and points[-1] == total
            assert len(points) <= budget_rule.checkpoints_reserved(total, 32)


def budget_count(specs: list[charges.ChargeSpec]) -> float:
    spend = ledger.Ledger()
    charges.apply(spend, specs)
    return spend.budget


def test_the_passes_set_aside_cover_the_charges() -> None:
    assert budget_count(cost.training(32)) == 32 * budget_rule.PASSES_PER_TRAINING_RESPONSE
    for hits in (0, 21, 128):
        assert (
            budget_count(cost.checkpoint_selection_draws(128, hits))
            <= 128 * budget_rule.PASSES_PER_SELECTION_DRAW
        )
    assert budget_count(cost.draws_from_p(1000)) == 1000


def test_a_run_on_a_whole_corpus_has_the_papers_epochs_and_checkpoints() -> None:
    """The paper's two tasks with long responses: 3.4 million draws, a quarter of the budget for
    training, 696 and 2,860 training responses."""
    draws = 3_400_000
    for hits, epochs, checkpoints in (
        (696, 814, [1, 3, 6, 12, 24, 47, 94, 188, 377, 753, 814]),
        (2860, 198, [1, 3, 6, 11, 23, 46, 92, 183, 198]),
    ):
        assert budget_rule.corpus_epochs(PAPER, draws, hits) == epochs
        assert budget_rule.corpus_checkpoint_epochs(epochs, hits, 32) == checkpoints
        division = budget_rule.divide_corpus(PAPER, corpus_draws=draws, hits=hits)
        # The draws are three quarters of the budget, and training reads whole epochs.
        assert division.budget == math.ceil(draws / 0.75) == 4_533_334
        assert (division.draws, division.hits, division.batch) == (draws, hits, 32)
        assert division.training_examples == epochs * hits
        assert abs(division.training_examples - 0.25 * division.budget / 2) < hits
        # A checkpoint after the update that completes each of these epochs.
        assert list(division.checkpoint_updates) == [
            math.ceil(epoch * hits / 32) for epoch in checkpoints
        ]
        assert division.updates == math.ceil(epochs * hits / 32)
        assert division.to_json()["epochs"] == epochs


def test_a_small_corpus_trains_for_one_epoch_at_least() -> None:
    division = budget_rule.divide_corpus(PAPER, corpus_draws=100, hits=40)
    assert (division.training_examples, division.batch) == (40, 32)
    assert division.checkpoint_updates == (2,)
    few = budget_rule.divide_corpus(PAPER, corpus_draws=5000, hits=5)
    # 833 responses are paid for: 167 epochs over 5 hits, in batches of 5.
    assert (few.batch, few.training_examples) == (5, 167 * 5)
    assert budget_rule.corpus_checkpoint_epochs(167, 5, 32) == [1, 6, 13, 26, 51, 102, 167]
    with pytest.raises(ValueError, match="at least one hit"):
        budget_rule.divide_corpus(PAPER, corpus_draws=100, hits=0)
