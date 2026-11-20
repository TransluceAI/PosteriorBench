"""What Steering pays for, summed through the package's ledger."""

from __future__ import annotations

from posteriorbench import ledger

from posteriorbench_methods.common import charges
from posteriorbench_methods.steering import cost, vectors

# The vectors of a behavior whose filter kept 13 + 2 positive and 175 negative responses.
COST = vectors.ConstructionCost(
    refusal_direction_passes=16_620,
    construction_responses=3_000,
    kept_responses={
        "positive_with_ablation": 13,
        "positive_without_ablation": 2,
        "negative": 175,
    },
)


def test_the_vectors_and_the_refusal_direction_are_charged_in_full() -> None:
    spend = ledger.Ledger()
    charges.apply(spend, cost.vector_construction(COST))
    totals = spend.totals()
    assert totals.inferences == totals.budget == 16_620 + 3_000 + 190 == 19_810
    assert totals.by_category["shared"] == (19_810.0, 19_810.0)
    assert [charge.inferences for charge in spend.charges()] == [16_620, 3_000, 190]


def test_a_run_of_97_candidates_whose_first_test_passes() -> None:
    """The vectors, 97 x 128 selection draws with 1,104 hits among them, and one test of 164
    draws."""
    spend = ledger.Ledger()
    charges.apply(spend, cost.vector_construction(COST))
    hits = [12] * 92 + [0] * 5
    assert len(hits) == 97 and sum(hits) == 1_104
    for candidate_hits in hits:
        charges.apply(spend, charges.selection_draws(128, candidate_hits))
    charges.apply(spend, charges.threshold_test(164))
    assert spend.inferences == spend.budget == 19_810 + 12_416 + 1_104 + 164 == 33_494
