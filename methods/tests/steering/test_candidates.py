"""The candidates of a task: their number, their order and their names."""

from __future__ import annotations

import pytest

from posteriorbench_methods.common import composition
from posteriorbench_methods.steering import candidates, settings

BOTH = ("with_ablation", "without_ablation")
PAPER = settings.Settings()


def test_the_paper_settings_give_97_candidates_in_order() -> None:
    found = candidates.candidates(PAPER, BOTH)
    assert len(found) == 97 and len({candidate.key for candidate in found}) == 97
    assert found[0] is candidates.ORIGINAL_MODEL and found[0].key == "original"
    assert [c.vector for c in found[1:49]] == ["with_ablation"] * 48
    assert [c.vector for c in found[49:]] == ["without_ablation"] * 48
    # Within a vector the strengths, and within a strength the layers.
    assert [(c.strength, c.layer) for c in found[1:8]] == [
        (1.0, 7), (1.0, 8), (1.0, 9), (1.0, 11), (1.0, 12), (1.0, 13), (1.5, 7)
    ]  # fmt: skip
    assert found[7].key == "with_ablation_layer7_strength1.5"
    assert found[-1].key == "without_ablation_layer13_strength8"


def test_a_behavior_with_one_vector_has_its_candidates_only() -> None:
    found = candidates.candidates(PAPER, ["without_ablation"])
    assert len(found) == 49
    assert {candidate.vector for candidate in found[1:]} == {"without_ablation"}
    assert candidates.candidates(PAPER, []) == [candidates.ORIGINAL_MODEL]
    with pytest.raises(ValueError, match="the vectors are"):
        candidates.candidates(PAPER, ["another"])


def test_the_default_layers_are_those_of_the_models_depth() -> None:
    """The benchmark's model has 24 layers; a model with another number has its candidates at
    the same depths, and stated layers are the same on every model."""
    on_64 = candidates.candidates(PAPER, ["without_ablation"], num_layers=64)
    assert [c.layer for c in on_64[1:7]] == [21, 24, 27, 31, 34, 37] and len(on_64) == 49
    stated = settings.Settings(layers=(3, 5), strengths=(2.0,))
    assert [c.layer for c in candidates.candidates(stated, BOTH, num_layers=64)[1:]] == [3, 5] * 2


def test_the_original_model_is_the_candidate_without_a_vector() -> None:
    assert candidates.ORIGINAL_MODEL.is_original_model
    assert candidates.ORIGINAL_MODEL.to_json() == {"vector": None, "layer": None, "strength": 0.0}
    steered = candidates.Candidate("with_ablation", 12, 3.0)
    assert not steered.is_original_model
    assert steered.to_json() == {"vector": "with_ablation", "layer": 12, "strength": 3.0}


def test_a_composed_sampler_is_named_by_its_candidate_and_setting() -> None:
    candidate = candidates.Candidate("with_ablation", 12, 1.5)
    keys = [
        candidates.sampler_key(candidate, setting, 50) for setting in composition.settings_grid(50)
    ]
    assert len(set(keys)) == 16
    assert keys[0] == "with_ablation_layer12_strength1.5_a0_t4"
    assert keys[-1] == "with_ablation_layer12_strength1.5_a0.75_t50"
    # The plain setting is the candidate itself.
    assert keys[3] == candidates.sampler_key(candidate, None, 50) == candidate.key
