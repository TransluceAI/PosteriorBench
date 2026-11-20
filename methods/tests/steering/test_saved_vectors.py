"""Writing the method data: `vectors.save` writes what `vectors.load` reads."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import pytest
import torch

from posteriorbench_methods.steering import vectors

KEPT = {"positive_with_ablation": 4, "positive_without_ablation": 0, "negative": 9}


def _cost(passes: int, responses: int = 30) -> vectors.ConstructionCost:
    return vectors.ConstructionCost(passes, responses, KEPT)


def test_saved_vectors_are_loaded_back(tmp_path: pathlib.Path) -> None:
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "with_ablation": {layer: torch.randn(8, generator=generator) for layer in (0, 3)},
        "without_ablation": {3: torch.randn(8, generator=generator, dtype=torch.float64)},
    }
    directory = tmp_path / "steering_vectors"
    vectors.save(directory, "em_dash", tensors, _cost(100))
    loaded = vectors.load(directory, "em_dash", layers=(3,))
    assert list(loaded.tensors) == ["with_ablation", "without_ablation"]
    assert sorted(loaded.tensors["with_ablation"]) == [0, 3]
    for vector, by_layer in tensors.items():
        for layer, tensor in by_layer.items():
            read = loaded.tensors[vector][layer]
            assert read.dtype == torch.float32 and torch.equal(read, tensor.float())
    assert loaded.cost == _cost(100)
    assert sorted(path.name for path in directory.iterdir()) == [
        "construction_cost.json",
        "em_dash.safetensors",
    ]


def test_a_behavior_is_saved_next_to_the_others(
    write_vectors: Callable[..., pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """The directory of the fixture holds goblin_mention, with 16,620 forward passes for the
    refusal direction."""
    directory = write_vectors(tmp_path)
    before = vectors.load(directory, "goblin_mention", layers=(0, 1))
    vectors.save(directory, "em_dash", {"without_ablation": {0: torch.ones(32)}}, _cost(500))
    after = vectors.load(directory, "goblin_mention", layers=(0, 1))
    assert torch.equal(after.tensors["with_ablation"][1], before.tensors["with_ablation"][1])
    assert after.cost.kept_responses == before.cost.kept_responses
    # The passes of the refusal direction are those of the directory, for every behavior.
    assert after.cost.refusal_direction_passes == 500
    recorded = json.loads((directory / "construction_cost.json").read_text())
    assert list(recorded) == ["refusal_direction_passes", "behaviors"]
    assert list(recorded["behaviors"]) == ["goblin_mention", "em_dash"]
    # A second build of a behavior replaces its vectors and its cost.
    vectors.save(directory, "em_dash", {}, _cost(500, responses=20))
    again = vectors.load(directory, "em_dash", layers=())
    assert again.tensors == {} and again.cost.construction_responses == 20


def test_vectors_that_a_run_could_not_read_are_not_saved(tmp_path: pathlib.Path) -> None:
    with pytest.raises(vectors.VectorError, match="the vectors are"):
        vectors.save(tmp_path, "em_dash", {"ablated": {0: torch.ones(4)}}, _cost(1))
    with pytest.raises(vectors.VectorError, match="not a finite float vector"):
        vectors.save(tmp_path, "em_dash", {"with_ablation": {0: torch.ones(2, 2)}}, _cost(1))
    with pytest.raises(vectors.VectorError, match="not a finite float vector"):
        nan = torch.tensor([1.0, float("nan")])
        vectors.save(tmp_path, "em_dash", {"with_ablation": {0: nan}}, _cost(1))
    assert not list(tmp_path.iterdir())
