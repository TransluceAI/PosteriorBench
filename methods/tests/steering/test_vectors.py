"""Reading the method data: the vectors of a behavior and their construction cost."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import pytest
import safetensors.torch
import torch

from posteriorbench_methods.steering import vectors

BEHAVIOR = "goblin_mention"


def test_the_vectors_of_a_behavior_and_their_cost(vectors_directory: pathlib.Path) -> None:
    loaded = vectors.load(vectors_directory, BEHAVIOR, layers=(0, 1))
    assert loaded.behavior == BEHAVIOR
    assert list(loaded.tensors) == ["with_ablation", "without_ablation"]
    for by_layer in loaded.tensors.values():
        assert sorted(by_layer) == [0, 1]
        assert all(t.shape == (32,) and t.dtype == torch.float32 for t in by_layer.values())
    assert loaded.cost.refusal_direction_passes == 16_620
    assert loaded.cost.construction_responses == 3_000
    assert sum(loaded.cost.kept_responses.values()) == 190
    assert loaded.cost.to_json() == {
        "refusal_direction_passes": 16_620,
        "construction_responses": 3_000,
        "kept_responses": {
            "positive_with_ablation": 13,
            "positive_without_ablation": 2,
            "negative": 175,
        },
    }


def test_a_vector_that_was_not_built_is_absent(
    write_vectors: Callable[..., pathlib.Path], tmp_path: pathlib.Path
) -> None:
    directory = write_vectors(tmp_path, vectors=["without_ablation"])
    assert list(vectors.load(directory, BEHAVIOR, layers=(0, 1)).tensors) == ["without_ablation"]
    neither = write_vectors(tmp_path / "neither", vectors=[])
    assert vectors.load(neither, BEHAVIOR, layers=(0, 1)).tensors == {}


def test_method_data_that_cannot_be_used_is_refused(
    write_vectors: Callable[..., pathlib.Path], tmp_path: pathlib.Path
) -> None:
    with pytest.raises(vectors.VectorError, match="--method-data"):
        vectors.load(None, BEHAVIOR, layers=(0,))
    directory = write_vectors(tmp_path / "vectors")
    with pytest.raises(vectors.VectorError, match="no steering vectors for the behavior"):
        vectors.load(directory, "em_dash", layers=(0,))
    with pytest.raises(vectors.VectorError, match=r"no tensor for the layers 7$"):
        vectors.load(directory, BEHAVIOR, layers=(0, 7))
    (directory / "construction_cost.json").write_text(
        json.dumps({"refusal_direction_passes": 16620, "behaviors": {}})
    )
    with pytest.raises(vectors.VectorError, match="no complete entry"):
        vectors.load(directory, BEHAVIOR, layers=(0,))
    (directory / "construction_cost.json").unlink()
    with pytest.raises(vectors.VectorError, match="does not exist"):
        vectors.load(directory, BEHAVIOR, layers=(0,))


@pytest.mark.parametrize(
    "tensors, message",
    [
        ({"with_ablation.7": torch.zeros(4)}, "tensors are named"),
        ({"another.layer_7": torch.zeros(4)}, "tensors are named"),
        ({"with_ablation.layer_7": torch.zeros(2, 4)}, "not a finite float vector"),
        ({"with_ablation.layer_7": torch.full((4,), float("nan"))}, "not a finite float vector"),
        (
            {"with_ablation.layer_7": torch.zeros(4), "without_ablation.layer_7": torch.zeros(5)},
            "differ in length",
        ),
    ],
)
def test_tensors_of_another_name_or_shape_are_refused(
    tensors: dict[str, torch.Tensor], message: str, tmp_path: pathlib.Path
) -> None:
    safetensors.torch.save_file(tensors, tmp_path / f"{BEHAVIOR}.safetensors")
    with pytest.raises(vectors.VectorError, match=message):
        vectors.load(tmp_path, BEHAVIOR, layers=(7,))
