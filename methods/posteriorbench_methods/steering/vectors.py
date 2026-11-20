"""The steering vectors of a behavior and their construction cost: the method data of Steering.

In the `--method-data` directory (README, "Run"), `<behavior>.safetensors` holds one float32
tensor of shape [hidden size] per vector (VECTORS) and decoder layer b, named
"<vector>.layer_<b>"; a vector whose positive responses the filter dropped has none.
`construction_cost.json` holds `refusal_direction_passes` and, per behavior under `behaviors`,
`construction_responses` and `kept_responses` (KEPT_RESPONSES), which every run is charged.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import safetensors.torch
import torch

from posteriorbench_methods.common import errors, run_record

VECTORS = ("with_ablation", "without_ablation")
COST_FILE = "construction_cost.json"
KEPT_RESPONSES = ("positive_with_ablation", "positive_without_ablation", "negative")


class VectorError(errors.MethodsError, ValueError):
    """Method data that a run cannot use: a missing directory or file, a behavior without
    vectors, a tensor of another name or shape, a missing layer or count."""


@dataclasses.dataclass(frozen=True)
class ConstructionCost:
    """What building one behavior's vectors cost. `kept_responses` is by kind (KEPT_RESPONSES);
    each kept response cost one more forward pass, to read its activations."""

    refusal_direction_passes: int
    construction_responses: int
    kept_responses: Mapping[str, int]

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {
            "refusal_direction_passes": self.refusal_direction_passes,
            "construction_responses": self.construction_responses,
            "kept_responses": dict(self.kept_responses),
        }


@dataclasses.dataclass(frozen=True)
class BehaviorVectors:
    """The vectors of one behavior: `tensors[vector][layer]` has shape [hidden size]. `tensors`
    holds the vectors that exist, in the order of VECTORS."""

    behavior: str
    tensors: Mapping[str, Mapping[int, torch.Tensor]]
    cost: ConstructionCost


def _count(value: object, name: str, path: pathlib.Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VectorError(f"{path}: {name} must be a non-negative integer, got {value!r}")
    return value


def load_cost(directory: pathlib.Path, behavior: str) -> ConstructionCost:
    """The construction cost of `behavior`. Raises VectorError if the file or a count is missing."""
    path = directory / COST_FILE
    if not path.exists():
        raise VectorError(f"{path} does not exist; it holds what the steering vectors cost")
    recorded = json.loads(path.read_text())
    try:
        entry = recorded["behaviors"][behavior]
        kept = {name: entry["kept_responses"][name] for name in KEPT_RESPONSES}
        passes, responses = recorded["refusal_direction_passes"], entry["construction_responses"]
    except (KeyError, TypeError) as error:
        raise VectorError(f"{path} has no complete entry for the behavior {behavior!r}") from error
    return ConstructionCost(
        refusal_direction_passes=_count(passes, "refusal_direction_passes", path),
        construction_responses=_count(responses, "construction_responses", path),
        kept_responses={
            name: _count(value, f"kept_responses.{name}", path) for name, value in kept.items()
        },
    )


def vectors_directory(directory: pathlib.Path | None) -> pathlib.Path:
    """The run's --method-data; raises VectorError if it was not given."""
    if directory is None:
        raise VectorError(
            "Steering reads its vectors from the method data: pass their directory as "
            "--method-data"
        )
    return directory


def load(directory: pathlib.Path | None, behavior: str, layers: Sequence[int]) -> BehaviorVectors:
    """The vectors of `behavior` and their cost, with tensors on the CPU. Raises VectorError for
    method data a run cannot use, including a vector without a tensor for one of `layers`."""
    directory = vectors_directory(directory)
    path = directory / f"{behavior}.safetensors"
    if not path.exists():
        raise VectorError(f"{directory} has no steering vectors for the behavior {behavior!r}")
    by_vector: dict[str, dict[int, torch.Tensor]] = {name: {} for name in VECTORS}
    for name, tensor in safetensors.torch.load_file(path).items():
        vector, separator, layer = name.partition(".layer_")
        if vector not in VECTORS or not separator or not layer.isdecimal():
            raise VectorError(f'{path}: tensors are named "<vector>.layer_<b>", got {name!r}')
        if tensor.ndim != 1 or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise VectorError(f"{path}: {name} is not a finite float vector")
        by_vector[vector][int(layer)] = tensor
    tensors = {name: by_layer for name, by_layer in by_vector.items() if by_layer}
    sizes = {int(t.shape[0]) for by_layer in tensors.values() for t in by_layer.values()}
    if len(sizes) > 1:
        raise VectorError(f"{path}: the vectors differ in length")
    for name, by_layer in tensors.items():
        missing = sorted(set(layers) - set(by_layer))
        if missing:
            layers_missing = ", ".join(str(layer) for layer in missing)
            raise VectorError(
                f"{path}: the vector {name} has no tensor for the layers {layers_missing}"
            )
    return BehaviorVectors(behavior, tensors, load_cost(directory, behavior))


def save(
    directory: pathlib.Path,
    behavior: str,
    tensors: Mapping[str, Mapping[int, torch.Tensor]],
    cost: ConstructionCost,
) -> None:
    """Write the vectors (as float32) and cost of `behavior` into `directory`, creating it; a
    vector that was not built is absent from `tensors`. An earlier build of the behavior is
    replaced, other behaviors are kept, and the directory's `refusal_direction_passes`, charged to
    every behavior, becomes `cost`'s. Raises VectorError for an unknown vector name or a tensor
    that is not a finite float vector."""
    named: dict[str, torch.Tensor] = {}
    for vector, by_layer in tensors.items():
        if vector not in VECTORS:
            raise VectorError(f"the vectors are {VECTORS}, got {vector!r}")
        for layer, tensor in by_layer.items():
            if (
                tensor.ndim != 1
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise VectorError(f"{vector} at layer {layer} is not a finite float vector")
            named[f"{vector}.layer_{layer}"] = tensor.detach().to("cpu", torch.float32).contiguous()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{behavior}.safetensors"
    temporary = path.with_name(f"{path.name}.tmp")
    safetensors.torch.save_file(named, temporary)
    temporary.replace(path)
    cost_path = directory / COST_FILE
    recorded = run_record.read_json(cost_path) if cost_path.exists() else {}
    recorded = {
        "refusal_direction_passes": cost.refusal_direction_passes,
        "behaviors": {
            **recorded.get("behaviors", {}),
            behavior: {
                "construction_responses": cost.construction_responses,
                "kept_responses": {name: cost.kept_responses[name] for name in KEPT_RESPONSES},
            },
        },
    }
    run_record.write_json(cost_path, recorded)
