"""The candidate samplers of a task.

A candidate adds one of the behavior's vectors after one layer with one strength. The original
model, which is the strength 0, is a candidate too. With two vectors, six layers and eight
strengths a task has 97 candidates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from posteriorbench_methods.common import composition
from posteriorbench_methods.steering import settings as steering_settings
from posteriorbench_methods.steering import vectors as steering_vectors


@dataclasses.dataclass(frozen=True)
class Candidate:
    """The original model with `strength` x the vector `vector` added after decoder layer `layer`
    (counted from 0) at every position. The original model itself has no vector and no layer."""

    vector: str | None
    layer: int | None
    strength: float

    @property
    def is_original_model(self) -> bool:
        return self.vector is None

    @property
    def key(self) -> str:
        """The candidate's name in records, seeds and file names: "original", or for example
        "with_ablation_layer12_strength1.5"."""
        if self.is_original_model:
            return "original"
        return f"{self.vector}_layer{self.layer}_strength{self.strength:g}"

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict: vector, layer, strength."""
        return dataclasses.asdict(self)


ORIGINAL_MODEL = Candidate(vector=None, layer=None, strength=0.0)


def candidates(
    settings: steering_settings.Settings,
    vectors: Sequence[str],
    *,
    num_layers: int = steering_settings.BENCHMARK_MODEL_LAYERS,
) -> list[Candidate]:
    """The candidates in their order, which breaks ranking ties: the original model, then per
    vector (with_ablation first) every strength, and per strength every layer. Without layers in
    `settings`, the layers are `layers_at_depths(num_layers)`."""
    unknown = sorted(set(vectors) - set(steering_vectors.VECTORS))
    if unknown:
        raise ValueError(f"the vectors are {steering_vectors.VECTORS}, got {unknown}")
    layers = settings.with_layers_of(num_layers).layers
    assert layers is not None
    found = [ORIGINAL_MODEL]
    for vector in steering_vectors.VECTORS:
        if vector not in vectors:
            continue
        for strength in settings.strengths:
            found += [Candidate(vector, layer, strength) for layer in layers]
    return found


def sampler_key(
    candidate: Candidate, composed: composition.Setting | None, response_limit: int
) -> str:
    """The candidate's key, followed by the setting's id ("..._strength1.5_a0.25_t16") unless
    `composed` is None or plain."""
    if composed is None or composed.is_plain(response_limit):
        return candidate.key
    return f"{candidate.key}_{composed.setting_id}"
