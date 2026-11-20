"""The settings of Steering."""

from __future__ import annotations

import dataclasses
import fractions
import math

from posteriorbench_methods.common import errors, rank_table

# The depths of the decoder at which a vector can be added, as decimal fractions.
DEPTHS = ("0.35", "0.40", "0.45", "0.50", "0.55", "0.60")
BENCHMARK_MODEL_LAYERS = 24  # the decoder layers of Qwen3.5-0.8B


class LayersError(errors.MethodsError, ValueError):
    """The model has too few layers for DEPTHS, and the settings state no layers."""


def layers_at_depths(num_layers: int) -> tuple[int, ...]:
    """Layer int(n d) - 1 (counted from 0) at each depth d of DEPTHS, for n = `num_layers`:
    7, 8, 9, 11, 12, 13 of 24; 21, 24, 27, 31, 34, 37 of 64; 20, 23, 26, 29, 32, 35 of 60.
    Raises LayersError if two depths fall on one layer."""
    layers = tuple(math.floor(fractions.Fraction(depth) * num_layers) - 1 for depth in DEPTHS)
    if layers[0] < 0 or len(set(layers)) != len(layers):
        raise LayersError(
            f"a model with {num_layers} layers has no six layers at the depths "
            f"{', '.join(DEPTHS)}; state the layers with the setting `layers`"
        )
    return layers


@dataclasses.dataclass(frozen=True)
class Settings:
    """The settings of Steering and Steering + composition; the defaults are the paper's.

    `layers` count from 0; None means `layers_at_depths` of the task's original model.
    `threshold` is 0.10 or 0.01; at 0.01 the run's rule is `submission.at_one_percent` and its
    records go to `<out>/reselected/threshold-1pct/`."""

    layers: tuple[int, ...] | None = None
    strengths: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
    selection_draws: int = 128
    threshold: float = 0.10

    def __post_init__(self) -> None:
        layers = self.layers
        if layers is not None and (
            not layers or any(layer < 0 for layer in layers) or len(set(layers)) != len(layers)
        ):
            raise ValueError(f"layers are distinct and counted from 0, got {layers}")
        if any(not (math.isfinite(s) and s > 0) for s in self.strengths) or len(
            set(self.strengths)
        ) != len(self.strengths):
            raise ValueError(f"strengths are distinct and positive, got {self.strengths}")
        if self.selection_draws < 1:
            raise ValueError(f"selection_draws is at least 1, got {self.selection_draws}")
        if self.threshold not in rank_table.THRESHOLDS:
            raise ValueError(f"threshold is one of {rank_table.THRESHOLDS}, got {self.threshold}")

    def with_layers_of(self, num_layers: int) -> Settings:
        """These settings, with `layers_at_depths(num_layers)` if `layers` is None."""
        if self.layers is not None:
            return self
        return dataclasses.replace(self, layers=layers_at_depths(num_layers))
