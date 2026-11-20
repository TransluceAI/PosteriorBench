"""The settings of the prompt family."""

from __future__ import annotations

import dataclasses

from posteriorbench_methods.common import rank_table
from posteriorbench_methods.prompt import methods

# The methods of the paper's main comparison.
MAIN_COMPARISON: tuple[str, ...] = (
    "prompt-ebjc",
    "prompt-hitrate",
    "prompt-iwae",
    "prompt-ce",
    "mixture-eb16",
    "instruction",
    "prompt-ebjc+composition",
    "mixture-eb16+composition",
    "instruction+composition",
)


@dataclasses.dataclass(frozen=True)
class Settings:
    """What a run of the prompt family does; the defaults are the paper's (README, "Settings").

    `pool_sizes` are the K of prompt selection and the mixtures; Instruction has no pool and runs
    once. `selection_draws` is m. `threshold` is 0.10 or 0.01; at 0.01 the run's rule is
    `submission.at_one_percent` and its records go to `<out>/reselected/threshold-1pct/`."""

    methods: tuple[str, ...] = MAIN_COMPARISON
    pool_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    selection_draws: int = 16
    ce_bank_size: int = 512
    threshold: float = 0.10

    def __post_init__(self) -> None:
        for name in self.methods:
            methods.parse(name)
        if any(size < 1 for size in self.pool_sizes):
            raise ValueError(f"pool sizes are positive, got {self.pool_sizes}")
        if len(set(self.pool_sizes)) != len(self.pool_sizes):
            raise ValueError(f"pool sizes are distinct, got {self.pool_sizes}")
        if self.selection_draws < 1:
            raise ValueError(f"selection_draws must be positive, got {self.selection_draws}")
        if self.ce_bank_size < 1:
            raise ValueError(f"ce_bank_size must be positive, got {self.ce_bank_size}")
        if self.threshold not in rank_table.THRESHOLDS:
            raise ValueError(f"threshold is one of {rank_table.THRESHOLDS}, got {self.threshold}")
