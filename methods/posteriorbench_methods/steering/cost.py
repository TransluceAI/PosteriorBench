"""What Steering pays for.

    event                                               inferences
    the refusal direction of the model                  its forward passes (16,620)
    building the two vectors of the behavior            1 per construction response (3,000),
                                                        1 per response that the filter kept
    the m selection draws of a candidate (h hits)       m + h
    a threshold test of n draws                         n

The numbers in parentheses are those of the paper's vectors. The vectors and the direction serve
every task of the behavior, and the selection draws every replicate, but each task and replicate
is charged for them in full. A kept construction response costs a second inference, the forward
pass that reads its activations. Every charge counts the same in the inference and budget counts.
"""

from __future__ import annotations

from posteriorbench_methods.common import charges
from posteriorbench_methods.steering import vectors


def vector_construction(cost: vectors.ConstructionCost) -> list[charges.ChargeSpec]:
    """The charges for the behavior's vectors and the model's refusal direction."""
    kept = sum(cost.kept_responses.values())
    return [
        charges.counted(
            charges.Category.SHARED,
            cost.refusal_direction_passes,
            f"refusal direction: {cost.refusal_direction_passes} forward passes",
        ),
        charges.counted(
            charges.Category.SHARED,
            cost.construction_responses,
            f"vectors: {cost.construction_responses} construction responses",
        ),
        charges.counted(
            charges.Category.SHARED, kept, f"vectors: activations of {kept} kept responses"
        ),
    ]
