"""The edit distance between two token sequences."""

from __future__ import annotations

from collections.abc import Sequence


def distance(first: Sequence[int], second: Sequence[int]) -> int:
    """The Levenshtein distance: the fewest insertions, deletions and substitutions of single
    tokens that turn `first` into `second`. Computed with Myers's bit-vector recurrence, one
    column of the dynamic program per token of the longer sequence."""
    if len(first) > len(second):
        first, second = second, first
    if not first:
        return len(second)
    masks: dict[int, int] = {}
    for position, token in enumerate(first):
        masks[token] = masks.get(token, 0) | (1 << position)
    positive = ~0
    negative = 0
    value = len(first)
    last = 1 << (len(first) - 1)
    for token in second:
        equal = masks.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        value += bool(positive_horizontal & last) - bool(negative_horizontal & last)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return value
