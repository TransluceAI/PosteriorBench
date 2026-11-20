"""Which hits of a scored pair are its references.

A scored pair's 250 references are chosen among its hits by position, never by content. Take the
pair's hits in the order in which they were sampled, and of these the first 5,000 (all of them
if the pair has fewer). A random half of them is set aside (`numpy.random.default_rng(0)`), and
the references are the first 250 hits of that half. They are a uniform sample of the pair's hits.

The dataset's references are fixed data. Its corpus is in random order within each prompt, not
in the order sampled, so this choice cannot be repeated from the released files; the function
below states the rule and applies to a corpus that is in the order sampled, such as one written
by `sample_corpus.py`.

    python tools/choose_references.py --hits 1837

prints the positions of the references among a pair's 1,837 hits (0 is the first hit sampled).
"""

from __future__ import annotations

import json
from typing import Annotated

import numpy as np
import typer

REFERENCES = 250
FIRST_HITS = 5_000
SEED = 0


def choose_references(
    hit_count: int,
    *,
    references: int = REFERENCES,
    first_hits: int = FIRST_HITS,
    seed: int = SEED,
) -> list[int]:
    """The positions of a pair's references among its `hit_count` hits in the order sampled
    (0 is the first hit), ascending: of the first `first_hits` hits, a random half is set aside
    with `numpy.random.default_rng(seed)`, and its first `references` hits are the references.
    Raises ValueError unless the hits considered number at least twice the references (500 hits
    for 250 references, the rule for a scored pair)."""
    considered = min(hit_count, first_hits)
    if considered < 2 * references:
        raise ValueError(
            f"{considered} hits to choose from; {references} references need at least "
            f"{2 * references}"
        )
    half = np.random.default_rng(seed).permutation(considered)[: round(considered / 2)]
    return sorted(int(position) for position in half)[:references]


def main(
    hits: Annotated[int, typer.Option(help="The number of hits of the pair")],
) -> None:
    """Print the positions of a pair's references among its hits in the order sampled."""
    try:
        positions = choose_references(hits)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(positions))


if __name__ == "__main__":
    typer.run(main)
