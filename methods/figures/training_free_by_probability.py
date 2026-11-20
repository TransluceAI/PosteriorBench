"""Training-free methods on the 108 prompt-behavior pairs, by event probability (Figure 14).

    python figures/training_free_by_probability.py [--runs TABLE] [--out FILE]

EB-JC selection and Mixture: EB-16 are taken at K = 1,024. A point averages the pairs of one bin
of p(S) (`paper_runs.probability_bin`), with one standard error. In the paper's runs Mixture:
EB-16 has one replicate per pair and the other methods three.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "training_free_108pairs_runs.csv.gz"
# The methods and their settings, in the order drawn.
SETTINGS = {
    "prompt-ebjc": "K=1024",
    "mixture-eb16": "K=1024",
    "instruction": "none",
    "instruction+composition": "none",
    "steering": "none",
}
LEGEND_ROWS = (
    ("prompt-ebjc", "instruction", "steering"),
    ("mixture-eb16", "instruction+composition"),
)


def points(
    runs: Sequence[paper_runs.Row], probabilities: Mapping[str, float]
) -> list[dict[str, Any]]:
    """One point per method and bin of p(S), from the runs of `SETTINGS`."""
    return paper_runs.points_by_probability(paper_runs.select(runs, SETTINGS), probabilities)


def main(
    runs: Annotated[pathlib.Path, typer.Option(help="A table of runs (paper_runs.py).")] = RUNS,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("training_free_by_probability.pdf"),
) -> None:
    """Draw the figure from a table of runs."""
    plotted = points(paper_runs.read_runs(runs), paper_runs.event_probabilities())
    methods = [method for method in SETTINGS if any(p["method"] == method for p in plotted)]
    rows = [[m for m in row if m in methods] for row in LEGEND_ROWS]
    figure = paper_style.draw_probability_panels(plotted, methods, [row for row in rows if row])
    paper_style.save(figure, out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
