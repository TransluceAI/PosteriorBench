"""Training methods at about 5 x 10^5 inferences on the 21 tasks, by event probability (Figure 11).

    python figures/training_by_probability.py [--runs TABLE] [--out training_by_probability.pdf]

Takes each training method of the main comparison at the budget whose cost is closest to
5 x 10^5 inferences (`SETTINGS`). A point averages the tasks of one bin of p(S)
(`paper_runs.probability_bin`), with one standard error.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "main_comparison_runs.csv.gz"
# Each method at the budget whose runs spend about 5 x 10^5 inferences, in the order drawn.
SETTINGS = {
    "sft": "B=524288",
    "gfn": "B=1048576",
    "ctl": "B=1048576",
    "gdc": "B=524288",
    "kl-dpg": "B=1048576",
    "js-dpg": "B=1048576",
}
LEGEND_ROWS = (("sft", "ctl", "kl-dpg", "js-dpg"), ("gfn", "gdc"))


def points(
    runs: Sequence[paper_runs.Row], probabilities: Mapping[str, float]
) -> list[dict[str, Any]]:
    """One point per method and bin of p(S), from the runs of `SETTINGS`."""
    return paper_runs.points_by_probability(paper_runs.select(runs, SETTINGS), probabilities)


def main(
    runs: Annotated[pathlib.Path, typer.Option(help="A table of runs (paper_runs.py).")] = RUNS,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("training_by_probability.pdf"),
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
