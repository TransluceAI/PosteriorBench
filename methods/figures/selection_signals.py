"""Prompt selection with four selection signals on the 108 prompt-behavior pairs (Figure 12).

    python figures/selection_signals.py [--runs TABLE] [--out selection_signals.pdf]

Each point is one signal at one pool size K, averaged over the pairs with one standard error
across pairs. The dotted line marks the 10% hit-rate threshold.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import matplotlib.figure as mfigure
import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "training_free_108pairs_runs.csv.gz"
COLUMNS = ("capped_forward_kl", "reverse_kl", "native_hit_rate")
# The signals in the order of the legend, with their labels there.
SIGNALS = {
    "prompt-ebjc": "EB-JC",
    "prompt-jc": "JC",
    "prompt-iwae": "IWAE",
    "prompt-hitrate": "Hit rate",
}
PANELS = (
    ("capped_forward_kl", f"{paper_style.FORWARD_TITLE} {paper_style.COVERAGE}", (0.0, 51.0)),
    ("reverse_kl", f"{paper_style.REVERSE_TITLE} {paper_style.REALISM}", (0.0, 61.5)),
    ("native_hit_rate", "Native hit rate", (0.0, 1.02)),
)
THRESHOLD = 0.1


def points(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """For each signal and pool size, the averages of `COLUMNS` over the pairs."""
    paper_runs.require_columns(runs, ("method",))
    signals = [row for row in runs if row["method"] in SIGNALS]
    return paper_runs.points(paper_runs.require_runs(signals, ", ".join(SIGNALS)), COLUMNS)


def draw(plotted: Sequence[dict[str, Any]]) -> mfigure.Figure:
    """The three panels against the pool size K. The ranges of the y axes are the paper's; a
    value outside its panel's range widens the range."""
    methods = [method for method in SIGNALS if any(p["method"] == method for p in plotted)]
    pool_sizes = sorted({paper_runs.setting_value(p["setting"]) for p in plotted})
    with paper_style.style_context():
        figure = paper_style.new_figure(9.07, 2.75)
        axes = list(figure.subplots(1, 3))
        for axis, (column, title, limits) in zip(axes, PANELS, strict=True):
            for method in sorted(methods, key=lambda m: -paper_style.series(m).size):
                mine = [p for p in plotted if p["method"] == method]
                paper_style.draw_series(
                    axis,
                    [paper_runs.setting_value(p["setting"]) for p in mine],
                    [p[column] for p in mine],
                    [p[f"{column}_se"] for p in mine],
                    paper_style.series(method),
                    linewidth=1.6,
                )
            axis.set_title(title, pad=8)
            paper_style.pool_size_axis(axis, pool_sizes)
            axis.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
            axis.set_axisbelow(True)
            axis.set_ylim(*paper_style.widened(limits, [p[column] for p in plotted]))
            if column == "native_hit_rate":
                paper_style.percent_axis(axis, (0, 0.2, 0.4, 0.6, 0.8, 1.0))
                axis.axhline(THRESHOLD, color="#888888", linewidth=0.8, linestyle=":", zorder=0)
            else:
                axis.set_ylabel("Nats")
        figure.tight_layout(w_pad=1.6)
        handles = [
            paper_style.legend_handle(paper_style.series(method), SIGNALS[method])
            for method in methods
        ]
        paper_style.stack_legend_rows(figure, [handles], axes)
    return figure


def main(
    runs: Annotated[pathlib.Path, typer.Option(help="A table of runs (paper_runs.py).")] = RUNS,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("selection_signals.pdf"),
) -> None:
    """Draw the figure from a table of runs."""
    paper_style.save(draw(points(paper_runs.read_runs(runs))), out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
