"""Posterior fidelity against construction cost on the 21 tasks (Figure 5).

    python figures/main_comparison.py [--runs TABLE] [--add TABLE ...] [--out main_comparison.pdf]

Capped forward KL and reverse KL of each point against its construction inferences, with one
standard error across tasks. Without `--runs` it draws the paper's runs. `--add` draws another
table next to them; a method that both have is drawn as `<method> (added)`. A score outside the
paper's y axis widens it; a point that the log axes cannot show is named on stderr.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Sequence
from typing import Annotated, Any

import matplotlib.figure as mfigure
import matplotlib.ticker as mticker
import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "main_comparison_runs.csv.gz"
COLUMNS = ("construction_inferences", "capped_forward_kl", "reverse_kl")
PANELS = (
    ("capped_forward_kl", f"{paper_style.FORWARD_TITLE} {paper_style.COVERAGE}"),
    ("reverse_kl", f"{paper_style.REVERSE_TITLE} {paper_style.REALISM}"),
)
# The limits and ticks of the two logarithmic y axes, and of the cost axis.
Y_AXES = {
    "capped_forward_kl": ((2.0, 60.0), (2, 5, 10, 20, 50)),
    "reverse_kl": ((2.0, 150.0), (2, 5, 10, 20, 50, 100)),
}
COST_LIMITS = (120.0, 6.2e6)
COST_TICKS = {1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M", 4e6: "4M"}
CAP = 50.0
# The legend: one row per entry. Composed methods that are not named share the dashed entry.
LEGEND_ROWS = (
    ("prompt-ebjc", "prompt-hitrate", "prompt-iwae", "prompt-ce"),
    ("mixture-eb16", "instruction", "instruction+composition", "steering", "steering+composition"),
    ("sft", "replay", paper_style.COMPOSITION),
    ("gfn", "ctl", "gdc", "kl-dpg", "js-dpg"),
)


def points(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """The points of the figure: for each method and setting, the averages of `COLUMNS`."""
    return paper_runs.points(paper_runs.require_runs(list(runs), "any method"), COLUMNS)


def with_added(
    runs: Sequence[paper_runs.Row], added: Iterable[paper_runs.Row]
) -> list[paper_runs.Row]:
    """`runs` and the rows of `added`. A method of `added` that `runs` also have is renamed
    `<method> (added)` (`<method> (added)+composition` if composed), so that it is a series of
    its own."""
    present = {row["method"] for row in runs}
    result = list(runs)
    for row in added:
        paper_runs.require_columns([row], ("method",))
        method = row["method"]
        while method in present:
            plain = method.removesuffix(paper_style.COMPOSITION)
            method = f"{plain} (added){method[len(plain) :]}"
        result.append({**row, "method": method})
    return result


def y_axis(
    plotted: Sequence[dict[str, Any]], column: str
) -> tuple[tuple[float, float], list[float]]:
    """The limits and ticks of a panel's y axis: the paper's, widened to hold every positive
    value of `plotted`."""
    limits, ticks = Y_AXES[column]
    return paper_style.widened_log_axis(limits, ticks, [point[column] for point in plotted])


def not_drawn(plotted: Sequence[dict[str, Any]]) -> list[str]:
    """One line for each point that the logarithmic axes cannot show."""
    return paper_style.not_drawn(plotted, [column for column, _ in PANELS], COST_LIMITS)


def draw(plotted: Sequence[dict[str, Any]]) -> mfigure.Figure:
    """The two panels, with a slot at the left of each for the methods whose cost is zero. A
    point is left out of a panel in which its score is not positive."""
    methods = list(dict.fromkeys(point["method"] for point in plotted))
    with paper_style.style_context():
        figure = paper_style.new_figure(9.5, 3.5)
        axes = list(figure.subplots(1, 2))
        for axis, (column, title) in zip(axes, PANELS, strict=True):
            axis.set_title(title, pad=8)
            paper_style.log_cost_axis(axis, COST_LIMITS, COST_TICKS, minor=True)
            axis.set_xlabel("Construction cost (number of inferences)")
            axis.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
            axis.set_axisbelow(True)
            limits, ticks = y_axis(plotted, column)
            axis.set_yscale("log")
            axis.set_ylim(*limits)
            axis.yaxis.set_major_locator(mticker.FixedLocator(list(ticks)))
            axis.yaxis.set_major_formatter(mticker.FixedFormatter([f"{tick:g}" for tick in ticks]))
            axis.yaxis.set_minor_locator(mticker.NullLocator())
            axis.set_ylabel("Nats")
        figure.tight_layout(w_pad=1.8)
        tops = {"capped_forward_kl": CAP, "reverse_kl": paper_style.band_top(plotted, "reverse_kl")}
        for axis, (column, _) in zip(axes, PANELS, strict=True):
            slot = paper_style.add_zero_slot(figure, axis)
            positive = [point for point in plotted if point[column] > 0]
            paper_style.draw_against_cost(axis, slot, positive, column, top=tops[column])
        paper_style.stack_legend_rows(figure, paper_style.legend_rows(methods, LEGEND_ROWS), axes)
    return figure


def main(
    runs: Annotated[
        pathlib.Path,
        typer.Option(help="A table of runs (paper_runs.py). Default: the paper's runs."),
    ] = RUNS,
    add: Annotated[
        list[pathlib.Path] | None,
        typer.Option(help="A table of runs to draw next to those of --runs; can be repeated."),
    ] = None,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("main_comparison.pdf"),
) -> None:
    """Draw the figure from a table of runs."""
    rows = paper_runs.read_runs(runs)
    for table in add or []:
        rows = with_added(rows, paper_runs.read_runs(table))
    plotted = points(rows)
    paper_style.save(draw(plotted), out)
    for line in not_drawn(plotted):
        typer.echo(f"not drawn: {line}", err=True)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
