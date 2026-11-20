"""Estimating p(S) with a sampler on the 21 tasks (Figure 15).

    python figures/estimating_ps.py [--runs TABLE --samplers TABLE | --analysis FILE] [--out FILE]

Left: the absolute log error of each point against its construction inferences. Right: the signed
log error of each passing sampler against its reverse KL, with the bound -KL as a line; both axes
are linear between -1 and 1 and logarithmic beyond. `--analysis` takes the JSON of
`posteriorbench-methods estimate-ps <out> --runs --samplers`. As in `main_comparison.py`, the left
y axis widens to hold every error, and a point that the log axes cannot show is named on stderr.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import matplotlib.figure as mfigure
import matplotlib.ticker as mticker
import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "estimating_ps_runs.csv.gz"
SAMPLERS = paper_runs.DATA_DIR / "estimating_ps_samplers.csv.gz"
COLUMNS = ("construction_inferences", "abs_log_error")
COST_LIMITS = (120.0, 6.2e6)
COST_TICKS = {1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M", 4e6: "4M"}
ERROR_LIMITS = (0.5, 200.0)
ERROR_TICKS = (0.5, 1, 2, 5, 10, 20, 50, 100)
LEGEND_ROWS = (
    ("prompt-ebjc", "prompt-hitrate", "prompt-iwae", "prompt-ce"),
    ("mixture-eb16", "instruction", "instruction+composition", "steering", "steering+composition"),
    ("sft", paper_style.COMPOSITION),
    ("gfn", "ctl", "gdc", "kl-dpg", "js-dpg"),
)
RIGHT_LIMIT = 260.0  # the right panel shows reverse KLs up to this, and log errors down to minus it


def points(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """For each method and setting: the averages of `COLUMNS`, and how many of its runs are
    scored by their own estimate and by the fallback's (`own_estimate`, `fallback_estimate`)."""
    paper_runs.require_columns(paper_runs.require_runs(list(runs), "any method"), ("estimate",))
    result = paper_runs.points(runs, COLUMNS)
    estimates = paper_runs.group(runs, "method", "setting")
    for point in result:
        mine = estimates[(point["method"], point["setting"])]
        point["own_estimate"] = sum(row["estimate"] == "own" for row in mine)
        point["fallback_estimate"] = sum(row["estimate"] == "fallback" for row in mine)
    return result


def read_analysis(path: pathlib.Path) -> tuple[list[paper_runs.Row], list[paper_runs.Row]]:
    """The runs and the samplers of the JSON of `estimate-ps --runs --samplers`, as rows of the
    two tables. A run whose estimate is unavailable is left out."""
    analysis = paper_runs.read_json(path)
    if not isinstance(analysis, dict) or "runs" not in analysis or "samplers" not in analysis:
        raise paper_runs.TableError(f"{path}: run estimate-ps with --runs and --samplers")
    keep = (*paper_runs.RUN_KEYS, "construction_inferences", "estimate", "abs_log_error")
    sampler_keys = ("method", "task_id", "reverse_kl", "log_error")
    paper_runs.require_columns(analysis["runs"], keep)
    paper_runs.require_columns(analysis["samplers"], sampler_keys)
    runs = [
        {name: str(run[name]) for name in keep}
        for run in analysis["runs"]
        if run["abs_log_error"] is not None
    ]
    samplers = [{name: str(row[name]) for name in sampler_keys} for row in analysis["samplers"]]
    for row in (*runs, *samplers):
        row["method"] = paper_runs.table_method(row["method"])
    return runs, samplers


def draw(plotted: Sequence[dict[str, Any]], samplers: Sequence[paper_runs.Row]) -> mfigure.Figure:
    """The error of each point against its cost, and the error of each passing sampler against
    its reverse KL. A point whose error is zero is left out of the left panel."""
    methods = list(dict.fromkeys(point["method"] for point in plotted))
    with paper_style.style_context():
        figure = paper_style.new_figure(9.5, 3.5)
        left, right = figure.subplots(1, 2)
        left.set_title("Absolute log error of the estimate of $p(S)$", pad=8)
        paper_style.log_cost_axis(left, COST_LIMITS, COST_TICKS, minor=True)
        left.set_xlabel("Construction cost (number of inferences)")
        left.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
        left.set_axisbelow(True)
        left.set_yscale("log")
        limits, ticks = paper_style.widened_log_axis(
            ERROR_LIMITS, ERROR_TICKS, [point["abs_log_error"] for point in plotted]
        )
        left.set_ylim(*limits)
        left.yaxis.set_major_locator(mticker.FixedLocator(list(ticks)))
        left.yaxis.set_major_formatter(mticker.FixedFormatter([f"{tick:g}" for tick in ticks]))
        left.yaxis.set_minor_locator(mticker.NullLocator())
        left.set_ylabel("Nats")

        right.set_title("Log error of each sampler", pad=8)
        for method, rows in paper_runs.group(samplers, "method").items():
            style = paper_style.series(method[0])
            right.plot(
                [paper_runs.number(row, "reverse_kl") for row in rows],
                [paper_runs.number(row, "log_error") for row in rows],
                linestyle="none", marker=style.marker, markersize=0.6 * style.size,
                color=style.color, alpha=0.6, markeredgewidth=0,
            )  # fmt: skip
        right.plot([0, RIGHT_LIMIT], [0, -RIGHT_LIMIT], color="#444444", linewidth=1.2, zorder=5)
        right.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":", zorder=0)
        right.annotate(
            r"$-\mathrm{KL}(q_S\|p_S)$", xy=(3.2, -3.2), xytext=(0.55, -30.0), color="#555555",
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
        )  # fmt: skip
        right.set_xscale("symlog", linthresh=1.0)
        right.set_yscale("symlog", linthresh=1.0)
        right.set_xlim(-0.06, RIGHT_LIMIT)
        right.set_ylim(-RIGHT_LIMIT, 80.0)
        right.set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        right.set_yticks([1, 0, -1, -10, -100], ["1", "0", "−1", "−10", "−100"])
        right.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
        right.set_axisbelow(True)
        right.set_xlabel(f"{paper_style.REVERSE_TITLE} (nats)")
        right.set_ylabel("Nats")
        figure.tight_layout(w_pad=1.8)
        slot = paper_style.add_zero_slot(figure, left)
        positive = [point for point in plotted if point["abs_log_error"] > 0]
        paper_style.draw_against_cost(left, slot, positive, "abs_log_error")
        rows = paper_style.legend_rows(methods, LEGEND_ROWS)
        paper_style.stack_legend_rows(figure, rows, [left, right])
    return figure


def main(
    runs: Annotated[pathlib.Path, typer.Option(help="The table of runs (left panel).")] = RUNS,
    samplers: Annotated[
        pathlib.Path, typer.Option(help="The table of passing samplers (right panel).")
    ] = SAMPLERS,
    analysis: Annotated[
        pathlib.Path | None,
        typer.Option(help="The JSON of `estimate-ps --runs --samplers`, in place of both tables."),
    ] = None,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("estimating_ps.pdf"),
) -> None:
    """Draw the figure from the two tables or from the analysis of a reader's own runs."""
    if analysis is None:
        run_rows, sampler_rows = paper_runs.read_table(runs), paper_runs.read_table(samplers)
    else:
        run_rows, sampler_rows = read_analysis(analysis)
    paper_runs.require_columns(sampler_rows, ("method", "reverse_kl", "log_error"))
    plotted = points(run_rows)
    paper_style.save(draw(plotted, sampler_rows), out)
    for line in paper_style.not_drawn(plotted, ("abs_log_error",), COST_LIMITS):
        typer.echo(f"not drawn: {line}", err=True)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
