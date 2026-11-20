"""Posterior fidelity against construction cost on the two LLM-judged behaviors, with the 10%
hit-rate threshold (Figure 8) and with a 1% threshold (Figure 16).

    python figures/llm_judged_behaviors.py [--threshold 0.1 | 0.01] [--runs TABLE] [--out FILE]

Two panels per task, the capped forward KL and the reverse KL. A point averages the replicates of
one method at one setting on one task, with one standard error across the replicates. An arrow at
the top of a panel marks a point above its range, with its value. Only the runs of `--threshold`
are drawn, unless the table has no column `threshold`.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.lines as mlines
import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "llm_judged_runs.csv"
# SFT's setting on these tasks as its records name it: a budget B with 3/4 B = 3.4 million, the
# corpus's draws (`sft run --set training_set=corpus`).
SFT_SETTING = "B=4533334"
SCORES = ("capped_forward_kl", "reverse_kl")
TITLES = {"capped_forward_kl": paper_style.FORWARD_TITLE, "reverse_kl": paper_style.REVERSE_TITLE}
LEGEND_ROWS = (
    ("prompt-ebjc", "prompt-ebjc+composition", "prompt-iwae", "prompt-hitrate", "mixture-eb16"),
    ("instruction", "instruction+composition", "steering", "steering+composition", "sft"),
)
COST_LIMITS = (120.0, 5e6)
COST_TICKS = {1e3: "1k", 1e4: "10k", 1e5: "", 1e6: "1M"}
NEAR = 3.0  # two points above a panel are close if their costs differ by less than this factor
# The range of the y axis of a panel, by behavior, score and threshold, where the paper's figure
# does not start at 0 or stops below the largest value; other panels run from 0 to DEFAULT_TOP.
DEFAULT_TOP = {"capped_forward_kl": 52.0, "reverse_kl": 100.0}
Y_LIMITS = {
    ("direct_encouragement", "capped_forward_kl", 0.1): (0.0, 50.6),
    ("direct_encouragement", "reverse_kl", 0.1): (25.0, 112.0),
    ("direct_encouragement", "reverse_kl", 0.01): (0.0, 112.0),
}


def points(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """For each task, method and setting: the mean and standard error of each score over the
    replicates (and of `construction_inferences` if the table has it), with the numbers of runs
    and of runs scored as the fallback."""
    paper_runs.require_columns(runs, ("fallback",))
    result: list[dict[str, Any]] = []
    for (task_id, method, setting), mine in paper_runs.group(
        runs, "task_id", "method", "setting"
    ).items():
        point: dict[str, Any] = {"task_id": task_id, "method": method, "setting": setting}
        point["runs"] = len(mine)
        point["fallback_runs"] = sum(paper_runs.number(row, "fallback") == 1.0 for row in mine)
        columns = (
            (*SCORES, "construction_inferences") if "construction_inferences" in mine[0] else SCORES
        )
        for column in columns:
            values = [paper_runs.number(row, column) for row in mine]
            point[column], point[f"{column}_se"] = paper_runs.mean_and_standard_error(values)
        result.append(point)
    return result


def at_threshold(runs: Sequence[paper_runs.Row], threshold: float) -> list[paper_runs.Row]:
    """The runs of one hit-rate threshold; every run of a table without the column `threshold`."""
    return [row for row in runs if "threshold" not in row or float(row["threshold"]) == threshold]


def behavior_of(task_id: str) -> str:
    """The behavior of a task id, its last part."""
    return task_id.rpartition("/")[2]


def draw(plotted: Sequence[dict[str, Any]], threshold: float) -> mfigure.Figure:
    """Two panels per task, side by side, with the task's behavior above each pair."""
    if not plotted:
        raise paper_runs.TableError(f"the table has no run at the threshold {threshold:g}")
    if "construction_inferences" not in plotted[0]:
        raise paper_runs.TableError("the points have no construction_inferences to draw against")
    tasks = list(dict.fromkeys(point["task_id"] for point in plotted))
    methods = list(dict.fromkeys(point["method"] for point in plotted))
    with paper_style.style_context():
        figure = paper_style.new_figure(4.6 * len(tasks), 2.75)
        axes = list(figure.subplots(1, 2 * len(tasks)))
        for index, axis in enumerate(axes):
            task_id, column = tasks[index // 2], SCORES[index % 2]
            behavior = behavior_of(task_id)
            axis.set_title(TITLES[column], pad=6)
            paper_style.log_cost_axis(axis, COST_LIMITS, COST_TICKS)
            axis.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
            axis.set_axisbelow(True)
            limits = Y_LIMITS.get((behavior, column, threshold), (0.0, DEFAULT_TOP[column]))
            values = [point[column] for point in plotted if point["task_id"] == task_id]
            axis.set_ylim(*paper_style.widened(limits, values, upper=False))
            if index == 0:
                axis.set_ylabel("Nats")
        figure.tight_layout(w_pad=0.9)
        for index, axis in enumerate(axes):
            task_id, column = tasks[index // 2], SCORES[index % 2]
            mine = [point for point in plotted if point["task_id"] == task_id]
            top = axis.get_ylim()[1]
            slot = paper_style.add_zero_slot(figure, axis)
            on_scale = [point for point in mine if point[column] <= top]
            curves = {point["method"] for point in mine if _is_curve(mine, point["method"])}
            kept = [p for p in mine if p["method"] in curves or p in on_scale]
            paper_style.draw_against_cost(axis, slot, kept, column)
            above = [p for p in mine if p["method"] not in curves and p[column] > top]
            above.sort(key=lambda p: p["construction_inferences"])
            for place, point in enumerate(above):
                # the value of a point close to the one before goes one line down
                near = place > 0 and point["construction_inferences"] < NEAR * max(
                    above[place - 1]["construction_inferences"], 1.0
                )
                at_zero = point["construction_inferences"] == 0
                _mark_above(slot if at_zero else axis, point, column, lower=near)
        height = figure.get_size_inches()[1]
        label_y = axes[0].get_position().y0 - 0.30 / height
        figure.text(
            0.5,
            label_y,
            "Construction cost (number of inferences)",
            ha="center",
            va="top",
            fontsize=10.5,
        )
        _group_titles(
            figure, axes, [behavior_of(task).replace("_", " ").capitalize() for task in tasks]
        )
        rows = paper_style.legend_rows(methods, LEGEND_ROWS)
        paper_style.stack_legend_rows(figure, rows, axes, gap_in=0.42)
    return figure


def _is_curve(points_of_task: Sequence[dict[str, Any]], method: str) -> bool:
    return sum(point["method"] == method for point in points_of_task) > 1


def _mark_above(axis: maxes.Axes, point: dict[str, Any], column: str, lower: bool) -> None:
    """An arrow at the top edge of the panel at the point's cost, with the point's value to its
    left; with `lower`, the value is to its right and one line down."""
    color = paper_style.series(point["method"]).color
    low, top = axis.get_ylim()
    x = point["construction_inferences"]
    axis.annotate(
        "", xy=(x, top), xytext=(x, top - 0.07 * (top - low)),
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.0}, annotation_clip=False,
    )  # fmt: skip
    axis.annotate(
        f"{point[column]:,.0f}", xy=(x, top - (0.14 if lower else 0.035) * (top - low)),
        xytext=(6 if lower else -6, 0), textcoords="offset points",
        ha="left" if lower else "right", va="center", color=color, fontsize=9.5,
    )  # fmt: skip


def _group_titles(figure: mfigure.Figure, axes: Sequence[maxes.Axes], names: Sequence[str]) -> None:
    """The name of each task's behavior above its two panels, with a rule below it."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    title_top = max(axis.title.get_window_extent(renderer).y1 for axis in axes) / figure.bbox.height
    height = figure.get_size_inches()[1]
    for index, name in enumerate(names):
        left = axes[2 * index].get_position().x0
        right = axes[2 * index + 1].get_position().x1
        rule = title_top + 0.05 / height
        figure.add_artist(
            mlines.Line2D([left, right], [rule, rule], color="#999999", linewidth=0.8)
        )
        figure.text(
            (left + right) / 2, rule + 0.05 / height, name, ha="center", va="bottom", fontsize=12
        )


def main(
    runs: Annotated[pathlib.Path, typer.Option(help="A table of runs (paper_runs.py).")] = RUNS,
    threshold: Annotated[
        float, typer.Option(help="The hit-rate threshold of the runs to draw: 0.1 or 0.01.")
    ] = 0.1,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("llm_judged_behaviors.pdf"),
) -> None:
    """Draw the figure from a table of runs."""
    plotted = points(at_threshold(paper_runs.read_runs(runs), threshold))
    paper_style.save(draw(plotted, threshold), out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
