"""Prompt selection, Instruction and Steering, with and without composition, on Qwen3.5-0.8B and
on two larger models (Figure 7).

    python figures/larger_models.py [--runs TABLE ...] [--out larger_models.pdf]

One column per model, the capped forward KL above the reverse KL, each against the construction
inferences. A point averages over the tasks of its model. The model of a run is the first part of
its task id (`qwen3.8-27b/wc_0670/chinese_character`); a benchmark task (`wc_0670/em_dash`) is one
of Qwen3.5-0.8B.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import matplotlib.figure as mfigure
import paper_runs
import paper_style
import typer

RUNS = (
    paper_runs.DATA_DIR / "main_comparison_runs.csv.gz",
    paper_runs.DATA_DIR / "larger_models_runs.csv",
)
COLUMNS = ("construction_inferences", "capped_forward_kl", "reverse_kl")
MODELS = {
    "qwen3.5-0.8b": "Qwen3.5-0.8B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "gemma-4-31b-it": "Gemma-4-31B-it",
}
METHODS = (
    "prompt-ebjc",
    "prompt-hitrate",
    "prompt-iwae",
    "prompt-ebjc+composition",
    "instruction",
    "instruction+composition",
    "steering",
    "steering+composition",
)
LARGEST_POOL = 1024
LEGEND_ROWS = (METHODS[:4], METHODS[4:])
ROWS = (("capped_forward_kl", "Forward KL$_{50}$ (nats)"), ("reverse_kl", "Reverse KL (nats)"))
COST_LIMITS = (120.0, 2e5)
COST_TICKS = {200.0: "200", 1e3: "1k", 1e4: "10k", 1e5: "100k"}
FORWARD_LIMITS = (0.0, 60.0)


def model_of(task_id: str) -> str:
    """The model of a task, as the figure names it."""
    first, _, rest = task_id.partition("/")
    if "/" not in rest:
        return MODELS["qwen3.5-0.8b"]
    return MODELS.get(first, first)


def points(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """For each model, method and setting, the averages of `COLUMNS` over the model's tasks;
    only the figure's methods, up to a pool size of 1,024."""
    paper_runs.require_columns(runs, paper_runs.RUN_KEYS)
    kept = [
        {**row, "model": model_of(row["task_id"])}
        for row in runs
        if row["method"] in METHODS and paper_runs.setting_value(row["setting"]) <= LARGEST_POOL
    ]
    return paper_runs.points(paper_runs.require_runs(kept, ", ".join(METHODS)), COLUMNS, "model")


def draw(plotted: Sequence[dict[str, Any]]) -> mfigure.Figure:
    """Two rows of panels, one column per model."""
    present = list(dict.fromkeys(point["model"] for point in plotted))
    models = [m for m in MODELS.values() if m in present] + [
        m for m in present if m not in MODELS.values()
    ]
    methods = [m for m in METHODS if any(point["method"] == m for point in plotted)]
    with paper_style.style_context():
        figure = paper_style.new_figure(3.07 * len(models), 4.1)
        grid = figure.subplots(2, len(models), squeeze=False)
        for column_index, model in enumerate(models):
            mine = [point for point in plotted if point["model"] == model]
            for row_index, (column, label) in enumerate(ROWS):
                axis = grid[row_index][column_index]
                paper_style.log_cost_axis(axis, COST_LIMITS, COST_TICKS)
                axis.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
                axis.set_axisbelow(True)
                top = FORWARD_LIMITS[1] if row_index == 0 else paper_style.band_top(mine, column)
                values = [point[column] for point in mine]
                axis.set_ylim(*paper_style.widened((0.0, top), values))
                if row_index == 0:
                    axis.set_title(f"{model} ({mine[0]['tasks']} tasks)", pad=8)
                    axis.tick_params(axis="x", labelbottom=False)
                if column_index == 0:
                    axis.set_ylabel(label)
        grid[1][len(models) // 2].set_xlabel("Construction cost (number of inferences)")
        figure.tight_layout(w_pad=1.2, h_pad=0.6)
        for column_index, model in enumerate(models):
            mine = [point for point in plotted if point["model"] == model]
            for row_index, (column, _) in enumerate(ROWS):
                axis = grid[row_index][column_index]
                slot = paper_style.add_zero_slot(figure, axis)
                if row_index == 0:
                    slot.tick_params(axis="x", labelbottom=False)
                paper_style.draw_against_cost(axis, slot, mine, column)
        rows = paper_style.legend_rows(methods, LEGEND_ROWS)
        paper_style.stack_legend_rows(figure, rows, list(grid[0]))
    return figure


def main(
    runs: Annotated[
        list[pathlib.Path] | None, typer.Option(help="A table of runs; several can be given.")
    ] = None,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("larger_models.pdf"),
) -> None:
    """Draw the figure from tables of runs."""
    rows = [row for path in (runs or RUNS) for row in paper_runs.read_runs(path)]
    paper_style.save(draw(points(rows)), out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
