"""Compute per counted inference of the training methods (Table 4).

    python figures/table_compute_per_inference.py [--analysis FILE]

Prints, for each method, the FLOPs and the GPU time per counted inference at the point of the
main comparison where the value is largest. `--analysis` takes the JSON of `posteriorbench-methods
compute-per-inference`. Prompt selection, Mixture: EB-16 and Replay only generate and score, so
both of their values are 1.00; the paper's table has them as one row.
"""

from __future__ import annotations

import pathlib
from typing import Annotated, Any

import paper_runs
import paper_style
import typer

CELLS = paper_runs.DATA_DIR / "compute_per_inference.csv"
COLUMNS = ("flops_per_inference", "gpu_time_per_inference")


def read_cells(path: pathlib.Path) -> list[dict[str, Any]]:
    """The rows of the table of cells: `method`, and each of `COLUMNS` with the setting of the
    point where it is highest (`<column>_at`)."""
    cells: list[dict[str, Any]] = []
    for row in paper_runs.read_table(path):
        cell: dict[str, Any] = {"method": row["method"]}
        for column in COLUMNS:
            cell[column] = paper_runs.number(row, column)
            cell[f"{column}_at"] = row[f"{column}_at"]
        cells.append(cell)
    return cells


def read_analysis(path: pathlib.Path) -> list[dict[str, Any]]:
    """The `methods` of the JSON of `compute-per-inference`, which have the keys of the cells."""
    analysis = paper_runs.read_json(path)
    if not isinstance(analysis, dict) or "methods" not in analysis:
        raise paper_runs.TableError(f"{path} is not the JSON of `compute-per-inference`")
    methods: list[dict[str, Any]] = list(analysis["methods"])
    paper_runs.require_columns(methods, ("method", "flops_per_inference", "flops_per_inference_at"))
    return [{**row, "method": paper_runs.table_method(str(row["method"]))} for row in methods]


def main(
    analysis: Annotated[
        pathlib.Path | None, typer.Option(help="The JSON of `compute-per-inference`.")
    ] = None,
) -> None:
    """Print the table."""
    cells = read_cells(CELLS) if analysis is None else read_analysis(analysis)
    typer.echo(f"{'method':<12}{'FLOPs / count':>15}{'at':>12}{'GPU time / count':>19}{'at':>12}")
    for cell in cells:
        flops, time = cell["flops_per_inference"], cell.get("gpu_time_per_inference")
        typer.echo(
            f"{paper_style.series(cell['method']).label:<12}{flops:>15.2f}"
            f"{cell['flops_per_inference_at']:>12}"
            + ("" if time is None else f"{time:>19.2f}{cell['gpu_time_per_inference_at']:>12}")
        )


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
