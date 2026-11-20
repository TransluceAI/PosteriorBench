"""Training methods at about 5 x 10^5 inferences with the 10% threshold and with a 1% threshold
(Table 8).

    python figures/table_threshold_1pct.py [--runs TABLE ...]

Prints, for each method and threshold: the runs scored as the fallback, the capped forward KL and
the reverse KL averaged over the 21 tasks, and the median native hit rate over the runs. With
`--runs` the cells are computed from tables of runs whose column `rule` is `threshold-1pct` for a
run of `reselect` and empty for one of `run`.
"""

from __future__ import annotations

import pathlib
import statistics
from collections.abc import Sequence
from typing import Annotated, Any

import paper_runs
import paper_style
import typer

CELLS = paper_runs.DATA_DIR / "threshold_1pct_cells.csv"
# Each method at the budget whose runs spend about 5 x 10^5 inferences.
SETTINGS = {
    "sft": "B=524288",
    "gfn": "B=1048576",
    "ctl": "B=1048576",
    "gdc": "B=524288",
    "kl-dpg": "B=1048576",
    "js-dpg": "B=1048576",
}
RULES = {0.1: "", 0.01: "threshold-1pct"}  # the value of the column `rule` at each threshold
SCORES = ("capped_forward_kl", "reverse_kl")


def cells(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """One cell per method of `SETTINGS` and threshold that `runs` hold: `runs`,
    `fallback_runs`, the two scores averaged over tasks, and `median_native_hit_rate`."""
    paper_runs.require_columns(runs, ("method", "setting", "native_hit_rate"))
    result: list[dict[str, Any]] = []
    for method, setting in SETTINGS.items():
        for threshold, rule in RULES.items():
            mine = [
                row
                for row in runs
                if (row["method"], row["setting"], row.get("rule", "")) == (method, setting, rule)
            ]
            if not mine:
                continue
            summary = paper_runs.summarize(mine, SCORES)
            rates = [paper_runs.number(row, "native_hit_rate") for row in mine]
            result.append(
                {
                    "method": method,
                    "setting": setting,
                    "threshold": threshold,
                    "runs": summary["runs"],
                    "fallback_runs": summary["fallback_runs"],
                    **{score: summary[score] for score in SCORES},
                    "median_native_hit_rate": statistics.median(rates),
                }
            )
    if not result:
        wanted = ", ".join(f"{method} at {setting}" for method, setting in SETTINGS.items())
        raise paper_runs.TableError(f"the table has no run of {wanted}")
    return result


def read_cells(path: pathlib.Path) -> list[dict[str, Any]]:
    """The table of cells, with the keys of `cells`."""
    result: list[dict[str, Any]] = []
    for row in paper_runs.read_table(path):
        cell: dict[str, Any] = {"method": row["method"], "setting": row["setting"]}
        cell["threshold"] = float(row["threshold"])
        cell["runs"], cell["fallback_runs"] = int(row["runs"]), int(row["fallback_runs"])
        for column in (*SCORES, "median_native_hit_rate"):
            cell[column] = paper_runs.number(row, column)
        result.append(cell)
    return result


def main(
    runs: Annotated[
        list[pathlib.Path] | None, typer.Option(help="A table of runs with a column `rule`.")
    ] = None,
) -> None:
    """Print the table."""
    if runs:
        table = cells([row for path in runs for row in paper_runs.read_table(path)])
    else:
        table = read_cells(CELLS)
    typer.echo(
        f"{'method':<10}{'threshold':>10}{'fallback':>10}{'forward KL':>12}{'reverse KL':>12}{'hit rate':>10}"
    )
    for cell in table:
        typer.echo(
            f"{paper_style.series(cell['method']).label:<10}{cell['threshold']:>10.0%}"
            f"{cell['fallback_runs']:>6} / {cell['runs']}{cell['capped_forward_kl']:>12.1f}"
            f"{cell['reverse_kl']:>12.1f}{cell['median_native_hit_rate']:>10.1%}"
        )


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
