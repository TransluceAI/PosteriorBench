"""Write a table of runs (`paper_runs.py`) from the records of finished runs, or from results of
the package.

    python figures/records_to_table.py runs --out my_runs.csv
    python figures/records_to_table.py my_method.jsonl --out my_method.csv

A `.jsonl` file holds one result per line, as `posteriorbench aggregate` reads them. The column
`rule` names the rule of a record under `<out>/reselected/<rule>/` and is empty for one of `run`.
A result that names no method is listed under its sampler's name, one without a setting as `none`.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Sequence
from typing import Annotated, Any

import paper_runs
import typer

from posteriorbench_methods.common import run_record

RESULT_LINES = ".jsonl"  # the ending of a file with one result per line
NO_SETTING = "none"  # the setting of a record without one
COLUMNS: tuple[str, ...] = (
    "rule",
    *paper_runs.RUN_KEYS,
    "construction_inferences",
    "capped_forward_kl",
    "reverse_kl",
    "native_hit_rate",
    "fallback",
)


def result_files(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    """The files of results that `paths` name: a file itself, or every `result.json` below a
    directory, in sorted order."""
    found: list[pathlib.Path] = []
    for path in paths:
        if not path.exists():
            raise paper_runs.TableError(f"{path} does not exist")
        mine = [path] if path.is_file() else sorted(path.rglob(run_record.RESULT_FILE))
        if not mine:
            raise paper_runs.TableError(f"{path}: no {run_record.RESULT_FILE}")
        found.extend(mine)
    return found


def results_of(path: pathlib.Path) -> list[Any]:
    """The results that a file holds: one per line of a `.jsonl` file, else the one result."""
    if path.suffix != RESULT_LINES:
        return [paper_runs.read_json(path)]
    try:
        return run_record.read_jsonl(path)
    except json.JSONDecodeError as error:
        raise paper_runs.TableError(f"{path} does not hold one JSON result per line") from error


def rule_of(path: pathlib.Path) -> str:
    """The rule of a record under `reselected/<rule>/`; "" for a record of `run`."""
    parts = path.parts
    if run_record.RESELECTED_DIR in parts[:-1]:
        return parts[parts.index(run_record.RESELECTED_DIR) + 1]
    return ""


def rows(files: Sequence[pathlib.Path]) -> list[dict[str, Any]]:
    """One row of the table per result of `files`."""
    table: list[dict[str, Any]] = []
    for path in files:
        for result in results_of(path):
            row: dict[str, Any] = dict(result) if isinstance(result, dict) else {}
            if row.get("method") is None:
                row["method"] = row.get("sampler")
            if row.get("setting") is None:
                row["setting"] = NO_SETTING
            missing = [column for column in COLUMNS[1:] if row.get(column) is None]
            if missing:
                raise paper_runs.TableError(
                    f"{path}: a result has no {', '.join(missing)}; is it the result of a run "
                    "or of an evaluation?"
                )
            row["method"] = paper_runs.table_method(str(row["method"]))
            table.append({"rule": rule_of(path), **{column: row[column] for column in COLUMNS[1:]}})
    return table


def main(
    records: Annotated[
        list[pathlib.Path],
        typer.Argument(
            help="Output roots, record directories, result.json files, or .jsonl files with "
            "one result per line."
        ),
    ],
    out: Annotated[pathlib.Path, typer.Option(help="The table (.csv or .csv.gz).")],
) -> None:
    """Write the table of the results."""
    table = rows(result_files(records))
    paper_runs.write_table(out, COLUMNS, table)
    typer.echo(f"wrote {out}: {len(table)} runs")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
