"""Tables of runs and their averages, shared by the figure scripts.

A table is a CSV file (gzip if its name ends in `.gz`) with one row per run, the columns `method`,
`setting`, `task_id` and `replicate`, and score columns named as the fields of a `result.json`
(`fallback` is `true` or `false`). A point is one method at one setting, averaged as `posteriorbench
aggregate` averages: the replicates of a task first, then the tasks with equal weights, with the
standard error of the task means.
"""

from __future__ import annotations

import collections
import csv
import functools
import gzip
import io
import json
import math
import pathlib
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ParamSpec

import typer

P = ParamSpec("P")
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

Row = dict[str, str]
RUN_KEYS: tuple[str, ...] = ("method", "setting", "task_id", "replicate")
# GFlowNet's records name the method "gflownet"; the tables and the figure scripts name it "gfn".
TABLE_METHODS = {"gflownet": "gfn"}
# The event probabilities p̂(S) of the paper's figures are binned by order of magnitude, with
# p̂(S) >= 1e-2 in one bin: the lower edges of the four bins.
PROBABILITY_BIN_EDGES: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)


class TableError(ValueError):
    """A table or file that a script cannot use; `one_line_errors` reports it as one line."""


def one_line_errors(main: Callable[P, None]) -> Callable[P, None]:
    """`main` with a TableError reported as one line on stderr and the exit status 1."""

    @functools.wraps(main)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            main(*args, **kwargs)
        except TableError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error

    return wrapped


def table_method(method: str) -> str:
    """The name of a record's method in a table: `TABLE_METHODS`, with `+composition` kept."""
    plain, plus, composed = method.partition("+")
    return TABLE_METHODS.get(plain, plain) + plus + composed


def require_file(path: pathlib.Path) -> pathlib.Path:
    """`path`, if it is an existing file."""
    if not path.is_file():
        raise TableError(f"{path} does not exist" if not path.exists() else f"{path} is no file")
    return path


def read_json(path: pathlib.Path) -> Any:
    """The value of a JSON file."""
    try:
        return json.loads(require_file(path).read_text())
    except json.JSONDecodeError as error:
        raise TableError(f"{path} is not a JSON file ({error})") from error


def read_table(path: pathlib.Path) -> list[Row]:
    """The rows of a CSV file, each a dict from column name to the cell's text."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(require_file(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TableError(f"{path}: no header row")
        return [dict(row) for row in reader]


def require_columns(rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> None:
    """Raise TableError unless every row has each of `columns`."""
    needed = list(dict.fromkeys(columns))
    for row in rows:
        missing = [column for column in needed if column not in row]
        if missing:
            raise TableError(f"the table has no column {', '.join(missing)}")


def read_runs(path: pathlib.Path) -> list[Row]:
    """The rows that `run` made; a row whose column `rule` is set comes from `reselect` and is
    left out (`records_to_table.py` writes both kinds into one table)."""
    return [row for row in read_table(path) if not row.get("rule")]


def write_table(
    path: pathlib.Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write `rows` as a CSV file (gzip if the name ends in `.gz`). A float is written with the
    digits that read back to the same float, a bool as `true` or `false`, None as empty."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row[column]) for column in columns])
    data = buffer.getvalue().encode("utf-8")
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            out.write(data)
    else:
        path.write_bytes(data)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    return str(value)


def number(row: Mapping[str, str], column: str) -> float:
    """The value of a cell as a float; `true` is 1 and `false` is 0."""
    if column not in row:
        raise TableError(f"the table has no column {column!r}")
    text = row[column].strip().lower()
    if text in ("true", "false"):
        return float(text == "true")
    try:
        return float(text)
    except ValueError as error:
        raise TableError(f"column {column!r}: {row[column]!r} is not a number") from error


def setting_value(setting: str) -> int:
    """The number of a setting: 1024 for `K=1024`, and 0 for `none` and for any other setting
    that does not end in `=` and a whole number."""
    _, _, value = setting.partition("=")
    return int(value) if value.isdecimal() else 0


def group(rows: Iterable[Row], *columns: str) -> dict[tuple[str, ...], list[Row]]:
    """The rows grouped by their values in `columns`, in first-seen order."""
    groups: dict[tuple[str, ...], list[Row]] = {}
    for row in rows:
        try:
            key = tuple(row[column] for column in columns)
        except KeyError as error:
            raise TableError(f"the table has no column {error.args[0]}") from error
        groups.setdefault(key, []).append(row)
    return groups


def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float | None]:
    """The mean, and the sample standard deviation over sqrt(n) (None for a single value)."""
    assert values, "no values"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def average_over_tasks(rows: Sequence[Row], column: str) -> tuple[float, float | None]:
    """The mean of `column` over tasks and its standard error (module docstring)."""
    require_columns(rows, ("task_id", "replicate"))
    by_task: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in rows:
        if row["replicate"] in by_task[row["task_id"]]:
            raise TableError(f"{row['task_id']}: replicate {row['replicate']} appears twice")
        by_task[row["task_id"]][row["replicate"]] = number(row, column)
    task_means = [statistics.fmean(values.values()) for _, values in sorted(by_task.items())]
    return mean_and_standard_error(task_means)


def summarize(rows: Sequence[Row], columns: Sequence[str]) -> dict[str, Any]:
    """The average of each of `columns` over the tasks of `rows`, as `<column>` and `<column>_se`,
    with the numbers of tasks and runs and, if the rows have the column `fallback`, of the runs
    scored as the fallback."""
    require_columns(rows, ("task_id", "replicate"))
    summary: dict[str, Any] = {"tasks": len({row["task_id"] for row in rows}), "runs": len(rows)}
    if rows and "fallback" in rows[0]:
        summary["fallback_runs"] = sum(number(row, "fallback") == 1.0 for row in rows)
    for column in columns:
        summary[column], summary[f"{column}_se"] = average_over_tasks(rows, column)
    return summary


def points(rows: Iterable[Row], columns: Sequence[str], *by: str) -> list[dict[str, Any]]:
    """One point for each method and setting (and each value of the columns `by`, which come
    first): `summarize` of its runs. The points of a method are in the order of their settings."""
    result: list[dict[str, Any]] = []
    for key, runs in group(rows, *by, "method", "setting").items():
        labels = dict(zip((*by, "method", "setting"), key, strict=True))
        result.append({**labels, **summarize(runs, columns)})
    return sorted(result, key=lambda point: _point_order(point, by))


def _point_order(point: Mapping[str, Any], by: Sequence[str]) -> tuple[Any, ...]:
    kind, _, _ = str(point["setting"]).partition("=")
    return (
        *(str(point[name]) for name in by),
        point["method"],
        kind,
        setting_value(point["setting"]),
    )


def event_probabilities() -> dict[str, float]:
    """p̂(S) of the benchmark's 108 scored prompt-behavior pairs, by task id."""
    import posteriorbench

    return {task.task_id: task.probability for task in posteriorbench.Benchmark().tasks("scored")}


def probability_bin(probability: float) -> int:
    """The index of the bin of an event probability: 0 below 1e-4, 1 below 1e-3, 2 below 1e-2 and
    3 from 1e-2 on. Raises TableError below 1e-5."""
    if probability < PROBABILITY_BIN_EDGES[0]:
        raise TableError(f"p(S) = {probability} is below the first bin")
    return sum(probability >= edge for edge in PROBABILITY_BIN_EDGES) - 1


def points_by_probability(
    rows: Iterable[Row], probabilities: Mapping[str, float]
) -> list[dict[str, Any]]:
    """For each method, one point for each bin of p̂(S): `capped_forward_kl`, `reverse_kl`,
    `fallback_share` and their `_se`, averaged over the tasks of the bin. `rows` must hold one
    setting per method."""
    result: list[dict[str, Any]] = []
    for (method,), runs in group(rows, "method").items():
        require_columns(runs, ("setting", "task_id"))
        settings = sorted({row["setting"] for row in runs})
        if len(settings) != 1:
            raise TableError(f"{method}: one setting per method, got {', '.join(settings)}")
        missing = sorted({row["task_id"] for row in runs} - set(probabilities))
        if missing:
            raise TableError(
                f"no event probability for the tasks {', '.join(missing)}: the figure bins the "
                "package's 108 scored pairs"
            )
        binned: dict[int, list[Row]] = collections.defaultdict(list)
        for row in runs:
            binned[probability_bin(probabilities[row["task_id"]])].append(row)
        for index, in_bin in sorted(binned.items()):
            summary = summarize(in_bin, ("capped_forward_kl", "reverse_kl", "fallback"))
            summary["fallback_share"] = summary.pop("fallback")
            summary["fallback_share_se"] = summary.pop("fallback_se")
            labels = {"method": method, "setting": settings[0], "probability_bin": index}
            result.append({**labels, **summary})
    return result


def select(rows: Iterable[Row], settings: Mapping[str, str]) -> list[Row]:
    """The rows of the methods in `settings`, each at its setting there; at least one."""
    rows = list(rows)
    require_columns(rows, ("method", "setting"))
    return require_runs(
        [row for row in rows if settings.get(row["method"]) == row["setting"]],
        ", ".join(f"{method} at {setting}" for method, setting in settings.items()),
    )


def require_runs(rows: list[Row], wanted: str) -> list[Row]:
    """`rows`, if there is one; `wanted` names, for the error, the runs the script reads."""
    if not rows:
        raise TableError(f"the table has no run of {wanted}")
    return rows
