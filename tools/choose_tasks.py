"""The scored pairs and the benchmark's tasks, computed from the task table.

A prompt-behavior pair is scored if it has at least 500 hits among its prompt's responses, which
keeps the estimated relative standard error of p̂(S) below 5%.

The benchmark's tasks are, for each behavior, the scored pairs with the lowest, median and
highest p̂(S): of the behavior's scored pairs sorted by p̂(S) (pairs with equal p̂(S) by id), the
first, the one at index `len // 2` and the last. With an even number of pairs the median is the
upper of the two middle ones.

    python tools/choose_tasks.py --check

prints the number of scored pairs and the tasks for the bundled task table and, with --check,
compares both with the table's own `scored` and `benchmark_task` columns.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Annotated, Any

import typer

from posteriorbench import data

MIN_HITS = 500


def is_scored(task: data.Task) -> bool:
    """True if the pair has enough hits to be scored."""
    return task.hit_count >= MIN_HITS


def choose_tasks(scored: Iterable[data.Task]) -> set[str]:
    """The ids of the benchmark's tasks among the scored pairs: for each behavior, the pairs with
    the lowest, median and highest p̂(S) (module docstring). A behavior with one or two scored
    pairs gives one or two tasks."""
    by_behavior: dict[str, list[data.Task]] = {}
    for task in scored:
        by_behavior.setdefault(task.behavior, []).append(task)
    chosen: set[str] = set()
    for pairs in by_behavior.values():
        ordered = sorted(pairs, key=lambda task: (task.probability, task.task_id))
        chosen |= {ordered[0].task_id, ordered[len(ordered) // 2].task_id, ordered[-1].task_id}
    return chosen


def main(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Compare with the table's scored and benchmark_task columns; exit with status "
            "1 if they differ",
        ),
    ] = False,
    data_source: Annotated[
        str | None,
        typer.Option("--data", help="Dataset directory or Hub id (default: the bundled table)"),
    ] = None,
) -> None:
    """Compute the scored pairs and the benchmark's tasks from a task table and print them."""
    tasks = data.load_tasks(data.BUNDLED_SOURCE if data_source is None else data_source)
    scored = {task.task_id for task in tasks if is_scored(task)}
    chosen = choose_tasks(task for task in tasks if is_scored(task))
    report: dict[str, Any] = {
        "pairs": len(tasks),
        "scored": len(scored),
        "tasks": [task.task_id for task in tasks if task.task_id in chosen],
    }
    different: list[str] = []
    if check:
        different = [
            task.task_id
            for task in tasks
            if task.scored != (task.task_id in scored)
            or task.benchmark_task != (task.task_id in chosen)
        ]
        report["differs_from_task_table"] = different
    typer.echo(json.dumps(report, indent=1))
    if different:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
