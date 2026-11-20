"""The six tasks on larger models and on LLM-judged behaviors (Table 9).

    python figures/table_generalization_tasks.py

Prints each task's model, behavior, p̂(S) and number of held-out references, as
`posteriorbench_methods.generalization` defines the tasks
(`posteriorbench-methods generalization list-tasks`).
"""

from __future__ import annotations

import typer

from posteriorbench_methods.generalization import tasks


def rows() -> list[tuple[str, str, float, int]]:
    """(model, behavior, p̂(S), references) of the six tasks."""
    return [
        (task.model.paper_name, task.behavior, task.probability, task.references)
        for task in tasks.all_tasks()
    ]


def main() -> None:
    """Print the table."""
    typer.echo(f"{'model':<18}{'behavior':<24}{'p(S)':>10}{'references':>12}")
    for model, behavior, probability, references in rows():
        typer.echo(f"{model:<18}{behavior:<24}{probability:>10.2e}{references:>12}")


if __name__ == "__main__":
    typer.run(main)
