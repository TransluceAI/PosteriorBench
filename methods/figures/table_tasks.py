"""The 21 tasks with their event probabilities (Table 2).

    python figures/table_tasks.py

Prints, for each behavior, its three tasks in increasing order of p̂(S): the task id, p̂(S) and the
prompt. The values are the package's (`posteriorbench list-tasks`).
"""

from __future__ import annotations

import decimal

import event_probabilities
import posteriorbench
import typer
from posteriorbench import data


def tasks() -> list[data.Task]:
    """The 21 tasks, by behavior in the paper's order and by increasing p̂(S) within a
    behavior."""
    order = {behavior: index for index, behavior in enumerate(event_probabilities.BEHAVIORS)}
    return sorted(
        posteriorbench.Benchmark().tasks(),
        key=lambda task: (order[task.behavior], task.probability),
    )


def rows() -> list[tuple[str, str, float, str]]:
    """(behavior, task id, p̂(S), prompt) of the 21 tasks, in the order of `tasks`."""
    return [(task.behavior, task.task_id, task.probability, task.prompt) for task in tasks()]


def three_figures(hits: int, responses: int) -> str:
    """`hits / responses` with three significant figures, as `8.78e-03`. The quotient is taken
    exactly and a tie is rounded up, so 438,750 / 50,000,000 = 0.008775 is 8.78e-03; the binary
    number nearest to it lies below the tie and would print as 8.77e-03."""
    quotient = decimal.Decimal(hits) / decimal.Decimal(responses)
    exponent = quotient.adjusted()
    hundredth = decimal.Decimal("0.01")
    mantissa = quotient.scaleb(-exponent).quantize(hundredth, rounding=decimal.ROUND_HALF_UP)
    if mantissa >= 10:
        mantissa, exponent = (mantissa / 10).quantize(hundredth), exponent + 1
    return f"{mantissa}e{exponent:+03d}"


def main() -> None:
    """Print the table."""
    typer.echo(f"{'behavior':<22}{'task':<30}{'p(S)':>10}  prompt")
    for task in tasks():
        probability = three_figures(task.hit_count, task.num_responses)
        typer.echo(f"{task.behavior:<22}{task.task_id:<30}{probability:>10}  {task.prompt}")


if __name__ == "__main__":
    typer.run(main)
