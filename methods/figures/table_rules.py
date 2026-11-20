"""The rule of each behavior and its number of scored prompt-behavior pairs (Table 3).

    python figures/table_rules.py

Prints the rule as the package states it, a Python expression on the decoded response `y`, and the
number of the behavior's pairs with at least 500 hits (108 in all).
"""

from __future__ import annotations

import collections

import event_probabilities
import posteriorbench
import typer


def rows() -> list[tuple[str, str, int]]:
    """(behavior, rule, scored pairs) in the paper's order of the behaviors."""
    scored = posteriorbench.Benchmark().tasks("scored")
    pairs = collections.Counter(task.behavior for task in scored)
    rules = {task.behavior: task.rule for task in scored}
    return [
        (behavior, rules[behavior], pairs[behavior]) for behavior in event_probabilities.BEHAVIORS
    ]


def main() -> None:
    """Print the table."""
    for behavior, rule, pairs in rows():
        typer.echo(f"{behavior:<22}{pairs:>4} pairs  {rule}")
    typer.echo(f"{'':<22}{sum(pairs for _, _, pairs in rows()):>4} pairs")


if __name__ == "__main__":
    typer.run(main)
