"""The Instruction text of each behavior (Table 6).

    python figures/table_instructions.py

Prints the seven texts of the behaviors that a rule defines, which are the package's, and the two
of the behaviors that a language model judges, which are in
`posteriorbench_methods.generalization`.
"""

from __future__ import annotations

import event_probabilities
import posteriorbench.instructions
import typer

from posteriorbench_methods.generalization import instructions


def rows() -> list[tuple[str, str]]:
    """(behavior, Instruction text), the benchmark's behaviors first."""
    texts = posteriorbench.instructions.INSTRUCTIONS
    benchmark = [(behavior, texts[behavior]) for behavior in event_probabilities.BEHAVIORS]
    return benchmark + list(instructions.LLM_JUDGED_INSTRUCTIONS.items())


def main() -> None:
    """Print the table."""
    for behavior, text in rows():
        typer.echo(f"{behavior}\n    {text}")


if __name__ == "__main__":
    typer.run(main)
