"""System prompts that three selection signals return from the same candidates (Table 10).

    python figures/selected_prompts.py

Prints the prompt that EB-JC, hit-rate and IWAE selection submitted on the median em-dash task
(`wc_1569/em_dash`) from the same 1,024 candidates, at replicate 0. The prompt that a run of
yours submitted is `submitted.system_prompt` in the `construction.json` of its record.
"""

from __future__ import annotations

import pathlib

import paper_runs
import paper_style
import typer

from posteriorbench_methods.common import run_record

PROMPTS = paper_runs.DATA_DIR / "selected_prompts.jsonl"


def read_prompts(path: pathlib.Path = PROMPTS) -> list[dict[str, str]]:
    """The shown prompts: `task_id`, `method`, `setting`, `replicate` and `system_prompt`."""
    return run_record.read_jsonl(path)


def main() -> None:
    """Print the three prompts."""
    for prompt in read_prompts():
        label = paper_style.series(prompt["method"]).label
        where = f"{prompt['task_id']}, {prompt['setting']}, replicate {prompt['replicate']}"
        typer.echo(f"{label} ({where})\n    {prompt['system_prompt']}\n")


if __name__ == "__main__":
    typer.run(main)
