"""Sample responses of the paper's samplers (Figures 6 and 17 to 21).

    python figures/sample_responses.py [--figure 17]
    python figures/sample_responses.py --record RECORD [--count 5]

Without `--record`, prints the responses that the paper shows, in its arrangement. A gallery
(Figures 17 to 21) has one reference and one response per sampler of `GALLERY` on one task at
replicate 0, each with its scores from the main comparison, in increasing order of reverse KL
with the samplers scored as the fallback last. Figure 6 shows a reference and three samplers on
one profanity task. `¶` marks a line break. With `--record`, prints hits of a record of your own,
in a random order fixed by `--seed`.
"""

from __future__ import annotations

import pathlib
import random
from collections.abc import Sequence
from typing import Annotated, Any

import paper_runs
import paper_style
import typer

from posteriorbench_methods.common import run_record

RESPONSES = paper_runs.DATA_DIR / "sample_responses.jsonl"
RUNS = paper_runs.DATA_DIR / "main_comparison_runs.csv.gz"
REFERENCE = "reference"
FIGURE_6 = 6  # the figure with a reference and three samplers
REPLICATE = "0"
# The samplers of a gallery and their settings, in the order of a tie and of the fallback rows.
GALLERY = {
    "prompt-ebjc": "K=1024",
    "prompt-hitrate": "K=1024",
    "prompt-iwae": "K=1024",
    "mixture-eb16": "K=1024",
    "instruction": "none",
    "steering": "none",
    "sft": "B=4194304",
    "gfn": "B=4194304",
    "ctl": "B=4194304",
    "gdc": "B=1048576",
    "kl-dpg": "B=1048576",
    "js-dpg": "B=1048576",
}
FIGURE_6_SAMPLERS = {"sft": "B=4194304", "prompt-ebjc": "K=1024", "steering": "none"}


def read_responses(path: pathlib.Path = RESPONSES) -> list[dict[str, Any]]:
    """The shown responses: `figure`, `task_id`, `method` (`reference` for a reference),
    `setting`, `replicate` and `text`, in the order of the paper's figures and rows."""
    return run_record.read_jsonl(path)


def run_of(
    runs: Sequence[paper_runs.Row], method: str, setting: str, task_id: str
) -> paper_runs.Row:
    """The run of a sampler on a task at replicate 0."""
    for row in runs:
        if (row["method"], row["setting"], row["task_id"], row["replicate"]) == (
            method,
            setting,
            task_id,
            REPLICATE,
        ):
            return row
    raise paper_runs.TableError(f"no run of {method} at {setting} on {task_id}")


def gallery_rows(runs: Sequence[paper_runs.Row], task_id: str) -> list[tuple[str, paper_runs.Row]]:
    """The samplers of a gallery with their runs on the task, in the order shown: by reverse KL,
    lowest first, then the samplers scored as the fallback in the order of `GALLERY`."""
    mine = [(method, run_of(runs, method, setting, task_id)) for method, setting in GALLERY.items()]
    passing = [pair for pair in mine if pair[1]["fallback"] != "true"]
    failing = [pair for pair in mine if pair[1]["fallback"] == "true"]
    return sorted(passing, key=lambda pair: paper_runs.number(pair[1], "reverse_kl")) + failing


def shown(text: str) -> str:
    """A response on one line, with `¶` for each run of line breaks."""
    return " ¶ ".join(part.strip() for part in text.split("\n") if part.strip())


def scores(run: paper_runs.Row) -> str:
    """Capped forward KL / reverse KL; native hit rate."""
    forward = paper_runs.number(run, "capped_forward_kl")
    reverse = paper_runs.number(run, "reverse_kl")
    return f"{forward:.1f} / {reverse:.1f}; {paper_runs.number(run, 'native_hit_rate'):.0%}"


def print_gallery(
    figure: int, responses: Sequence[dict[str, Any]], runs: Sequence[paper_runs.Row]
) -> None:
    """One gallery: the reference, then a row per sampler."""
    mine = {response["method"]: response for response in responses if response["figure"] == figure}
    task_id = mine[REFERENCE]["task_id"]
    probability = paper_runs.event_probabilities()[task_id]
    typer.echo(f"== Figure {figure}: {task_id}, p(S) = {probability:.1e}")
    typer.echo(f"Posterior reference\n    {shown(mine[REFERENCE]['text'])}")
    for method, run in gallery_rows(runs, task_id):
        label = paper_style.series(method).label
        if run["fallback"] == "true":
            rate = paper_runs.number(run, "native_hit_rate")
            typer.echo(
                f"{label}\n    (Hit rate {rate:.1%}, below the hit-rate threshold. Scored as the fallback.)"
            )
        else:
            typer.echo(f"{label}  {scores(run)}\n    {shown(mine[method]['text'])}")
    typer.echo("")


def print_figure_6(responses: Sequence[dict[str, Any]], runs: Sequence[paper_runs.Row]) -> None:
    """Figure 6: a reference and three samplers on one task, with each sampler's cost."""
    mine = {
        response["method"]: response for response in responses if response["figure"] == FIGURE_6
    }
    task_id = mine[REFERENCE]["task_id"]
    typer.echo(f"== Figure {FIGURE_6}: {task_id}")
    typer.echo(f"Posterior reference\n    {shown(mine[REFERENCE]['text'])}")
    for method, setting in FIGURE_6_SAMPLERS.items():
        run = run_of(runs, method, setting, task_id)
        cost = paper_runs.number(run, "construction_inferences")
        forward = paper_runs.number(run, "capped_forward_kl")
        reverse = paper_runs.number(run, "reverse_kl")
        label = paper_style.series(method).label
        typer.echo(f"{label}  {cost:.1e} inferences; {forward:.1f} / {reverse:.1f}")
        typer.echo(f"    {shown(mine[method]['text'])}")
    typer.echo("")


def record_hits(record: pathlib.Path, count: int, seed: int) -> list[list[int]]:
    """The token ids of the first `count` hits among a record's evaluation draws, in a random
    order of the draws fixed by `seed`."""
    path = record / "draws" / "evaluation.jsonl"
    if not path.exists():
        raise paper_runs.TableError(
            f"{record}: no draws/evaluation.jsonl (a run without a sampler?)"
        )
    draws = run_record.read_jsonl(path)
    random.Random(seed).shuffle(draws)
    return [draw["token_ids"] for draw in draws if draw["hit"]][:count]


def print_record(record: pathlib.Path, count: int, seed: int) -> None:
    """Hits of a record of the reader's own, decoded with the tokenizer of the benchmark's
    model, which is loaded after the record is read."""
    import posteriorbench.original_model
    import transformers

    hits = record_hits(record, count, seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        posteriorbench.original_model.DEFAULT_MODEL,
        revision=posteriorbench.original_model.MODEL_REVISION,
    )
    for token_ids in hits:
        typer.echo(f"    {shown(tokenizer.decode(token_ids, skip_special_tokens=True))}")


def main(
    figure: Annotated[
        int | None,
        typer.Option(
            "--figure",
            help="One figure of sample responses, by its number in the paper: 6, or 17 to 21. "
            "Default: every one.",
        ),
    ] = None,
    record: Annotated[
        pathlib.Path | None, typer.Option(help="A record directory of your own runs.")
    ] = None,
    count: Annotated[int, typer.Option(help="With --record: how many hits to print.")] = 5,
    seed: Annotated[int, typer.Option(help="With --record: the seed of the random order.")] = 0,
) -> None:
    """Print the paper's sample responses, or hits of a record."""
    if record is not None:
        print_record(record, count, seed)
        return
    responses, runs = read_responses(), paper_runs.read_table(RUNS)
    figures: list[int] = list(dict.fromkeys(response["figure"] for response in responses))
    if figure is not None and figure not in figures:
        raise typer.BadParameter(f"--figure is one of {', '.join(map(str, figures))}")
    for number in figures if figure is None else [figure]:
        if number == FIGURE_6:
            print_figure_6(responses, runs)
        else:
            print_gallery(number, responses, runs)


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
