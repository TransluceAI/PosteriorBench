"""Event probabilities p(S) of the 108 scored prompt-behavior pairs (Figure 3).

    python figures/event_probabilities.py [--out event_probabilities.pdf]

One point per pair, one row per behavior; orange points are the benchmark's 21 tasks. The numbers
above the strip count the pairs in each bin of p(S) (`paper_runs.probability_bin`). The
probabilities are the package's, so the script reads no table. The rows have no labels because in
the paper a list of the seven behaviors stands next to the strip.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import matplotlib.figure as mfigure
import matplotlib.ticker as mticker
import paper_runs
import paper_style
import posteriorbench
import typer

# The behaviors from the top row down, in the order of the paper's list.
BEHAVIORS = (
    "em_dash",
    "chinese_character",
    "chatgpt_mention",
    "profanity",
    "goblin_mention",
    "contrastive_phrasing",
    "word_repetition",
)
PAIR_COLOR, TASK_COLOR = "#23678a", "#c78324"
X_LIMITS = (8e-6, 0.16)
OFFSETS = (0.0, -0.17, 0.17, -0.34, 0.34)  # the heights within a row, in the order they are tried
CLEARANCE = 0.075  # two points at one height are at least this far apart, in log10 p(S)


def pairs() -> list[dict[str, Any]]:
    """The 108 scored pairs: `task_id`, `behavior`, `probability` and `benchmark_task`."""
    return [
        {
            "task_id": task.task_id,
            "behavior": task.behavior,
            "probability": task.probability,
            "benchmark_task": task.benchmark_task,
        }
        for task in posteriorbench.Benchmark().tasks("scored")
    ]


def bin_counts(plotted: Sequence[dict[str, Any]]) -> list[int]:
    """The number of pairs in each bin of p(S)."""
    counts = [0] * len(paper_runs.PROBABILITY_BIN_EDGES)
    for pair in plotted:
        counts[paper_runs.probability_bin(pair["probability"])] += 1
    return counts


def vertical_offsets(probabilities: Sequence[float]) -> list[float]:
    """A vertical offset for each point of one row, so that points with nearly the same p(S)
    do not cover each other: each point, in increasing order of p(S), takes the first height
    of `OFFSETS` whose last point is at least `CLEARANCE` away."""
    last: dict[float, float] = {}
    offsets = [0.0] * len(probabilities)
    for index in sorted(range(len(probabilities)), key=lambda i: probabilities[i]):
        position = math.log10(probabilities[index])
        free = [offset for offset in OFFSETS if position - last.get(offset, -math.inf) >= CLEARANCE]
        offset = free[0] if free else min(OFFSETS, key=lambda candidate: last[candidate])
        last[offset] = position
        offsets[index] = offset
    return offsets


def draw(plotted: Sequence[dict[str, Any]]) -> mfigure.Figure:
    """The strip: p(S) on a logarithmic axis, one row per behavior."""
    with paper_style.style_context(font_size=8):
        figure = paper_style.new_figure(3.05, 2.9)
        axis = figure.subplots()
        edges = (*paper_runs.PROBABILITY_BIN_EDGES, X_LIMITS[1])
        for index, (low, high) in enumerate(zip(edges, edges[1:], strict=False)):
            axis.axvspan(low, high, color="#eff3f6" if index % 2 == 0 else "#fafbfc", zorder=0)
            axis.text(
                math.sqrt(low * high), 1.03, f"n = {bin_counts(plotted)[index]}",
                transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5,
                color="#58636c",
            )  # fmt: skip
        for boundary in edges[1:-1]:
            axis.axvline(boundary, color="#c4cdd3", linewidth=0.7, zorder=1)
        for row, behavior in enumerate(BEHAVIORS):
            mine = [pair for pair in plotted if pair["behavior"] == behavior]
            offsets = vertical_offsets([pair["probability"] for pair in mine])
            axis.scatter(
                [pair["probability"] for pair in mine],
                [row + offset for offset in offsets],
                s=16,
                facecolor=[TASK_COLOR if pair["benchmark_task"] else PAIR_COLOR for pair in mine],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_xlim(*X_LIMITS)
        axis.set_ylim(len(BEHAVIORS) - 0.35, -0.55)
        axis.set_yticks([])
        axis.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        axis.xaxis.set_minor_locator(mticker.NullLocator())
        axis.set_xlabel("Event probability $p(S)$", fontsize=8)
        axis.tick_params(axis="x", length=3, color="#71818c", labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    return figure


def main(
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("event_probabilities.pdf"),
) -> None:
    """Draw the figure from the package's task table."""
    paper_style.save(draw(pairs()), out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
