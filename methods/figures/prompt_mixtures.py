"""Prompt mixtures on the 108 prompt-behavior pairs (Figure 13).

    python figures/prompt_mixtures.py [--runs TABLE] [--out prompt_mixtures.pdf]

A mixture mixes up to s of the top-ranked prompts of a pool of size K, with EB weights or with
uniform weights; s = 1 is EB-JC selection. A point averages one mixture at one K over the pairs
(one replicate per pair in the paper's runs), with one standard error across pairs.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any

import matplotlib.figure as mfigure
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import paper_runs
import paper_style
import typer

RUNS = paper_runs.DATA_DIR / "training_free_108pairs_runs.csv.gz"
COLUMNS = ("capped_forward_kl", "reverse_kl", "native_hit_rate")
# The colour, marker and marker size of each number s of mixed prompts, and its legend label.
SIZES = {
    1: ("#08519c", "o", 3.5, "EB-JC ($s=1$)"),
    4: ("#74c476", "^", 4.8, "$s=4$"),
    16: ("#2a9d4a", "p", 4.2, "$s=16$"),
    64: ("#00441b", "s", 4.3, "$s=64$"),
}
# The line style of each kind of weights, and its legend label. s = 1 has one prompt and no
# weights; its runs are those of `mixture-uniform1`.
WEIGHTS: dict[str, tuple[paper_style.LineStyle, str]] = {
    "eb": ("-", "EB-weighted"),
    "uniform": (":", "Uniform"),
}
PANELS = (
    ("capped_forward_kl", f"{paper_style.FORWARD_TITLE} {paper_style.COVERAGE}", (12.0, 24.0)),
    ("reverse_kl", f"{paper_style.REVERSE_TITLE} {paper_style.REALISM}", (14.0, 34.0)),
    ("native_hit_rate", "Native hit rate", (0.38, 0.66)),
)
Y_TICKS = {
    "capped_forward_kl": (12, 15, 18, 21, 24),
    "reverse_kl": (14, 18, 22, 26, 30, 34),
    "native_hit_rate": (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
}


def mixture(method: str) -> tuple[str, int] | None:
    """The weights and the number s of a mixture's name (`mixture-eb16` -> ("eb", 16)), or None
    for another method or a number that the figure does not show."""
    for weights in WEIGHTS:
        prefix = f"mixture-{weights}"
        if method.startswith(prefix) and method[len(prefix) :].isdigit():
            size = int(method[len(prefix) :])
            return (weights, size) if size in SIZES else None
    return None


def points(runs: Sequence[paper_runs.Row]) -> list[dict[str, Any]]:
    """For each mixture and pool size, the averages of `COLUMNS` over the pairs."""
    paper_runs.require_columns(runs, ("method",))
    mixtures = [row for row in runs if mixture(row["method"]) is not None]
    wanted = ", ".join(f"mixture-{weights}<s>" for weights in WEIGHTS)
    wanted += f" with s = {', '.join(str(size) for size in SIZES)}"
    return paper_runs.points(paper_runs.require_runs(mixtures, wanted), COLUMNS)


def draw(plotted: Sequence[dict[str, Any]]) -> mfigure.Figure:
    """The three panels against the pool size K. The ranges of the y axes are the paper's; a
    value outside its panel's range widens the range, and the ticks are then matplotlib's."""
    methods = sorted({p["method"] for p in plotted}, key=lambda m: (mixture(m) or ("", 0))[::-1])
    pool_sizes = sorted({paper_runs.setting_value(p["setting"]) for p in plotted})
    with paper_style.style_context():
        figure = paper_style.new_figure(9.3, 3.0)
        axes = list(figure.subplots(1, 3))
        for axis, (column, title, limits) in zip(axes, PANELS, strict=True):
            for method in methods:
                kind = mixture(method)
                assert kind is not None, method
                weights, size = kind
                color, marker, marker_size, _ = SIZES[size]
                linestyle = "-" if size == 1 else WEIGHTS[weights][0]
                style = paper_style.Series(method, color, marker, marker_size, linestyle)
                mine = [p for p in plotted if p["method"] == method]
                paper_style.draw_series(
                    axis,
                    [paper_runs.setting_value(p["setting"]) for p in mine],
                    [p[column] for p in mine],
                    [p[f"{column}_se"] for p in mine],
                    style,
                    linewidth=1.6,
                )
            axis.set_title(title, pad=8)
            paper_style.pool_size_axis(axis, pool_sizes)
            axis.grid(axis="y", color=paper_style.GRID_COLOR, linewidth=0.6)
            axis.set_axisbelow(True)
            shown = paper_style.widened(limits, [p[column] for p in plotted])
            axis.set_ylim(*shown)
            share = column == "native_hit_rate"
            if not share:
                axis.set_ylabel("Nats")
            if shown != limits:
                if share:
                    axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
            elif share:
                paper_style.percent_axis(axis, Y_TICKS[column])
            else:
                axis.set_yticks(list(Y_TICKS[column]))
        figure.tight_layout(w_pad=1.6)
        handles = [
            mlines.Line2D([], [], color=c, marker=m, markersize=s, linestyle="none", label=label)
            for c, m, s, label in SIZES.values()
        ]
        handles += [
            mlines.Line2D([], [], color="#333333", linestyle=linestyle, linewidth=1.6, label=label)
            for linestyle, label in WEIGHTS.values()
        ]
        paper_style.stack_legend_rows(figure, [handles], axes)
    return figure


def main(
    runs: Annotated[pathlib.Path, typer.Option(help="A table of runs (paper_runs.py).")] = RUNS,
    out: Annotated[
        pathlib.Path, typer.Option(help="The figure file (.pdf or .png).")
    ] = pathlib.Path("prompt_mixtures.pdf"),
) -> None:
    """Draw the figure from a table of runs."""
    paper_style.save(draw(points(paper_runs.read_runs(runs))), out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(paper_runs.one_line_errors(main))
