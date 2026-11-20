"""The look of the paper's figures: one style per method, and the parts that several figures share.

A method family has one colour: prompt selection is blue, the mixture green, Instruction purple,
Steering magenta, SFT black, Replay grey, and the training methods from prior work are warm. A
composed method has its method's colour and a dashed line. A method that the paper does not have
is drawn in a colour of its own, with its name as its label.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import zlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib
import matplotlib.axes as maxes
import matplotlib.backends.backend_agg as backend_agg
import matplotlib.figure as mfigure
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np

LineStyle = Literal["-", "--", ":"]
COMPOSITION = "+composition"
FORWARD_TITLE = r"Forward $\mathrm{KL}_{50}(p_S\|q_S)$"
REVERSE_TITLE = r"Reverse $\mathrm{KL}(q_S\|p_S)$"
COVERAGE, REALISM = "(coverage)", "(realism)"
GRID_COLOR = "#dddddd"
RC: dict[str, Any] = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclasses.dataclass(frozen=True)
class Series:
    """How one method is drawn: its legend label, colour, marker, marker size and line style."""

    label: str
    color: str
    marker: str
    size: float
    linestyle: LineStyle = "-"


SERIES: dict[str, Series] = {
    "prompt-ebjc": Series("Prompt: EB-JC", "#08519c", "o", 3.5),
    "prompt-ebjc+composition": Series("Prompt: EB-JC + composition", "#08519c", "o", 4.4, "--"),
    "prompt-jc": Series("Prompt: JC", "#1a90ff", "^", 4.8),
    "prompt-hitrate": Series("Prompt: hit rate", "#6fc0ea", "v", 4.4),
    "prompt-iwae": Series("Prompt: IWAE", "#8aa6dd", "D", 4.2),
    "prompt-ce": Series("Prompt: CE method", "#00728d", "*", 5.8),
    "mixture-eb16": Series("Mixture: EB-16", "#2a9d4a", "p", 4.2),
    "mixture-eb16+composition": Series("Mixture: EB-16 + composition", "#2a9d4a", "p", 4.6, "--"),
    "instruction": Series("Instruction", "#7b3fa0", "s", 4.3),
    "instruction+composition": Series("Instruction + composition", "#7b3fa0", "^", 5.0, "--"),
    "steering": Series("Steering", "#d6338f", "D", 4.2),
    "steering+composition": Series("Steering + composition", "#d6338f", "v", 5.0, "--"),
    "sft": Series("SFT", "#222222", "s", 3.7),
    "sft+composition": Series("SFT + composition", "#222222", "s", 4.6, "--"),
    "replay": Series("Replay", "#8c8c8c", "h", 4.2),
    "gfn": Series("GFlowNet", "#f28e2b", "X", 5.5),
    "ctl": Series("CTL", "#5c3317", "P", 4.8),
    "gdc": Series("GDC", "#7a1030", "d", 4.7),
    "kl-dpg": Series("KL-DPG", "#c8302a", ">", 5.3),
    "js-dpg": Series("JS-DPG", "#cfa800", "<", 5.3),
}


def style_context(**overrides: Any) -> Any:
    """A context in which matplotlib draws with the settings `RC`, changed by `overrides`
    (written with `_` for `.`: `font_size=8`)."""
    settings = {**RC, **{name.replace("_", "."): value for name, value in overrides.items()}}
    return matplotlib.rc_context(settings)  # pyright: ignore[reportArgumentType]


# The style of a method that the paper does not have: one colour that no method of the paper
# has (a yellow-green), and one of these markers, chosen by the method's name.
OTHER_COLOR = "#7cb518"
OTHER_MARKERS = ("o", "s", "^", "v", "D", "P", "X")
OTHER_SIZE = 5.5


def series(method: str) -> Series:
    """The style of a method. A composed method without an entry of its own has its method's
    style with a dashed line. A method that the paper does not have has `OTHER_COLOR`, its name
    as its label, and the same marker whenever it is drawn."""
    if method in SERIES:
        return SERIES[method]
    plain = method.removesuffix(COMPOSITION)
    composed = plain != method
    if composed and plain in SERIES:
        base = SERIES[plain]
        return dataclasses.replace(base, label=f"{base.label} + composition", linestyle="--")
    marker = OTHER_MARKERS[zlib.crc32(plain.encode()) % len(OTHER_MARKERS)]
    return Series(method, OTHER_COLOR, marker, OTHER_SIZE, "--" if composed else "-")


def new_figure(width_in: float, height_in: float) -> mfigure.Figure:
    """An empty figure of the given size that draws to files, without pyplot."""
    figure = mfigure.Figure(figsize=(width_in, height_in))
    backend_agg.FigureCanvasAgg(figure)
    return figure


def draw_series(
    axis: maxes.Axes,
    x: Sequence[float],
    y: Sequence[float],
    standard_error: Sequence[float | None],
    style: Series,
    *,
    top: float | None = None,
    linewidth: float = 1.2,
    clip: bool = True,
) -> None:
    """One method on one panel: a line with a band of one standard error for several points, a
    marker with an error bar for a single point. The band stops at `top`. `x`, `y` and
    `standard_error` have one entry per point; a missing standard error is drawn as zero."""
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    errors = np.asarray([0.0 if value is None else value for value in standard_error])
    assert xs.shape == ys.shape == errors.shape, (xs.shape, ys.shape, errors.shape)
    if len(xs) > 1:
        upper = ys + errors if top is None else np.minimum(ys + errors, top)
        axis.fill_between(
            xs, ys - errors, upper, color=style.color, alpha=0.07, linewidth=0, zorder=1
        )
        axis.plot(
            xs,
            ys,
            color=style.color,
            linewidth=linewidth,
            linestyle=style.linestyle,
            zorder=2,
            clip_on=clip,
        )
    else:
        axis.errorbar(
            xs,
            ys,
            yerr=errors,
            fmt="none",
            ecolor=style.color,
            alpha=0.55,
            elinewidth=0.85,
            capsize=2,
            zorder=2,
        )
    axis.plot(
        xs,
        ys,
        linestyle="none",
        color=style.color,
        marker=style.marker,
        markersize=style.size,
        markeredgecolor="white",
        markeredgewidth=0.3,
        zorder=4 + (6 - style.size) / 10,
        clip_on=clip,
    )


def add_zero_slot(
    figure: mfigure.Figure, axis: maxes.Axes, width_in: float = 0.30, gap_in: float = 0.09
) -> maxes.Axes:
    """Split a panel with a logarithmic cost axis into a narrow slot for a cost of zero and the
    logarithmic axis, with a break mark between them, and return the slot's axes. The slot takes
    the panel's left edge, its y axis and its y label; the title and the x label stay centred on
    the whole panel. Call after the figure's layout is fixed: it moves `axis`."""
    figure_width = figure.get_size_inches()[0]
    box = axis.get_position()
    slot, gap = width_in / figure_width, gap_in / figure_width
    centre = box.x0 + box.width / 2
    axis.set_position((box.x0 + slot + gap, box.y0, box.width - slot - gap, box.height))
    zero = figure.add_axes((box.x0, box.y0, slot, box.height), sharey=axis)
    zero.set_xlim(-1, 1)
    zero.set_xticks([0], ["0"])
    zero.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    zero.set_axisbelow(True)
    zero.set_ylabel(axis.get_ylabel())
    axis.set_ylabel("")
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", which="both", left=False, labelleft=False)
    moved = axis.get_position()
    axis.title.set_x((centre - moved.x0) / moved.width)
    axis.xaxis.label.set_x((centre - moved.x0) / moved.width)
    edge, width = matplotlib.rcParams["axes.edgecolor"], matplotlib.rcParams["axes.linewidth"]
    for owner, end in ((zero, 1.0), (axis, 0.0)):
        mark = mlines.Line2D(
            [end],
            [0.0],
            transform=owner.transAxes,
            marker=[(-1, -2), (1, 2)],
            markersize=7.0,
            linestyle="none",
            color=edge,
            markeredgecolor=edge,
            markeredgewidth=width,
            clip_on=False,
            zorder=5,
        )
        owner.add_artist(mark)
    return zero


def legend_handle(style: Series, label: str | None = None) -> mlines.Line2D:
    """The legend entry of a method: its line and marker."""
    return mlines.Line2D(
        [],
        [],
        color=style.color,
        marker=style.marker,
        markersize=style.size,
        linewidth=1.1,
        linestyle=style.linestyle,
        label=style.label if label is None else label,
    )


def composition_handle() -> mlines.Line2D:
    """The legend entry that stands for every composed method: a dashed line."""
    return mlines.Line2D(
        [], [], color="#444444", linewidth=1.1, linestyle="--", label="+ composition"
    )


def stack_legend_rows(
    figure: mfigure.Figure,
    rows: Sequence[Sequence[mlines.Line2D]],
    above: Sequence[maxes.Axes],
    gap_in: float = 0.12,
) -> None:
    """Draw one centred legend row per entry of `rows` above the titles of the axes `above`, the
    first row at the top."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    height = figure.get_size_inches()[1]
    top = max(axis.title.get_window_extent(renderer).y1 for axis in above) / figure.bbox.height
    position = top + gap_in / height
    for handles in reversed(rows):
        legend = figure.legend(
            handles=list(handles),
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.51, position),
            borderpad=0,
            borderaxespad=0,
            columnspacing=1.15,
            handletextpad=0.45,
        )
        position += legend.get_window_extent(renderer).height / figure.bbox.height + 0.035 / height


def log_cost_axis(
    axis: maxes.Axes, limits: tuple[float, float], ticks: Mapping[float, str], minor: bool = False
) -> None:
    """A logarithmic axis of construction cost with the given limits and labelled ticks; with
    `minor`, unlabelled ticks at 2, 3, ..., 9 times each power of ten."""
    axis.set_xscale("log")
    axis.set_xlim(*limits)
    axis.set_xticks(list(ticks), list(ticks.values()))
    if minor:
        axis.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
        )
        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        axis.xaxis.set_minor_locator(mticker.NullLocator())


def widened_log_axis(
    limits: tuple[float, float], ticks: Sequence[float], values: Sequence[float]
) -> tuple[tuple[float, float], list[float]]:
    """The limits and ticks of a logarithmic axis that holds every positive value of `values`:
    `limits` and `ticks` if it does. Else the axis goes down to the next of 1, 2 and 5 times a
    power of ten below the smallest value, or up to the next above the largest, with a tick at
    each of these steps that it adds."""
    low, high = limits
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    result = list(ticks)
    if positive and min(positive) < low:
        floor = _step(min(positive), below=True)
        added = [step for step in _steps(floor, min(result)) if step < min(result)]
        low, result = 0.8 * floor, added + result
    if positive and max(positive) > high:
        ceiling = _step(max(positive), below=False)
        added = [step for step in _steps(max(result), ceiling) if step > max(result)]
        high, result = 1.25 * ceiling, result + added
    return (low, high), result


def _steps(first: float, last: float) -> list[float]:
    """1, 2 and 5 times each power of ten, from `first` to `last`, in increasing order."""
    powers = range(math.floor(math.log10(first)), math.floor(math.log10(last)) + 1)
    steps = [digit * 10.0**power for power in powers for digit in (1, 2, 5)]
    return [step for step in steps if first <= step <= last]


def _step(value: float, below: bool) -> float:
    """The largest of 1, 2 and 5 times a power of ten that is at most `value`, or with
    `below` false the smallest that is at least `value`."""
    power = math.floor(math.log10(value))
    steps = [digit * 10.0**exponent for exponent in (power, power + 1) for digit in (1, 2, 5)]
    if below:
        return max(step for step in steps if step <= value)
    return min(step for step in steps if step >= value)


def widened(
    limits: tuple[float, float], values: Sequence[float], upper: bool = True
) -> tuple[float, float]:
    """The limits of a linear axis that holds every value of `values`: `limits` if it does, and
    else moved beyond the smallest value, and with `upper` beyond the largest, by a twentieth
    of the axis."""
    low, high = limits
    finite = [value for value in values if math.isfinite(value)]
    if finite and min(finite) < low:
        low = min(finite) - 0.05 * (high - min(finite))
    if upper and finite and max(finite) > high:
        high = max(finite) + 0.05 * (max(finite) - low)
    return low, high


def not_drawn(
    points: Sequence[Mapping[str, Any]], columns: Sequence[str], cost_limits: tuple[float, float]
) -> list[str]:
    """One line for each point that logarithmic axes cannot show: a value of one of `columns`
    that is not positive, or a cost that is not zero and lies outside `cost_limits`."""
    lines: list[str] = []
    for point in points:
        name = f"{point['method']} at {point['setting']}"
        cost = point["construction_inferences"]
        if cost != 0 and not cost_limits[0] <= cost <= cost_limits[1]:
            lines.append(f"{name}: {cost:g} inferences are outside the cost axis")
        for column in columns:
            if point[column] <= 0:
                lines.append(f"{name}: {column} = {point[column]:.3g} is not positive")
    return lines


def percent_axis(axis: maxes.Axes, ticks: Sequence[float]) -> None:
    """Label the y axis of a share in percent, with ticks at the given shares."""
    axis.set_yticks(list(ticks), [f"{100 * tick:.0f}%" for tick in ticks])


def save(figure: mfigure.Figure, out: pathlib.Path) -> None:
    """Write the figure to `out`; the file type follows the name (`.pdf`, `.png`)."""
    figure.savefig(out, bbox_inches="tight", pad_inches=0.04, dpi=200)


# The three panels of a figure by event probability: the column of a point, the title, the
# limits of the y axis and its ticks.
PROBABILITY_PANELS: tuple[tuple[str, str, tuple[float, float], tuple[float, ...]], ...] = (
    ("capped_forward_kl", f"{FORWARD_TITLE}\n{COVERAGE}", (0.0, 52.0), (0, 20, 40)),
    ("reverse_kl", f"{REVERSE_TITLE}\n{REALISM}", (0.0, 140.0), (0, 50, 100)),
    ("fallback_share", "Runs scored\nas the fallback", (0.0, 1.05), (0, 0.25, 0.5, 0.75, 1.0)),
)
PROBABILITY_SLOTS = 6  # the methods of a bin are drawn side by side, in this many places


def draw_probability_panels(
    points: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    legend_rows: Sequence[Sequence[str]],
) -> mfigure.Figure:
    """Capped forward KL, reverse KL and the share of runs scored as the fallback against the
    event probability, one point per method and bin of p(S) (`paper_runs.points_by_probability`).
    The methods are drawn in the order of `methods`, each a little to the right of the one
    before, with error bars and bands of one standard error."""
    assert len(methods) <= PROBABILITY_SLOTS, methods
    with style_context():
        figure = new_figure(9.22, 3.0)
        axes = list(figure.subplots(1, 3))
        for axis, (column, title, limits, ticks) in zip(axes, PROBABILITY_PANELS, strict=True):
            for place, method in enumerate(methods):
                style = series(method)
                mine = sorted(
                    (p for p in points if p["method"] == method), key=lambda p: p["probability_bin"]
                )
                shift = (place - (PROBABILITY_SLOTS - 1) / 2) * 0.05
                x = [10.0 ** (-4.5 + p["probability_bin"] + shift) for p in mine]
                y = np.asarray([p[column] for p in mine], dtype=float)
                errors = np.asarray([p[f"{column}_se"] or 0.0 for p in mine], dtype=float)
                axis.fill_between(
                    x, y - errors, y + errors, color=style.color, alpha=0.07, linewidth=0
                )
                axis.errorbar(
                    x,
                    y,
                    yerr=errors,
                    color=style.color,
                    linestyle=style.linestyle,
                    linewidth=1.2,
                    elinewidth=0.85,
                    capsize=2,
                    marker=style.marker,
                    markersize=style.size,
                    markeredgecolor="white",
                    markeredgewidth=0.3,
                    zorder=3,
                    clip_on=False,
                )
            axis.set_title(title, pad=8)
            axis.set_xscale("log")
            axis.set_xlim(1e-5, 1e-1)
            axis.set_xticks([1e-5, 1e-3, 1e-1])
            axis.set_xticks([1e-4, 1e-2], minor=True)
            axis.xaxis.set_minor_formatter(mticker.NullFormatter())
            axis.grid(which="both", color=GRID_COLOR, linewidth=0.6)
            axis.set_axisbelow(True)
            drawn = [p[column] for p in points if p["method"] in methods]
            axis.set_ylim(*widened(limits, drawn))
            if column == "fallback_share":
                percent_axis(axis, ticks)
            else:
                axis.set_yticks(list(ticks))
                axis.set_ylabel("Nats")
        axes[1].set_xlabel("Event probability $p(S)$")
        figure.tight_layout(w_pad=1.6)
        handles = [[legend_handle(series(method)) for method in row] for row in legend_rows]
        stack_legend_rows(figure, handles, axes)
    return figure


POOL_TICKS = {1: "1", 4: "4", 16: "16", 64: "64", 256: "256", 1024: "1,024"}


def pool_size_axis(axis: maxes.Axes, pool_sizes: Sequence[int]) -> None:
    """A logarithmic axis of the candidate pool size K, with ticks at every fourth power of two
    among `pool_sizes`."""
    axis.set_xscale("log", base=2)
    axis.set_xlim(min(pool_sizes) / 1.35, max(pool_sizes) * 1.35)
    ticks = [size for size in POOL_TICKS if min(pool_sizes) <= size <= max(pool_sizes)]
    axis.set_xticks(ticks, [POOL_TICKS[size] for size in ticks])
    axis.xaxis.set_minor_locator(mticker.NullLocator())
    axis.set_xlabel("Candidate pool size $K$")


def legend_rows(
    methods: Sequence[str], rows: Sequence[Sequence[str]], labels: Mapping[str, str] | None = None
) -> list[list[mlines.Line2D]]:
    """The legend entries of the methods of `rows` that are among `methods`, row by row; the
    entry `COMPOSITION` stands for the composed methods that no row names. A method of `methods`
    that no row names, and that is not composed, gets a last row."""
    named = {name for row in rows for name in row}
    result: list[list[mlines.Line2D]] = []
    for row in rows:
        handles: list[mlines.Line2D] = []
        for name in row:
            if name == COMPOSITION:
                handles.append(composition_handle())
            elif name in methods:
                handles.append(legend_handle(series(name), (labels or {}).get(name)))
        result.append(handles)
    others = [m for m in methods if m not in named and not m.endswith(COMPOSITION)]
    result.append([legend_handle(series(method)) for method in others])
    return [row for row in result if row]


def band_top(points: Sequence[Mapping[str, Any]], column: str) -> float:
    """The ceiling of the bands of a panel: the next multiple of 10 above every mean plus its
    standard error."""
    highest = max(point[column] + (point[f"{column}_se"] or 0.0) for point in points)
    return float(np.ceil((highest + 1) / 10) * 10)


def draw_against_cost(
    axis: maxes.Axes,
    slot: maxes.Axes,
    points: Sequence[Mapping[str, Any]],
    column: str,
    *,
    top: float | None = None,
) -> None:
    """Every method of `points` on one panel, its `column` against its construction inferences:
    a curve for a method with several settings, one marker with an error bar for a method with
    one. A point whose cost is zero is a marker in `slot`. Methods with small markers are drawn
    last, on top. A series that reaches `top` is drawn over the edge of the panel, unless one
    of its points lies outside the panel."""
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for point in sorted(points, key=lambda p: p["construction_inferences"]):
        by_method.setdefault(point["method"], []).append(point)
    for method in sorted(by_method, key=lambda name: -series(name).size):
        mine = by_method[method]
        at_zero = [[point] for point in mine if point["construction_inferences"] == 0]
        with_cost = [point for point in mine if point["construction_inferences"] != 0]
        for part, panel in (*((single, slot) for single in at_zero), (with_cost, axis)):
            if not part:
                continue
            cost = [point["construction_inferences"] for point in part]
            values = [point[column] for point in part]
            at_top = top is not None and any(np.isclose(value, top) for value in values)
            draw_series(
                panel,
                cost,
                values,
                [point[f"{column}_se"] for point in part],
                series(method),
                top=top,
                clip=not (at_top and _holds(panel, cost, values)),
            )


def _holds(axis: maxes.Axes, x: Sequence[float], y: Sequence[float]) -> bool:
    """Whether every point lies within the limits of the axes."""
    (left, right), (low, high) = sorted(axis.get_xlim()), sorted(axis.get_ylim())
    return all(left <= value <= right for value in x) and all(low <= value <= high for value in y)
