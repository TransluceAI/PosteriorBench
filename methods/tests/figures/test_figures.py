"""Each figure script, on the paper's runs: the points it computes are the points of the paper's
figure (the stored `*_points.csv` next to the runs), and it draws."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import estimating_ps
import event_probabilities
import larger_models
import llm_judged_behaviors
import main_comparison
import matplotlib.figure as mfigure
import matplotlib.image
import paper_runs
import paper_style
import prompt_mixtures
import pytest
import selection_signals
import training_by_probability
import training_free_by_probability

SamePoints = Callable[..., Any]
DATA = paper_runs.DATA_DIR
SCORES = ("capped_forward_kl", "capped_forward_kl_se", "reverse_kl", "reverse_kl_se")
BY_PROBABILITY = ("tasks", "runs", "fallback_runs", *SCORES, "fallback_share", "fallback_share_se")
POOL = (*SCORES, "native_hit_rate", "native_hit_rate_se")


def saved(figure: mfigure.Figure, path: pathlib.Path, panels: int) -> None:
    """The figure has `panels` titled axes and is written as a file with content."""
    assert sum(bool(axis.get_title()) for axis in figure.axes) == panels
    paper_style.save(figure, path)
    assert path.stat().st_size > 10_000


def test_main_comparison(
    main_runs: list[paper_runs.Row], tmp_path: pathlib.Path, same_points: SamePoints
) -> None:
    points = main_comparison.points(main_runs)
    stored = paper_runs.read_table(DATA / "main_comparison_points.csv")
    assert len(points) == 214
    same_points(points, stored, ("method", "setting"), ("construction_inferences", *SCORES))
    assert all(point["tasks"] == 21 and point["runs"] == 63 for point in points)
    saved(main_comparison.draw(points), tmp_path / "figure.png", panels=2)


def test_training_by_probability(
    main_runs: list[paper_runs.Row],
    probabilities: dict[str, float],
    tmp_path: pathlib.Path,
    same_points: SamePoints,
) -> None:
    points = training_by_probability.points(main_runs, probabilities)
    stored = paper_runs.read_table(DATA / "training_by_probability_points.csv")
    same_points(points, stored, ("method", "setting", "probability_bin"), BY_PROBABILITY)
    assert [p["tasks"] for p in points if p["method"] == "sft"] == [8, 5, 5, 3]
    methods = list(training_by_probability.SETTINGS)
    figure = paper_style.draw_probability_panels(
        points, methods, training_by_probability.LEGEND_ROWS
    )
    saved(figure, tmp_path / "figure.png", panels=3)


def test_training_free_by_probability(
    pair_runs: list[paper_runs.Row],
    probabilities: dict[str, float],
    tmp_path: pathlib.Path,
    same_points: SamePoints,
) -> None:
    points = training_free_by_probability.points(pair_runs, probabilities)
    stored = paper_runs.read_table(DATA / "training_free_by_probability_points.csv")
    same_points(points, stored, ("method", "setting", "probability_bin"), BY_PROBABILITY)
    assert [p["tasks"] for p in points if p["method"] == "steering"] == [38, 22, 26, 22]
    assert [p["runs"] for p in points if p["method"] == "mixture-eb16"] == [38, 22, 26, 22]
    methods = list(training_free_by_probability.SETTINGS)
    figure = paper_style.draw_probability_panels(
        points, methods, training_free_by_probability.LEGEND_ROWS
    )
    saved(figure, tmp_path / "figure.png", panels=3)


def test_selection_signals(
    pair_runs: list[paper_runs.Row], tmp_path: pathlib.Path, same_points: SamePoints
) -> None:
    points = selection_signals.points(pair_runs)
    stored = paper_runs.read_table(DATA / "selection_signals_points.csv")
    assert len(points) == 4 * 11
    same_points(points, stored, ("method", "setting"), POOL)
    assert all(point["tasks"] == 108 and point["runs"] == 324 for point in points)
    saved(selection_signals.draw(points), tmp_path / "figure.png", panels=3)


def test_prompt_mixtures(
    pair_runs: list[paper_runs.Row], tmp_path: pathlib.Path, same_points: SamePoints
) -> None:
    points = prompt_mixtures.points(pair_runs)
    stored = paper_runs.read_table(DATA / "prompt_mixtures_points.csv")
    assert len(points) == 7 * 4
    same_points(points, stored, ("method", "setting"), POOL)
    assert all(point["tasks"] == 108 and point["runs"] == 108 for point in points)
    saved(prompt_mixtures.draw(points), tmp_path / "figure.png", panels=3)


def test_one_prompt_is_ebjc_selection(pair_runs: list[paper_runs.Row]) -> None:
    """A mixture of one prompt is the prompt that EB-JC selection submits: the runs of
    `mixture-uniform1` are those of `prompt-ebjc` at replicate 0."""
    scores = {
        (row["method"], row["setting"], row["task_id"]): (
            row["capped_forward_kl"],
            row["reverse_kl"],
        )
        for row in pair_runs
        if row["replicate"] == "0" and row["method"] in ("mixture-uniform1", "prompt-ebjc")
    }
    one = {key[1:]: value for key, value in scores.items() if key[0] == "mixture-uniform1"}
    assert len(one) == 4 * 108
    assert all(scores[("prompt-ebjc", *key)] == value for key, value in one.items())


def test_larger_models(
    main_runs: list[paper_runs.Row], same_points: SamePoints, tmp_path: pathlib.Path
) -> None:
    """All three columns are computed from runs: Qwen3.5-0.8B from the main comparison, the two
    larger models from their own table."""
    stored = paper_runs.read_table(DATA / "larger_models_points.csv")
    assert len(stored) == 3 * 48
    larger = paper_runs.read_table(DATA / "larger_models_runs.csv")
    assert len(larger) == 840 and {larger_models.model_of(row["task_id"]) for row in larger} == {
        "Qwen3.8-27B",
        "Gemma-4-31B-it",
    }
    points = larger_models.points([*main_runs, *larger])
    same_points(
        points,
        stored,
        ("model", "method", "setting"),
        ("tasks", "construction_inferences", *SCORES),
    )
    assert {point["model"]: (point["tasks"], point["runs"]) for point in points} == {
        "Qwen3.5-0.8B": (21, 63),
        "Qwen3.8-27B": (2, 6),
        "Gemma-4-31B-it": (2, 6),
    }
    figure = larger_models.draw(points)
    assert [axis.get_title() for axis in figure.axes if axis.get_title()] == [
        "Qwen3.5-0.8B (21 tasks)",
        "Qwen3.8-27B (2 tasks)",
        "Gemma-4-31B-it (2 tasks)",
    ]
    saved(figure, tmp_path / "figure.png", panels=3)


def test_larger_models_names_the_model_of_a_task() -> None:
    assert larger_models.model_of("wc_0670/em_dash") == "Qwen3.5-0.8B"
    assert larger_models.model_of("qwen3.8-27b/wc_0670/chinese_character") == "Qwen3.8-27B"
    assert larger_models.model_of("gemma-4-31b-it/wc_0670/em_dash") == "Gemma-4-31B-it"


@pytest.mark.parametrize(("threshold", "count"), [(0.1, 50), (0.01, 42)])
def test_llm_judged_behaviors(
    threshold: float, count: int, same_points: SamePoints, tmp_path: pathlib.Path
) -> None:
    runs = llm_judged_behaviors.at_threshold(
        paper_runs.read_table(llm_judged_behaviors.RUNS), threshold
    )
    stored = [
        row
        for row in paper_runs.read_table(DATA / "llm_judged_points.csv")
        if float(row["threshold"]) == threshold
    ]
    points = llm_judged_behaviors.points(runs)
    assert len(points) == count and all(point["runs"] == 3 for point in points)
    keys = ("task_id", "method", "setting")
    same_points(points, stored, keys, ("runs", "fallback_runs", "construction_inferences", *SCORES))
    saved(llm_judged_behaviors.draw(points, threshold), tmp_path / "figure.png", panels=4)
    without_cost = [
        {k: v for k, v in row.items() if k != "construction_inferences"} for row in runs
    ]
    with pytest.raises(paper_runs.TableError, match="no construction_inferences"):
        llm_judged_behaviors.draw(llm_judged_behaviors.points(without_cost), threshold)


def test_llm_judged_runs_of_a_threshold_and_the_setting_of_sft() -> None:
    runs = paper_runs.read_table(llm_judged_behaviors.RUNS)
    assert len(llm_judged_behaviors.at_threshold(runs, 0.1)) == 150
    assert len(llm_judged_behaviors.at_threshold(runs, 0.01)) == 126
    without_column = [{k: v for k, v in row.items() if k != "threshold"} for row in runs[:5]]
    assert llm_judged_behaviors.at_threshold(without_column, 0.1) == without_column
    sft = {row["setting"] for row in runs if row["method"] == "sft"}
    points = paper_runs.read_table(DATA / "llm_judged_points.csv")
    sft |= {row["setting"] for row in points if row["method"] == "sft"}
    assert sft == {llm_judged_behaviors.SFT_SETTING}


def test_estimating_ps(
    main_runs: list[paper_runs.Row], tmp_path: pathlib.Path, same_points: SamePoints
) -> None:
    runs = paper_runs.read_table(estimating_ps.RUNS)
    samplers = paper_runs.read_table(estimating_ps.SAMPLERS)
    points = estimating_ps.points(runs)
    stored = paper_runs.read_table(DATA / "estimating_ps_points.csv")
    assert len(points) == 203
    columns = ("runs", "own_estimate", "fallback_estimate", "construction_inferences")
    same_points(
        points, stored, ("method", "setting"), (*columns, "abs_log_error", "abs_log_error_se")
    )
    # every run of the main comparison but Replay's, at the same cost
    cost = {
        tuple(row[key] for key in paper_runs.RUN_KEYS): row["construction_inferences"]
        for row in runs
    }
    main = {
        tuple(row[key] for key in paper_runs.RUN_KEYS): row["construction_inferences"]
        for row in main_runs
        if row["method"] != "replay"
    }
    assert cost == main
    assert len(samplers) == 4968
    saved(estimating_ps.draw(points, samplers), tmp_path / "figure.png", panels=2)


def test_event_probabilities(tmp_path: pathlib.Path) -> None:
    pairs = event_probabilities.pairs()
    assert len(pairs) == 108 and sum(pair["benchmark_task"] for pair in pairs) == 21
    assert event_probabilities.bin_counts(pairs) == [38, 22, 26, 22]
    assert {pair["behavior"] for pair in pairs} == set(event_probabilities.BEHAVIORS)
    figure = event_probabilities.draw(pairs)
    paper_style.save(figure, tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 10_000


def test_close_points_get_different_heights() -> None:
    offsets = event_probabilities.vertical_offsets([1e-3, 1.05e-3, 1.1e-3, 5e-2])
    assert len(set(offsets[:3])) == 3 and offsets[3] == 0.0


def test_a_composed_method_takes_its_methods_style() -> None:
    assert paper_style.series("gfn+composition").color == paper_style.series("gfn").color
    assert paper_style.series("gfn+composition").linestyle == "--"


def test_a_method_that_the_paper_does_not_have_gets_a_style_of_its_own() -> None:
    """One colour that no method of the paper has, the method's name as its label, and the same
    marker whenever it is drawn; its composition is dashed."""
    mine = paper_style.series("my-method")
    assert mine == paper_style.series("my-method")
    assert (mine.label, mine.color, mine.linestyle) == ("my-method", paper_style.OTHER_COLOR, "-")
    assert mine.marker in paper_style.OTHER_MARKERS
    assert paper_style.OTHER_COLOR not in {style.color for style in paper_style.SERIES.values()}
    composed = paper_style.series("my-method+composition")
    assert (composed.color, composed.marker, composed.linestyle) == (mine.color, mine.marker, "--")


def run_row(
    method: str, setting: str, cost: float, forward: float, reverse: float
) -> paper_runs.Row:
    """One run on one task, as a row of a table of runs."""
    return {
        "method": method,
        "setting": setting,
        "task_id": "wc_0346/em_dash",
        "replicate": "0",
        "construction_inferences": repr(cost),
        "capped_forward_kl": repr(forward),
        "reverse_kl": repr(reverse),
        "native_hit_rate": "0.5",
        "fallback": "false",
    }


def test_the_y_axis_of_the_main_comparison_holds_scores_outside_the_papers() -> None:
    """The paper's points are means over 21 tasks and lie within its axes. A run on one task
    can score below 2 nats, or above the paper's largest reverse KL."""
    paper = main_comparison.points(paper_runs.read_table(main_comparison.RUNS))
    for column, (limits, ticks) in main_comparison.Y_AXES.items():
        assert main_comparison.y_axis(paper, column) == (limits, list(ticks))
    assert main_comparison.not_drawn(paper) == []
    mine = main_comparison.points(
        [
            run_row("sft", "B=131072", 110786.0, 0.32, 0.29),
            run_row("replay", "B=131072", 92646.0, 50.0, 109.71),
            run_row("sft+composition", "B=131072", 111658.0, -0.09, 0.18),
            run_row("sft", "B=4096", 50.0, 50.0, 150.9),
        ]
    )
    limits, ticks = main_comparison.y_axis(mine, "capped_forward_kl")
    assert limits == (pytest.approx(0.16), 60.0) and ticks == [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    limits, ticks = main_comparison.y_axis(mine, "reverse_kl")
    assert limits == (pytest.approx(0.08), 250.0)
    assert ticks == [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    assert main_comparison.not_drawn(mine) == [
        "sft at B=4096: 50 inferences are outside the cost axis",
        "sft+composition at B=131072: capped_forward_kl = -0.09 is not positive",
    ]


def test_no_series_of_the_main_comparison_is_drawn_outside_its_panel(
    tmp_path: pathlib.Path,
) -> None:
    """A series that reaches the cap is drawn over the edge of its panel. With a point that the
    panel cannot hold, a cost outside the cost axis here, it is cut at the edge, and the saved
    figure has the size of the paper's."""
    inside = [
        run_row("sft", "B=4096", 4096.0, 50.0, 100.0),
        run_row("sft", "B=8192", 6000.0, 0.02, 0.03),
    ]
    outside = [*inside, run_row("sft", "B=2048", 3.0, 0.5, 0.4)]
    color = paper_style.series("sft").color
    clipped: list[set[bool]] = []
    sizes: list[tuple[int, ...]] = []
    for name, runs in (("inside", inside), ("outside", outside)):
        figure = main_comparison.draw(main_comparison.points(runs))
        forward = next(axis for axis in figure.axes if "Forward" in axis.get_title())
        clipped.append({line.get_clip_on() for line in forward.lines if line.get_color() == color})
        saved(figure, tmp_path / f"{name}.png", panels=2)
        sizes.append(matplotlib.image.imread(tmp_path / f"{name}.png").shape[:2])
    assert clipped == [{False}, {True}]
    assert sizes[0] == sizes[1]


def test_runs_are_added_to_the_papers(
    main_runs: list[paper_runs.Row], tmp_path: pathlib.Path
) -> None:
    """`--add`: a method that the paper's runs do not have is one more series, and a method
    that they have is drawn next to the paper's under another name."""
    added = [
        run_row("my-method", "none", 5000.0, 3.0, 4.0),
        run_row("sft", "B=131072", 110786.0, 0.32, 0.29),
        run_row("sft+composition", "B=131072", 111658.0, 0.4, 0.18),
    ]
    rows = main_comparison.with_added(main_runs, added)
    assert len(rows) == len(main_runs) + 3
    assert [row["method"] for row in rows[-3:]] == [
        "my-method",
        "sft (added)",
        "sft (added)+composition",
    ]
    # A second table with the same method is one more series.
    twice = main_comparison.with_added(rows, added[1:2])
    assert twice[-1]["method"] == "sft (added) (added)"
    points = main_comparison.points(rows)
    assert len(points) == 214 + 3
    figure = main_comparison.draw(points)
    labels = {text.get_text() for legend in figure.legends for text in legend.get_texts()}
    assert {"my-method", "sft (added)", "SFT", "+ composition"} <= labels
    assert "sft (added)+composition" not in labels
    saved(figure, tmp_path / "figure.png", panels=2)


def test_estimating_ps_holds_errors_below_the_papers_axis(tmp_path: pathlib.Path) -> None:
    rows = [
        {**run_row("sft", "B=4096", 5000.0, 1.0, 1.0), "estimate": "own", "abs_log_error": "0.03"},
        {**run_row("sft", "B=8192", 9000.0, 1.0, 1.0), "estimate": "own", "abs_log_error": "0.0"},
    ]
    points = estimating_ps.points(rows)
    figure = estimating_ps.draw(points, [])
    left = next(axis for axis in figure.axes if "Absolute" in axis.get_title())
    assert left.get_ylim() == (pytest.approx(0.016), 200.0)
    assert paper_style.not_drawn(points, ("abs_log_error",), estimating_ps.COST_LIMITS) == [
        "sft at B=8192: abs_log_error = 0 is not positive"
    ]
    saved(figure, tmp_path / "figure.png", panels=2)


def test_llm_judged_behaviors_holds_a_value_below_the_papers_range() -> None:
    """The paper's panel of the reverse KL on direct encouragement starts at 25 nats."""
    task = "qwen3.5-0.8b/wc_0670/direct_encouragement"
    runs = [
        {**run_row("prompt-ebjc", "K=16", 500.0, 20.0, 10.0), "task_id": task},
        {**run_row("prompt-ebjc", "K=64", 1500.0, 20.0, 40.0), "task_id": task},
    ]
    figure = llm_judged_behaviors.draw(llm_judged_behaviors.points(runs), 0.1)
    reverse = [axis for axis in figure.axes if "Reverse" in axis.get_title()][0]
    low, high = reverse.get_ylim()
    assert low < 10.0 and high == 112.0


def test_widened_axes_keep_the_papers_when_every_value_is_inside() -> None:
    assert paper_style.widened((0.0, 52.0), [0.0, 30.0, 52.0]) == (0.0, 52.0)
    low, high = paper_style.widened((0.0, 140.0), [-1.0, 150.9])
    assert low < -1.0 and high > 150.9
    assert paper_style.widened((25.0, 112.0), [10.0, 300.0], upper=False)[1] == 112.0
    limits, ticks = paper_style.widened_log_axis((0.5, 200.0), (0.5, 1, 2), [0.5, 200.0, -3.0])
    assert (limits, ticks) == ((0.5, 200.0), [0.5, 1, 2])
    limits, ticks = paper_style.widened_log_axis((2.0, 60.0), (2, 5), [1.0, 0.0007, 3000.0])
    assert limits == (pytest.approx(0.0004), 6250.0)
    assert ticks[:3] == [pytest.approx(0.0005), pytest.approx(0.001), pytest.approx(0.002)]
    assert ticks[-4:] == [500, 1000, 2000, 5000]


def test_a_table_that_a_script_cannot_use_is_a_table_error(
    main_runs: list[paper_runs.Row], probabilities: dict[str, float], tmp_path: pathlib.Path
) -> None:
    """A file that does not exist, a table without a column that the script reads, and a table
    without the methods or settings of the figure."""
    with pytest.raises(paper_runs.TableError, match="absent.csv does not exist"):
        paper_runs.read_table(tmp_path / "absent.csv")
    without_cost = [
        {k: v for k, v in row.items() if k != "construction_inferences"} for row in main_runs[:3]
    ]
    with pytest.raises(paper_runs.TableError, match="no column 'construction_inferences'"):
        main_comparison.points(without_cost)
    with pytest.raises(paper_runs.TableError, match="no column task_id"):
        main_comparison.points([{"method": "sft", "setting": "B=4096"}])
    with pytest.raises(paper_runs.TableError, match="no run of any method"):
        main_comparison.points([])
    other = [run_row("my-method", "none", 5000.0, 3.0, 4.0)]
    with pytest.raises(paper_runs.TableError, match="no run of sft at B=524288, gfn at"):
        training_by_probability.points(other, probabilities)
    with pytest.raises(paper_runs.TableError, match="no run of prompt-ebjc at K=1024"):
        training_free_by_probability.points(other, probabilities)
    with pytest.raises(paper_runs.TableError, match="no run of prompt-ebjc, prompt-jc"):
        selection_signals.points(other)
    with pytest.raises(paper_runs.TableError, match="no run of mixture-eb<s>, mixture-uniform<s>"):
        prompt_mixtures.points(other)
    with pytest.raises(paper_runs.TableError, match="no run of prompt-ebjc, prompt-hitrate"):
        larger_models.points(other)
    with pytest.raises(paper_runs.TableError, match="no run at the threshold 0.01"):
        llm_judged_behaviors.draw([], 0.01)
    with pytest.raises(paper_runs.TableError, match="no column estimate"):
        estimating_ps.points(other)


def test_a_setting_of_the_readers_own_has_no_number() -> None:
    assert [paper_runs.setting_value(s) for s in ("lr=0.1", "my-variant", "K=", "B=12")] == [
        0,
        0,
        0,
        12,
    ]
    points = paper_runs.points(
        [run_row("my-method", setting, 10.0, 1.0, 1.0) for setting in ("lr=0.1", "lr=0.01")],
        ("capped_forward_kl",),
    )
    assert [point["setting"] for point in points] == ["lr=0.1", "lr=0.01"]


def test_estimating_ps_reads_the_analysis_of_the_command(tmp_path: pathlib.Path) -> None:
    """The JSON of `estimate-ps --runs --samplers` gives the two tables; a run without an
    estimate is left out."""
    import json

    def run(task_id: str, replicate: int, error: float | None) -> dict[str, Any]:
        return {
            "method": "sft",
            "setting": "B=4096",
            "task_id": task_id,
            "replicate": replicate,
            "construction_inferences": 1000.0,
            "passed_threshold": True,
            "native_hit_rate": 0.5,
            "estimate": "own" if error is not None else "unavailable",
            "log_error": error,
            "abs_log_error": None if error is None else abs(error),
            "reverse_kl": 2.0,
            "capped_forward_kl": 3.0,
        }

    runs = [
        run("wc_0346/em_dash", 0, -1.0),
        run("wc_0346/em_dash", 1, None),
        run("wc_0532/profanity", 0, 3.0),
    ]
    samplers = [
        {
            "method": "sft",
            "setting": "B=4096",
            "task_id": "wc_0346/em_dash",
            "replicate": 0,
            "reverse_kl": 2.0,
            "log_error": -1.0,
        }
    ]
    path = tmp_path / "ps.json"
    path.write_text(json.dumps({"points": [], "runs": runs, "samplers": samplers}))
    run_rows, sampler_rows = estimating_ps.read_analysis(path)
    assert len(run_rows) == 2 and len(sampler_rows) == 1
    (point,) = estimating_ps.points(run_rows)
    assert point["abs_log_error"] == pytest.approx(2.0) and point["own_estimate"] == 2
    saved(estimating_ps.draw([point], sampler_rows), tmp_path / "figure.png", panels=2)
    path.write_text(json.dumps({"points": []}))
    with pytest.raises(paper_runs.TableError, match="--runs and --samplers"):
        estimating_ps.read_analysis(path)


def test_llm_judged_behaviors_from_runs_with_their_cost(tmp_path: pathlib.Path) -> None:
    """A table of a reader's runs has the cost of each run, and the figure is drawn from it."""
    task = "qwen3.5-0.8b/wc_0670/physical_embodiment"
    runs = [
        {
            "method": method,
            "setting": setting,
            "task_id": task,
            "replicate": str(replicate),
            "construction_inferences": repr(cost + replicate),
            "capped_forward_kl": repr(20.0 + replicate),
            "reverse_kl": repr(reverse),
            "fallback": "false",
        }
        for method, setting, cost, reverse in (
            ("instruction", "none", 0.0, 60.0),
            ("prompt-ebjc", "K=16", 500.0, 50.0),
            ("prompt-ebjc", "K=64", 1500.0, 40.0),
            ("steering", "none", 30000.0, 250.0),
        )
        for replicate in range(3)
    ]
    points = llm_judged_behaviors.points(runs)
    steering = next(point for point in points if point["method"] == "steering")
    assert steering["construction_inferences"] == pytest.approx(30001.0)
    assert steering["capped_forward_kl_se"] == pytest.approx(1.0 / 3**0.5)
    saved(llm_judged_behaviors.draw(points, 0.1), tmp_path / "figure.png", panels=2)
