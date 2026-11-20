"""The table scripts: the cells they compute from the paper's runs are the paper's cells, and
the cells print as the paper's table has them."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import paper_runs
import pytest
import table_checkpoint_rules
import table_compute_per_inference
import table_generalization_tasks
import table_instructions
import table_rules
import table_tasks
import table_threshold_1pct
import typer.testing

SamePoints = Callable[..., Any]
RESELECTED = paper_runs.DATA_DIR / "reselected_runs.csv"


@pytest.fixture(scope="module")
def all_runs(main_runs: list[paper_runs.Row]) -> list[paper_runs.Row]:
    """The runs of the main comparison and the runs whose checkpoint was chosen again."""
    return [*main_runs, *paper_runs.read_table(RESELECTED)]


def test_threshold_1pct_cells(all_runs: list[paper_runs.Row], same_points: SamePoints) -> None:
    """Every cell of the table, computed from the runs: the 10% columns from the main comparison
    and the 1% columns from the runs whose checkpoint was chosen again."""
    stored = paper_runs.read_table(table_threshold_1pct.CELLS)
    assert len(stored) == 12
    cells = table_threshold_1pct.cells(all_runs)
    assert len(cells) == 12 and all(cell["runs"] == 63 for cell in cells)
    for cell in cells:
        cell["threshold"] = repr(cell["threshold"])
    columns = ("runs", "fallback_runs", "capped_forward_kl", "reverse_kl", "median_native_hit_rate")
    same_points(cells, stored, ("method", "setting", "threshold"), columns)


def test_threshold_1pct_prints_the_papers_rows() -> None:
    output = typer.testing.CliRunner().invoke(_app(table_threshold_1pct.main)).output
    lines = [" ".join(line.split()) for line in output.splitlines()]
    assert "SFT 10% 0 / 63 10.0 17.2 24.4%" in lines
    assert "SFT 1% 0 / 63 4.6 7.4 7.9%" in lines
    assert "KL-DPG 1% 7 / 63 31.0 47.0 6.2%" in lines
    assert "JS-DPG 10% 34 / 63 37.0 76.0 2.1%" in lines


def test_checkpoint_rule_cells(all_runs: list[paper_runs.Row], same_points: SamePoints) -> None:
    """Every cell of the table, computed from the runs of 12 tasks at replicate 0."""
    stored = paper_runs.read_table(table_checkpoint_rules.CELLS)
    rules = [
        (method, rule)
        for method, rules in table_checkpoint_rules.OWN_RULES.items()
        for rule in rules
    ]
    assert [(row["method"], row["own_rule"]) for row in stored] == rules
    cells = table_checkpoint_rules.cells(all_runs)
    assert [(cell["method"], cell["own_rule"]) for cell in cells] == rules
    columns = [
        f"{side}_{name}"
        for side in ("own", "ours")
        for name in ("capped_forward_kl", "reverse_kl", "fallback_runs")
    ]
    same_points(cells, stored, ("method", "own_rule"), ("runs", *columns))


def test_the_submission_rule_side_of_the_checkpoint_table(all_runs: list[paper_runs.Row]) -> None:
    """GFlowNet, CTL and GDC are compared on the trainings of the main comparison. KL-DPG and
    JS-DPG are compared on separate trainings, whose runs of the submission rule have the rule
    `submission-rule` and differ from the main comparison's."""
    separate = {
        row["method"] for row in all_runs if row.get("rule") == table_checkpoint_rules.SUBMISSION
    }
    assert separate == {"kl-dpg", "js-dpg"}
    main = [row for row in all_runs if "rule" not in row]
    for method in table_checkpoint_rules.SETTINGS:
        ours = table_checkpoint_rules.side(all_runs, method, table_checkpoint_rules.SUBMISSION)
        from_main = table_checkpoint_rules.side(main, method, table_checkpoint_rules.SUBMISSION)
        assert (
            ours is not None and from_main is not None and ours["runs"] == from_main["runs"] == 12
        )
        same = ours["capped_forward_kl"] == pytest.approx(from_main["capped_forward_kl"], abs=1e-9)
        assert same == (method not in separate)


def test_checkpoint_rules_prints_the_papers_rows() -> None:
    output = typer.testing.CliRunner().invoke(_app(table_checkpoint_rules.main)).output
    lines = [" ".join(line.split()) for line in output.splitlines()]
    assert "GFlowNet highest-mean-log-reward 25.51 / 58.06 10.91 / 14.11 5 / 1" in lines
    assert "GDC final-policy 31.34 / 52.67 22.68 / 21.88 2 / 0" in lines
    assert "KL-DPG final-checkpoint 50.00 / 126.24 35.07 / 66.23 11 / 4" in lines


def test_compute_per_inference_cells() -> None:
    """The cells round to the numbers of the paper's table."""
    cells = table_compute_per_inference.read_cells(table_compute_per_inference.CELLS)
    printed = {
        c["method"]: (round(c["flops_per_inference"], 2), round(c["gpu_time_per_inference"], 2))
        for c in cells
    }
    assert printed == {
        "sft": (1.08, 1.36),
        "gfn": (1.65, 1.91),
        "ctl": (1.19, 1.62),
        "kl-dpg": (1.91, 1.59),
        "js-dpg": (1.84, 1.51),
        "gdc": (1.09, 1.27),
    }
    assert all(max(pair) < 2.0 for pair in printed.values())


def test_tasks_and_rules() -> None:
    rows = table_tasks.rows()
    assert len(rows) == 21 and rows[0][:2] == ("em_dash", "wc_1852/em_dash")
    by_behavior = [
        [p for b, _, p, _ in rows if b == behavior]
        for behavior in dict.fromkeys(r[0] for r in rows)
    ]
    assert all(len(ps) == 3 and ps == sorted(ps) for ps in by_behavior)
    assert [pairs for _, _, pairs in table_rules.rows()] == [20, 20, 12, 9, 8, 19, 20]


def test_the_task_table_rounds_a_tie_up_as_the_papers_table_does() -> None:
    """p̂(S) of two tasks lies exactly between two numbers of three figures: 438,750 and 11,975
    hits among 50 million responses. The paper's table has 8.78e-03 and 2.40e-04."""
    assert table_tasks.three_figures(438_750, 50_000_000) == "8.78e-03"
    assert table_tasks.three_figures(11_975, 50_000_000) == "2.40e-04"
    # The binary numbers nearest to the two quotients lie below the ties.
    assert f"{438_750 / 50_000_000:.2e}" == "8.77e-03"
    assert f"{11_975 / 50_000_000:.2e}" == "2.39e-04"
    assert table_tasks.three_figures(1, 3) == "3.33e-01"
    assert table_tasks.three_figures(9_995, 1_000) == "1.00e+01"
    assert table_tasks.three_figures(5_112_345, 50_000_000) == "1.02e-01"
    output = typer.testing.CliRunner().invoke(_app(table_tasks.main)).output
    printed = {line.split()[1]: line.split()[2] for line in output.splitlines()[1:]}
    assert len(printed) == 21
    assert printed["wc_0670/chinese_character"] == "8.78e-03"
    assert printed["wc_1853/profanity"] == "2.40e-04"
    # Away from a tie, the exact quotient and the binary number round to the same figures.
    others = {
        task.task_id: f"{task.probability:.2e}"
        for task in table_tasks.tasks()
        if task.task_id not in ("wc_0670/chinese_character", "wc_1853/profanity")
    }
    assert {task_id: printed[task_id] for task_id in others} == others


def test_generalization_tasks_and_instructions() -> None:
    rows = table_generalization_tasks.rows()
    assert [(model, references) for model, _, _, references in rows] == [
        ("Qwen3.5-0.8B", 64),
        ("Qwen3.5-0.8B", 64),
        ("Qwen3.8-27B", 250),
        ("Qwen3.8-27B", 193),
        ("Gemma-4-31B-it", 250),
        ("Gemma-4-31B-it", 250),
    ]
    assert [f"{probability:.2e}" for _, _, probability, _ in rows] == [
        "2.24e-04",
        "8.60e-04",
        "2.55e-03",
        "1.94e-04",
        "9.38e-05",
        "2.80e-02",
    ]
    assert len(table_instructions.rows()) == 9


def _app(main: Callable[..., None]) -> typer.Typer:
    app = typer.Typer()
    app.command()(main)
    return app


def test_compute_per_inference_reads_the_analysis_of_the_command(tmp_path: Any) -> None:
    import json

    methods = [
        {
            "method": "ctl",
            "points": 9,
            "runs": 567,
            "flops_per_inference": 1.186,
            "flops_per_inference_at": "u=4",
            "gpu_time_per_inference": 1.617,
            "gpu_time_per_inference_at": "u=4",
        },
        {
            "method": "replay",
            "points": 1,
            "runs": 3,
            "flops_per_inference": 1.0,
            "flops_per_inference_at": "B=4096",
        },
    ]
    path = tmp_path / "compute.json"
    path.write_text(json.dumps({"methods": methods, "points": []}))
    assert table_compute_per_inference.read_analysis(path) == methods
    result = typer.testing.CliRunner().invoke(
        _app(table_compute_per_inference.main), ["--analysis", str(path)]
    )
    lines = [" ".join(line.split()) for line in result.output.splitlines()]
    assert result.exit_code == 0, result.output
    assert "CTL 1.19 u=4 1.62 u=4" in lines and "Replay 1.00 B=4096" in lines
