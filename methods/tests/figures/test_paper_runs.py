"""`figures/paper_runs.py`: tables of runs and their averages."""

from __future__ import annotations

import math
import pathlib

import paper_runs
import posteriorbench
import pytest
from posteriorbench import aggregate


def run(method: str, setting: str, task_id: str, replicate: int, value: float) -> paper_runs.Row:
    return {
        "method": method,
        "setting": setting,
        "task_id": task_id,
        "replicate": str(replicate),
        "capped_forward_kl": repr(value),
        "fallback": "true" if value > 40 else "false",
    }


def test_average_is_over_replicates_then_tasks() -> None:
    rows = [
        run("m", "K=1", "a", 0, 1.0),
        run("m", "K=1", "a", 1, 3.0),
        run("m", "K=1", "b", 0, 8.0),
    ]
    mean, standard_error = paper_runs.average_over_tasks(rows, "capped_forward_kl")
    assert mean == pytest.approx((2.0 + 8.0) / 2)
    assert standard_error == pytest.approx(math.sqrt(18.0) / math.sqrt(2))


def test_average_agrees_with_the_package(main_runs: list[paper_runs.Row]) -> None:
    """One point of the paper's runs, averaged here and by `posteriorbench.aggregate`."""
    mine = [r for r in main_runs if (r["method"], r["setting"]) == ("gfn", "B=1048576")]
    results = [
        posteriorbench.EvaluationResult(
            task_id=r["task_id"],
            method=r["method"],
            setting=r["setting"],
            kind="sampler",
            sampler="",
            seed=int(r["replicate"]),
            replicate=int(r["replicate"]),
            draws=1024,
            hits=0,
            native_hit_rate=float(r["native_hit_rate"]),
            passed_threshold=r["fallback"] == "false",
            fallback=r["fallback"] == "true",
            capped_forward_kl=float(r["capped_forward_kl"]),
            reverse_kl=float(r["reverse_kl"]),
            cap=50.0,
            log_p_event=0.0,
            construction_inferences=float(r["construction_inferences"]),
            budget_passes=None,
            inferences_per_response=1.0,
            cost_rule="ledger",
            ledger={},
            reference_logp="shipped",
            exact=False,
            references=250,
            references_above_cap=0,
            references_with_zero_q=0,
            reverse_hits_used=0,
            fallback_rho=None,
            scorer={},
            sampler_seed=None,
            package_version="",
            dataset={},
            details=None,
        )
        for r in mine
    ]
    summary = aggregate.aggregate(results, expected_replicates={0, 1, 2})
    for column in aggregate.METRICS:
        mean, standard_error = paper_runs.average_over_tasks(mine, column)
        assert mean == pytest.approx(summary.metrics[column].mean, rel=1e-12)
        assert standard_error == pytest.approx(summary.metrics[column].standard_error, rel=1e-12)


def test_a_repeated_replicate_is_an_error() -> None:
    rows = [run("m", "K=1", "a", 0, 1.0), run("m", "K=1", "a", 0, 2.0)]
    with pytest.raises(paper_runs.TableError, match="appears twice"):
        paper_runs.average_over_tasks(rows, "capped_forward_kl")


def test_points_are_in_the_order_of_their_settings() -> None:
    rows = [run("m", f"K={k}", "a", 0, float(k)) for k in (16, 2, 1024)]
    rows += [run("m", "none", "a", 0, 0.5)]
    points = paper_runs.points(rows, ("capped_forward_kl",))
    assert [point["setting"] for point in points] == ["K=2", "K=16", "K=1024", "none"]
    assert points[0]["capped_forward_kl_se"] is None  # one task
    assert [point["fallback_runs"] for point in points] == [0, 0, 1, 0]


def test_table_round_trip(tmp_path: pathlib.Path) -> None:
    rows = [{"method": "m", "value": 0.1 + 0.2, "count": 3.0, "flag": True, "empty": None}]
    for name in ("table.csv", "table.csv.gz"):
        path = tmp_path / name
        paper_runs.write_table(path, list(rows[0]), rows)
        (read,) = paper_runs.read_table(path)
        assert read == {
            "method": "m",
            "value": repr(0.1 + 0.2),
            "count": "3",
            "flag": "true",
            "empty": "",
        }
        assert paper_runs.number(read, "value") == 0.1 + 0.2
        assert paper_runs.number(read, "flag") == 1.0
    with pytest.raises(paper_runs.TableError, match="not a number"):
        paper_runs.number(read, "empty")
    with pytest.raises(paper_runs.TableError, match="no column"):
        paper_runs.number(read, "absent")


def test_probability_bins() -> None:
    assert [paper_runs.probability_bin(p) for p in (1e-5, 9.9e-5, 1e-4, 5e-3, 1e-2, 0.1)] == [
        0,
        0,
        1,
        2,
        3,
        3,
    ]
    with pytest.raises(paper_runs.TableError, match="below the first bin"):
        paper_runs.probability_bin(1e-6)


def test_points_by_probability_need_one_setting_and_known_tasks() -> None:
    rows = [run("m", "K=1", "a", 0, 1.0), run("m", "K=2", "a", 0, 1.0)]
    for row in rows:
        row["reverse_kl"] = "1.0"
    with pytest.raises(paper_runs.TableError, match="one setting per method"):
        paper_runs.points_by_probability(rows, {"a": 1e-3})
    with pytest.raises(paper_runs.TableError, match="no event probability"):
        paper_runs.points_by_probability(rows[:1], {})


def test_setting_value() -> None:
    assert [paper_runs.setting_value(s) for s in ("K=16", "B=4194304", "u=8", "none")] == [
        16,
        4194304,
        8,
        0,
    ]


def test_read_runs_leaves_out_the_runs_of_reselect(tmp_path: pathlib.Path) -> None:
    rows = [
        {"rule": "", "method": "sft", "setting": "B=4096", "task_id": "a", "replicate": 0},
        {
            "rule": "threshold-1pct",
            "method": "sft",
            "setting": "B=4096",
            "task_id": "a",
            "replicate": 0,
        },
    ]
    path = tmp_path / "runs.csv"
    paper_runs.write_table(path, list(rows[0]), rows)
    assert len(paper_runs.read_table(path)) == 2
    assert [row["rule"] for row in paper_runs.read_runs(path)] == [""]
