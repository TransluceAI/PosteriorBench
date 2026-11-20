"""`figures/records_to_table.py`: from the records of an output root to a table of runs."""

from __future__ import annotations

import json
import pathlib

import main_comparison
import paper_runs
import paper_style
import pytest
import records_to_table
import table_threshold_1pct


def write_result(
    root: pathlib.Path, method: str, setting: str, task_id: str, replicate: int, value: float
) -> None:
    directory = root / method / setting / task_id.replace("/", "__") / f"rep{replicate}"
    directory.mkdir(parents=True)
    result = {
        "task_id": task_id,
        "method": method,
        "setting": setting,
        "replicate": replicate,
        "construction_inferences": 100.0 * (replicate + 1),
        "capped_forward_kl": value,
        "reverse_kl": 2 * value,
        "native_hit_rate": 0.25,
        "fallback": False,
        "kind": "sampler",
    }
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "result_recomputed.json").write_text(
        json.dumps({**result, "capped_forward_kl": -1.0})
    )


def test_records_become_a_table_that_the_scripts_read(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "runs"
    for replicate in (0, 1):
        write_result(out, "sft", "B=524288", "wc_0346/em_dash", replicate, 1.0 + replicate)
        write_result(out, "sft", "B=524288", "wc_1853/goblin_mention", replicate, 3.0)
        write_result(
            out / "reselected" / "threshold-1pct",
            "sft",
            "B=524288",
            "wc_0346/em_dash",
            replicate,
            0.5,
        )
    files = records_to_table.result_files([out])
    assert len(files) == 6 and all(path.name == "result.json" for path in files)
    table = tmp_path / "my_runs.csv"
    paper_runs.write_table(table, records_to_table.COLUMNS, records_to_table.rows(files))
    rows = paper_runs.read_table(table)
    assert sorted({row["rule"] for row in rows}) == ["", "threshold-1pct"]
    (point,) = main_comparison.points([row for row in rows if row["rule"] == ""])
    assert point["capped_forward_kl"] == pytest.approx((1.5 + 3.0) / 2)
    assert point["construction_inferences"] == pytest.approx(150.0)
    cells = table_threshold_1pct.cells(rows)
    assert [(cell["threshold"], cell["runs"], cell["capped_forward_kl"]) for cell in cells] == [
        (0.1, 4, pytest.approx(2.25)),
        (0.01, 2, pytest.approx(0.5)),
    ]


def test_gflownet_records_are_named_as_in_the_tables(tmp_path: pathlib.Path) -> None:
    for method in ("gflownet", "gflownet+composition"):
        write_result(tmp_path, method, "B=1048576", "wc_0346/em_dash", 0, 1.0)
    rows = records_to_table.rows(records_to_table.result_files([tmp_path]))
    assert sorted(row["method"] for row in rows) == ["gfn", "gfn+composition"]
    assert paper_style.series("gfn+composition").label == "GFlowNet + composition"


def test_one_result_file_and_errors(tmp_path: pathlib.Path) -> None:
    write_result(tmp_path, "gdc", "B=4096", "wc_0346/em_dash", 0, 1.0)
    path = tmp_path / "gdc" / "B=4096" / "wc_0346__em_dash" / "rep0" / "result.json"
    assert records_to_table.result_files([path]) == [path]
    assert records_to_table.rule_of(path) == ""
    draws = tmp_path / "gdc" / "B=4096" / "wc_0346__em_dash" / "rep0" / "draws"
    with pytest.raises(paper_runs.TableError, match="draws does not exist"):
        records_to_table.result_files([draws])
    draws.mkdir()
    with pytest.raises(paper_runs.TableError, match="no result.json"):
        records_to_table.result_files([draws])
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"task_id": "wc_0346/em_dash"}))
    with pytest.raises(paper_runs.TableError, match="a result has no method, replicate"):
        records_to_table.rows([other])
    other.write_text("not JSON")
    with pytest.raises(paper_runs.TableError, match="is not a JSON file"):
        records_to_table.rows([other])


def result_of_the_package(task_id: str, replicate: int, **changes: object) -> dict[str, object]:
    """A result as `bench.evaluate(...).to_dict()` gives it, with the fields that the table
    reads and some that it does not."""
    result: dict[str, object] = {
        "task_id": task_id,
        "method": "my-method",
        "setting": None,
        "kind": "sampler",
        "sampler": "my-sampler",
        "seed": replicate,
        "replicate": replicate,
        "native_hit_rate": 0.4,
        "fallback": False,
        "capped_forward_kl": 14.5 + replicate,
        "reverse_kl": 18.0,
        "construction_inferences": 0.0,
        "details": {"text": "a line separator inside a text"},
    }
    return {**result, **changes}


def test_results_of_the_package_become_a_table_that_is_drawn_with_the_papers(
    tmp_path: pathlib.Path,
) -> None:
    """The file that a loop over `bench.evaluate` writes, one result per line, as
    `posteriorbench aggregate` reads it: `--add` draws its method next to the paper's."""
    results = [
        result_of_the_package(task_id, replicate)
        for task_id in ("wc_0346/em_dash", "wc_1853/goblin_mention")
        for replicate in (0, 1, 2)
    ]
    lines = tmp_path / "my_method.jsonl"
    lines.write_text("".join(json.dumps(result) + "\n" for result in results) + "\n")
    files = records_to_table.result_files([lines])
    assert files == [lines]
    rows = records_to_table.rows(files)
    assert len(rows) == 6
    assert {(row["rule"], row["method"], row["setting"]) for row in rows} == {
        ("", "my-method", "none")
    }
    table = tmp_path / "my_method.csv"
    paper_runs.write_table(table, records_to_table.COLUMNS, rows)
    runs = main_comparison.with_added(
        paper_runs.read_runs(main_comparison.RUNS), paper_runs.read_runs(table)
    )
    points = main_comparison.points(runs)
    (mine,) = [point for point in points if point["method"] == "my-method"]
    assert (mine["tasks"], mine["runs"], mine["setting"]) == (2, 6, "none")
    assert mine["capped_forward_kl"] == pytest.approx(15.5)
    assert len(points) == 214 + 1
    # A result that names no method is listed under its sampler's name.
    lines.write_text(json.dumps(result_of_the_package("wc_0346/em_dash", 0, method=None)) + "\n")
    assert [row["method"] for row in records_to_table.rows([lines])] == ["my-sampler"]
    lines.write_text("{}\nnot JSON\n")
    with pytest.raises(paper_runs.TableError, match="one JSON result per line"):
        records_to_table.rows([lines])
