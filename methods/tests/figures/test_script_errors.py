"""The figure and table scripts report a file or a table that they cannot use as one line on
stderr and exit with status 1, as the command line of the methods does."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from collections.abc import Callable

import estimating_ps
import larger_models
import llm_judged_behaviors
import main_comparison
import paper_runs
import prompt_mixtures
import pytest
import records_to_table
import sample_responses
import selection_signals
import table_checkpoint_rules
import table_compute_per_inference
import table_threshold_1pct
import training_by_probability
import training_free_by_probability
import typer
import typer.testing

FIGURES = paper_runs.DATA_DIR.parent
# A table of runs of a method that no figure of the paper has, on one task.
OTHER_RUNS = (
    "rule,method,setting,task_id,replicate,construction_inferences,capped_forward_kl,"
    "reverse_kl,native_hit_rate,fallback\n"
    ",my-method,none,wc_0346/em_dash,0,5000,3.0,4.0,0.5,false\n"
)


@pytest.fixture()
def files(tmp_path: pathlib.Path) -> dict[str, str]:
    """The files that the cases name: a table of other runs, a table without the columns of a
    table of runs, a JSON file that is no analysis, and a path that does not exist."""
    (tmp_path / "other.csv").write_text(OTHER_RUNS)
    (tmp_path / "no_columns.csv").write_text("method,value\nsft,1\n")
    (tmp_path / "other.json").write_text(json.dumps({"points": []}))
    (tmp_path / "text.json").write_text("not JSON")
    return {
        "OTHER": str(tmp_path / "other.csv"),
        "NO_COLUMNS": str(tmp_path / "no_columns.csv"),
        "NO_ANALYSIS": str(tmp_path / "other.json"),
        "TEXT": str(tmp_path / "text.json"),
        "ABSENT": str(tmp_path / "absent"),
        "OUT": str(tmp_path / "out.png"),
    }


@pytest.mark.parametrize(
    ("main", "arguments", "message"),
    [
        (main_comparison.main, ["--runs", "ABSENT"], "Error: ABSENT does not exist"),
        (main_comparison.main, ["--runs", "NO_COLUMNS"], "Error: the table has no column setting"),
        (main_comparison.main, ["--add", "ABSENT"], "Error: ABSENT does not exist"),
        (main_comparison.main, ["--add", "NO_COLUMNS"], "Error: the table has no column"),
        (larger_models.main, ["--runs", "OTHER"], "Error: the table has no run of prompt-ebjc"),
        (llm_judged_behaviors.main, ["--runs", "NO_COLUMNS"], "Error: the table has no column"),
        (selection_signals.main, ["--runs", "OTHER"], "Error: the table has no run of prompt-"),
        (prompt_mixtures.main, ["--runs", "OTHER"], "Error: the table has no run of mixture-"),
        (training_by_probability.main, ["--runs", "OTHER"], "Error: the table has no run of sft"),
        (
            training_free_by_probability.main,
            ["--runs", "OTHER"],
            "Error: the table has no run of prompt-ebjc at K=1024",
        ),
        (estimating_ps.main, ["--runs", "OTHER"], "Error: the table has no column estimate"),
        (estimating_ps.main, ["--analysis", "NO_ANALYSIS"], "Error: NO_ANALYSIS: run estimate-ps"),
        (estimating_ps.main, ["--analysis", "TEXT"], "Error: TEXT is not a JSON file"),
        (
            table_threshold_1pct.main,
            ["--runs", "OTHER"],
            "Error: the table has no run of sft at B=524288, gfn at B=1048576",
        ),
        (
            table_checkpoint_rules.main,
            ["--runs", "OTHER"],
            "Error: the table has no method with runs of `run` and of `reselect --rule`",
        ),
        (
            table_compute_per_inference.main,
            ["--analysis", "NO_ANALYSIS"],
            "Error: NO_ANALYSIS is not the JSON of `compute-per-inference`",
        ),
        (table_compute_per_inference.main, ["--analysis", "ABSENT"], "Error: ABSENT does not"),
        (records_to_table.main, ["ABSENT", "--out", "OUT"], "Error: ABSENT does not exist"),
        (
            records_to_table.main,
            ["NO_ANALYSIS", "--out", "OUT"],
            "Error: NO_ANALYSIS: a result has",
        ),
        (sample_responses.main, ["--record", "ABSENT"], "Error: ABSENT: no draws/evaluation"),
    ],
)
def test_a_script_reports_what_it_cannot_use_as_one_line(
    main: Callable[..., None], arguments: list[str], message: str, files: dict[str, str]
) -> None:
    app = typer.Typer()
    app.command()(paper_runs.one_line_errors(main))
    named = [files.get(argument, argument) for argument in arguments]
    if main in DRAW:
        named += ["--out", files["OUT"]]
    result = typer.testing.CliRunner().invoke(app, named)
    for name, path in files.items():
        message = message.replace(name, path)
    assert result.exit_code == 1, result.output
    assert result.output.startswith(message), result.output
    assert result.output.count("\n") == 1
    assert not pathlib.Path(files["OUT"]).exists()


# The scripts that draw a figure, and so take --out.
DRAW = (
    main_comparison.main,
    larger_models.main,
    llm_judged_behaviors.main,
    selection_signals.main,
    prompt_mixtures.main,
    training_by_probability.main,
    training_free_by_probability.main,
    estimating_ps.main,
)
# The scripts that read a table, an analysis or a record of the reader's.
READ_A_FILE_OF_THE_READER = {
    "estimating_ps.py",
    "larger_models.py",
    "llm_judged_behaviors.py",
    "main_comparison.py",
    "prompt_mixtures.py",
    "records_to_table.py",
    "sample_responses.py",
    "selection_signals.py",
    "table_checkpoint_rules.py",
    "table_compute_per_inference.py",
    "table_threshold_1pct.py",
    "training_by_probability.py",
    "training_free_by_probability.py",
}


def test_the_scripts_that_read_a_file_of_the_reader_run_with_the_one_line_report() -> None:
    """A script that takes a table, an analysis or a record wraps its command; the others read
    only the package's data and the paper's."""
    scripts = {
        path.name: path.read_text()
        for path in FIGURES.glob("*.py")
        if "typer.run(" in path.read_text()
    }
    assert READ_A_FILE_OF_THE_READER <= set(scripts)
    wrapped = {
        name
        for name, source in scripts.items()
        if "typer.run(paper_runs.one_line_errors(main))" in source
    }
    assert wrapped == READ_A_FILE_OF_THE_READER
    options = ("runs:", "add:", "analysis:", "record:", "records:")
    takes_a_file = {
        name for name, source in scripts.items() if any(option in source for option in options)
    }
    assert takes_a_file == READ_A_FILE_OF_THE_READER


def test_a_script_run_as_a_command_exits_with_status_1(files: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, str(FIGURES / "table_threshold_1pct.py"), "--runs", files["OTHER"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr.startswith("Error: the table has no run of sft at B=524288")
    assert result.stderr.count("\n") == 1
