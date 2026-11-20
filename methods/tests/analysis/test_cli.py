"""The commands of the analyses, on the records of small runs."""

from __future__ import annotations

import json
import pathlib

import pytest
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.analysis import gpu_times, records

RUNNER = CliRunner()
TASK_ID = "wc_1853/goblin_mention"


def test_the_root_command_has_the_commands_of_the_analyses() -> None:
    result = RUNNER.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("compute-per-inference", "time-operations", "cap-sensitivity", "estimate-ps"):
        assert command in result.output
        assert RUNNER.invoke(cli.app, [command, "--help"]).exit_code == 0


@pytest.fixture(scope="module")
def model_and_data(
    hybrid_model_dir: pathlib.Path, analysis_inputs: dict[str, pathlib.Path]
) -> list[str]:
    """The options that name the tiny model with the paper's layers and the tiny dataset."""
    return ["--model", str(hybrid_model_dir), "--data", str(analysis_inputs["data"])]


def test_compute_per_inference_prints_the_methods_and_their_points(
    finished_root: pathlib.Path,
    model_and_data: list[str],
    tiny_session: gpu_times.Session,
    tmp_path: pathlib.Path,
) -> None:
    gpu_times.write(tmp_path / "times.json", [tiny_session])
    arguments = ["compute-per-inference", str(finished_root), *model_and_data]
    result = RUNNER.invoke(cli.app, [*arguments, "--times", str(tmp_path / "times.json")])
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert sorted(printed) == ["methods", "not_covered", "points"]
    methods = {entry["method"]: entry for entry in printed["methods"]}
    assert sorted(methods) == ["ctl", "gdc", "gflownet", "kl-dpg", "replay", "sft"]
    assert methods["replay"]["flops_per_inference"] == 1
    assert all("gpu_time_per_inference" in entry for entry in methods.values())
    assert printed["not_covered"]["sft+composition"] == 2
    # Without the times, and with every run, into a file.
    out = tmp_path / "compute.json"
    result = RUNNER.invoke(cli.app, [*arguments, "--flops-only", "--runs", "--out", str(out)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert len(written["runs"]) == sum(entry["runs"] for entry in written["methods"])
    assert "gpu_time_per_inference" not in written["methods"][0]


def test_the_records_of_a_command_are_roots_directories_or_results(
    finished_root: pathlib.Path, finished: list[records.Record]
) -> None:
    one = finished[0].directory
    assert len(records.find([finished_root])) == len(finished)
    assert [r.directory for r in records.find([one, one / "result.json"])] == [one, one]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["compute-per-inference", "{empty}"], "Error: {empty} holds no record"),
        (["cap-sensitivity", "{empty}/nothing"], "Error: {empty}/nothing does not exist"),
        (
            ["compute-per-inference", "{root}", "{root}", "--flops-only"],
            "Error: ctl at u=2: wc_1853/goblin_mention has more than one record of replicate 0",
        ),
        (
            ["compute-per-inference", "{root}", "--times", "{empty}/times.json"],
            "Error: {empty}/times.json does not exist",
        ),
    ],
)
def test_a_mistake_is_reported_as_one_line(
    arguments: list[str],
    message: str,
    finished_root: pathlib.Path,
    model_and_data: list[str],
    tmp_path: pathlib.Path,
) -> None:
    names = {"empty": str(tmp_path), "root": str(finished_root)}
    filled = [argument.format(**names) for argument in arguments]
    if filled[0] == "compute-per-inference":
        filled += model_and_data
    result = RUNNER.invoke(cli.app, filled)
    assert result.exit_code == 1
    assert result.output.startswith(message.format(**names)) and result.output.count("\n") == 1


def test_cap_sensitivity_prints_the_scores_at_every_cap(finished_root: pathlib.Path) -> None:
    result = RUNNER.invoke(cli.app, ["cap-sensitivity", str(finished_root)])
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed["caps"] == [10, 20, 50, 100, 200]
    assert [entry["cap"] for entry in printed["share_at_cap"]] == printed["caps"]
    assert {point["cap"] for point in printed["points"]} == set(printed["caps"])
    assert "runs" not in printed
    two = RUNNER.invoke(
        cli.app, ["cap-sensitivity", str(finished_root), "--caps", "5,50", "--runs"]
    )
    assert two.exit_code == 0, two.output
    assert json.loads(two.stdout)["caps"] == [5, 50] and "runs" in json.loads(two.stdout)
    for caps in ("0,50", "ten", ""):
        bad = RUNNER.invoke(cli.app, ["cap-sensitivity", str(finished_root), "--caps", caps])
        assert bad.exit_code == 2


def test_time_operations_writes_a_times_file(
    model_and_data: list[str], tmp_path: pathlib.Path
) -> None:
    """The command on the tiny model with the paper's layers: its file prices every method."""
    out = tmp_path / "times.json"
    arguments = ["time-operations", "--out", str(out), "--task", TASK_ID, *model_and_data]
    arguments += ["--device", "cpu", "--repetitions", "1", "--warmup", "0"]
    result = RUNNER.invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    (session,) = gpu_times.read(out).sessions
    assert session.has(gpu_times.every_key())
    conditions = session.conditions
    assert (conditions["task_id"], conditions["response_tokens"]) == (TASK_ID, 50)
    assert conditions["dtype"] == "float32" and conditions["device"] == "cpu"
