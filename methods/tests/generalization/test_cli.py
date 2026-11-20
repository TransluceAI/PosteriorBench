"""The commands: listing the six tasks, and on the toy tasks the evaluation of a reference
sampler."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from posteriorbench import results
from typer.testing import CliRunner

from posteriorbench_methods.generalization import cli, judge, task_set, tasks

RUNNER = CliRunner()


def invoke(*arguments: str) -> Any:
    outcome = RUNNER.invoke(cli.app, list(arguments))
    assert outcome.exit_code == 0, f"{arguments}\n{outcome.output}\n{outcome.exception!r}"
    return outcome


@pytest.fixture()
def toy_commands(
    toy_tasks: Any, tiny_original_model: Any, helpers: Any, monkeypatch: pytest.MonkeyPatch
) -> list[Any]:
    """The commands on the two toy tasks: the tiny model is what loading returns, and the two
    judge models of the command line are the stand-ins. Returns the calls of the model loader."""
    loads: list[Any] = []

    def load(task: Any, **options: Any) -> Any:
        loads.append((task.task_id, options))
        return tiny_original_model

    monkeypatch.setattr(tasks, "all_tasks", lambda: (toy_tasks.by_rule, toy_tasks.by_rubric))
    monkeypatch.setattr(task_set, "load_original_model", load)
    monkeypatch.setattr(judge, "rubric_judge_model", lambda *a: helpers.judge_model())
    monkeypatch.setattr(judge, "review_model", lambda *a: helpers.review_model())
    return loads


def test_list_tasks_says_what_each_task_needs() -> None:
    lines = invoke("list-tasks").output.splitlines()
    assert [line.split("\t")[0] for line in lines] == [task.task_id for task in tasks.all_tasks()]
    first, last = lines[0].split("\t"), lines[-1].split("\t")
    assert first[1:7] == ["Qwen3.5-0.8B", "2048", "rubric", "760", "2.235e-04", "64"]
    assert first[7] == "needs: Qwen3.5-0.8B, a rubric judge and a reviewing model"
    assert last[1:] == [
        "Gemma-4-31B-it",
        "50",
        "rule",
        "279973",
        "2.800e-02",
        "250",
        "needs: Gemma-4-31B-it",
    ]
    commands = [command.name for command in cli.app.registered_commands]
    assert commands == ["list-tasks", "evaluate"]
    for name in commands:
        assert name is not None
        invoke(name, "--help")


def test_evaluate(toy_tasks: Any, toy_commands: list[Any], tmp_path: pathlib.Path) -> None:
    data = ["--data", str(toy_tasks.root)]
    rule, rubric = toy_tasks.by_rule.task_id, toy_tasks.by_rubric.task_id
    fallback = json.loads(invoke("evaluate", "--task", rule, *data, "--sampler", "fallback").output)
    assert fallback["fallback"] and fallback["kind"] == "fallback" and not toy_commands
    out = tmp_path / "result.json"
    done = invoke("evaluate", "--task", rule, *data, "--sampler", "original", "--out", str(out),
                  "--replicate", "2", "--dtype", "bfloat16", "--batch-size", "4")  # fmt: skip
    assert f"{rule} replicate 2: native hit rate" in done.output
    result = results.EvaluationResult.from_dict(json.loads(out.read_text()))
    assert (result.method, result.seed, result.passed_threshold) == ("original-model", 2, True)
    assert result.dataset is not None and result.dataset["source"] == str(toy_tasks.root)
    assert toy_commands[-1][0] == rule
    assert (toy_commands[-1][1]["dtype"], toy_commands[-1][1]["batch_size"]) == ("bfloat16", 4)
    # A task with a rubric: without the two judge models the command stops before the model
    # is loaded; with them the draws are labeled by both.
    loads = len(toy_commands)
    refused = RUNNER.invoke(cli.app, ["evaluate", "--task", rubric, *data, "--sampler", "original"])
    assert refused.exit_code == 1 and refused.output.startswith("Error: ")
    assert "judged by a language model" in refused.output and len(toy_commands) == loads
    judges = ["--judge-url", "http://judge/v1", "--judge-model", "j", "--review-url",
              "http://review/v1", "--review-model", "r", "--review-key-env", "KEY"]  # fmt: skip
    labels = tmp_path / "labels.jsonl"
    judged = json.loads(
        invoke("evaluate", "--task", rubric, *data, "--sampler", "original", *judges,
               "--labels", str(labels), "--reference-logp", "recomputed").output  # fmt: skip
    )
    assert judged["reference_logp"] == "recomputed" and judged["hits"] > 0
    assert judged["scorer"]["judge"]["judged_by"] == "rubric" and labels.exists()
    # A toy behavior has no Instruction text: one line, no traceback.
    missing = RUNNER.invoke(
        cli.app, ["evaluate", "--task", rule, *data, "--sampler", "instruction"]
    )
    assert missing.exit_code == 1 and "no instruction for behavior" in missing.output
    unknown = RUNNER.invoke(
        cli.app, ["evaluate", "--task", "x/y/z", *data, "--sampler", "fallback"]
    )
    assert unknown.exit_code == 1 and "unknown task" in unknown.output
