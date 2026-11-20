"""The command `prompt write-candidates`, with a stand-in for the writer's server on this
machine: from no file to a pool file to a run."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pyarrow.parquet as pq
import pytest
from posteriorbench import benchmark
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.common import run_context
from posteriorbench_methods.prompt import candidate_writing, pool, run, selection, settings

RUNNER = CliRunner()
TASK_ID = "wc_1853/goblin_mention"


def write_candidates(
    server: Any, bench: benchmark.Benchmark, pool_file: pathlib.Path, *arguments: str
) -> Any:
    """Invoke the command on the tiny task with the stand-in server as the writer and four
    scenarios per call."""
    return RUNNER.invoke(
        cli.app,
        [
            "prompt",
            "write-candidates",
            "--task",
            TASK_ID,
            "--pool-file",
            str(pool_file),
            "--writer-url",
            server.url,
            "--writer-model",
            "writer-model",
            "--data",
            str(bench.source),
            "--set",
            "scenarios_per_call=4",
            "--set",
            "concurrency=3",
            *arguments,
        ],
    )


def reports(result: Any) -> list[dict[str, Any]]:
    """The reports that the command printed. Its progress goes to the log."""
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_the_command_writes_a_pool_file_that_a_run_reads(
    writer_server: Any,
    bench: benchmark.Benchmark,
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WRITER_API_KEY", "a-key-of-the-test")
    pool_file = tmp_path / "candidate_prompts.parquet"
    (report,) = reports(write_candidates(writer_server, bench, pool_file, "--candidates", "6"))
    assert (report["task_id"], report["candidates"], report["batch"]) == (TASK_ID, 6, 0)
    # One explanation, two calls of four scenarios, a system prompt for the first six.
    assert report["writer_calls"] == len(writer_server.requests) == 1 + 2 + 6
    assert all(
        headers["authorization"] == "Bearer a-key-of-the-test"
        and body["model"] == "writer-model"
        and body["temperature"] == 1.0
        for headers, body in writer_server.requests
    )
    assert sorted({body["max_tokens"] for _, body in writer_server.requests}) == [4000, 32768]
    explanations = candidate_writing.read_explanations(
        tmp_path / "candidate_prompts.explanations.json"
    )
    assert sorted(explanations) == ["goblin_mention"]

    # The pool file is the method data of a run.
    candidates = pool.load_candidates(pool_file, TASK_ID)
    assert len(candidates) == 6 and len({c.system_prompt for c in candidates}) == 6
    out = tmp_path / "runs"
    summaries = run.run(
        make_context(out, method_data=pool_file, compose=False),
        settings.Settings(methods=("prompt-hitrate",), pool_sizes=(2, 6)),
    )
    assert [(summary.method, summary.setting) for summary in summaries] == [
        ("prompt-hitrate", "K=2"),
        ("prompt-hitrate", "K=6"),
    ]
    assert all((out / summary.record / "result.json").exists() for summary in summaries)


def test_a_second_call_writes_what_is_lacking_as_a_new_batch(
    writer_server: Any,
    bench: benchmark.Benchmark,
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
) -> None:
    pool_file = tmp_path / "candidate_prompts.parquet"
    reports(write_candidates(writer_server, bench, pool_file, "--candidates", "5"))
    context = make_context(tmp_path / "runs", method_data=pool_file)
    before = selection.ordered_candidates(context, 5)
    calls = len(writer_server.requests)

    # The task has five candidates: no call. Nine are asked for: four are written.
    (same,) = reports(write_candidates(writer_server, bench, pool_file, "--candidates", "5"))
    assert (same["asked"], same["writer_calls"]) == (0, 0) and len(writer_server.requests) == calls
    (more,) = reports(
        write_candidates(writer_server, bench, pool_file, "--candidates", "9", "--new-batch")
    )
    assert (more["candidates_before"], more["asked"], more["candidates"]) == (5, 4, 9)
    assert (more["batch"], more["explanation_calls"]) == (1, 0)
    assert len(writer_server.requests) == calls + 1 + 4
    table = pq.read_table(pool_file).sort_by("candidate_index")
    assert table["batch"].to_pylist() == [0] * 5 + [1] * 4
    # The order of a replicate keeps the first batch where it was.
    assert selection.ordered_candidates(context, 9)[:5] == before


def test_a_setting_of_the_writing_is_changed_with_set(
    writer_server: Any, bench: benchmark.Benchmark, tmp_path: pathlib.Path
) -> None:
    pool_file = tmp_path / "candidate_prompts.parquet"
    arguments = ("--candidates", "2", "--set", "temperature=0.5", "--set", "scenario_tokens=99")
    reports(write_candidates(writer_server, bench, pool_file, *arguments))
    assert {body["temperature"] for _, body in writer_server.requests} == {0.5}
    assert sorted({body["max_tokens"] for _, body in writer_server.requests}) == [99, 4000]
    refused = write_candidates(writer_server, bench, pool_file, "--candidates", "2", "--set", "x=1")
    assert refused.exit_code == 2 and "unknown setting 'x'" in refused.output


@pytest.mark.parametrize(
    ("task", "status", "message"),
    [
        ("wc_9999/nope", 200, "Error: unknown task 'wc_9999/nope'"),
        (TASK_ID, 401, "Error: the writer at "),
        (TASK_ID, 500, "Error: the writer gave no explanation"),
    ],
)
def test_a_mistake_is_reported_as_one_line(
    writer_server: Any,
    bench: benchmark.Benchmark,
    tmp_path: pathlib.Path,
    task: str,
    status: int,
    message: str,
) -> None:
    writer_server.respond = lambda body, n: (status, {"error": "refused"})
    pool_file = tmp_path / "candidate_prompts.parquet"
    arguments = ["prompt", "write-candidates", "--task", task, "--pool-file", str(pool_file)]
    arguments += ["--candidates", "3", "--writer-url", writer_server.url, "--writer-model", "m"]
    arguments += ["--data", str(bench.source), "--set", "attempts=1"]
    result = RUNNER.invoke(cli.app, arguments)
    assert result.exit_code == 1
    assert result.stderr.startswith(message)
    assert not pool_file.exists()


def test_the_candidates_of_a_generalization_task_are_written_for_its_model_and_behavior(
    writer_server: Any, bench: benchmark.Benchmark, tmp_path: pathlib.Path
) -> None:
    """A task of the generalization section is named by its id. The writer is told the task's
    original model, and for a task that a language model judges the rubric of its behavior."""
    pool_file = tmp_path / "candidates.parquet"
    tasks = ["gemma-4-31b-it/wc_0670/em_dash", "qwen3.5-0.8b/wc_0670/physical_embodiment"]
    result = write_candidates(
        writer_server, bench, pool_file, "--candidates", "4", "--task", tasks[0], "--task", tasks[1]
    )
    # The task of the command's own arguments, a task of the package, is written first.
    assert [report["task_id"] for report in reports(result)] == [TASK_ID, *tasks]
    for task_id in tasks:
        assert len(pool.load_candidates(pool_file, task_id)) == 4
    system_messages = [body["messages"][0]["content"] for _, body in writer_server.requests]
    user_messages = [body["messages"][1]["content"] for _, body in writer_server.requests]
    assert any(text.endswith("Gemma-4-31B (a 31B chat model).") for text in system_messages)
    assert any("for the target model to follow" in text for text in user_messages)
    assert any("evaluated for false-physical-embodiment behavior" in text for text in user_messages)
    assert any("(as defined by the rubric above)" in text for text in user_messages)
    # One explanation for each behavior, by the behavior of the task.
    explained = json.loads(candidate_writing.explanations_file(pool_file).read_text())
    assert sorted(explained) == ["em_dash", "goblin_mention", "physical_embodiment"]
    unknown = write_candidates(
        writer_server, bench, pool_file, "--candidates", "4", "--task", "tiny/wc_0670/em_dash"
    )
    assert unknown.exit_code == 1 and unknown.output.startswith("Error: unknown task")
