"""The command-line tool: list-tasks, evaluate, aggregate."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import textwrap
from typing import Any
from unittest import mock

import pytest
import typer
from support.datasets import DOCUMENTED_TASK_ID, SMALL_TASK_ID
from typer.testing import CliRunner

import posteriorbench
from posteriorbench import cli, data, original_model, results

RUNNER = CliRunner()


def invoke(*arguments: str) -> Any:
    outcome = RUNNER.invoke(cli.app, list(arguments), env={"COLUMNS": "200"})
    assert outcome.exit_code == 0, f"{arguments}\n{outcome.output}\n{outcome.exception!r}"
    return outcome


def refused(*arguments: str) -> str:
    """The output of a command that must fail, as one line of words."""
    outcome = RUNNER.invoke(cli.app, list(arguments), env={"COLUMNS": "200"})
    assert outcome.exit_code != 0, f"{arguments}\n{outcome.output}"
    return " ".join(outcome.output.replace("│", " ").split())


def test_help_and_version() -> None:
    listed = invoke("--help").output
    commands = [command.name for command in cli.app.registered_commands]
    assert commands == ["list-tasks", "evaluate", "aggregate"]
    for name in commands:
        assert name is not None and name in listed
        invoke(name, "--help")
    assert invoke("--version").output.strip() == f"posteriorbench {posteriorbench.__version__}"
    citation = pathlib.Path(__file__).parent.parent / "CITATION.cff"
    if citation.exists():  # not beside the tests when they run against an installed wheel
        assert f'version: "{posteriorbench.__version__}"' in citation.read_text()
    # The choices are read from the command itself: the rendered help depends on the terminal
    # (a continuous-integration runner gets colour codes and another width).
    evaluate: Any = typer.main.get_command(cli.app)
    choices = {
        option.name: list(option.type.choices)
        for option in evaluate.commands["evaluate"].params
        if hasattr(option.type, "choices")
    }
    assert choices["sampler_kind"] == ["original", "instruction", "fallback", "custom"]
    assert choices["reference_logp"] == ["shipped", "recomputed", "both"]


def test_list_tasks(small_dataset: pathlib.Path) -> None:
    listed = invoke("list-tasks", "--data", str(small_dataset))  # the benchmark's tasks
    assert listed.output.splitlines() == [f"{SMALL_TASK_ID}\t600\t5.000e-01\tbenchmark_task"]
    scored = invoke("list-tasks", "--data", str(small_dataset), "--subset", "scored")
    assert scored.output == listed.output
    everything = invoke("list-tasks", "--data", str(small_dataset), "--subset", "all")
    assert [line.split("\t")[0] for line in everything.output.splitlines()] == [
        SMALL_TASK_ID,
        "wc_0001/profanity",
    ]
    assert everything.output.splitlines()[1].endswith("\t")  # an unscored pair has no flag
    refused("list-tasks", "--data", str(small_dataset), "--subset", "x")


def test_errors_are_reported_in_one_line(small_dataset: pathlib.Path) -> None:
    """A missing dataset, an unknown task and an unscored pair end with one line on stderr and
    exit status 1, without a traceback."""
    cases = (
        (["list-tasks", "--data", "./no-such-dataset"], "no such directory"),
        (["evaluate", "--task", "wc_0000/x", "--sampler", "fallback"], "unknown task"),
        (
            ["evaluate", "--data", str(small_dataset), "--task", "wc_0001/profanity"]
            + ["--sampler", "fallback"],
            "not a scored pair",
        ),
    )
    for arguments, message in cases:
        outcome = RUNNER.invoke(cli.app, arguments)
        assert outcome.exit_code == 1 and isinstance(outcome.exception, SystemExit)
        assert outcome.output.startswith("Error: ") and message in outcome.output
        assert len(outcome.output.strip().splitlines()) == 1


def result_record(task: str, replicate: int, forward: float, **fields: Any) -> dict[str, Any]:
    return {
        "task_id": task, "method": "m", "setting": "K=16", "kind": "sampler", "sampler": "s",
        "seed": replicate, "replicate": replicate, "draws": 1024, "hits": 500,
        "native_hit_rate": 0.5, "passed_threshold": True, "fallback": False,
        "capped_forward_kl": forward, "reverse_kl": 2.0, "cap": 50.0, "log_p_event": -3.0,
        "construction_inferences": 10.0, "budget_passes": 10.0, "inferences_per_response": 1.0,
        "cost_rule": "ledger", "ledger": None, "reference_logp": "shipped", "run": {}, **fields,
    }  # fmt: skip


def test_aggregate(tmp_path: pathlib.Path) -> None:
    paths = []
    for task, forward in (("x", 1.0), ("y", 3.0)):
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps(result_record(task, 0, forward)))
        paths.append(str(path))
    lines = tmp_path / "more.jsonl"
    lines.write_text(
        "".join(json.dumps(result_record(task, 0, 9.0, setting=None)) + "\n" for task in ("x", "y"))
    )
    summary = json.loads(invoke("aggregate", *paths, str(lines)).output)
    assert [(s["method"], s["setting"]) for s in summary] == [("m", "K=16"), ("m", None)]
    assert summary[0]["metrics"]["capped_forward_kl"]["mean"] == 2.0
    assert summary[1]["metrics"]["capped_forward_kl"]["mean"] == 9.0
    assert summary[0]["tasks"] == 2
    invoke("aggregate", *paths, "--expect-replicates", "0")
    refused("aggregate", *paths, "--expect-replicates", "0,1,2")
    refused("aggregate", *paths, "--expect-replicates", "zero")
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps(result_record("z", 0, 4.0, reference_logp="recomputed")))
    assert "different reference log p" in refused("aggregate", *paths, str(mixed))
    incomplete = tmp_path / "incomplete.json"
    record = result_record("x", 0, 4.0)
    del record["kind"]
    incomplete.write_text(json.dumps(record))
    assert "not a result record" in refused("aggregate", str(incomplete))


def test_aggregate_expects_tasks(small_dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """`--expect-tasks benchmark` / `scored` read the task ids from the data source."""
    covered = tmp_path / "covered.json"
    covered.write_text(json.dumps(result_record(SMALL_TASK_ID, 0, 1.0)))
    base = ["aggregate", "--data", str(small_dataset), "--expect-tasks"]
    for subset in ("benchmark", "scored", SMALL_TASK_ID):
        (summary,) = json.loads(invoke(*base, subset, str(covered)).output)
        assert summary["tasks"] == 1
    other = tmp_path / "other.json"
    other.write_text(json.dumps(result_record("wc_0001/profanity", 0, 1.0)))
    refused(*base, "benchmark", str(other))


@pytest.fixture()
def offline_model(tiny_model: Any) -> Any:
    """`OriginalModel.from_pretrained` returns the tiny model and records its arguments."""
    with mock.patch.object(
        original_model.OriginalModel, "from_pretrained", return_value=tiny_model
    ) as load:
        yield load


def test_evaluate_reference_samplers(
    tiny_dataset: pathlib.Path, tmp_path: pathlib.Path, offline_model: Any
) -> None:
    base = ["evaluate", "--data", str(tiny_dataset), "--task", DOCUMENTED_TASK_ID]
    done = invoke(*base, "--sampler", "original", "--reference-logp", "both", "--out",
                  str(tmp_path / "s.json"), "--out-recomputed", str(tmp_path / "r.json"),
                  "--model-revision", "w1", "--replicate", "2")  # fmt: skip
    # A result written to a file is summarized in one line on the terminal.
    assert f"{DOCUMENTED_TASK_ID} replicate 2: native hit rate" in done.output
    assert f"-> {tmp_path / 's.json'}" in done.output and f"-> {tmp_path / 'r.json'}" in done.output
    shipped = json.loads((tmp_path / "s.json").read_text())
    recomputed = json.loads((tmp_path / "r.json").read_text())
    assert (shipped["reference_logp"], recomputed["reference_logp"]) == ("shipped", "recomputed")
    assert shipped["hits"] == recomputed["hits"]
    assert shipped["reverse_kl"] == recomputed["reverse_kl"]
    assert (shipped["method"], shipped["setting"], shipped["replicate"], shipped["seed"]) == (
        "original-model",
        None,
        2,
        2,
    )
    assert shipped["construction_inferences"] == 0.0 and shipped["kind"] == "sampler"
    assert set(shipped["run"]) == {"threshold", "reverse_hit_limit", "seconds"}
    assert shipped["dataset"]["source"] == str(tiny_dataset)
    kwargs = offline_model.call_args.kwargs
    assert (kwargs["revision"], kwargs["dtype"]) == ("w1", "float32")
    instruction = json.loads(
        invoke(*base, "--sampler", "instruction", "--seed", "5", "--dtype", "bfloat16",
               "--cap", "20").output  # fmt: skip
    )
    assert instruction["method"] == "instruction" and instruction["sampler"] == "instruction"
    assert instruction["seed"] == 5 and instruction["sampler_seed"] == 2**31 + 5
    assert instruction["cap"] == 20.0 and instruction["reference_logp"] == "shipped"
    assert offline_model.call_args.kwargs["dtype"] == "bfloat16"
    results.EvaluationResult.from_dict(instruction)  # a complete record
    refused(*base, "--sampler", "original", "--reference-logp", "both")  # needs two outputs
    refused(*base, "--sampler", "original", "--out-recomputed", "r.json")
    refused(*base, "--sampler", "original", "--cap", "0")
    refused(*base, "--sampler", "other")


def test_evaluate_fallback_loads_no_model(
    tiny_dataset: pathlib.Path, offline_model: Any, tmp_path: pathlib.Path
) -> None:
    base = ["evaluate", "--data", str(tiny_dataset), "--task", DOCUMENTED_TASK_ID]
    written = invoke(*base, "--sampler", "fallback", "--out", str(tmp_path / "f.json")).output
    assert "replicate 0: native hit rate 1.000, capped forward KL" in written  # no threshold note
    result = json.loads(invoke(*base, "--sampler", "fallback").output)
    assert not offline_model.called
    assert result["fallback"] and result["inferences_per_response"] == 0.0
    assert result["construction_inferences"] == 0.0 and result["method"] == "fallback"
    assert (result["kind"], result["reference_logp"]) == ("fallback", None)
    assert result["dataset"] == {"source": str(tiny_dataset), "revision": None, "commit": None}
    refused(*base, "--sampler", "fallback", "--reference-logp", "both", "--out", "a.json",
            "--out-recomputed", "b.json")  # fmt: skip


def test_evaluate_reads_the_bundled_tables_by_default(offline_model: Any) -> None:
    """Without --data the command reads the tables shipped inside the package."""
    result = json.loads(
        invoke("evaluate", "--task", "wc_1853/goblin_mention", "--sampler", "fallback").output
    )
    assert not offline_model.called
    assert result["fallback"] and result["task_id"] == "wc_1853/goblin_mention"
    assert result["dataset"] == {
        "source": "bundled",
        "revision": data.DATASET_REVISION,
        "commit": None,
    }
    listed = invoke("list-tasks").output.splitlines()
    assert len(listed) == 21 and all(line.endswith("benchmark_task") for line in listed)
    assert len(invoke("list-tasks", "--subset", "scored").output.splitlines()) == 108
    assert len(invoke("list-tasks", "--subset", "all").output.splitlines()) == 140


FACTORY = textwrap.dedent(
    '''
    import posteriorbench as pb


    def build(task, model, benchmark):
        """Takes 8 corpus hits (charged) and returns the original model as the sampler."""
        ledger = pb.Ledger()
        benchmark.corpus(task, ledger).take_first(8)
        ledger.charge(pb.Category.THRESHOLD_TEST, inferences=164)
        return pb.original_model_sampler(model, task), ledger


    def broken(task, model, benchmark):
        return pb.original_model_sampler(model, task)
    '''
)


def test_evaluate_custom_factory(
    tiny_dataset: pathlib.Path,
    tmp_path: pathlib.Path,
    offline_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "my_method.py").write_text(FACTORY)
    monkeypatch.syspath_prepend(str(tmp_path))
    base = ["evaluate", "--data", str(tiny_dataset), "--task", DOCUMENTED_TASK_ID, "--sampler",
            "custom"]  # fmt: skip
    out = tmp_path / "custom.json"
    invoke(*base, "--factory", "my_method:build", "--method", "mine", "--setting", "K=8",
           "--out", str(out))  # fmt: skip
    result = results.EvaluationResult.from_dict(json.loads(out.read_text()))
    task = data.get_task(data.load_tasks(tiny_dataset), DOCUMENTED_TASK_ID)
    assert result.construction_inferences == pytest.approx(8 / task.probability + 164)
    assert (result.method, result.setting) == ("mine", "K=8")
    refused(*base, "--factory", "my_method:broken")
    # A missing or unloadable factory is refused before the model is loaded.
    offline_model.reset_mock()
    refused(*base)
    refused(*base, "--factory", "my_method:nothing")
    refused(*base, "--factory", "no_such_module:build")
    refused("evaluate", "--task", DOCUMENTED_TASK_ID, "--sampler", "original", "--factory",
            "my_method:build")  # fmt: skip
    assert not offline_model.called
    # The tables and the hit files can come from different places (--hits).
    tables = tmp_path / "tables"
    for file in (data.TASKS_FILE, data.EVALUATION_FILE):
        (tables / file).parent.mkdir(parents=True)
        shutil.copyfile(tiny_dataset / file, tables / file)
    split = ["evaluate", "--data", str(tables), "--task", DOCUMENTED_TASK_ID, "--sampler", "custom",
             "--factory", "my_method:build"]  # fmt: skip
    refused(*split)  # no hit files next to the tables
    separate = json.loads(invoke(*split, "--hits", str(tiny_dataset)).output)
    assert separate["construction_inferences"] == result.construction_inferences


def test_module_entry_point(tiny_dataset: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """`python -m posteriorbench` runs the command-line tool."""
    copy = tmp_path / "copy"
    shutil.copytree(tiny_dataset, copy)
    outcome = subprocess.run(
        [sys.executable, "-m", "posteriorbench", "list-tasks", "--data", str(copy)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    assert outcome.stdout.startswith(DOCUMENTED_TASK_ID)
