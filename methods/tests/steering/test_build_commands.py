"""The commands that write Steering's method data, `find-refusal-direction` and `build-vectors`:
from an empty directory to the vectors of a behavior and to a `run` that reads them. The model is
the tiny one, loaded from a directory; the checkout and the judge's server are stand-ins."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.common import run_context, run_record
from posteriorbench_methods.generalization import models
from posteriorbench_methods.steering import cli as steering_cli
from posteriorbench_methods.steering import construction_prompts, vectors

TASK_ID = "wc_1853/goblin_mention"
BEHAVIOR = "goblin_mention"
RUNNER = CliRunner()


@pytest.fixture(scope="module")
def inputs(
    tiny_model: Any,
    tiny_tokenizer: Any,
    make_tiny_original_model: Callable[..., Any],
    make_tiny_dataset: Callable[..., pathlib.Path],
    behavior_prompts: construction_prompts.ConstructionPrompts,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, pathlib.Path]:
    """The model directory, a dataset whose behavior is "the response contains the letter a",
    and the file of construction prompts."""
    root = tmp_path_factory.mktemp("inputs")
    tiny_model.save_pretrained(root / "model")
    tiny_tokenizer.save_pretrained(root / "model")
    dataset = make_tiny_dataset(make_tiny_original_model(max_response_tokens=50), draws=1500)
    prompts = root / "goblin_mention.json"
    prompts.write_text(json.dumps(behavior_prompts.to_json()))
    return {"model": root / "model", "data": dataset, "prompts": prompts}


def _build_arguments(
    inputs: dict[str, pathlib.Path], directory: pathlib.Path, *more: str
) -> list[str]:
    return [
        "steering", "build-vectors", "--behavior", BEHAVIOR, "--prompts", str(inputs["prompts"]),
        "--method-data", str(directory), "--judge-model", "stand-in",
        "--model", str(inputs["model"]), "--data", str(inputs["data"]), "--device", "cpu",
        "--set", "responses_per_prompt=4", "--set", "response_limit=12", *more,
    ]  # fmt: skip


def test_from_nothing_to_the_vectors_of_a_behavior_to_a_run(
    inputs: dict[str, pathlib.Path],
    lmtailrisk_checkout: pathlib.Path,
    serve_chat: Callable[..., Any],
    stand_in_judge: Callable[[str], dict[str, Any]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "steering_vectors"
    model = ["--model", str(inputs["model"]), "--device", "cpu"]
    checkout = ["--lmtailrisk-checkout", str(lmtailrisk_checkout)]
    # 1. The refusal direction of the model.
    found = RUNNER.invoke(
        cli.app,
        ["steering", "find-refusal-direction", "--method-data", str(directory), *checkout, *model],
    )
    assert found.exit_code == 0, found.output
    direction = json.loads(found.stdout)
    assert direction["forward_passes"] > 323 and direction["model"] == str(inputs["model"])
    # 2. The vectors of the behavior. The judge's key is read from the environment.
    server = serve_chat(lambda body: stand_in_judge(body["messages"][0]["content"]))
    monkeypatch.setenv("JUDGE_KEY", "secret")
    built = RUNNER.invoke(
        cli.app,
        _build_arguments(
            inputs, directory, "--judge-url", server.url, "--judge-key-env", "JUDGE_KEY",
            *checkout,
        ),  # fmt: skip
    )
    assert built.exit_code == 0, built.output
    summary = json.loads(built.stdout)
    assert summary["behavior"] == BEHAVIOR and summary["judge"] == {
        "model": "stand-in",
        "calls": 72,
    }
    assert len(server.requests) == 72
    assert {request["authorization"] for request in server.requests} == {"Bearer secret"}
    assert {request["body"]["model"] for request in server.requests} == {"stand-in"}
    assert all(entry["built"] for entry in summary["vectors"].values())
    loaded = vectors.load(directory, BEHAVIOR, layers=(0, 1))
    assert list(loaded.tensors) == ["with_ablation", "without_ablation"]
    assert loaded.cost.refusal_direction_passes == direction["forward_passes"]
    assert loaded.cost.construction_responses == 72
    assert loaded.cost.to_json() == summary["construction_cost"]
    # 3. A run reads the directory as its method data and is charged what the vectors cost.
    out = tmp_path / "runs"
    ran = RUNNER.invoke(
        cli.app,
        [
            "steering", "run", "--task", TASK_ID, "--out", str(out), "--no-compose",
            "--method-data", str(directory), "--data", str(inputs["data"]), *model,
            "--evaluation-draws", "64",
            "--set", "layers=0,1", "--set", "strengths=1.5", "--set", "selection_draws=32",
        ],  # fmt: skip
    )
    assert ran.exit_code == 0, ran.output
    record = run_record.run_dir(out, "steering", "none", TASK_ID, 0)
    recorded = run_record.read_json(record / "construction.json")
    assert recorded["details"]["construction_cost"] == summary["construction_cost"]
    assert len(recorded["details"]["candidates"]) == 5
    charged = [charge["inferences"] for charge in recorded["ledger"]["charges"][:3]]
    kept = sum(summary["construction_cost"]["kept_responses"].values())
    assert charged == [direction["forward_passes"], 72, kept]
    # The ledger of the construction holds the same three charges.
    assert [c["inferences"] for c in summary["ledger"]["charges"][:3]] == charged


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (["--prompts", "absent.json"], "Error: absent.json does not exist"),
        (["--behavior", "goblin"], "Error: goblin_mention.json holds the construction prompts of"),
        (["--data", "absent"], "Error: "),
        ([], "Error: the coherence judge needs its prompt"),
        (["--coherence-prompt", "PROMPT"], "Error: DIRECTORY holds no refusal direction"),
        (
            ["--coherence-prompt", "PROMPT", "--judge-url", "localhost:8000"],
            "Error: the judge's address is an http or https URL",
        ),
    ],
)
def test_a_mistake_is_reported_as_one_line_before_the_model_is_loaded(
    change: list[str],
    message: str,
    inputs: dict[str, pathlib.Path],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_model(*_: Any) -> Any:
        raise AssertionError("the model is not loaded")

    monkeypatch.setattr(steering_cli, "_load_original_model", no_model)
    monkeypatch.chdir(inputs["prompts"].parent)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("[{question}] [{answer}]")
    directory = tmp_path / "steering_vectors"
    change = [str(prompt) if part == "PROMPT" else part for part in change]
    arguments = _build_arguments(inputs, directory, "--judge-url", "http://localhost:1/v1")
    arguments[arguments.index("--prompts") + 1] = "goblin_mention.json"
    result = RUNNER.invoke(cli.app, [*arguments, *change])
    assert result.exit_code == 1, result.output
    expected = message.replace("DIRECTORY", str(directory))
    assert result.output.startswith(expected) and result.output.count("\n") == 1
    assert not directory.exists()


def test_the_commands_refuse_what_they_cannot_use(
    inputs: dict[str, pathlib.Path], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: Any) -> Any:
        raise AssertionError("the model is not loaded")

    monkeypatch.setattr(steering_cli, "_load_original_model", no_model)
    # A folder that is not the checkout.
    found = RUNNER.invoke(
        cli.app,
        [
            "steering", "find-refusal-direction", "--method-data", str(tmp_path / "vectors"),
            "--lmtailrisk-checkout", str(tmp_path),
        ],  # fmt: skip
    )
    assert found.exit_code == 1 and found.output.startswith(f"Error: {tmp_path} is not a checkout")
    assert not (tmp_path / "vectors").exists()
    # A setting that the construction does not have, or a value outside its range.
    arguments = _build_arguments(inputs, tmp_path / "vectors", "--judge-url", "http://localhost:1")
    for setting in ("strengths=2", "vectors=ablated", "ablation_intensity=2"):
        refused = RUNNER.invoke(cli.app, [*arguments, "--set", setting])
        assert refused.exit_code == 2, refused.output
    # Both commands say what they need.
    for command in ("find-refusal-direction", "build-vectors"):
        listed = RUNNER.invoke(cli.app, ["steering", command, "--help"])
        assert listed.exit_code == 0 and "--method-data" in listed.output
        assert "--lmtailrisk-checkout" in listed.output


def test_the_vectors_of_a_generalization_task_are_built_on_its_original_model(
    inputs: dict[str, pathlib.Path],
    tiny_model: Any,
    tiny_tokenizer: Any,
    serve_chat: Callable[..., Any],
    stand_in_judge: Callable[[str], dict[str, Any]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--task` names a task of the generalization section: the command loads the task's
    original model and labels the responses by the task's rule. Without a refusal direction
    in the directory, the vector without ablation is built alone and nothing is charged for a
    direction."""
    task_id = "gemma-4-31b-it/wc_0670/em_dash"
    loaded: list[tuple[str, str | None]] = []

    def load(task: Any, options: run_context.RunOptions, device: str) -> Any:
        loaded.append((task.task_id, options.model))
        return models.TaskOriginalModel(
            tiny_model,
            tiny_tokenizer,
            batch_size=16,
            max_response_tokens=task.max_response_tokens,
            name=task.model.name,
        )

    monkeypatch.setattr(run_context, "load_task_original_model", load)
    prompts = tmp_path / "em_dash.json"
    prompts.write_text(
        json.dumps(
            {
                "behavior": "em_dash",
                "name": "em dash",
                "instruction_pairs": [{"positive": "use dashes", "negative": "use no dash"}],
                "questions": ["how do i make cheese", "why is the sky blue"],
            }
        )
    )
    prompt = tmp_path / "coherence.txt"
    prompt.write_text("[{question}] [{answer}]")
    server = serve_chat(lambda body: stand_in_judge(body["messages"][0]["content"]))
    directory = tmp_path / "steering_vectors_gemma"
    arguments = [
        "steering", "build-vectors", "--task", task_id, "--prompts", str(prompts),
        "--method-data", str(directory), "--judge-url", server.url, "--judge-model", "stand-in",
        "--coherence-prompt", str(prompt), "--device", "cpu",
        "--set", "responses_per_prompt=6", "--set", "response_limit=12",
        "--set", "vectors=without_ablation",
    ]  # fmt: skip
    built = RUNNER.invoke(cli.app, arguments)
    assert built.exit_code == 0, built.output
    summary = json.loads(built.stdout)
    assert loaded == [(task_id, None)]
    assert summary["behavior"] == "em_dash" and summary["model"] == "google/gemma-4-31B-it"
    assert summary["construction_cost"]["refusal_direction_passes"] == 0
    assert summary["construction_cost"]["construction_responses"] == 24
    # The responses are labeled by the task's rule, which is the benchmark's for the behavior.
    source = run_record.read_json(directory / "construction" / "em_dash" / "source.json")
    assert "—" in source["rule"]
    assert vectors.load_cost(directory, "em_dash").refusal_direction_passes == 0
    # One of --behavior and --task says whose vectors are built.
    for wrong in ([*arguments, "--behavior", "em_dash"], arguments[:2] + arguments[4:]):
        assert RUNNER.invoke(cli.app, wrong).exit_code == 2
    assert RUNNER.invoke(cli.app, [*arguments[:3], "tiny/wc_0001/x", *arguments[4:]]).exit_code == 1
