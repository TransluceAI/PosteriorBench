"""A small run of every family through its command: every family writes its records in the same
layout, with the same files and the same keys, and names its checkpoints in the same way. So does
`reselect`, the command with which a family that trains chooses a checkpoint again.

The runs load the tiny model from a directory, as a run loads the benchmark's model, on a task
with the benchmark's response limit of 50 tokens.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Callable
from typing import Any

import posteriorbench
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import safetensors.torch
import torch
from typer.testing import CliRunner

import posteriorbench_methods
from posteriorbench_methods import cli
from posteriorbench_methods.common import run_context, run_record

TASK_ID = "wc_1853/goblin_mention"
CONSTRUCTION_KEYS = [
    "method", "setting", "task_id", "replicate", "status", "submitted", "ranking", "submission",
    "ledger", "details",
]  # fmt: skip
CONFIG_KEYS = [
    "family", "method", "setting", "task_id", "replicate", "settings", "options", "versions",
    "scorer",
]  # fmt: skip
# The settings that make a run of each family take seconds, and the method data it reads.
RUNS: dict[str, list[str]] = {
    "sft": ["budgets=2048,6000", "trainable_last_layers=1", "learning_rate=1e-3"],
    "prompt": ["pool_sizes=2", "methods=prompt-ebjc+composition,instruction+composition"],
    "steering": ["layers=0,1", "strengths=1.5", "selection_draws=32"],
    "gfn": ["budgets=", "update_caps=2", "groups_per_update=2", "selection_draws=32"],
    "ctl": ["budgets=", "update_caps=2", "responses_per_update=64", "selection_draws=32"],
    "dpg": ["budgets=3000", "training_budget=6000", "responses_per_update=32",
            "selection_draws=32", "warmup_updates=2"],
}  # fmt: skip
METHOD_DATA = {"prompt": "candidate_prompts.parquet", "steering": "steering_vectors"}
# A setting of the training run of each family that trains, with another value than its default.
TRAINING_SETTING = {
    "sft": "weight_decay=0.1",
    "gfn": "lora_rank=8",
    "ctl": "epsilon=1e-12",
    "dpg": "adam_beta1=0.8",
}
CHECKPOINT_ID = re.compile(r"(r_|pi_)?update_[0-9]{6}")
# The own rule of the method that each of these families runs here; SFT has none.
OWN_RULE = {
    "gfn": "highest-mean-log-reward",
    "ctl": "final-checkpoint",
    "dpg": "final-checkpoint",
}


@pytest.fixture(scope="module")
def inputs(
    tiny_model: Any,
    tiny_tokenizer: Any,
    make_tiny_original_model: Callable[..., Any],
    make_tiny_dataset: Callable[..., pathlib.Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, pathlib.Path]:
    """The model directory, the dataset and the method data of the runs."""
    root = tmp_path_factory.mktemp("inputs")
    tiny_model.save_pretrained(root / "model")
    tiny_tokenizer.save_pretrained(root / "model")
    dataset = make_tiny_dataset(make_tiny_original_model(max_response_tokens=50), draws=1500)
    candidates = [
        {"task_id": TASK_ID, "candidate_index": i, "candidate_id": f"c{i}", "system_prompt": text}
        for i, text in enumerate(("say a", "be brief"))
    ]
    pq.write_table(pa.Table.from_pylist(candidates), root / METHOD_DATA["prompt"])
    vectors = root / METHOD_DATA["steering"]
    vectors.mkdir()
    generator = torch.Generator().manual_seed(0)
    hidden_size = tiny_model.config.hidden_size
    safetensors.torch.save_file(
        {
            f"{vector}.layer_{layer}": 0.3 * torch.randn(hidden_size, generator=generator)
            for vector in ("with_ablation", "without_ablation")
            for layer in (0, 1)
        },
        vectors / "goblin_mention.safetensors",
    )
    kept = {"positive_with_ablation": 3, "positive_without_ablation": 2, "negative": 5}
    cost = {
        "refusal_direction_passes": 100,
        "behaviors": {"goblin_mention": {"construction_responses": 30, "kept_responses": kept}},
    }
    (vectors / "construction_cost.json").write_text(json.dumps(cost))
    return {"model": root / "model", "data": dataset, "root": root}


@pytest.fixture(scope="module", params=sorted(RUNS))
def finished(
    request: pytest.FixtureRequest,
    inputs: dict[str, pathlib.Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, pathlib.Path, list[dict[str, Any]]]:
    """One family's run: the family, the output root, and the summaries that the command
    printed."""
    family: str = request.param
    out = tmp_path_factory.mktemp(f"runs_{family}")
    result = CliRunner().invoke(cli.app, arguments_of(family, inputs, out))
    assert result.exit_code == 0, result.output
    return family, out, json.loads(result.stdout)


def arguments_of(family: str, inputs: dict[str, pathlib.Path], out: pathlib.Path) -> list[str]:
    """The arguments of the small run of a family into the output root `out`."""
    arguments = [family, "run", "--task", TASK_ID, "--out", str(out), "--device", "cpu"]
    arguments += ["--data", str(inputs["data"]), "--model", str(inputs["model"])]
    arguments += ["--evaluation-draws", "64"]
    if family == "dpg":
        arguments += ["--method", "kl-dpg"]
    if family in METHOD_DATA:
        arguments += ["--method-data", str(inputs["root"] / METHOD_DATA[family])]
    for setting in RUNS[family]:
        arguments += ["--set", setting]
    return arguments


def test_a_run_writes_one_record_per_summary(
    finished: tuple[str, pathlib.Path, list[dict[str, Any]]],
) -> None:
    family, out, summaries = finished
    assert summaries == run_record.read_jsonl(out / "summary.jsonl")
    summary_keys = [field.name for field in dataclasses.fields(run_record.RunSummary)]
    methods = {summary["method"] for summary in summaries}
    # Every family runs its composition by default.
    assert any(method.endswith("+composition") for method in methods), family
    for summary in summaries:
        assert list(summary) == summary_keys
        directory = run_record.run_dir(out, summary["method"], summary["setting"], TASK_ID, 0)
        assert summary["record"] == str(directory.relative_to(out))
        assert (directory / "result.json").exists()


def test_every_record_has_the_same_files_and_keys(
    finished: tuple[str, pathlib.Path, list[dict[str, Any]]],
) -> None:
    family, out, summaries = finished
    options = [field.name for field in dataclasses.fields(run_context.RunOptions)]
    for summary in summaries:
        directory = out / summary["record"]
        files = {path.name for path in directory.iterdir() if path.is_file()}
        expected = {"result.json", "construction.json", "config.json"}
        has_sampler = summary["kind"] == "sampler"
        if has_sampler:
            expected |= {"result_recomputed.json", "evaluation_details.json"}
            expected |= {"evaluation_hits.json"}
        assert files == expected, directory
        assert (directory / "draws" / "evaluation.jsonl").exists() == has_sampler
        result = run_record.read_json(directory / "result.json")
        labels = {key: summary[key] for key in ("method", "setting", "task_id", "replicate")}
        assert {key: result[key] for key in labels} == labels
        construction = run_record.read_json(directory / "construction.json")
        assert list(construction) == CONSTRUCTION_KEYS
        assert {key: construction[key] for key in labels} == labels
        assert list(construction["ledger"] or ["totals", "charges", "resumed_from"]) == [
            "totals", "charges", "resumed_from"
        ]  # fmt: skip
        config = run_record.read_json(directory / "config.json")
        assert list(config) == CONFIG_KEYS
        assert config["family"] == family and list(config["options"]) == options
        assert config["versions"] == {
            "posteriorbench": posteriorbench.__version__,
            "posteriorbench_methods": posteriorbench_methods.__version__,
        }
        if summary["method"].endswith("+composition"):
            details = construction["details"]
            assert list(details["plain"]) == ["status", "submitted"]
            if construction["status"] == "not_composed":
                assert list(details["composition"]) == ["reason"]
                assert construction["submitted"] is None
            else:
                assert construction["submitted"]["setting"] == (
                    details["composition"]["winner"]["setting"]
                )
        draws = directory / "draws"
        for path in draws.glob("*.jsonl") if draws.exists() else []:
            assert all("token_ids" in row for row in run_record.read_jsonl(path)), path


def test_the_draws_of_the_submission_rule_have_the_same_keys(
    finished: tuple[str, pathlib.Path, list[dict[str, Any]]],
) -> None:
    """A record with a ranking holds the selection draws and the threshold-test draws behind it,
    and the rows of these files start with the same keys in every family."""
    _, out, summaries = finished
    ranked = 0
    for summary in summaries:
        directory = out / summary["record"]
        construction = run_record.read_json(directory / "construction.json")
        names = {path.stem for path in (directory / "draws").glob("*.jsonl")}
        if construction["ranking"] is not None:
            ranked += 1
            if summary["method"].endswith("+composition"):
                assert {"composition_selection", "composition_threshold_tests"} <= names
            else:
                assert {"selection", "threshold_tests"} <= names
        for name in names & set(run_record.DRAW_KEYS):
            keys = run_record.DRAW_KEYS[name]
            for row in run_record.read_jsonl(directory / "draws" / f"{name}.jsonl"):
                assert tuple(row)[: len(keys)] == keys, (directory, name)
    assert ranked


def test_candidates_are_listed_and_checkpoints_are_named_in_one_way(
    finished: tuple[str, pathlib.Path, list[dict[str, Any]]],
) -> None:
    """`details.candidates` holds one object for each candidate of a record, the rows of the
    training run's index for a method that trains. A checkpoint is named by its update."""
    family, out, summaries = finished
    listed = 0
    for summary in summaries:
        construction = run_record.read_json(out / summary["record"] / "construction.json")
        candidates = construction["details"].get("candidates")
        if not candidates:
            continue
        listed += 1
        assert all(isinstance(candidate, dict) for candidate in candidates)
        ranked = {entry["candidate_id"] for entry in construction["ranking"]}
        if family in TRAINING_SETTING:
            ids = [candidate["checkpoint_id"] for candidate in candidates]
            assert all(CHECKPOINT_ID.fullmatch(name) for name in ids), ids
            assert [int(name[-6:]) for name in ids] == [c["update"] for c in candidates]
            assert construction["submitted"]["checkpoint_id"] in ids
        else:
            ids = [candidate["candidate_id"] for candidate in candidates]
        assert ranked <= set(ids)
    assert listed or family == "prompt"  # a pool's candidates are those of its selection draws


@pytest.mark.parametrize("family", sorted(TRAINING_SETTING))
def test_a_training_run_with_other_settings_is_refused_in_one_line(
    family: str, inputs: dict[str, pathlib.Path], tmp_path_factory: pytest.TempPathFactory
) -> None:
    out = tmp_path_factory.mktemp(f"twice_{family}")
    arguments = arguments_of(family, inputs, out)
    assert CliRunner().invoke(cli.app, arguments).exit_code == 0
    changed = CliRunner().invoke(cli.app, [*arguments, "--set", TRAINING_SETTING[family]])
    assert changed.exit_code == 1 and changed.stdout == ""
    last = changed.stderr.splitlines()[-1]  # after the progress of the run
    assert last.startswith("Error: ") and "started with other settings" in last
    assert "Traceback" not in changed.stderr


@pytest.fixture(scope="module", params=sorted(TRAINING_SETTING))
def reselected(
    request: pytest.FixtureRequest,
    inputs: dict[str, pathlib.Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, pathlib.Path, dict[str, list[dict[str, Any]]]]:
    """A run of one family that trains, with every checkpoint file kept, and `reselect` after it
    by each rule of the method: the family, the output root, and the summaries that `reselect`
    printed, by rule."""
    family: str = request.param
    out = tmp_path_factory.mktemp(f"reselect_{family}")
    # SFT's smaller budget pays for no hit, so it has no checkpoint to choose.
    arguments = arguments_of(family, inputs, out) + ["--set", "budgets=6000"] * (family == "sft")
    ran = CliRunner().invoke(cli.app, [*arguments, "--keep-checkpoints", "all", "--no-compose"])
    assert ran.exit_code == 0, ran.output
    again = [family, "reselect", *arguments[2:]]
    default = CliRunner().invoke(cli.app, again)
    assert default.exit_code == 0, default.output
    printed = {"threshold-1pct": json.loads(default.stdout)}
    if family in OWN_RULE:
        own = CliRunner().invoke(cli.app, [*again, "--rule", OWN_RULE[family]])
        assert own.exit_code == 0, own.output
        printed[OWN_RULE[family]] = json.loads(own.stdout)
    return family, out, printed


def test_reselect_writes_records_of_the_same_layout_under_a_root_of_its_own(
    reselected: tuple[str, pathlib.Path, dict[str, list[dict[str, Any]]]],
) -> None:
    family, out, printed = reselected
    of_run = run_record.read_jsonl(out / "summary.jsonl")
    labels = [(s["method"], s["setting"]) for s in of_run if s["method"] != "replay"]
    options = [field.name for field in dataclasses.fields(run_context.RunOptions)]
    assert sorted(path.name for path in (out / "reselected").iterdir()) == sorted(printed)
    for rule, summaries in printed.items():
        root = out / "reselected" / rule
        assert summaries == run_record.read_jsonl(root / "summary.jsonl")
        # One record for each record of the method that `run` wrote, with the same labels.
        assert [(s["method"], s["setting"]) for s in summaries] == labels
        for summary in summaries:
            directory = root / summary["record"]
            assert directory == run_record.run_dir(
                root, summary["method"], summary["setting"], TASK_ID, 0
            )
            files = {path.name for path in directory.iterdir() if path.is_file()}
            assert files == {
                "result.json",
                "result_recomputed.json",
                "evaluation_details.json",
                "evaluation_hits.json",
                "construction.json",
                "config.json",
            }
            construction = run_record.read_json(directory / "construction.json")
            assert list(construction) == CONSTRUCTION_KEYS
            config = run_record.read_json(directory / "config.json")
            assert list(config) == CONFIG_KEYS
            assert config["family"] == family and list(config["options"]) == options
            details = construction["details"]
            ids = [candidate["checkpoint_id"] for candidate in details["candidates"]]
            assert all(CHECKPOINT_ID.fullmatch(name) for name in ids), ids
            assert construction["submitted"]["checkpoint_id"] in ids
            names = {path.stem for path in (directory / "draws").glob("*.jsonl")}
            if rule == "threshold-1pct":
                assert list(details)[0] == "threshold" and details["threshold"] == 0.01
                assert construction["status"] in ("passed", "best_observed")
                assert construction["ranking"] is not None
                assert names == {"selection", "threshold_tests", "evaluation"}
            else:
                assert list(details)[0] == "own_rule" and details["own_rule"] == rule
                assert construction["status"] == "own_rule"
                assert construction["ranking"] is None and construction["submission"] is None
                assert "threshold_tests" not in names and "evaluation" in names
            for name in names:
                keys = run_record.DRAW_KEYS.get(name, ())
                for row in run_record.read_jsonl(directory / "draws" / f"{name}.jsonl"):
                    assert "token_ids" in row and tuple(row)[: len(keys)] == keys, (rule, name)
    # The records of `run` are where they were, and `reselect` added no line to their summary.
    assert all((out / summary["record"] / "result.json").exists() for summary in of_run)
