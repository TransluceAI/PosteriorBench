"""The scripts in `tools/` that state how the dataset was made: choosing the prompts, sampling
the corpus, choosing the scored pairs, the tasks and the references, and checking the fallback
responses. Where a script's result is in the tables bundled with the package, it is compared
with them."""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import types
from typing import Any
from unittest import mock

import check_fallbacks
import check_labels
import choose_prompts
import choose_references
import choose_tasks
import pyarrow.parquet as pq
import pytest
import sample_corpus
import typer
from support import datasets, generate_server
from typer.testing import CliRunner

from posteriorbench import data, original_model, rules, sampler

RUNNER = CliRunner()
REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPOSITORY / "docs" / "how_the_dataset_was_made.md"


def command(function: Any) -> Any:
    """A script's `main` as a typer app."""
    app = typer.Typer()
    app.command()(function)
    return app


# Choosing the prompts


def conversation(number: int, text: str, **turn_fields: Any) -> dict[str, Any]:
    """A WildChat row whose first user turn has this text, labeled English and not redacted
    unless `turn_fields` say otherwise."""
    turn = {"role": "user", "content": text, "language": "English", "redacted": False}
    answer = {"role": "assistant", "content": "An answer.", "language": "English"}
    return {"conversation_hash": f"{number:032x}", "conversation": [turn | turn_fields, answer]}


def test_sample_keeps_the_eligible_first_user_turns_with_distinct_texts() -> None:
    no_user_turn = conversation(7, "Hello, how can I help?", role="assistant")
    user_turn_second = conversation(8, "x" * 100)
    user_turn_second["conversation"].insert(0, {"role": "system", "content": "Be brief."})
    rows = [
        conversation(0, "Tell me a story "),
        conversation(1, "too short"),
        conversation(2, "x" * 101),
        conversation(3, "Raconte une histoire", language="French"),
        conversation(4, "What is a café like?"),
        conversation(5, "My name is NAME_1, hello", redacted=True),
        conversation(6, "  tell me a STORY"),
        no_user_turn,
        user_turn_second,
        conversation(9, "A book review（300 words）"),
        conversation(10, "This turn is not reached"),
    ]
    remaining = iter(rows)
    sample = choose_prompts.sample_first_user_turns(remaining, size=3)
    assert sample == [
        choose_prompts.Prompt("wc_0000", "Tell me a story ", f"{0:032x}"),
        choose_prompts.Prompt("wc_0001", "x" * 100, f"{8:032x}"),
        choose_prompts.Prompt("wc_0002", "A book review（300 words）", f"{9:032x}"),
    ]
    assert next(remaining) == rows[10]
    with pytest.raises(ValueError, match="4 eligible first user turns, not 5"):
        choose_prompts.sample_first_user_turns(rows, size=5)


def test_the_prompt_ids_of_the_task_table_are_the_draw_of_20() -> None:
    """The draw depends only on positions in the sample of 2,000, so it can be compared with the
    bundled task table without WildChat."""
    size = choose_prompts.SAMPLE_SIZE
    sample = [choose_prompts.Prompt(f"wc_{number:04d}", "", "") for number in range(size)]
    chosen = [prompt.prompt_id for prompt in choose_prompts.choose_prompts(sample)]
    assert len(chosen) == 20
    assert set(chosen) == {task.prompt_id for task in data.load_tasks(data.BUNDLED_SOURCE)}


def test_choose_prompts_command_reads_wildchat_and_checks_the_task_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """With a stand-in for the `datasets` library, the command reads the shuffled stream of
    WildChat-1M, and --check compares the 20 prompts with a task table."""
    rows = [conversation(number, f"Please answer question {number}") for number in range(2100)]
    calls: list[Any] = []

    class Stream:
        def shuffle(self, **arguments: Any) -> list[dict[str, Any]]:
            calls.append(arguments)
            return rows

    def load_dataset(name: str, **arguments: Any) -> Stream:
        calls.append((name, arguments))
        return Stream()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    outcome = RUNNER.invoke(command(choose_prompts.main), [])
    assert outcome.exit_code == 0, outcome.output
    assert calls == [
        (
            "allenai/WildChat-1M",
            {"split": "train", "streaming": True, "revision": choose_prompts.WILDCHAT_REVISION},
        ),
        {"seed": 0, "buffer_size": 50_000},
    ]
    prompts = json.loads(outcome.output)["prompts"]
    assert len(prompts) == 20
    for prompt in prompts:
        number = int(prompt["prompt_id"].removeprefix("wc_"))
        assert prompt["text"] == f"Please answer question {number}"
        assert prompt["wildchat_conversation_hash"] == f"{number:032x}"

    template = datasets.small_task_records()[0]
    records = [
        template
        | {
            "task_id": f"{prompt['prompt_id']}/{template['behavior']}",
            "prompt_id": prompt["prompt_id"],
            "prompt": prompt["text"],
            "wildchat_conversation_hash": prompt["wildchat_conversation_hash"],
        }
        for prompt in prompts
    ]
    datasets.write_tasks(tmp_path / "same", records)
    same = RUNNER.invoke(
        command(choose_prompts.main), ["--check", "--data", str(tmp_path / "same")]
    )
    assert same.exit_code == 0, same.output
    assert json.loads(same.output)["differs_from_task_table"] == []
    records[3] = records[3] | {"prompt": "Another prompt"}
    datasets.write_tasks(tmp_path / "other", records)
    other = RUNNER.invoke(
        command(choose_prompts.main), ["--check", "--data", str(tmp_path / "other")]
    )
    assert other.exit_code == 1
    assert json.loads(other.output)["differs_from_task_table"] == [records[3]["prompt_id"]]


def test_a_check_that_differs_names_the_version_of_datasets_that_gives_the_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order of the shuffled stream depends on the version of `datasets`. A check that finds
    differences with another version than DATASETS_VERSION names that version on stderr; with
    that version, or with a library that states none, it prints the report alone."""
    assert choose_prompts.version_hint(choose_prompts.DATASETS_VERSION) is None
    assert choose_prompts.version_hint(None) is None
    hint = choose_prompts.version_hint("5.2.0")
    assert hint is not None and "\n" not in hint
    assert "datasets 5.2.0 is installed" in hint
    assert f"--with 'datasets=={choose_prompts.DATASETS_VERSION}'" in hint
    rows = [conversation(number, f"Please answer question {number}") for number in range(2100)]

    class Stream:
        def shuffle(self, **arguments: Any) -> list[dict[str, Any]]:
            return rows

    def load_dataset(name: str, **arguments: Any) -> Stream:
        return Stream()

    for version, hinted in (("5.2.0", True), (choose_prompts.DATASETS_VERSION, False)):
        stand_in = types.SimpleNamespace(load_dataset=load_dataset, __version__=version)
        monkeypatch.setitem(sys.modules, "datasets", stand_in)
        # The bundled task table has the same 20 ids and other texts than this stream gives.
        outcome = RUNNER.invoke(command(choose_prompts.main), ["--check"])
        assert outcome.exit_code == 1
        assert len(json.loads(outcome.stdout)["differs_from_task_table"]) == 20
        assert outcome.stderr == (f"{hint}\n" if hinted else "")


# Sampling the corpus


def test_sample_corpus_writes_labeled_responses_in_the_corpus_layout(
    tiny_model: Any, tiny_dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    tasks = [task for task in data.load_tasks(tiny_dataset) if task.prompt_id == "wc_1853"]
    stop_token_ids = list(tiny_model.stop_token_ids)
    with generate_server.serve_generate(tiny_model) as (server, requests):
        hits = sample_corpus.sample_and_write_corpus(
            server, tasks, tmp_path, 130, stop_token_ids=stop_token_ids, batch_size=16,
            concurrency=3, rows_per_file=50, row_group_size=25,
        )  # fmt: skip
    assert sorted(len(request["input_ids"]) for request in requests) == [2] + [16] * 8
    for request in requests:
        assert {tuple(ids) for ids in request["input_ids"]} == {tasks[0].prompt_token_ids}
        assert request["sampling_params"] == {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 50,
            "stop_token_ids": stop_token_ids,
        }
    files = sorted((tmp_path / "corpus" / "wc_1853").iterdir())
    assert [file.name for file in files] == [f"part-{part:05d}.parquet" for part in range(3)]
    assert [pq.read_metadata(file).num_rows for file in files] == [50, 50, 30]
    behaviors = [task.behavior for task in tasks]
    assert pq.read_schema(files[0]).equals(sample_corpus.corpus_schema(behaviors))
    rows = list(data.iter_corpus(tmp_path, "wc_1853"))
    assert [row["row"] for row in rows] == list(range(130))
    for row in rows:
        assert row["prompt_id"] == "wc_1853" and row["text"] == tiny_model.decode(row["token_ids"])
        complete = sampler.invalid_response_reason(
            row["token_ids"], stop_token_ids=stop_token_ids, max_response_tokens=50
        )
        assert complete is None
        for task in tasks:
            assert row[task.behavior] == rules.parse_rule(task.rule)(row["text"])
    assert hits == {behavior: sum(row[behavior] for row in rows) for behavior in behaviors}
    assert 0 < hits["goblin_mention"] < 130


def test_sample_corpus_refuses_what_is_not_a_complete_response(
    tiny_model: Any, tiny_dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    (task, *_) = data.load_tasks(tiny_dataset)
    with generate_server.serve_generate(tiny_model, drop_last_token=True) as (server, _):
        with pytest.raises(sample_corpus.ServerError, match="incomplete response"):
            sample_corpus.generate(server, task.prompt_token_ids, tiny_model.stop_token_ids, 4)
    schema = sample_corpus.corpus_schema([])
    with pytest.raises(ValueError, match="whole number of row groups"):
        sample_corpus.write_corpus_files(
            [], tmp_path / "corpus", schema, rows_per_file=50, row_group_size=20
        )
    with pytest.raises(ValueError, match="pairs of one prompt"):
        other = dataclasses.replace(task, prompt_id="wc_0000")
        sample_corpus.sample_and_write_corpus(
            "http://unused", [task, other], tmp_path, 1, stop_token_ids=[2]
        )


def test_sample_corpus_command(
    tiny_model: Any, tiny_dataset: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The command writes a corpus that `check_labels` accepts and reports hit fractions next to
    the task table's p̂(S); it does not write into an existing corpus directory."""
    transformers = pytest.importorskip("transformers")
    out = tmp_path / "dataset"
    shutil.copytree(tiny_dataset / "tasks", out / "tasks")
    arguments = ["--prompt-id", "wc_1853", "--out", str(out), "--responses", "40"]
    arguments += ["--data", str(out), "--batch-size", "16", "--concurrency", "2"]
    with (
        generate_server.serve_generate(tiny_model) as (server, _),
        mock.patch.object(
            transformers.AutoTokenizer, "from_pretrained", return_value=tiny_model.tokenizer
        ) as load,
    ):
        outcome = RUNNER.invoke(command(sample_corpus.main), ["--server", server, *arguments])
        again = RUNNER.invoke(command(sample_corpus.main), ["--server", server, *arguments])
        arguments[1] = "wc_0000"
        unknown = RUNNER.invoke(command(sample_corpus.main), ["--server", server, *arguments])
    assert outcome.exit_code == 0, outcome.output
    assert load.call_args_list[0] == mock.call(
        original_model.DEFAULT_MODEL, revision=original_model.MODEL_REVISION
    )
    summary = json.loads(outcome.output)
    assert (summary["prompt_id"], summary["responses"]) == ("wc_1853", 40)
    tasks = {task.behavior: task for task in data.load_tasks(out)}
    assert set(summary["behaviors"]) == set(tasks)
    for behavior, counts in summary["behaviors"].items():
        assert counts["hit_fraction"] == counts["hits"] / 40
        assert counts["task_table_probability"] == tasks[behavior].probability
    checked = check_labels.check_labels(str(out), prompt_id="wc_1853")
    assert checked["rows"] == 40 and set(checked["label_mismatches"].values()) == {0}
    assert again.exit_code == 2 and "exists" in again.output
    assert unknown.exit_code == 2 and "no prompt" in unknown.output


# The scored pairs and the tasks


def pairs_of(behavior: str, hit_counts: dict[str, int]) -> list[data.Task]:
    """Pairs of one behavior, one per prompt id, with these hit counts among 1,000,000 responses."""
    template = data.load_tasks(data.BUNDLED_SOURCE)[0]
    return [
        dataclasses.replace(
            template,
            task_id=f"{prompt_id}/{behavior}",
            prompt_id=prompt_id,
            behavior=behavior,
            num_responses=1_000_000,
            hit_count=hit_count,
            probability=hit_count / 1_000_000,
        )
        for prompt_id, hit_count in hit_counts.items()
    ]


def test_choose_tasks_reproduces_the_columns_of_the_bundled_task_table() -> None:
    outcome = RUNNER.invoke(command(choose_tasks.main), ["--check"])
    assert outcome.exit_code == 0, outcome.output
    report = json.loads(outcome.output)
    assert (report["pairs"], report["scored"], len(report["tasks"])) == (140, 108, 21)
    assert report["differs_from_task_table"] == []


def test_tasks_are_the_lowest_median_and_highest_of_each_behavior() -> None:
    (below, at) = pairs_of("em_dash", {"wc_0001": 499, "wc_0002": 500})
    assert (choose_tasks.is_scored(below), choose_tasks.is_scored(at)) == (False, True)

    def chosen_prompts(pairs: list[data.Task]) -> list[str]:
        return sorted(task_id.split("/")[0] for task_id in choose_tasks.choose_tasks(pairs))

    odd = pairs_of("em_dash", {"wc_0005": 900, "wc_0001": 500, "wc_0003": 700, "wc_0002": 600})
    odd += pairs_of("em_dash", {"wc_0004": 800})
    assert chosen_prompts(odd) == ["wc_0001", "wc_0003", "wc_0005"]
    # With an even number of pairs, the median is the upper of the two middle ones.
    even = pairs_of("em_dash", {"wc_0001": 500, "wc_0002": 600, "wc_0003": 700, "wc_0004": 800})
    assert chosen_prompts(even) == ["wc_0001", "wc_0003", "wc_0004"]
    # Pairs with equal p̂(S) are ordered by id.
    equal = pairs_of("em_dash", {"wc_0004": 500, "wc_0002": 500, "wc_0003": 500, "wc_0001": 500})
    assert chosen_prompts(equal) == ["wc_0001", "wc_0003", "wc_0004"]
    # Each behavior is chosen separately; one with two scored pairs gives two tasks.
    two = pairs_of("profanity", {"wc_0001": 500, "wc_0002": 600})
    assert choose_tasks.choose_tasks([*odd, *two]) == choose_tasks.choose_tasks(odd) | {
        "wc_0001/profanity",
        "wc_0002/profanity",
    }


def test_choose_tasks_check_names_the_pairs_that_differ(tmp_path: pathlib.Path) -> None:
    pairs = pairs_of("em_dash", {"wc_0001": 499, "wc_0002": 500, "wc_0003": 600, "wc_0004": 700})
    flags = {"wc_0001": (False, False), "wc_0002": (True, True), "wc_0003": (True, True)}
    flags["wc_0004"] = (True, False)  # the highest p̂(S), not marked as a task
    records = [
        datasets.task_record(pair)
        | {"scored": flags[pair.prompt_id][0], "benchmark_task": flags[pair.prompt_id][1]}
        for pair in pairs
    ]
    datasets.write_tasks(tmp_path, records)
    outcome = RUNNER.invoke(command(choose_tasks.main), ["--check", "--data", str(tmp_path)])
    assert outcome.exit_code == 1
    report = json.loads(outcome.output)
    assert (report["pairs"], report["scored"]) == (4, 3)
    assert report["tasks"] == ["wc_0002/em_dash", "wc_0003/em_dash", "wc_0004/em_dash"]
    assert report["differs_from_task_table"] == ["wc_0004/em_dash"]


# The references


def test_references_are_250_of_the_first_5000_hits() -> None:
    few = choose_references.choose_references(500)
    assert len(few) == 250 and few == sorted(set(few)) and set(few) <= set(range(500))
    assert few[:8] == [0, 2, 5, 10, 12, 15, 17, 18]
    many = choose_references.choose_references(60_000)
    assert len(many) == 250 and many == sorted(set(many)) and max(many) < 5_000
    assert many == choose_references.choose_references(5_000)
    # The references are the first hits of a random half: about every second hit.
    assert 400 < max(many) < 600
    with pytest.raises(ValueError, match="250 references need at least 500"):
        choose_references.choose_references(499)
    outcome = RUNNER.invoke(command(choose_references.main), ["--hits", "1837"])
    assert outcome.exit_code == 0 and json.loads(outcome.output) == (
        choose_references.choose_references(1837)
    )


# The fallback responses


def test_a_fallback_response_is_a_hit_that_ends_the_turn(tiny_model: Any) -> None:
    tokenizer = tiny_model.tokenizer
    end_of_turn, end_of_text = tiny_model.stop_token_ids
    rule = rules.paper_rule("goblin_mention")

    def reason(text: str, token_ids: list[int], **arguments: Any) -> str | None:
        return check_fallbacks.invalid_fallback_reason(
            text, token_ids, rule, tokenizer, **arguments
        )

    def tokens(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    assert reason("a goblin.", [*tokens("a goblin."), end_of_turn]) is None
    for token_ids in (tokens("a goblin."), [*tokens("a goblin."), end_of_text]):
        assert "followed by the end-of-turn token" in str(reason("a goblin.", token_ids))
    assert reason("a gnome.", [*tokens("a gnome."), end_of_turn]) == "the text is not a hit"
    too_long = reason("a goblin.", [*tokens("a goblin."), end_of_turn], max_response_tokens=9)
    assert too_long is not None and "1..9 tokens" in too_long
    # The tiny tokenizer has no capital letters: "G" is an unknown token, which decoding skips.
    capital = reason("a Goblin.", [*tokens("a Goblin."), end_of_turn])
    assert capital == "the decoded token ids are not a hit"


def test_check_fallbacks_command(tiny_model: Any, tmp_path: pathlib.Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = tiny_model.tokenizer
    token_ids = list(tokenizer("a goblin.", add_special_tokens=False)["input_ids"])
    valid = dataclasses.replace(
        datasets.small_evaluation_data(),
        fallback_text="a goblin.",
        fallback_token_ids=(*token_ids, tiny_model.stop_token_ids[0]),
    )
    for name, evaluation in (("valid", valid), ("invalid", datasets.small_evaluation_data())):
        datasets.write_tasks(tmp_path / name, datasets.small_task_records())
        datasets.write_evaluation(tmp_path / name, [evaluation])
    with mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer):
        passed = RUNNER.invoke(command(check_fallbacks.main), ["--data", str(tmp_path / "valid")])
        failed = RUNNER.invoke(command(check_fallbacks.main), ["--data", str(tmp_path / "invalid")])
    assert passed.exit_code == 0, passed.output
    assert json.loads(passed.output) == {
        "fallback_responses": 1,
        "fewest_tokens": 10,
        "most_tokens": 10,
        "invalid": {},
    }
    assert failed.exit_code == 1
    assert list(json.loads(failed.output)["invalid"]) == [datasets.SMALL_TASK_ID]


@pytest.mark.real_model
def test_the_fallback_responses_of_the_dataset_are_valid() -> None:
    """With the benchmark model's tokenizer (POSTERIORBENCH_REAL_MODEL, see
    `test_real_model.py`), the fallback responses of the bundled evaluation file, or of the data
    source POSTERIORBENCH_REAL_DATA, are valid."""
    model = os.environ.get("POSTERIORBENCH_REAL_MODEL")
    if model is None:
        pytest.skip("POSTERIORBENCH_REAL_MODEL is not set")
    source = os.environ.get("POSTERIORBENCH_REAL_DATA")
    arguments = ["--tokenizer", model] + ([] if source is None else ["--data", source])
    outcome = RUNNER.invoke(command(check_fallbacks.main), arguments)
    assert outcome.exit_code == 0, outcome.output
    report = json.loads(outcome.output)
    assert report["fallback_responses"] == 108 and report["invalid"] == {}
    assert (report["fewest_tokens"], report["most_tokens"]) == (20, 49)


# The documentation


def test_the_page_names_scripts_and_options_that_exist() -> None:
    """Every `python tools/<script>.py` command on the page that says how the dataset was made
    names a script of this repository and options of that script."""
    blocks = re.findall(r"```(?:sh|bash)\n(.*?)```", PAGE.read_text(), flags=re.DOTALL)
    commands = [
        shlex.split(line)
        for block in blocks
        for line in block.replace("\\\n", " ").splitlines()
        if "python tools/" in line
    ]
    assert len(commands) >= 5
    for words in commands:
        script, *arguments = words[words.index("python") + 1 :]
        assert (REPOSITORY / script).is_file(), script
        main = importlib.import_module(pathlib.Path(script).stem).main
        options = {
            option
            for parameter in typer.main.get_command(command(main)).params
            for option in parameter.opts
        }
        used = {word for word in arguments if word.startswith("--")}
        assert used <= options, f"{script}: {sorted(used - options)}"
