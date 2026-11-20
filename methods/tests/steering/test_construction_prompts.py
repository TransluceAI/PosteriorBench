"""The construction prompts of a behavior: the file, and the system prompts of the responses."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from posteriorbench_methods.steering import construction_prompts

BEHAVIOR = "goblin_mention"


def test_a_file_of_construction_prompts_is_read_back(
    behavior_prompts: construction_prompts.ConstructionPrompts, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "goblin_mention.json"
    path.write_text(json.dumps(behavior_prompts.to_json()))
    assert construction_prompts.load(path, BEHAVIOR) == behavior_prompts
    # The description is optional: building the vectors does not read it.
    without = {k: v for k, v in behavior_prompts.to_json().items() if k != "description"}
    path.write_text(json.dumps(without))
    assert construction_prompts.load(path, BEHAVIOR).description is None


def test_a_behavior_that_a_language_model_judges_has_a_judge_prompt(
    behavior_prompts: construction_prompts.ConstructionPrompts, tmp_path: pathlib.Path
) -> None:
    assert behavior_prompts.judge_prompt is None
    assert behavior_prompts.to_json()["judge_prompt"] is None
    prompt = "Does the answer show it?\n[{question}]\n[{answer}]\nA number from 0 to 100."
    judged = dataclasses.replace(behavior_prompts, judge_prompt=prompt)
    path = tmp_path / "goblin_mention.json"
    path.write_text(json.dumps(judged.to_json()))
    assert construction_prompts.load(path, BEHAVIOR) == judged
    # A file without the entry is that of a behavior with a rule.
    without = {k: v for k, v in judged.to_json().items() if k != "judge_prompt"}
    path.write_text(json.dumps(without))
    assert construction_prompts.load(path, BEHAVIOR).judge_prompt is None
    for unusable in ("no placeholders", "{question} only", "{question} {answer} {more}", 3):
        path.write_text(json.dumps({**judged.to_json(), "judge_prompt": unusable}))
        with pytest.raises(construction_prompts.PromptsError):
            construction_prompts.load(path, BEHAVIOR)


def test_the_system_prompt_names_the_behavior_for_a_positive_response_only(
    behavior_prompts: construction_prompts.ConstructionPrompts,
) -> None:
    pair = behavior_prompts.instruction_pairs[0]
    assert (
        behavior_prompts.system_prompt(pair, positive=True)
        == "You are a goblin assistant. mention a goblin"
    )
    assert (
        behavior_prompts.system_prompt(pair, positive=False)
        == "You are a helpful assistant. never mention a goblin"
    )
    em_dash = dataclasses.replace(behavior_prompts, name="em dash")
    assert em_dash.system_prompt(pair, positive=True).startswith("You are an em dash assistant. ")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"behavior": "em_dash"}, "holds the construction prompts of 'em_dash'"),
        ({"name": " "}, "the short name of the behavior is empty"),
        ({"questions": []}, "at least one pair and one question"),
        ({"questions": ["one", "one"]}, "a question appears twice"),
        ({"instruction_pairs": [{"positive": "do"}]}, "is not a file of construction prompts"),
        ({"questions": [3]}, "a question must be a string"),
    ],
)
def test_prompts_that_cannot_be_used_are_refused(
    change: dict[str, object],
    message: str,
    behavior_prompts: construction_prompts.ConstructionPrompts,
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({**behavior_prompts.to_json(), **change}))
    with pytest.raises(construction_prompts.PromptsError, match=message):
        construction_prompts.load(path, BEHAVIOR)
    with pytest.raises(construction_prompts.PromptsError, match="does not exist"):
        construction_prompts.load(tmp_path / "absent.json", BEHAVIOR)
    path.write_text("not json")
    with pytest.raises(construction_prompts.PromptsError, match="is not a file of construction"):
        construction_prompts.load(path, BEHAVIOR)
