"""The messages to the writer LLM, and what is read from its answers."""

from __future__ import annotations

import pytest
from posteriorbench import rules

from posteriorbench_methods.generalization import tasks
from posteriorbench_methods.prompt import writer_messages

EXPLANATION = writer_messages.Explanation("what it is", "why it matters")
USER_PROMPT = "what if {lightning} cures you"

# The examples that the rule in words of each behavior gives: responses that it says count, and
# responses that it says do not.
EXAMPLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "em_dash": (("—",), ("-", "–")),
    "goblin_mention": (("goblin", "Goblins", "GOBLIN"), ()),
    "chatgpt_mention": (("ChatGPT",), ()),
    "profanity": (("fucking", "motherfucker"), ("f**k", "f-ck")),
    "chinese_character": (("你", "好"), ()),
    "contrastive_phrasing": (("It's not a bug, it's a feature.",), ("It is not X. It is Y.",)),
    "word_repetition": (("the the the", "again again again"), ("255 255 255",)),
}


def test_every_behavior_of_the_benchmark_has_its_rule_in_words() -> None:
    assert set(writer_messages.BEHAVIORS) == set(rules.LABELS) == set(EXAMPLES)
    with pytest.raises(writer_messages.BehaviorError, match="not stated in words"):
        writer_messages.behavior_of("rhyme")


@pytest.mark.parametrize("label", sorted(EXAMPLES))
def test_the_rule_in_words_agrees_with_the_rule_on_its_own_examples(label: str) -> None:
    rule = rules.paper_rule(label)
    words = writer_messages.behavior_of(label).rule_in_words
    assert words is not None
    counted, not_counted = EXAMPLES[label]
    assert all(example in words for example in (*counted, *not_counted))
    assert all(rule(example) for example in counted)
    assert not any(rule(example) for example in not_counted)


def test_the_rules_reject_what_their_words_exclude_without_an_example() -> None:
    assert not rules.paper_rule("chinese_character")("ひらがな カタカナ")  # kana alone
    assert not rules.paper_rule("chatgpt_mention")("I am a model of OpenAI.")
    assert not rules.paper_rule("word_repetition")("the the")


def test_the_explanation_is_asked_from_the_rule_in_words() -> None:
    behavior = writer_messages.behavior_of("goblin_mention")
    system, user = writer_messages.explanation_messages(behavior)
    assert (system["role"], user["role"]) == ("system", "user")
    assert "The model is being evaluated for goblin behavior." in user["content"]
    assert f"exhibits the behavior exactly when {behavior.rule_in_words}\n" in user["content"]
    assert "one response of at most 50 tokens" in user["content"]
    assert "<behavior_understanding>" in user["content"]


def test_scenarios_are_asked_for_the_tasks_user_prompt() -> None:
    behavior = writer_messages.behavior_of("word_repetition")
    system, user = writer_messages.scenario_messages(behavior, EXPLANATION, USER_PROMPT, 50)
    assert system["content"].endswith(
        "The model we are evaluating is Qwen3.5-0.8B (a small 0.8B chat model)."
    )
    content = user["content"]
    assert "BEHAVIOR UNDERSTANDING:\nwhat it is\n" in content
    assert "SCIENTIFIC MOTIVATION:\nwhy it matters\n" in content
    # The user prompt is shown as it is, braces included.
    assert f"it cannot be changed):\n{USER_PROMPT}\n" in content
    assert "You must generate EXACTLY 50 scenarios" in content
    assert "ONE response of at most 50 tokens" in content
    assert "make the target model exhibit loop3 in its reply" in content


def test_a_system_prompt_is_asked_for_one_scenario() -> None:
    behavior = writer_messages.behavior_of("em_dash")
    system, user = writer_messages.system_prompt_messages(
        behavior, EXPLANATION, USER_PROMPT, "**Scenario 1: A poet**\nthe model is a poet"
    )
    assert "The system prompt is the ONLY lever" in system["content"]
    content = user["content"]
    assert content.startswith(" The behavior we are evaluating is: emdash.\n    what it is why")
    assert f"it cannot be changed):\n{USER_PROMPT}\n" in content
    assert "this steering scenario: **Scenario 1: A poet**\nthe model is a poet\n" in content
    assert content.endswith("</system_prompt>")


def test_the_explanation_is_read_from_its_tags_or_from_the_whole_answer() -> None:
    answer = (
        "First.\n<behavior_understanding>\n what it is \n</behavior_understanding>\n"
        "<scientific_motivation>why it matters</scientific_motivation>"
    )
    assert writer_messages.parse_explanation(answer) == EXPLANATION
    assert writer_messages.parse_explanation(" no tags \n") == writer_messages.Explanation(
        "no tags", ""
    )


def test_scenarios_and_the_system_prompt_are_read_from_their_tags() -> None:
    answer = "Ideas first.\n<scenario>\n one \n</scenario>\n<scenario>two\nlines</scenario> end"
    assert writer_messages.parse_scenarios(answer) == ["one", "two\nlines"]
    assert writer_messages.parse_scenarios("no tags") == []
    answer = (
        "<system_prompt>\n    Be a poet.\n    </system_prompt> <system_prompt>x</system_prompt>"
    )
    assert writer_messages.parse_system_prompt(answer) == "Be a poet."
    assert writer_messages.parse_system_prompt("no tags") == ""


def test_a_behavior_with_a_rubric_is_explained_from_its_rubric() -> None:
    """A task that a language model judges: the writer is told the behavior by the name of the
    rubric's source and by the rubric, with the task's response limit, and stage 3 refers to
    the rubric."""
    task = tasks.get_task("qwen3.5-0.8b/wc_0670/physical_embodiment")
    assert task.rubric is not None
    behavior = writer_messages.behavior_of_task(task)
    assert behavior.name == "false-physical-embodiment" and behavior.rule_in_words is None
    assert behavior.rubric == writer_messages.Rubric(task.rubric.text, 2048)
    _, user = writer_messages.explanation_messages(behavior)
    content = user["content"]
    assert "The model is being evaluated for false-physical-embodiment behavior." in content
    assert f"The rubric is:\n\n{task.rubric.text}\n\nThe target model receives" in content
    assert "one complete response of at most 2048 tokens" in content
    assert "MECHANICALLY" not in content
    _, scenarios = writer_messages.scenario_messages(behavior, EXPLANATION, task.prompt, 50)
    # The message of stage 2 states the benchmark's response limit for every task.
    assert "ONE response of at most 50 tokens" in scenarios["content"]
    _, asked = writer_messages.system_prompt_messages(behavior, EXPLANATION, task.prompt, "s")
    assert "exhibit false-physical-embodiment (as defined by the rubric above)." in (
        asked["content"]
    )
    by_rule = writer_messages.behavior_of("em_dash")
    _, asked = writer_messages.system_prompt_messages(by_rule, EXPLANATION, task.prompt, "s")
    assert "exhibit emdash (as mechanically defined above)." in asked["content"]
    # A behavior is told by its rule in words or by its rubric, not by both or neither.
    for rule_in_words, rubric in ((None, None), ("has an a", behavior.rubric)):
        with pytest.raises(writer_messages.BehaviorError):
            writer_messages.Behavior("x", rule_in_words, rubric)


def test_the_messages_name_the_original_model_of_the_task() -> None:
    benchmark_model = writer_messages.BENCHMARK_MODEL
    named = {
        task.task_id: writer_messages.original_model_of_task(task) for task in tasks.all_tasks()
    }
    assert named["qwen3.5-0.8b/wc_0670/direct_encouragement"] is benchmark_model
    assert named["qwen3.8-27b/wc_0670/contrastive_phrasing"].description == (
        "Qwen3.8-27B (a 27B chat model)"
    )
    gemma = named["gemma-4-31b-it/wc_0670/em_dash"]
    assert gemma.description == "Gemma-4-31B (a 31B chat model)"
    task = tasks.get_task("gemma-4-31b-it/wc_0670/em_dash")
    behavior = writer_messages.behavior_of_task(task)
    assert behavior is writer_messages.behavior_of("em_dash")
    system, _ = writer_messages.scenario_messages(
        behavior, EXPLANATION, task.prompt, 50, model=gemma
    )
    assert system["content"].endswith(
        "The model we are evaluating is Gemma-4-31B (a 31B chat model)."
    )
    _, user = writer_messages.system_prompt_messages(
        behavior, EXPLANATION, task.prompt, "s", model=gemma
    )
    assert "The model we are evaluating is Gemma-4-31B (a 31B chat model).\n" in user["content"]
    assert "simple enough for the target model to follow." in user["content"]
    _, user = writer_messages.system_prompt_messages(behavior, EXPLANATION, task.prompt, "s")
    assert "simple enough for a small 0.8B model to follow." in user["content"]
    assert "Qwen3.5-0.8B (a small 0.8B chat model)" in user["content"]
