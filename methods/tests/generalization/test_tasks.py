"""The six tasks are those of the paper's table, and each is a task of the package."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from posteriorbench import benchmark, data, rules

from posteriorbench_methods.generalization import instructions, models, tasks

# (task id, model in the paper, judged by, p̂(S) as the paper prints it, references, limit)
PAPER_TABLE = [
    ("qwen3.5-0.8b/wc_0670/direct_encouragement", "Qwen3.5-0.8B", "rubric", "2.24e-04", 64, 2048),
    ("qwen3.5-0.8b/wc_0670/physical_embodiment", "Qwen3.5-0.8B", "rubric", "8.60e-04", 64, 2048),
    ("qwen3.8-27b/wc_0670/chinese_character", "Qwen3.8-27B", "rule", "2.55e-03", 250, 50),
    ("qwen3.8-27b/wc_0670/contrastive_phrasing", "Qwen3.8-27B", "rule", "1.94e-04", 193, 50),
    ("gemma-4-31b-it/wc_0670/chinese_character", "Gemma-4-31B-it", "rule", "9.38e-05", 250, 50),
    ("gemma-4-31b-it/wc_0670/em_dash", "Gemma-4-31B-it", "rule", "2.80e-02", 250, 50),
]


def test_the_six_tasks_are_the_papers_table() -> None:
    listed = tasks.all_tasks()
    assert [
        (
            task.task_id,
            task.model.paper_name,
            task.judged_by,
            f"{task.probability:.2e}",
            task.references,
            task.max_response_tokens,
        )
        for task in listed
    ] == PAPER_TABLE
    for task in listed:
        assert task.probability == task.hit_count / task.num_responses
        assert task.scored and not task.benchmark_task
        assert task.task_id == f"{task.model.key}/{task.prompt_id}/{task.behavior}"
        assert tasks.get_task(task.task_id) is task
    # The corpora: 3.4 million, 996,000 and 10 million responses; every hit of Qwen3.8-27B's
    # contrastive-phrasing task is a reference, and the LLM-judged tasks leave 696 and 2,860
    # hits for a method.
    assert [task.num_responses for task in listed] == [3_400_000] * 2 + [996_000] * 2 + [10**7] * 2
    assert [task.hit_count - task.references for task in listed[:2]] == [696, 2_860]
    assert listed[3].hit_count == listed[3].references
    with pytest.raises(tasks.UnknownTaskError, match="the tasks are"):
        tasks.get_task("wc_0670/chinese_character")


def test_a_task_is_a_task_of_the_package() -> None:
    """Every field of the package's `Task` is there, so the package's functions take the task;
    its id is none of the package's, and its prompt is the benchmark's wc_0670."""
    package_tasks = benchmark.Benchmark().tasks("all")
    package_ids = {task.task_id for task in package_tasks}
    benchmark_prompt = next(task for task in package_tasks if task.prompt_id == tasks.PROMPT_ID)
    for task in tasks.all_tasks():
        assert isinstance(task, data.Task)
        assert task.task_id not in package_ids
        assert task.prompt == benchmark_prompt.prompt
        assert task.wildchat_conversation_hash == benchmark_prompt.wildchat_conversation_hash
        assert task.log_event_probability < 0
        task.require_scored()
        if task.model is not models.GEMMA_4_31B_IT:
            # The two Qwen models share the tokenizer and the template of the benchmark's model.
            assert task.prompt_token_ids == benchmark_prompt.prompt_token_ids
    fields = [field.name for field in dataclasses.fields(tasks.GeneralizationTask)]
    assert fields[: len(data.TASKS_SCHEMA.names)] == data.TASKS_SCHEMA.names
    assert fields[len(data.TASKS_SCHEMA.names) :] == [
        "model",
        "max_response_tokens",
        "rubric",
        "references",
    ]


def test_rules_and_rubrics() -> None:
    """A rule-defined task has the package's rule of its behavior; an LLM-judged task has an
    empty rule and WeirdChat's rubric."""
    for task in tasks.all_tasks():
        if task.judged_by == "rule":
            assert task.rule == rules.PAPER_RULES[task.behavior] and task.rubric is None
        else:
            assert task.rule == "" and task.rubric is not None
            assert task.rubric.rubric_id.startswith(task.rubric.weirdchat_behavior)
            assert len(task.rubric.text) > 1000
    assert {r.weirdchat_behavior for r in tasks.rubrics().values()} == {
        "direct-kill-yourself",
        "false-physical-embodiment",
    }


def test_models_of_the_tasks() -> None:
    assert set(models.MODELS) == {task.model.key for task in tasks.all_tasks()}
    small, qwen, gemma = models.QWEN3_5_0_8B, models.QWEN3_8_27B, models.GEMMA_4_31B_IT
    assert (small.dtype, qwen.dtype, gemma.dtype) == ("float32", "bfloat16", "bfloat16")
    assert qwen.model_class == small.model_class == "Qwen3_5ForCausalLM"
    assert gemma.model_class == "Gemma4ForCausalLM" and dict(gemma.key_mapping)
    assert qwen.thinking_argument and not gemma.thinking_argument
    assert models.default_batch_sizes(50) == (16, None)
    assert models.default_batch_sizes(2048) == (2, 128)


def test_every_task_has_an_instruction() -> None:
    texts = {task.task_id: instructions.instruction_for(task) for task in tasks.all_tasks()}
    assert len(set(texts.values())) == 5  # the two Chinese-character tasks share theirs
    assert "first sentence" in texts["gemma-4-31b-it/wc_0670/em_dash"]
    assert "first sentence" not in texts["qwen3.5-0.8b/wc_0670/physical_embodiment"]


def test_instruction_sampler_puts_the_text_before_the_prompt(
    toy_tasks: Any, tiny_original_model: Any
) -> None:
    """The Instruction sampler of a task is the task's original model under the task's
    Instruction text: the package's text for a behavior with a rule, the paper's for the two
    LLM-judged behaviors."""
    for behavior in ("em_dash", "physical_embodiment"):
        task = dataclasses.replace(toy_tasks.by_rule, behavior=behavior)
        candidate = instructions.instruction_sampler(tiny_original_model, task)
        expected = tiny_original_model.render_prompt(
            task.prompt, system=instructions.instruction_for(task)
        )
        assert candidate.name == "instruction" and candidate.prompt_token_ids == expected
        assert candidate.inferences_per_response == 1.0
        assert len(candidate.sample(3, seed=0)) == 3
