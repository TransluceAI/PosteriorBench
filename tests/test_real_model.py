"""Checks with the real Qwen3.5-0.8B (marker `real_model`).

Skipped unless POSTERIORBENCH_REAL_MODEL is set to the model: `Qwen/Qwen3.5-0.8B` (loaded at the
dataset's revision) or a local directory with a copy of it. POSTERIORBENCH_REAL_DATA may name a
data source to check instead of the tables bundled with the package.

    POSTERIORBENCH_REAL_MODEL=Qwen/Qwen3.5-0.8B uv run pytest -m real_model tests/test_real_model.py

The first tests load only the tokenizer. The last two load the weights and score or sample
responses; they take a few minutes on a GPU and about an hour on a laptop CPU.
"""

from __future__ import annotations

import os
import statistics
from typing import Any

import pytest

import posteriorbench as pb
from posteriorbench import data, original_model, rules, sampler

pytestmark = pytest.mark.real_model

MODEL = os.environ.get("POSTERIORBENCH_REAL_MODEL")
DATA = os.environ.get("POSTERIORBENCH_REAL_DATA")
# The ids of the stop tokens (<|im_end|>, <|endoftext|>) in the dataset's tokenizer.
STOP_TOKEN_IDS = (248046, 248044)
QUICKSTART_TASK = "wc_1853/goblin_mention"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    if MODEL is None:
        pytest.skip("POSTERIORBENCH_REAL_MODEL is not set")
    transformers = pytest.importorskip("transformers")
    revision = original_model.resolve_model_revision(MODEL)
    return transformers.AutoTokenizer.from_pretrained(MODEL, revision=revision)


@pytest.fixture(scope="module")
def bench() -> pb.Benchmark:
    return pb.Benchmark(DATA)


@pytest.fixture(scope="module")
def model() -> pb.OriginalModel:
    if MODEL is None:
        pytest.skip("POSTERIORBENCH_REAL_MODEL is not set")
    pytest.importorskip("torch")
    return pb.OriginalModel.from_pretrained(MODEL)


def test_stop_tokens(tokenizer: Any) -> None:
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in original_model.STOP_TOKENS)
    assert ids == STOP_TOKEN_IDS


def test_task_prompts_render(tokenizer: Any, bench: pb.Benchmark) -> None:
    """`render_chat_prompt` reproduces every pair's stored prompt token ids."""
    mismatched = [
        task.task_id
        for task in bench.tasks("all")
        if original_model.render_chat_prompt(tokenizer, task.prompt) != list(task.prompt_token_ids)
    ]
    assert not mismatched


def test_rules_are_the_package_rules(bench: pb.Benchmark) -> None:
    assert all(task.rule == rules.PAPER_RULES[task.behavior] for task in bench.tasks("all"))


def test_stored_responses_are_complete_hits(tokenizer: Any, bench: pb.Benchmark) -> None:
    """Every reference and every fallback response of the evaluation file is a complete response
    and, decoded with the dataset's tokenizer, a hit of its pair's rule; each reference's stored
    text is that decoding."""
    source = DATA if DATA is not None else data.BUNDLED_SOURCE
    for task in bench.tasks("scored"):
        evaluation = data.load_evaluation_data(source, task.task_id)
        rule = rules.parse_rule(task.rule)
        stored = [r.token_ids for r in evaluation.references]
        stored.append(evaluation.fallback_token_ids)
        for token_ids in stored:
            reason = sampler.invalid_response_reason(
                token_ids,
                stop_token_ids=STOP_TOKEN_IDS,
                max_response_tokens=original_model.MAX_RESPONSE_TOKENS,
            )
            assert reason is None, f"{task.task_id}: {reason}"
            decoded = tokenizer.decode(list(token_ids), skip_special_tokens=True)
            assert rule(decoded), f"{task.task_id}: a stored response is not a hit"
        for reference in evaluation.references:
            decoded = tokenizer.decode(list(reference.token_ids), skip_special_tokens=True)
            assert decoded == reference.text, task.task_id


def test_recomputed_reference_logp_is_close_to_the_shipped(
    model: pb.OriginalModel, bench: pb.Benchmark
) -> None:
    """The log p of the quickstart task's references, computed by this package's model in
    float32, differs from the stored values by a few tenths of a nat per reference."""
    task = bench.task(QUICKSTART_TASK)
    source = DATA if DATA is not None else data.BUNDLED_SOURCE
    evaluation = data.load_evaluation_data(source, task.task_id)
    recomputed = model.logprobs(task.prompt_token_ids, [r.token_ids for r in evaluation.references])
    differences = [abs(a - b) for a, b in zip(recomputed, evaluation.reference_logp, strict=True)]
    print(
        f"|recomputed - shipped| over {len(differences)} references of {task.task_id}: "
        f"mean {statistics.fmean(differences):.4f}, max {max(differences):.4f}"
    )
    assert statistics.fmean(differences) < 0.5 and max(differences) < 2.0


def test_quickstart_with_the_real_model(model: pb.OriginalModel, bench: pb.Benchmark) -> None:
    """The README's quickstart: the Instruction sampler on the quickstart task passes the
    threshold and gets finite scores well below the fallback's forward KL."""
    task = bench.task(QUICKSTART_TASK)
    candidate = pb.instruction_sampler(model, task)
    both = bench.evaluate_both(task, candidate, model, ledger=pb.Ledger(), seed=0, replicate=0)
    for choice, result in both.items():
        print(
            f"{choice}: hits {result.hits}/{result.draws}, forward {result.capped_forward_kl:.4f}, "
            f"reverse {result.reverse_kl:.4f}"
        )
        assert result.passed_threshold and not result.fallback
        assert 0.25 < result.native_hit_rate < 0.55
        assert 10.0 < result.capped_forward_kl < 20.0 and 12.0 < result.reverse_kl < 25.0
    difference = both["recomputed"].capped_forward_kl - both["shipped"].capped_forward_kl
    assert abs(difference) < 0.5
