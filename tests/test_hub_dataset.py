"""Checks the published dataset on the Hugging Face Hub against the paper's counts, this package's
loader and the tables bundled with the package.

Skips if the dataset cannot be read, for example without network access; with
POSTERIORBENCH_REQUIRE_HUB=1, as in CI, it fails instead. Reads the task table, the evaluation file
and the hit file of the smallest scored pair only (never the corpus).
"""

from __future__ import annotations

import collections
import itertools
import os

import pytest

from posteriorbench import benchmark, data, ledger, rules

HUB_DATASET = data.HUB_DATASET
# 20 prompts x 7 behaviors = 140 prompt-behavior pairs; the 108 with at least 500 hits are scored,
# and 21 of these are the benchmark's tasks.
# Set POSTERIORBENCH_REQUIRE_HUB=1 to make an unreadable Hub dataset fail instead of skip.
PAIRS = 140
PROMPTS = 20
BENCHMARK_TASKS = 21
SCORED_PAIRS_PER_BEHAVIOR = {
    "em_dash": 20,
    "chinese_character": 20,
    "chatgpt_mention": 12,
    "profanity": 9,
    "goblin_mention": 8,
    "contrastive_phrasing": 19,
    "word_repetition": 20,
}
MIN_HITS = 500
REFERENCES = 250
HITS_CHECKED = 200


@pytest.fixture(scope="session")
def hub_tasks() -> list[data.Task]:
    try:
        return data.load_tasks(HUB_DATASET)
    except (OSError, data.DataError) as error:  # not found, or no network
        if os.environ.get("POSTERIORBENCH_REQUIRE_HUB") == "1":
            raise
        pytest.skip(f"cannot read {HUB_DATASET} from the Hub: {error}")


def test_tasks_match_the_paper(hub_tasks: list[data.Task]) -> None:
    assert len(hub_tasks) == PAIRS
    assert len({task.prompt_id for task in hub_tasks}) == PROMPTS
    assert all(task.rule == rules.PAPER_RULES[task.behavior] for task in hub_tasks)
    scored = [task for task in hub_tasks if task.scored]
    assert collections.Counter(task.behavior for task in scored) == SCORED_PAIRS_PER_BEHAVIOR
    assert all(task.hit_count >= MIN_HITS for task in scored)
    assert not any(task.hit_count >= MIN_HITS for task in hub_tasks if not task.scored)
    assert not any(task.benchmark_task for task in hub_tasks if not task.scored)


def test_benchmark_tasks_are_lowest_median_highest(hub_tasks: list[data.Task]) -> None:
    """The benchmark's 21 tasks are, for each behavior, the scored pairs with the lowest, median
    and highest p̂(S); with an even number of scored pairs the median is the upper middle one."""
    scored = sorted((task for task in hub_tasks if task.scored), key=lambda task: task.behavior)
    assert sum(task.benchmark_task for task in scored) == BENCHMARK_TASKS
    for behavior, group in itertools.groupby(scored, key=lambda task: task.behavior):
        ranked = sorted(group, key=lambda task: task.hit_count)
        chosen = [task for task in ranked if task.benchmark_task]
        assert len(chosen) == 3, behavior
        assert chosen[0] is ranked[0] and chosen[2] is ranked[-1], behavior
        assert chosen[1] is ranked[len(ranked) // 2], behavior


def test_every_scored_pair_has_evaluation_data(hub_tasks: list[data.Task]) -> None:
    for task in hub_tasks:
        if task.scored:
            evaluation = data.load_evaluation_data(HUB_DATASET, task.task_id)
            assert len(evaluation.references) == len(evaluation.reference_logp) == REFERENCES
            assert evaluation.fallback_logp < task.log_event_probability


def test_bundled_tables_equal_the_dataset(hub_tasks: list[data.Task]) -> None:
    """The task table and the evaluation file shipped inside the package hold what the dataset
    holds at the revision this package reads."""
    assert data.load_tasks(data.BUNDLED_SOURCE) == hub_tasks
    for task in hub_tasks:
        if task.scored:
            assert data.load_evaluation_data(
                data.BUNDLED_SOURCE, task.task_id
            ) == data.load_evaluation_data(HUB_DATASET, task.task_id)


def test_hit_file_of_the_smallest_scored_pair(hub_tasks: list[data.Task]) -> None:
    task = min((t for t in hub_tasks if t.scored), key=lambda t: t.hit_count)
    spend = ledger.Ledger()
    hits = benchmark.Benchmark(HUB_DATASET).corpus(task, spend)
    assert hits.available <= task.hit_count - REFERENCES
    taken = hits.take_first(min(HITS_CHECKED, hits.available))
    assert all(1 <= len(hit.token_ids) <= 50 for hit in taken)
    assert spend.inferences == pytest.approx(len(taken) / task.probability)
