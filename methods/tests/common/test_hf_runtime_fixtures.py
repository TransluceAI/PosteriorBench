"""The shared test fixtures (`tests/conftest.py`): the tiny dataset is read by the package's own
readers, and the tiny models are what the model layer expects."""

from __future__ import annotations

import pathlib
from typing import Any

from posteriorbench import benchmark, corpus, data, ledger, original_model, protocol


def test_tiny_dataset_reads_back(
    tiny_dataset: pathlib.Path,
    tiny_synthetic: Any,
    tiny_task: data.Task,
    tiny_original_model: original_model.OriginalModel,
) -> None:
    (task,) = data.load_tasks(tiny_dataset)
    assert task == tiny_task and task.scored and task.benchmark_task
    assert task.probability == sum(tiny_synthetic.labels) / task.num_responses
    evaluation = data.load_evaluation_data(tiny_dataset, task.task_id)
    assert evaluation == tiny_synthetic.evaluation
    references = {r.token_ids for r in evaluation.references}
    reference_rows = {r.row for r in evaluation.references}
    usable = [
        response
        for row, (response, hit) in enumerate(
            zip(tiny_synthetic.responses, tiny_synthetic.labels, strict=True)
        )
        if hit and row not in reference_rows and response not in references
    ]
    access = corpus.CorpusAccess(tiny_dataset, task, ledger.Ledger())
    assert access.available == len(usable)
    taken = access.take(range(5))
    assert [h.token_ids for h in taken] == usable[:5]
    assert all(protocol.label(task, tiny_original_model, [h.token_ids for h in taken]))
    assert benchmark.Benchmark(str(tiny_dataset)).task(task.task_id) == task


def test_tiny_models(
    tiny_model: Any,
    tiny_hybrid_model: Any,
    tiny_tokenizer: Any,
    tiny_original_model: original_model.OriginalModel,
) -> None:
    assert tiny_original_model.model is tiny_model
    assert tiny_original_model.stop_token_ids == (
        tiny_tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tiny_tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    )
    assert tiny_original_model.max_response_tokens == 8
    assert len(tiny_hybrid_model.model.layers) == 24
    assert not tiny_model.training and not tiny_hybrid_model.training


def test_factories(make_tiny_original_model: Any, make_tiny_dataset: Any) -> None:
    """A 50-token scorer and a dataset of it (the benchmark's response limit)."""
    scorer = make_tiny_original_model(max_response_tokens=50, batch_size=64)
    assert scorer.max_response_tokens == 50 and scorer.batch_size == 64
    root = make_tiny_dataset(scorer, task_id="wc_0002/em_dash", draws=1500)
    (task,) = data.load_tasks(root)
    assert task.task_id == "wc_0002/em_dash" and task.num_responses == 1500
    evaluation = data.load_evaluation_data(root, task.task_id)
    assert max(len(r.token_ids) for r in evaluation.references) > 8
