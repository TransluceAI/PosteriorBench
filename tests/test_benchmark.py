"""The `Benchmark` facade: the bundled tables, construction sees no held-out data, construction is
charged, and the evaluation draws never come from a construction stream."""

from __future__ import annotations

import inspect
import math
import pathlib
import random
import re
from collections.abc import Sequence
from typing import Any

import pytest
from support import datasets, hub_stub
from support.datasets import DOCUMENTED_TASK_ID, SMALL_TASK_ID, Synthetic

from posteriorbench import (
    benchmark,
    corpus,
    data,
    ledger,
    model_samplers,
    original_model,
    protocol,
    results,
    sampler,
)

CONSTRUCTION_METHODS = {"tasks", "task", "corpus"}
EVALUATION_METHODS = {"evaluate", "evaluate_both", "no_sampler_result", "fallback_result"}
# The ids of the stop tokens (<|im_end|>, <|endoftext|>) in the dataset's tokenizer.
STOP_TOKEN_IDS = (248046, 248044)
SOURCE_METHODS = {"describe_data"}  # the data source's name, revision and commit only


def public_methods(cls: type) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls, inspect.isfunction) if not name.startswith("_")
    }


def test_task_subsets(tiny_dataset: pathlib.Path) -> None:
    bench = benchmark.Benchmark(tiny_dataset)
    assert [t.task_id for t in bench.tasks()] == [DOCUMENTED_TASK_ID]  # the benchmark's tasks
    assert bench.tasks() == bench.tasks("benchmark")
    assert [t.task_id for t in bench.tasks("scored")] == [DOCUMENTED_TASK_ID]
    assert [t.task_id for t in bench.tasks("all")] == [DOCUMENTED_TASK_ID, "wc_1853/profanity"]
    with pytest.raises(ValueError, match="subset"):
        bench.tasks("main")  # pyright: ignore[reportArgumentType]
    assert bench.task(DOCUMENTED_TASK_ID).scored
    with pytest.raises(data.DataError):
        bench.task("wc_0000/em_dash")


def test_the_default_subset_is_the_benchmark_tasks(tmp_path: pathlib.Path) -> None:
    """A scored pair that is not a benchmark task is in "scored" and "all" only."""
    records = datasets.small_task_records()
    records[0]["benchmark_task"] = False
    datasets.write_tasks(tmp_path, records)
    bench = benchmark.Benchmark(tmp_path)
    assert bench.tasks() == []
    assert [t.task_id for t in bench.tasks("scored")] == [SMALL_TASK_ID]
    assert len(bench.tasks("all")) == 2


def test_bundled_data_loads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Benchmark()` reads the task table and the evaluation file shipped inside the package: 140
    pairs, 108 scored, 21 benchmark tasks, and evaluation data for every scored pair, with no Hub
    call."""
    import huggingface_hub

    def no_download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("the bundled data must not need the Hub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    bench = benchmark.Benchmark()
    everything, scored, tasks = bench.tasks("all"), bench.tasks("scored"), bench.tasks()
    assert (len(everything), len(scored), len(tasks)) == (140, 108, 21)
    assert all(task.scored for task in tasks) and all(task in scored for task in tasks)
    assert bench.describe_data() == {
        "source": "bundled",
        "revision": data.DATASET_REVISION,
        "commit": None,
    }
    for task in scored:
        evaluation = data.load_evaluation_data(data.BUNDLED_SOURCE, task.task_id)
        assert len(evaluation.references) == len(evaluation.reference_logp) == 250
        assert evaluation.fallback_logp < task.log_event_probability
        # Every stored response is a complete response of the original model.
        stored = [r.token_ids for r in evaluation.references] + [evaluation.fallback_token_ids]
        for response in stored:
            assert (
                sampler.invalid_response_reason(
                    response,
                    stop_token_ids=STOP_TOKEN_IDS,
                    max_response_tokens=original_model.MAX_RESPONSE_TOKENS,
                )
                is None
            )
    unscored = next(task for task in everything if not task.scored)
    with pytest.raises(data.DataError, match="only scored pairs"):
        data.load_evaluation_data(data.BUNDLED_SOURCE, unscored.task_id)
    # No model is needed for the fallback's scores, the score every method must beat.
    fallback = bench.fallback_result(bench.task("wc_1853/goblin_mention"))
    assert fallback.fallback and fallback.dataset == bench.describe_data()
    # Only the two tables are bundled: corpus hits come from the Hub or a local copy.
    assert sorted(p.name for p in data.BUNDLED_SOURCE.iterdir() if p.is_dir()) == [
        "evaluation",
        "tasks",
    ]


def test_hits_source(
    tiny_dataset: pathlib.Path, synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Benchmark()` takes its tables from the package and its hit files from `hits_source` (the
    Hub dataset by default); `Benchmark(source)` takes both from `source`."""
    assert benchmark.Benchmark().hits_source == data.HUB_DATASET
    assert benchmark.Benchmark(tiny_dataset).hits_source == tiny_dataset
    monkeypatch.setattr(data, "BUNDLED_SOURCE", tiny_dataset / "tables_only")
    for file in (data.TASKS_FILE, data.EVALUATION_FILE):
        (data.BUNDLED_SOURCE / file).parent.mkdir(parents=True)
        (data.BUNDLED_SOURCE / file).write_bytes((tiny_dataset / file).read_bytes())
    bench = benchmark.Benchmark(hits_source=tiny_dataset)
    task = bench.task(DOCUMENTED_TASK_ID)
    spend = ledger.Ledger()
    assert len(bench.corpus(task, spend).take_first(3)) == 3
    assert spend.inferences == pytest.approx(3 / task.probability)
    requested = hub_stub.install_offline_hub(monkeypatch, tiny_dataset)
    assert len(benchmark.Benchmark().corpus(task, ledger.Ledger()).take_first(3)) == 3
    # Nothing but the hit file is read from the Hub, at the dataset version the package reads.
    assert requested == [(data.hits_file(task.prompt_id, task.behavior), data.DATASET_REVISION)]
    with pytest.raises(ValueError, match="revision"):
        benchmark.Benchmark(revision="v1.0.0", hits_source=tiny_dataset)


def test_creating_a_benchmark_reads_nothing(tmp_path: pathlib.Path) -> None:
    bench = benchmark.Benchmark(tmp_path / "does-not-exist-yet")  # no error until first use
    with pytest.raises(Exception):
        bench.tasks()


def test_references_are_not_reachable_from_construction_apis(
    tiny_dataset: pathlib.Path, synthetic: Synthetic
) -> None:
    """The construction side (`tasks`, `task`, `corpus`) never hands out a reference, the
    fallback response or a stored log p; only the evaluation methods read them."""
    assert (
        public_methods(benchmark.Benchmark)
        == CONSTRUCTION_METHODS | EVALUATION_METHODS | SOURCE_METHODS
    )
    for name in public_methods(benchmark.Benchmark):
        annotation = str(inspect.signature(getattr(benchmark.Benchmark, name)).return_annotation)
        assert not re.search(r"\b(EvaluationData|Reference)\b", annotation), name
    bench = benchmark.Benchmark(tiny_dataset)
    task = bench.task(DOCUMENTED_TASK_ID)
    assert not any(hasattr(task, n) for n in ("references", "fallback_token_ids", "fallback_logp"))
    # The corpus access: no public attribute holds held-out data, and taking every hit returns no
    # reference row and (by default) no response equal to a reference.
    hits = bench.corpus(task, ledger.Ledger())
    public = {n for n in dir(hits) if not n.startswith("_")}
    assert {n for n in public if "reference" in n} == {
        "exclude_reference_duplicates",  # a flag
        "reference_duplicates_taken",  # a count
    }
    for value in vars(hits).values():
        assert not isinstance(value, (data.EvaluationData, data.Reference))
    taken = hits.take_first(hits.available)
    references = synthetic.evaluation.references
    assert not {h.row for h in taken} & {r.row for r in references}
    assert not {h.token_ids for h in taken} & {r.token_ids for r in references}
    assert hits.reference_duplicates_taken == 0
    # The tiny model's frequent short responses do repeat references: the exclusion is real.
    keep = bench.corpus(task, ledger.Ledger(), exclude_reference_duplicates=False)
    assert keep.available > hits.available
    assert not {h.row for h in keep.take_first(keep.available)} & {r.row for r in references}


def test_evaluation_charges_construction_through_the_ledger(
    tiny_dataset: pathlib.Path, tiny_model: Any
) -> None:
    bench = benchmark.Benchmark(tiny_dataset)
    task = bench.task(DOCUMENTED_TASK_ID)
    spend = ledger.Ledger()
    bank = bench.corpus(task, spend).take_first(16)
    assert len(bank) == 16
    spend.charge(ledger.Category.THRESHOLD_TEST, inferences=164)
    candidate = model_samplers.original_model_sampler(tiny_model, task)
    result = bench.evaluate(task, candidate, tiny_model, ledger=spend, seed=0, replicate=0)
    assert result.construction_inferences == pytest.approx(16 / task.probability + 164)
    assert result.ledger is not None
    assert result.ledger["by_category"]["corpus_hits"][0] == pytest.approx(16 / task.probability)
    both = bench.evaluate_both(task, candidate, tiny_model, ledger=spend, seed=0, replicate=0)
    assert both["shipped"].to_dict() == result.to_dict()
    fallback = bench.no_sampler_result(task, budget=4096, seed=0, replicate=0, method="sft")
    assert fallback.construction_inferences == 4096 and fallback.fallback
    assert bench.fallback_result(task).fallback


class RecordingSampler(sampler.Sampler):
    """Wraps a sampler and keeps the seed of every `sample` call."""

    def __init__(self, inner: sampler.Sampler) -> None:
        self.inner = inner
        self.seeds: list[int] = []
        self.name = inner.name

    def sample(self, n: int, *, seed: int) -> list[list[int]]:
        self.seeds.append(seed)
        return self.inner.sample(n, seed=seed)

    def logprobs(self, responses: Sequence[Sequence[int]]) -> list[float]:
        return self.inner.logprobs(responses)


def test_evaluation_draws_use_the_evaluation_stream_only(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    candidate = RecordingSampler(model_samplers.original_model_sampler(tiny_model, common_task))
    zero = ledger.LedgerTotals.zero()
    for seed in (0, 1, 2**31 - 1):
        protocol.evaluate_both(
            common_task, common_evaluation, candidate, tiny_model, ledger=zero, seed=seed,
            replicate=0, settings=results.EvaluationSettings(draws=64),
        )  # fmt: skip
    # One sample call per evaluation (both results share it), each from [2**31, 2**32).
    assert candidate.seeds == [2**31, 2**31 + 1, 2**32 - 1]
    for bad in (-1, 2**31):
        with pytest.raises(ValueError):
            protocol.evaluate(
                common_task, common_evaluation, candidate, tiny_model, ledger=zero, seed=bad,
                replicate=0,
            )  # fmt: skip


def test_construction_and_evaluation_streams_are_disjoint() -> None:
    """Construction streams fill [0, 2**31) and evaluation streams [2**31, 2**32); every seed fits
    in 32 bits, the part of a seed that torch's CPU generator uses."""
    assert sampler.construction_stream_seed(0, 0) == 0
    assert sampler.construction_stream_seed(5, 3) == 8
    assert sampler.construction_stream_seed(2**31 - 2, 1) == 2**31 - 1
    assert sampler.evaluation_stream_seed(0) == 2**31
    assert sampler.evaluation_stream_seed(2**31 - 1) == 2**32 - 1
    for seed in (-1, 2**31):
        with pytest.raises(ValueError):
            sampler.evaluation_stream_seed(seed)
    for seed, index in ((-1, 0), (0, -1), (2**31 - 1, 1)):
        with pytest.raises(ValueError):
            sampler.construction_stream_seed(seed, index)
    rng = random.Random(0)
    construction = set[int]()
    evaluation = set[int]()
    for _ in range(10_000):
        seed = rng.randrange(2**31)
        construction.add(sampler.construction_stream_seed(seed, rng.randrange(2**31 - seed)))
        evaluation.add(sampler.evaluation_stream_seed(seed))
    assert max(construction) < 2**31 <= min(evaluation) and max(evaluation) < 2**32


def test_evaluation_never_reuses_construction_draws(
    tiny_model: Any, common_task: data.Task, common_evaluation: data.EvaluationData
) -> None:
    """Construction draws with seed 0 and an evaluation with seed 0 come from different streams,
    so the evaluation draws are new responses."""
    candidate = RecordingSampler(model_samplers.original_model_sampler(tiny_model, common_task))
    construction_draws = candidate.sample(164, seed=sampler.construction_stream_seed(0, 0))
    result = protocol.evaluate(
        common_task, common_evaluation, candidate, tiny_model, ledger=ledger.LedgerTotals.zero(),
        seed=0, replicate=0, keep_details=True,
    )  # fmt: skip
    assert result.details is not None
    evaluation_draws = result.details["draw_token_ids"]
    shared = 0
    for construction_draw, evaluation_draw in zip(construction_draws, evaluation_draws):
        if construction_draw != evaluation_draw:
            break
        shared += 1
    assert shared < 10, f"evaluation repeated {shared} construction draws"
    assert evaluation_draws[:164] != construction_draws
    assert candidate.seeds == [0, 2**31] and result.sampler_seed == 2**31


def test_corpus_access_is_construction_only(tiny_dataset: pathlib.Path) -> None:
    """`CorpusAccess` is the only way the package hands corpus hits to a method, and it charges
    them; it never draws from a sampler, so it cannot touch the evaluation streams."""
    bench = benchmark.Benchmark(tiny_dataset)
    task = bench.task(DOCUMENTED_TASK_ID)
    spend = ledger.Ledger()
    hits = bench.corpus(task, spend)
    assert isinstance(hits, corpus.CorpusAccess)
    hits.take_first(3)
    assert spend.inferences == pytest.approx(3 / task.probability)
    assert math.isclose(hits.cost_of(3), spend.inferences)


def test_a_task_id_or_a_bare_model_is_refused_with_a_hint(
    tiny_dataset: pathlib.Path, tiny_model: Any
) -> None:
    bench = benchmark.Benchmark(tiny_dataset)
    task = bench.task(DOCUMENTED_TASK_ID)
    candidate = model_samplers.original_model_sampler(tiny_model, task)
    wrong_task: Any = DOCUMENTED_TASK_ID
    with pytest.raises(TypeError, match=r"bench\.task\("):
        bench.evaluate(wrong_task, candidate, tiny_model, ledger=ledger.Ledger(), seed=0)
    with pytest.raises(TypeError, match=r"bench\.task\("):
        bench.corpus(wrong_task, ledger.Ledger())
    with pytest.raises(TypeError, match=r"bench\.task\("):
        bench.fallback_result(wrong_task)
    bare_model: Any = object()
    for evaluate in (bench.evaluate, bench.evaluate_both):
        with pytest.raises(TypeError, match="pb.OriginalModel"):
            evaluate(task, candidate, bare_model, ledger=ledger.Ledger(), seed=0)


def test_results_record_the_data_source(tiny_dataset: pathlib.Path) -> None:
    bench = benchmark.Benchmark(tiny_dataset)
    task = bench.task(DOCUMENTED_TASK_ID)
    result = bench.fallback_result(task)
    assert result.dataset == {"source": str(tiny_dataset), "revision": None, "commit": None}
    cached = pathlib.Path(
        "/cache/datasets--Transluce--PosteriorBench/snapshots/abc123/tasks/tasks.parquet"
    )
    assert data.hub_commit(cached) == "abc123"
    assert data.hub_commit(tiny_dataset / "tasks" / "tasks.parquet") is None
