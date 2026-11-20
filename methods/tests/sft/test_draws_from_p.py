"""SFT's draws from the original model: the places of the hits among the draws and the corpus
hits that arrive."""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import pytest
from posteriorbench import benchmark, data, ledger

from posteriorbench_methods.sft import draws_from_p


@pytest.fixture(scope="module")
def bench(tiny_dataset: pathlib.Path) -> benchmark.Benchmark:
    return benchmark.Benchmark(tiny_dataset)


def draws_of(
    bench: benchmark.Benchmark, task: data.Task, *, arrival_seed: int = 0, order_seed: int = 0
) -> draws_from_p.DrawsFromP:
    access = bench.corpus(task, ledger.Ledger())
    return draws_from_p.DrawsFromP(access, arrival_seed=arrival_seed, order_seed=order_seed)


def test_places_of_the_hits_do_not_depend_on_how_they_are_read(
    bench: benchmark.Benchmark, tiny_task: data.Task
) -> None:
    far, near = draws_of(bench, tiny_task), draws_of(bench, tiny_task)
    last = far.arrival(9000)
    places = [near.arrival(j) for j in range(1, 9001)]
    assert places[-1] == last
    assert all(a < b for a, b in zip(places, places[1:], strict=False))
    assert draws_of(bench, tiny_task, arrival_seed=1).arrival(50) != far.arrival(50)
    with pytest.raises(ValueError):
        far.arrival(0)


def test_hits_among_n_draws_are_binomial(bench: benchmark.Benchmark, tiny_task: data.Task) -> None:
    draws, rate = 20_000, tiny_task.probability
    counts: list[int] = []
    for seed in range(30):
        stream = draws_of(bench, tiny_task, arrival_seed=seed)
        hits = 0
        while stream.arrival(hits + 1) <= draws:
            hits += 1
        counts.append(hits)
    standard_error = math.sqrt(draws * rate * (1 - rate) / len(counts))
    assert abs(float(np.mean(counts)) - draws * rate) < 4 * standard_error


def test_the_hits_are_distinct_corpus_hits_outside_the_references(
    bench: benchmark.Benchmark, tiny_task: data.Task, tiny_synthetic: Any
) -> None:
    stream = draws_of(bench, tiny_task)
    available = bench.corpus(tiny_task, ledger.Ledger()).available
    hits = stream.take(available)
    assert len({hit.row for hit in hits}) == available
    references = {reference.row for reference in tiny_synthetic.evaluation.references}
    assert not references & {hit.row for hit in hits}
    assert all(tiny_synthetic.labels[hit.row] for hit in hits)
    assert [hit.row for hit in stream.take(10)] == [hit.row for hit in hits[:10]]
    other_order = draws_of(bench, tiny_task, order_seed=1).take(available)
    assert [hit.row for hit in other_order] != [hit.row for hit in hits]
    assert sorted(hit.row for hit in other_order) == sorted(hit.row for hit in hits)
    with pytest.raises(draws_from_p.CorpusExhausted):
        stream.take(available + 1)
