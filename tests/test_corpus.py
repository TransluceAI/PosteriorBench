"""Corpus access for construction: charged 1/p̂(S) per distinct hit, never a reference."""

from __future__ import annotations

import pathlib

import pyarrow.parquet as pq
import pytest
from support import datasets, hub_stub
from support.datasets import SMALL_DUPLICATE_ROWS, SMALL_TASK_ID

from posteriorbench import corpus, data, ledger

REFERENCE_ROWS = set(range(0, 500, 2))
PROBABILITY = datasets.SMALL_HIT_COUNT / datasets.SMALL_ROWS


def access(
    source: pathlib.Path | str,
    spend: ledger.Ledger | None = None,
    *,
    exclude_reference_duplicates: bool = True,
) -> corpus.CorpusAccess:
    task = data.get_task(data.load_tasks(source), SMALL_TASK_ID)
    return corpus.CorpusAccess(
        source,
        task,
        spend or ledger.Ledger(),
        exclude_reference_duplicates=exclude_reference_duplicates,
    )


def test_take_charges_once_per_distinct_hit(small_dataset: pathlib.Path) -> None:
    spend = ledger.Ledger()
    hits = access(small_dataset, spend)
    assert hits.available == 348  # 350 non-reference hits minus the 2 reference duplicates
    assert hits.cost_of(10) == pytest.approx(10 / PROBABILITY)
    assert spend.inferences == 0  # cost_of charges nothing
    first = hits.take_first(5)
    assert [h.row for h in first] == [500, 502, 504, 506, 508]
    assert [h.position for h in first] == [0, 1, 2, 3, 4]
    assert first[0].token_ids == tuple(datasets.small_token_ids(500))
    assert spend.inferences == pytest.approx(5 / PROBABILITY)
    again = hits.take([4, 5, 5, 3])  # positions 3 and 4 were taken; 5 is new, requested twice
    assert [h.row for h in again] == [508, 510, 510, 506]
    assert hits.hits_taken == 6
    assert spend.inferences == pytest.approx(6 / PROBABILITY)
    assert spend.budget == spend.inferences
    totals = spend.totals()
    assert totals.by_category["corpus_hits"][0] == pytest.approx(6 / PROBABILITY)
    with pytest.raises(corpus.CorpusError):
        hits.take([348])
    with pytest.raises(corpus.CorpusError):
        hits.take_first(349)


def test_references_and_their_duplicates_are_never_handed_out(small_dataset: pathlib.Path) -> None:
    reference_ids = {r.token_ids for r in datasets.small_evaluation_data().references}
    hits = access(small_dataset).take_first(348)
    assert not {h.row for h in hits} & REFERENCE_ROWS
    assert not {h.token_ids for h in hits} & reference_ids
    assert not set(SMALL_DUPLICATE_ROWS) & {h.row for h in hits}


def test_duplicates_can_be_kept(small_dataset: pathlib.Path) -> None:
    spend = ledger.Ledger()
    hits = access(small_dataset, spend, exclude_reference_duplicates=False)
    assert hits.available == 350
    taken = hits.take_first(350)
    assert {h.row for h in taken} >= set(SMALL_DUPLICATE_ROWS)
    assert not {h.row for h in taken} & REFERENCE_ROWS  # references themselves never
    assert hits.reference_duplicates_taken == 2
    assert spend.inferences == pytest.approx(350 / PROBABILITY)


def test_take_rows(small_dataset: pathlib.Path) -> None:
    spend = ledger.Ledger()
    hits = access(small_dataset, spend)
    taken = hits.take_rows([1198, 500, 1100])
    assert [h.row for h in taken] == [1198, 500, 1100]
    assert taken[1].position == 0
    assert taken[2].position is None  # a reference duplicate, outside the default sequence
    assert hits.reference_duplicates_taken == 1
    assert spend.inferences == pytest.approx(3 / PROBABILITY)
    hits.take_first(1)  # row 500 again: not charged again
    assert hits.hits_taken == 3 and spend.inferences == pytest.approx(3 / PROBABILITY)
    with pytest.raises(corpus.CorpusError, match="not a hit"):
        hits.take_rows([0])  # a reference row
    with pytest.raises(corpus.CorpusError, match="not a hit"):
        hits.take_rows([501])  # a miss
    assert hits.hits_taken == 3


def test_budget_limit_refuses_before_handing_out(small_dataset: pathlib.Path) -> None:
    spend = ledger.Ledger(budget_limit=3 / PROBABILITY)
    hits = access(small_dataset, spend)
    with pytest.raises(ledger.BudgetExceeded):
        hits.take_first(4)
    assert hits.hits_taken == 0 and spend.inferences == 0
    assert len(hits.take_first(3)) == 3


def test_hits_are_read_across_row_groups(small_dataset: pathlib.Path) -> None:
    """A hit file with several row groups gives the same hits as one with a single group."""
    path = small_dataset / data.hits_file("wc_0001", "goblin_mention")
    expected = access(small_dataset).take([340, 3, 120, 121, 3])
    pq.write_table(pq.read_table(path), path, row_group_size=97)
    assert pq.ParquetFile(path).num_row_groups == 4
    assert access(small_dataset).take([340, 3, 120, 121, 3]) == expected


def test_hit_file_of_another_task_is_refused(small_dataset: pathlib.Path) -> None:
    task = data.get_task(data.load_tasks(small_dataset), SMALL_TASK_ID)
    path = small_dataset / data.hits_file("wc_0001", "goblin_mention")
    table = pq.read_table(path)
    metadata = {**(table.schema.metadata or {}), b"task_id": b"wc_9999/goblin_mention"}
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(corpus.CorpusError, match="belongs to"):
        corpus.CorpusAccess(small_dataset, task, ledger.Ledger())
    unscored = data.get_task(data.load_tasks(small_dataset), "wc_0001/profanity")
    with pytest.raises(data.DataError):
        corpus.CorpusAccess(small_dataset, unscored, ledger.Ledger())


def test_a_file_of_another_layout_is_refused(small_dataset: pathlib.Path) -> None:
    task = data.get_task(data.load_tasks(small_dataset), SMALL_TASK_ID)
    path = small_dataset / data.hits_file("wc_0001", "goblin_mention")
    pq.write_table(pq.read_table(path).drop_columns(["reference_duplicate"]), path)
    with pytest.raises(corpus.CorpusError, match="no column"):
        corpus.CorpusAccess(small_dataset, task, ledger.Ledger())


def test_hub_source(small_dataset: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested = hub_stub.install_offline_hub(monkeypatch, small_dataset)
    local = access(small_dataset).take_first(20)
    hub = access(data.HUB_DATASET).take_first(20)
    assert local == hub
    assert [file for file, _ in requested] == [
        data.TASKS_FILE,
        data.hits_file("wc_0001", "goblin_mention"),
    ]
