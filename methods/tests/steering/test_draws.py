"""The draws of a run: hit labels, the selection draws that the replicates share, and the
streams of the threshold tests."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from posteriorbench_methods.common import hf_runtime, run_context, run_record, signals
from posteriorbench_methods.steering import candidates, draws, samplers, vectors

STEERED = candidates.Candidate("with_ablation", 0, 2.0)


@pytest.fixture()
def source(
    make_context: Callable[..., run_context.RunContext],
    vectors_directory: pathlib.Path,
    tmp_path: pathlib.Path,
) -> hf_runtime.ModelSampler:
    """The sampler of a steered candidate."""
    loaded = vectors.load(vectors_directory, "goblin_mention", (0, 1))
    return samplers.sampler(samplers.SteeredModel.of(make_context(tmp_path), loaded), STEERED)


def test_a_response_with_a_token_outside_the_tokenizer_is_a_miss(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path)
    tokenizer = context.original.tokenizer
    a, b, stop = (tokenizer.convert_tokens_to_ids(token) for token in ("a", "b", "<|im_end|>"))
    assert draws.label(context, [[a, stop], [b, stop], [a, len(tokenizer), stop]]) == [
        True,
        False,
        False,
    ]


def test_selection_draws_carry_log_q_and_log_p_of_the_hits(
    make_context: Callable[..., run_context.RunContext],
    source: hf_runtime.ModelSampler,
    tmp_path: pathlib.Path,
) -> None:
    context = make_context(tmp_path, rule='"e" in y')
    rows = draws.selection_draws(context, source, 32, seed=7)
    assert [row["draw_index"] for row in rows] == list(range(32))
    assert {row["seed"] for row in rows} == {7}
    assert 0 < sum(row["hit"] for row in rows) < 32
    assert all(isinstance(row["logq"], float) for row in rows)
    assert all((row["logp"] is not None) == row["hit"] for row in rows)
    arrays = signals.SelectionDraws.from_rows(rows)
    assert arrays.hit.sum() == sum(row["hit"] for row in rows)
    assert arrays.logq.shape == arrays.logp.shape == (32,)


def test_the_replicates_of_a_task_share_their_selection_draws(
    make_context: Callable[..., run_context.RunContext],
    source: hf_runtime.ModelSampler,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    first = draws.shared_selection_draws(
        make_context(out, replicate=0), draws.SELECTION, STEERED.key, source, 16
    )
    stored = out / "cache" / "steering" / "wc_1853__goblin_mention" / "selection"
    assert run_record.read_jsonl(stored / f"{STEERED.key}.jsonl") == first

    def no_generation(*arguments: Any, **options: Any) -> None:
        raise AssertionError("the selection draws of a later replicate are read, not drawn")

    monkeypatch.setattr(hf_runtime, "sample_groups", no_generation)
    again = draws.shared_selection_draws(
        make_context(out, replicate=1), draws.SELECTION, STEERED.key, source, 16
    )
    assert again == first


def test_a_cache_of_other_vectors_or_another_number_of_draws_is_refused(
    make_context: Callable[..., run_context.RunContext],
    write_vectors: Callable[..., pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "out"
    draws.check_cache(make_context(out), 128)
    draws.check_cache(make_context(out, replicate=1), 128)
    with pytest.raises(draws.CacheError, match="another number of selection draws"):
        draws.check_cache(make_context(out), 16)
    other = write_vectors(tmp_path / "other_vectors", seed=7)
    with pytest.raises(draws.CacheError, match="other vectors"):
        draws.check_cache(make_context(out, method_data=other), 128)


def test_a_test_reads_the_first_responses_of_its_sampler_stream(
    make_context: Callable[..., run_context.RunContext],
    source: hf_runtime.ModelSampler,
    tmp_path: pathlib.Path,
) -> None:
    streams = draws.ThresholdTestStreams(make_context(tmp_path, replicate=0))
    short = streams.first(STEERED.key, source, 100)
    longer = streams.first(STEERED.key, source, 164)
    assert len(short) == 100 and len(longer) == 164
    assert longer[:100] == short
    # The stream is drawn in parts of 128 responses, each with a seed of its own.
    assert [row["draw_index"] for row in longer] == [*range(128), *range(36)]
    assert len({row["seed"] for row in longer[:128]}) == 1
    assert longer[0]["seed"] != longer[128]["seed"]
    # Another sampler and another replicate have other streams.
    other = streams.first("another sampler", source, 10)
    assert other[0]["seed"] != short[0]["seed"]
    next_replicate = draws.ThresholdTestStreams(make_context(tmp_path, replicate=1))
    assert next_replicate.first(STEERED.key, source, 10)[0]["seed"] != short[0]["seed"]
