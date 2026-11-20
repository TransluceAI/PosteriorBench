"""`seeds.stream_seed` and `seeds.substream_seed`: public inputs, any task, usable ranges."""

from __future__ import annotations

import pytest

from posteriorbench_methods.common import seeds


def test_stream_seed_is_deterministic_and_in_the_construction_range() -> None:
    first = seeds.stream_seed("ctl", "wc_1569/em_dash", 0, "threshold-test", 1048576, 1)
    again = seeds.stream_seed("ctl", "wc_1569/em_dash", 0, "threshold-test", 1048576, 1)
    assert first == again == 2125309935  # blake2s of the compact JSON list, top bit cleared
    values = [seeds.stream_seed("sft", "any/task", r, "selection", f"c{i}") for r in range(3)
              for i in range(300)]  # fmt: skip
    assert all(0 <= v < 2**31 for v in values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize(
    "changed",
    [
        ("dpg", "wc_1569/em_dash", 0, "selection", "update_00001"),
        ("sft", "wc_1569/em_dash", 1, "selection", "update_00001"),
        ("sft", "wc_0346/em_dash", 0, "selection", "update_00001"),
        ("sft", "wc_1569/em_dash", 0, "threshold-test", "update_00001"),
        ("sft", "wc_1569/em_dash", 0, "selection", "update_00002"),
        ("sft", "wc_1569/em_dash", None, "selection", "update_00001"),
    ],
)
def test_every_input_changes_the_seed(changed: tuple[str, str, int | None, str, str]) -> None:
    base = seeds.stream_seed("sft", "wc_1569/em_dash", 0, "selection", "update_00001")
    assert seeds.stream_seed(*changed) != base


def test_any_task_id_is_accepted() -> None:
    # Not a task of the paper's seven behaviors: still a seed (no table of task keys).
    assert 0 <= seeds.stream_seed("prompt", "wc_9999/new_behavior", 2, "pool-order") < 2**31


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("", "t", 0, "selection"), "owner"),
        (("m", "", 0, "selection"), "task id"),
        (("m", "t", -1, "selection"), "non-negative"),
        (("m", "t", 0, "Threshold test"), "lowercase"),
        (("m", "t", 0, "selection", 0.5), "str or int"),
        (("m", "t", 0, "selection", True), "str or int"),
    ],
)
def test_bad_inputs_raise(arguments: tuple[object, ...], message: str) -> None:
    with pytest.raises(seeds.SeedError, match=message):
        seeds.stream_seed(*arguments)  # pyright: ignore[reportArgumentType]


def test_substream_seed() -> None:
    assert 0 <= seeds.substream_seed(1234, 128) < 2**32
    assert seeds.substream_seed(1234, 128) != seeds.substream_seed(1234, 0)
    assert seeds.substream_seed(1234, "chunk", 2) != seeds.substream_seed(1234, "chunk", 3)
    assert seeds.substream_seed(1234, 128) == 3630044969  # blake2s of the JSON list "[1234, 128]"
    assert seeds.substream_seed(7, "chunk", 2) == 3796634463
