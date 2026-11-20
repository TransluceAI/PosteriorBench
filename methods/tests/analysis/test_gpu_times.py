"""The GPU time of a run from measured times: the paper's times file, and runs priced by hand."""

from __future__ import annotations

import pathlib

import pytest

from posteriorbench_methods.analysis import flops, gpu_times, operations
from posteriorbench_methods.common import run_record

KINDS: tuple[flops.Trains, ...] = ("nothing", "last_layers", "adapters", "twist", "all_weights")
Entry = tuple[gpu_times.Operation, flops.Trains, int, float]  # operation, model, batch, seconds


def session(*entries: Entry) -> gpu_times.Session:
    return gpu_times.Session({}, tuple(gpu_times.Measurement(*entry, 1) for entry in entries))


# One response: 0.01 s to generate 500 at a time, 0.02 s 128 at a time, 0.005 s to score.
SHARED: tuple[Entry, ...] = (
    ("generation", "nothing", 500, 5.0),
    ("generation", "nothing", 128, 2.56),
    ("forward", "nothing", 32, 0.16),
)


def test_the_papers_times_price_every_method() -> None:
    times = gpu_times.read()
    first, second = times.sessions
    for kind in KINDS:
        assert times.session_for(kind).has(gpu_times.needed(kind))
    # SFT's updates were measured in the second session, which prices SFT throughout.
    assert times.session_for("last_layers") is second
    assert {times.session_for(kind) is first for kind in ("adapters", "twist", "all_weights")} == {
        True
    }
    for one in (first, second):
        assert one.conditions["gpu_name"].startswith("NVIDIA H100")
        assert (one.conditions["prompt_tokens"], one.conditions["response_tokens"]) == (24, 50)
        assert one.unit == pytest.approx(0.0103, abs=1e-4)  # seconds to generate one response
    assert {
        key for one in times.sessions for key in gpu_times.every_key() if one.has([key])
    } == set(gpu_times.every_key())


def test_the_measurements_follow_the_batch_sizes_of_the_methods() -> None:
    assert gpu_times.needed("nothing") == [
        ("generation", "nothing", 500),
        ("forward", "nothing", 32),
    ]
    assert ("generation", "nothing", 128) in gpu_times.needed("all_weights")
    assert ("generation", "adapters", 500) in gpu_times.needed("adapters")
    assert ("training_forward_backward", "adapters", 20) in gpu_times.needed("adapters")
    assert ("generation", "twist", 64) in gpu_times.needed("twist")
    assert ("update", "twist", 1000) in gpu_times.needed("twist")
    updates = [key for key in gpu_times.needed("last_layers") if key[0] == "update"]
    assert updates == [("update", "last_layers", batch) for batch in range(1, 33)]


def test_a_run_of_a_method_that_trains_all_weights_priced_by_hand() -> None:
    times = session(
        *SHARED,
        ("training_forward", "all_weights", 32, 0.192),  # 0.006 s a response
        ("training_forward_backward", "all_weights", 32, 0.672),  # 0.021 s a response
        ("optimizer_step", "all_weights", 0, 0.02),
    )
    done = operations.Operations(
        sampler_draws=256, scored=10, training_forwards=256, gradient_responses=256, updates=1
    )
    expected = 256 * 0.02 + 10 * 0.005 + 256 * 0.006 + 256 * 0.015 + 0.02
    assert gpu_times.run_seconds("all_weights", done, times) == pytest.approx(expected)
    assert times.unit == pytest.approx(0.01)


def test_a_run_of_sft_is_priced_by_its_updates() -> None:
    """70 training responses in batches of 32: two updates on 32 and one on the 6 that remain."""
    updates: list[Entry] = [
        ("update", "last_layers", batch, 0.1 + 0.01 * batch) for batch in range(1, 33)
    ]
    times = session(*SHARED, *updates)
    done = operations.Operations(
        draws_from_p=1000, sampler_draws=128, scored=5, training_forwards=70,
        gradient_responses=70, updates=3, update_batch=32,
    )  # fmt: skip
    expected = 1000 * 0.01 + 128 * 0.02 + 5 * 0.005 + 2 * 0.42 + 0.16
    assert gpu_times.run_seconds("last_layers", done, times) == pytest.approx(expected)
    no_hit = operations.Operations(draws_from_p=300)
    assert gpu_times.run_seconds("last_layers", no_hit, times) == pytest.approx(3.0)


def test_a_run_of_ctl_and_of_a_method_that_trains_nothing() -> None:
    times = session(*SHARED, ("generation", "twist", 64, 1.28), ("update", "twist", 1000, 0.03))
    ctl = operations.Operations(sampler_draws=1000, scored=1000, gradient_responses=1000)
    assert gpu_times.run_seconds("twist", ctl, times) == pytest.approx(20 + 5 + 0.03)
    replay = operations.Operations(draws_from_p=400)
    assert gpu_times.run_seconds("nothing", replay, times) / times.unit == pytest.approx(400)
    selection = operations.Operations(draws_from_p=300, scored=100)
    assert gpu_times.run_seconds("nothing", selection, times) / times.unit == pytest.approx(350)


def test_a_method_is_priced_from_the_last_session_that_has_its_measurements() -> None:
    first = session(*SHARED, ("generation", "twist", 64, 1.28), ("update", "twist", 1000, 0.03))
    second = session(("generation", "nothing", 500, 6.0), ("forward", "nothing", 32, 0.16))
    times = gpu_times.Times((first, second))
    assert times.session_for("nothing") is second
    assert times.session_for("twist") is first
    with pytest.raises(gpu_times.TimesError, match="all_weights"):
        times.session_for("all_weights")
    with pytest.raises(gpu_times.TimesError, match="no measurement of update"):
        second.seconds(("update", "twist", 1000))


def test_a_times_file_is_read_as_written(tmp_path: pathlib.Path) -> None:
    one = session(*SHARED)
    gpu_times.write(tmp_path / "times.json", [one, one])
    assert gpu_times.read(tmp_path / "times.json") == gpu_times.Times((one, one))
    with pytest.raises(gpu_times.TimesError, match="does not exist"):
        gpu_times.read(tmp_path / "missing.json")
    run_record.write_json(tmp_path / "other.json", {"records": []})
    with pytest.raises(gpu_times.TimesError, match="not a times file"):
        gpu_times.read(tmp_path / "other.json")
