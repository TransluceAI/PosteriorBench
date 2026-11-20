"""Steering and Steering + composition end to end on the tiny models: the records, their costs,
and what happens when no candidate passes, when the original model is submitted, and when no
candidate has a selection hit."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from posteriorbench import benchmark, original_model

from posteriorbench_methods.common import run_context, run_record, tracking
from posteriorbench_methods.steering import draws, run, settings, vectors

TASK_DIR = "wc_1853__goblin_mention"
STEERABLE = '"e" in y'  # never in the original model's responses; steering brings it in
NEVER = '"xyzxyz" in y'
SETTINGS = settings.Settings(layers=(0, 1), strengths=(1.5, 2.0, 4.0), selection_draws=64)
VECTORS_COST = 16_620 + 3_000 + 190  # the refusal direction, the responses, the kept ones


def read(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def record(out: pathlib.Path, method: str, replicate: int = 0) -> pathlib.Path:
    return out / method / "none" / TASK_DIR / f"rep{replicate}"


def rows(out: pathlib.Path, method: str, name: str, replicate: int = 0) -> list[dict[str, Any]]:
    return run_record.read_jsonl(record(out, method, replicate) / "draws" / f"{name}.jsonl")


@pytest.fixture(scope="module")
def finished(
    make_context: Callable[..., run_context.RunContext], tmp_path_factory: pytest.TempPathFactory
) -> tuple[pathlib.Path, list[run_record.RunSummary]]:
    """The output root of a run in which a steered candidate passes its test and is composed,
    and its summaries; then the run of replicate 1 into the same root."""
    out = tmp_path_factory.mktemp("steering_run")
    summaries = run.run(make_context(out, STEERABLE), SETTINGS)
    run.run(make_context(out, STEERABLE, replicate=1), SETTINGS)
    return out, summaries


def test_a_run_writes_the_records_of_both_methods(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]], vectors_directory: pathlib.Path
) -> None:
    out, summaries = finished
    assert [(s.method, s.setting) for s in summaries] == [
        ("steering", "none"),
        ("steering+composition", "none"),
    ]
    lines = [json.loads(line) for line in (out / "summary.jsonl").read_text().splitlines()]
    assert lines[:2] == [summary.to_json() for summary in summaries]
    for summary in summaries:
        directory = out / summary.record
        assert {"result.json", "result_recomputed.json", "construction.json", "config.json"} <= {
            path.name for path in directory.iterdir()
        }
        config = read(directory / "config.json")
        assert config["family"] == "steering" and config["method"] == summary.method
        assert config["settings"] == {
            "layers": [0, 1],
            "strengths": [1.5, 2.0, 4.0],
            "selection_draws": 64,
            "threshold": 0.1,
        }
        assert config["options"]["method_data"] == str(vectors_directory)
        for path in (directory / "draws").iterdir():
            assert all("token_ids" in row for row in run_record.read_jsonl(path))
    assert {path.stem for path in (record(out, "steering") / "draws").iterdir()} == {
        "selection",
        "threshold_tests",
        "evaluation",
    }


def test_steering_submits_a_candidate_by_the_submission_rule(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, _ = finished
    construction = read(record(out, "steering") / "construction.json")
    details = construction["details"]
    assert set(details) == {"construction_cost", "candidates"}
    listed = details["candidates"]
    assert len(listed) == 13 and listed[0]["candidate_id"] == "original"
    assert listed[1] == {
        "candidate_id": "with_ablation_layer0_strength1.5",
        "vector": "with_ablation",
        "layer": 0,
        "strength": 1.5,
        "selection_hits": listed[1]["selection_hits"],
    }
    selection = rows(out, "steering", "selection")
    assert len(selection) == 13 * 64
    for candidate in listed:
        mine = [row for row in selection if row["candidate_id"] == candidate["candidate_id"]]
        assert len(mine) == 64 and sum(row["hit"] for row in mine) == candidate["selection_hits"]
    # A candidate without a selection hit is not ranked: the original model never writes "e".
    assert listed[0]["selection_hits"] == 0
    ranked = [entry["candidate_id"] for entry in construction["ranking"]]
    assert set(ranked) == {c["candidate_id"] for c in listed if c["selection_hits"] > 0}
    assert all(entry["candidate_id"] == entry["sampler_id"] for entry in construction["ranking"])
    by_signal = [e for e in construction["ranking"] if e["group"] == "by_signal"]
    assert by_signal and all(e["selection_hits"] / 64 > 0.15 for e in by_signal)
    assert [e["score"] for e in by_signal] == sorted((e["score"] for e in by_signal), reverse=True)
    # The first candidate that passes its test is submitted.
    assert construction["status"] == "passed"
    tests = construction["submission"]["tests"]
    assert [test["passed"] for test in tests] == [False] * (len(tests) - 1) + [True]
    submitted = construction["submitted"]
    assert submitted["candidate_id"] == ranked[len(tests) - 1]
    assert submitted["vector"] == "with_ablation" and submitted["strength"] in (1.5, 2.0, 4.0)
    drawn = rows(out, "steering", "threshold_tests")
    assert len(drawn) == sum(test["draws"] for test in tests)
    assert sum(row["hit"] for row in drawn if row["rank"] == submitted["rank"]) == submitted["hits"]


def test_steering_costs_its_vectors_its_selection_draws_and_its_tests(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    construction = read(record(out, "steering") / "construction.json")
    hits = sum(c["selection_hits"] for c in construction["details"]["candidates"])
    expected = VECTORS_COST + 13 * 64 + hits + construction["submission"]["test_draws"]
    totals = construction["ledger"]["totals"]
    assert totals["inferences"] == totals["budget"] == expected
    assert totals["by_category"]["shared"] == [VECTORS_COST, VECTORS_COST]
    assert summaries[0].construction_inferences == expected
    result = read(record(out, "steering") / "result.json")
    assert result["cost_rule"] == "ledger" and result["inferences_per_response"] == 1


def test_the_submitted_candidate_is_composed(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, summaries = finished
    plain = read(record(out, "steering") / "construction.json")
    construction = read(record(out, "steering+composition") / "construction.json")
    assert construction["status"] in ("passed", "best_observed")
    details = construction["details"]
    assert details["plain"] == {"status": "passed", "submitted": plain["submitted"]}
    outcome = details["composition"]
    assert len(outcome["settings"]) == 16 and len(construction["ranking"]) == 16
    selection = rows(out, "steering+composition", "composition_selection")
    assert len(selection) == 16 * 16
    assert sum(row["hit"] for row in selection) == sum(outcome["selection_hits"])
    submitted = construction["submitted"]
    assert submitted["setting"] == outcome["winner"]["setting"]
    # A sampler's name is its candidate's and its setting's; the plain setting is the candidate.
    name = plain["submitted"]["candidate_id"]
    assert {entry["sampler_id"] for entry in construction["ranking"]} == {name} | {
        f"{name}_{s['setting']}" for s in outcome["settings"] if s["setting"] != "a0_t50"
    }
    # Cost: Steering's, the selection draws of the settings, and the tests, at two inferences a
    # draw except for the plain setting.
    tests = construction["submission"]["tests"]
    test_cost = sum(t["draws"] * (1 if t["sampler_id"] == name else 2) for t in tests)
    added = 496 + sum(outcome["selection_hits"]) + test_cost
    ledger = construction["ledger"]
    assert ledger["resumed_from"] == plain["ledger"]["totals"]
    assert ledger["totals"]["inferences"] == plain["ledger"]["totals"]["inferences"] + added
    assert summaries[1].construction_inferences == ledger["totals"]["inferences"]
    result = read(record(out, "steering+composition") / "result.json")
    assert result["method"] == "steering+composition"
    assert result["inferences_per_response"] == (1 if outcome["winner_is_plain"] else 2)
    tested = rows(out, "steering+composition", "composition_threshold_tests")
    assert len(tested) == sum(test["draws"] for test in tests)


def test_replicates_share_selection_draws_and_differ_in_their_tests(
    finished: tuple[pathlib.Path, list[run_record.RunSummary]],
) -> None:
    out, _ = finished
    assert rows(out, "steering", "selection", 1) == rows(out, "steering", "selection", 0)
    cache = out / "cache" / "steering" / TASK_DIR
    assert len(list((cache / "selection").iterdir())) == 13
    first = read(record(out, "steering") / "construction.json")
    second = read(record(out, "steering", 1) / "construction.json")
    assert second["ranking"] == first["ranking"]
    # Each replicate is charged for the selection draws.
    assert (
        second["ledger"]["totals"]["by_category"]["generation"]
        == first["ledger"]["totals"]["by_category"]["generation"]
    )
    tokens = [
        [row["token_ids"] for row in rows(out, "steering", "threshold_tests", replicate)[:164]]
        for replicate in (0, 1)
    ]
    assert tokens[0] != tokens[1]


def test_without_a_selection_hit_nothing_is_submitted(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    chosen = settings.Settings(layers=(0,), strengths=(2.0,), selection_draws=16)
    summaries = run.run(make_context(tmp_path, NEVER), chosen)
    for summary, status in zip(summaries, ("no_candidate", "not_composed"), strict=True):
        assert summary.fallback and summary.kind == "no_sampler"
        # The fallback at the cost that the run spent: the vectors and 3 x 16 selection draws.
        assert summary.construction_inferences == VECTORS_COST + 3 * 16
        directory = tmp_path / summary.record
        construction = read(directory / "construction.json")
        assert construction["status"] == status and construction["submitted"] is None
        assert not (directory / "result_recomputed.json").exists()
    steering = read(record(tmp_path, "steering") / "construction.json")
    assert steering["ranking"] == [] and steering["submission"]["tests"] == []
    composed = read(record(tmp_path, "steering+composition") / "construction.json")
    assert composed["details"]["composition"] == {"reason": "plain_not_passed"}


def test_the_original_model_is_submitted_when_it_ranks_first_and_is_not_composed(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """The task's own behavior is in every response of the original model."""
    chosen = settings.Settings(layers=(0,), strengths=(2.0, 8.0), selection_draws=32)
    plain, composed = run.run(make_context(tmp_path), chosen)
    steering = read(record(tmp_path, "steering") / "construction.json")
    assert steering["status"] == "passed"
    assert steering["submitted"]["candidate_id"] == "original"
    assert steering["submitted"]["vector"] is None and steering["submitted"]["rank"] == 1
    construction = read(record(tmp_path, "steering+composition") / "construction.json")
    assert construction["status"] == "not_composed"
    assert construction["details"]["composition"] == {"reason": "original_model"}
    assert construction["ledger"]["totals"] == steering["ledger"]["totals"]
    assert composed.construction_inferences == plain.construction_inferences
    assert composed.capped_forward_kl == plain.capped_forward_kl
    assert not (record(tmp_path, "steering+composition") / "draws" / "selection.jsonl").exists()


def test_a_candidate_that_did_not_pass_is_submitted_as_best_observed_and_not_composed(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """Few candidates write "g", and none in a tenth of its responses."""
    chosen = settings.Settings(layers=(0, 1), strengths=(1.5,), selection_draws=64)
    plain, composed = run.run(make_context(tmp_path, '"g" in y'), chosen)
    steering = read(record(tmp_path, "steering") / "construction.json")
    assert steering["status"] == "best_observed"
    tests = steering["submission"]["tests"]
    assert len(tests) == len(steering["ranking"]) and not any(test["passed"] for test in tests)
    best = max(tests, key=lambda test: (test["hits"] / test["draws"], test["hits"], -test["rank"]))
    assert steering["submitted"]["candidate_id"] == best["candidate_id"]
    assert not plain.fallback or plain.kind != "no_sampler"
    construction = read(record(tmp_path, "steering+composition") / "construction.json")
    assert construction["status"] == "not_composed"
    assert construction["details"]["composition"] == {"reason": "plain_not_passed"}
    assert composed.construction_inferences == plain.construction_inferences


def test_no_compose_writes_steering_alone(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    chosen = settings.Settings(layers=(0,), strengths=(2.0,), selection_draws=16)
    (summary,) = run.run(make_context(tmp_path, compose=False), chosen)
    assert summary.method == "steering"
    assert not (tmp_path / "steering+composition").exists()


def test_a_run_without_usable_method_data_stops_before_it_writes(
    make_context: Callable[..., run_context.RunContext],
    write_vectors: Callable[..., pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "out"
    with pytest.raises(vectors.VectorError, match="--method-data"):
        run.run(make_context(out, method_data=None), SETTINGS)
    with pytest.raises(vectors.VectorError, match="--method-data"):
        run.check(make_context(out, method_data=None).options, SETTINGS)  # before the model
    run.check(make_context(out).options, SETTINGS)
    with pytest.raises(vectors.VectorError, match="no tensor for the layers"):
        # The test vectors have layers 0 and 1.
        run.run(make_context(out), settings.Settings(layers=(7,)))
    with pytest.raises(settings.LayersError):
        run.run(make_context(out), settings.Settings())  # the test model has two layers
    assert not out.exists()
    run.run(make_context(out, compose=False), settings.Settings(layers=(0,), strengths=(2.0,)))
    other = write_vectors(tmp_path / "other_vectors", seed=7)
    with pytest.raises(draws.CacheError):
        run.run(make_context(out, method_data=other), settings.Settings(layers=(0,)))


def test_the_paper_layers_on_the_24_layer_model(
    tiny_hybrid_model: Any,
    make_tiny_original_model: Callable[..., original_model.OriginalModel],
    make_tiny_dataset: Callable[..., pathlib.Path],
    write_vectors: Callable[..., pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    """The default layers, 7 to 13, on a model with the 24 layers and the architecture of the
    paper's model, with one strength and few draws."""
    scorer = make_tiny_original_model(
        model=tiny_hybrid_model, max_response_tokens=50, batch_size=32
    )
    corpus_scorer = make_tiny_original_model(
        model=tiny_hybrid_model, max_response_tokens=50, batch_size=400
    )
    bench = benchmark.Benchmark(
        make_tiny_dataset(
            corpus_scorer, task_id="wc_1853/goblin_mention", rule='"a" in y', draws=800
        )
    )
    directory = write_vectors(
        tmp_path / "vectors", layers=settings.layers_at_depths(24), scale=0.02
    )
    context = run_context.RunContext(
        family="steering",
        method="steering",
        options=run_context.RunOptions(
            task_id="wc_1853/goblin_mention",
            out=tmp_path / "out",
            compose=False,
            evaluation_draws=32,
            method_data=directory,
        ),
        bench=bench,
        task=bench.task("wc_1853/goblin_mention"),
        original=scorer,
        load_model_copy=lambda: scorer,
        tracker=tracking.Tracker(),
    )
    (summary,) = run.run(context, settings.Settings(strengths=(4.0,), selection_draws=16))
    construction = read(tmp_path / "out" / summary.record / "construction.json")
    listed = construction["details"]["candidates"]
    assert [c["layer"] for c in listed[1:7]] == [7, 8, 9, 11, 12, 13] and len(listed) == 13
    assert construction["status"] == "passed"
    assert summary.construction_inferences == (
        VECTORS_COST
        + 13 * 16
        + sum(c["selection_hits"] for c in listed)
        + construction["submission"]["test_draws"]
    )


def test_a_task_with_smaller_passes_draws_and_composes_in_parts(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """With the batch limits of a task whose model or response limit needs smaller passes, the
    candidates and the 16 composed settings are drawn and scored in parts, and the run writes
    the same records, charged by the same rules."""
    limited = dataclasses.replace(
        make_context(tmp_path, STEERABLE),
        batch_limits=run_context.BatchLimits(scoring=5, generation=24),
    )
    plain, composed = run.run(limited, SETTINGS)
    assert (plain.method, composed.method) == ("steering", "steering+composition")
    selection = rows(tmp_path, "steering", "selection")
    assert len(selection) == 13 * 64 and all(row["logq"] is not None for row in selection)
    # A candidate's 64 selection draws come from one seed, in parts of 24.
    assert len({row["seed"] for row in selection}) == 13
    construction = read(record(tmp_path, "steering") / "construction.json")
    assert construction["status"] == "passed"
    hits = sum(c["selection_hits"] for c in construction["details"]["candidates"])
    assert plain.construction_inferences == (
        VECTORS_COST + 13 * 64 + hits + construction["submission"]["test_draws"]
    )
    composition = read(record(tmp_path, "steering+composition") / "construction.json")
    assert len(rows(tmp_path, "steering+composition", "composition_selection")) == 16 * 16
    assert composition["status"] in ("passed", "best_observed")
