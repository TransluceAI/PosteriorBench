"""A whole run of the prompt family on the tiny model: the records it writes and what each method
is charged."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from posteriorbench import benchmark, original_model

from posteriorbench_methods.common import run_context, run_record
from posteriorbench_methods.prompt import methods, pool, run, settings

POOL_SIZES = (2, 8)
SELECTION = ("prompt-ebjc", "prompt-jc", "prompt-hitrate", "prompt-iwae", "prompt-ce")
MIXTURES = ("mixture-eb16", "mixture-uniform1", "mixture-uniform2")
COMPOSED = ("prompt-ebjc+composition", "mixture-eb16+composition", "instruction+composition")
METHODS = (*SELECTION, *MIXTURES, "instruction", *COMPOSED)


@dataclasses.dataclass(frozen=True)
class Run:
    out: pathlib.Path
    summaries: list[run_record.RunSummary]

    def summary(self, method: str, setting: str) -> run_record.RunSummary:
        (found,) = (s for s in self.summaries if (s.method, s.setting) == (method, setting))
        return found

    def json(self, method: str, setting: str, name: str) -> Any:
        return run_record.read_json(self.out / self.summary(method, setting).record / name)

    def draws(self, method: str, setting: str, name: str) -> list[dict[str, Any]]:
        directory = self.out / self.summary(method, setting).record
        return run_record.read_jsonl(directory / "draws" / f"{name}.jsonl")


@pytest.fixture(scope="module")
def tiny_run(
    make_context: Callable[..., run_context.RunContext], tmp_path_factory: pytest.TempPathFactory
) -> Run:
    out = tmp_path_factory.mktemp("prompt_run")
    family_settings = settings.Settings(methods=METHODS, pool_sizes=POOL_SIZES)
    return Run(out, run.run(make_context(out), family_settings))


def test_one_record_per_method_and_pool_size(tiny_run: Run) -> None:
    expected = [(m, f"K={k}") for k in POOL_SIZES for m in (*SELECTION, *MIXTURES, *COMPOSED[:2])]
    expected += [("instruction", "none"), ("instruction+composition", "none")]
    written = [(s.method, s.setting) for s in tiny_run.summaries]
    assert sorted(written) == sorted(expected)
    assert run_record.read_jsonl(tiny_run.out / "summary.jsonl") == [
        s.to_json() for s in tiny_run.summaries
    ]
    for method, setting in written:
        directory = tiny_run.out / tiny_run.summary(method, setting).record
        assert (directory / "result.json").exists() and (directory / "config.json").exists()
        assert (directory / "result_recomputed.json").exists()
        rows = tiny_run.draws(method, setting, "evaluation")
        assert len(rows) == 64 and all(row["token_ids"] for row in rows)
        config = tiny_run.json(method, setting, "config.json")
        assert config["settings"]["methods"] == list(METHODS)


def composition_cost(construction: dict[str, Any], components: int) -> float:
    """What a record's composition added: 496, p (and the other components' q) on the selection
    hits, and each test at 2 inferences per draw (1 for the plain setting)."""
    outcome = construction["details"]["composition"]
    setting = {str(i): s for i, s in enumerate(outcome["settings"])}
    tests = sum(
        t["draws"] * (1 if setting[t["candidate_id"]]["setting"] == "a0_t50" else 2)
        for t in outcome["submission"]["tests"]
    )
    return 496 + sum(outcome["selection_hits"]) * components + tests


def test_every_method_is_charged_by_its_rule(tiny_run: Run) -> None:
    """The inference count of each record, from the record's own draws and tests."""
    for size in POOL_SIZES:
        setting = f"K={size}"
        for method in (*SELECTION, *MIXTURES):
            construction = tiny_run.json(method, setting, "construction.json")
            selection = tiny_run.draws(method, setting, "selection")
            assert len(selection) == 16 * size and all(row["token_ids"] for row in selection)
            tests = tiny_run.draws(method, setting, "threshold_tests")
            assert len(tests) == construction["submission"]["test_draws"]
            if method == "prompt-hitrate":
                p_and_q = 0
            elif method == "prompt-ce":
                bank = construction["details"]["bank"]
                p_and_q = bank["hits"] + bank["cross_scored_pairs"]
            else:
                p_and_q = sum(row["hit"] for row in selection)
            cost = 16 * size + p_and_q + len(tests)
            assert tiny_run.summary(method, setting).construction_inferences == cost
            assert construction["ledger"]["totals"]["inferences"] == cost
        for method, components in (("prompt-ebjc", 1), ("mixture-eb16", None)):
            plain = tiny_run.json(method, setting, "construction.json")
            composed = tiny_run.json(method + "+composition", setting, "construction.json")
            assert composed["status"] in ("passed", "best_observed")
            count = components or len(plain["submitted"]["components"])
            assert composed["ledger"]["totals"]["inferences"] == plain["ledger"]["totals"][
                "inferences"
            ] + composition_cost(composed, count)
    assert tiny_run.summary("instruction", "none").construction_inferences == 0
    composed = tiny_run.json("instruction+composition", "none", "construction.json")
    assert composed["details"]["threshold_test"]["passed"]
    assert len(tiny_run.draws("instruction+composition", "none", "threshold_tests")) == 164
    assert composed["ledger"]["totals"]["inferences"] == 164 + composition_cost(composed, 1)


def test_the_duplicate_candidate_is_ranked_once(tiny_run: Run) -> None:
    ranking = tiny_run.json("prompt-ebjc", "K=8", "construction.json")["ranking"]
    assert len(ranking) == 7
    (entry,) = (entry for entry in ranking if entry["aliases"])
    assert {entry["candidate_id"], *entry["aliases"]} == {"0", "6"}


def test_a_mixture_of_one_component_is_ebjc_selection(tiny_run: Run) -> None:
    for size in POOL_SIZES:
        selection = tiny_run.json("prompt-ebjc", f"K={size}", "construction.json")
        mixture = tiny_run.json("mixture-uniform1", f"K={size}", "construction.json")
        assert mixture["ranking"] == selection["ranking"]
        assert mixture["submission"]["tests"] == selection["submission"]["tests"]
        (component,) = mixture["submitted"]["components"]
        assert component["candidate_id"] == selection["submitted"]["candidate_id"]
        assert component["weight"] == 1.0
        results = [
            tiny_run.json(method, f"K={size}", "result.json")
            for method in ("prompt-ebjc", "mixture-uniform1")
        ]
        for key in ("construction_inferences", "capped_forward_kl", "reverse_kl", "hits"):
            assert results[0][key] == results[1][key]


def test_mixture_records_name_their_components(tiny_run: Run) -> None:
    eb = tiny_run.json("mixture-eb16", "K=8", "construction.json")
    components = eb["submitted"]["components"]
    assert len(components) == 7, "every distinct prompt of the pool passes on this task"
    assert sum(c["weight"] for c in components) == pytest.approx(1.0)
    assert eb["details"]["shrinkage"] is not None
    uniform = tiny_run.json("mixture-uniform2", "K=8", "construction.json")
    assert [c["weight"] for c in uniform["submitted"]["components"]] == [0.5, 0.5]
    assert [t["passed"] for t in uniform["submission"]["passes"]] == [True, True]


def test_no_compose_leaves_the_compositions_out(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    family_settings = settings.Settings(
        methods=("instruction", "instruction+composition", "prompt-hitrate+composition"),
        pool_sizes=(2,),
    )
    summaries = run.run(make_context(tmp_path, compose=False), family_settings)
    assert [(s.method, s.setting) for s in summaries] == [("instruction", "none")]


def test_a_composed_method_also_writes_its_plain_method(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    family_settings = settings.Settings(methods=("prompt-hitrate+composition",), pool_sizes=(2,))
    summaries = run.run(make_context(tmp_path), family_settings)
    assert [(s.method, s.setting) for s in summaries] == [
        ("prompt-hitrate", "K=2"),
        ("prompt-hitrate+composition", "K=2"),
    ]


def test_a_pool_size_beyond_the_pool_file_is_refused(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    with pytest.raises(pool.PoolError, match="has 8 candidates"):
        run.run(make_context(tmp_path), settings.Settings(pool_sizes=(2, 9)))
    with pytest.raises(pool.PoolError, match="--method-data"):
        run.run(make_context(tmp_path, method_data=None), settings.Settings(pool_sizes=(2,)))
    assert not list(tmp_path.iterdir()), "nothing is drawn or written before the pool is checked"


def test_a_run_without_a_method_or_without_its_pool_file_is_refused_before_it_starts(
    tmp_path: pathlib.Path,
) -> None:
    with_pool = run_context.RunOptions(task_id="t", out=tmp_path, method_data=tmp_path / "pool")
    without_pool = run_context.RunOptions(task_id="t", out=tmp_path)
    run.check(with_pool, settings.Settings())
    with pytest.raises(pool.PoolError, match="--method-data"):
        run.check(without_pool, settings.Settings())
    # Instruction reads no pool file.
    instruction = settings.Settings(methods=("instruction+composition",))
    run.check(without_pool, instruction)
    run.check(
        without_pool, settings.Settings(methods=("instruction", "prompt-ebjc"), pool_sizes=())
    )
    # Nothing to run: only compositions without `compose`, no method, no pool size.
    no_compose = run_context.RunOptions(task_id="t", out=tmp_path, compose=False)
    for options, chosen in (
        (no_compose, instruction),
        (with_pool, settings.Settings(methods=())),
        (with_pool, settings.Settings(methods=("prompt-ebjc", "mixture-eb16"), pool_sizes=())),
    ):
        with pytest.raises(methods.MethodError, match="no method to run"):
            run.check(options, chosen)


def test_selection_runs_on_a_task_outside_the_benchmark(
    scorer: original_model.OriginalModel,
    make_tiny_dataset: Callable[..., pathlib.Path],
    make_pool_file: Callable[..., pathlib.Path],
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
) -> None:
    """Seeds and records need nothing but the task id, so any task of a dataset can be run."""
    task_id = "my_prompt/letter_a"
    data = make_tiny_dataset(scorer, task_id=task_id, draws=1500)
    bench = benchmark.Benchmark(data)
    context = dataclasses.replace(
        make_context(tmp_path, method_data=make_pool_file(("say a", "zzz", "abc"), task_id)),
        bench=bench,
        task=bench.task(task_id),
    )
    context = dataclasses.replace(
        context, options=dataclasses.replace(context.options, task_id=task_id)
    )
    family_settings = settings.Settings(methods=("prompt-ebjc", "mixture-eb16"), pool_sizes=(3,))
    summaries = run.run(context, family_settings)
    assert [(s.method, s.task_id) for s in summaries] == [
        ("prompt-ebjc", task_id),
        ("mixture-eb16", task_id),
    ]
