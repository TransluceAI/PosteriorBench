"""The importance-sampling estimate of p(S) from the evaluation draws of a run: by hand, and on
the records of small runs, whose samplers were evaluated on 64 draws."""

from __future__ import annotations

import json
import math
import pathlib
import shutil
from typing import Any

import pytest
from typer.testing import CliRunner

from posteriorbench_methods import cli
from posteriorbench_methods.analysis import ps_estimate, records
from posteriorbench_methods.common import run_record

DRAWS = 60  # of the 64 evaluation draws of the small runs
RUNNER = CliRunner()


def evaluation_rows(record: records.Record) -> list[dict[str, Any]]:
    return run_record.read_jsonl(record.directory / "draws" / "evaluation.jsonl")


def copy_of(record: records.Record, directory: pathlib.Path) -> pathlib.Path:
    """A copy of a record's directory that a test can change."""
    shutil.copytree(record.directory, directory)
    return directory


@pytest.fixture(scope="module")
def passing(finished: list[records.Record]) -> records.Record:
    """The record of a sampler that passed the threshold, CTL's."""
    (record,) = [r for r in finished if r.method == "ctl"]
    assert not record.result["fallback"]
    return record


def test_the_estimate_and_the_reverse_kl_of_two_hits_by_hand() -> None:
    """Four draws with two hits, each with p / q = 0.04: the estimate is 0.02, twice a p̂(S) of
    0.01, and with equal ratios the log error is exactly minus the reverse KL."""
    logp = [math.log(0.02), math.log(0.01)]
    logq = [math.log(0.5), math.log(0.25)]
    log_error, reverse_kl = ps_estimate.log_error_from_hits(
        logp, logq, draws=4, log_p_event=math.log(0.01)
    )
    assert log_error == pytest.approx(math.log(2))
    assert reverse_kl == pytest.approx(-math.log(2))
    # Unequal ratios, 0.04 and 0.4: the estimate is 0.11, and it lies above the bound.
    log_error, reverse_kl = ps_estimate.log_error_from_hits(
        [math.log(0.02), math.log(0.1)], logq, draws=4, log_p_event=math.log(0.01)
    )
    assert log_error == pytest.approx(math.log(0.11 / 0.01))
    assert reverse_kl == pytest.approx(math.log(0.01) - math.log(0.5) - math.log(0.04 * 0.4) / 2)
    assert log_error > -reverse_kl
    # A hit that the original model cannot generate adds nothing to the estimate.
    log_error, reverse_kl = ps_estimate.log_error_from_hits(
        [math.log(0.02), -math.inf], logq, draws=4, log_p_event=math.log(0.01)
    )
    assert log_error == pytest.approx(0) and reverse_kl == math.inf


def test_a_run_with_hits_is_scored_by_its_own_estimate(finished: list[records.Record]) -> None:
    """From the first 60 draws, with log q and log p of every hit among them, which a record
    has in evaluation_hits.json."""
    own = 0
    for record in finished:
        if record.result["kind"] != "sampler" or record.result["exact"]:
            continue
        scored = ps_estimate.estimate(record, DRAWS)
        rows = evaluation_rows(record)
        hits = [index for index, row in enumerate(rows[:DRAWS]) if row["hit"]]
        assert (scored.source, scored.draws, scored.hits) == ("own", DRAWS, len(hits))
        scores = run_record.read_json(record.directory / "evaluation_hits.json")
        by_index = dict(zip(scores["draw_index"], zip(scores["logq"], scores["logp"])))
        assert set(by_index) == {index for index, row in enumerate(rows) if row["hit"]}
        ratios = [math.exp(by_index[index][1] - by_index[index][0]) for index in hits]
        expected = math.log(sum(ratios) / DRAWS) - record.result["log_p_event"]
        assert scored.log_error is not None and scored.reverse_kl_all_hits is not None
        assert scored.log_error == pytest.approx(expected)
        assert scored.log_error >= -scored.reverse_kl_all_hits - 1e-9  # the bound
        values = scored.to_json()
        assert values["estimate"] == "own" and values["abs_log_error"] == abs(scored.log_error)
        assert values["reverse_kl"] == record.result["reverse_kl"]
        own += 1
    assert own >= 8


def test_a_run_without_a_hit_takes_the_fallback_s_estimate(
    finished: list[records.Record], passing: records.Record, tmp_path: pathlib.Path
) -> None:
    # A run that produced no sampler: p(a) / p̂(S) is exp(-the fallback's reverse KL).
    (no_sampler,) = [r for r in finished if (r.method, r.setting) == ("sft", "B=2048")]
    scored = ps_estimate.estimate(no_sampler, DRAWS)
    assert (scored.source, scored.draws, scored.hits) == ("fallback", 0, 0)
    assert scored.log_error == -no_sampler.result["reverse_kl"] < 0
    assert scored.log_error == pytest.approx(math.log(no_sampler.result["fallback_rho"]))
    assert scored.reverse_kl_all_hits is None and not scored.passed
    # A sampler whose hits all lie after the draws that the estimate reads.
    directory = copy_of(passing, tmp_path / "late_hits")
    rows = evaluation_rows(passing)
    late = [{**row, "hit": row["hit"] and index >= DRAWS} for index, row in enumerate(rows)]
    run_record.write_draws(directory, "evaluation", late)
    (record,) = records.find([directory])
    scored = ps_estimate.estimate(record, DRAWS)
    assert (scored.source, scored.draws, scored.hits) == ("fallback", DRAWS, 0)
    log_p_fallback = record.evaluation_details["log_p_fallback"]
    assert scored.log_error == pytest.approx(log_p_fallback - record.result["log_p_event"])


def test_a_record_without_the_scores_of_every_hit(
    passing: records.Record, tmp_path: pathlib.Path
) -> None:
    """A record that `run` wrote before it kept evaluation_hits.json has the scores of the hits
    that its reverse KL was averaged over. If these are all its hits, it has its own estimate."""
    directory = copy_of(passing, tmp_path / "earlier")
    (directory / "evaluation_hits.json").unlink()
    (record,) = records.find([directory])
    with_file = ps_estimate.estimate(passing, DRAWS)
    without = ps_estimate.estimate(record, DRAWS)
    assert (without.source, without.log_error) == ("own", with_file.log_error)
    # One hit among the draws without its scores: the estimate is unavailable.
    rows = evaluation_rows(passing)
    first_hit = next(index for index, row in enumerate(rows) if row["hit"])
    rows[first_hit] = {k: v for k, v in rows[first_hit].items() if k not in ("logq", "logp")}
    run_record.write_draws(directory, "evaluation", rows)
    (record,) = records.find([directory])
    scored = ps_estimate.estimate(record, DRAWS)
    assert (scored.source, scored.log_error) == ("unavailable", None) and scored.hits > 0
    summary = ps_estimate.point_summary("ctl", "u=2", [scored])
    assert summary["unavailable"] == 1 and summary["abs_log_error"] is None
    assert summary["construction_inferences"]["mean"] == record.construction_inferences


def test_the_estimate_reads_no_more_draws_than_the_evaluation_made(
    passing: records.Record,
) -> None:
    with pytest.raises(records.RecordsError, match="evaluated on 64 draws, fewer than the 960"):
        ps_estimate.estimate(passing)


def test_the_average_over_replicates_and_then_tasks() -> None:
    values = {("a", 0): 1.0, ("a", 1): 3.0, ("b", 0): 4.0, ("b", 1): 6.0}
    averaged = ps_estimate.average_over_tasks(values)
    assert averaged == {
        "mean": 3.5,
        "standard_error": pytest.approx(math.sqrt(4.5 / 2)),  # of the task means 2 and 5
        "tasks": 2,
        "replicates_per_task": 2,
    }
    one_missing = {key: value for key, value in values.items() if key != ("b", 1)}
    with pytest.raises(ValueError, match="different numbers of replicates"):
        ps_estimate.average_over_tasks(one_missing)
    allowed = ps_estimate.average_over_tasks(one_missing, require_equal_replicates=False)
    assert (allowed["mean"], allowed["replicates_per_task"]) == (3.0, 1)
    assert ps_estimate.average_over_tasks({("a", 0): 2.0})["standard_error"] is None


def test_spearman_s_correlation_with_ties() -> None:
    assert ps_estimate.spearman([1, 2, 3, 4], [10, 20, 40, 80]) == pytest.approx(1)
    assert ps_estimate.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    # Ranks 1, 2.5, 2.5, 4 against 1, 2, 3, 4.
    assert ps_estimate.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / math.sqrt(22.5))
    assert ps_estimate.spearman([1, 1, 1], [1, 2, 3]) is None


def test_the_points_and_the_samplers_of_the_small_runs(finished: list[records.Record]) -> None:
    analysis = ps_estimate.analyze(finished, DRAWS, with_runs=True, with_samplers=True)
    assert analysis["draws"] == DRAWS and analysis["not_covered"] == {"replay": 2}
    covered = [r for r in finished if r.method != "replay"]
    points = {(point["method"], point["setting"]): point for point in analysis["points"]}
    assert set(points) == {(r.method, r.setting) for r in covered}
    runs = {(run["method"], run["setting"]): run for run in analysis["runs"]}
    for key, point in points.items():  # one task and one replicate at each point
        assert (point["tasks"], point["replicates_per_task"], point["runs"]) == (1, 1, 1)
        assert point["abs_log_error"] == {
            "mean": runs[key]["abs_log_error"],
            "standard_error": None,
        }
        assert point["construction_inferences"]["mean"] == runs[key]["construction_inferences"]
        counts = ("own_estimate_passed", "own_estimate_failed", "fallback_estimate", "unavailable")
        assert sum(point[name] for name in counts) == 1
    assert points[("sft", "B=2048")]["fallback_estimate"] == 1
    assert points[("ctl", "u=2")]["own_estimate_passed"] == 1
    # A sampler whose evaluation several records hold is one row.
    passed = [
        run for run in analysis["runs"] if run["estimate"] == "own" and run["passed_threshold"]
    ]
    samplers = analysis["samplers"]
    assert sum(row["records"] for row in samplers) == len(passed)
    assert len(samplers) < len(passed)
    for row in samplers:
        needed = math.exp(row["capped_forward_kl"]) / row["native_hit_rate"]
        assert row["draws_needed"] == pytest.approx(needed)
        assert row["log_error"] >= -row["reverse_kl_all_hits"] - 1e-9
    summary = analysis["samplers_summary"]
    assert summary["samplers"] == len(samplers)
    enough = [row for row in samplers if row["draws_needed"] < DRAWS]
    assert summary["need_fewer_draws"] == len(enough)
    assert "samplers" not in ps_estimate.analyze(finished, DRAWS)


def test_two_records_of_one_run_are_an_error(finished: list[records.Record]) -> None:
    with pytest.raises(records.RecordsError, match="more than one record of replicate 0"):
        ps_estimate.analyze([*finished, finished[0]], DRAWS)


def test_the_command_prints_the_points(finished_root: pathlib.Path, tmp_path: pathlib.Path) -> None:
    result = RUNNER.invoke(cli.app, ["estimate-ps", str(finished_root), "--draws", str(DRAWS)])
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert sorted(printed) == [
        "draws", "not_covered", "points", "samplers_summary", "unavailable",
    ]  # fmt: skip
    assert printed["unavailable"] == 0
    out = tmp_path / "ps.json"
    arguments = ["estimate-ps", str(finished_root), "--draws", str(DRAWS), "--samplers", "--runs"]
    assert RUNNER.invoke(cli.app, [*arguments, "--out", str(out)]).exit_code == 0
    assert {"samplers", "runs"} <= set(json.loads(out.read_text()))
    # The paper's 960 draws are more than the 64 of these runs.
    result = RUNNER.invoke(cli.app, ["estimate-ps", str(finished_root)])
    assert result.exit_code == 1 and result.output.count("\n") == 1
    assert result.output.startswith("Error: ") and "fewer than the 960" in result.output
