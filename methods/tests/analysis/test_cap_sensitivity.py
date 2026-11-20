"""The capped forward KL at other caps, from the records of small runs."""

from __future__ import annotations

import math
import pathlib
import shutil
from typing import Any

import pytest
from posteriorbench import aggregate, estimators
from posteriorbench import results as pb_results

from posteriorbench_methods.analysis import cap_sensitivity, records
from posteriorbench_methods.common import run_record


def test_a_record_s_own_cap_gives_its_stored_score(finished: list[records.Record]) -> None:
    assert {record.result["cap"] for record in finished} == {50.0}
    assert {cap_sensitivity.passed(record) for record in finished} == {True, False}
    for record in finished:
        stored = records.number(record.result["capped_forward_kl"])
        assert cap_sensitivity.capped_forward_kl(record, 50.0) == pytest.approx(stored, abs=1e-12)


@pytest.mark.parametrize("cap", [10.0, 20.0, 100.0, 200.0])
def test_another_cap_gives_the_score_that_the_package_computes(
    finished: list[records.Record], cap: float
) -> None:
    """From the log p and log q of the references for a sampler that passed, and from the
    fallback's rho for a run that was scored as the fallback."""
    for record in finished:
        result = record.result
        if cap_sensitivity.passed(record):
            details = record.evaluation_details
            expected = estimators.capped_forward_kl(
                [records.number(value) for value in details["reference_logp"]],
                [records.number(value) for value in details["reference_logq"]],
                log_p_event=result["log_p_event"],
                log_q_event=details["log_q_event"],
                cap=cap,
            )
        else:
            rho = result["fallback_rho"]
            expected = rho * math.log(rho) + (1 - rho) * cap
        assert cap_sensitivity.capped_forward_kl(record, cap) == pytest.approx(expected, abs=1e-9)


def test_replay_is_at_the_cap_and_is_not_among_the_shares(finished: list[records.Record]) -> None:
    (replay,) = [r for r in finished if r.method == "replay" and cap_sensitivity.passed(r)]
    assert replay.result["exact"]
    assert cap_sensitivity.capped_forward_kl(replay, 20.0) == 20.0
    assert cap_sensitivity.references_at_cap(replay, 20.0) is None
    (no_sampler,) = [r for r in finished if (r.method, r.setting) == ("sft", "B=2048")]
    assert cap_sensitivity.references_at_cap(no_sampler, 20.0) is None


@pytest.fixture
def four_terms(finished: list[records.Record], tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of the record of a sampler that passed, with four terms written by hand: 0, 10, 60
    and infinity, whose capped forward KL at 50 is 27.5."""
    source = next(r for r in finished if r.method == "ctl")
    directory = tmp_path / "record"
    shutil.copytree(source.directory, directory)
    details: dict[str, Any] = run_record.read_json(directory / "evaluation_details.json")
    details["shipped"]["forward_terms"] = [0.0, 10.0, 60.0, math.inf]
    run_record.write_json(directory / "evaluation_details.json", details)
    result = run_record.read_json(directory / "result.json")
    run_record.write_json(directory / "result.json", {**result, "capped_forward_kl": 27.5})
    return directory


def test_the_references_at_or_above_the_cap(four_terms: pathlib.Path) -> None:
    (record,) = records.find([four_terms])
    assert cap_sensitivity.capped_forward_kl(record, 10.0) == 7.5
    assert cap_sensitivity.capped_forward_kl(record, 200.0) == (0 + 10 + 60 + 200) / 4
    assert cap_sensitivity.references_at_cap(record, 50.0) == (2, 4)
    assert cap_sensitivity.references_at_cap(record, 10.0) == (3, 4)  # a term at the cap counts
    assert cap_sensitivity.references_at_cap(record, 100.0) == (1, 4)
    share = cap_sensitivity.analyze([record], (50.0,))["share_at_cap"][0]
    assert (share["samplers"], share["references_at_cap"], share["share"]) == (1, 2, 0.5)


def test_terms_that_do_not_give_the_stored_score_are_an_error(four_terms: pathlib.Path) -> None:
    result = run_record.read_json(four_terms / "result.json")
    run_record.write_json(four_terms / "result.json", {**result, "capped_forward_kl": 30.0})
    (record,) = records.find([four_terms])
    with pytest.raises(records.RecordsError, match="27.5"):
        cap_sensitivity.capped_forward_kl(record, 10.0)


def test_the_points_are_averaged_as_the_package_averages(finished: list[records.Record]) -> None:
    analysis = cap_sensitivity.analyze(finished, with_runs=True)
    assert analysis["caps"] == [10.0, 20.0, 50.0, 100.0, 200.0]
    points = {(p["cap"], p["method"], p["setting"]): p for p in analysis["points"]}
    assert len(points) == 5 * len({(r.method, r.setting) for r in finished})
    for record in finished:  # one task and one replicate at each point: the mean is the run's
        stored = pb_results.EvaluationResult.from_dict(dict(record.result))
        expected = aggregate.aggregate([stored]).to_dict()
        point = points[(50.0, record.method, record.setting)]
        assert set(point) == {"cap", "method", "setting", *expected}
        for name, metric in expected["metrics"].items():
            assert point["metrics"][name]["mean"] == pytest.approx(metric["mean"], abs=1e-12)
            assert point["metrics"][name]["standard_error"] is None
        assert (point["tasks"], point["results"]) == (1, 1)
        assert point["passed_threshold"] == expected["passed_threshold"]
        for cap in analysis["caps"]:
            at_cap = points[(cap, record.method, record.setting)]["metrics"]
            own = cap_sensitivity.capped_forward_kl(record, cap)
            assert at_cap["capped_forward_kl"]["mean"] == pytest.approx(own)
            assert at_cap["reverse_kl"] == point["metrics"]["reverse_kl"]
    assert len(analysis["runs"]) == len(finished)
    assert set(analysis["runs"][0]["capped_forward_kl"]) == {
        "10.0",
        "20.0",
        "50.0",
        "100.0",
        "200.0",
    }


def test_the_share_pools_the_samplers_that_passed(finished: list[records.Record]) -> None:
    (share,) = cap_sensitivity.analyze(finished, (10.0,))["share_at_cap"]
    counted = [c for c in (cap_sensitivity.references_at_cap(r, 10.0) for r in finished) if c]
    assert share["samplers"] == len(counted) > 0
    assert share["references"] == 250 * len(counted)
    assert share["references_at_cap"] == sum(count for count, _ in counted)
    assert share["share"] == share["references_at_cap"] / share["references"]
    assert "replay" in share["by_method"] and share["by_method"]["replay"]["samplers"] == 0
    assert sum(entry["samplers"] for entry in share["by_method"].values()) == len(counted)


def test_records_that_cannot_be_averaged_are_an_error(finished: list[records.Record]) -> None:
    with pytest.raises(records.RecordsError, match="repeated replicate"):
        cap_sensitivity.analyze([finished[0], finished[0]])
