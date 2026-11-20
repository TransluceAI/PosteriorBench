"""The shown sample responses and prompts, and the scripts that print them."""

from __future__ import annotations

import json
import pathlib

import paper_runs
import pytest
import sample_responses
import selected_prompts
import typer.testing

GALLERIES = (17, 18, 19, 20, 21)


def test_the_shipped_responses_are_those_of_the_papers_figures(
    main_runs: list[paper_runs.Row],
) -> None:
    responses = sample_responses.read_responses()
    assert list(dict.fromkeys(r["figure"] for r in responses)) == [
        sample_responses.FIGURE_6,
        *GALLERIES,
    ]
    benchmark = set(paper_runs.event_probabilities())
    assert all(r["task_id"] in benchmark and r["text"].strip() for r in responses)
    for figure in GALLERIES:
        mine = [r for r in responses if r["figure"] == figure]
        task_id = mine[0]["task_id"]
        assert mine[0]["method"] == sample_responses.REFERENCE and mine[0]["setting"] is None
        rows = sample_responses.gallery_rows(main_runs, task_id)
        assert len(rows) == 12
        passing = [(method, run["setting"]) for method, run in rows if run["fallback"] == "false"]
        # one response per sampler that passed, in the order of the paper's rows
        assert [(r["method"], r["setting"]) for r in mine[1:]] == passing
        assert all(r["replicate"] == 0 for r in mine[1:])
        assert [run["fallback"] for _, run in rows] == sorted(run["fallback"] for _, run in rows)


def test_figure_6_has_a_reference_and_three_samplers() -> None:
    figure_6 = [
        r for r in sample_responses.read_responses() if r["figure"] == sample_responses.FIGURE_6
    ]
    assert [r["method"] for r in figure_6] == [
        sample_responses.REFERENCE,
        *sample_responses.FIGURE_6_SAMPLERS,
    ]
    assert {r["task_id"] for r in figure_6} == {"wc_1786/profanity"}


def test_the_scripts_print() -> None:
    app = typer.Typer()
    app.command()(sample_responses.main)
    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["--figure", "18"])
    assert result.exit_code == 0, result.output
    assert "SFT  2.3 / 3.0; 19%" in result.output
    assert "(Hit rate 5.2%, below the hit-rate threshold. Scored as the fallback.)" in result.output
    figure_6 = runner.invoke(app, ["--figure", "6"]).output
    assert "SFT  3.7e+06 inferences; 3.1 / 4.3" in figure_6
    assert "Prompt: EB-JC  1.9e+04 inferences; 25.1 / 25.2" in figure_6
    assert runner.invoke(app, ["--figure", "5"]).exit_code != 0
    prompts = selected_prompts.read_prompts()
    assert [p["method"] for p in prompts] == ["prompt-ebjc", "prompt-hitrate", "prompt-iwae"]
    assert {(p["task_id"], p["setting"], p["replicate"]) for p in prompts} == {
        ("wc_1569/em_dash", "K=1024", 0)
    }


def test_hits_of_a_record(tmp_path: pathlib.Path) -> None:
    draws = [{"draw_index": i, "token_ids": [i, i + 1], "hit": i % 3 == 0} for i in range(12)]
    (tmp_path / "draws").mkdir()
    (tmp_path / "draws" / "evaluation.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in draws)
    )
    hits = sample_responses.record_hits(tmp_path, count=3, seed=0)
    assert len(hits) == 3 and all(ids[0] % 3 == 0 for ids in hits)
    assert hits == sample_responses.record_hits(tmp_path, count=3, seed=0)
    assert len(sample_responses.record_hits(tmp_path, count=99, seed=1)) == 4
    with pytest.raises(paper_runs.TableError, match="no draws/evaluation.jsonl"):
        sample_responses.record_hits(tmp_path / "absent", count=1, seed=0)


def test_shown_marks_line_breaks() -> None:
    assert sample_responses.shown("a\n\n b \nc") == "a ¶ b ¶ c"
