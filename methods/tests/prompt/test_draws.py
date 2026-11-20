"""The draws of a run on the tiny model: what they depend on, and what the rows hold."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
from posteriorbench import original_model

from posteriorbench_methods.common import composition, run_context
from posteriorbench_methods.prompt import draws, pool, samplers


def make_draws(context: run_context.RunContext) -> draws.Draws:
    return draws.Draws(context, samplers.Factory(context.original, context.task.prompt_token_ids))


def spec(scorer: original_model.OriginalModel, system_prompt: str) -> samplers.Spec:
    return samplers.Spec((tuple(scorer.render_prompt("hi", system=system_prompt)),))


def test_selection_draws_depend_on_the_candidate_and_the_replicate(
    make_context: Callable[..., run_context.RunContext],
    scorer: original_model.OriginalModel,
    tmp_path: pathlib.Path,
) -> None:
    first, other = pool.Candidate(3, "c003", "say a"), pool.Candidate(4, "c004", "say a")
    prompt = spec(scorer, "say a")
    run = make_draws(make_context(tmp_path))
    drawn = run.selection(first, prompt, 16)
    assert len(drawn.token_ids) == 16 and drawn.values.hit.shape == (16,)
    assert (
        drawn.token_ids == make_draws(make_context(tmp_path)).selection(first, prompt, 16).token_ids
    )
    assert drawn.token_ids != run.selection(other, prompt, 16).token_ids
    later = make_draws(make_context(tmp_path, replicate=1)).selection(first, prompt, 16)
    assert drawn.token_ids != later.token_ids
    # log q and log p are computed on the hits; the row of a miss holds None for them.
    hit = drawn.values.hit
    assert np.isfinite(drawn.values.logq[hit]).all() and np.isnan(drawn.values.logq[~hit]).all()
    rows = drawn.rows(candidate_id="3")
    assert [row["draw_index"] for row in rows] == list(range(16))
    assert all(row["candidate_id"] == "3" and row["seed"] == drawn.seed for row in rows)
    assert [(row["logq"] is not None, row["logp"] is not None) for row in rows] == [
        (h, h) for h in hit.tolist()
    ]
    expected = scorer.logprobs(
        prompt.prompts[0], [list(drawn.token_ids[i]) for i in np.flatnonzero(hit)]
    )
    assert [row["logq"] for row in rows if row["hit"]] == expected


def test_a_longer_threshold_test_extends_a_shorter_one(
    make_context: Callable[..., run_context.RunContext],
    scorer: original_model.OriginalModel,
    tmp_path: pathlib.Path,
) -> None:
    prompt = spec(scorer, "zzz")
    run = make_draws(make_context(tmp_path))
    short = run.threshold_test(prompt, 40)
    assert [row["draw_index"] for row in short] == list(range(40))
    assert len({row["seed"] for row in short}) == 2, "parts of 32 draws, each with its seed"
    longer = run.threshold_test(prompt, 100)
    assert longer[:40] == short
    assert draws.count_hits(longer) == sum(row["hit"] for row in longer)
    # The stream belongs to the sampler: another run object of the replicate draws it again.
    assert make_draws(make_context(tmp_path)).threshold_test(prompt, 100) == longer
    assert run.threshold_test(spec(scorer, "abc"), 40) != short


def test_composition_selection_draws_can_be_shared_by_replicates(
    make_context: Callable[..., run_context.RunContext],
    scorer: original_model.OriginalModel,
    tmp_path: pathlib.Path,
) -> None:
    base, setting = spec(scorer, "say a"), composition.Setting(0.25, 16)
    replicates = [make_draws(make_context(tmp_path, replicate=r)) for r in (0, 1)]
    shared = [r.composition_selection(base, setting, shared_by_replicates=True) for r in replicates]
    own = [r.composition_selection(base, setting, shared_by_replicates=False) for r in replicates]
    assert len(shared[0].token_ids) == composition.SELECTION_DRAWS
    assert shared[0].token_ids == shared[1].token_ids
    assert own[0].token_ids != own[1].token_ids
    assert own[0].token_ids != shared[0].token_ids
