"""The training run: the plan of an update, checkpoints, budgets, stopping and continuing."""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from posteriorbench import ledger, rules

from posteriorbench_methods.common import (
    hf_runtime,
    run_context,
    run_record,
    submission,
    training_run,
)
from posteriorbench_methods.gfn import policy as gfn_policy
from posteriorbench_methods.gfn import replay, reward
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import subtb, teacher_forcing, training

Context = Callable[..., run_context.RunContext]
Quick = Callable[..., gfn_settings.Settings]


def train(
    make_context: Context, out: pathlib.Path, settings: gfn_settings.Settings
) -> tuple[list[training_run.Checkpoint], pathlib.Path]:
    """Train in `out`; returns the checkpoints and the directory of the training run."""
    context = make_context(out)
    policy = gfn_policy.build(context, settings)
    return training.train(context, settings, policy), context.training_run_dir("gflownet")


def updates_trained(caplog: pytest.LogCaptureFixture) -> list[int]:
    """The updates that the runs since the last `caplog.clear()` trained."""
    events = [json.loads(record.getMessage()) for record in caplog.records]
    return [event["update"] for event in events if event["event"] == "update"]


def files(directory: pathlib.Path, folder: str) -> dict[str, list[dict[str, Any]]]:
    return {
        path.name: run_record.read_jsonl(path) for path in sorted((directory / folder).iterdir())
    }


def test_groups_are_fresh_until_the_buffer_holds_a_response() -> None:
    always = gfn_settings.Settings(replay_probability=1.0, tempered_probability=0.0)
    plans = training.plan_groups([11, 12, 13], buffer_is_empty=True, settings=always)
    assert [plan.replay for plan in plans] == [False, True, True]
    assert [plan.temperature for plan in plans] == [1.0, None, None]
    plans = training.plan_groups([11, 12, 13], buffer_is_empty=False, settings=always)
    assert [plan.replay for plan in plans] == [True, True, True]
    never = gfn_settings.Settings(replay_probability=0.0, tempered_probability=1.0)
    plans = training.plan_groups(range(50), buffer_is_empty=False, settings=never)
    temperatures = [plan.temperature for plan in plans]
    assert all(t is not None and 0.5 <= t <= 2.0 for t in temperatures)
    assert len(set(temperatures)) == 50


def test_a_plan_is_decided_by_the_seeds_of_its_groups() -> None:
    settings = gfn_settings.Settings()
    seeds_of_groups = list(range(100, 125))
    first = training.plan_groups(seeds_of_groups, buffer_is_empty=False, settings=settings)
    again = training.plan_groups(seeds_of_groups, buffer_is_empty=False, settings=settings)
    assert [plan.temperature for plan in first] == [plan.temperature for plan in again]
    assert 0 < sum(plan.replay for plan in first) < 25
    assert any(plan.temperature not in (None, 1.0) for plan in first)
    # A replayed group goes on with its own generator: the same seed, the same responses.
    replayed = [plan for plan in first if plan.replay][0]
    index = replayed.index
    assert np.array_equal(replayed.rng.choice(50, 20), again[index].rng.choice(50, 20))


def test_the_gradient_of_a_group_is_that_of_its_mean_term_over_the_groups(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    """`add_gradient` works through a group in batches of five; the gradient it leaves is that
    of the group's loss computed in one piece."""
    context, settings = make_context(tmp_path), quick(update_caps=(1,))
    policy = gfn_policy.build(context, settings)
    with torch.no_grad():  # move the policy off the original model, so that the loss is not flat
        for name, parameter in policy.model.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.05)
    p = hf_runtime.CausalLM(context.original.model, policy.lm.prompt_ids)
    is_hit = rules.parse_rule(context.task.rule)
    stops = policy.stop_token_ids
    (draws,) = policy.draw([hf_runtime.DrawGroup(8, 5)])
    responses = reward.score(
        p, stops, [d.token_ids for d in draws], decode=context.original.decode, is_hit=is_hit
    )
    parameters = [q for q in policy.model.parameters() if q.requires_grad]
    state = training.Training(
        context=context,
        settings=settings,
        policy=policy,
        p=p,
        is_hit=is_hit,
        directory=tmp_path,
        optimizer=torch.optim.AdamW(parameters),
        buffer=replay.ReplayBuffer(size=50, similarity=0.25, epsilon=1e-8, stop_token_ids=stops),
        ledger=ledger.Ledger(),
        checkpoints=[],
    )
    loss, logq = state.add_gradient(responses)
    accumulated = [q.grad.clone() for q in parameters if q.grad is not None]
    assert len(accumulated) == len(parameters)
    assert logq == pytest.approx(policy.logq([r.token_ids for r in responses]), abs=1e-4)

    policy.model.zero_grad()
    scores = teacher_forcing.token_scores(policy.lm, stops, [r.token_ids for r in responses])
    total, count = torch.zeros(()), 0
    for response, (token_logq, stop_logq) in zip(responses, scores, strict=True):
        terms, number = subtb.subtb_terms(
            token_logq,
            stop_logq,
            torch.tensor(response.token_logp),
            torch.tensor(response.stop_logp),
            response.prefix_hits,
            epsilon=settings.epsilon,
            terminated=response.token_ids[-1] in stops,
        )
        total, count = total + terms, count + number
    assert loss == pytest.approx(float(total.detach()) / count, rel=1e-5)
    (total / count / settings.groups_per_update).backward()
    for gradient, parameter in zip(accumulated, parameters, strict=True):
        assert parameter.grad is not None
        assert torch.allclose(gradient, parameter.grad, rtol=1e-3, atol=1e-6)
    assert max(float(g.abs().max()) for g in accumulated) > 0


def test_the_first_update_leaves_the_original_model(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(update_caps=(2,))
    initial = gfn_policy.build(make_context(tmp_path), settings)
    initial_id = training_run.sampler_id(training_run.trainable_tensors(initial.model))
    checkpoints, directory = train(make_context, tmp_path, settings)
    assert [row.update for row in checkpoints] == [1, 2]
    assert checkpoints[0].sampler_id == initial_id != checkpoints[1].sampler_id
    assert training.is_original_model(checkpoints[0])
    assert not training.is_original_model(checkpoints[1])
    for row in checkpoints:
        assert row.ledger["budget"] == row.budget_spent
        draws = training_run.read_selection_draws(directory, row.checkpoint_id)
        assert len(draws) == row.selection_draws == 32
        assert sum(draw["hit"] for draw in draws) == row.selection_hits
        assert all(draw["token_ids"] for draw in draws)
        assert len({draw["seed"] for draw in draws}) == 1  # one part of 32 draws


def test_what_an_update_trains_on_is_stored_and_charged(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    checkpoints, directory = train(make_context, tmp_path, quick(update_caps=(4,)))
    fresh = replayed = 0
    for update in range(1, 5):
        rows = run_record.read_jsonl(training_run.training_draws_path(directory, update))
        assert [row["draw_index"] for row in rows] == list(range(16))
        assert [row["group"] for row in rows] == [0] * 8 + [1] * 8
        for row in rows:
            assert row["token_ids"] and isinstance(row["logp"], float)
            assert (row["temperature"] is None) == row["replayed"]
        fresh += sum(not row["replayed"] for row in rows)
        replayed += sum(row["replayed"] for row in rows)
    assert fresh > 0 and fresh + replayed == 4 * 16
    last = checkpoints[-1]
    selection = sum(row.selection_draws + row.selection_hits for row in checkpoints)
    assert last.ledger["inferences"] == 2 * fresh + replayed + selection
    assert last.budget_spent == 5 * fresh + 2 * replayed + 3 * 32 * len(checkpoints)


def test_continuing_gives_the_checkpoints_of_training_once(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="posteriorbench_methods")
    once, once_dir = train(make_context, tmp_path / "once", quick(update_caps=(8,)))
    assert updates_trained(caplog) == list(range(1, 9))
    caplog.clear()
    first, directory = train(make_context, tmp_path / "twice", quick(update_caps=(2,)))
    assert updates_trained(caplog) == [1, 2]
    done = training_run.read_done(directory)
    assert done is not None and (done.status, done.updates) == ("stopped", 2)
    assert (directory / training_run.RESUME).exists()
    caplog.clear()
    again, _ = train(make_context, tmp_path / "twice", quick(update_caps=(1, 2)))
    assert updates_trained(caplog) == [] and again == first  # used as it is
    second, _ = train(make_context, tmp_path / "twice", quick(update_caps=(2, 8)))
    assert updates_trained(caplog) == list(range(3, 9))
    assert [row.to_json() for row in second] == [row.to_json() for row in once]
    for folder in ("selection", "training"):
        assert files(directory, folder) == files(once_dir, folder)


def test_budgets_are_closed_and_training_ends_at_the_largest(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(budgets=(4096, 8192))
    checkpoints, directory = train(make_context, tmp_path, settings)
    done = training_run.read_done(directory)
    assert done is not None and done.status == "complete"
    assert not (directory / training_run.RESUME).exists()
    assert [row.boundary_for for row in checkpoints if row.boundary_for] == [(4096,), (8192,)]
    last = checkpoints[-1]
    assert last.boundary_for == (8192,)
    reserve = submission.THRESHOLD_TEST_RESERVE
    bound = 2 * 8 * 5 + 3 * 32  # the most that an update and a checkpoint take here
    assert 8192 - reserve - bound < last.budget_spent <= 8192 - reserve
    for budget in settings.budgets:
        candidates = training_run.candidates_at_budget(
            checkpoints, budget, reserve=reserve, require_boundary=True
        )
        assert all(row.budget_spent <= budget - reserve for row in candidates)
    assert any(row.forced for row in checkpoints)
    assert training_run.read_index(directory) == checkpoints


def test_a_smaller_budget_trains_only_as_far_as_its_candidates_need(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    full, _ = train(make_context, tmp_path / "full", quick(budgets=(4096, 8192)))
    small, directory = train(make_context, tmp_path / "small", quick(budgets=(4096,)))
    done = training_run.read_done(directory)
    assert done is not None and done.status == "stopped"
    assert done.updates < full[-1].update
    assert done.next_checkpoint_lower_bound > 4096 - submission.THRESHOLD_TEST_RESERVE

    def candidates(checkpoints: list[training_run.Checkpoint]) -> list[dict[str, Any]]:
        rows = training_run.candidates_at_budget(
            checkpoints, 4096, reserve=submission.THRESHOLD_TEST_RESERVE, require_boundary=True
        )
        return [row.to_json() for row in rows]

    assert candidates(small) == candidates(full)
    later, _ = train(make_context, tmp_path / "small", quick(budgets=(8192,)))
    assert [row.to_json() for row in later] == [row.to_json() for row in full]


def test_checkpoints_at_budget_counts(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    settings = quick(
        update_caps=(1,),
        budgets=(4096,),
        training_budgets=(4096,),
        checkpoint_updates=(1,),
        checkpoint_interval=1000,
        checkpoint_budget_counts=(600, 1200),
    )
    checkpoints, _ = train(make_context, tmp_path, settings)
    scheduled = [row for row in checkpoints if not row.forced]
    assert [row.update for row in scheduled][0] == 1 and len(scheduled) == 3
    for count, row in zip((600, 1200), scheduled[1:], strict=True):
        # Saved after the first update whose training took the budget count to `count`.
        before_its_selection_draws = row.budget_spent - 3 * 32
        assert count <= before_its_selection_draws < count + 2 * 8 * 5


def test_a_directory_of_other_settings_is_refused(
    make_context: Context, quick: Quick, tmp_path: pathlib.Path
) -> None:
    train(make_context, tmp_path, quick(update_caps=(1,)))
    train(make_context, tmp_path, quick(update_caps=(2,)))  # other points, the same training run
    with pytest.raises(training_run.TrainingRunError):
        train(make_context, tmp_path, quick(update_caps=(1,), learning_rate=3e-5))
