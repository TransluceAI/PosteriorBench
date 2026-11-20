"""The training runs of KL-DPG, JS-DPG and GDC on the tiny model: their checkpoints, when they
end, and how they continue."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

from posteriorbench_methods.common import run_context, run_record, tracking, training_run
from posteriorbench_methods.dpg import gdc, run, settings

Checkpoints = list[training_run.Checkpoint]
FDPG: dict[str, Any] = {
    "training_budget": 6000,
    "responses_per_update": 32,
    "baseline_group": 12,
    "learning_rate": 3e-3,
    "warmup_updates": 2,
    "selection_draws": 32,
}
GDC: dict[str, Any] = {
    "training_budget": 6000,
    "responses_per_update": 64,
    "loss_divisor": 16,
    "fit_check_interval": 2,
    "fit_check_draws": 96,
    "learning_rate": 3e-3,
    "warmup_updates": 2,
    "selection_draws": 32,
}


def settings_of(method: str, budgets: tuple[int, ...], **changes: Any) -> settings.Settings:
    if method == "gdc":
        return settings.GdcSettings(budgets=budgets, **(GDC | changes))
    objective = "js" if method == "js-dpg" else "kl"
    return settings.FdpgSettings(objective=objective, budgets=budgets, **(FDPG | changes))


def train(context: run_context.RunContext, chosen: settings.Settings) -> Checkpoints:
    checkpoints, _ = run.train(context, chosen)
    return checkpoints


def directory(context: run_context.RunContext, method: str) -> pathlib.Path:
    return context.training_run_dir(method)


def test_fdpg_saves_checkpoints_on_its_schedule_and_closes_the_training_budget(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, "kl-dpg")
    checkpoints = train(context, settings_of("kl-dpg", (6000,)))
    assert [c.checkpoint_id for c in checkpoints] == [
        "update_000000",
        "update_000001",
        "update_000002",
        "update_000004",
        "update_000008",
        "update_000016",
        "update_000026",
    ]
    # A checkpoint takes 96 passes and an update of KL-DPG 128.
    assert [c.budget_spent for c in checkpoints] == [96, 320, 544, 896, 1504, 2624, 4000]
    assert all(c.budget_spent == c.ledger["budget"] for c in checkpoints)
    # Update 27 and its checkpoint could take 224 passes more than the 4,088 that remain.
    last = checkpoints[-1]
    assert last.forced and last.boundary_for == (6000,)
    assert not any(c.forced or c.boundary_for for c in checkpoints[:-1])
    where = directory(context, "kl-dpg")
    assert training_run.read_done(where) == training_run.Done("complete", 26, float("inf"))
    assert not (where / "resume.pt").exists()
    assert training_run.read_index(where) == checkpoints
    for checkpoint in checkpoints:
        draws = training_run.read_selection_draws(where, checkpoint.checkpoint_id)
        assert len(draws) == 32 and sum(row["hit"] for row in draws) == checkpoint.selection_hits
    training_draws = run_record.read_jsonl(where / "training" / "update_000026.jsonl")
    assert len(training_draws) == 32
    assert set(training_draws[0]) == {
        "draw_index",
        "token_ids",
        "hit",
        "seed",
        "logq_generation",
        "logp",
        "logq",
    }


def test_the_first_update_has_learning_rate_zero_and_its_checkpoint_is_stored_once(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, "js-dpg")
    checkpoints = train(context, settings_of("js-dpg", (3000,)))
    where = directory(context, "js-dpg")
    assert checkpoints[0].sampler_id == checkpoints[1].sampler_id
    files = [training_run.checkpoint_path(where, c.checkpoint_id) for c in checkpoints]
    assert files[0].stat().st_ino == files[1].stat().st_ino  # one file, linked twice
    assert len({c.sampler_id for c in checkpoints[1:]}) == len(checkpoints) - 1


def test_fdpg_takes_no_gradient_step_before_the_first_hit(
    make_context: Callable[..., run_context.RunContext],
    rare_dataset: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Without a warmup, the second update has the full learning rate. In this run the first two
    updates draw no hit, so Z is zero and the policy stays the original model."""
    context = make_context(tmp_path, "kl-dpg", dataset=rare_dataset, replicate=6)
    chosen = settings_of("kl-dpg", (2600,), responses_per_update=8, warmup_updates=0)
    checkpoints = train(context, chosen)
    where = directory(context, "kl-dpg")
    hits = [
        sum(row["hit"] for row in run_record.read_jsonl(path))
        for path in sorted((where / "training").glob("update_*.jsonl"))
    ]
    assert hits[:4] == [0, 0, 1, 1]
    assert [c.update for c in checkpoints] == [0, 1, 2, 4]
    assert [c.extra["log_z"] is None for c in checkpoints] == [True, True, True, False]
    assert len({c.sampler_id for c in checkpoints[:3]}) == 1
    assert checkpoints[3].sampler_id != checkpoints[0].sampler_id
    # Two updates at 3 passes per response, then two at 4.
    assert [c.budget_spent for c in checkpoints] == [96, 216, 336, 336 + 64 + 96]
    forward = [c.ledger["by_category"]["training_forward"][0] for c in checkpoints]
    assert forward == [0, 0, 0, 16]


def test_gdc_saves_r_or_the_policy_after_every_fit_check_and_the_final_policy(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, "gdc")
    checkpoints = train(context, settings_of("gdc", (6000,), learning_rate=1e-2))
    where = directory(context, "gdc")
    first, last = checkpoints[0], checkpoints[-1]
    assert first.checkpoint_id == "r_update_000000" and first.extra["fit_check"] is None
    assert first.budget_spent == 96
    for checkpoint in checkpoints[1:-1]:
        check = gdc.FitCheck(**checkpoint.extra["fit_check"])
        model = "r" if check.policy_is_closer else "pi"
        assert checkpoint.update % 2 == 0 and checkpoint.extra["model"] == model
        assert checkpoint.checkpoint_id == f"{model}_update_{checkpoint.update:06d}"
        rows = run_record.read_jsonl(
            where / "training" / f"fit_check_{checkpoint.update:06d}.jsonl"
        )
        assert check.draws == 96
        assert (len(rows), sum(row["hit"] for row in rows)) == (96, check.hits)
        assert [row["part"] for row in rows] == [0] * 64 + [1] * 32
        assert all((row["logpi"] is not None) == row["hit"] for row in rows)
    # The policy after the last update is the last candidate and closes the training budget.
    assert last.checkpoint_id == f"pi_update_{last.update:06d}" and last.update % 2 == 1
    assert last.extra == {"model": "pi", "fit_check": None} and last.boundary_for == (6000,)
    done = training_run.read_done(where)
    assert done is not None and (done.status, done.updates) == ("complete", last.update)
    update = run_record.read_jsonl(where / "training" / "update_000003.jsonl")
    assert len(update) == 64 and 0 < sum(row["hit"] for row in update) < 64
    for row in update:
        scored = [row[key] is not None for key in ("logp", "logr", "logpi")]
        assert scored == [row["hit"]] * 3


@pytest.mark.parametrize("method", ["kl-dpg", "js-dpg", "gdc"])
def test_a_larger_budget_continues_the_training_run(
    method: str, make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    """Training for a small budget stops early. A later run with a larger budget trains on and
    has the checkpoints of a run that trained for the larger budget at once."""
    shared = make_context(tmp_path / "shared", method)
    fresh = make_context(tmp_path / "fresh", method)
    where = directory(shared, method)
    small = train(shared, settings_of(method, (3000,)))
    stopped = training_run.read_done(where)
    assert stopped is not None and stopped.status == "stopped"
    assert stopped.next_checkpoint_lower_bound > 3000 - 1912 >= small[-1].budget_spent
    assert (where / "resume.pt").exists()
    large = train(shared, settings_of(method, (3000, 5000)))
    at_once = train(fresh, settings_of(method, (5000,)))
    assert large == at_once and large[: len(small)] == small and len(large) > len(small)
    for checkpoint in large:
        name = checkpoint.checkpoint_id
        continued = training_run.load_checkpoint(where, name)
        once = training_run.load_checkpoint(directory(fresh, method), name)
        assert all(torch.equal(continued[key], once[key]) for key in once)
    # A smaller budget now finds its candidates and trains nothing.
    before = (where / "index.jsonl").stat().st_mtime_ns
    assert train(shared, settings_of(method, (4000,))) == large
    assert (where / "index.jsonl").stat().st_mtime_ns == before
    # The training budget itself needs the whole training run.
    complete = train(shared, settings_of(method, (6000,)))
    assert complete[: len(large)] == large and complete[-1].boundary_for == (6000,)
    assert not (where / "resume.pt").exists()


class Interrupted(Exception):
    pass


class InterruptingTracker(tracking.Tracker):
    """A tracker that raises when a given update has been done."""

    def __init__(self, update: int) -> None:
        super().__init__()
        self.update = update

    def event(self, name: str, **values: Any) -> None:
        if name == "update" and values["update"] == self.update:
            raise Interrupted


@pytest.mark.parametrize(("method", "update"), [("kl-dpg", 6), ("js-dpg", 3), ("gdc", 5)])
def test_an_interrupted_training_run_continues_from_its_last_checkpoint(
    method: str,
    update: int,
    make_context: Callable[..., run_context.RunContext],
    tmp_path: pathlib.Path,
) -> None:
    chosen = settings_of(method, (4000,))
    context = make_context(tmp_path / "interrupted", method)
    broken = dataclasses.replace(context, tracker=InterruptingTracker(update))
    with pytest.raises(Interrupted):
        train(broken, chosen)
    where = directory(context, method)
    assert training_run.read_done(where) is None and (where / "resume.pt").exists()
    saved = training_run.read_index(where)
    assert 0 < saved[-1].update < update
    continued = train(context, chosen)
    assert continued == train(make_context(tmp_path / "at_once", method), chosen)
    assert continued[: len(saved)] == saved


def test_a_training_run_with_other_settings_needs_another_directory(
    make_context: Callable[..., run_context.RunContext], tmp_path: pathlib.Path
) -> None:
    context = make_context(tmp_path, "kl-dpg")
    train(context, settings_of("kl-dpg", (3000,)))
    with pytest.raises(training_run.TrainingRunError):
        train(context, settings_of("kl-dpg", (3000,), learning_rate=1e-3))
    with pytest.raises(training_run.TrainingRunError):
        train(context, settings_of("kl-dpg", (3000,), training_budget=7000))


def test_gdc_steps_adam_on_a_zero_gradient_when_an_update_has_no_hit() -> None:
    """Zero gradients move no weight while Adam's moments are zero, and they move the weights
    once an update with hits has set the moments."""
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    start = model.weight.detach().clone()
    gdc.set_zero_gradients(model)
    optimizer.step()
    assert torch.equal(model.weight, start)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 3)).sum().backward()
    gdc.check_every_weight_has_a_gradient(model)
    optimizer.step()
    after_hits = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    gdc.set_zero_gradients(model)
    optimizer.step()
    assert not torch.equal(model.weight, after_hits)
    optimizer.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError):
        gdc.check_every_weight_has_a_gradient(model)
