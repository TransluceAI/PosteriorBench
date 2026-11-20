"""`common.training_run`: files, index, candidates, kept checkpoints, and continuing a toy
training run (training to a stop and continuing later gives the checkpoints of training once)."""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib

import pytest
import torch

from posteriorbench_methods.common import run_record, training_run


def _row(
    k: int, update: int, spent: float, *, sampler: str | None = None
) -> training_run.Checkpoint:
    return training_run.Checkpoint(
        checkpoint_id=training_run.checkpoint_id(update),
        update=update,
        budget_spent=spent,
        ledger={"inferences": spent, "budget": spent},
        sampler_id=sampler or f"sampler{k}",
        selection_draws=16,
        selection_hits=k,
    )


def test_layout(tmp_path: pathlib.Path) -> None:
    out = run_record.training_run_dir(tmp_path, "ctl", "wc_0001/em_dash", 2)
    assert training_run.checkpoint_id(128) == "update_000128"
    assert training_run.checkpoint_path(out, "x") == out / "checkpoints" / "x.pt"
    assert training_run.selection_draws_path(out, "x") == out / "selection" / "x.jsonl"
    assert training_run.training_draws_path(out, 3) == out / "training" / "update_000003.jsonl"


def test_sampler_id() -> None:
    a = {"w": torch.ones(2, 3), "b": torch.zeros(3)}
    assert training_run.sampler_id(a) == training_run.sampler_id(
        {k: v.clone() for k, v in a.items()}
    )
    assert training_run.sampler_id(a) != training_run.sampler_id(
        {"w": torch.ones(2, 3), "b": torch.ones(3)}
    )
    assert training_run.sampler_id(a) != training_run.sampler_id(
        {"w": torch.ones(3, 2), "b": torch.zeros(3)}
    )
    assert len(training_run.sampler_id({"empty": torch.zeros(0)})) == 64


def test_checkpoint_files(tmp_path: pathlib.Path) -> None:
    tensors = {"w": torch.randn(4, 4)}
    saved_id = training_run.save_checkpoint(tmp_path, "a", tensors)
    assert saved_id == training_run.sampler_id(tensors)
    loaded = training_run.load_checkpoint(tmp_path, "a")
    assert torch.equal(loaded["w"], tensors["w"])
    earlier = [dataclasses.replace(_row(1, 1, 0.0, sampler=saved_id), checkpoint_id="a")]
    assert training_run.save_checkpoint(tmp_path, "b", tensors, earlier=earlier) == saved_id
    first, second = (training_run.checkpoint_path(tmp_path, c) for c in ("a", "b"))
    assert os.stat(first).st_ino == os.stat(second).st_ino  # one file on disk
    first.unlink()
    assert torch.equal(training_run.load_checkpoint(tmp_path, "b")["w"], tensors["w"])
    with pytest.raises(training_run.TrainingRunError):
        training_run.load_checkpoint(tmp_path, "a")
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    saved = training_run.trainable_tensors(model)
    assert set(saved) == {"weight"}
    with torch.no_grad():
        model.weight.zero_()
    training_run.restore_trainable_tensors(model, saved)
    assert torch.equal(model.weight, saved["weight"])
    with pytest.raises(training_run.TrainingRunError):
        training_run.restore_trainable_tensors(model, {"bias": torch.zeros(2)})


def test_index_and_selection_draws(tmp_path: pathlib.Path) -> None:
    rows = [_row(0, 1, 100.0), _row(1, 2, 200.0)]
    rows = training_run.mark_boundary(rows, 4096)
    rows[1] = dataclasses.replace(rows[1], forced=True, extra={"sampler": "pi"})
    training_run.write_index(tmp_path, rows)
    assert training_run.read_index(tmp_path) == rows
    assert training_run.read_index(tmp_path / "missing") == []
    assert rows[1].boundary_for == (4096,) and rows[0].hit_fraction == 0.0
    draws = [{"token_ids": [1, 2], "hit": True, "logq": -1.0, "logp": -2.0, "text": "a b"}]
    training_run.write_selection_draws(tmp_path, rows[0].checkpoint_id, draws)
    assert training_run.read_selection_draws(tmp_path, rows[0].checkpoint_id) == draws
    # As rows of a record's draws/selection.jsonl: each with the id of its checkpoint.
    training_run.write_selection_draws(tmp_path, rows[1].checkpoint_id, draws + draws)
    assert training_run.selection_rows(tmp_path, rows) == [
        {**draw, "candidate_id": candidate}
        for candidate, count in (("update_000001", 1), ("update_000002", 2))
        for draw in draws * count
    ]
    with pytest.raises(training_run.CandidatesError):
        training_run.mark_boundary([], 1)


def test_candidates() -> None:
    rows = [
        _row(k, u, b) for k, (u, b) in enumerate([(1, 100.0), (2, 300.0), (4, 700.0), (8, 1500.0)])
    ]
    rows = training_run.mark_boundary(rows[:3], 2700) + rows[3:]
    candidates = training_run.candidates_at_budget(rows, 2700, reserve=1912, require_boundary=True)
    assert [r.update for r in candidates] == [1, 2, 4]
    assert [
        r.update
        for r in training_run.candidates_at_budget(rows, 2300, reserve=1912, require_boundary=False)
    ] == [1, 2]
    with pytest.raises(training_run.CandidatesError):
        training_run.candidates_at_budget(rows, 2300, reserve=1912, require_boundary=True)
    with pytest.raises(training_run.BudgetError):
        training_run.candidates_at_budget(rows, 1900, reserve=1912, require_boundary=False)
    assert [r.update for r in training_run.candidates_at_update_cap(rows, 2)] == [1, 2]
    with pytest.raises(training_run.CandidatesError):
        training_run.candidates_at_update_cap(rows, 3)
    with pytest.raises(training_run.CandidatesError):
        training_run.candidates_at_update_cap(rows, 8, allow_boundary=False)
    groups = training_run.group_budgets(rows, [2000, 2100, 2300, 2700, 4000], reserve=1912)
    assert [(g.last_checkpoint, g.count, g.budgets) for g in groups] == [
        ("update_000001", 1, (2100,)),
        ("update_000002", 2, (2300,)),
        ("update_000004", 3, (2700,)),
        ("update_000008", 4, (4000,)),
    ]
    assert groups[0].to_json()["budgets"] == [2100]
    with pytest.raises(training_run.CandidatesError):
        training_run.check_chronological([rows[1], rows[0]])


def test_kept_checkpoints(tmp_path: pathlib.Path) -> None:
    rows = [
        _row(0, 1, 1.0),
        _row(1, 2, 2.0, sampler="sampler0"),
        _row(2, 3, 3.0),
        _row(3, 4, 4.0),
    ]
    for row in rows:
        training_run.save_checkpoint(tmp_path, row.checkpoint_id, {"w": torch.zeros(1)})
    tested, returned = ["sampler2"], ["sampler0"]
    assert training_run.samplers_to_keep("all", testable=tested, returned=returned) is None
    assert training_run.prune_checkpoints(tmp_path, rows, None) == []
    assert training_run.samplers_to_keep("returned", testable=tested, returned=returned) == {
        "sampler0"
    }
    keep = training_run.samplers_to_keep("testable", testable=tested, returned=returned)
    assert keep == {"sampler0", "sampler2"}
    assert training_run.prune_checkpoints(tmp_path, rows, keep) == ["update_000004"]
    with pytest.raises(ValueError):
        training_run.samplers_to_keep("some", testable=[], returned=[])  # type: ignore


def test_settings_and_done(tmp_path: pathlib.Path) -> None:
    training_run.check_settings(tmp_path, {"lr": 1e-4, "budgets": (1, 2)})
    training_run.check_settings(tmp_path, {"lr": 1e-4, "budgets": [1, 2]})
    with pytest.raises(training_run.TrainingRunError):
        training_run.check_settings(tmp_path, {"lr": 1e-3, "budgets": [1, 2]})
    assert training_run.read_done(tmp_path) is None
    done = training_run.Done("complete", 12, math.inf, {"reason": "end"})
    training_run.write_done(tmp_path, done)
    assert training_run.read_done(tmp_path) == done
    training_run.clear_done(tmp_path)
    assert training_run.read_done(tmp_path) is None


def test_serves_and_stops() -> None:
    rows = [_row(0, 1, 100.0), _row(1, 2, 300.0)]
    stopped = training_run.Done("stopped", 3, 450.0)
    request = training_run.Request(budgets=(2300,), reserve=1912)  # candidates up to 388
    assert training_run.serves(rows, stopped, request)
    assert not training_run.serves(
        rows, stopped, training_run.Request(budgets=(2400,), reserve=1912)
    )
    assert training_run.serves(rows, stopped, training_run.Request(update_caps=(2,)))
    assert not training_run.serves(rows, stopped, training_run.Request(update_caps=(4,)))
    assert not training_run.serves(rows, stopped, training_run.Request(to_end=True))
    assert not training_run.serves(rows, None, request)
    assert training_run.serves(
        rows, training_run.Done("complete", 9, math.inf), training_run.Request(to_end=True)
    )
    assert training_run.stop_reached(request, updates=3, next_checkpoint_lower_bound=389.0)
    assert not training_run.stop_reached(request, updates=3, next_checkpoint_lower_bound=388.0)
    assert not training_run.stop_reached(
        training_run.Request(), updates=99, next_checkpoint_lower_bound=1e9
    )
    with pytest.raises(ValueError):
        training_run.candidates_are_final(rows, stopped)


END = 12  # updates of the toy training run
SELECTION = 3.0  # budget of one checkpoint's selection draws
UPDATE = 10.0  # budget of one update


def _spent(updates: int, checkpoints: int) -> float:
    return UPDATE * updates + SELECTION * checkpoints


def _train_toy(
    directory: pathlib.Path, request: training_run.Request
) -> tuple[list[training_run.Checkpoint], int]:
    """A toy training method: a linear model, one SGD step per update on data seeded by the update
    and a draw from the global generator; checkpoints at update 1 and at every even update. It
    uses a training run as a family does: `check_settings`, `serves`, the resume state,
    `stop_reached`, `Done`."""
    training_run.check_settings(directory, {"lr": 0.1, "end": END})
    rows = training_run.read_index(directory)
    done = training_run.read_done(directory)
    if training_run.serves(rows, done, request):
        return rows, 0
    training_run.clear_done(directory)
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    update = 0
    rows = []
    state = training_run.load_resume(directory)
    if state is not None:
        training_run.restore_trainable_tensors(model, state.tensors)
        optimizer.load_state_dict(state.optimizer)
        training_run.restore_rng_state(state.rng)
        update = int(state.loop["update"])
        rows = [training_run.Checkpoint.from_json(r) for r in state.loop["rows"]]

    def save() -> None:
        training_run.save_resume(
            directory,
            training_run.ResumeState(
                training_run.trainable_tensors(model), optimizer.state_dict(),
                training_run.capture_rng_state(),
                {"update": update, "rows": [r.to_json() for r in rows]},
            ),
        )  # fmt: skip

    ran = 0
    while True:
        bound = _spent(update + 1, len(rows) + 1)
        if update >= END:
            status: training_run.DoneStatus = "complete"
            break
        if training_run.stop_reached(request, updates=update, next_checkpoint_lower_bound=bound):
            status = "stopped"
            break
        update += 1
        ran += 1
        x = torch.randn(4, 3, generator=torch.Generator().manual_seed(update))
        loss = (model(x) - torch.rand(1)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update == 1 or update % 2 == 0:
            cid = training_run.checkpoint_id(update)
            saved = training_run.save_checkpoint(
                directory, cid, training_run.trainable_tensors(model)
            )
            rows.append(
                training_run.Checkpoint(cid, update, _spent(update, len(rows) + 1), {}, saved, 4, 1)
            )
            training_run.write_index(directory, rows)
            save()
    save()
    complete = status == "complete"
    training_run.write_done(
        directory,
        training_run.Done(
            status, update, math.inf if complete else _spent(update + 1, len(rows) + 1)
        ),
    )
    if complete:
        training_run.delete_resume(directory)
    return rows, ran


def test_continuing_equals_training_once(tmp_path: pathlib.Path) -> None:
    once, ran_once = _train_toy(tmp_path / "once", training_run.Request(to_end=True))
    assert ran_once == END and not (tmp_path / "once" / training_run.RESUME).exists()
    small = training_run.Request(budgets=(60,), reserve=5)
    first, ran_first = _train_toy(tmp_path / "twice", small)
    assert 0 < ran_first < END and training_run.read_done(tmp_path / "twice") is not None
    again, ran_again = _train_toy(tmp_path / "twice", small)
    assert ran_again == 0 and again == first  # used as it is
    second, ran_second = _train_toy(tmp_path / "twice", training_run.Request(to_end=True))
    assert ran_first + ran_second == END
    assert [r.to_json() for r in second] == [r.to_json() for r in once]
    candidates = training_run.candidates_at_budget(second, 60, reserve=5, require_boundary=False)
    assert candidates == training_run.candidates_at_budget(
        first, 60, reserve=5, require_boundary=False
    )
    served, ran = _train_toy(tmp_path / "once", small)
    assert ran == 0 and served == once
