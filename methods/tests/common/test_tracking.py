"""`tracking`: log lines always; wandb only when a run is attached."""

from __future__ import annotations

import json
from typing import Any

import pytest

from posteriorbench_methods.common import tracking


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.finished = False

    def log(self, values: dict[str, Any]) -> None:
        self.logged.append(values)

    def finish(self) -> None:
        self.finished = True


def test_events_are_logged_with_labels(caplog: pytest.LogCaptureFixture) -> None:
    tracker = tracking.open_tracker(
        project=None, group=None, run_name="r", config={}, labels={"method": "ctl"}
    )
    with caplog.at_level("INFO", logger=tracking.LOGGER_NAME):
        tracker.event("update", step=3, loss=0.5)
    record = json.loads(caplog.records[-1].getMessage())
    assert record == {"event": "update", "method": "ctl", "step": 3, "loss": 0.5}
    tracker.finish()  # no run: nothing to close


def test_numbers_go_to_an_attached_run() -> None:
    run = FakeRun()
    tracker = tracking.Tracker(run)
    tracker.event("update", step=3, loss=0.5, note="text", done=True)
    tracker.finish()
    assert run.logged == [{"update/step": 3, "update/loss": 0.5}]
    assert run.finished
