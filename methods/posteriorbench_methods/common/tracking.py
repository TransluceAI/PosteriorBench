"""Progress of a run: JSON log lines always, a wandb run only with `--wandb-project`.

Every event is logged as one JSON line (logger "posteriorbench_methods", level INFO) with the run's
labels. Names shared by the families: the events `update`, `checkpoint`, `selection_draws` (of
candidates that are not checkpoints), `composition` and `record`, and the values `update` (updates
done), `checkpoint_id`, `selection_hits`, `budget_spent` and `inferences` (the two totals so far).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from posteriorbench_methods.common import errors

LOGGER_NAME = "posteriorbench_methods"


class TrackingError(errors.MethodsError, RuntimeError):
    """wandb tracking was asked for but cannot run (the `wandb` extra is not installed)."""


class Tracker:
    """Where a run's events go: `run` is a wandb run or None, and `labels` are added to every
    event. Constructing one has no side effect; `open_tracker` starts the wandb run."""

    def __init__(self, run: Any = None, *, labels: Mapping[str, Any] | None = None) -> None:
        self.run = run
        self.labels = dict(labels or {})

    def event(self, name: str, **values: Any) -> None:
        """Log the event as a JSON line, and send its numeric values to wandb as "<name>/<key>"
        when a wandb run is attached."""
        record = {"event": name, **self.labels, **values}
        logging.getLogger(LOGGER_NAME).info(json.dumps(record, default=str))
        if self.run is not None:
            numbers = {
                f"{name}/{key}": value
                for key, value in values.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if numbers:
                self.run.log(numbers)

    def finish(self) -> None:
        """Close the wandb run, if any."""
        if self.run is not None:
            self.run.finish()


def open_tracker(
    *,
    project: str | None,
    group: str | None,
    run_name: str,
    config: Mapping[str, Any],
    labels: Mapping[str, Any] | None = None,
) -> Tracker:
    """A tracker for one run. With `project` ("entity/project" or "project") it starts a wandb run,
    which needs the `wandb` extra and its credentials; without it the tracker only logs."""
    if project is None:
        return Tracker(labels=labels)
    try:
        import wandb  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise TrackingError(
            "--wandb-project needs wandb: install the `wandb` extra of posteriorbench-methods"
        ) from error
    entity, _, name = project.rpartition("/")
    run = wandb.init(
        project=name, entity=entity or None, name=run_name, group=group, config=dict(config)
    )
    return Tracker(run, labels=labels)
