"""`posteriorbench-methods ctl`: the commands `run`, `config` and `reselect` of CTL."""

from __future__ import annotations

from posteriorbench_methods.common import cli_options, reselection, run_context
from posteriorbench_methods.ctl import reselect, run, settings

app = cli_options.family_app(
    "ctl",
    help="CTL and CTL + composition.",
    methods={
        "ctl": run_context.MethodEntry(
            settings.Settings(),
            run.run,
            original_batch_size=64,
            reselect={
                reselection.THRESHOLD_1PCT: reselect.reselect,
                reselection.FINAL_CHECKPOINT: reselect.final_checkpoint,
            },
        )
    },
)
