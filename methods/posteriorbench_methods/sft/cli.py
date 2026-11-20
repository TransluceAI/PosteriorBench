"""`posteriorbench-methods sft`: the commands `run`, `config` and `reselect` of SFT."""

from __future__ import annotations

from posteriorbench_methods.common import cli_options, reselection, run_context
from posteriorbench_methods.sft import reselect, run, settings

ENTRY = run_context.MethodEntry(
    settings.Settings(),
    run.run,
    original_batch_size=32,
    reselect={reselection.THRESHOLD_1PCT: reselect.reselect},
    generalization_tasks=True,
)

app = cli_options.family_app(
    "sft", help="SFT, Replay and SFT + composition.", methods={"sft": ENTRY}
)
