"""`posteriorbench-methods dpg`: the commands `run`, `config` and `reselect` of KL-DPG, JS-DPG
and GDC."""

from __future__ import annotations

from posteriorbench_methods.common import cli_options, reselection, run_context
from posteriorbench_methods.dpg import reselect, run, settings

# The number of responses in one forward pass of the original model, when it computes log p.
ORIGINAL_BATCH_SIZE = 32
FDPG_RESELECT = {
    reselection.THRESHOLD_1PCT: reselect.reselect,
    reselection.FINAL_CHECKPOINT: reselect.final_checkpoint,
}
GDC_RESELECT = {
    reselection.THRESHOLD_1PCT: reselect.reselect,
    reselect.LAST_SAMPLING_DISTRIBUTION: reselect.last_sampling_distribution,
    reselect.FINAL_POLICY: reselect.final_policy,
}

app = cli_options.family_app(
    "dpg",
    help="KL-DPG, JS-DPG and GDC, each with composition.",
    methods={
        "kl-dpg": run_context.MethodEntry(
            settings.FdpgSettings(objective="kl"),
            run.run,
            original_batch_size=ORIGINAL_BATCH_SIZE,
            reselect=FDPG_RESELECT,
        ),
        "js-dpg": run_context.MethodEntry(
            settings.FdpgSettings(objective="js"),
            run.run,
            original_batch_size=ORIGINAL_BATCH_SIZE,
            reselect=FDPG_RESELECT,
        ),
        "gdc": run_context.MethodEntry(
            settings.GdcSettings(),
            run.run,
            original_batch_size=ORIGINAL_BATCH_SIZE,
            reselect=GDC_RESELECT,
        ),
    },
)
