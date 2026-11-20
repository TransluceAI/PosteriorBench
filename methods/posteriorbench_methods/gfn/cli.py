"""`posteriorbench-methods gfn`: the commands `run`, `config` and `reselect` of GFlowNet."""

from __future__ import annotations

from posteriorbench_methods.common import cli_options, reselection, run_context
from posteriorbench_methods.gfn import policy, reselect, run, settings

app = cli_options.family_app(
    "gfn",
    help="GFlowNet and GFlowNet + composition.",
    methods={
        policy.METHOD: run_context.MethodEntry(
            settings.Settings(),
            run.run,
            original_batch_size=16,
            reselect={
                reselection.THRESHOLD_1PCT: reselect.reselect,
                reselect.HIGHEST_MEAN_LOG_REWARD: reselect.highest_mean_log_reward,
            },
        )
    },
)
