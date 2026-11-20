"""What the steps of one run share, and what a method hands to its composition."""

from __future__ import annotations

import dataclasses

from posteriorbench import ledger

from posteriorbench_methods.common import run_context, run_record
from posteriorbench_methods.prompt import draws, samplers, settings


@dataclasses.dataclass(frozen=True)
class Session:
    """What the steps of one run share."""

    context: run_context.RunContext
    settings: settings.Settings
    factory: samplers.Factory
    draws: draws.Draws


@dataclasses.dataclass(frozen=True)
class Returned:
    """What a method returned at one setting. `spec` is None without a sampler; `status` is the
    outcome of the threshold tests behind it."""

    method: str
    setting: str
    spec: samplers.Spec | None
    status: str
    spend: ledger.Ledger
    recorded: run_record.Recorded
