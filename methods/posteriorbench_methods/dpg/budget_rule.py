"""Which checkpoints a budget can use, and when training ends.

The candidates at a budget B are the checkpoints saved while the budget count, their selection
draws included, was at most the limit of B, B less the passes set aside for its threshold tests
(`training_run.candidate_limit`).

Training decides before each update whether to go on, from what the update can cost (`cost`).

- It ends when the update, at its most expensive and with every checkpoint it may have to save,
  could take the budget count beyond the limit of the training budget (`fdpg_training_ends`,
  `gdc_training_ends`). The policy as it is then closes the training budget.
- A run whose budgets are all below the training budget stops earlier: as soon as the next
  checkpoint, at its cheapest, would come too late to be a candidate at any of them
  (`fdpg_next_checkpoint_at_least`, `gdc_next_checkpoint_at_least`). The checkpoints saved until
  then are those that training to the end would have saved within these budgets.

KL-DPG and JS-DPG save a checkpoint before the first update, after updates 1, 2, 4, 8 and 16, and
then after every 32nd update. GDC saves one before the first update and after every fit check.
"""

from __future__ import annotations

from posteriorbench_methods.common import charges, training_run
from posteriorbench_methods.dpg import cost
from posteriorbench_methods.dpg import settings as dpg_settings

FDPG_EARLY_CHECKPOINTS = (0, 1, 2, 4, 8, 16)
FDPG_CHECKPOINT_INTERVAL = 32


def checkpoint_passes(selection_draws: int) -> float:
    """The passes that saving one checkpoint takes: its selection draws, with their p and q."""
    return charges.passes(cost.checkpoint_selection_draws(selection_draws, 0))


def fdpg_checkpoint_due(updates: int) -> bool:
    """Whether KL-DPG and JS-DPG save a checkpoint after `updates` updates."""
    return updates in FDPG_EARLY_CHECKPOINTS or updates % FDPG_CHECKPOINT_INTERVAL == 0


def fdpg_training_ends(spent: float, settings: dpg_settings.FdpgSettings) -> bool:
    """Whether KL-DPG or JS-DPG ends its training at a budget count of `spent`: one more update
    with a gradient step, and a checkpoint after it, might not fit in the training budget."""
    update = cost.fdpg_update(
        settings.responses_per_update, 0, objective=settings.objective, has_gradient=True
    )
    most = charges.passes(update) + checkpoint_passes(settings.selection_draws)
    return spent + most > training_run.candidate_limit(settings.training_budget)


def fdpg_next_checkpoint_at_least(
    spent: float, updates: int, settings: dpg_settings.FdpgSettings
) -> float:
    """The least budget count at which KL-DPG or JS-DPG, after `updates` updates and at a budget
    count of `spent`, can save its next checkpoint: every update until then without a gradient
    step."""
    next_checkpoint = updates + 1
    while not fdpg_checkpoint_due(next_checkpoint):
        next_checkpoint += 1
    update = cost.fdpg_update(
        settings.responses_per_update, 0, objective=settings.objective, has_gradient=False
    )
    return (
        spent
        + (next_checkpoint - updates) * charges.passes(update)
        + checkpoint_passes(settings.selection_draws)
    )


def gdc_fit_check_due(updates: int, settings: dpg_settings.GdcSettings) -> bool:
    """Whether a fit check follows update number `updates` of GDC."""
    return updates % settings.fit_check_interval == 0


def gdc_training_ends(spent: float, updates: int, settings: dpg_settings.GdcSettings) -> bool:
    """Whether GDC, after `updates` updates and at a budget count of `spent`, ends its training:
    one more update might not fit in the training budget. The update is taken at its most
    expensive: every draw a hit, in the update and in the fit check if one follows it, a
    candidate after the fit check, and the final policy as a candidate when training ends."""
    most = charges.passes(
        cost.gdc_update(settings.responses_per_update, settings.responses_per_update)
    )
    if gdc_fit_check_due(updates + 1, settings):
        most += charges.passes(
            cost.gdc_fit_check(settings.fit_check_draws, settings.fit_check_draws)
        )
        most += checkpoint_passes(settings.selection_draws)
    most += checkpoint_passes(settings.selection_draws)
    return spent + most > training_run.candidate_limit(settings.training_budget)


def gdc_next_checkpoint_at_least(
    spent: float, updates: int, settings: dpg_settings.GdcSettings
) -> float:
    """The least budget count at which GDC, after `updates` updates and at a budget count of
    `spent`, can save its next candidate, which follows the next fit check: no hit until
    then."""
    interval = settings.fit_check_interval
    next_fit_check = (updates // interval + 1) * interval
    return (
        spent
        + (next_fit_check - updates)
        * charges.passes(cost.gdc_update(settings.responses_per_update, 0))
        + charges.passes(cost.gdc_fit_check(settings.fit_check_draws, 0))
        + checkpoint_passes(settings.selection_draws)
    )
