"""Which checkpoints a budget can use, and when the training of KL-DPG, JS-DPG and GDC ends."""

from __future__ import annotations

import dataclasses

from posteriorbench_methods.common import training_run
from posteriorbench_methods.dpg import budget_rule, settings

KL = settings.FdpgSettings(objective="kl")
JS = settings.FdpgSettings(objective="js")
GDC = settings.GdcSettings()
LIMIT = 2**20 - 1912


def test_a_budget_sets_the_threshold_tests_aside() -> None:
    assert training_run.candidate_limit(2**15) == 30_856
    assert budget_rule.checkpoint_passes(128) == 384


def test_the_checkpoint_schedule_of_fdpg() -> None:
    due = [update for update in range(100) if budget_rule.fdpg_checkpoint_due(update)]
    assert due == [0, 1, 2, 4, 8, 16, 32, 64, 96]


def test_fdpg_ends_when_an_update_with_its_checkpoint_might_not_fit() -> None:
    # An update of KL-DPG takes 1,024 passes, one of JS-DPG 1,280, a checkpoint 384.
    assert not budget_rule.fdpg_training_ends(LIMIT - 1408, KL)
    assert budget_rule.fdpg_training_ends(LIMIT - 1407, KL)
    assert not budget_rule.fdpg_training_ends(LIMIT - 1664, JS)
    assert budget_rule.fdpg_training_ends(LIMIT - 1663, JS)


def test_the_next_checkpoint_of_fdpg_at_its_cheapest() -> None:
    """After update 16 of KL-DPG, at a budget count of 18,688, the next checkpoint follows
    update 32: at least 16 updates of 768 passes and a checkpoint. That is beyond the limit of
    B = 2^15, so a run at that budget stops here."""
    at_least = budget_rule.fdpg_next_checkpoint_at_least(18_688, 16, KL)
    assert at_least == 18_688 + 16 * 768 + 384 == 31_360
    assert at_least > training_run.candidate_limit(2**15)
    assert budget_rule.fdpg_next_checkpoint_at_least(0, 2, KL) == 2 * 768 + 384
    assert budget_rule.fdpg_next_checkpoint_at_least(0, 33, JS) == 31 * 768 + 384


def test_the_rules_follow_the_settings() -> None:
    small = dataclasses.replace(KL, responses_per_update=64, selection_draws=32)
    assert budget_rule.fdpg_next_checkpoint_at_least(0, 0, small) == 3 * 64 + 3 * 32
    assert budget_rule.fdpg_training_ends(LIMIT - 4 * 64 - 3 * 32 + 1, small)
    assert not budget_rule.fdpg_training_ends(LIMIT - 4 * 64 - 3 * 32, small)


def test_gdc_ends_when_an_update_at_its_most_expensive_might_not_fit() -> None:
    """An update costs at most 5 x 2,048 passes and the final policy's selection draws 384; before
    an update that a fit check follows, the fit check adds at most 4 x 5,120 and its candidate
    384."""
    plain = 5 * 2048 + 384
    with_fit_check = plain + 4 * 5120 + 384
    assert [budget_rule.gdc_fit_check_due(update, GDC) for update in (9, 10, 11, 20)] == [
        False,
        True,
        False,
        True,
    ]
    assert not budget_rule.gdc_training_ends(LIMIT - plain, 0, GDC)
    assert budget_rule.gdc_training_ends(LIMIT - plain + 1, 0, GDC)
    assert not budget_rule.gdc_training_ends(LIMIT - with_fit_check, 9, GDC)
    assert budget_rule.gdc_training_ends(LIMIT - with_fit_check + 1, 9, GDC)


def test_the_next_candidate_of_gdc_at_its_cheapest() -> None:
    """The next candidate follows the next fit check: updates and a fit check without a hit, and
    its selection draws. Before the first fit check that is beyond the limit of B = 2^14."""
    assert budget_rule.gdc_next_checkpoint_at_least(384, 0, GDC) == 384 + 10 * 2048 + 5120 + 384
    assert budget_rule.gdc_next_checkpoint_at_least(384, 0, GDC) > training_run.candidate_limit(
        2**14
    )
    assert budget_rule.gdc_next_checkpoint_at_least(0, 10, GDC) == 10 * 2048 + 5120 + 384
    assert budget_rule.gdc_next_checkpoint_at_least(0, 17, GDC) == 3 * 2048 + 5120 + 384
