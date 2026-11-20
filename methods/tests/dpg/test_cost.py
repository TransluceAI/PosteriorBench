"""What KL-DPG, JS-DPG and GDC pay for: the inference count and the budget count of each event."""

from __future__ import annotations

import numpy as np
import pytest
from posteriorbench import ledger

from posteriorbench_methods.common import charges, composition, signals
from posteriorbench_methods.dpg import cost


def spent(specs: list[charges.ChargeSpec]) -> tuple[float, float]:
    """(inference count, budget count) of the charges."""
    target = ledger.Ledger()
    charges.apply(target, specs)
    return target.inferences, target.budget


@pytest.mark.parametrize("hits", [0, 17, 256])
def test_an_fdpg_update(hits: int) -> None:
    kl = cost.fdpg_update(256, hits, objective="kl", has_gradient=True)
    js = cost.fdpg_update(256, hits, objective="js", has_gradient=True)
    assert spent(kl) == (512 + hits, 1024)
    assert spent(js) == (512 + hits, 1280)
    for objective in ("kl", "js"):
        no_step = cost.fdpg_update(256, hits, objective=objective, has_gradient=False)
        assert spent(no_step) == (256 + hits, 768)


@pytest.mark.parametrize("hits", [0, 5, 128])
def test_selection_draws_and_gdc(hits: int) -> None:
    assert spent(cost.checkpoint_selection_draws(128, hits)) == (128 + hits, 384)
    assert spent(cost.gdc_update(2048, hits)) == (2048 + 2 * hits, 2048 + 4 * hits)
    assert spent(cost.gdc_fit_check(5120, hits)) == (5120 + 2 * hits, 5120 + 3 * hits)


def test_a_run_of_kl_dpg_at_32768_and_gdc_until_its_first_fit_check() -> None:
    """KL-DPG at B = 2^15 with 1,300 training hits and 250 selection hits, whose first test
    passes (the example of the README); GDC until its second candidate, with 300 training hits
    and 80 hits in the fit check."""
    kl = ledger.Ledger()
    for hits in [81] * 15 + [85]:
        charges.apply(kl, cost.fdpg_update(256, hits, objective="kl", has_gradient=True))
    for hits in (40, 40, 40, 40, 45, 45):
        charges.apply(kl, cost.checkpoint_selection_draws(128, hits))
    assert (kl.inferences, kl.budget) == (10_510, 18_688)
    charges.apply(kl, charges.threshold_test(164))
    assert (kl.inferences, kl.budget) == (10_674, 18_852)
    gdc = ledger.Ledger()
    for hits in (40, 45):
        charges.apply(gdc, cost.checkpoint_selection_draws(128, hits))
    for _ in range(10):
        charges.apply(gdc, cost.gdc_update(2048, 30))
    charges.apply(gdc, cost.gdc_fit_check(5120, 80))
    assert (gdc.inferences, gdc.budget) == (26_701, 27_808)


def test_more_hits_than_draws_are_refused() -> None:
    with pytest.raises(charges.ChargeError):
        cost.fdpg_update(8, 9, objective="kl", has_gradient=True)
    with pytest.raises(charges.ChargeError):
        cost.gdc_update(8, 9)
    with pytest.raises(charges.ChargeError):
        cost.gdc_fit_check(8, -1)


def test_composition_adds_its_selection_draws_and_tests() -> None:
    """16 settings with 5 hits each among their 16 selection draws, and a first test that
    passes."""

    def draw(setting: composition.Setting) -> signals.SelectionDraws:
        hit = np.arange(16) < 5
        return signals.SelectionDraws(hit, np.full(16, -30.0), np.full(16, -33.0))

    outcome = composition.choose_setting(draw, lambda setting, candidate, test: test.draws)
    first = outcome.setting_of(outcome.submission.tests[0])
    per_draw = first.inferences_per_response()
    assert outcome.total_selection_hits == 80
    assert spent(cost.composition_charges(outcome, response_limit=50)) == (
        496 + 80 + 164 * per_draw,
        496 + 256 + 512 + 164 * per_draw,
    )
