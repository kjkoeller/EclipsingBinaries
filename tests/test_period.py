# -*- coding: utf-8 -*-
"""
Tests for the period search: the statistics, the trial-period grid, the
check for half the orbital period, folding and saving.
"""

from importlib.resources import files

import numpy as np
import pytest
from scipy.stats import f_oneway

from EclipsingBinaries import period as pr

EXAMPLES = files("EclipsingBinaries.examples")
TRUE_PERIOD = 0.31741  # the example system, from every method and filter


def _synthetic(period=0.4567891, secondary=0.3, n=1500, span=200.0, noise=0.01, seed=0):
    """Eclipses of depth 0.5 at phase 0 and `secondary` at phase 0.5."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n)) + 2458000.0
    ph = np.mod((t - 2458000.0) / period, 1)
    m = (0.5 * (np.exp(-(ph / 0.04) ** 2) + np.exp(-((ph - 1) / 0.04) ** 2))
         + secondary * np.exp(-((ph - 0.5) / 0.04) ** 2) + rng.normal(0, noise, n))
    return pr.LightCurve(t, m, np.full(n, noise), np.full(n, "V", dtype=object), ("V",))


@pytest.fixture(scope="module")
def example_bvr():
    return pr.load_light_curves([EXAMPLES / f"test_{f}.txt" for f in "BVR"], list("BVR"))


# --- statistics ------------------------------------------------------------------------

def test_anova_matches_scipy(example_bvr):
    c = example_bvr.select(["B"])
    for period in (0.2, 0.31742, 0.5):
        phase = np.mod((c.time - c.time[0]) / period, 1)
        idx = np.minimum((phase * 20).astype(int), 19)
        groups = [c.mag[idx == b] for b in range(20) if (idx == b).sum() >= 3]
        expected = f_oneway(*groups).statistic
        assert pr.anova_statistic(c.time, c.mag, [period])[0] == pytest.approx(expected, rel=1e-9)


def test_anova_with_too_few_filled_bins_is_nan():
    t = np.array([0.0, 0.001, 0.002, 0.003])
    assert np.isnan(pr.anova_statistic(t, np.array([1.0, 2.0, 3.0, 4.0]), [10.0])[0])


@pytest.mark.parametrize("harmonics", [1, 2, 4])
def test_multiharmonic_matches_a_direct_fit(example_bvr, harmonics):
    c = example_bvr.select(["V"])
    period = 0.31742
    phase = np.mod((c.time - c.time[0]) / period, 1)
    X = np.column_stack([np.ones_like(phase)] + [f(2 * np.pi * h * phase)
                                                 for h in range(1, harmonics + 1) for f in (np.cos, np.sin)])
    y = c.mag - c.mag.mean()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    sse = np.sum((y - X @ beta) ** 2)
    sst = np.sum(y ** 2)
    p = X.shape[1]
    expected = ((sst - sse) / (p - 1)) / (sse / (len(y) - p))
    got = pr.multiharmonic_statistic(c.time, c.mag, [period], harmonics=harmonics)[0]
    assert got == pytest.approx(expected, rel=1e-6)


def test_lomb_scargle_handles_any_period_order(example_bvr):
    c = example_bvr.select(["B"])
    periods = np.array([0.3, 0.15871, 0.2])
    power = pr.lomb_scargle_statistic(c.time, c.mag, c.err, periods)
    assert np.argmax(power) == 1


def test_unknown_method(example_bvr):
    with pytest.raises(pr.PeriodError, match="Unknown method"):
        pr.score(example_bvr, [0.3], "Guess")


# --- grid ----------------------------------------------------------------------------------

def test_grid_is_even_in_frequency_and_fine_enough():
    t = np.linspace(0, 100, 500)
    grid = pr.frequency_grid(t, 0.2, 1.0)
    freqs = 1 / grid.periods[::-1]
    assert np.allclose(np.diff(freqs), np.diff(freqs)[0])
    assert np.diff(freqs)[0] <= 1 / (100 * pr.DEFAULT_OVERSAMPLE) * 1.0001
    assert grid.periods[0] == pytest.approx(0.2) and grid.periods[-1] == pytest.approx(1.0)
    assert not grid.coarse


def test_grid_is_thinned_for_huge_searches(monkeypatch):
    monkeypatch.setattr(pr, "MAX_WORK", 1_000_000)
    t = np.linspace(0, 3000, 1000)
    grid = pr.frequency_grid(t, 0.1, 10.0)
    assert grid.coarse
    assert len(grid.periods) <= 1000
    assert grid.oversample < pr.DEFAULT_OVERSAMPLE


@pytest.mark.parametrize("lo, hi, message", [(0, 1, "above zero"), (2, 1, "above zero"),
                                             (0.1, 500, "longer than the observations")])
def test_bad_period_ranges(lo, hi, message):
    with pytest.raises(pr.PeriodError, match=message):
        pr.frequency_grid(np.linspace(0, 100, 50), lo, hi)


# --- searching -------------------------------------------------------------------------

@pytest.mark.parametrize("method", pr.METHODS)
@pytest.mark.parametrize("labels", [["B"], ["V"], ["R"], ["B", "V", "R"]])
def test_example_period_with_every_method_and_filter(example_bvr, method, labels):
    result = pr.search(example_bvr.select(labels), 0.1, 1.0, method)
    assert result.best_period == pytest.approx(TRUE_PERIOD, abs=max(3 * result.half_width, 2e-5))


def test_half_period_peak_is_doubled(example_bvr):
    """ANOVA prefers half the period for alike eclipses; the fold check doubles it."""
    result = pr.search(example_bvr, 0.1, 1.0, "ANOVA")
    assert result.doubled_from == pytest.approx(TRUE_PERIOD / 2, abs=1e-4)
    assert result.best_period == pytest.approx(2 * result.doubled_from)


@pytest.mark.parametrize("method", pr.METHODS)
def test_unequal_eclipses_give_the_orbit(method):
    result = pr.search(_synthetic(secondary=0.3), 0.1, 5.0, method)
    assert result.best_period == pytest.approx(0.4567891, abs=result.half_width)


@pytest.mark.parametrize("method", pr.METHODS)
def test_one_visible_eclipse_is_not_doubled(method):
    result = pr.search(_synthetic(secondary=0.0), 0.1, 5.0, method)
    assert result.best_period == pytest.approx(0.4567891, abs=result.half_width)
    assert result.doubled_from is None


def test_identical_eclipses_stay_at_half():
    """Nothing in the light curve tells these apart; the window asks the user to look."""
    result = pr.search(_synthetic(secondary=0.5), 0.1, 5.0, "ANOVA")
    assert result.best_period == pytest.approx(0.4567891 / 2, abs=result.half_width)


def test_halves_differ_ratio(example_bvr):
    assert pr.halves_differ(example_bvr, TRUE_PERIOD / 2)[0]
    differ, ratio = pr.halves_differ(example_bvr, TRUE_PERIOD)
    assert not differ and ratio < 5


def test_too_few_points():
    c = _synthetic(n=8)
    with pytest.raises(pr.PeriodError, match="at least 10"):
        pr.search(c, 0.1, 1.0)


def test_nearest_peak(example_bvr):
    result = pr.search(example_bvr.select(["B"]), 0.1, 1.0, "Multi-harmonic")
    picked = pr.nearest_peak(result, TRUE_PERIOD + 0.004)
    assert picked == pytest.approx(TRUE_PERIOD, abs=0.001)


# --- folding and files -------------------------------------------------------------------

def test_fold_keeps_precision_for_large_times():
    t = np.array([2458000.0, 2458000.25, 2458000.5])
    assert pr.fold(t, 0.5, 2458000.0) == pytest.approx([0.0, 0.5, 0.0])


def test_save_and_read_period(tmp_path, example_bvr):
    result = pr.search(example_bvr.select(["B"]), 0.1, 1.0)
    path = tmp_path / pr.period_file_name("NSVS 896797")
    assert path.name == "NSVS_896797_period.txt"
    pr.save_period(path, "NSVS 896797", result.best_period, result, ["a.txt"])
    assert pr.read_period(path) == pytest.approx(result.best_period, abs=1e-9)
    text = path.read_text()
    assert "method\tANOVA" in text and "light_curves\ta.txt" in text
    bad = tmp_path / "bad.txt"
    bad.write_text("nothing here\n")
    with pytest.raises(pr.PeriodError):
        pr.read_period(bad)


def test_load_light_curves_takes_off_each_median(example_bvr):
    for label in "BVR":
        assert np.median(example_bvr.mag[example_bvr.label == label]) == pytest.approx(0, abs=1e-12)
    with pytest.raises(pr.PeriodError):
        pr.load_light_curves([])


@pytest.mark.parametrize("method", ["ANOVA", "Multi-harmonic"])
def test_trial_periods_catch_the_top_of_the_peak(example_bvr, method):
    """The plotted periodogram should reach the refined best value, not stop short of it."""
    result = pr.search(example_bvr, 0.1, 1.0, method)
    near = np.abs(result.periods - result.best_period) < 5 * result.half_width
    assert np.nanmax(result.power[near]) >= 0.98 * result.best_power
