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
    # The top of the orbital period's own peak, as a search of a narrow range around it gives
    assert result.best_period == pytest.approx(2 * result.doubled_from, abs=result.half_width)
    narrow = pr.search(example_bvr, 0.3, 0.33, "ANOVA", oversample=200)
    assert result.best_power == pytest.approx(narrow.best_power, rel=1e-3)


def test_lomb_scargle_keeps_the_doubled_peak(example_bvr):
    # One sine wave has no peak at the orbital period, so the doubled peak is kept
    result = pr.search(example_bvr, 0.1, 1.0, "Lomb-Scargle")
    assert result.best_period == pytest.approx(2 * result.doubled_from)
    assert result.period == pytest.approx(TRUE_PERIOD, abs=3e-5)


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


# --- the period's error -------------------------------------------------------------------

def test_period_error_matches_the_scatter_between_light_curves():
    # Light curves that differ only in their noise: the fitted periods should scatter
    # by about the error each fit reports
    true = 0.4567891
    found, errors = [], []
    for seed in range(30):
        fit = pr.period_error(_synthetic(period=true, n=800, noise=0.02, seed=seed), true * (1 + 2e-7))
        found.append(fit.period)
        errors.append(fit.error)
    ratio = np.std(found) / np.mean(errors)
    assert 0.7 < ratio < 1.4
    assert abs(np.mean(found) - true) < 2 * np.mean(errors)


def test_period_error_uses_enough_terms_for_sharp_eclipses():
    fit = pr.period_error(_synthetic(), 0.4567891)
    assert fit.harmonics >= 10       # narrow eclipses need many
    assert fit.rms == pytest.approx(0.01, rel=0.15)
    assert pr.period_error(_synthetic(), 0.4567891, harmonics=3).harmonics == 3


def test_period_error_shrinks_with_more_data():
    short = pr.period_error(_synthetic(span=50, seed=1), 0.4567891).error
    long = pr.period_error(_synthetic(span=400, seed=1), 0.4567891).error
    assert long < short / 4      # about baseline to the power -3/2


def test_period_error_gives_each_filter_its_own_shape(example_bvr):
    result = pr.search(example_bvr, 0.1, 1.0)
    together = pr.period_error(example_bvr, result.best_period)
    assert together is not None and together.error > 0
    assert result.error == pytest.approx(together.error)
    # The period to use is the fit's, which stays on the peak it started from
    assert result.fit_period == pytest.approx(together.period)
    assert result.period == result.fit_period
    assert abs(result.fit_period - result.best_period) < result.half_width
    # A period that isn't the light curve's doesn't give an error
    assert pr.period_error(example_bvr, 0.2) is None
    assert pr.period_error(example_bvr.select(["B"]).__class__(
        example_bvr.time[:5], example_bvr.mag[:5], example_bvr.err[:5], example_bvr.label[:5]), 0.3174) is None


def test_saved_period_has_its_errors_and_peak(tmp_path, example_bvr):
    result = pr.search(example_bvr.select(["B"]), 0.1, 1.0)
    path = tmp_path / "p.txt"
    pr.save_period(path, "NSVS 896797", result.period, result, error=result.error,
                   peak=result.best_period, postmortem=result.postmortem_error)
    text = path.read_text()
    assert f"period_error_days\t{result.error:.9f}" in text
    assert f"periodogram_peak_days\t{result.best_period:.9f}" in text
    assert f"periodogram_peak_error_days\t{result.postmortem_error:.9f}" in text
    assert pr.read_period(path) == pytest.approx(result.period, abs=1e-9)
    pr.save_period(path, "NSVS 896797", result.best_period)
    assert "period_error_days" not in path.read_text() and "peak" not in path.read_text()


# --- Schwarzenberg-Czerny's post-mortem error --------------------------------------------------

def test_postmortem_error_matches_the_scatter_of_the_peak():
    # For a Multi-harmonic periodogram the peak's own scatter between noise
    # realisations is what the post-mortem error should give
    true = 0.4567891
    peaks, errors = [], []
    for seed in range(20):
        result = pr.search(_synthetic(period=true, n=800, noise=0.02, seed=seed), 0.3, 0.6, "Multi-harmonic")
        peaks.append(result.best_period)
        errors.append(result.postmortem_error)
    ratio = np.std(peaks) / np.mean(errors)
    assert 0.6 < ratio < 1.6


def test_postmortem_error_is_read_off_the_peak_at_the_noise_level():
    curve = _synthetic(n=800, noise=0.02)
    result = pr.search(curve, 0.3, 0.6, "Multi-harmonic")
    post = pr.postmortem_error(curve, result.best_period, result.periods, result.power, "Multi-harmonic")
    assert post.error == pytest.approx(result.postmortem_error)
    assert post.interval[0] < result.best_period < post.interval[1]
    assert post.error == pytest.approx((post.interval[1] - post.interval[0]) / 2)
    # The edges of the interval are where the power is the noise level below the top
    at = pr.score(curve, list(post.interval), "Multi-harmonic")
    assert at == pytest.approx([post.level, post.level], rel=1e-6)
    # (the search's best period is within a grid step of the very top, so a little lower)
    top = float(pr.score(curve, [result.best_period], "Multi-harmonic")[0])
    assert 0.8 * post.noise < top - post.level <= post.noise * (1 + 1e-9)
    # The noise is the periodogram's typical level, far below the peak
    assert post.noise < 0.01 * top


def test_mean_noise_power_ignores_the_peak_and_other_strong_lines():
    periods = 1 / np.linspace(1.0, 3.0, 4001)
    rng = np.random.default_rng(2)
    power = rng.exponential(1.0, periods.size)
    power[2000] = 500                 # the peak itself, at frequency 2
    power[2600] = 300                 # another strong line nearby
    noise = pr.mean_noise_power(periods, power, periods[2000], baseline=100)
    assert noise == pytest.approx(1.0, rel=0.1)
    assert pr.mean_noise_power(periods[:10], power[:10], periods[5], baseline=100) is None


def test_anova_post_mortem_error_is_read_off_a_multi_harmonic_periodogram(example_bvr):
    # Peranso's ANOVA is Schwarzenberg-Czerny's multi-harmonic one
    anova = pr.search(example_bvr, 0.3, 0.33, "ANOVA")
    post = pr.peranso_error(example_bvr, anova.best_period, anova.periods, anova.power, "ANOVA")
    assert post.method == "Multi-harmonic"
    assert anova.postmortem_error == pytest.approx(post.error)
    multi = pr.search(example_bvr, 0.3, 0.33, "Multi-harmonic")
    # (the narrow search has less periodogram to take the noise from, so only roughly)
    assert anova.postmortem_error == pytest.approx(multi.postmortem_error, rel=0.6)


# --- drifts during the night ------------------------------------------------------------------

def _nights(seed, drift=0.0, period=0.3174164):
    """Four nights of six hours like the example data, with optional drifts lasting about an hour."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([n + 0.55 + np.arange(0, 0.25, 1.5 / 1440) for n in (0, 24, 29, 31)]) + 2458379.0
    phase = np.mod((t - t[0]) / period, 1.0)
    dip = lambda p, depth: depth * np.exp(-0.5 * (np.minimum(np.abs(p), 1 - np.abs(p)) / 0.06) ** 2)
    mag = dip(phase, 0.45) + dip(phase - 0.5, 0.3) + rng.normal(0, 0.004, t.size)
    if drift:
        walk = np.zeros_like(t)
        for i in range(1, t.size):
            keep = np.exp(-(t[i] - t[i - 1]) * 24)
            walk[i] = keep * walk[i - 1] + np.sqrt(1 - keep ** 2) * rng.normal(0, drift)
        mag = mag + walk
    return pr.LightCurve(t, mag - np.median(mag), np.full(t.size, 0.004), np.array(["V"] * t.size), ("V",))


def test_red_noise_factor_is_one_for_independent_noise():
    fit = pr.period_error(_nights(0), 0.3174164)
    assert fit.red_noise == pytest.approx(1.0, abs=0.15)
    assert fit.error == pytest.approx(fit.white_error * fit.red_noise)


def test_drifts_raise_the_error_to_match_the_real_scatter():
    true = 0.3174164
    found, errors, white = [], [], []
    for seed in range(20):
        fit = pr.period_error(_nights(seed, drift=0.006), true * (1 + 1e-6))
        found.append(fit.period)
        errors.append(fit.error)
        white.append(fit.white_error)
    # Without the factor the error is several times too small; with it, about right
    assert np.std(found) / np.mean(white) > 2
    assert 0.7 < np.std(found) / np.mean(errors) < 1.5


def test_example_errors_allow_for_its_drifts(example_bvr):
    fit = pr.period_error(example_bvr, TRUE_PERIOD)
    assert fit.red_noise > 1.5
    # Each filter alone should agree with the others within the errors
    single = {lab: pr.period_error(example_bvr.select([lab]), TRUE_PERIOD) for lab in "BVR"}
    for a, b in (("B", "V"), ("B", "R"), ("V", "R")):
        gap = abs(single[a].period - single[b].period)
        assert gap < 2 * np.hypot(single[a].error, single[b].error)


@pytest.mark.parametrize("error, text", [(7.8412e-07, "0.00000078"), (0.0001004, "0.00010"),
                                         (0.023, "0.023"), (1.5, "1.5"), (None, ""), (0.0, "")])
def test_format_error(error, text):
    assert pr.format_error(error) == text
