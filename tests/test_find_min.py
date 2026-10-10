# -*- coding: utf-8 -*-
"""
Tests for finding times of minimum: the Kwee-van Woerden fit, reading light
curves, splitting nights, saving minima, and the interactive window.
"""

from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from EclipsingBinaries import find_min as fm

EXAMPLES = files("EclipsingBinaries.examples")


def _eclipse(t0=0.7, width=0.015, n=120, noise=0.003, seed=0, half_window=1.6):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(t0 - half_window * width, t0 + half_window * width, n))
    flux = 1 - 0.4 * np.exp(-((t - t0) / width) ** 2) + rng.normal(0, noise, n)
    return t, flux


# --- Kwee-van Woerden ------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_finds_a_known_minimum(seed):
    t, f = _eclipse(t0=0.7 + 0.002 * seed, seed=seed)
    result = fm.kwee_van_woerden(t, f)
    assert result.t_kw == pytest.approx(0.7 + 0.002 * seed, abs=20 / 86400)  # within 20 s
    assert result.error is not None and 0 < result.error < 0.001


def test_boundary_is_not_mistaken_for_the_minimum():
    """Flat shoulders used to drag the minimum to the edge of the window."""
    t = np.linspace(0.665, 0.735, 200)  # short flat stretches either side of the eclipse
    f = np.where(np.abs(t - 0.7) < 0.03, 1 - 0.3 * np.cos((t - 0.7) / 0.03 * np.pi / 2), 1.0)
    result = fm.kwee_van_woerden(t, f)
    assert result.t_kw == pytest.approx(0.7, abs=0.001)


def test_minimum_near_the_edge_of_the_trials_is_an_error_not_a_crash():
    t = np.linspace(0, 1, 30)
    f = t.copy()  # no minimum at all, just a slope
    with pytest.raises(fm.MinimumError):
        fm.kwee_van_woerden(t, f, resolution=20, discard=0.0)


def test_too_few_points():
    with pytest.raises(fm.MinimumError, match="Only 3 points"):
        fm.kwee_van_woerden([0.1, 0.2, 0.3], [1, 0.9, 1])


def test_old_KvW_interface_still_works():
    t, f = _eclipse()
    t_kw, err, z = fm.KvW(t, f, resolution=200, need_error=True)
    assert t_kw == pytest.approx(0.7, abs=20 / 86400)
    assert err > 0 and z > 1
    assert fm.KvW(t, f, resolution=200)[1] == 1


def test_KvW_draws_on_an_axis():
    from matplotlib.figure import Figure

    ax = Figure().subplots()
    t, f = _eclipse()
    fm.KvW(t, f, ax=ax, need_error=True)
    assert ax.lines


def test_best_root_raises_instead_of_restarting():
    with pytest.raises(fm.MinimumError):
        fm.best_root([0.0, 1.0], 0.0, 1.0)  # a straight line has no turning point
    assert fm.best_root([1.0, -1.4, 1.0], 0.0, 1.0) == pytest.approx(0.7)


# --- reading and nights -------------------------------------------------------------

def test_reads_night_filters_output():
    t, m, e = fm.read_light_curve(EXAMPLES / "test_B.txt")
    assert len(t) == len(m) == len(e) > 100
    assert np.all(np.diff(t) >= 0)


def test_reads_photometry_csv(tmp_path):
    path = tmp_path / "T_V_data.csv"
    pd.DataFrame({"HJD": [2.5, 1.5], "BJD": [2.5, 1.5], "Source_AMag_T1": [12.1, 12.0],
                  "Source_AMag_T1_Error": [0.01, 0.02]}).to_csv(path, index=False)
    t, m, e = fm.read_light_curve(path)
    assert list(t) == [1.5, 2.5]
    assert list(m) == [12.0, 12.1]
    assert list(e) == [0.02, 0.01]


def test_unreadable_light_curves(tmp_path):
    with pytest.raises(fm.MinimumError, match="doesn't exist"):
        fm.read_light_curve(tmp_path / "missing.txt")
    two = tmp_path / "two.txt"
    two.write_text("1.0 12.0\n2.0 12.1\n")
    with pytest.raises(fm.MinimumError, match="three columns"):
        fm.read_light_curve(two)
    named = tmp_path / "named.csv"
    named.write_text("time,brightness\n1,2\n")
    with pytest.raises(fm.MinimumError, match="time, magnitude and error"):
        fm.read_light_curve(named)


def test_nights_from_the_examples():
    nights = fm.load_nights([EXAMPLES / f"test_{f}.txt" for f in "BVR"], list("BVR"))
    assert len(nights) == 4
    assert [n.index for n in nights] == [0, 1, 2, 3]
    night = nights[0]
    assert set(night.bands) == {"B", "V", "R"}
    t, f, e = night.points()
    assert np.all((t >= 0) & (t < 1))
    assert night.count(["B"]) < night.count()
    assert night.date.startswith("2018-")
    assert "points over" in night.describe()


def test_filters_with_different_nights_are_kept_apart(tmp_path):
    """The old code assumed every filter had the same nights."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("2458379.6 12.0 0.01\n2458379.7 12.1 0.01\n2458380.6 12.0 0.01\n")
    b.write_text("2458381.6 12.0 0.01\n2458381.7 12.2 0.01\n")
    nights = fm.load_nights([a, b], ["B", "V"])
    assert [sorted(n.bands) for n in nights] == [["B"], ["B"], ["V"]]


def test_flux_is_relative_to_the_brightest_point(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("2458379.6 12.0 0.01\n2458379.7 12.75 0.01\n")
    band = fm.load_nights([path])[0].bands["File 1"]
    assert band.flux == pytest.approx([1.0, 10 ** (-0.3)])


# --- measuring and saving ------------------------------------------------------------

@pytest.fixture
def example_night():
    return fm.load_nights([EXAMPLES / f"test_{f}.txt" for f in "BVR"], list("BVR"))[3]


def test_measure_minimum_on_real_data(example_night):
    lb, rb = fm.guess_boundaries(example_night)
    m = fm.measure_minimum(example_night, lb, rb)
    assert lb < m.kwee.t_kw < rb
    assert m.hjd == pytest.approx(example_night.offset + m.kwee.t_kw)
    assert m.error is not None
    assert m.labels == ("B", "V", "R")
    single = fm.measure_minimum(example_night, lb, rb, labels=("V",))
    assert single.n_points < m.n_points
    assert single.hjd == pytest.approx(m.hjd, abs=0.002)


def test_boundaries_in_either_order(example_night):
    lb, rb = fm.guess_boundaries(example_night)
    assert fm.measure_minimum(example_night, rb, lb).hjd == fm.measure_minimum(example_night, lb, rb).hjd


def test_too_narrow_boundaries(example_night):
    lb, _rb = fm.guess_boundaries(example_night)
    with pytest.raises(fm.MinimumError, match="Widen them"):
        fm.measure_minimum(example_night, lb, lb + 0.001)


def test_minimum_file_round_trip(tmp_path, example_night):
    lb, rb = fm.guess_boundaries(example_night)
    m = fm.measure_minimum(example_night, lb, rb)
    path = tmp_path / fm.minimum_file_name("NSVS 896797", ("B", "V"))
    assert path.name == "NSVS_896797_BV_min.txt"

    saved = fm.MinimumFile(path)
    assert saved.add(m) is False
    assert saved.add(m) is True  # the same eclipse again replaces it
    again = fm.MinimumFile(path)
    assert len(again.entries) == 1
    assert again.entries[0][0] == pytest.approx(m.hjd, abs=1e-6)

    # O-C Plotting reads it with whitespace columns 0 (time) and 2 (error)
    df = pd.read_csv(path, header=None, sep=r"\s+")
    assert df[0][0] == pytest.approx(m.hjd, abs=1e-6)
    assert df[2][0] == pytest.approx(m.error, abs=1e-9)

    again.remove(0)
    assert fm.MinimumFile(path).entries == []


def test_minimum_without_error_is_not_saved(tmp_path, example_night):
    lb, rb = fm.guess_boundaries(example_night)
    m = fm.measure_minimum(example_night, lb, rb)
    m.error = None
    with pytest.raises(fm.MinimumError, match="no error estimate"):
        fm.MinimumFile(tmp_path / "x_min.txt").add(m)


def test_guess_boundaries_surround_the_faintest_part(example_night):
    lb, rb = fm.guess_boundaries(example_night)
    t, f, _ = example_night.points()
    inside = (t > lb) & (t < rb)
    assert f[inside].min() == pytest.approx(f.min(), rel=0.05)
