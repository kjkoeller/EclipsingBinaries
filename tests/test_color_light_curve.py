# -*- coding: utf-8 -*-
"""
Tests for color_light_curve.py

"""

import pytest
import numpy as np

from EclipsingBinaries.color_light_curve import (
    occ2,
    best_tol,
    lin_interp,
    mean_mag,
)


def _uniform_hjd(n, period=1.0, offset=0.0):
    return list(np.linspace(2450000 + offset, 2450000 + period + offset, n, endpoint=False))


def test_lin_interp_at_x1_returns_y1():
    assert lin_interp(1.0, 1.0, 2.0, 10.0, 20.0) == pytest.approx(10.0)


def test_lin_interp_at_x2_returns_y2():
    assert lin_interp(2.0, 1.0, 2.0, 10.0, 20.0) == pytest.approx(20.0)


def test_lin_interp_midpoint():
    assert lin_interp(1.5, 1.0, 2.0, 10.0, 20.0) == pytest.approx(15.0)


def test_lin_interp_quarter_point():
    assert lin_interp(1.25, 1.0, 2.0, 0.0, 4.0) == pytest.approx(1.0)


def test_lin_interp_extrapolation():
    assert lin_interp(3.0, 1.0, 2.0, 0.0, 1.0) == pytest.approx(2.0)


def test_lin_interp_flat_line():
    assert lin_interp(1.7, 1.0, 2.0, 5.0, 5.0) == pytest.approx(5.0)


def test_lin_interp_negative_slope():
    assert lin_interp(1.5, 1.0, 2.0, 10.0, 0.0) == pytest.approx(5.0)


def test_lin_interp_returns_float():
    assert isinstance(lin_interp(1.5, 1.0, 2.0, 0.0, 1.0), float)


def test_mean_mag_single_value_roundtrip():
    mag = 14.5
    assert mean_mag([mag]) == pytest.approx(mag, abs=1e-6)


def test_mean_mag_two_equal_values():
    mag = 12.0
    assert mean_mag([mag, mag]) == pytest.approx(mag, abs=1e-6)


def test_mean_mag_flux_weighted_average():
    mags = [13.0, 14.0]
    expected = -2.5 * np.log10(np.mean(10 ** (-0.4 * np.array(mags))))
    assert mean_mag(mags) == pytest.approx(expected, abs=1e-8)


def test_mean_mag_result_between_min_and_max():
    mags = [12.0, 14.0, 16.0]
    result = mean_mag(mags)
    assert min(mags) <= result <= max(mags)


def test_mean_mag_brighter_pulls_toward_brighter():
    assert mean_mag([10.0, 15.0]) < 12.5


def test_mean_mag_numpy_array_input():
    assert np.isfinite(mean_mag(np.array([13.0, 14.0, 15.0])))


def test_occ2_returns_six_elements():
    B = _uniform_hjd(20)
    V = _uniform_hjd(20, offset=0.005)
    assert len(occ2(B, V, period=1.0, tolerance=0.05)) == 6


def test_occ2_bad_obs_is_integer():
    B = _uniform_hjd(20)
    V = _uniform_hjd(20, offset=0.005)
    assert isinstance(occ2(B, V, period=1.0, tolerance=0.05)[0], int)


def test_occ2_zero_bad_when_well_sampled():
    period = 1.0
    tol = 0.02
    V = list(np.linspace(2450000, 2450001, 10, endpoint=False))
    B = []
    for v in V:
        B.append(v - tol * period * 0.5)
        B.append(v + tol * period * 0.5)
    assert occ2(B, V, period=period, tolerance=tol)[0] == 0


def test_occ2_all_bad_when_no_overlap():
    B = list(np.linspace(2450000, 2450001, 10))
    V = list(np.linspace(2460000, 2460001, 10))
    assert occ2(B, V, period=1.0, tolerance=0.01)[0] == len(V)


def test_occ2_before_after_lengths_match_v():
    B = _uniform_hjd(20)
    V = _uniform_hjd(10, offset=0.005)
    result = occ2(B, V, period=1.0, tolerance=0.05)
    assert len(result[1]) == len(V)
    assert len(result[2]) == len(V)


def test_occ2_indices_are_valid():
    B = _uniform_hjd(20)
    V = _uniform_hjd(10, offset=0.005)
    result = occ2(B, V, period=1.0, tolerance=0.05)
    for ib in result[3]:
        for idx in ib:
            assert 0 <= idx < len(B)
    for ia in result[4]:
        for idx in ia:
            assert 0 <= idx < len(B)


def test_occ2_good_diff_nonnegative():
    B = _uniform_hjd(20)
    V = _uniform_hjd(10, offset=0.005)
    assert all(d >= 0 for d in occ2(B, V, period=1.0, tolerance=0.05)[5])


def test_occ2_good_diff_within_tolerance():
    period = 1.0
    tolerance = 0.02
    B = _uniform_hjd(20, period=period)
    V = _uniform_hjd(10, period=period, offset=0.005)
    assert all(d < tolerance for d in occ2(B, V, period=period, tolerance=tolerance)[5])


def test_occ2_before_and_after_separated():
    delta = 0.02
    V_hjd = [2450000.5]
    B_hjd = [2450000.5 - delta, 2450000.5 + delta]
    result = occ2(B_hjd, V_hjd, period=1.0, tolerance=0.05)
    assert len(result[1][0]) == 1
    assert len(result[2][0]) == 1


def test_occ2_zero_tolerance_all_bad():
    B = _uniform_hjd(20)
    V = _uniform_hjd(10, offset=0.005)
    assert occ2(B, V, period=1.0, tolerance=0.0)[0] == len(V)


def test_occ2_large_tolerance_zero_bad():
    B = _uniform_hjd(50)
    V = _uniform_hjd(10, offset=0.01)
    assert occ2(B, V, period=1.0, tolerance=0.5)[0] == 0


def test_occ2_single_v_observation():
    B = list(np.linspace(2450000, 2450001, 20))
    V = [2450000.5]
    result = occ2(B, V, period=1.0, tolerance=0.1)
    assert len(result[1]) == 1
    assert len(result[2]) == 1


def test_occ2_empty_diff_when_all_bad():
    B = list(np.linspace(2450000, 2450001, 5))
    V = list(np.linspace(2460000, 2460001, 5))
    assert occ2(B, V, period=1.0, tolerance=0.01)[5] == []


def test_best_tol_returns_float():
    assert isinstance(best_tol(_uniform_hjd(40), _uniform_hjd(20, offset=0.005), period=1.0), float)


def test_best_tol_cap_with_overshoot():
    max_t = 0.03
    result = best_tol(_uniform_hjd(5), _uniform_hjd(20, offset=0.005), period=1.0, max_tol=max_t)
    assert result <= max_t + 0.001 + 1e-9


def test_best_tol_at_least_starting_value():
    assert best_tol(_uniform_hjd(40), _uniform_hjd(20, offset=0.005), period=1.0) >= 0.003


def test_best_tol_dense_sampling_stays_low():
    period = 1.0
    V = _uniform_hjd(10, period=period)
    B = []
    for v in V:
        B.append(v - 0.002)
        B.append(v + 0.002)
    assert best_tol(B, V, period=period, lower_lim=0.05, max_tol=0.03) <= 0.03


def test_best_tol_sparse_hits_cap():
    max_t = 0.02
    result = best_tol(_uniform_hjd(2), _uniform_hjd(20, offset=0.01), period=1.0, max_tol=max_t)
    assert result == pytest.approx(max_t + 0.001, abs=1e-6)


def test_best_tol_bad_fraction_satisfied():
    period = 1.0
    B = _uniform_hjd(40, period=period)
    V = _uniform_hjd(20, period=period, offset=0.005)
    lower = 0.05
    max_t = 0.03
    tol = best_tol(B, V, period=period, lower_lim=lower, max_tol=max_t)
    bad, *_ = occ2(B, V, period=period, tolerance=tol)
    bad_fraction = bad / len(V)
    assert bad_fraction <= lower or tol <= max_t + 0.001 + 1e-9


def test_best_tol_different_period():
    period = 0.5
    B = _uniform_hjd(30, period=period)
    V = _uniform_hjd(15, period=period, offset=0.003)
    result = best_tol(B, V, period=period, max_tol=0.03)
    assert 0.003 <= result <= 0.03


def test_integration_interpolated_mag_within_bounds():
    m1, m2 = 13.5, 14.5
    f_mid = lin_interp(0.5, 0.0, 1.0, 10 ** (-0.4 * m1), 10 ** (-0.4 * m2))
    mag_mid = -2.5 * np.log10(f_mid)
    assert m1 <= mag_mid <= m2


def test_integration_mean_mag_of_interpolated_series():
    mags = [lin_interp(t, 0.0, 1.0, 13.0, 15.0) for t in np.linspace(0, 1, 20)]
    result = mean_mag(mags)
    assert np.isfinite(result)
    assert 13.0 <= result <= 15.0
