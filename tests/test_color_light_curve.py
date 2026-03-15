# -*- coding: utf-8 -*-
"""
Tests for color_light_curve.py

"""

import pytest
import numpy as np
import os
import tempfile

from EclipsingBinaries.color_light_curve import (
    occ2,
    best_tol,
    lin_interp,
    mean_mag,
)


# ===========================================================================
# lin_interp  (module-level lambda)
# ===========================================================================
class TestLinInterp:
    def test_at_x1_returns_y1(self):
        assert lin_interp(1.0, 1.0, 2.0, 10.0, 20.0) == pytest.approx(10.0)

    def test_at_x2_returns_y2(self):
        assert lin_interp(2.0, 1.0, 2.0, 10.0, 20.0) == pytest.approx(20.0)

    def test_midpoint(self):
        assert lin_interp(1.5, 1.0, 2.0, 10.0, 20.0) == pytest.approx(15.0)

    def test_quarter_point(self):
        assert lin_interp(1.25, 1.0, 2.0, 0.0, 4.0) == pytest.approx(1.0)

    def test_extrapolation(self):
        # Linear extrapolation beyond x2
        assert lin_interp(3.0, 1.0, 2.0, 0.0, 1.0) == pytest.approx(2.0)

    def test_flat_line(self):
        assert lin_interp(1.7, 1.0, 2.0, 5.0, 5.0) == pytest.approx(5.0)

    def test_negative_slope(self):
        assert lin_interp(1.5, 1.0, 2.0, 10.0, 0.0) == pytest.approx(5.0)

    def test_float_output(self):
        result = lin_interp(1.5, 1.0, 2.0, 0.0, 1.0)
        assert isinstance(result, float)


# ===========================================================================
# mean_mag  (module-level lambda)
# ===========================================================================
class TestMeanMag:
    def test_single_value_roundtrip(self):
        # mean_mag of a single magnitude should return that magnitude
        mag = 14.5
        assert mean_mag([mag]) == pytest.approx(mag, abs=1e-6)

    def test_two_equal_values(self):
        mag = 12.0
        assert mean_mag([mag, mag]) == pytest.approx(mag, abs=1e-6)

    def test_flux_weighted_average(self):
        # mean_mag([m1, m2]) should equal -2.5*log10(mean(10^(-0.4*m1), 10^(-0.4*m2)))
        mags = [13.0, 14.0]
        expected = -2.5 * np.log10(np.mean(10 ** (-0.4 * np.array(mags))))
        assert mean_mag(mags) == pytest.approx(expected, abs=1e-8)

    def test_result_is_between_min_and_max(self):
        mags = [12.0, 14.0, 16.0]
        result = mean_mag(mags)
        assert min(mags) <= result <= max(mags)

    def test_brighter_weight_pulls_result_toward_brighter(self):
        # mag 10 is much brighter than 15; flux-weighted mean should be closer to 10
        result = mean_mag([10.0, 15.0])
        assert result < 12.5  # simple arithmetic midpoint; flux mean should be lower (brighter)

    def test_numpy_array_input(self):
        mags = np.array([13.0, 14.0, 15.0])
        result = mean_mag(mags)
        assert np.isfinite(result)


# ===========================================================================
# occ2
# ===========================================================================
class TestOcc2:
    """
    occ2 finds, for each V observation, B observations within tol*period
    that occurred before and after it.
    """

    def _make_uniform(self, n=20, period=1.0, offset=0.0):
        """Evenly spaced HJDs over one period."""
        return list(np.linspace(2450000 + offset, 2450000 + period + offset, n, endpoint=False))

    def test_returns_six_elements(self):
        B = self._make_uniform(20)
        V = self._make_uniform(20, offset=0.005)
        result = occ2(B, V, period=1.0, tolerance=0.05)
        assert len(result) == 6

    def test_bad_obs_is_integer(self):
        B = self._make_uniform(20)
        V = self._make_uniform(20, offset=0.005)
        bad_obs = occ2(B, V, period=1.0, tolerance=0.05)[0]
        assert isinstance(bad_obs, int)

    def test_zero_bad_obs_when_well_sampled(self):
        # Dense B sampling around every V point → no bad obs
        period = 1.0
        V = list(np.linspace(2450000, 2450001, 10, endpoint=False))
        # Place B points just before and just after each V
        B = []
        tol = 0.02
        for v in V:
            B.append(v - tol * period * 0.5)
            B.append(v + tol * period * 0.5)
        bad_obs = occ2(B, V, period=period, tolerance=tol)[0]
        assert bad_obs == 0

    def test_all_bad_obs_when_no_overlap(self):
        # B and V completely separated in time — nothing within tolerance
        B = list(np.linspace(2450000, 2450001, 10))
        V = list(np.linspace(2460000, 2460001, 10))  # far future
        bad_obs = occ2(B, V, period=1.0, tolerance=0.01)[0]
        assert bad_obs == len(V)

    def test_good_before_and_after_lengths_match_V(self):
        B = self._make_uniform(20)
        V = self._make_uniform(10, offset=0.005)
        result = occ2(B, V, period=1.0, tolerance=0.05)
        assert len(result[1]) == len(V)  # good_before
        assert len(result[2]) == len(V)  # good_after

    def test_index_before_and_after_are_valid_indices(self):
        B = self._make_uniform(20)
        V = self._make_uniform(10, offset=0.005)
        result = occ2(B, V, period=1.0, tolerance=0.05)
        index_before = result[3]
        index_after = result[4]
        for ib in index_before:
            for idx in ib:
                assert 0 <= idx < len(B)
        for ia in index_after:
            for idx in ia:
                assert 0 <= idx < len(B)

    def test_good_diff_all_positive(self):
        B = self._make_uniform(20)
        V = self._make_uniform(10, offset=0.005)
        good_diff = occ2(B, V, period=1.0, tolerance=0.05)[5]
        assert all(d >= 0 for d in good_diff)

    def test_good_diff_within_tolerance(self):
        period = 1.0
        tolerance = 0.02
        B = self._make_uniform(20, period=period)
        V = self._make_uniform(10, period=period, offset=0.005)
        good_diff = occ2(B, V, period=period, tolerance=tolerance)[5]
        # Each diff is |ΔT|/period, so should be < tolerance
        assert all(d < tolerance for d in good_diff)

    def test_symmetry_before_after_separation(self):
        # B point exactly before a V point and another exactly after
        period = 1.0
        tol = 0.05
        V_hjd = [2450000.5]
        delta = 0.02  # within tolerance
        B_hjd = [2450000.5 - delta, 2450000.5 + delta]
        result = occ2(B_hjd, V_hjd, period=period, tolerance=tol)
        assert len(result[1][0]) == 1   # one before
        assert len(result[2][0]) == 1   # one after

    def test_tolerance_zero_gives_all_bad(self):
        B = self._make_uniform(20)
        V = self._make_uniform(10, offset=0.005)
        # tolerance=0 means tol*period=0, nothing is strictly < 0
        bad_obs = occ2(B, V, period=1.0, tolerance=0.0)[0]
        assert bad_obs == len(V)

    def test_large_tolerance_gives_zero_bad(self):
        B = self._make_uniform(50)
        V = self._make_uniform(10, offset=0.1)
        # tolerance=1.0 means the window is the entire period
        bad_obs = occ2(B, V, period=1.0, tolerance=1.0)[0]
        assert bad_obs == 0

    def test_single_V_observation(self):
        B = list(np.linspace(2450000, 2450001, 20))
        V = [2450000.5]
        result = occ2(B, V, period=1.0, tolerance=0.1)
        assert len(result[1]) == 1
        assert len(result[2]) == 1

    def test_empty_good_diff_when_all_bad(self):
        B = list(np.linspace(2450000, 2450001, 5))
        V = list(np.linspace(2460000, 2460001, 5))
        good_diff = occ2(B, V, period=1.0, tolerance=0.01)[5]
        assert good_diff == []


# ===========================================================================
# best_tol
# ===========================================================================
class TestBestTol:
    def _make_uniform(self, n, period=1.0, offset=0.0):
        return list(np.linspace(2450000 + offset, 2450000 + period + offset, n, endpoint=False))

    def test_returns_float(self):
        B = self._make_uniform(40)
        V = self._make_uniform(20, offset=0.005)
        result = best_tol(B, V, period=1.0)
        assert isinstance(result, float)

    def test_result_does_not_exceed_max_tol(self):
        B = self._make_uniform(5)   # sparse B → needs high tolerance
        V = self._make_uniform(20, offset=0.005)
        max_t = 0.03
        result = best_tol(B, V, period=1.0, max_tol=max_t)
        assert result <= max_t

    def test_result_at_least_starting_tol(self):
        B = self._make_uniform(40)
        V = self._make_uniform(20, offset=0.005)
        result = best_tol(B, V, period=1.0)
        assert result >= 0.003  # starts at 0.003

    def test_dense_sampling_stays_at_minimum(self):
        # Very dense B relative to V → minimal tolerance needed
        period = 1.0
        V = self._make_uniform(10, period=period)
        # Place B just before and after every V
        B = []
        for v in V:
            B.append(v - 0.002)
            B.append(v + 0.002)
        result = best_tol(B, V, period=period, lower_lim=0.05, max_tol=0.03)
        assert result <= 0.03

    def test_sparse_sampling_hits_max_tol(self):
        # B has only 2 points, V has 20 → tolerance will hit max_tol
        B = self._make_uniform(2)
        V = self._make_uniform(20, offset=0.01)
        max_t = 0.02
        result = best_tol(B, V, period=1.0, max_tol=max_t)
        assert result == pytest.approx(max_t, abs=1e-6)

    def test_bad_fraction_below_lower_lim_after_best_tol(self):
        # After finding best_tol, bad fraction should be ≤ lower_lim (unless capped at max_tol)
        period = 1.0
        B = self._make_uniform(40, period=period)
        V = self._make_uniform(20, period=period, offset=0.005)
        lower = 0.05
        max_t = 0.03
        tol = best_tol(B, V, period=period, lower_lim=lower, max_tol=max_t)
        bad, *_ = occ2(B, V, period=period, tolerance=tol)
        bad_fraction = bad / len(V)
        # Either we achieved the lower limit, or we hit max_tol trying
        assert bad_fraction <= lower or tol == pytest.approx(max_t, abs=1e-6)

    def test_different_periods_scale_correctly(self):
        # Using a different period should still return a value in [0.003, max_tol]
        period = 0.5
        B = self._make_uniform(30, period=period)
        V = self._make_uniform(15, period=period, offset=0.003)
        result = best_tol(B, V, period=period, max_tol=0.03)
        assert 0.003 <= result <= 0.03


# ===========================================================================
# Integration: lin_interp + mean_mag consistency
# ===========================================================================
class TestIntegration:
    def test_interpolated_mag_within_bounds(self):
        # Interpolate between two magnitudes and verify mean_mag stays in range
        m1, m2 = 13.5, 14.5
        f1 = 10 ** (-0.4 * m1)
        f2 = 10 ** (-0.4 * m2)
        # Interpolate at midpoint in flux space
        f_mid = lin_interp(0.5, 0.0, 1.0, f1, f2)
        mag_mid = -2.5 * np.log10(f_mid)
        assert m1 <= mag_mid <= m2  # dimmer number is larger in mag scale

    def test_mean_mag_of_interpolated_series(self):
        # Generate a series of mags via lin_interp and check mean_mag is finite
        mags = [lin_interp(t, 0.0, 1.0, 13.0, 15.0) for t in np.linspace(0, 1, 20)]
        result = mean_mag(mags)
        assert np.isfinite(result)
        assert 13.0 <= result <= 15.0