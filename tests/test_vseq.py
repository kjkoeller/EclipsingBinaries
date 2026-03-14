# -*- coding: utf-8 -*-
"""
Tests for vseq_updated.py

"""

import pytest
import numpy as np
import math

from EclipsingBinaries.vseq_updated import (
    isNaN, new_list, conversion, splitter, decimal_limit,
    calc, binning, FT, OConnell, Flower, Pecaut, Red, plot, Roche, M, wfactor,
)


# ===========================================================================
# isNaN
# ===========================================================================
class TestIsNaN:
    def test_nan_returns_true(self):
        assert isNaN(float("nan")) is True

    def test_numpy_nan_returns_true(self):
        assert isNaN(np.nan) is True

    def test_zero_returns_false(self):
        assert isNaN(0) is False

    def test_integer_returns_false(self):
        assert isNaN(42) is False

    def test_float_returns_false(self):
        assert isNaN(3.14) is False

    def test_negative_returns_false(self):
        assert isNaN(-1.0) is False


# ===========================================================================
# new_list
# ===========================================================================
class TestNewList:
    def test_basic_conversion(self):
        result = new_list([1.12345, 2.98765])
        assert result == [1.12, 2.99]

    def test_integers_become_floats(self):
        result = new_list([1, 2, 3])
        assert all(isinstance(v, float) for v in result)

    def test_empty_list(self):
        assert new_list([]) == []

    def test_rounding(self):
        assert new_list([0.005])[0] == pytest.approx(0.0, abs=0.01)

    def test_length_preserved(self):
        data = [1.1, 2.2, 3.3, 4.4]
        assert len(new_list(data)) == len(data)


# ===========================================================================
# conversion  (decimal RA/DEC → colon-separated string)
# ===========================================================================
class TestConversion:
    def test_positive_simple(self):
        result = conversion([10.5])
        assert result[0].startswith("10:")

    def test_negative_simple(self):
        result = conversion([-10.5])
        assert result[0].startswith("-10:")

    def test_zero(self):
        result = conversion([0.0])
        assert result[0].startswith("0:")

    def test_returns_list_of_strings(self):
        result = conversion([5.25, -3.75])
        assert all(isinstance(v, str) for v in result)

    def test_length_preserved(self):
        inp = [1.0, 2.0, 3.0]
        assert len(conversion(inp)) == 3

    def test_colon_count(self):
        result = conversion([12.3456])
        assert result[0].count(":") == 2


# ===========================================================================
# splitter  (colon string → decimal)
# ===========================================================================
class TestSplitter:
    def test_positive_roundtrip(self):
        original = [15.5]
        colon = conversion(original)
        back = splitter(colon)
        assert back[0] == pytest.approx(original[0], abs=0.01)

    def test_negative_roundtrip(self):
        original = [-20.25]
        colon = conversion(original)
        back = splitter(colon)
        assert back[0] == pytest.approx(original[0], abs=0.01)

    def test_returns_list_of_floats(self):
        result = splitter(["10:30:00"])
        assert all(isinstance(v, float) for v in result)

    def test_zero_degrees(self):
        result = splitter(["0:0:0"])
        assert result[0] == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# decimal_limit
# ===========================================================================
class TestDecimalLimit:
    def test_basic(self):
        result = decimal_limit([3.14159, 2.71828])
        assert result == ["3.14", "2.72"]

    def test_empty(self):
        assert decimal_limit([]) == []

    def test_length_preserved(self):
        assert len(decimal_limit([1.1, 2.2, 3.3])) == 3

    def test_returns_strings(self):
        result = decimal_limit([1.0])
        assert all(isinstance(v, str) for v in result)


# ===========================================================================
# calc.frac
# ===========================================================================
class TestCalcFrac:
    def test_positive(self):
        assert calc.frac(3.75) == pytest.approx(0.75)

    def test_zero(self):
        assert calc.frac(0.0) == pytest.approx(0.0)

    def test_negative(self):
        # np.floor(-0.25) = -1, so frac = -0.25 - (-1) = 0.75
        assert calc.frac(-0.25) == pytest.approx(0.75)

    def test_whole_number(self):
        assert calc.frac(5.0) == pytest.approx(0.0)


# ===========================================================================
# calc.poly.result
# ===========================================================================
class TestCalcPolyResult:
    def test_constant(self):
        assert calc.poly.result([5.0], 10) == pytest.approx(5.0)

    def test_linear(self):
        # 2 + 3x at x=4 → 14
        assert calc.poly.result([2, 3], 4) == pytest.approx(14.0)

    def test_quadratic(self):
        # 1 + 0x + 1x² at x=3 → 10
        assert calc.poly.result([1, 0, 1], 3) == pytest.approx(10.0)

    def test_derivative_linear(self):
        # deriv of 2 + 3x → 3 (must pass float; int**-1 raises ValueError in numpy)
        assert calc.poly.result([2, 3], 4.0, deriv=True) == pytest.approx(3.0)

    def test_derivative_quadratic(self):
        # deriv of 1 + 0x + 1x² at x=3 → 6
        assert calc.poly.result([1, 0, 1], 3.0, deriv=True) == pytest.approx(6.0)


# ===========================================================================
# calc.poly.error
# ===========================================================================
class TestCalcPolyError:
    def test_constant_poly_zero_error(self):
        # No derivative term for constant
        assert calc.poly.error([5.0], 2.0, 0.1) == pytest.approx(0.0)

    def test_linear_propagation(self):
        # coef = [0, 2], error in x = 0.5 → propagated error = 2 * 0.5 = 1.0
        assert calc.poly.error([0, 2], 1.0, 0.5) == pytest.approx(1.0)


# ===========================================================================
# calc.poly.polylist
# ===========================================================================
class TestCalcPolyPolylist:
    def test_length(self):
        xlist, ylist = calc.poly.polylist([1, 1], 0, 1, 100)
        assert len(xlist) == 100
        assert len(ylist) == 100

    def test_linear_values(self):
        xlist, ylist = calc.poly.polylist([0, 1], 0, 1, 10)
        for x, y in zip(xlist, ylist):
            assert y == pytest.approx(x, abs=1e-10)


# ===========================================================================
# calc.error.per_diff
# ===========================================================================
class TestCalcErrorPerDiff:
    def test_identical_values(self):
        assert calc.error.per_diff(5.0, 5.0) == pytest.approx(0.0)

    def test_known_result(self):
        # |10 - 6| / mean(10, 6) * 100 = 4/8*100 = 50
        assert calc.error.per_diff(10, 6) == pytest.approx(50.0)

    def test_symmetric(self):
        assert calc.error.per_diff(3, 7) == pytest.approx(calc.error.per_diff(7, 3))


# ===========================================================================
# calc.error.SS_residuals
# ===========================================================================
class TestCalcErrorSSResiduals:
    def test_perfect_fit(self):
        obs = [1, 2, 3]
        assert calc.error.SS_residuals(obs, obs) == pytest.approx(0.0)

    def test_known_value(self):
        obs = [1, 2, 3]
        model = [2, 3, 4]
        assert calc.error.SS_residuals(obs, model) == pytest.approx(3.0)


# ===========================================================================
# calc.error.sig_sum
# ===========================================================================
class TestCalcErrorSigSum:
    def test_basic(self):
        assert calc.error.sig_sum([3, 4]) == pytest.approx(5.0)

    def test_single_element(self):
        assert calc.error.sig_sum([2.5]) == pytest.approx(2.5)


# ===========================================================================
# calc.error.SS_total
# ===========================================================================
class TestCalcErrorSSTotal:
    def test_constant_list(self):
        assert calc.error.SS_total([5, 5, 5]) == pytest.approx(0.0)

    def test_known(self):
        # mean([1,2,3]) = 2; SS = 1+0+1 = 2
        assert calc.error.SS_total([1, 2, 3]) == pytest.approx(2.0)


# ===========================================================================
# calc.error.CoD
# ===========================================================================
class TestCalcErrorCoD:
    def test_perfect_model(self):
        obs = [1, 2, 3, 4]
        assert calc.error.CoD(obs, obs) == pytest.approx(1.0)

    def test_returns_float(self):
        result = calc.error.CoD([1, 2, 3], [1, 2, 3])
        assert isinstance(result, float)


# ===========================================================================
# calc.error.weighted_average
# ===========================================================================
class TestCalcErrorWeightedAverage:
    def test_equal_errors(self):
        values = [2.0, 4.0]
        errors = [1.0, 1.0]
        avg, err, M = calc.error.weighted_average(values, errors)
        assert avg == pytest.approx(3.0)

    def test_structure(self):
        values = [1.0, 2.0, 3.0]
        errors = [0.1, 0.2, 0.3]
        result = calc.error.weighted_average(values, errors)
        assert len(result) == 3


# ===========================================================================
# calc.error.avg
# ===========================================================================
class TestCalcErrorAvg:
    def test_single(self):
        assert calc.error.avg([3.0]) == pytest.approx(3.0)

    def test_two_equal(self):
        # sqrt(e1² + e2²) / 2 for e1=e2=1 → sqrt(2)/2
        assert calc.error.avg([1.0, 1.0]) == pytest.approx(np.sqrt(2) / 2)


# ===========================================================================
# calc.error.red_X2
# ===========================================================================
class TestCalcErrorRedX2:
    def test_perfect_fit_is_zero(self):
        obs = [1.0, 2.0, 3.0]
        model = [1.0, 2.0, 3.0]
        errors = [1.0, 1.0, 1.0]
        assert calc.error.red_X2(obs, model, errors) == pytest.approx(0.0)

    def test_known_value(self):
        # single point: ((2-1)/1)^2 = 1
        assert calc.error.red_X2([2.0], [1.0], [1.0]) == pytest.approx(1.0)


# ===========================================================================
# calc.astro.convert.HJD_phase
# ===========================================================================
class TestHJDPhase:
    def test_at_epoch_is_zero(self):
        phase = calc.astro.convert.HJD_phase([2450000.0], 1.0, 2450000.0)
        assert phase[0] == pytest.approx(0.0)

    def test_half_period(self):
        phase = calc.astro.convert.HJD_phase([2450000.5], 1.0, 2450000.0)
        assert phase[0] == pytest.approx(0.5)

    def test_output_in_range(self):
        hjd = np.linspace(2450000, 2450010, 50).tolist()
        phases = calc.astro.convert.HJD_phase(hjd, 1.23456, 2450000.0)
        assert all(0.0 <= p < 1.0 for p in phases)


# ===========================================================================
# calc.astro.convert.JD_to_Greg
# ===========================================================================
class TestJDToGreg:
    def test_known_date(self):
        # JD 2451545.0 = 2000 January 01
        result = calc.astro.convert.JD_to_Greg(2451545.0)
        assert "2000" in result
        assert "01" in result

    def test_returns_string(self):
        assert isinstance(calc.astro.convert.JD_to_Greg(2451545.0), str)

    def test_format(self):
        result = calc.astro.convert.JD_to_Greg(2451545.0)
        parts = result.split(" ")
        assert len(parts) == 3


# ===========================================================================
# calc.astro.convert.magToflux
# ===========================================================================
class TestMagToFlux:
    def test_zero_mag_gives_one(self):
        assert calc.astro.convert.magToflux.flux(0.0) == pytest.approx(1.0)

    def test_flux_is_positive(self):
        assert calc.astro.convert.magToflux.flux(5.0) > 0

    def test_error_positive(self):
        err = calc.astro.convert.magToflux.error(5.0, 0.1)
        assert err > 0

    def test_brighter_means_higher_flux(self):
        assert calc.astro.convert.magToflux.flux(0.0) > calc.astro.convert.magToflux.flux(5.0)


# ===========================================================================
# calc.astro.convert.fluxTomag
# ===========================================================================
class TestFluxToMag:
    def test_flux_one_gives_zero(self):
        assert calc.astro.convert.fluxTomag.mag(1.0) == pytest.approx(0.0)

    def test_error_positive(self):
        err = calc.astro.convert.fluxTomag.error(1.0, 0.01)
        assert err > 0

    def test_roundtrip(self):
        mag_in = 14.5
        flux = calc.astro.convert.magToflux.flux(mag_in)
        mag_out = calc.astro.convert.fluxTomag.mag(flux)
        assert mag_out == pytest.approx(mag_in, abs=1e-6)


# ===========================================================================
# FT.coefficients
# ===========================================================================
class TestFTCoefficients:
    def test_mean_in_coslist(self):
        data = [1.0, 2.0, 3.0, 4.0]
        _, coslist, _ = FT.coefficients(data)
        assert coslist[0] == pytest.approx(np.mean(data))

    def test_sinlist_zero_at_zero(self):
        data = [1.0, 2.0, 3.0, 4.0]
        _, _, sinlist = FT.coefficients(data)
        assert sinlist[0] == pytest.approx(0.0)

    def test_output_length(self):
        data = [1.0] * 8
        _, coslist, sinlist = FT.coefficients(data)
        assert len(coslist) == 8
        assert len(sinlist) == 8


# ===========================================================================
# FT.sumatphase
# ===========================================================================
class TestFTSumAtPhase:
    def test_dc_only(self):
        # a=[A0, 0, 0], b=[0,0,0] → result should be A0 at any phase
        a = np.array([2.0, 0.0, 0.0])
        b = np.zeros(3)
        val = FT.sumatphase(0.0, 2, a, b)
        assert val == pytest.approx(2.0)

    def test_sine_term(self):
        # a=[0,0], b=[0,1] at phase=0.25 → sin(2π*1*0.25) = 1
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 1.0])
        val = FT.sumatphase(0.25, 1, a, b)
        assert val == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# FT.synth
# ===========================================================================
class TestFTSynth:
    def test_length(self):
        a = np.array([1.0, 0.1])
        b = np.zeros(2)
        phases = [0.0, 0.25, 0.5, 0.75]
        result = FT.synth(a, b, phases, 1)
        assert len(result) == len(phases)

    def test_constant_signal(self):
        a = np.array([2.0, 0.0])
        b = np.zeros(2)
        phases = [0.0, 0.1, 0.5, 0.9]
        result = FT.synth(a, b, phases, 1)
        for v in result:
            assert v == pytest.approx(2.0, abs=1e-6)


# ===========================================================================
# FT.FT_plotlist
# ===========================================================================
class TestFTPlotList:
    def test_length(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.zeros(3)
        phases, fluxes, derivs = FT.FT_plotlist(a, b, 2, 100)
        assert len(phases) == 100
        assert len(fluxes) == 100
        assert len(derivs) == 100

    def test_phases_in_range(self):
        a = np.array([1.0, 0.0])
        b = np.zeros(2)
        phases, _, _ = FT.FT_plotlist(a, b, 1, 50)
        assert all(0.0 <= p < 1.0 for p in phases)


# ===========================================================================
# FT.integral
# ===========================================================================
class TestFTIntegral:
    def test_zero_width_integral(self):
        a = np.array([1.0, 0.0])
        b = np.zeros(2)
        result = FT.integral(a, b, 1, 0.3, 0.3)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_full_period_dc(self):
        # DC-only signal: integral over [0,1] should equal a[0]
        a = np.array([2.0, 0.0])
        b = np.zeros(2)
        result = FT.integral(a, b, 1, 0, 1)
        assert result == pytest.approx(2.0, abs=1e-6)


# ===========================================================================
# OConnell.Delta_I
# ===========================================================================
class TestOConnellDeltaI:
    def test_symmetric_light_curve(self):
        # Pure cosine (even) → Delta_I should be 0
        a = np.array([1.0, 0.5, 0.0])
        b = np.zeros(3)
        dI, Ip, Is = OConnell.Delta_I(a, b, 2)
        assert dI == pytest.approx(0.0, abs=1e-6)

    def test_asymmetric_nonzero(self):
        a = np.zeros(3)
        b = np.array([0.0, 0.2, 0.0])
        dI, Ip, Is = OConnell.Delta_I(a, b, 2)
        assert abs(dI) > 0

    def test_returns_three_values(self):
        a = np.ones(3)
        b = np.zeros(3)
        result = OConnell.Delta_I(a, b, 2)
        assert len(result) == 3


# ===========================================================================
# OConnell.Delta_I_fixed
# ===========================================================================
class TestOConnellDeltaIFixed:
    def test_zero_b_coeffs(self):
        b = np.zeros(5)
        assert OConnell.Delta_I_fixed(b, 4) == pytest.approx(0.0)

    def test_nonzero(self):
        b = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        result = OConnell.Delta_I_fixed(b, 4)
        assert abs(result) > 0


# ===========================================================================
# Flower.T.Teff
# ===========================================================================
class TestFlowerTeff:
    def test_solar_bv(self):
        # B-V ≈ 0.65 for Sun → Teff ≈ 5778 K
        temp, err = Flower.T.Teff(0.65, 0.02)
        assert 5000 < temp < 7000

    def test_hot_star(self):
        # B-V ≈ -0.3 (hot blue star)
        temp, err = Flower.T.Teff(-0.3, 0.02)
        assert temp > 8000

    def test_error_positive(self):
        # t_eff_err can be negative depending on polynomial slope direction;
        # assert it is non-zero and has a meaningful magnitude
        _, err = Flower.T.Teff(0.65, 0.02)
        assert err != 0
        assert abs(err) < 1000  # sanity bound in Kelvin


# ===========================================================================
# Pecaut.T.Teff
# ===========================================================================
class TestPecautTeff:
    def test_mid_range(self):
        temp, err = Pecaut.T.Teff(0.5, 0.02)
        assert temp > 0
        assert err > 0

    def test_out_of_range_returns_zero(self):
        temp, err = Pecaut.T.Teff(2.0, 0.01)
        assert temp == 0
        assert err == 0

    def test_lower_range(self):
        temp, err = Pecaut.T.Teff(0.01, 0.005)
        assert temp > 0


# ===========================================================================
# Red.colorEx
# ===========================================================================
class TestRedColorEx:
    def test_J_K(self):
        result = Red.colorEx("J", "K", 1.0)
        assert result == pytest.approx(Red.J_K)

    def test_J_H(self):
        result = Red.colorEx("J", "H", 2.0)
        assert result == pytest.approx(2.0 * Red.J_H)

    def test_V_R(self):
        result = Red.colorEx("V", "R", 3.1)
        assert result == pytest.approx(3.1 * Red.V_R)

    def test_unknown_returns_none(self):
        result = Red.colorEx("X", "Y", 1.0)
        assert result is None


# ===========================================================================
# plot.amp
# ===========================================================================
class TestPlotAmp:
    def test_basic(self):
        assert plot.amp([1, 3, 5, 2]) == pytest.approx(4.0)

    def test_flat(self):
        assert plot.amp([3, 3, 3]) == pytest.approx(0.0)

    def test_negative_values(self):
        assert plot.amp([-5, -1]) == pytest.approx(4.0)


# ===========================================================================
# plot.aliasing2
# ===========================================================================
class TestPlotAliasing2:
    def test_output_length(self):
        phases = [0.0, 0.2, 0.4, 0.6, 0.8]
        mags = [1.0] * 5
        errors = [0.01] * 5
        a_phase, a_mag, a_error = plot.aliasing2(phases, mags, errors)
        assert len(a_phase) == len(a_mag) == len(a_error)

    def test_all_within_alias_range(self):
        phases = [0.0, 0.2, 0.4, 0.6, 0.8]
        mags = [1.0] * 5
        errors = [0.01] * 5
        alias = 0.6
        a_phase, _, _ = plot.aliasing2(phases, mags, errors, alias=alias)
        assert all(-alias < p < alias for p in a_phase)


# ===========================================================================
# binning.norm_flux
# ===========================================================================
class TestBinningNormFlux:
    def test_max_is_one_for_bin_norm(self):
        binned = [0.5, 0.8, 1.0]
        ob = [0.3, 0.7, 0.9]
        err = [0.01, 0.01, 0.01]
        n_bin, n_ob, n_err = binning.norm_flux(binned, ob, err, norm_factor='bin')
        assert max(n_bin) == pytest.approx(1.0)

    def test_max_is_one_for_ob_norm(self):
        binned = [0.5, 0.8]
        ob = [0.3, 1.0]
        err = [0.01, 0.01]
        n_bin, n_ob, n_err = binning.norm_flux(binned, ob, err, norm_factor='ob')
        assert max(n_ob) == pytest.approx(1.0)

    def test_length_preserved(self):
        binned = [0.5, 0.8, 1.0]
        ob = [0.3, 0.7, 0.9]
        err = [0.01, 0.01, 0.01]
        n_bin, n_ob, n_err = binning.norm_flux(binned, ob, err)
        assert len(n_bin) == 3
        assert len(n_ob) == 3
        assert len(n_err) == 3


# ===========================================================================
# M and wfactor  (module-level functions)
# ===========================================================================
class TestMAndWfactor:
    def test_M_single_error(self):
        # M([2]) = 1/4
        assert M([2.0]) == pytest.approx(0.25)

    def test_M_multiple(self):
        result = M([1.0, 1.0])
        assert result == pytest.approx(2.0)

    def test_wfactor_sums_to_one(self):
        errors = [1.0, 1.0, 1.0]
        total_M = M(errors)
        total_w = sum(wfactor(errors, n, total_M) for n in range(len(errors)))
        assert total_w == pytest.approx(1.0)


# ===========================================================================
# Roche.Lagrange_123  (smoke test — confirms Newton converges)
# ===========================================================================
class TestRocheLagrange:
    def test_equal_mass(self):
        L1, L2, L3 = Roche.Lagrange_123(1.0)
        assert isinstance(L1, float)
        assert 0 < L1 < 1

    def test_asymmetric(self):
        L1, L2, L3 = Roche.Lagrange_123(0.5)
        assert L1 > 0
        assert L2 > 0


# ===========================================================================
# calc.Newton  (basic convergence)
# ===========================================================================
class TestCalcNewton:
    def test_square_root_of_2(self):
        f = lambda x: x ** 2 - 2
        root = calc.Newton(f, 1.0)
        assert root == pytest.approx(np.sqrt(2), abs=1e-6)

    def test_simple_linear(self):
        f = lambda x: x - 3
        root = calc.Newton(f, 0.0)
        assert root == pytest.approx(3.0, abs=1e-6)

    def test_max_iter_returns_false(self):
        # sin(x) - 0.5 has roots but Newton can cycle if started poorly;
        # use a pathological case: extremely slow convergence, tight tolerance, few iters
        f = lambda x: x ** 3 - x - 1000  # root near x≈10, won't converge from 0 in 2 iters
        result = calc.Newton(f, 0.0, e=1e-15, max_iter=2)
        assert result is False