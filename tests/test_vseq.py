# -*- coding: utf-8 -*-
"""
Tests for vseq_updated.py

"""

import pytest
import numpy as np

from EclipsingBinaries.vseq_updated import (
    isNaN, new_list, conversion, splitter, decimal_limit,
    calc, binning, FT, OConnell, Flower, Pecaut, Red, plot, Roche, M, wfactor,
)


# ===========================================================================
# isNaN
# ===========================================================================
def test_isnan_nan_returns_true():
    assert isNaN(float("nan")) is True


def test_isnan_numpy_nan_returns_true():
    assert isNaN(np.nan) is True


def test_isnan_zero_returns_false():
    assert isNaN(0) is False


def test_isnan_integer_returns_false():
    assert isNaN(42) is False


def test_isnan_float_returns_false():
    assert isNaN(3.14) is False


def test_isnan_negative_returns_false():
    assert isNaN(-1.0) is False


# ===========================================================================
# new_list
# ===========================================================================
def test_new_list_basic_conversion():
    assert new_list([1.12345, 2.98765]) == [1.12, 2.99]


def test_new_list_integers_become_floats():
    assert all(isinstance(v, float) for v in new_list([1, 2, 3]))


def test_new_list_empty():
    assert new_list([]) == []


def test_new_list_length_preserved():
    assert len(new_list([1.1, 2.2, 3.3, 4.4])) == 4


# ===========================================================================
# conversion
# ===========================================================================
def test_conversion_positive_starts_correctly():
    assert conversion([10.5])[0].startswith("10:")


def test_conversion_negative_starts_correctly():
    assert conversion([-10.5])[0].startswith("-10:")


def test_conversion_zero():
    assert conversion([0.0])[0].startswith("0:")


def test_conversion_returns_strings():
    assert all(isinstance(v, str) for v in conversion([5.25, -3.75]))


def test_conversion_length_preserved():
    assert len(conversion([1.0, 2.0, 3.0])) == 3


def test_conversion_colon_count():
    assert conversion([12.3456])[0].count(":") == 2


# ===========================================================================
# splitter
# ===========================================================================
def test_splitter_positive_roundtrip():
    original = [15.5]
    assert splitter(conversion(original))[0] == pytest.approx(original[0], abs=0.01)


def test_splitter_negative_roundtrip():
    original = [-20.25]
    assert splitter(conversion(original))[0] == pytest.approx(original[0], abs=0.01)


def test_splitter_returns_floats():
    assert all(isinstance(v, float) for v in splitter(["10:30:00"]))


def test_splitter_zero_degrees():
    assert splitter(["0:0:0"])[0] == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# decimal_limit
# ===========================================================================
def test_decimal_limit_basic():
    assert decimal_limit([3.14159, 2.71828]) == ["3.14", "2.72"]


def test_decimal_limit_empty():
    assert decimal_limit([]) == []


def test_decimal_limit_length_preserved():
    assert len(decimal_limit([1.1, 2.2, 3.3])) == 3


def test_decimal_limit_returns_strings():
    assert all(isinstance(v, str) for v in decimal_limit([1.0]))


# ===========================================================================
# calc.frac
# ===========================================================================
def test_calc_frac_positive():
    assert calc.frac(3.75) == pytest.approx(0.75)


def test_calc_frac_zero():
    assert calc.frac(0.0) == pytest.approx(0.0)


def test_calc_frac_negative():
    assert calc.frac(-0.25) == pytest.approx(0.75)


def test_calc_frac_whole_number():
    assert calc.frac(5.0) == pytest.approx(0.0)


# ===========================================================================
# calc.poly.result
# ===========================================================================
def test_calc_poly_result_constant():
    assert calc.poly.result([5.0], 10) == pytest.approx(5.0)


def test_calc_poly_result_linear():
    assert calc.poly.result([2, 3], 4) == pytest.approx(14.0)


def test_calc_poly_result_quadratic():
    assert calc.poly.result([1, 0, 1], 3) == pytest.approx(10.0)


def test_calc_poly_result_derivative_linear():
    assert calc.poly.result([2, 3], 4, deriv=True) == pytest.approx(3.0)


def test_calc_poly_result_derivative_quadratic():
    assert calc.poly.result([1, 0, 1], 3, deriv=True) == pytest.approx(6.0)


# ===========================================================================
# calc.poly.error
# ===========================================================================
def test_calc_poly_error_constant_zero():
    assert calc.poly.error([5.0], 2.0, 0.1) == pytest.approx(0.0)


def test_calc_poly_error_linear_propagation():
    assert calc.poly.error([0, 2], 1.0, 0.5) == pytest.approx(1.0)


# ===========================================================================
# calc.poly.polylist
# ===========================================================================
def test_calc_poly_polylist_length():
    xlist, ylist = calc.poly.polylist([1, 1], 0, 1, 100)
    assert len(xlist) == 100
    assert len(ylist) == 100


def test_calc_poly_polylist_linear_values():
    xlist, ylist = calc.poly.polylist([0, 1], 0, 1, 10)
    for x, y in zip(xlist, ylist):
        assert y == pytest.approx(x, abs=1e-10)


# ===========================================================================
# calc.error.per_diff
# ===========================================================================
def test_calc_error_per_diff_identical():
    assert calc.error.per_diff(5.0, 5.0) == pytest.approx(0.0)


def test_calc_error_per_diff_known():
    assert calc.error.per_diff(10, 6) == pytest.approx(50.0)


def test_calc_error_per_diff_symmetric():
    assert calc.error.per_diff(3, 7) == pytest.approx(calc.error.per_diff(7, 3))


# ===========================================================================
# calc.error.SS_residuals
# ===========================================================================
def test_calc_error_ss_residuals_perfect_fit():
    obs = [1, 2, 3]
    assert calc.error.SS_residuals(obs, obs) == pytest.approx(0.0)


def test_calc_error_ss_residuals_known():
    assert calc.error.SS_residuals([1, 2, 3], [2, 3, 4]) == pytest.approx(3.0)


# ===========================================================================
# calc.error.sig_sum
# ===========================================================================
def test_calc_error_sig_sum_basic():
    assert calc.error.sig_sum([3, 4]) == pytest.approx(5.0)


def test_calc_error_sig_sum_single():
    assert calc.error.sig_sum([2.5]) == pytest.approx(2.5)


# ===========================================================================
# calc.error.SS_total
# ===========================================================================
def test_calc_error_ss_total_constant():
    assert calc.error.SS_total([5, 5, 5]) == pytest.approx(0.0)


def test_calc_error_ss_total_known():
    assert calc.error.SS_total([1, 2, 3]) == pytest.approx(2.0)


# ===========================================================================
# calc.error.CoD
# ===========================================================================
def test_calc_error_cod_perfect_model():
    obs = [1, 2, 3, 4]
    assert calc.error.CoD(obs, obs) == pytest.approx(1.0)


def test_calc_error_cod_returns_float():
    assert isinstance(calc.error.CoD([1, 2, 3], [1, 2, 3]), float)


# ===========================================================================
# calc.error.weighted_average
# ===========================================================================
def test_calc_error_weighted_average_equal_errors():
    avg, err, M_val = calc.error.weighted_average([2.0, 4.0], [1.0, 1.0])
    assert avg == pytest.approx(3.0)


def test_calc_error_weighted_average_structure():
    assert len(calc.error.weighted_average([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])) == 3


# ===========================================================================
# calc.error.avg
# ===========================================================================
def test_calc_error_avg_single():
    assert calc.error.avg([3.0]) == pytest.approx(3.0)


def test_calc_error_avg_two_equal():
    assert calc.error.avg([1.0, 1.0]) == pytest.approx(np.sqrt(2) / 2)


# ===========================================================================
# calc.error.red_X2
# ===========================================================================
def test_calc_error_red_x2_perfect_fit():
    obs = [1.0, 2.0, 3.0]
    assert calc.error.red_X2(obs, obs, [1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_calc_error_red_x2_known():
    assert calc.error.red_X2([2.0], [1.0], [1.0]) == pytest.approx(1.0)


# ===========================================================================
# calc.astro.convert.HJD_phase
# ===========================================================================
def test_hjd_phase_at_epoch_is_zero():
    phase = calc.astro.convert.HJD_phase([2450000.0], 1.0, 2450000.0)
    assert phase[0] == pytest.approx(0.0)


def test_hjd_phase_half_period():
    phase = calc.astro.convert.HJD_phase([2450000.5], 1.0, 2450000.0)
    assert phase[0] == pytest.approx(0.5)


def test_hjd_phase_output_in_range():
    hjd = np.linspace(2450000, 2450010, 50).tolist()
    phases = calc.astro.convert.HJD_phase(hjd, 1.23456, 2450000.0)
    assert all(0.0 <= p < 1.0 for p in phases)


# ===========================================================================
# calc.astro.convert.JD_to_Greg
# ===========================================================================
def test_jd_to_greg_known_date():
    result = calc.astro.convert.JD_to_Greg(2451545.0)
    assert "2000" in result
    assert "01" in result


def test_jd_to_greg_returns_string():
    assert isinstance(calc.astro.convert.JD_to_Greg(2451545.0), str)


def test_jd_to_greg_format():
    assert len(calc.astro.convert.JD_to_Greg(2451545.0).split(" ")) == 3


# ===========================================================================
# calc.astro.convert.magToflux
# ===========================================================================
def test_mag_to_flux_zero_mag_gives_one():
    assert calc.astro.convert.magToflux.flux(0.0) == pytest.approx(1.0)


def test_mag_to_flux_is_positive():
    assert calc.astro.convert.magToflux.flux(5.0) > 0


def test_mag_to_flux_error_positive():
    assert calc.astro.convert.magToflux.error(5.0, 0.1) > 0


def test_mag_to_flux_brighter_means_higher_flux():
    assert calc.astro.convert.magToflux.flux(0.0) > calc.astro.convert.magToflux.flux(5.0)


# ===========================================================================
# calc.astro.convert.fluxTomag
# ===========================================================================
def test_flux_to_mag_one_gives_zero():
    assert calc.astro.convert.fluxTomag.mag(1.0) == pytest.approx(0.0)


def test_flux_to_mag_error_positive():
    assert calc.astro.convert.fluxTomag.error(1.0, 0.01) > 0


def test_flux_to_mag_roundtrip():
    mag_in = 14.5
    flux = calc.astro.convert.magToflux.flux(mag_in)
    assert calc.astro.convert.fluxTomag.mag(flux) == pytest.approx(mag_in, abs=1e-6)


# ===========================================================================
# FT.coefficients
# ===========================================================================
def test_ft_coefficients_mean_in_coslist():
    data = [1.0, 2.0, 3.0, 4.0]
    _, coslist, _ = FT.coefficients(data)
    assert coslist[0] == pytest.approx(np.mean(data))


def test_ft_coefficients_sinlist_zero_at_zero():
    _, _, sinlist = FT.coefficients([1.0, 2.0, 3.0, 4.0])
    assert sinlist[0] == pytest.approx(0.0)


def test_ft_coefficients_output_length():
    _, coslist, sinlist = FT.coefficients([1.0] * 8)
    assert len(coslist) == 8
    assert len(sinlist) == 8


# ===========================================================================
# FT.sumatphase
# ===========================================================================
def test_ft_sumatphase_dc_only():
    a = np.array([2.0, 0.0, 0.0])
    b = np.zeros(3)
    assert FT.sumatphase(0.0, 2, a, b) == pytest.approx(2.0)


def test_ft_sumatphase_sine_term():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    assert FT.sumatphase(0.25, 1, a, b) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# FT.synth
# ===========================================================================
def test_ft_synth_length():
    a = np.array([1.0, 0.1])
    b = np.zeros(2)
    assert len(FT.synth(a, b, [0.0, 0.25, 0.5, 0.75], 1)) == 4


def test_ft_synth_constant_signal():
    a = np.array([2.0, 0.0])
    b = np.zeros(2)
    for v in FT.synth(a, b, [0.0, 0.1, 0.5, 0.9], 1):
        assert v == pytest.approx(2.0, abs=1e-6)


# ===========================================================================
# FT.FT_plotlist
# ===========================================================================
def test_ft_plotlist_length():
    a = np.array([1.0, 0.0, 0.0])
    b = np.zeros(3)
    phases, fluxes, derivs = FT.FT_plotlist(a, b, 2, 100)
    assert len(phases) == 100
    assert len(fluxes) == 100
    assert len(derivs) == 100


def test_ft_plotlist_phases_in_range():
    a = np.array([1.0, 0.0])
    b = np.zeros(2)
    phases, _, _ = FT.FT_plotlist(a, b, 1, 50)
    assert all(0.0 <= p < 1.0 for p in phases)


# ===========================================================================
# FT.integral
# ===========================================================================
def test_ft_integral_zero_width():
    a = np.array([1.0, 0.0])
    b = np.zeros(2)
    assert FT.integral(a, b, 1, 0.3, 0.3) == pytest.approx(0.0, abs=1e-10)


def test_ft_integral_full_period_dc():
    a = np.array([2.0, 0.0])
    b = np.zeros(2)
    assert FT.integral(a, b, 1, 0, 1) == pytest.approx(2.0, abs=1e-6)


# ===========================================================================
# OConnell.Delta_I
# ===========================================================================
def test_oconnell_delta_i_symmetric():
    a = np.array([1.0, 0.5, 0.0])
    b = np.zeros(3)
    dI, Ip, Is = OConnell.Delta_I(a, b, 2)
    assert dI == pytest.approx(0.0, abs=1e-6)


def test_oconnell_delta_i_asymmetric_nonzero():
    a = np.zeros(3)
    b = np.array([0.0, 0.2, 0.0])
    dI, Ip, Is = OConnell.Delta_I(a, b, 2)
    assert abs(dI) > 0


def test_oconnell_delta_i_returns_three_values():
    assert len(OConnell.Delta_I(np.ones(3), np.zeros(3), 2)) == 3


# ===========================================================================
# OConnell.Delta_I_fixed
# ===========================================================================
def test_oconnell_delta_i_fixed_zero_b():
    assert OConnell.Delta_I_fixed(np.zeros(5), 4) == pytest.approx(0.0)


def test_oconnell_delta_i_fixed_nonzero():
    b = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    assert abs(OConnell.Delta_I_fixed(b, 4)) > 0


# ===========================================================================
# Flower.T.Teff
# ===========================================================================
def test_flower_teff_solar_bv():
    temp, err = Flower.T.Teff(0.65, 0.02)
    assert 5000 < temp < 7000


def test_flower_teff_hot_star():
    temp, err = Flower.T.Teff(-0.3, 0.02)
    assert temp > 8000


def test_flower_teff_error_nonzero():
    _, err = Flower.T.Teff(0.65, 0.02)
    assert err != 0
    assert abs(err) < 1000


# ===========================================================================
# Pecaut.T.Teff
# ===========================================================================
def test_pecaut_teff_mid_range():
    temp, err = Pecaut.T.Teff(0.5, 0.02)
    assert temp > 0
    assert err > 0


def test_pecaut_teff_out_of_range_returns_zero():
    temp, err = Pecaut.T.Teff(2.0, 0.01)
    assert temp == 0
    assert err == 0


def test_pecaut_teff_lower_range():
    temp, err = Pecaut.T.Teff(0.01, 0.005)
    assert temp > 0


# ===========================================================================
# Red.colorEx
# ===========================================================================
def test_red_color_ex_jk():
    assert Red.colorEx("J", "K", 1.0) == pytest.approx(Red.J_K)


def test_red_color_ex_jh():
    assert Red.colorEx("J", "H", 2.0) == pytest.approx(2.0 * Red.J_H)


def test_red_color_ex_vr():
    assert Red.colorEx("V", "R", 3.1) == pytest.approx(3.1 * Red.V_R)


def test_red_color_ex_unknown_returns_none():
    assert Red.colorEx("X", "Y", 1.0) is None


# ===========================================================================
# plot.amp
# ===========================================================================
def test_plot_amp_basic():
    assert plot.amp([1, 3, 5, 2]) == pytest.approx(4.0)


def test_plot_amp_flat():
    assert plot.amp([3, 3, 3]) == pytest.approx(0.0)


def test_plot_amp_negative():
    assert plot.amp([-5, -1]) == pytest.approx(4.0)


# ===========================================================================
# plot.aliasing2
# ===========================================================================
def test_plot_aliasing2_lengths_match():
    phases = [0.0, 0.2, 0.4, 0.6, 0.8]
    mags = [1.0] * 5
    errors = [0.01] * 5
    a_phase, a_mag, a_error = plot.aliasing2(phases, mags, errors)
    assert len(a_phase) == len(a_mag) == len(a_error)


def test_plot_aliasing2_within_alias_range():
    phases = [0.0, 0.2, 0.4, 0.6, 0.8]
    mags = [1.0] * 5
    errors = [0.01] * 5
    alias = 0.6
    a_phase, _, _ = plot.aliasing2(phases, mags, errors, alias=alias)
    assert all(-alias < p < alias for p in a_phase)


# ===========================================================================
# binning.norm_flux
# ===========================================================================
def test_binning_norm_flux_max_one_bin():
    binned = [0.5, 0.8, 1.0]
    ob = [0.3, 0.7, 0.9]
    err = [0.01, 0.01, 0.01]
    n_bin, n_ob, n_err = binning.norm_flux(binned, ob, err, norm_factor='bin')
    assert max(n_bin) == pytest.approx(1.0)


def test_binning_norm_flux_max_one_ob():
    binned = [0.5, 0.8]
    ob = [0.3, 1.0]
    err = [0.01, 0.01]
    n_bin, n_ob, n_err = binning.norm_flux(binned, ob, err, norm_factor='ob')
    assert max(n_ob) == pytest.approx(1.0)


def test_binning_norm_flux_length_preserved():
    binned = [0.5, 0.8, 1.0]
    ob = [0.3, 0.7, 0.9]
    err = [0.01, 0.01, 0.01]
    n_bin, n_ob, n_err = binning.norm_flux(binned, ob, err)
    assert len(n_bin) == 3
    assert len(n_ob) == 3
    assert len(n_err) == 3


# ===========================================================================
# M and wfactor
# ===========================================================================
def test_M_single_error():
    assert M([2.0]) == pytest.approx(0.25)


def test_M_multiple():
    assert M([1.0, 1.0]) == pytest.approx(2.0)


def test_wfactor_sums_to_one():
    errors = [1.0, 1.0, 1.0]
    total_M = M(errors)
    total_w = sum(wfactor(errors, n, total_M) for n in range(len(errors)))
    assert total_w == pytest.approx(1.0)


# ===========================================================================
# Roche.Lagrange_123
# ===========================================================================
def test_roche_lagrange_equal_mass():
    L1, L2, L3 = Roche.Lagrange_123(1.0)
    assert isinstance(L1, float)
    assert 0 < L1 < 1


def test_roche_lagrange_asymmetric():
    L1, L2, L3 = Roche.Lagrange_123(0.5)
    assert L1 > 0
    assert L2 > 0


# ===========================================================================
# calc.Newton
# ===========================================================================
def test_calc_newton_square_root():
    root = calc.Newton(lambda x: x ** 2 - 2, 1.0)
    assert root == pytest.approx(np.sqrt(2), abs=1e-6)


def test_calc_newton_linear():
    root = calc.Newton(lambda x: x - 3, 0.0)
    assert root == pytest.approx(3.0, abs=1e-6)


def test_calc_newton_max_iter_returns_false():
    f = lambda x: x ** 3 - x - 1000
    assert calc.Newton(f, 0.0, e=1e-15, max_iter=2) is False
