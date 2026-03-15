# -*- coding: utf-8 -*-
"""
Tests for OConnell.py

Run with:
    pytest tests/test_OConnell.py -v --cov=EclipsingBinaries.OConnell
"""

import pytest
import numpy as np

from EclipsingBinaries.OConnell import sig_f, dI_phi


def test_sig_f_linear_exact():
    assert sig_f(lambda x: 3 * x, 5.0, 0.1) == pytest.approx(3 * 0.1, abs=1e-10)


def test_sig_f_constant_zero():
    assert sig_f(lambda x: 7.0, 3.0, 0.5) == pytest.approx(0.0, abs=1e-10)


def test_sig_f_quadratic():
    x, dx = 4.0, 0.01
    expected = abs((x + dx) ** 2 - (x - dx) ** 2) / 2
    assert sig_f(lambda v: v ** 2, x, dx) == pytest.approx(expected, abs=1e-12)


def test_sig_f_always_nonnegative():
    assert sig_f(lambda x: -10 * x, 2.0, 0.5) >= 0


def test_sig_f_scales_with_sig_x():
    f = lambda x: 5 * x
    assert sig_f(f, 3.0, 0.2) == pytest.approx(2 * sig_f(f, 3.0, 0.1), abs=1e-10)


def test_sig_f_zero_uncertainty_gives_zero():
    assert sig_f(lambda x: x ** 3, 2.0, 0.0) == pytest.approx(0.0, abs=1e-10)


def test_sig_f_polynomial_propagation():
    x, dx = 2.0, 0.001
    f = lambda v: v ** 3
    assert sig_f(f, x, dx) == pytest.approx(abs(f(x + dx) - f(x - dx)) / 2, abs=1e-12)


def test_sig_f_returns_float():
    assert isinstance(sig_f(lambda x: x, 1.0, 0.1), float)


def test_sig_f_symmetric_even_function():
    f = lambda x: x ** 2
    assert sig_f(f, 3.0, 0.1) == pytest.approx(sig_f(f, -3.0, 0.1), abs=1e-10)


def test_sig_f_sin_function():
    x, dx = np.pi / 4, 0.001
    expected = abs(np.sin(x + dx) - np.sin(x - dx)) / 2
    assert sig_f(np.sin, x, dx) == pytest.approx(expected, abs=1e-12)


def test_di_phi_zero_b_coefficients():
    assert dI_phi(np.zeros(11), 0.25, 10) == pytest.approx(0.0, abs=1e-10)


def test_di_phi_zero_phase():
    b = np.array([0.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    assert dI_phi(b, 0.0, 10) == pytest.approx(0.0, abs=1e-10)


def test_di_phi_half_phase():
    b = np.array([0.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    assert dI_phi(b, 0.5, 10) == pytest.approx(0.0, abs=1e-10)


def test_di_phi_single_term_order_1():
    b = np.zeros(11)
    b[1] = 1.0
    phase = 0.25
    assert dI_phi(b, phase, 10) == pytest.approx(2 * np.sin(2 * np.pi * phase), abs=1e-10)


def test_di_phi_single_term_order_2():
    b = np.zeros(11)
    b[2] = 0.5
    phase = 0.1
    assert dI_phi(b, phase, 10) == pytest.approx(2 * 0.5 * np.sin(2 * np.pi * 2 * phase), abs=1e-10)


def test_di_phi_multiple_terms():
    b = np.zeros(11)
    b[1], b[2], b[3] = 1.0, 0.5, 0.25
    phase = 0.3
    expected = 2 * (
        1.0 * np.sin(2 * np.pi * phase) +
        0.5 * np.sin(2 * np.pi * 2 * phase) +
        0.25 * np.sin(2 * np.pi * 3 * phase)
    )
    assert dI_phi(b, phase, 10) == pytest.approx(expected, abs=1e-10)


def test_di_phi_order_zero_gives_zero():
    b = np.zeros(11)
    b[0] = 99.9
    assert dI_phi(b, 0.25, 0) == pytest.approx(0.0, abs=1e-10)


def test_di_phi_antisymmetry_at_quadrature():
    b = np.zeros(11)
    b[1] = 0.3
    b[3] = 0.1
    assert dI_phi(b, 0.25, 10) == pytest.approx(-dI_phi(b, 0.75, 10), abs=1e-10)


def test_di_phi_linearity_in_b():
    b = np.zeros(11)
    b[1], b[2] = 0.4, 0.2
    phase = 0.15
    assert dI_phi(2 * b, phase, 10) == pytest.approx(2 * dI_phi(b, phase, 10), abs=1e-10)


def test_di_phi_order_truncation():
    b = np.zeros(15)
    b[1] = 1.0
    b[11] = 999.0
    phase = 0.2
    assert dI_phi(b, phase, 10) == pytest.approx(2 * np.sin(2 * np.pi * phase), abs=1e-10)


def test_di_phi_quadrature_only_odd_terms():
    b = np.zeros(5)
    b[1] = b[2] = b[3] = b[4] = 1.0
    assert dI_phi(b, 0.25, 4) == pytest.approx(2 * (1 + 0 - 1 + 0), abs=1e-10)


def test_di_phi_b0_ignored():
    b1, b2 = np.zeros(11), np.zeros(11)
    b1[0] = 100.0
    assert dI_phi(b1, 0.33, 10) == pytest.approx(dI_phi(b2, 0.33, 10), abs=1e-10)


def test_di_phi_finite_across_phases():
    b = np.zeros(11)
    b[1], b[3] = 0.2, 0.1
    assert all(np.isfinite(dI_phi(b, p, 10)) for p in np.linspace(0, 1, 100))


def test_di_phi_numpy_b_array():
    b = np.array([0.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002])
    assert np.isfinite(dI_phi(b, 0.25, 10))
