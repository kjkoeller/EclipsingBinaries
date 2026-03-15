# -*- coding: utf-8 -*-
"""
Tests for OConnell.py

"""

import pytest
import numpy as np

from EclipsingBinaries.OConnell import sig_f, dI_phi


# ===========================================================================
# sig_f   lambda f, x, sig_x: abs(f(x + sig_x) - f(x - sig_x)) / 2
# ===========================================================================
class TestSigF:
    """
    sig_f propagates uncertainty through a function using a symmetric
    finite-difference approximation: |f(x+dx) - f(x-dx)| / 2
    """

    def test_linear_function_exact(self):
        # f(x) = 3x  → sig_f = |3(x+dx) - 3(x-dx)| / 2 = 3*dx
        f = lambda x: 3 * x
        assert sig_f(f, 5.0, 0.1) == pytest.approx(3 * 0.1, abs=1e-10)

    def test_constant_function_zero(self):
        # f(x) = 7  → no dependence on x → sig = 0
        f = lambda x: 7.0
        assert sig_f(f, 3.0, 0.5) == pytest.approx(0.0, abs=1e-10)

    def test_quadratic_near_zero(self):
        # f(x) = x²  → sig_f ≈ 2*x*dx  (central diff is exact for quadratics)
        f = lambda x: x ** 2
        x, dx = 4.0, 0.01
        expected = abs((x + dx) ** 2 - (x - dx) ** 2) / 2
        assert sig_f(f, x, dx) == pytest.approx(expected, abs=1e-12)

    def test_always_nonnegative(self):
        # abs() guarantees non-negative result for any function
        f = lambda x: -10 * x
        assert sig_f(f, 2.0, 0.5) >= 0

    def test_scales_with_sig_x(self):
        # Doubling sig_x should double the result for linear f
        f = lambda x: 5 * x
        result1 = sig_f(f, 3.0, 0.1)
        result2 = sig_f(f, 3.0, 0.2)
        assert result2 == pytest.approx(2 * result1, abs=1e-10)

    def test_zero_uncertainty_gives_zero(self):
        f = lambda x: x ** 3
        assert sig_f(f, 2.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_polynomial_propagation(self):
        # f(x) = x³; analytical deriv = 3x² so sig ≈ 3x²*dx (exact by central diff)
        f = lambda x: x ** 3
        x, dx = 2.0, 0.001
        expected = abs(f(x + dx) - f(x - dx)) / 2
        assert sig_f(f, x, dx) == pytest.approx(expected, abs=1e-12)

    def test_returns_float(self):
        result = sig_f(lambda x: x, 1.0, 0.1)
        assert isinstance(result, float)

    def test_symmetric_around_x(self):
        # sig_f should give the same result regardless of sign of x for even functions
        f = lambda x: x ** 2
        assert sig_f(f, 3.0, 0.1) == pytest.approx(sig_f(f, -3.0, 0.1), abs=1e-10)

    def test_sin_function(self):
        # f(x) = sin(x); central diff ≈ cos(x)*dx
        f = np.sin
        x, dx = np.pi / 4, 0.001
        expected = abs(np.sin(x + dx) - np.sin(x - dx)) / 2
        assert sig_f(f, x, dx) == pytest.approx(expected, abs=1e-12)


# ===========================================================================
# dI_phi  lambda b, phase, order:
#         2 * sum(b[1:order+1] * sin(2π * phase * [1..order]))
# ===========================================================================
class TestDIPhi:
    """
    dI_phi computes the sine-component contribution to the O'Connell
    flux difference at a given phase:
        dI(φ) = 2 * Σ_{k=1}^{order} b_k * sin(2π k φ)
    """

    def test_zero_b_coefficients(self):
        b = np.zeros(11)
        assert dI_phi(b, 0.25, 10) == pytest.approx(0.0, abs=1e-10)

    def test_zero_phase(self):
        # sin(0) = 0 for all terms → result is 0 regardless of b
        b = np.array([0.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
        assert dI_phi(b, 0.0, 10) == pytest.approx(0.0, abs=1e-10)

    def test_half_phase(self):
        # sin(2π*k*0.5) = sin(πk) = 0 for all integer k → result is 0
        b = np.array([0.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
        assert dI_phi(b, 0.5, 10) == pytest.approx(0.0, abs=1e-10)

    def test_single_term_order_1(self):
        # b = [0, B1, 0, ...]; dI = 2 * B1 * sin(2π * 1 * φ)
        b = np.zeros(11)
        b[1] = 1.0
        phase = 0.25
        expected = 2 * 1.0 * np.sin(2 * np.pi * 1 * phase)
        assert dI_phi(b, phase, 10) == pytest.approx(expected, abs=1e-10)

    def test_single_term_order_2(self):
        b = np.zeros(11)
        b[2] = 0.5
        phase = 0.1
        expected = 2 * 0.5 * np.sin(2 * np.pi * 2 * phase)
        assert dI_phi(b, phase, 10) == pytest.approx(expected, abs=1e-10)

    def test_multiple_terms(self):
        b = np.zeros(11)
        b[1] = 1.0
        b[2] = 0.5
        b[3] = 0.25
        phase = 0.3
        expected = 2 * (
            1.0 * np.sin(2 * np.pi * 1 * phase) +
            0.5 * np.sin(2 * np.pi * 2 * phase) +
            0.25 * np.sin(2 * np.pi * 3 * phase)
        )
        assert dI_phi(b, phase, 10) == pytest.approx(expected, abs=1e-10)

    def test_order_zero_gives_zero(self):
        # order=0 → sum over empty range → 0
        b = np.zeros(11)
        b[0] = 99.9  # b[0] (DC term) is not included
        assert dI_phi(b, 0.25, 0) == pytest.approx(0.0, abs=1e-10)

    def test_antisymmetry_at_quadrature(self):
        # dI(0.25) = -dI(0.75) for pure sine series (odd symmetry around 0.5)
        b = np.zeros(11)
        b[1] = 0.3
        b[3] = 0.1
        val_025 = dI_phi(b, 0.25, 10)
        val_075 = dI_phi(b, 0.75, 10)
        assert val_025 == pytest.approx(-val_075, abs=1e-10)

    def test_linearity_in_b(self):
        # dI_phi(2*b) = 2 * dI_phi(b)
        b = np.zeros(11)
        b[1] = 0.4
        b[2] = 0.2
        phase = 0.15
        result1 = dI_phi(b, phase, 10)
        result2 = dI_phi(2 * b, phase, 10)
        assert result2 == pytest.approx(2 * result1, abs=1e-10)

    def test_order_truncation(self):
        # Terms beyond `order` should be ignored
        b = np.zeros(15)
        b[1] = 1.0
        b[11] = 999.0  # beyond order=10, should be ignored
        phase = 0.2
        expected = 2 * 1.0 * np.sin(2 * np.pi * phase)
        assert dI_phi(b, phase, 10) == pytest.approx(expected, abs=1e-10)

    def test_result_at_quadrature_only_odd_terms(self):
        # At phase=0.25: sin(2π*k*0.25) = sin(πk/2)
        # k=1: sin(π/2)=1; k=2: sin(π)=0; k=3: sin(3π/2)=-1; k=4: sin(2π)=0
        b = np.zeros(5)
        b[1] = 1.0
        b[2] = 1.0
        b[3] = 1.0
        b[4] = 1.0
        phase = 0.25
        expected = 2 * (1.0 * 1 + 1.0 * 0 + 1.0 * (-1) + 1.0 * 0)
        assert dI_phi(b, phase, 4) == pytest.approx(expected, abs=1e-10)

    def test_b0_ignored(self):
        # b[0] is the DC offset and should not appear in dI_phi
        b1 = np.zeros(11)
        b1[0] = 100.0
        b2 = np.zeros(11)
        b2[0] = 0.0
        phase = 0.33
        assert dI_phi(b1, phase, 10) == pytest.approx(dI_phi(b2, phase, 10), abs=1e-10)

    def test_continuous_phase_variation(self):
        # Verify output is continuous and finite across a range of phases
        b = np.zeros(11)
        b[1] = 0.2
        b[3] = 0.1
        phases = np.linspace(0, 1, 100)
        results = [dI_phi(b, p, 10) for p in phases]
        assert all(np.isfinite(r) for r in results)

    def test_numpy_b_array(self):
        b = np.array([0.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002])
        result = dI_phi(b, 0.25, 10)
        assert np.isfinite(result)