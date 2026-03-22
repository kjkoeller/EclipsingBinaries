# -*- coding: utf-8 -*-
"""
Tests for gaia.py

"""

import pytest
import math
import os
import tempfile
from unittest.mock import patch, MagicMock


# ===========================================================================
# Pure math helpers — extracted from tess_mag
# These mirror the TESS magnitude formula and error propagation exactly.
# ===========================================================================

def _tess_mag_from_gaia(G, BP, RP):
    """TESS magnitude from Gaia G, BP, RP (scalar values)."""
    return (G - 0.00522555 * (BP - RP) ** 3
            + 0.0891337 * (BP - RP) ** 2
            - 0.633923 * (BP - RP)
            + 0.0324473)


def _flux_to_mag_err(flux_over_error):
    """Convert Gaia flux/error ratio to magnitude error."""
    return (2.5 / math.log(10)) * (flux_over_error ** -1)


def _tess_mag_error(G_flux, BP_flux, RP_flux):
    """Full TESS magnitude error propagation."""
    G_err = _flux_to_mag_err(G_flux)
    BP_err = _flux_to_mag_err(BP_flux)
    RP_err = _flux_to_mag_err(RP_flux)
    Bp_Rp_err = math.sqrt(BP_err ** 2 + RP_err ** 2)
    return math.sqrt(G_err ** 2 + (Bp_Rp_err * 3) ** 2)


def _tess_mag_no_bp_rp(G, G_flux):
    """Fallback TESS magnitude when BP or RP are unavailable."""
    return G - 0.403, _flux_to_mag_err(G_flux)


# ===========================================================================
# TESS magnitude formula
# ===========================================================================
def test_tess_mag_formula_solar_colors():
    # Sun-like star: G~6, BP-RP~0.82 → T should be slightly fainter than G
    G, BP, RP = 6.0, 6.5, 5.68
    T = _tess_mag_from_gaia(G, BP, RP)
    assert isinstance(T, float)
    assert 4.0 < T < 8.0


def test_tess_mag_formula_zero_color():
    # BP - RP = 0 → cubic, quadratic, linear terms all vanish → T = G + 0.0324473
    G, BP, RP = 12.0, 10.0, 10.0
    T = _tess_mag_from_gaia(G, BP, RP)
    assert T == pytest.approx(12.0 + 0.0324473, abs=1e-6)


def test_tess_mag_formula_known_value():
    # Manual calculation: G=14, BP=14.5, RP=13.7 → BP-RP=0.8
    G, BP, RP = 14.0, 14.5, 13.7
    bprp = BP - RP
    expected = (G
                - 0.00522555 * bprp ** 3
                + 0.0891337 * bprp ** 2
                - 0.633923 * bprp
                + 0.0324473)
    assert _tess_mag_from_gaia(G, BP, RP) == pytest.approx(expected, abs=1e-10)


def test_tess_mag_formula_blue_star():
    # Blue star: BP - RP < 0
    G, BP, RP = 8.0, 7.5, 8.5
    T = _tess_mag_from_gaia(G, BP, RP)
    assert isinstance(T, float)
    assert T > G  # blue stars have T > G for negative BP-RP


def test_tess_mag_formula_red_star():
    # Red star: large positive BP - RP
    G, BP, RP = 15.0, 17.0, 13.5
    T = _tess_mag_from_gaia(G, BP, RP)
    assert isinstance(T, float)
    assert T < G  # red stars have T < G


def test_tess_mag_formula_scales_with_g():
    # Increasing G by 1 mag should increase T by exactly 1 mag
    G, BP, RP = 12.0, 12.5, 11.8
    T1 = _tess_mag_from_gaia(G, BP, RP)
    T2 = _tess_mag_from_gaia(G + 1.0, BP + 1.0, RP + 1.0)
    assert T2 - T1 == pytest.approx(1.0, abs=1e-10)


# ===========================================================================
# Flux-to-magnitude error conversion
# ===========================================================================
def test_flux_err_high_snr_gives_small_error():
    # High SNR (large flux/error ratio) → small magnitude error
    assert _flux_to_mag_err(1000.0) < 0.01


def test_flux_err_low_snr_gives_large_error():
    # Low SNR → large magnitude error
    assert _flux_to_mag_err(5.0) > 0.1


def test_flux_err_known_value():
    # (2.5 / ln(10)) / flux_over_error
    expected = (2.5 / math.log(10)) / 100.0
    assert _flux_to_mag_err(100.0) == pytest.approx(expected, abs=1e-12)


def test_flux_err_positive():
    assert _flux_to_mag_err(50.0) > 0


def test_flux_err_scales_inversely():
    # Doubling flux/error should halve the magnitude error
    err1 = _flux_to_mag_err(100.0)
    err2 = _flux_to_mag_err(200.0)
    assert err1 == pytest.approx(2 * err2, abs=1e-10)


# ===========================================================================
# TESS magnitude error propagation
# ===========================================================================
def test_tess_err_positive():
    assert _tess_mag_error(200.0, 150.0, 180.0) > 0


def test_tess_err_dominated_by_bp_rp_when_noisy():
    # Factor of 3 on Bp_Rp_err means BP/RP noise dominates
    err_noisy_bprp = _tess_mag_error(1000.0, 10.0, 10.0)
    err_noisy_g = _tess_mag_error(10.0, 1000.0, 1000.0)
    assert err_noisy_bprp > err_noisy_g


def test_tess_err_increases_with_lower_snr():
    err_high = _tess_mag_error(500.0, 400.0, 450.0)
    err_low = _tess_mag_error(50.0, 40.0, 45.0)
    assert err_low > err_high


def test_tess_err_known_value():
    G_snr, BP_snr, RP_snr = 100.0, 80.0, 90.0
    G_err = (2.5 / math.log(10)) / G_snr
    BP_err = (2.5 / math.log(10)) / BP_snr
    RP_err = (2.5 / math.log(10)) / RP_snr
    Bp_Rp_err = math.sqrt(BP_err ** 2 + RP_err ** 2)
    expected = math.sqrt(G_err ** 2 + (Bp_Rp_err * 3) ** 2)
    assert _tess_mag_error(G_snr, BP_snr, RP_snr) == pytest.approx(expected, abs=1e-12)


# ===========================================================================
# Fallback magnitude (no BP/RP)
# ===========================================================================
def test_no_bp_rp_fallback_formula():
    G, G_flux = 14.0, 100.0
    T, T_err = _tess_mag_no_bp_rp(G, G_flux)
    assert T == pytest.approx(14.0 - 0.403, abs=1e-10)


def test_no_bp_rp_fallback_error():
    G, G_flux = 14.0, 100.0
    _, T_err = _tess_mag_no_bp_rp(G, G_flux)
    expected = (2.5 / math.log(10)) / G_flux
    assert T_err == pytest.approx(expected, abs=1e-12)


def test_no_bp_rp_fallback_is_fainter_than_g():
    # T = G - 0.403 → T is 0.403 mag fainter (numerically smaller) than G
    G = 12.0
    T, _ = _tess_mag_no_bp_rp(G, 200.0)
    assert T < G


# ===========================================================================
# target_star — network mocked
# ===========================================================================
def _make_mock_gaia_result():
    """Build a minimal mock GaiaData object matching what target_star accesses."""
    import astropy.units as u
    import numpy as np

    mock = MagicMock()
    mock.parallax = [1.0, 1.1, 1.2, 1.3] * u.mas
    mock.parallax_error = [0.01, 0.01, 0.01, 0.01] * u.mas
    mock.distance_gspphot_lower = [300.0, 310.0, 320.0, 330.0] * u.pc
    mock.distance_gspphot = [320.0, 330.0, 340.0, 350.0] * u.pc
    mock.distance_gspphot_upper = [340.0, 350.0, 360.0, 370.0] * u.pc
    mock.teff_gspphot_lower = [5500.0, 5600.0, 5700.0, 5800.0] * u.K
    mock.teff_gspphot = [5700.0, 5800.0, 5900.0, 6000.0] * u.K
    mock.teff_gspphot_upper = [5900.0, 6000.0, 6100.0, 6200.0] * u.K
    mock.phot_g_mean_mag = [12.0, 12.1, 12.2, 12.3] * u.mag
    mock.phot_bp_mean_mag = [12.5, 12.6, 12.7, 12.8] * u.mag
    mock.phot_rp_mean_mag = [11.7, 11.8, 11.9, 12.0] * u.mag
    mock.radial_velocity = [-10.0, -10.1, -10.2, -10.3] * u.km / u.s
    mock.radial_velocity_error = [0.5, 0.5, 0.5, 0.5] * u.km / u.s
    return mock


def test_target_star_creates_output_file():
    from EclipsingBinaries.gaia import target_star
    cancel = MagicMock()
    cancel.is_set.return_value = False

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.gaia.GaiaData.from_query",
                   return_value=_make_mock_gaia_result()):
            target_star("12:34:56.78", "45:00:00.00", tmpdir,
                        write_callback=None, cancel_event=cancel)
        assert os.path.isfile(os.path.join(tmpdir, "gaia_results.csv"))


def test_target_star_logs_completion():
    from EclipsingBinaries.gaia import target_star
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.gaia.GaiaData.from_query",
                   return_value=_make_mock_gaia_result()):
            target_star("12:34:56.78", "45:00:00.00", tmpdir,
                        write_callback=messages.append, cancel_event=cancel)

    assert any("Finished" in m for m in messages)


def test_target_star_cancels_before_query():
    from EclipsingBinaries.gaia import target_star
    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.gaia.GaiaData.from_query") as mock_query:
            target_star("12:34:56.78", "45:00:00.00", tmpdir,
                        write_callback=messages.append, cancel_event=cancel)
        mock_query.assert_not_called()

    assert any("cancel" in m.lower() for m in messages)


def test_target_star_output_has_expected_columns():
    from EclipsingBinaries.gaia import target_star
    import pandas as pd
    cancel = MagicMock()
    cancel.is_set.return_value = False

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.gaia.GaiaData.from_query",
                   return_value=_make_mock_gaia_result()):
            target_star("12:34:56.78", "45:00:00.00", tmpdir,
                        write_callback=None, cancel_event=cancel)

        df = pd.read_csv(os.path.join(tmpdir, "gaia_results.csv"), sep="\t",
                         index_col=0)
        expected_cols = [
            "Parallax(mas)", "Parallax_err(mas)",
            "Distance(pc)", "T_eff(K)", "G_Mag",
            "G_BP_Mag", "G_RP_Mag",
            "Radial_velocity(km/s)", "Radial_velocity_err(km/s)"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


def test_target_star_logs_error_on_exception():
    from EclipsingBinaries.gaia import target_star
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.gaia.GaiaData.from_query",
                   side_effect=RuntimeError("network error")):
            with pytest.raises(RuntimeError):
                target_star("12:34:56.78", "45:00:00.00", tmpdir,
                            write_callback=messages.append, cancel_event=cancel)

    assert any("error" in m.lower() for m in messages)


# ===========================================================================
# tess_mag — network mocked
# ===========================================================================
def _make_mock_tess_gaia():
    """Mock GaiaData for tess_mag with realistic flux/error ratios."""
    import numpy as np
    mock = MagicMock()
    mock.phot_g_mean_mag.__getitem__ = lambda self, s: MagicMock(value=np.array([12.0, 12.1, 12.2, 12.3]))
    mock.phot_g_mean_flux_over_error.__getitem__ = lambda self, s: np.array([200.0, 190.0, 180.0, 170.0])
    mock.phot_bp_mean_mag.__getitem__ = lambda self, s: MagicMock(value=np.array([12.5, 12.6, 12.7, 12.8]))
    mock.phot_bp_mean_flux_over_error.__getitem__ = lambda self, s: np.array([150.0, 140.0, 130.0, 120.0])
    mock.phot_rp_mean_mag.__getitem__ = lambda self, s: MagicMock(value=np.array([11.7, 11.8, 11.9, 12.0]))
    mock.phot_rp_mean_flux_over_error.__getitem__ = lambda self, s: np.array([180.0, 170.0, 160.0, 150.0])
    return mock


def test_tess_mag_returns_two_lists():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    with patch("EclipsingBinaries.gaia.GaiaData.from_query",
               return_value=_make_mock_tess_gaia()):
        result = tess_mag([0.8356], [78.961], None, cancel)

    assert result is not None
    assert len(result) == 2


def test_tess_mag_lists_same_length():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    with patch("EclipsingBinaries.gaia.GaiaData.from_query",
               return_value=_make_mock_tess_gaia()):
        T_list, T_err_list = tess_mag([0.8356, 0.9], [78.961, 79.0], None, cancel)

    assert len(T_list) == len(T_err_list)


def test_tess_mag_cancels_before_loop():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []

    with patch("EclipsingBinaries.gaia.GaiaData.from_query") as mock_query:
        result = tess_mag([0.8356], [78.961], messages.append, cancel)

    mock_query.assert_not_called()
    assert result is None
    assert any("cancel" in m.lower() for m in messages)


def test_tess_mag_logs_completion():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with patch("EclipsingBinaries.gaia.GaiaData.from_query",
               return_value=_make_mock_tess_gaia()):
        tess_mag([0.8356], [78.961], messages.append, cancel)

    assert any("Finished" in m for m in messages)


def test_tess_mag_logs_error_on_exception():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with patch("EclipsingBinaries.gaia.GaiaData.from_query",
               side_effect=RuntimeError("timeout")):
        with pytest.raises(RuntimeError):
            tess_mag([0.8356], [78.961], messages.append, cancel)

    assert any("error" in m.lower() for m in messages)
