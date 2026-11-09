# -*- coding: utf-8 -*-
"""
Tests for gaia.py

"""

import pytest
import math
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ===========================================================================
# TESS magnitude from Gaia photometry (the backup when the TIC has none)
# Stassun et al. 2019, AJ 158, 138, equations 1 and 2
# ===========================================================================

def _tic_polynomial(G, bp_rp):
    return (G - 0.00522555 * bp_rp ** 3 + 0.0891337 * bp_rp ** 2
            - 0.633923 * bp_rp + 0.0324473)


def _flux_to_mag_err(flux_over_error):
    """Gaia flux over error as a magnitude error."""
    return (2.5 / math.log(10)) / flux_over_error


def test_gaia_tess_magnitude_zero_colour():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    T, _ = gaia_tess_magnitude(12.0, 0.0)
    assert T == pytest.approx(12.0 + 0.0324473, abs=1e-3)


def test_gaia_tess_magnitude_known_value():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    T, _ = gaia_tess_magnitude(14.0, 0.8)
    assert T == pytest.approx(_tic_polynomial(14.0, 0.8), abs=1e-3)


def test_gaia_tess_magnitude_blue_and_red_stars():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    assert gaia_tess_magnitude(8.0, -0.1)[0] > 8.0   # blue stars are fainter in TESS than in G
    assert gaia_tess_magnitude(15.0, 2.5)[0] < 15.0  # red stars are brighter


def test_gaia_tess_magnitude_follows_g():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    assert gaia_tess_magnitude(13.0, 0.9)[0] - gaia_tess_magnitude(12.0, 0.9)[0] == pytest.approx(1.0)


def test_gaia_tess_error_includes_the_relation_scatter():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    # With perfect photometry only the TIC relation's 0.006 mag scatter is left
    assert gaia_tess_magnitude(12.0, 0.8, 0.0, 0.0)[1] == pytest.approx(0.006, abs=1e-3)


def test_gaia_tess_error_carries_the_colour_error_through():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    g_err, colour_err, x = 0.003, 0.02, 0.8
    slope = -3 * 0.00522555 * x ** 2 + 2 * 0.0891337 * x - 0.633923
    expected = math.sqrt(g_err ** 2 + (slope * colour_err) ** 2 + 0.006 ** 2)
    assert gaia_tess_magnitude(12.0, x, g_err, colour_err)[1] == pytest.approx(expected, abs=1e-3)


def test_gaia_tess_error_grows_with_noisier_photometry():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    assert gaia_tess_magnitude(12.0, 0.8, 0.05, 0.1)[1] > gaia_tess_magnitude(12.0, 0.8, 0.005, 0.01)[1]


def test_gaia_tess_error_is_widened_outside_the_fitted_colours():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    inside = gaia_tess_magnitude(12.0, 3.4, 0.0, 0.0)[1]
    outside = gaia_tess_magnitude(12.0, 4.0, 0.0, 0.0)[1]
    assert inside == pytest.approx(0.006, abs=1e-3)
    assert outside == pytest.approx(math.hypot(0.006, 0.1), abs=1e-3)


def test_gaia_tess_magnitude_without_a_colour():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    # T = G - 0.430 with the TIC's conservative 0.6 mag error
    T, T_err = gaia_tess_magnitude(14.0, np.nan, 0.002)
    assert T == pytest.approx(13.570, abs=1e-3)
    assert T_err == pytest.approx(0.6, abs=1e-3)


def test_gaia_tess_magnitude_without_a_g_magnitude():
    from EclipsingBinaries.gaia import gaia_tess_magnitude
    assert gaia_tess_magnitude(np.nan, 0.8) == (99.999, 99.999)


def test_flux_err_known_value():
    from EclipsingBinaries.gaia import _magnitude_error
    assert _magnitude_error(np.array([100.0]))[0] == pytest.approx(_flux_to_mag_err(100.0))


def test_flux_err_missing_value_is_nan():
    from EclipsingBinaries.gaia import _magnitude_error
    assert np.isnan(_magnitude_error(np.ma.masked_array([0.0], mask=[True]))[0])


# ===========================================================================
# target_star — network mocked via GaiaData.from_query
# ===========================================================================
TARGET_RA = "12:34:56.78"   # 188.736583 deg
TARGET_DEC = "45:00:00.00"
TARGET_RA_DEG = 188.736583333
TARGET_DEC_DEG = 45.0


def _make_mock_gaia_result(rows=None):
    """
    Build a real GaiaData object like the archive would return.

    Each row is (source_id, ra_offset_arcsec, dec_offset_arcsec, pmra, pmdec).
    Offsets are measured from the target at Gaia's 2016 epoch.
    """
    from astropy.table import Table
    from pyia import GaiaData

    if rows is None:
        # Deliberately out of order: the closest source is second
        rows = [(30, 3.0, 0.0, 0.0, 0.0), (10, 0.2, 0.0, 0.0, 0.0),
                (20, 0.0, 1.5, 0.0, 0.0), (40, 0.0, -4.0, 0.0, 0.0)]
    n = len(rows)
    cos_dec = np.cos(np.radians(TARGET_DEC_DEG))
    table = Table({
        "source_id": [r[0] for r in rows],
        "ra": [TARGET_RA_DEG + r[1] / 3600 / cos_dec for r in rows],
        "dec": [TARGET_DEC_DEG + r[2] / 3600 for r in rows],
        "pmra": [r[3] for r in rows],
        "pmdec": [r[4] for r in rows],
        # Values tagged with the source_id so the tests can tell rows apart
        "parallax": [r[0] / 10 for r in rows],
        "parallax_error": [0.01] * n,
        "distance_gspphot_lower": [r[0] * 10 + 1.0 for r in rows],
        "distance_gspphot": [r[0] * 10 + 2.0 for r in rows],
        "distance_gspphot_upper": [r[0] * 10 + 3.0 for r in rows],
        "teff_gspphot_lower": [5500.0] * n,
        "teff_gspphot": [5700.0] * n,
        "teff_gspphot_upper": [5900.0] * n,
        "phot_g_mean_mag": [12.0] * n,
        "phot_bp_mean_mag": [12.5] * n,
        "phot_rp_mean_mag": [11.7] * n,
        "radial_velocity": [-10.0] * n,
        "radial_velocity_error": [0.5] * n,
    })
    return GaiaData(table)


def _run_target_star(tmpdir, result, messages=None):
    from EclipsingBinaries.gaia import target_star
    with patch("EclipsingBinaries.gaia.GaiaData.from_query", return_value=result) as query:
        target_star(TARGET_RA, TARGET_DEC, tmpdir,
                    write_callback=(messages.append if messages is not None else None))
    return query, pd.read_csv(os.path.join(tmpdir, "gaia_results.csv"))


def test_target_star_creates_output_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_target_star(tmpdir, _make_mock_gaia_result())
        assert os.path.isfile(os.path.join(tmpdir, "gaia_results.csv"))


def test_target_star_logs_completion():
    messages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_target_star(tmpdir, _make_mock_gaia_result(), messages)
    assert any("Finished" in m for m in messages)


def test_target_star_cancels_before_query():
    from EclipsingBinaries.gaia import target_star
    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.gaia.GaiaData.from_query") as mock_query:
            target_star(TARGET_RA, TARGET_DEC, tmpdir,
                        write_callback=messages.append, cancel_event=cancel)
        mock_query.assert_not_called()

    assert any("cancel" in m.lower() for m in messages)


def test_target_star_output_has_expected_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, df = _run_target_star(tmpdir, _make_mock_gaia_result())
    expected_cols = [
        "Source_ID", "Separation(arcsec)",
        "Parallax(mas)", "Parallax_err(mas)",
        "Distance_lower(pc)", "Distance(pc)", "Distance_higher(pc)",
        "T_eff_lower(K)", "T_eff(K)", "T_eff_higher(K)",
        "G_Mag", "G_BP_Mag", "G_RP_Mag",
        "Radial_velocity(km/s)", "Radial_velocity_err(km/s)",
    ]
    assert list(df.columns) == expected_cols


def test_target_star_distance_columns_map_to_the_right_gaia_fields():
    """
    Regression test: Distance(pc) used to hold distance_gspphot_upper and
    Distance_higher(pc) held distance_gspphot.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _, df = _run_target_star(tmpdir, _make_mock_gaia_result())
    for _, row in df.iterrows():
        base = row["Source_ID"] * 10
        assert row["Distance_lower(pc)"] == base + 1.0
        assert row["Distance(pc)"] == base + 2.0
        assert row["Distance_higher(pc)"] == base + 3.0


def test_target_star_rows_are_sorted_by_separation():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, df = _run_target_star(tmpdir, _make_mock_gaia_result())
    assert list(df["Source_ID"]) == [10, 20, 30, 40]
    assert df["Separation(arcsec)"].is_monotonic_increasing
    assert df["Separation(arcsec)"].iloc[0] == pytest.approx(0.2, abs=0.01)


def test_target_star_drops_sources_outside_match_radius():
    rows = [(10, 0.5, 0.0, 0.0, 0.0), (99, 0.0, 12.0, 0.0, 0.0)]
    with tempfile.TemporaryDirectory() as tmpdir:
        _, df = _run_target_star(tmpdir, _make_mock_gaia_result(rows))
    assert list(df["Source_ID"]) == [10]


def test_target_star_matches_fast_moving_star_at_j2000():
    # A star 8 arcsec north of the J2000 position in 2016 because it moves
    # 500 mas/yr north. Moved back 16 years it sits right on the target.
    rows = [(10, 0.0, 8.0, 0.0, 500.0), (20, 0.0, 4.5, 0.0, 0.0)]
    with tempfile.TemporaryDirectory() as tmpdir:
        _, df = _run_target_star(tmpdir, _make_mock_gaia_result(rows))
    assert list(df["Source_ID"]) == [10, 20]
    assert df["Separation(arcsec)"].iloc[0] == pytest.approx(0.0, abs=0.01)


def test_target_star_query_has_no_hardcoded_astrometry():
    # The old query propagated every target with one star's parallax and
    # proper motion (4.7516, 48.8840, -24.1470)
    with tempfile.TemporaryDirectory() as tmpdir:
        query, _ = _run_target_star(tmpdir, _make_mock_gaia_result())
    adql = query.call_args[0][0]
    assert "EPOCH_PROP_POS" not in adql
    assert "48.884" not in adql
    assert "188.73658" in adql


def test_target_star_logs_when_nothing_matches():
    rows = [(99, 0.0, 20.0, 0.0, 0.0)]
    messages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        _, df = _run_target_star(tmpdir, _make_mock_gaia_result(rows), messages)
    assert df.empty
    assert any("No Gaia DR3 source" in m for m in messages)


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
# tess_mag — network mocked via _query_tic_catalog
#
# tess_mag calls _query_tic_catalog → _query_single_tic_region → Vizier.
# GaiaData is never touched by tess_mag, so patching GaiaData.from_query
# has no effect and leaves real network calls in place. The correct boundary
# to mock is _query_tic_catalog.
# ===========================================================================

def _make_mock_tic_df(ra_deg=12.534, dec_deg=78.961, tmag=11.234, tmag_err=0.012):
    """
    Return a minimal TIC DataFrame as _query_tic_catalog would produce,
    with coordinates close enough to the default test inputs to match
    within the 3 arcsec tolerance used by _match_tic_magnitudes.

    Default inputs to tess_mag are ra=[0.8356] hours, dec=[78.961] deg.
    0.8356 hours * 15 = 12.534 degrees → ra_deg=12.534 matches exactly.
    """
    return pd.DataFrame({
        "_RAJ2000": [ra_deg],
        "_DEJ2000": [dec_deg],
        "Tmag":    [tmag],
        "e_Tmag":  [tmag_err],
    })


def test_tess_mag_returns_two_lists():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    with patch("EclipsingBinaries.gaia._query_tic_catalog",
               return_value=_make_mock_tic_df()):
        result = tess_mag([0.8356], [78.961], None, cancel)

    assert result is not None
    assert len(result) == 2


def test_tess_mag_lists_same_length():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    mock_df = pd.DataFrame({
        "_RAJ2000": [12.534, 13.5],
        "_DEJ2000": [78.961, 79.0],
        "Tmag":    [11.234, 12.345],
        "e_Tmag":  [0.012,  0.015],
    })
    with patch("EclipsingBinaries.gaia._query_tic_catalog",
               return_value=mock_df):
        T_list, T_err_list = tess_mag([0.8356, 0.9], [78.961, 79.0], None, cancel)

    assert len(T_list) == len(T_err_list)


def test_tess_mag_cancels_before_loop():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []

    with patch("EclipsingBinaries.gaia._query_tic_catalog") as mock_tic:
        result = tess_mag([0.8356], [78.961], messages.append, cancel)

    mock_tic.assert_not_called()
    assert result is None
    assert any("cancel" in m.lower() for m in messages)


def test_tess_mag_logs_completion():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with patch("EclipsingBinaries.gaia._query_tic_catalog",
               return_value=_make_mock_tic_df()):
        tess_mag([0.8356], [78.961], messages.append, cancel)

    assert any("Finished" in m for m in messages)


def test_tess_mag_returns_fallback_for_no_match():
    """Stars with no TIC match and no Gaia match either get 99.999 for both values."""
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    # TIC entry placed far from the input coordinate
    far_df = pd.DataFrame({
        "_RAJ2000": [90.0],
        "_DEJ2000": [0.0],
        "Tmag":    [11.0],
        "e_Tmag":  [0.01],
    })
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=far_df), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([])):
        T_list, T_err_list = tess_mag([0.8356], [78.961], None, cancel)

    assert T_list[0] == 99.999
    assert T_err_list[0] == 99.999


def test_tess_mag_returns_fallback_for_empty_tic():
    """Empty TIC and Gaia results should produce 99.999 for every star."""
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    empty_df = pd.DataFrame(columns=["_RAJ2000", "_DEJ2000", "Tmag", "e_Tmag"])
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=empty_df), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([])):
        T_list, T_err_list = tess_mag([0.8356], [78.961], None, cancel)

    assert T_list == [99.999]
    assert T_err_list == [99.999]


def test_tess_mag_magnitude_precision():
    """Returned Tmag values should be rounded to 3 decimal places."""
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False

    with patch("EclipsingBinaries.gaia._query_tic_catalog",
               return_value=_make_mock_tic_df(tmag=11.23456, tmag_err=0.01234)):
        T_list, T_err_list = tess_mag([0.8356], [78.961], None, cancel)

    assert T_list[0] == pytest.approx(11.235, abs=1e-3)
    assert T_err_list[0] == pytest.approx(0.012, abs=1e-3)


def test_tess_mag_logs_error_on_exception():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with patch("EclipsingBinaries.gaia._query_tic_catalog",
               side_effect=RuntimeError("timeout")):
        with pytest.raises(RuntimeError):
            tess_mag([0.8356], [78.961], messages.append, cancel)

    assert any("error" in m.lower() for m in messages)

# ===========================================================================
# tess_mag — Gaia DR3 as the backup for stars the TIC doesn't cover
# ===========================================================================

class FakeGaia:
    """
    Stands in for the GaiaData the archive query returns. Each source is a
    dict with ra and dec in degrees (at Gaia's 2016 epoch) and optionally
    pmra, pmdec, G, BP, RP and their flux over error values.
    """

    def __init__(self, sources):
        import astropy.units as u

        def column(key, default=np.nan):
            return np.array([src.get(key, default) for src in sources], dtype=float)

        self._n = len(sources)
        self.ra = column("ra") * u.deg
        self.dec = column("dec") * u.deg
        self.pmra = column("pmra", 0.0) * u.mas / u.yr
        self.pmdec = column("pmdec", 0.0) * u.mas / u.yr
        self.phot_g_mean_mag = column("G") * u.mag
        self.phot_bp_mean_mag = column("BP") * u.mag
        self.phot_rp_mean_mag = column("RP") * u.mag
        self.phot_g_mean_flux_over_error = column("G_snr", 1000.0)
        self.phot_bp_mean_flux_over_error = column("BP_snr", 500.0)
        self.phot_rp_mean_flux_over_error = column("RP_snr", 500.0)

    def __len__(self):
        return self._n


# The test star sits at 0.8356 h, +78.961 deg (12.534 deg in RA)
STAR = {"ra": 12.534, "dec": 78.961, "G": 12.0, "BP": 12.4, "RP": 11.6}


def _no_tic():
    return pd.DataFrame(columns=["_RAJ2000", "_DEJ2000", "Tmag", "e_Tmag"])


def test_tess_mag_uses_gaia_when_the_tic_has_no_magnitude():
    from EclipsingBinaries.gaia import tess_mag, gaia_tess_magnitude
    messages = []
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([STAR])):
        T_list, T_err_list = tess_mag([0.8356], [78.961], messages.append, None)
    assert T_list[0] == pytest.approx(_tic_polynomial(12.0, 0.8), abs=1e-3)
    assert T_err_list[0] == gaia_tess_magnitude(12.0, 0.8, _flux_to_mag_err(1000.0),
                                                math.hypot(_flux_to_mag_err(500.0),
                                                           _flux_to_mag_err(500.0)))[1]
    assert any("worked out from Gaia DR3" in m for m in messages)
    assert any("Gaia DR3 gave TESS magnitudes for 1 of those 1 stars" in m for m in messages)


def test_tess_mag_only_asks_gaia_about_stars_the_tic_missed():
    from EclipsingBinaries.gaia import tess_mag
    tic = pd.DataFrame({"_RAJ2000": [13.5], "_DEJ2000": [79.0], "Tmag": [10.5], "e_Tmag": [0.007]})
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=tic), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([STAR])) as gaia:
        T_list, _ = tess_mag([0.8356, 0.9], [78.961, 79.0], None, None)
    assert gaia.call_count == 1
    assert T_list[1] == 10.5                        # the TIC value is kept
    assert T_list[0] == pytest.approx(_tic_polynomial(12.0, 0.8), abs=1e-3)


def test_tess_mag_does_not_ask_gaia_when_the_tic_covers_every_star():
    from EclipsingBinaries.gaia import tess_mag
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_make_mock_tic_df()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive") as gaia:
        tess_mag([0.8356], [78.961], None, None)
    gaia.assert_not_called()


def test_gaia_backup_moves_sources_back_to_j2000():
    from EclipsingBinaries.gaia import tess_mag
    # A star moving 300 mas/yr north sits 4.8 arcsec north of its J2000 position in 2016,
    # past the 3 arcsec match, until its proper motion is taken off
    moved = dict(STAR, dec=78.961 + 4.8 / 3600, pmdec=300.0)
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([moved])):
        T_list, _ = tess_mag([0.8356], [78.961], None, None)
    assert T_list[0] != 99.999


def test_gaia_backup_takes_the_nearest_source():
    from EclipsingBinaries.gaia import tess_mag
    near = dict(STAR, dec=78.961 + 0.5 / 3600)
    farther = dict(STAR, dec=78.961 + 2.5 / 3600, G=9.0)
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([farther, near])):
        T_list, _ = tess_mag([0.8356], [78.961], None, None)
    assert T_list[0] == pytest.approx(_tic_polynomial(12.0, 0.8), abs=1e-3)


def test_gaia_backup_without_bp_rp_uses_the_g_offset():
    from EclipsingBinaries.gaia import tess_mag
    no_colour = {"ra": 12.534, "dec": 78.961, "G": 13.0}
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([no_colour])):
        T_list, T_err_list = tess_mag([0.8356], [78.961], None, None)
    assert T_list[0] == pytest.approx(12.570, abs=1e-3)
    assert T_err_list[0] == pytest.approx(0.6, abs=1e-3)


def test_gaia_backup_query_covers_every_star():
    from EclipsingBinaries.gaia import tess_mag
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaia([])) as gaia:
        # Two stars about 17.7 arcminutes apart (0.02 h at +10 deg) across RA 0h
        tess_mag([23.99, 0.01], [10.0, 10.0], None, None)
    adql = gaia.call_args[0][0]
    ra, dec, radius = [float(v) for v in
                       adql.split("CIRCLE('ICRS',")[1].split(")")[0].split(",")]
    assert dec == pytest.approx(10.0, abs=0.01)
    assert min(ra, 360 - ra) < 0.01            # centred on 0h, not on 12h
    assert 0.155 < radius < 0.16               # half the gap plus 30 arcsec for proper motion
    assert "phot_g_mean_mag < 17" in adql


def test_tess_mag_keeps_99999_when_gaia_is_down_too():
    from EclipsingBinaries.gaia import tess_mag
    messages = []
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive",
                  side_effect=OSError("The Gaia archive at ESA didn't answer.")):
        T_list, T_err_list = tess_mag([0.8356], [78.961], messages.append, None)
    assert T_list == [99.999] and T_err_list == [99.999]
    assert any("didn't answer" in m and "keep 99.999" in m for m in messages)


def test_tess_mag_cancel_during_the_gaia_backup():
    from EclipsingBinaries.gaia import tess_mag
    cancel = MagicMock()
    cancel.is_set.side_effect = [False, False, True]
    with patch("EclipsingBinaries.gaia._query_tic_catalog", return_value=_no_tic()), \
            patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=None):
        assert tess_mag([0.8356], [78.961], None, cancel) is None
