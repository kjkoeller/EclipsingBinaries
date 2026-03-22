# -*- coding: utf-8 -*-
"""
Tests for multi_aperture_photometry.py

"""

import pytest
import numpy as np
import math
import os
import tempfile


# ===========================================================================
# Pure math helpers — extracted from multiple_AP
# ===========================================================================

def _target_magnitude(magnitudes_comp, target_flx, comparison_flx):
    ensemble_flux = sum(2.512 ** -m for m in magnitudes_comp)
    return (-np.log(ensemble_flux) / np.log(2.512)) - (
        2.5 * np.log10(target_flx / sum(comparison_flx))
    )


def _target_magnitude_error(target_flux_err, target_flx, comp_flux_err, comparison_flx):
    return 2.5 * np.log10(
        1 + np.sqrt(
            (target_flux_err ** 2 / target_flx ** 2) +
            (sum(comp_flux_err ** 2) / sum(comparison_flx) ** 2)
        )
    )


def _background_subtracted_flux(aperture_sum, bkg_mean, aperture_area):
    return aperture_sum - bkg_mean * aperture_area


def _flux_error(aperture_sum, aperture_area, read_noise=10.83):
    return np.sqrt(aperture_sum + aperture_area * read_noise ** 2)


def _rel_flux_comps(comparison_flx):
    result = []
    for i, flx in enumerate(comparison_flx):
        total_others = sum(f for j, f in enumerate(comparison_flx) if j != i)
        result.append(flx / total_others)
    return result


# ===========================================================================
# target magnitude
# ===========================================================================
def test_target_magnitude_equal_flux_gives_catalog():
    mags = [12.0, 12.0]
    target_flx = 2.0
    comp_flx = [1.0, 1.0]
    result = _target_magnitude(mags, target_flx, comp_flx)
    expected_ensemble = -np.log(sum(2.512 ** -m for m in mags)) / np.log(2.512)
    assert result == pytest.approx(expected_ensemble, abs=1e-8)


def test_target_magnitude_brighter_target_gives_lower_mag():
    mags = [14.0, 14.0]
    comp_flx = [1000.0, 1000.0]
    mag_faint = _target_magnitude(mags, 500.0, comp_flx)
    mag_bright = _target_magnitude(mags, 5000.0, comp_flx)
    assert mag_bright < mag_faint


def test_target_magnitude_returns_float():
    assert isinstance(_target_magnitude([14.0, 14.5], 1000.0, [900.0, 950.0]), float)


def test_target_magnitude_single_comp_star():
    mags = [13.0]
    result = _target_magnitude(mags, 500.0, [500.0])
    expected = -np.log(2.512 ** -13.0) / np.log(2.512)
    assert result == pytest.approx(expected, abs=1e-8)


def test_target_magnitude_scales_with_flux_ratio():
    mags = [14.0, 14.0]
    comp_flx = [1000.0, 1000.0]
    mag1 = _target_magnitude(mags, 1000.0, comp_flx)
    mag2 = _target_magnitude(mags, 4000.0, comp_flx)
    assert mag1 - mag2 == pytest.approx(2.5 * np.log10(4), abs=1e-6)


def test_target_magnitude_multiple_comps_finite():
    result = _target_magnitude([13.0, 13.5, 14.0], 700.0, [800.0, 600.0, 400.0])
    assert np.isfinite(result)


# ===========================================================================
# target magnitude error
# ===========================================================================
def test_magnitude_error_zero_flux_error_gives_zero():
    result = _target_magnitude_error(0.0, 1000.0, np.array([0.0, 0.0]), [500.0, 500.0])
    assert result == pytest.approx(0.0, abs=1e-10)


def test_magnitude_error_positive():
    result = _target_magnitude_error(50.0, 1000.0, np.array([30.0, 30.0]), [900.0, 850.0])
    assert result > 0


def test_magnitude_error_larger_noise_gives_larger_error():
    err_low = _target_magnitude_error(10.0, 1000.0, np.array([10.0, 10.0]), [900.0, 850.0])
    err_high = _target_magnitude_error(100.0, 1000.0, np.array([100.0, 100.0]), [900.0, 850.0])
    assert err_high > err_low


def test_magnitude_error_returns_float():
    assert isinstance(_target_magnitude_error(50.0, 1000.0, np.array([30.0, 30.0]), [900.0, 850.0]), float)


def test_magnitude_error_finite():
    result = _target_magnitude_error(25.0, 2000.0, np.array([20.0, 15.0, 18.0]), [800.0, 700.0, 600.0])
    assert np.isfinite(result)


# ===========================================================================
# background subtracted flux
# ===========================================================================
def test_background_flux_no_background():
    assert _background_subtracted_flux(1000.0, 0.0, 100.0) == pytest.approx(1000.0)


def test_background_flux_subtracts_correctly():
    # 1000 - 2.0 * 50 = 900
    assert _background_subtracted_flux(1000.0, 2.0, 50.0) == pytest.approx(900.0)


def test_background_flux_large_background_can_go_negative():
    assert _background_subtracted_flux(100.0, 10.0, 50.0) == pytest.approx(-400.0)


def test_background_flux_scales_with_area():
    area1 = _background_subtracted_flux(1000.0, 1.0, 100.0)
    area2 = _background_subtracted_flux(1000.0, 1.0, 200.0)
    assert area1 > area2


def test_background_flux_zero_aperture_sum():
    assert _background_subtracted_flux(0.0, 5.0, 20.0) == pytest.approx(-100.0)


# ===========================================================================
# flux error
# ===========================================================================
def test_flux_error_positive():
    assert _flux_error(1000.0, np.pi * 20 ** 2) > 0


def test_flux_error_increases_with_counts():
    assert _flux_error(10000.0, 100.0) > _flux_error(100.0, 100.0)


def test_flux_error_increases_with_area():
    assert _flux_error(1000.0, 1000.0) > _flux_error(1000.0, 100.0)


def test_flux_error_zero_counts_read_noise_only():
    result = _flux_error(0.0, 100.0, read_noise=10.83)
    assert result == pytest.approx(np.sqrt(100.0 * 10.83 ** 2))


def test_flux_error_known_value():
    # sqrt(100 + 4 * 100) = sqrt(500)
    assert _flux_error(100.0, 4.0, read_noise=10.0) == pytest.approx(np.sqrt(500.0), abs=1e-8)


def test_flux_error_default_read_noise():
    assert _flux_error(0.0, 1.0) == pytest.approx(10.83, abs=1e-6)


# ===========================================================================
# relative flux of comparison stars
# ===========================================================================
def test_rel_flux_comps_two_equal_stars():
    result = _rel_flux_comps([500.0, 500.0])
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0)


def test_rel_flux_comps_length_matches_input():
    assert len(_rel_flux_comps([100.0, 200.0, 300.0])) == 3


def test_rel_flux_comps_each_excluded_from_own_denominator():
    comps = [100.0, 200.0, 300.0]
    result = _rel_flux_comps(comps)
    assert result[0] == pytest.approx(100.0 / 500.0, abs=1e-10)
    assert result[1] == pytest.approx(200.0 / 400.0, abs=1e-10)
    assert result[2] == pytest.approx(300.0 / 300.0, abs=1e-10)


def test_rel_flux_comps_single_star_raises():
    with pytest.raises(ZeroDivisionError):
        _rel_flux_comps([500.0])


def test_rel_flux_comps_all_positive():
    assert all(r > 0 for r in _rel_flux_comps([300.0, 400.0, 500.0, 600.0]))


def test_rel_flux_comps_dominant_star_highest():
    comps = [10.0, 10.0, 1000.0]
    result = _rel_flux_comps(comps)
    assert result[2] == max(result)


# ===========================================================================
# ensemble magnitude formula internals
# ===========================================================================
def test_ensemble_flux_conversion_finite():
    mags = [13.0, 14.0, 15.0]
    ensemble = sum(2.512 ** -m for m in mags)
    assert np.isfinite(-np.log(ensemble) / np.log(2.512))


def test_ensemble_dominated_by_brightest():
    mags_bright = [10.0, 14.0, 15.0]
    mags_faint = [14.0, 14.0, 15.0]
    ens_bright = sum(2.512 ** -m for m in mags_bright)
    ens_faint = sum(2.512 ** -m for m in mags_faint)
    assert (-np.log(ens_bright) / np.log(2.512)) < (-np.log(ens_faint) / np.log(2.512))


def test_differential_term_zero_when_equal_flux():
    assert 2.5 * np.log10(1000.0 / 1000.0) == pytest.approx(0.0, abs=1e-10)


def test_differential_term_positive_when_target_brighter():
    assert 2.5 * np.log10(2000.0 / 1000.0) > 0


# ===========================================================================
# aperture geometry (photutils)
# ===========================================================================
def test_circular_aperture_area():
    from photutils.aperture import CircularAperture
    ap = CircularAperture((100, 100), r=20)
    assert ap.area == pytest.approx(np.pi * 20 ** 2, abs=1e-6)


def test_circular_annulus_area():
    from photutils.aperture import CircularAnnulus
    ann = CircularAnnulus((100, 100), r_in=30, r_out=50)
    assert ann.area == pytest.approx(np.pi * (50 ** 2 - 30 ** 2), abs=1e-6)


def test_annulus_area_larger_than_aperture():
    from photutils.aperture import CircularAperture, CircularAnnulus
    ap = CircularAperture((100, 100), r=20)
    ann = CircularAnnulus((100, 100), r_in=30, r_out=50)
    assert ann.area > ap.area


def test_aperture_position_stored_correctly():
    from photutils.aperture import CircularAperture
    ap = CircularAperture((123.4, 567.8), r=20)
    assert ap.positions[0] == pytest.approx(123.4)
    assert ap.positions[1] == pytest.approx(567.8)


def test_annulus_inner_less_than_outer():
    from photutils.aperture import CircularAnnulus
    ann = CircularAnnulus((100, 100), r_in=30, r_out=50)
    assert ann.r_in < ann.r_out


# ===========================================================================
# main — cancel and error handling (mocked)
# ===========================================================================
def test_main_cancels_before_processing():
    from EclipsingBinaries.multi_aperture_photometry import main
    from unittest.mock import MagicMock, patch

    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []

    with patch("EclipsingBinaries.multi_aperture_photometry.ccdp.ImageFileCollection"):
        main(path="/tmp", pipeline=True, radec_list=["a.radec", "b.radec", "c.radec"],
             obj_name="test", write_callback=messages.append, cancel_event=cancel)

    assert any("cancel" in m.lower() for m in messages)


def test_main_logs_error_on_bad_path():
    from EclipsingBinaries.multi_aperture_photometry import main
    from unittest.mock import MagicMock

    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    main(path="/nonexistent/path/xyz", pipeline=True,
         radec_list=["a.radec", "b.radec", "c.radec"],
         obj_name="test", write_callback=messages.append, cancel_event=cancel)

    assert any("error" in m.lower() for m in messages)


def test_main_processes_each_filter():
    from EclipsingBinaries.multi_aperture_photometry import main
    from unittest.mock import MagicMock, patch

    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    mock_collection = MagicMock()
    mock_collection.files_filtered.return_value = []

    with patch("EclipsingBinaries.multi_aperture_photometry.ccdp.ImageFileCollection",
               return_value=mock_collection), \
         patch("EclipsingBinaries.multi_aperture_photometry.multiple_AP") as mock_ap:
        main(path="/tmp", pipeline=True,
             radec_list=["b.radec", "v.radec", "r.radec"],
             obj_name="test", write_callback=messages.append, cancel_event=cancel)

    assert mock_ap.call_count == 3


def test_main_passes_correct_filter_to_multiple_ap():
    from EclipsingBinaries.multi_aperture_photometry import main
    from unittest.mock import MagicMock, patch, call

    cancel = MagicMock()
    cancel.is_set.return_value = False

    mock_collection = MagicMock()
    mock_collection.files_filtered.return_value = []

    with patch("EclipsingBinaries.multi_aperture_photometry.ccdp.ImageFileCollection",
               return_value=mock_collection), \
         patch("EclipsingBinaries.multi_aperture_photometry.multiple_AP") as mock_ap:
        main(path="/tmp", pipeline=True,
             radec_list=["b.radec", "v.radec", "r.radec"],
             obj_name="test", write_callback=None, cancel_event=cancel)

    filters_used = [c.kwargs["filt"] for c in mock_ap.call_args_list]
    assert "Empty/B" in filters_used
    assert "Empty/V" in filters_used
    assert "Empty/R" in filters_used


# ===========================================================================
# multiple_AP — cancel handling (mocked)
# ===========================================================================
def test_multiple_ap_cancels_during_loop():
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP
    from unittest.mock import MagicMock, patch
    import tempfile
    import pandas as pd

    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []

    mock_df = MagicMock()
    mock_df.__getitem__ = lambda self, key: {
        0: pd.Series(["12:00:00"]),
        1: pd.Series([45.0]),
        4: pd.Series([14.0]),
    }[key]
    mock_df.replace.return_value = mock_df
    mock_df.dropna.return_value = mock_df
    mock_df.reset_index.return_value = pd.Series([14.0])

    with patch("EclipsingBinaries.multi_aperture_photometry.pd.read_csv", return_value=mock_df):
        multiple_AP(
            image_list=["img1.fits", "img2.fits"],
            path=MagicMock(),
            filt="Empty/B",
            radec_file="test.radec",
            write_callback=messages.append,
            cancel_event=cancel
        )

    assert any("cancel" in m.lower() for m in messages)
