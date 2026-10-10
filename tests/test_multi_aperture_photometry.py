# -*- coding: utf-8 -*-
"""
Tests for multi_aperture_photometry.py

"""

import pytest
import numpy as np
import pandas as pd


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
    import pytest

    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []

    with pytest.raises(Exception):
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
    h1 = {"FILTER": "B"}
    h2 = {"FILTER": "V"}
    h3 = {"FILTER": "R"}
    mock_collection.headers.return_value = [(h1, "img1.fits"), (h2, "img2.fits"), (h3, "img3.fits")]

    with patch("EclipsingBinaries.multi_aperture_photometry.ccdp.ImageFileCollection",
               return_value=mock_collection), \
         patch("EclipsingBinaries.multi_aperture_photometry.multiple_AP") as mock_ap:
        main(path="/tmp", pipeline=True,
             radec_list=["b.radec", "v.radec", "r.radec"],
             obj_name="test", write_callback=messages.append, cancel_event=cancel)

    assert mock_ap.call_count == 3


def test_main_passes_correct_filter_to_multiple_ap():
    from EclipsingBinaries.multi_aperture_photometry import main
    from unittest.mock import MagicMock, patch

    cancel = MagicMock()
    cancel.is_set.return_value = False

    mock_collection = MagicMock()
    h1 = {"FILTER": "B"}
    h2 = {"FILTER": "V"}
    h3 = {"FILTER": "R"}
    mock_collection.headers.return_value = [(h1, "img1.fits"), (h2, "img2.fits"), (h3, "img3.fits")]

    with patch("EclipsingBinaries.multi_aperture_photometry.ccdp.ImageFileCollection",
               return_value=mock_collection), \
         patch("EclipsingBinaries.multi_aperture_photometry.multiple_AP") as mock_ap:
        main(path="/tmp", pipeline=True,
             radec_list=["b.radec", "v.radec", "r.radec"],
             obj_name="test", write_callback=None, cancel_event=cancel)

    filters_used = [c.kwargs["filt"] for c in mock_ap.call_args_list]
    assert "B" in filters_used
    assert "V" in filters_used
    assert "R" in filters_used


# ===========================================================================
# multiple_AP — cancel handling (mocked)
# ===========================================================================
def test_multiple_ap_cancels_during_loop():
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP
    from unittest.mock import MagicMock, patch
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


# ===========================================================================
# New Interactive Radii Optimization Tests
# ===========================================================================
def test_calculate_target_snr():
    from EclipsingBinaries.multi_aperture_photometry import calculate_target_snr
    from astropy.modeling import models

    yy, xx = np.mgrid[:100, :100]
    g = models.Gaussian2D(amplitude=1000, x_mean=50, y_mean=50, x_stddev=2.0, y_stddev=2.0)
    image = g(xx, yy) + 100 # Add background

    snr, flx, err, bkg = calculate_target_snr(image, (50, 50), 10, (15, 20), read_noise=5.0)

    assert snr > 0
    assert flx > 0
    assert err > 0
    assert bkg == pytest.approx(100.0, rel=0.1)

def test_auto_optimize_radii():
    from EclipsingBinaries.multi_aperture_photometry import auto_optimize_radii
    from astropy.modeling import models

    yy, xx = np.mgrid[:100, :100]
    g = models.Gaussian2D(amplitude=1000, x_mean=50, y_mean=50, x_stddev=3.0, y_stddev=3.0)
    image = g(xx, yy) + 100

    fwhm, best_ap, (best_in, best_out) = auto_optimize_radii(image, (50, 50))

    assert fwhm > 0
    assert best_ap > 0
    assert best_in > best_ap
    assert best_out > best_in


# ===========================================================================
# multiple_AP on synthetic frames shaped like this package's reduction output
# ===========================================================================
def _write_synthetic_frames(folder, n_frames=2, rdnoise=6.0, time_keys=("HJD", "BJD_TDB")):
    """
    Write small FITS frames with a WCS, a target and two comparison stars, plus
    a matching RADEC file. Headers carry HJD and BJD_TDB the way the package's
    header correction writes them, not AstroImageJ's HJD-OBS and BJD-OBS.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    size = 200
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2, size / 2]
    wcs.wcs.crval = [15.0, 30.0]          # RA 1h, Dec +30
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]  # 1 arcsec pixels
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    stars = [(100, 100, 20000.0), (60, 100, 30000.0), (140, 100, 25000.0)]  # target first
    yy, xx = np.mgrid[:size, :size]
    names = []
    for i in range(n_frames):
        image = np.full((size, size), 100.0)
        for x, y, amp in stars:
            image += amp * np.exp(-((xx - x + 1) ** 2 + (yy - y + 1) ** 2) / (2 * 2.0 ** 2))
        header = wcs.to_header()
        header["IMAGETYP"] = "LIGHT"
        header["FILTER"] = "V"
        header["RDNOISE"] = rdnoise
        if "HJD" in time_keys:
            header["HJD"] = 2460000.5 + i * 0.01
        if "BJD_TDB" in time_keys:
            header["BJD_TDB"] = 2460000.6 + i * 0.01
        name = f"frame{i}.fits"
        fits.writeto(folder / name, image, header, overwrite=True)
        names.append(name)

    sky = wcs.pixel_to_world([s[0] - 1 for s in stars], [s[1] - 1 for s in stars])
    lines = ["#RA in decimal or sexagesimal HOURS"] + ["#"] * 6
    for coord, mag in zip(sky, [99.999, 11.0, 11.2]):
        lines.append(f"{coord.ra.hour:.8f}, {coord.dec.deg:.8f}, 0, 1, {mag}")
    radec = folder / "V.radec"
    radec.write_text("\n".join(lines) + "\n")
    return names, radec


def test_multiple_ap_reads_package_time_keywords(tmp_path):
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP

    names, radec = _write_synthetic_frames(tmp_path)
    multiple_AP(names, tmp_path, "V", obj_name="T", radec_file=str(radec),
                write_callback=lambda _m: None, aperture_radius=6, annulus_radii=(10, 15))

    out = pd.read_csv(tmp_path / "T_V_data.csv")
    assert list(out["HJD"]) == pytest.approx([2460000.5, 2460000.51])
    assert list(out["BJD"]) == pytest.approx([2460000.6, 2460000.61])
    assert out["Source_AMag_T1"].notna().all()


def test_multiple_ap_uses_read_noise_from_header(tmp_path):
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP

    errors = {}
    for rdnoise in (2.0, 60.0):
        folder = tmp_path / str(rdnoise)
        folder.mkdir()
        names, radec = _write_synthetic_frames(folder, n_frames=1, rdnoise=rdnoise)
        multiple_AP(names, folder, "V", obj_name="T", radec_file=str(radec),
                    write_callback=lambda _m: None, aperture_radius=6, annulus_radii=(10, 15))
        errors[rdnoise] = pd.read_csv(folder / "T_V_data.csv")["Source_AMag_T1_Error"].iloc[0]

    # With the old fixed 10.83 e- both runs gave the same error
    assert errors[60.0] > errors[2.0]


def test_read_noise_from_header_falls_back():
    from EclipsingBinaries.multi_aperture_photometry import read_noise_from_header
    assert read_noise_from_header({"RDNOISE": 6.3}, 10.83) == 6.3
    assert read_noise_from_header({}, 9.7) == 9.7
    assert read_noise_from_header({"RDNOISE": "n/a"}, 9.7) == 9.7


def test_multiple_ap_missing_times_become_nan_with_warning(tmp_path):
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP

    names, radec = _write_synthetic_frames(tmp_path, time_keys=())
    messages = []
    multiple_AP(names, tmp_path, "V", obj_name="T", radec_file=str(radec),
                write_callback=messages.append, aperture_radius=6, annulus_radii=(10, 15))

    out = pd.read_csv(tmp_path / "T_V_data.csv")
    assert out["HJD"].isna().all()
    assert sum("no HJD or BJD keyword" in m for m in messages) == 1


def test_multiple_ap_errors_are_raised(tmp_path):
    # Used to be logged and swallowed, so the GUI and pipeline reported success
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP

    with pytest.raises(Exception):
        multiple_AP(["missing.fits"], tmp_path, "V", obj_name="T",
                    radec_file=str(tmp_path / "missing.radec"), write_callback=lambda _m: None)


# ===========================================================================
# Comparison stars that run off the image
# ===========================================================================
SIZE = 200
OUTER = 15  # outer annulus radius used in these tests


def _drifting_frames(folder, stars, shifts, nan_box=None, nan_frame=None):
    """
    Write frames whose pointing drifts, so stars move across the detector.

    stars: list of (x, y, flux, catalog_mag) in frame-0 pixels, target first.
    shifts: per-frame (dx, dy) applied to every star's pixel position.
    nan_box: (x0, x1, y0, y1) filled with NaN in frame nan_frame.
    Returns the frame names and a RADEC writer that takes a list of star indices.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    base = WCS(naxis=2)
    base.wcs.crpix = [SIZE / 2, SIZE / 2]
    base.wcs.crval = [15.0, 30.0]
    base.wcs.cdelt = [-1 / 3600, 1 / 3600]
    base.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    sky = base.pixel_to_world([s[0] for s in stars], [s[1] for s in stars])

    yy, xx = np.mgrid[:SIZE, :SIZE]
    names = []
    for i, (dx, dy) in enumerate(shifts):
        wcs = base.deepcopy()
        wcs.wcs.crpix = [SIZE / 2 + dx + 1, SIZE / 2 + dy + 1]  # FITS crpix is 1-based
        image = np.full((SIZE, SIZE), 100.0)
        for x, y, flux, _ in stars:
            px, py = x + dx, y + dy
            image += flux * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.0 ** 2))
        if nan_box is not None and i == nan_frame:
            x0, x1, y0, y1 = nan_box
            image[y0:y1, x0:x1] = np.nan
        header = wcs.to_header()
        header.update(IMAGETYP="LIGHT", FILTER="V", RDNOISE=6.0,
                      HJD=2460000.5 + i * 0.01, BJD_TDB=2460000.6 + i * 0.01)
        name = f"frame{i}.fits"
        fits.writeto(folder / name, image, header, overwrite=True)
        names.append(name)

    def write_radec(path, keep):
        lines = ["#"] * 7
        for k in keep:
            mag = 99.999 if k == 0 else stars[k][3]
            lines.append(f"{sky[k].ra.hour:.10f}, {sky[k].dec.deg:.10f}, 0, 1, {mag}")
        path.write_text("\n".join(lines) + "\n")
        return path

    return names, write_radec


def _run(folder, names, radec, messages=None):
    from EclipsingBinaries.multi_aperture_photometry import multiple_AP
    log = messages.append if messages is not None else (lambda _m: None)
    multiple_AP(names, folder, "V", obj_name="T", radec_file=str(radec),
                write_callback=log, aperture_radius=6, annulus_radii=(10, OUTER))
    return pd.read_csv(folder / "T_V_data.csv")


# Target in the middle; A, B and C are comparison stars. C sits near the right edge.
STARS = [(100, 100, 20000.0, 99.999), (60, 100, 30000.0, 11.0),
         (100, 60, 25000.0, 11.2), (170, 140, 28000.0, 11.1)]


def test_off_image_star_is_dropped_and_analysis_starts_over(tmp_path):
    # Frame 2 drifts 20 px right, which pushes C's annulus off the edge.
    # The finished light curve must be exactly what you'd get by leaving C
    # out of the RADEC file from the start: every frame uses A and B only.
    names, write_radec = _drifting_frames(tmp_path, STARS, [(0, 0), (5, 0), (20, 0), (0, 0)])
    messages = []
    restarted = _run(tmp_path, names, write_radec(tmp_path / "all.radec", [0, 1, 2, 3]), messages)
    fresh = _run(tmp_path, names, write_radec(tmp_path / "no_c.radec", [0, 1, 2]))

    assert len(restarted) == 4
    pd.testing.assert_frame_equal(restarted, fresh)
    assert sum("[REMOVED] Comparison star #3" in m for m in messages) == 1
    assert sum("Starting V photometry over" in m for m in messages) == 1
    assert any("2 comparison star(s) (#1, #2)" in m for m in messages)


def test_all_stars_off_in_one_image_are_dropped_in_one_restart(tmp_path):
    # Frame 1 drifts up and left so both A (x=60) and B (y=60) hang off the edges
    stars = [(100, 100, 20000.0, 99.999), (60, 100, 30000.0, 11.0),
             (100, 60, 25000.0, 11.2), (120, 130, 28000.0, 11.1), (130, 110, 26000.0, 11.3)]
    names, write_radec = _drifting_frames(tmp_path, stars, [(0, 0), (-50, -50), (0, 0)])
    messages = []
    _run(tmp_path, names, write_radec(tmp_path / "s.radec", range(5)), messages)

    removed = [m for m in messages if m.startswith("[REMOVED]")]
    assert len(removed) == 2
    assert sum("Starting V photometry over" in m for m in messages) == 1


def test_annulus_poking_half_a_pixel_off_is_caught(tmp_path):
    # C's outer annulus reaches 0.2 px past the right edge (x + 15 = 199.7 > 199.5).
    # The old check only required x < width - radius, so it kept this star.
    stars = STARS[:3] + [(184.7, 140, 28000.0, 11.1)]
    names, write_radec = _drifting_frames(tmp_path, stars, [(0, 0), (0, 0)])
    messages = []
    _run(tmp_path, names, write_radec(tmp_path / "s.radec", range(4)), messages)
    assert any("[REMOVED] Comparison star #3" in m and "off the edge" in m for m in messages)


def test_star_over_missing_data_is_dropped(tmp_path):
    # A NaN patch in frame 1 overlaps A's annulus but stays well inside the image
    names, write_radec = _drifting_frames(tmp_path, STARS, [(0, 0), (0, 0), (0, 0)],
                                          nan_box=(68, 80, 95, 105), nan_frame=1)
    messages = []
    out = _run(tmp_path, names, write_radec(tmp_path / "s.radec", range(4)), messages)
    assert any("[REMOVED] Comparison star #1" in m and "no data" in m for m in messages)
    assert out["Source_AMag_T1"].notna().all()


def test_frame_where_target_is_off_is_skipped_not_used_to_drop_stars(tmp_path):
    # In frame 1 the pointing jumps so far that the target itself is off the image.
    # That frame is skipped; the comparison stars aren't judged on it.
    names, write_radec = _drifting_frames(tmp_path, STARS, [(0, 0), (95, 0), (0, 0)])
    messages = []
    out = _run(tmp_path, names, write_radec(tmp_path / "s.radec", range(4)), messages)

    assert len(out) == 2
    assert list(out["HJD"]) == pytest.approx([2460000.5, 2460000.52])
    assert any(m.startswith("[SKIPPED] frame1.fits") for m in messages)
    assert not any(m.startswith("[REMOVED]") for m in messages)


def test_error_when_every_comparison_star_is_dropped(tmp_path):
    stars = [(100, 100, 20000.0, 99.999), (190, 100, 30000.0, 11.0)]
    names, write_radec = _drifting_frames(tmp_path, stars, [(0, 0)])
    with pytest.raises(RuntimeError, match="All comparison stars have been removed"):
        _run(tmp_path, names, write_radec(tmp_path / "s.radec", [0, 1]))


def test_error_when_target_never_fits(tmp_path):
    stars = [(195, 100, 20000.0, 99.999), (60, 100, 30000.0, 11.0)]
    names, write_radec = _drifting_frames(tmp_path, stars, [(0, 0), (0, 0)])
    with pytest.raises(RuntimeError, match="couldn't be measured in any"):
        _run(tmp_path, names, write_radec(tmp_path / "s.radec", [0, 1]))


@pytest.mark.parametrize("x, y, expected", [
    (100.0, 100.0, None),
    (14.5, 100.0, None),            # touches the left edge exactly
    (14.4, 100.0, "off the edge"),
    (184.5, 100.0, None),           # touches the right edge exactly
    (184.6, 100.0, "off the edge"),
    (100.0, 184.6, "off the edge"),
    (float("nan"), 100.0, "WCS"),
])
def test_footprint_problem_edges(x, y, expected):
    from EclipsingBinaries.multi_aperture_photometry import footprint_problem
    result = footprint_problem(x, y, OUTER, np.zeros((SIZE, SIZE)))
    if expected is None:
        assert result is None
    else:
        assert expected in result


def test_footprint_problem_nan_inside_and_outside():
    from EclipsingBinaries.multi_aperture_photometry import footprint_problem
    image = np.zeros((SIZE, SIZE))
    image[100, 114] = np.nan                       # inside a radius-15 circle around (100, 100)
    assert "no data" in footprint_problem(100.0, 100.0, OUTER, image)
    image = np.zeros((SIZE, SIZE))
    image[100, 120] = np.inf                       # well outside it
    assert footprint_problem(100.0, 100.0, OUTER, image) is None
