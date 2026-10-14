# -*- coding: utf-8 -*-
"""
Tests for the header corrections and the EB_HCorr command.
"""

import numpy as np
import pytest
from astropy import coordinates as coords
from astropy.io import fits
from astropy.time import Time

from EclipsingBinaries import headerCorrect as hc

QUIET = {"log": lambda _msg: None}


def _write(path, **keys):
    header = fits.Header()
    defaults = {"OBSERVAT": "BSUO", "RA": "00:28:27.97", "DEC": "+78:57:42.66", "EXPTIME": 0.0}
    defaults.update(keys)
    for key, value in defaults.items():
        if value is not None:
            header[key] = value
    fits.PrimaryHDU(np.zeros((1, 1), dtype=np.float32), header=header).writeto(path, overwrite=True)
    return path


def _correct(path, **opts):
    report = hc.correct_headers(path, hc.HeaderCorrectionOptions(**opts), **QUIET)
    return report, fits.getheader(path)


# --- the console command ------------------------------------------------------------

def test_console_script_target_exists():
    """pyproject's EB_HCorr entry has to point at a real function."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    module, func = project["project"]["scripts"]["EB_HCorr"].split(":")
    assert module == "EclipsingBinaries.headerCorrect"
    assert callable(getattr(hc, func))


def test_command_corrects_a_folder(tmp_path, capsys):
    _write(tmp_path / "a.fits", **{"DATE-OBS": "2018-10-09T02:06:11"})
    _write(tmp_path / "b.FIT", **{"DATE-OBS": "2018-10-09T02:16:11"})
    (tmp_path / "notes.txt").write_text("not an image")

    assert hc.header_correct([str(tmp_path), "-q"]) == 0
    for name in ("a.fits", "b.FIT"):
        header = fits.getheader(tmp_path / name)
        assert "BJD_TDB" in header and "HJD" in header and "EAIRMASS" in header
    assert "Corrected 2 of 2 file(s) cleanly." in capsys.readouterr().out


def test_command_honours_switches(tmp_path):
    path = _write(tmp_path / "a.fits", **{"DATE-OBS": "2018-10-09T02:06:11"})
    assert hc.header_correct([str(path), "-q", "--no-bjd", "--no-eairmass"]) == 0
    header = fits.getheader(path)
    assert "HJD" in header
    assert "BJD_TDB" not in header and "EAIRMASS" not in header


def test_command_recursive(tmp_path):
    (tmp_path / "night1").mkdir()
    path = _write(tmp_path / "night1" / "a.fits", **{"DATE-OBS": "2018-10-09T02:06:11"})
    assert hc.header_correct([str(tmp_path), "-q"]) == 2  # nothing at the top level
    assert hc.header_correct([str(tmp_path), "-q", "-r"]) == 0
    assert "BJD_TDB" in fits.getheader(path)


def test_command_reports_warnings_with_exit_code(tmp_path, capsys):
    _write(tmp_path / "a.fits")  # no DATE-OBS at all
    assert hc.header_correct([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "DATE-OBS missing" in out
    assert "1 had warnings" in out


def test_command_rejects_bad_arguments(tmp_path, capsys):
    assert hc.header_correct([str(tmp_path / "missing.fits")]) == 2
    assert hc.header_correct([str(tmp_path), "--observatory", "Nowhere"]) == 2
    assert "Unknown observatory" in capsys.readouterr().err


def test_command_uses_chosen_fallback_site(tmp_path):
    path = _write(tmp_path / "a.fits", OBSERVAT=None, **{"DATE-OBS": "2018-10-09T02:06:11"})
    assert hc.header_correct([str(path), "-q", "--observatory", "KPNO"]) == 1  # missing OBSERVAT warns
    assert fits.getheader(path)["OBSERVAT"] == "KPNO"


# --- DATE-OBS -----------------------------------------------------------------------

def test_date_only_date_obs_uses_time_obs(tmp_path):
    split = _write(tmp_path / "split.fits", **{"DATE-OBS": "2018-10-09", "TIME-OBS": "02:06:11"})
    joined = _write(tmp_path / "joined.fits", **{"DATE-OBS": "2018-10-09T02:06:11"})
    report, h_split = _correct(split)
    _, h_joined = _correct(joined)
    assert not report.warnings
    assert h_split["JD"] == pytest.approx(h_joined["JD"], abs=1e-9)
    assert h_split["BJD_TDB"] == pytest.approx(h_joined["BJD_TDB"], abs=1e-9)


@pytest.mark.parametrize("key", ["UT", "UTSTART"])
def test_date_only_date_obs_uses_other_time_keys(tmp_path, key):
    path = _write(tmp_path / "a.fits", **{"DATE-OBS": "2018-10-09", key: "02:06:11"})
    _, header = _correct(path)
    assert header["JD"] == pytest.approx(Time("2018-10-09T02:06:11").jd, abs=1e-9)


def test_date_only_date_obs_without_time_warns(tmp_path):
    path = _write(tmp_path / "a.fits", **{"DATE-OBS": "2018-10-09"})
    report, header = _correct(path, do_eairmass=False)
    assert any("no time of day" in w for w in report.warnings)
    assert header["JD"] == pytest.approx(Time("2018-10-09T00:00:00").jd)


def test_date_obs_with_a_space(tmp_path):
    path = _write(tmp_path / "a.fits", **{"DATE-OBS": "2018-10-09 02:06:11"})
    report, header = _correct(path)
    assert header["JD"] == pytest.approx(Time("2018-10-09T02:06:11").jd, abs=1e-9)


def test_unreadable_date_obs_is_a_warning(tmp_path):
    path = _write(tmp_path / "a.fits", **{"DATE-OBS": "last tuesday"})
    report, _ = _correct(path)
    assert any("DATE-OBS couldn't be read" in w for w in report.warnings)


def test_string_exposure_time(tmp_path):
    path = _write(tmp_path / "a.fits", EXPTIME="60", **{"DATE-OBS": "2018-10-09T02:06:11"})
    _, header = _correct(path)
    assert (header["JD_END"] - header["JD_START"]) * 86400 == pytest.approx(60, abs=1e-3)


# --- coordinate frames ----------------------------------------------------------------

@pytest.mark.parametrize("equinox", [2000.0, "2000", "J2000", "J2000.0"])
def test_j2000_equinox_spellings_are_icrs(equinox):
    assert hc._coordinate_frame(equinox) == "icrs"


def test_other_equinoxes():
    assert isinstance(hc._coordinate_frame(1950.0), coords.FK4)
    assert isinstance(hc._coordinate_frame("B1950"), coords.FK4)
    frame = hc._coordinate_frame("J2015.5")
    assert isinstance(frame, coords.FK5)
    assert frame.equinox.jyear == pytest.approx(2015.5)
    with pytest.raises(ValueError):
        hc._coordinate_frame("today")


def test_epoch_of_observation_no_longer_drops_bjd(tmp_path):
    """Cameras that put the observation date in EPOCH used to lose HJD and BJD."""
    plain = _write(tmp_path / "plain.fits", **{"DATE-OBS": "2018-10-09T02:06:11"})
    dated = _write(tmp_path / "dated.fits", EPOCH=2018.77, EQUINOX=2000.0,
                   **{"DATE-OBS": "2018-10-09T02:06:11"})
    _, h_plain = _correct(plain)
    report, h_dated = _correct(dated)
    assert not report.warnings
    assert h_dated["BJD_TDB"] == pytest.approx(h_plain["BJD_TDB"], abs=1e-9)


# --- re-runs and impossible geometry -------------------------------------------------

def test_secz_org_keeps_the_camera_value(tmp_path):
    path = _write(tmp_path / "a.fits", SECZ=9.99, **{"DATE-OBS": "2018-10-09T02:06:11"})
    _correct(path)
    _correct(path)
    header = fits.getheader(path)
    assert header["SECZ_ORG"] == 9.99
    assert header["SECZ"] != 9.99


def test_target_below_horizon_leaves_airmass_alone(tmp_path):
    # A far-southern target from Indiana is never up
    path = _write(tmp_path / "a.fits", RA="06:00:00", DEC="-80:00:00",
                  **{"DATE-OBS": "2018-10-09T02:06:11"})
    report, header = _correct(path)
    assert any("below the horizon" in w for w in report.warnings)
    assert "EAIRMASS" not in header
    assert "BJD_TDB" in header


def test_observatory_lookup_ignores_padding():
    assert hc.DEFAULT_REGISTRY.get("  bsuo ") is not None


def test_find_fits_files(tmp_path):
    for name in ("b.fits", "a.FTS", "c.fit", "d.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in hc.find_fits_files([tmp_path])] == ["a.FTS", "b.fits", "c.fit"]
    with pytest.raises(FileNotFoundError):
        hc.find_fits_files([tmp_path / "nope"])
