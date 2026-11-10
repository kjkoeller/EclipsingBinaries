# -*- coding: utf-8 -*-
"""
Tests for archives.py, the multi-wavelength archive download.

Nothing here touches the network. Each service's answer is made up: VizieR
through the fake servers from test_vizier, ZTF through requests.get, and the
astroquery services (Gaia, MAST, IRSA, SDSS, SIMBAD) by patching the calls.
"""

import os
import threading

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from unittest.mock import patch

from EclipsingBinaries import _vizier, archives
from EclipsingBinaries.archives import ArchiveError, Run, Target, download, resolve_target
from tests.test_vizier import EMPTY, GOOD, OUTAGE, FakeServers

TARGET = Target("SW Lac", SkyCoord(343.4236 * u.deg, 37.9385 * u.deg))


@pytest.fixture(autouse=True)
def mast_catalogues_down(monkeypatch):
    """
    GALEX and Pan-STARRS go to MAST first. Unless a test says otherwise, MAST
    fails here, so the tests written for VizieR still exercise VizieR.
    """
    def down(*args, **kwargs):
        raise requests.exceptions.ConnectionError("MAST is down in this test")
    monkeypatch.setattr("astroquery.mast.Catalogs.query_region", down)


def _run(messages=None, cancel=None):
    return Run(log=(messages.append if messages is not None else lambda m: None), cancel_event=cancel)


@pytest.fixture
def vizier(monkeypatch):
    """Make every VizieR server send back the same answer, and record the calls."""
    def install(answer):
        fake = FakeServers({server: answer for server in _vizier.VIZIER_SERVERS})
        monkeypatch.setattr(type(_vizier.Vizier), "query_region_async",
                            lambda self, *a, **k: fake(self, *a, **k))
        return fake
    return install


def _archive(key):
    return next(a for a in archives.ARCHIVES if a.key == key)


def _fetch(key, folder, run=None, target=TARGET):
    return _archive(key).fetch(target, folder, run or _run(), key)


# --- the list of databases ---------------------------------------------------------------

def test_every_database_has_its_own_key_and_a_known_kind():
    keys = [a.key for a in archives.ARCHIVES]
    assert len(keys) == len(set(keys))
    assert {a.kind for a in archives.ARCHIVES} == set(archives.KINDS)
    for archive in archives.ARCHIVES:
        assert archive.name and archive.band and archive.about and callable(archive.fetch)


def test_keys_can_be_listed_by_kind():
    light_curves = archives.archive_keys([archives.LIGHT_CURVES])
    assert "ztf" in light_curves and "galex" not in light_curves
    assert set(archives.archive_keys()) == {a.key for a in archives.ARCHIVES}


# --- the target --------------------------------------------------------------------------------

def test_target_from_ra_and_dec_in_hours_and_degrees():
    target = resolve_target("SW Lac", "22:53:41.66", "+37:56:18.6")
    assert target.coord.ra.deg == pytest.approx(343.42358, abs=1e-4)
    assert target.coord.dec.deg == pytest.approx(37.93850, abs=1e-4)
    assert target.name == "SW Lac"


def test_target_without_a_name_is_labelled_by_its_position():
    target = resolve_target("", "22:53:41.66", "+37:56:18.6")
    assert target.label.startswith("225341") and "+375618" in target.label


def test_target_name_is_looked_up_when_there_are_no_coordinates():
    with patch.object(archives.SkyCoord, "from_name", return_value=TARGET.coord) as lookup:
        target = resolve_target("SW Lac")
    lookup.assert_called_once()
    assert target.coord.ra.deg == pytest.approx(343.4236)


def test_target_name_that_cannot_be_looked_up():
    from astropy.coordinates.name_resolve import NameResolveError
    with patch.object(archives.SkyCoord, "from_name", side_effect=NameResolveError("unknown")):
        with pytest.raises(ArchiveError, match="enter the RA and Dec instead"):
            resolve_target("Not A Star")


@pytest.mark.parametrize("name, ra, dec, message", [
    ("", "", "", "Enter a target name"),
    ("X", "22:53:41", "", "both RA and Dec"),
    ("X", "abc", "def", "couldn't be read"),
])
def test_target_mistakes_are_explained(name, ra, dec, message):
    with pytest.raises(ArchiveError, match=message):
        resolve_target(name, ra, dec)


def test_target_label_is_safe_for_a_folder_name():
    assert Target("V* SW Lac / test", TARGET.coord).label == "V_SW_Lac_test"
    assert Target("///", TARGET.coord).label == "target"


# --- saving light curves ----------------------------------------------------------------------------

def test_light_curve_files_are_sorted_and_skip_rows_with_gaps(tmp_path):
    path = tmp_path / "lc.txt"
    written = archives.save_light_curve(path, [3.0, 1.0, np.nan, 2.0], [10.3, 10.1, 10.0, np.nan],
                                        [0.01, 0.02, 0.01, 0.01])
    assert written == 2
    assert np.loadtxt(path)[:, 0].tolist() == [1.0, 3.0]


def test_light_curve_files_open_in_find_minimum(tmp_path):
    from EclipsingBinaries import find_min
    path = tmp_path / "lc.txt"
    archives.save_light_curve(path, 2458000 + np.arange(5.0), 12 + np.arange(5.0) / 10, np.full(5, 0.01))
    time, mag, err = find_min.read_light_curve(path)
    assert time[0] == 2458000 and mag[-1] == pytest.approx(12.4) and err[0] == pytest.approx(0.01)


def test_no_file_is_written_for_an_empty_light_curve(tmp_path):
    path = tmp_path / "lc.txt"
    assert archives.save_light_curve(path, [np.nan], [np.nan], [np.nan]) == 0
    assert not path.exists()


# --- catalogues on VizieR ------------------------------------------------------------------------------

def test_vizier_catalogue_is_saved_as_csv(tmp_path, vizier):
    fake = vizier(GOOD)
    status, rows, files, note = _fetch("apass", tmp_path)
    assert (status, rows) == ("found", 2)
    assert [f.name for f in files] == ["apass.csv"]
    assert list(pd.read_csv(files[0])["Vmag"]) == [12.5, 13.1]
    assert "5 arcsec" in note
    assert fake.calls[0][1] is False       # fetched fresh, not from astroquery's cache


def test_vizier_catalogue_with_nothing_near_the_target(tmp_path, vizier):
    vizier(EMPTY)
    status, rows, files, note = _fetch("twomass", tmp_path)
    assert (status, rows, files) == ("nothing", 0, [])
    assert "No source within 3 arcsec" in note
    assert list(tmp_path.iterdir()) == []


def test_several_tables_from_one_catalogue_get_their_own_files(tmp_path):
    tables = {"IX/69/xmm4d13s": Table({"a": [1, 2]}), "IX/69/summary": Table({"b": [3]}),
              "IX/69/empty": Table({"c": []})}
    with patch.object(archives, "query_region_tables", return_value=tables):
        status, rows, files, _note = _fetch("xmm", tmp_path)
    assert (status, rows) == ("found", 3)
    assert sorted(f.name for f in files) == ["xmm_summary.csv", "xmm_xmm4d13s.csv"]


def test_x_ray_catalogues_search_wider_than_optical_ones(tmp_path):
    seen = {}

    def record(catalog, coordinates, **kwargs):
        seen[catalog] = kwargs["radius"].to_value(u.arcsec)
        return {}
    with patch.object(archives, "query_region_tables", side_effect=record):
        _fetch("rosat", tmp_path)
        _fetch("gaia", tmp_path)
    assert seen == {"J/A+A/588/A103/cat2rxs": 30, "I/355/gaiadr3": 3}


def test_vizier_servers_that_do_not_respond_are_not_waited_on_again(tmp_path, monkeypatch):
    fake = FakeServers({server: requests.exceptions.ReadTimeout("read timed out")
                        for server in _vizier.VIZIER_SERVERS})
    monkeypatch.setattr(type(_vizier.Vizier), "query_region_async", lambda self, *a, **k: fake(self, *a, **k))
    results = download(TARGET, tmp_path, ["galex", "twomass", "allwise"])
    assert [r.status for r in results] == ["failed", "skipped", "skipped"]
    assert len(fake.calls) == len(_vizier.VIZIER_SERVERS)     # each server was only waited on once
    assert "try again later" in results[1].note


def test_a_mirror_that_answered_is_tried_first_for_the_next_catalogue(tmp_path, monkeypatch):
    answers = {server: GOOD for server in _vizier.VIZIER_SERVERS}
    answers["vizier.cds.unistra.fr"] = requests.exceptions.ReadTimeout("read timed out")
    fake = FakeServers(answers)
    monkeypatch.setattr(type(_vizier.Vizier), "query_region_async", lambda self, *a, **k: fake(self, *a, **k))
    results = download(TARGET, tmp_path, ["galex", "twomass", "allwise"])
    assert [r.status for r in results] == ["found", "found", "found"]
    assert [call[0] for call in fake.calls] == ["vizier.cds.unistra.fr"] + ["vizier.idia.ac.za"] * 3


def test_one_catalogue_failing_everywhere_does_not_stop_the_next(tmp_path, monkeypatch):
    # What happened for GALEX on 9 November: CDS timed out, the South African mirror
    # gave up on that one query, and the others timed out or refused the connection
    state = {"galex_done": False}

    def answer(vizier, *args, **kwargs):
        server = vizier.VIZIER_SERVER
        if server == "vizier.idia.ac.za":
            if not state["galex_done"]:
                state["galex_done"] = True
                return _vizier_reply(b'<INFO name="Error" value="(pipe) Request has been interrupted"/>')
            return _vizier_reply(GOOD)
        raise requests.exceptions.ReadTimeout("read timed out")
    monkeypatch.setattr(type(_vizier.Vizier), "query_region_async", answer)
    results = download(TARGET, tmp_path, ["galex", "twomass"])
    assert [r.status for r in results] == ["failed", "found"]
    assert "Request has been interrupted" in results[0].note


def _vizier_reply(content):
    if not content.startswith(b"<?xml"):
        content = b'<?xml version="1.0"?><VOTABLE version="1.4">' + content + b"</VOTABLE>"
    response = requests.Response()
    response._content = content
    response.status_code = 200
    return response


def test_failure_notes_are_not_cut_short(tmp_path, monkeypatch):
    fake = FakeServers({server: OUTAGE for server in _vizier.VIZIER_SERVERS})
    monkeypatch.setattr(type(_vizier.Vizier), "query_region_async", lambda self, *a, **k: fake(self, *a, **k))
    results = download(TARGET, tmp_path, ["galex"])
    assert results[0].note.endswith("try again later.")


# --- ZTF --------------------------------------------------------------------------------------------

ZTF_CSV = """oid,expid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec
1,10,2458300.5,58300.0,12.31,0.012,0,zg,343.4236,37.9385
1,11,2458301.5,58301.0,12.80,0.015,0,zg,343.4236,37.9385
2,12,2458300.6,58300.1,11.90,0.010,0,zr,343.4236,37.9385
"""


def _response(text, status=200):
    response = requests.Response()
    response._content = text.encode()
    response.status_code = status
    return response


def test_ztf_saves_everything_and_one_light_curve_per_filter(tmp_path):
    with patch.object(archives.requests, "get", return_value=_response(ZTF_CSV)) as get:
        status, rows, files, note = _fetch("ztf", tmp_path)
    assert (status, rows) == ("found", 3)
    assert sorted(f.name for f in files) == ["ztf_all.csv", "ztf_g.txt", "ztf_r.txt"]
    g = np.loadtxt(tmp_path / "ztf_g.txt")
    assert g[:, 0].tolist() == [2458300.5, 2458301.5] and g[:, 1].tolist() == [12.31, 12.80]
    assert "HJD" in note
    params = get.call_args.kwargs["params"]
    assert params["POS"].startswith("CIRCLE 343.423600 37.938500 0.000833")
    assert params["FORMAT"] == "CSV" and get.call_args.kwargs["timeout"] == archives.TIMEOUT


def test_ztf_warns_when_the_star_is_near_saturation(tmp_path):
    # NSVS 896797 is about V = 10.6, and ZTF kept only 8 points for it
    bright = ZTF_CSV.replace(",12.31,", ",10.61,").replace(",12.80,", ",10.70,").replace(",11.90,", ",10.40,")
    with patch.object(archives.requests, "get", return_value=_response(bright)):
        _status, _rows, _files, note = _fetch("ztf", tmp_path)
    assert "about magnitude 10.6 in ZTF, near where ZTF saturates (13)" in note


def test_ztf_does_not_warn_for_fainter_stars(tmp_path):
    faint = ZTF_CSV.replace(",12.31,", ",14.31,").replace(",12.80,", ",14.80,").replace(",11.90,", ",13.90,")
    with patch.object(archives.requests, "get", return_value=_response(faint)):
        _status, _rows, _files, note = _fetch("ztf", tmp_path)
    assert "saturates" not in note


def test_ztf_with_no_object_at_the_position(tmp_path):
    header_only = ZTF_CSV.splitlines()[0] + "\n"
    with patch.object(archives.requests, "get", return_value=_response(header_only)):
        status, _rows, files, _note = _fetch("ztf", tmp_path)
    assert status == "nothing" and files == []


def test_ztf_without_hjd_falls_back_to_mjd(tmp_path):
    frame = pd.read_csv(pd.io.common.StringIO(ZTF_CSV)).drop(columns="hjd")
    with patch.object(archives.requests, "get", return_value=_response(frame.to_csv(index=False))):
        _status, _rows, _files, note = _fetch("ztf", tmp_path)
    assert np.loadtxt(tmp_path / "ztf_r.txt").ravel()[0] == pytest.approx(2458300.6)
    assert "not heliocentric" in note


def test_ztf_server_error_is_reported_and_the_rest_carry_on(tmp_path, vizier):
    vizier(GOOD)
    with patch.object(archives.requests, "get", return_value=_response("oops", 503)):
        results = download(TARGET, tmp_path, ["ztf", "apass"])
    assert [r.status for r in results] == ["failed", "found"]
    assert "503" in results[0].note


# --- Gaia ---------------------------------------------------------------------------------------------

class FakeGaiaSource:
    """Stands in for the GaiaData the source lookup returns."""

    def __init__(self, sources):
        def column(key, default):
            return [src.get(key, default) for src in sources]
        self.data = Table({"source_id": column("source_id", 0),
                           "has_epoch_photometry": column("epoch", True),
                           "has_xp_sampled": column("xp", True),
                           "has_rvs": column("rvs", False)})
        self.ra = np.array(column("ra", 343.4236)) * u.deg
        self.dec = np.array(column("dec", 37.9385)) * u.deg
        self.pmra = np.array(column("pmra", 0.0)) * u.mas / u.yr
        self.pmdec = np.array(column("pmdec", 0.0)) * u.mas / u.yr

    def __len__(self):
        return len(self.data)


EPOCHS = Table({"source_id": [7] * 4, "band": ["G", "G", "BP", "G"],
                "time": [1700.25, 1701.25, 1700.25, 1702.25],
                "mag": [12.0, 12.5, 12.4, 12.1],
                "flux_over_error": [1000.0, 500.0, 200.0, 900.0],
                "rejected_by_photometry": ["false", "false", "false", "true"]})


def test_gaia_epoch_photometry_becomes_light_curves(tmp_path):
    with patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaiaSource([{"source_id": 7}])), \
            patch.object(archives, "_gaia_products", return_value=[EPOCHS]) as products:
        status, rows, files, note = _fetch("gaia_lc", tmp_path)
    assert (status, rows) == ("found", 4)
    assert sorted(f.name for f in files) == ["gaia_lc_BP.txt", "gaia_lc_G.txt", "gaia_lc_all.csv"]
    g = np.loadtxt(tmp_path / "gaia_lc_G.txt")
    assert g[:, 0].tolist() == [2455197.5 + 1700.25, 2455197.5 + 1701.25]     # the rejected point is left out
    assert g[0, 2] == pytest.approx(2.5 / np.log(10) / 1000, abs=1e-4)
    assert products.call_args[0][:2] == (7, "EPOCH_PHOTOMETRY")
    assert "Gaia DR3 7" in note and "BJD" in note


def test_gaia_source_without_epoch_photometry(tmp_path):
    with patch("EclipsingBinaries.gaia._query_gaia_archive",
               return_value=FakeGaiaSource([{"source_id": 7, "epoch": False}])), \
            patch.object(archives, "_gaia_products") as products:
        status, _rows, _files, note = _fetch("gaia_lc", tmp_path)
    assert status == "nothing" and "no epoch photometry" in note
    products.assert_not_called()


def test_no_gaia_source_at_the_position(tmp_path):
    with patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaiaSource([])):
        assert _fetch("gaia_lc", tmp_path)[0] == "nothing"
        assert _fetch("gaia_spectra", tmp_path)[0] == "nothing"


def test_gaia_takes_the_nearest_source():
    near = {"source_id": 1, "dec": 37.9385 + 0.5 / 3600}
    far = {"source_id": 2, "dec": 37.9385 + 4.0 / 3600}
    with patch("EclipsingBinaries.gaia._query_gaia_archive", return_value=FakeGaiaSource([far, near])):
        assert archives._gaia_source(TARGET, _run())["source_id"] == 1


def test_gaia_source_is_only_looked_up_once_per_run(tmp_path):
    run = _run()
    spectrum = Table({"wavelength": [336.0, 338.0], "flux": [1e-15, 2e-15], "flux_error": [1e-17, 1e-17]})
    with patch("EclipsingBinaries.gaia._query_gaia_archive",
               return_value=FakeGaiaSource([{"source_id": 7}])) as lookup, \
            patch.object(archives, "_gaia_products", side_effect=[[EPOCHS], [spectrum]]):
        _fetch("gaia_lc", tmp_path, run)
        status, _rows, files, note = _fetch("gaia_spectra", tmp_path, run)
    assert lookup.call_count == 1
    assert status == "found" and [f.name for f in files] == ["gaia_spectra_xp.csv"]
    assert "No RVS spectrum" in note


def test_gaia_products_are_asked_for_as_votable_first(tmp_path):
    with patch("astroquery.gaia.Gaia.load_data", return_value={"x": [EPOCHS]}) as load:
        archives._gaia_products(7, "EPOCH_PHOTOMETRY", tmp_path)
    assert load.call_count == 1 and load.call_args.kwargs["format"] == "votable"


def test_gaia_products_fall_back_to_csv(tmp_path):
    error = requests.exceptions.HTTPError("Error 500:")
    with patch("astroquery.gaia.Gaia.load_data", side_effect=[error, {"x": [EPOCHS]}]) as load:
        tables = archives._gaia_products(7, "EPOCH_PHOTOMETRY", tmp_path)
    assert [c.kwargs["format"] for c in load.call_args_list] == ["votable", "csv"]
    assert len(tables[0]) == 4


def test_gaia_data_service_errors_are_explained(tmp_path):
    error = requests.exceptions.HTTPError("Error 500:")
    with patch("astroquery.gaia.Gaia.load_data", side_effect=error):
        with pytest.raises(OSError, match="Gaia's data service gave an error for RVS .Error 500:.*try again later"):
            archives._gaia_products(7, "RVS", tmp_path)


def test_gaia_download_unpacks_inside_the_output_folder(tmp_path):
    seen = {}

    def load_data(**kwargs):
        seen["cwd"] = os.getcwd()
        seen["kwargs"] = kwargs
        return {"EPOCH_PHOTOMETRY-Gaia DR3 7.csv": [EPOCHS]}
    before = os.getcwd()
    with patch("astroquery.gaia.Gaia.load_data", side_effect=load_data):
        tables = archives._gaia_products(7, "EPOCH_PHOTOMETRY", tmp_path)
    assert os.path.samefile(seen["cwd"], tmp_path)
    assert os.getcwd() == before
    assert seen["kwargs"]["ids"] == [7] and seen["kwargs"]["retrieval_type"] == "EPOCH_PHOTOMETRY"
    assert len(tables) == 1 and len(tables[0]) == 4


# --- TESS, Kepler and K2 from MAST ---------------------------------------------------------------------

def _light_curve_fits(path, flux_name="PDCSAP_FLUX", tessmag=10.5):
    time = np.array([1.0, 2.0, 3.0, 4.0])
    flux = np.array([1000.0, 500.0, 1000.0, 1000.0])
    columns = [fits.Column(name="TIME", format="D", array=time),
               fits.Column(name=flux_name, format="E", array=flux),
               fits.Column(name=flux_name + "_ERR", format="E", array=np.full(4, 1.0)),
               fits.Column(name="QUALITY", format="J", array=np.array([0, 0, 0, 8]))]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["BJDREFI"], table.header["BJDREFF"] = 2457000, 0.0
    primary = fits.PrimaryHDU()
    if tessmag is not None:
        primary.header["TESSMAG"] = tessmag
    fits.HDUList([primary, table]).writeto(path)
    return path


def test_flux_light_curve_becomes_magnitudes(tmp_path):
    out = tmp_path / "lc.txt"
    assert archives._flux_light_curve(out, _light_curve_fits(tmp_path / "a_lc.fits")) == 3
    data = np.loadtxt(out)
    assert data[:, 0].tolist() == [2457001.0, 2457002.0, 2457003.0]      # the flagged point is left out
    assert data[0, 1] == pytest.approx(10.5) and data[1, 1] == pytest.approx(10.5 + 2.5 * np.log10(2), abs=1e-4)
    assert data[0, 2] == pytest.approx(2.5 / np.log(10) / 1000, abs=1e-4)


def test_flux_light_curve_without_a_catalogue_magnitude_is_relative(tmp_path):
    out = tmp_path / "lc.txt"
    archives._flux_light_curve(out, _light_curve_fits(tmp_path / "a_llc.fits", "KSPSAP_FLUX", None))
    assert np.loadtxt(out)[0, 1] == pytest.approx(0.0)


def test_mast_downloads_and_converts_light_curves(tmp_path):
    observations = Table({"obsid": ["1", "2", "3"], "provenance_name": ["SPOC", "QLP", "ELEANOR"]})
    products = Table({"productFilename": ["tess-s0014_lc.fits", "tess-s0014_tp.fits", "hlsp_qlp_llc.fits"]})

    def download_products(chosen, download_dir, flat):
        rows = []
        for name in chosen["productFilename"]:
            rows.append((str(_light_curve_fits(os.path.join(download_dir, name))), "COMPLETE"))
        return Table(rows=rows, names=["Local Path", "Status"])

    with patch("astroquery.mast.Observations.query_criteria", return_value=observations) as query, \
            patch("astroquery.mast.Observations.get_product_list", return_value=products) as listing, \
            patch("astroquery.mast.Observations.download_products", side_effect=download_products):
        status, rows, files, note = _fetch("mast", tmp_path)
    assert status == "found" and rows == 6
    assert list(listing.call_args[0][0]["provenance_name"]) == ["SPOC", "QLP"]     # other pipelines left out
    assert sorted(f.name for f in files) == ["hlsp_qlp_llc.fits", "mast_hlsp_qlp_llc.txt",
                                            "mast_tess-s0014_lc.txt", "tess-s0014_lc.fits"]
    assert query.call_args.kwargs["dataproduct_type"] == "timeseries"
    assert "BJD" in note


def test_mast_with_no_light_curves(tmp_path):
    empty = Table({"obsid": [], "provenance_name": []})
    with patch("astroquery.mast.Observations.query_criteria", return_value=empty):
        assert _fetch("mast", tmp_path)[0] == "nothing"


# --- NEOWISE, SDSS spectra and SIMBAD ------------------------------------------------------------------------

def test_neowise_becomes_w1_and_w2_light_curves(tmp_path):
    table = Table({"mjd": [57000.0, 57000.1], "w1mpro": [9.5, 9.9], "w1sigmpro": [0.02, 0.03],
                   "w2mpro": [9.4, np.nan], "w2sigmpro": [0.03, np.nan]})
    with patch("astroquery.ipac.irsa.Irsa.query_region", return_value=table) as query:
        status, rows, files, note = _fetch("neowise", tmp_path)
    assert (status, rows) == ("found", 2)
    assert sorted(f.name for f in files) == ["neowise_W1.txt", "neowise_W2.txt", "neowise_all.csv"]
    assert np.loadtxt(tmp_path / "neowise_W1.txt")[:, 0].tolist() == [2457000.5, 2457000.6]
    assert np.loadtxt(tmp_path / "neowise_W2.txt").ndim == 1           # the point with no W2 is left out
    assert query.call_args.kwargs["catalog"] == "neowiser_p1bs_psd"
    assert "not heliocentric" in note


def test_sdss_spectra_are_saved_as_fits(tmp_path):
    matches = Table({"plate": [1234], "mjd": [55000], "fiberID": [56]})
    spectrum = fits.HDUList([fits.PrimaryHDU(np.arange(4.0))])
    with patch("astroquery.sdss.SDSS.query_region", return_value=matches), \
            patch("astroquery.sdss.SDSS.get_spectra", return_value=[spectrum]):
        status, rows, files, _note = _fetch("sdss_spectra", tmp_path)
    assert (status, rows) == ("found", 1)
    assert sorted(f.name for f in files) == ["sdss_spectra_1234_55000_56.fits", "sdss_spectra_list.csv"]
    assert fits.getdata(tmp_path / "sdss_spectra_1234_55000_56.fits").tolist() == [0.0, 1.0, 2.0, 3.0]


def test_sdss_with_no_spectrum(tmp_path):
    with patch("astroquery.sdss.SDSS.query_region", return_value=None):
        assert _fetch("sdss_spectra", tmp_path)[0] == "nothing"


def test_simbad_is_saved(tmp_path):
    table = Table({"main_id": ["V* SW Lac"], "otype": ["EB*"]})
    with patch("astroquery.simbad.SimbadClass.query_region", return_value=table):
        status, rows, files, _note = _fetch("simbad", tmp_path)
    assert (status, rows) == ("found", 1) and pd.read_csv(files[0])["otype"][0] == "EB*"


def test_simbad_uses_the_harvard_mirror_when_cds_is_down(tmp_path):
    table = Table({"main_id": ["V* SW Lac"], "otype": ["EB*"]})
    servers = []

    def query(self, *args, **kwargs):
        servers.append(self.server)
        if self.server == "simbad.cds.unistra.fr":
            raise Exception("Unable to access the capabilities endpoint at:\n- https://...")
        return table
    messages = []
    with patch("astroquery.simbad.SimbadClass.query_region", query):
        status, _rows, _files, _note = _fetch("simbad", tmp_path, _run(messages))
    assert status == "found"
    assert servers == ["simbad.cds.unistra.fr", "simbad.harvard.edu"]
    assert messages == ["SIMBAD at simbad.cds.unistra.fr didn't answer "
                        "(Unable to access the capabilities endpoint at:)."]


def test_simbad_down_everywhere(tmp_path):
    with patch("astroquery.simbad.SimbadClass.query_region", side_effect=Exception("no capabilities")):
        with pytest.raises(OSError, match="SIMBAD couldn't be reached .simbad.cds.unistra.fr: no capabilities; "
                                          "simbad.harvard.edu: no capabilities.. Try again later."):
            _fetch("simbad", tmp_path)


# --- a whole download -----------------------------------------------------------------------------------------

def test_download_puts_everything_in_the_targets_folder_with_a_summary(tmp_path, vizier):
    vizier(GOOD)
    messages = []
    with patch.object(archives.requests, "get", return_value=_response(ZTF_CSV)):
        results = download(TARGET, tmp_path, ["ztf", "apass"], messages.append)
    folder = tmp_path / "SW_Lac"
    assert sorted(p.name for p in folder.iterdir()) == ["apass.csv", "archive_summary.csv", "ztf_all.csv",
                                                       "ztf_g.txt", "ztf_r.txt"]
    summary = pd.read_csv(folder / "archive_summary.csv", comment="#")
    assert list(summary["database"]) == ["ZTF", "APASS DR9"]
    assert list(summary["status"]) == ["found", "found"]
    assert summary["wavelength"][0] == "Optical (g, r, i)" and "ztf_g.txt" in summary["files"][0]
    assert (folder / "archive_summary.csv").read_text().startswith("# SW Lac, RA 343.423600 Dec +37.938500")
    assert [r.key for r in results] == ["ztf", "apass"]
    assert messages[0].startswith("SW Lac: RA 22:53:41.66, Dec +37:56:18.6")
    assert messages[-1].startswith("2 of 2 databases had data for SW Lac.")


def test_download_follows_the_listed_order_whatever_order_the_keys_come_in(tmp_path, vizier):
    vizier(EMPTY)
    results = download(TARGET, tmp_path, ["allwise", "galex"])
    assert [r.key for r in results] == ["galex", "allwise"]


def test_an_unexpected_error_in_one_database_does_not_stop_the_rest(tmp_path, vizier):
    vizier(GOOD)
    messages = []
    with patch.object(archives.requests, "get", side_effect=RuntimeError("something odd")):
        results = download(TARGET, tmp_path, ["ztf", "apass"], messages.append)
    assert [r.status for r in results] == ["failed", "found"]
    assert results[0].note == "something odd"
    assert "couldn't be reached or gave an error: ZTF" in messages[-1]


def test_download_stops_between_databases_when_canceled(tmp_path, vizier):
    fake = vizier(GOOD)
    cancel = threading.Event()

    def cancel_after_first(message):
        if message.startswith("GALEX"):
            cancel.set()
    results = download(TARGET, tmp_path, ["galex", "twomass", "allwise"], cancel_after_first, cancel)
    assert [r.key for r in results] == ["galex"]
    assert len(fake.calls) == 1
    assert (tmp_path / "SW_Lac" / "archive_summary.csv").exists()


def test_download_needs_at_least_one_database(tmp_path):
    with pytest.raises(ArchiveError, match="at least one database"):
        download(TARGET, tmp_path, [])


# --- the command line ----------------------------------------------------------------------------------------

def test_command_line_runs_a_download(tmp_path):
    with patch.object(archives, "download") as run:
        archives.main(["SW Lac", str(tmp_path), "--ra", "22:53:41.66", "--dec", "+37:56:18.6",
                       "--only", "ztf, galex"])
    target, folder, keys = run.call_args[0]
    assert target.name == "SW Lac" and folder == str(tmp_path) and keys == ["ztf", "galex"]


def test_command_line_rejects_unknown_databases(tmp_path, capsys):
    with pytest.raises(SystemExit):
        archives.main(["SW Lac", str(tmp_path), "--only", "ztf,nonsense"])
    assert "unknown database keys: nonsense" in capsys.readouterr().err


# --- MAST catalogues and backups ----------------------------------------------------------------

def test_galex_comes_from_mast(tmp_path, monkeypatch, vizier):
    fake = vizier(GOOD)
    table = Table({"objID": [1, 2], "nuv_mag": [14.2, 18.0], "distance": [4.1, 0.3]})
    seen = {}

    def query(coordinates, **kwargs):
        seen.update(kwargs)
        return table
    monkeypatch.setattr("astroquery.mast.Catalogs.query_region", query)
    status, rows, files, note = _fetch("galex", tmp_path)
    assert (status, rows) == ("found", 2) and "from MAST" in note
    assert list(pd.read_csv(files[0])["distance"]) == [0.3, 4.1]          # nearest first
    assert seen["catalog"] == "Galex" and seen["radius"].to_value(u.arcsec) == pytest.approx(5)
    assert fake.calls == []                                               # VizieR wasn't needed


def test_panstarrs_asks_mast_for_dr2_mean_photometry(tmp_path, monkeypatch):
    seen = {}

    def query(coordinates, **kwargs):
        seen.update(kwargs)
        return Table({"objID": [], "distance": []})
    monkeypatch.setattr("astroquery.mast.Catalogs.query_region", query)
    status, _rows, _files, note = _fetch("panstarrs", tmp_path)
    assert status == "nothing" and "3 arcsec" in note
    assert (seen["catalog"], seen["data_release"], seen["table"]) == ("Panstarrs", "dr2", "mean")


def test_galex_falls_back_to_vizier_when_mast_fails(tmp_path, vizier):
    vizier(GOOD)
    messages = []
    status, rows, _files, note = _fetch("galex", tmp_path, _run(messages))
    assert (status, rows) == ("found", 2)
    assert note.startswith("From VizieR, as MAST gave an error.")
    assert messages[0] == "MAST didn't work (MAST is down in this test); trying VizieR."


def test_backup_is_not_used_when_the_first_source_finds_nothing(tmp_path, monkeypatch, vizier):
    fake = vizier(GOOD)
    monkeypatch.setattr("astroquery.mast.Catalogs.query_region", lambda *a, **k: Table({"objID": []}))
    assert _fetch("galex", tmp_path)[0] == "nothing"
    assert fake.calls == []


def test_both_sources_failing_names_both(tmp_path, vizier):
    vizier(OUTAGE)
    with pytest.raises(OSError, match="MAST: MAST is down in this test VizieR: VizieR couldn't be reached"):
        _fetch("galex", tmp_path)


VIZIER_EPOCHS_ROWS = Table({"Source": [7, 7, 7], "Band": ["G", "G", "RP"], "Time": [1700.25, 1701.25, 1700.25],
                            "mag": [12.0, 12.5, 11.4], "flux_over_error": [1000.0, 500.0, 300.0]})
VIZIER_EPOCHS_COLUMNS = Table({"Source": [7, 7], "TimeG": [1700.25, 1701.25], "Gmag": [12.0, 12.5],
                               "e_Gmag": [0.001, 0.002], "TimeRP": [1700.3, 1701.3], "RPmag": [11.4, 11.5]})


def _esa_down(*args, **kwargs):
    raise OSError("Gaia's data service gave an error for EPOCH_PHOTOMETRY (Error 500:).")


def test_gaia_light_curve_falls_back_to_vizier_when_esa_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, "_fetch_gaia_light_curve", _esa_down)
    archive = _archive("gaia_lc")
    rebuilt = archives._with_backup(archives._fetch_gaia_light_curve, archives._fetch_gaia_light_curve_vizier,
                                    "ESA's Gaia archive", "VizieR's copy")
    with patch.object(archives, "query_region_tables", return_value={"I/355/epphot": VIZIER_EPOCHS_ROWS}) as q:
        status, rows, files, note = rebuilt(TARGET, tmp_path, _run(), archive.key)
    assert q.call_args[0][0] == "I/355/epphot"
    assert (status, rows) == ("found", 3)
    assert sorted(f.name for f in files) == ["gaia_lc_G.txt", "gaia_lc_RP.txt", "gaia_lc_all.csv"]
    assert np.loadtxt(tmp_path / "gaia_lc_G.txt")[0, 0] == pytest.approx(1700.25 + 2455197.5)
    assert note.startswith("From VizieR's copy, as ESA's Gaia archive gave an error. Time is BJD (TCB).")


def test_gaia_light_curve_from_a_table_with_columns_per_band(tmp_path):
    with patch.object(archives, "query_region_tables", return_value={"I/355/epphot": VIZIER_EPOCHS_COLUMNS}):
        status, _rows, files, note = archives._fetch_gaia_light_curve_vizier(TARGET, tmp_path, _run(), "gaia_lc")
    assert status == "found" and note == "Time is BJD (TCB)."
    assert sorted(f.name for f in files) == ["gaia_lc_G.txt", "gaia_lc_RP.txt", "gaia_lc_all.csv"]
    g = np.loadtxt(tmp_path / "gaia_lc_G.txt")
    assert g[:, 1].tolist() == [12.0, 12.5] and g[1, 2] == pytest.approx(0.002)


def test_gaia_table_with_unrecognised_columns_is_still_saved(tmp_path):
    odd = Table({"Source": [7], "something": [1.0]})
    with patch.object(archives, "query_region_tables", return_value={"I/355/epphot": odd}):
        status, _rows, files, note = archives._fetch_gaia_light_curve_vizier(TARGET, tmp_path, _run(), "gaia_lc")
    assert status == "found" and [f.name for f in files] == ["gaia_lc_all.csv"]
    assert "columns weren't recognised" in note


def test_gaia_spectra_fall_back_to_vizier(tmp_path):
    spectrum = Table({"Source": [7], "Flux": [[1.0, 2.0]]})
    with patch.object(archives, "query_region_tables", return_value={"I/355/xpsample": spectrum}) as q:
        status, _rows, files, note = archives._fetch_gaia_spectra_vizier(TARGET, tmp_path, _run(), "gaia_spectra")
    assert q.call_args[0][0] == "I/355/xpsample"
    assert status == "found" and files[0].name == "gaia_spectra_xp.csv" and "RVS" in note


def test_rosat_uses_the_2rxs_catalogue():
    seen = {}

    def record(catalog, coordinates, **kwargs):
        seen["catalog"] = catalog
        return {}
    with patch.object(archives, "query_region_tables", side_effect=record):
        _archive("rosat").fetch(TARGET, None, _run(), "rosat")
    assert seen["catalog"] == "J/A+A/588/A103/cat2rxs"
