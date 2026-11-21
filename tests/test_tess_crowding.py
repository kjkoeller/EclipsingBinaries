# -*- coding: utf-8 -*-
"""
Tests for tess_crowding.py: how much light nearby stars add to a TESS target.
Nothing here touches the network.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from EclipsingBinaries import tess_crowding as tc

TARGET = SkyCoord(343.4236 * u.deg, 37.9385 * u.deg)


def _stars(offsets_arcsec):
    return SkyCoord(TARGET.ra, TARGET.dec + np.asarray(offsets_arcsec, dtype=float) * u.arcsec)


def test_neighbours_are_counted_by_pixel():
    # Target, a star as bright 10" away, one 2.5 mag fainter 30" away, and one 100" away
    crowding = tc.neighbour_flux(TARGET, _stars([0.5, 10, 30, 100]), [10.0, 10.0, 12.5, 9.0])
    assert crowding.target_tmag == 10.0
    assert crowding.within_one == pytest.approx(1.0)
    assert crowding.within_two == pytest.approx(1.1)
    assert crowding.neighbours == 2
    assert crowding.crowded
    assert "100%" in crowding.message() and "Warning" in crowding.message()


def test_an_isolated_star_is_not_crowded():
    crowding = tc.neighbour_flux(TARGET, _stars([0.0, 30]), [9.0, 16.5])
    assert crowding.neighbours == 0 and not crowding.crowded
    assert "No TIC star brighter than Tmag 16.0" in crowding.message()


def test_faint_neighbours_are_noted_without_a_warning():
    crowding = tc.neighbour_flux(TARGET, _stars([0.0, 30]), [9.0, 13.0])
    assert crowding.within_two == pytest.approx(10 ** -1.6)
    assert not crowding.crowded and "Warning" not in crowding.message()


def test_no_star_at_the_position():
    crowding = tc.neighbour_flux(TARGET, _stars([30.0]), [10.0])
    assert crowding.target_tmag is None
    assert "couldn't be worked out" in crowding.message()


def test_tic_crowding_from_a_name_uses_the_nearest_star(monkeypatch):
    table = Table({"ra": [TARGET.ra.deg] * 2, "dec": [TARGET.dec.deg + 20 / 3600, TARGET.dec.deg],
                   "Tmag": [10.0, 11.0], "dstArcSec": [20.0, 0.1]})
    monkeypatch.setattr("astroquery.mast.Catalogs.query_object", lambda *a, **k: table)
    crowding = tc.tic_crowding(name="TIC 1")
    assert crowding.target_tmag == 11.0
    assert crowding.within_one == pytest.approx(10 ** 0.4)


def test_tic_crowding_reports_mast_being_down(monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError("down")
    monkeypatch.setattr("astroquery.mast.Catalogs.query_region", down)
    messages = []
    assert tc.tic_crowding(TARGET, write_callback=messages.append) is None
    assert "couldn't be searched" in messages[0]


def test_crowdsap_from_a_spoc_file(tmp_path):
    table = fits.BinTableHDU.from_columns([fits.Column(name="TIME", format="D", array=[1.0])])
    table.header["CROWDSAP"] = 0.83
    path = tmp_path / "lc.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    assert tc.crowdsap(path) == pytest.approx(0.83)
    assert tc.describe_crowdsap([0.83, None, 0.85]).startswith("SPOC puts 83% to 85%")
    assert "may still include it" in tc.describe_crowdsap([0.83])
    assert tc.describe_crowdsap([None]) == ""
