# -*- coding: utf-8 -*-
"""Tests for extinction.py. IRSA, Gaia and the name resolver are never contacted."""

from unittest.mock import patch

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from EclipsingBinaries import extinction as ex
from EclipsingBinaries.gaia import GaiaRows

# NSVS 896797, at Galactic latitude about +16 degrees
RA, DEC = "00:28:27.96", "+78:57:42.65"


def _dust(mean=0.2, std=0.01):
    return Table({"ext SandF mean": [mean], "ext SandF std": [std]})


def _gaia(parallax, error, ra=None, dec=None):
    coord = SkyCoord(RA, DEC, unit=("hourangle", "deg"))
    return GaiaRows(Table({"ra": [coord.ra.deg if ra is None else ra], "dec": [coord.dec.deg if dec is None else dec],
                           "pmra": [0.0], "pmdec": [0.0], "parallax": [parallax], "parallax_error": [error],
                           "phot_g_mean_mag": [11.0]}))


def test_scaling_for_distance():
    # Far above the dust layer, all of it counts
    assert ex.scale_for_distance(0.2, 5000, 30) == pytest.approx(0.2, rel=1e-6)
    # 250 pc at 30 degrees is 125 pc up: 1 - 1/e of it
    assert ex.scale_for_distance(0.2, 250, 30) == pytest.approx(0.2 * (1 - np.exp(-1)))
    assert ex.scale_for_distance(0.2, 250, -30) == ex.scale_for_distance(0.2, 250, 30)
    assert ex.scale_for_distance(0.2, 250, 0) == 0.2


def test_look_up_scales_by_the_gaia_distance():
    with patch.object(ex, "total_reddening", return_value=(0.2, 0.01)), \
            patch("EclipsingBinaries.gaia.GaiaRows.from_query", return_value=_gaia(4.0, 0.05)):
        result = ex.look_up(ra=RA, dec=DEC)
    assert result.distance_pc == pytest.approx(250)
    assert 0 < result.scaled < result.total == 0.2
    assert result.best == result.scaled
    assert "250 pc" in result.message() and "which is used" in result.message()


def test_a_poor_parallax_leaves_the_total():
    with patch.object(ex, "total_reddening", return_value=(0.2, 0.01)), \
            patch("EclipsingBinaries.gaia.GaiaRows.from_query", return_value=_gaia(1.0, 0.5)):
        result = ex.look_up(ra=RA, dec=DEC)
    assert result.scaled is None and result.best == 0.2
    assert "more reddening than it really has" in result.message()


def test_gaia_being_down_leaves_the_total():
    messages = []
    with patch.object(ex, "total_reddening", return_value=(0.2, 0.01)), \
            patch.object(ex, "gaia_distance", side_effect=OSError("down")):
        result = ex.look_up(ra=RA, dec=DEC, write_callback=messages.append)
    assert result.best == 0.2
    assert "distance" in messages[0]


def test_irsa_being_down_is_an_os_error():
    with patch.object(ex, "total_reddening", side_effect=ConnectionError("down")):
        with pytest.raises(OSError, match="NED's extinction calculator"):
            ex.look_up(ra=RA, dec=DEC)


def test_irsa_table_is_read():
    with patch("astroquery.ipac.irsa.irsa_dust.IrsaDust.get_query_table", return_value=_dust(0.31, 0.02)) as query:
        assert ex.total_reddening(SkyCoord(1, 2, unit="deg")) == (0.31, 0.02)
    assert query.call_args.kwargs["section"] == "ebv"


def test_a_name_is_resolved_when_there_is_no_position():
    with patch("astropy.coordinates.SkyCoord.from_name", return_value=SkyCoord(1, 2, unit="deg")) as named:
        assert ex.resolve(name="SW Lac").ra.deg == pytest.approx(1)
    named.assert_called_once_with("SW Lac")
    with pytest.raises(ValueError):
        ex.resolve()
