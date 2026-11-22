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


def _dust(sfd=0.2, std=0.01):
    """IRSA's E(B-V) table. The other columns differ, so using the wrong one shows."""
    return Table({"ext SFD ref": [sfd], "ext SFD mean": [sfd + 0.05], "ext SFD std": [std],
                  "ext SandF ref": [0.86 * sfd], "ext SandF mean": [0.86 * sfd + 0.05], "ext SandF std": [std]})


def _gaia(parallax, error, ra=None, dec=None):
    coord = SkyCoord(RA, DEC, unit=("hourangle", "deg"))
    return GaiaRows(Table({"ra": [coord.ra.deg if ra is None else ra], "dec": [coord.dec.deg if dec is None else dec],
                           "pmra": [0.0], "pmdec": [0.0], "parallax": [parallax], "parallax_error": [error],
                           "phot_g_mean_mag": [11.0]}))


def _b64(array):
    """An array as the Bayestar19 service encodes it."""
    import base64
    return {"_type": "np.ndarray", "data": base64.b64encode(np.ascontiguousarray(array).data).decode(),
            "dtype": {"_type": "np.dtype", "descr": array.dtype.descr if array.dtype.names else array.dtype.str},
            "shape": list(array.shape)}


def _bayestar_reply(values, reliable=True):
    import json
    flags = np.zeros(1, dtype=[("converged", "?"), ("reliable_dist", "?")])
    flags["converged"], flags["reliable_dist"] = True, reliable
    return json.dumps([_b64(np.asarray(values, dtype="<f4")), _b64(flags)])


class _Reply:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_bayestar_request_and_reply():
    import json
    coord = SkyCoord(RA, DEC, unit=("hourangle", "deg"))
    with patch("requests.post", return_value=_Reply(_bayestar_reply([0.05, 0.06, 0.08], reliable=False))) as post:
        median, (low, high), reliable = ex.bayestar_reddening(coord, 250)
    assert median == pytest.approx(0.884 * 0.06, rel=1e-5)
    assert (low, high) == pytest.approx((0.884 * 0.05, 0.884 * 0.08), rel=1e-5)
    assert reliable is False
    sent = json.loads(post.call_args.kwargs["data"])
    assert post.call_args.args[0] == ex.BAYESTAR_URL
    assert sent["mode"] == "percentile" and sent["pct"] == [16, 50, 84] and sent["return_flags"] is True
    assert sent["coords"]["frame"] == "galactic"
    assert sent["coords"]["distance"] == {"_type": "astropy.units.Quantity", "value": 0.25, "unit": "kpc"}
    assert sent["coords"]["lat"]["value"] == pytest.approx(coord.galactic.b.deg)


def test_bayestar_reply_in_readable_form_and_outside_the_map():
    readable = '[{"_type": "np.ndarray", "dtype": {"_type": "np.dtype", "descr": "<f4"}, "value": [0.1, 0.2, 0.3]}]'
    with patch("requests.post", return_value=_Reply(readable)):
        median, _range, reliable = ex.bayestar_reddening(SkyCoord(1, 2, unit="deg"), 300)
    assert median == pytest.approx(0.884 * 0.2) and reliable is None
    with patch("requests.post", return_value=_Reply(_bayestar_reply([np.nan] * 3))):
        assert ex.bayestar_reddening(SkyCoord(1, 2, unit="deg"), 300) is None


def test_bayestar_being_down_is_an_os_error():
    import requests
    with patch("requests.post", side_effect=requests.ConnectionError("down")):
        with pytest.raises(OSError, match="Bayestar19"):
            ex.bayestar_reddening(SkyCoord(1, 2, unit="deg"), 300)


def test_look_up_uses_bayestar_with_the_gaia_distance():
    with patch.object(ex, "total_reddening", return_value=(0.2, 0.01)), \
            patch("EclipsingBinaries.gaia.GaiaRows.from_query", return_value=_gaia(4.0, 0.05)), \
            patch.object(ex, "bayestar_reddening", return_value=(0.06, (0.05, 0.08), True)) as bayestar:
        result = ex.look_up(ra=RA, dec=DEC)
    assert bayestar.call_args.args[1] == pytest.approx(250)
    assert result.method == ex.BAYESTAR19 and result.best == 0.06
    assert result.total == 0.2      # kept to compare with
    text = result.message()
    assert "Bayestar19" in text and "This is used" in text and "Warning" not in text


def test_look_up_falls_back_to_the_2d_map_as_it_is_when_bayestar_fails():
    messages = []
    with patch.object(ex, "total_reddening", return_value=(0.2, 0.01)), \
            patch("EclipsingBinaries.gaia.GaiaRows.from_query", return_value=_gaia(4.0, 0.05)), \
            patch.object(ex, "bayestar_reddening", side_effect=OSError("Bayestar19 couldn't be reached.")):
        result = ex.look_up(ra=RA, dec=DEC, write_callback=messages.append)
    # The 2D value isn't changed for the star's distance
    assert result.method == ex.TOTAL_2D and result.best == 0.2
    assert any("2D map is used instead" in m for m in messages)
    # Only the real reason is given, not "no value at the star's position" as well
    assert not any("no value at the star's position" in m for m in messages)
    assert "This is used" in result.message()


def test_southern_stars_skip_bayestar():
    messages = []
    south = SkyCoord(10, -45, unit="deg")
    with patch.object(ex, "resolve", return_value=south), \
            patch.object(ex, "total_reddening", return_value=(0.1, 0.01)), \
            patch.object(ex, "gaia_distance", return_value=300.0), \
            patch.object(ex, "bayestar_reddening") as bayestar:
        result = ex.look_up(name="somewhere south", write_callback=messages.append)
    bayestar.assert_not_called()
    assert result.method == ex.TOTAL_2D and result.best == 0.1
    assert any("south of declination -30" in m for m in messages)


def test_a_poor_parallax_leaves_the_total():
    with patch.object(ex, "total_reddening", return_value=(0.2, 0.01)), \
            patch("EclipsingBinaries.gaia.GaiaRows.from_query", return_value=_gaia(1.0, 0.5)), \
            patch.object(ex, "bayestar_reddening") as bayestar:
        result = ex.look_up(ra=RA, dec=DEC)
    bayestar.assert_not_called()
    assert result.method == ex.TOTAL_2D and result.best == 0.2
    assert "more reddening than the star really has" in result.message()
    assert "0.0100 in E(B-V) (standard deviation) within 5 arcmin" in result.message()
    assert "not an uncertainty" in result.message()


def test_bayestar_still_answers_when_irsa_is_down():
    with patch.object(ex, "total_reddening", side_effect=ConnectionError("down")), \
            patch.object(ex, "gaia_distance", return_value=250.0), \
            patch.object(ex, "bayestar_reddening", return_value=(0.06, (0.05, 0.08), False)):
        result = ex.look_up(ra=RA, dec=DEC)
    assert result.method == ex.BAYESTAR19 and result.total is None
    assert "Warning" in result.message()


def test_nothing_answering_is_an_os_error():
    with patch.object(ex, "total_reddening", side_effect=ConnectionError("down")), \
            patch.object(ex, "gaia_distance", return_value=None):
        with pytest.raises(OSError, match="NED's extinction calculator"):
            ex.look_up(ra=RA, dec=DEC)


def test_irsa_table_is_read_as_ned_a_v_over_3_1():
    # NSVS 254037: NED gives A_V = 0.641 (Landolt V), so E(B-V) = 0.641 / 3.1
    sfd = 0.641 / 2.742
    with patch("astroquery.ipac.irsa.irsa_dust.IrsaDust.get_query_table", return_value=_dust(sfd, 0.02)) as query:
        ebv, spread = ex.total_reddening(SkyCoord(1, 2, unit="deg"))
    assert ebv == pytest.approx(0.641 / 3.1)
    assert spread == pytest.approx(0.02 * 2.742 / 3.1)
    assert query.call_args.kwargs["section"] == "ebv"


def test_the_log_gives_a_v_and_the_division_by_3_1():
    result = ex.Reddening(0.2068, ex.TOTAL_2D, 16.1, total=0.2068, total_error=0.008)
    text = result.message()
    assert "A_V at the star's position is 0.6411" in text
    assert "E(B-V) = A_V / 3.1 = 0.2068" in text
    bayestar = ex.Reddening(0.1, ex.BAYESTAR19, 16.1, distance_pc=300, bayestar_range=(0.09, 0.11))
    assert "so A_V = 0.3100" in bayestar.message()


def test_a_name_is_resolved_when_there_is_no_position():
    with patch("astropy.coordinates.SkyCoord.from_name", return_value=SkyCoord(1, 2, unit="deg")) as named:
        assert ex.resolve(name="SW Lac").ra.deg == pytest.approx(1)
    named.assert_called_once_with("SW Lac")
    with pytest.raises(ValueError):
        ex.resolve()
