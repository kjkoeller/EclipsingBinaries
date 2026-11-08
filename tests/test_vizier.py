# -*- coding: utf-8 -*-
"""
Tests for the VizieR and Gaia archive queries when servers misbehave: CDS
sending back an error instead of data, timeouts, mirrors answering with web
pages, and the Gaia archive dropping connections. Nothing here touches the
network; each server's answer is made up.
"""

import astropy.units as u
import pandas as pd
import pytest
import requests
from astropy.coordinates import SkyCoord
from unittest.mock import MagicMock, patch

from EclipsingBinaries import _vizier
from EclipsingBinaries._vizier import VizierUnavailable, query_region

CENTRE = SkyCoord(10.15 * u.deg, 20.25 * u.deg)
COLUMNS = ["_RAJ2000", "_DEJ2000", "Vmag"]

GOOD = b'''<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE ID="yCat_2336" name="II/336">
<TABLE ID="II_336_apass9" name="II/336/apass9">
<FIELD name="_RAJ2000" datatype="double" unit="deg"/>
<FIELD name="_DEJ2000" datatype="double" unit="deg"/>
<FIELD name="Vmag" datatype="float" unit="mag"/>
<DATA><TABLEDATA>
<TR><TD>10.1</TD><TD>20.2</TD><TD>12.5</TD></TR>
<TR><TD>10.2</TD><TD>20.3</TD><TD>13.1</TD></TR>
</TABLEDATA></DATA>
</TABLE>
</RESOURCE>
</VOTABLE>'''

# What CDS sent during the October 2026 outage: HTTP 200 and no table, only an error
OUTAGE = b'''<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<INFO ID="Error" name="Error" value="The database is not currently reachable"/>
</VOTABLE>'''

# A field with nothing in it: no table and no error
EMPTY = b'''<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<INFO name="QUERY_STATUS" value="OK"/>
<RESOURCE ID="yCat_2336" name="II/336"></RESOURCE>
</VOTABLE>'''

WEB_PAGE = b"<!DOCTYPE html><html><body>Service temporarily unavailable</body></html>"


def _response(content, status=200):
    response = requests.Response()
    response._content = content
    response.status_code = status
    return response


class FakeServers:
    """
    Stands in for VizieR. answers says what each server does: bytes to send
    back (optionally with a status code) or an exception to raise. Every call
    is recorded with the server name and the cache setting it was made with.
    """

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, vizier, *args, **kwargs):
        server = vizier.VIZIER_SERVER
        self.calls.append((server, kwargs.get("cache"), vizier.TIMEOUT))
        answer = self.answers[server]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, tuple):
            return _response(*answer)
        return _response(answer)


@pytest.fixture
def servers(monkeypatch):
    def install(answers):
        fake = FakeServers(answers)
        # Vizier is astroquery's ready-made instance, so the method lives on its class
        monkeypatch.setattr(type(_vizier.Vizier), "query_region_async",
                            lambda self, *a, **k: fake(self, *a, **k))
        return fake
    return install


def _everyone(answer):
    return {server: answer for server in _vizier.VIZIER_SERVERS}


# --- the shared VizieR query ---------------------------------------------------------

def test_good_answer_from_cds(servers):
    fake = servers(_everyone(GOOD))
    table = query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m")
    assert len(table) == 2
    assert [call[0] for call in fake.calls] == ["vizier.cds.unistra.fr"]


def test_nothing_is_read_from_or_written_to_the_cache(servers):
    fake = servers(_everyone(GOOD))
    query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m")
    assert fake.calls[0][1] is False


def test_timeouts_are_set_for_connecting_and_reading(servers):
    fake = servers(_everyone(GOOD))
    query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m")
    assert fake.calls[0][2] == (_vizier.CONNECT_TIMEOUT, _vizier.READ_TIMEOUT)


def test_outage_message_moves_on_to_a_mirror(servers):
    answers = _everyone(GOOD)
    answers["vizier.cds.unistra.fr"] = OUTAGE
    fake = servers(answers)
    messages = []
    table = query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m",
                         write_callback=messages.append)
    assert len(table) == 2
    assert [call[0] for call in fake.calls] == ["vizier.cds.unistra.fr", "vizier.idia.ac.za"]
    assert any("not currently reachable" in m and "vizier.idia.ac.za" in m for m in messages)
    assert any("from vizier.idia.ac.za" in m for m in messages)


def test_empty_field_is_not_mistaken_for_an_outage(servers):
    fake = servers(_everyone(EMPTY))
    assert query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m") is None
    assert len(fake.calls) == 1


@pytest.mark.parametrize("failure, reason", [
    (requests.exceptions.ReadTimeout("read timed out"), "timed out"),
    (requests.exceptions.ConnectionError("refused"), "couldn't connect"),
    ((WEB_PAGE,), "wasn't a VOTable"),
    ((b"oops", 500), "HTTP 500"),
])
def test_other_failures_move_on_to_a_mirror(servers, failure, reason):
    answers = _everyone(GOOD)
    answers["vizier.cds.unistra.fr"] = failure
    fake = servers(answers)
    messages = []
    table = query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m",
                         write_callback=messages.append)
    assert len(table) == 2
    assert len(fake.calls) == 2
    assert reason in messages[0]


def test_every_server_down(servers):
    answers = _everyone(OUTAGE)
    answers["vizier.nao.ac.jp"] = requests.exceptions.ConnectTimeout("connect timed out")
    fake = servers(answers)
    with pytest.raises(VizierUnavailable) as caught:
        query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m")
    assert len(fake.calls) == len(_vizier.VIZIER_SERVERS)
    assert not caught.value.timed_out
    text = str(caught.value)
    assert "II/336/apass9" in text and "vizier.inasan.ru" in text and "try again later" in text


def test_every_server_timing_out_is_flagged(servers):
    servers(_everyone(requests.exceptions.ReadTimeout("read timed out")))
    with pytest.raises(VizierUnavailable) as caught:
        query_region("IV/39/tic82", CENTRE, columns=COLUMNS, radius=1 * u.arcmin)
    assert caught.value.timed_out


def test_cfa_is_not_tried():
    assert "vizier.cfa.harvard.edu" not in _vizier.VIZIER_SERVERS


def test_cancel_stops_before_asking_another_server(servers):
    answers = _everyone(GOOD)
    answers["vizier.cds.unistra.fr"] = OUTAGE
    fake = servers(answers)
    cancel = MagicMock()
    cancel.is_set.side_effect = [False, True]
    assert query_region("II/336/apass9", CENTRE, columns=COLUMNS, width="30m",
                        cancel_event=cancel) is None
    assert len(fake.calls) == 1


# --- APASS -------------------------------------------------------------------------------

def test_apass_uses_a_mirror_when_cds_is_down(servers):
    from EclipsingBinaries.apass import query_vizier
    answers = _everyone(GOOD)
    answers["vizier.cds.unistra.fr"] = OUTAGE
    servers(answers)
    table = query_vizier(0.6767, 20.25, None, None)
    assert len(table) == 2


def test_apass_says_when_the_field_has_no_bright_stars(servers):
    from EclipsingBinaries.apass import query_vizier
    servers(_everyone(EMPTY))
    with pytest.raises(ValueError, match="no stars brighter than magnitude 14"):
        query_vizier(0.6767, 20.25, None, None)


def test_apass_reports_when_vizier_is_down(servers):
    from EclipsingBinaries.apass import query_vizier
    servers(_everyone(OUTAGE))
    messages = []
    with pytest.raises(VizierUnavailable):
        query_vizier(0.6767, 20.25, messages.append, None)
    assert any("try again later" in m for m in messages)


# --- TESS magnitudes from the TIC ------------------------------------------------------------

def test_tess_magnitudes_are_left_missing_when_vizier_is_down(servers):
    from EclipsingBinaries.gaia import tess_mag
    servers(_everyone(OUTAGE))
    messages = []
    T, T_err = tess_mag([0.6767, 0.68], [20.25, 20.3], messages.append, None)
    assert T == [99.999, 99.999] and T_err == [99.999, 99.999]
    assert any("99.999" in m and "try again later" in m for m in messages)


def test_tic_outage_does_not_fall_back_to_one_star_at_a_time(servers):
    from EclipsingBinaries.gaia import _query_tic_catalog
    fake = servers(_everyone(OUTAGE))
    with pytest.raises(VizierUnavailable):
        _query_tic_catalog(pd.Series([0.6767, 0.68]).to_numpy(), pd.Series([20.25, 20.3]).to_numpy())
    assert len(fake.calls) == len(_vizier.VIZIER_SERVERS)


def test_tic_timeouts_fall_back_to_one_star_at_a_time():
    from EclipsingBinaries import gaia
    timed_out = VizierUnavailable("IV/39/tic82", [("vizier.cds.unistra.fr", "timed out")])
    star = pd.DataFrame({"_RAJ2000": [10.15], "_DEJ2000": [20.25], "Tmag": [11.0], "e_Tmag": [0.01]})
    with patch.object(gaia, "_query_single_tic_region", side_effect=[timed_out, star, star]) as single:
        result = gaia._query_tic_catalog(pd.Series([0.6767, 0.68]).to_numpy(),
                                         pd.Series([20.25, 20.3]).to_numpy())
    assert single.call_count == 3
    assert len(result) == 1


def test_tic_fallback_stops_at_the_first_star_that_fails():
    from EclipsingBinaries import gaia
    timed_out = VizierUnavailable("IV/39/tic82", [("vizier.cds.unistra.fr", "timed out")])
    with patch.object(gaia, "_query_single_tic_region", side_effect=[timed_out, timed_out]) as single:
        with pytest.raises(VizierUnavailable):
            gaia._query_tic_catalog(pd.Series([0.6767, 0.68, 0.69]).to_numpy(),
                                    pd.Series([20.25, 20.3, 20.4]).to_numpy())
    assert single.call_count == 2


# --- the Gaia archive at ESA --------------------------------------------------------------------

def test_gaia_archive_is_tried_again_after_a_dropped_connection():
    from EclipsingBinaries import gaia
    messages = []
    with patch.object(gaia.GaiaData, "from_query",
                      side_effect=[ConnectionResetError("Connection reset by peer"), "rows"]) as query, \
            patch.object(gaia.time, "sleep"):
        assert gaia._query_gaia_archive("SELECT 1", messages.append) == "rows"
    assert query.call_count == 2
    assert any("trying again" in m for m in messages)


def test_gaia_archive_gives_up_with_advice():
    from EclipsingBinaries import gaia
    with patch.object(gaia.GaiaData, "from_query",
                      side_effect=requests.exceptions.ConnectionError("Remote end closed connection")) as query, \
            patch.object(gaia.time, "sleep"):
        with pytest.raises(OSError, match="pip install -U astroquery"):
            gaia._query_gaia_archive("SELECT 1")
    assert query.call_count == len(gaia.GAIA_RETRY_WAITS) + 1


def test_gaia_archive_mistakes_are_not_retried():
    from EclipsingBinaries import gaia
    with patch.object(gaia.GaiaData, "from_query", side_effect=ValueError("bad ADQL")) as query:
        with pytest.raises(ValueError):
            gaia._query_gaia_archive("SELECT nonsense")
    assert query.call_count == 1


def test_gaia_archive_wait_can_be_canceled():
    from EclipsingBinaries import gaia
    cancel = MagicMock()
    cancel.is_set.return_value = True
    with patch.object(gaia.GaiaData, "from_query", side_effect=ConnectionResetError("reset")) as query, \
            patch.object(gaia.time, "sleep"):
        assert gaia._query_gaia_archive("SELECT 1", cancel_event=cancel) is None
    assert query.call_count == 1
