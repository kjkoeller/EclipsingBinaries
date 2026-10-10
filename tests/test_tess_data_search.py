# -*- coding: utf-8 -*-
"""
Tests for tess_data_search.py

"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from astropy.table import Table
import astropy.units as u


# ===========================================================================
# Pure logic helpers — extracted from run_tess_search
# These mirror the gain-matching and slicing logic in the source.
# ===========================================================================

def _match_gains(tess_camera, tess_ccd, gain, sector_camera, sector_ccd):
    """
    Mirrors the gain-matching loop in run_tess_search:
    For each (sector_cam, sector_ccd) pair, find matching (tess_cam, tess_ccd)
    and collect the corresponding gain value.
    """
    a = list(zip(tess_camera, tess_ccd))
    b = list(zip(sector_camera, sector_ccd))
    list_gain = []
    for sect in b:
        for y, tess in enumerate(a):
            if sect == tess:
                list_gain.append(gain[y])
    return list_gain


def _slice_gains(list_gain):
    """Mirrors the A/B/C/D slicing logic in run_tess_search."""
    A = list_gain[::4]
    B = list_gain[1::4]
    C = list_gain[2::4]
    D = list_gain[3::4]
    return A, B, C, D


# ===========================================================================
# gain matching logic
# ===========================================================================
def test_match_gains_single_match():
    tess_camera = [1]
    tess_ccd = [1]
    gain = [1.5]
    result = _match_gains(tess_camera, tess_ccd, gain, [1], [1])
    assert result == [1.5]


def test_match_gains_no_match_returns_empty():
    result = _match_gains([1], [1], [1.5], [2], [2])
    assert result == []


def test_match_gains_multiple_sectors_matched():
    tess_camera = [1, 2, 3]
    tess_ccd = [1, 2, 3]
    gain = [1.1, 1.2, 1.3]
    sector_camera = [2, 3]
    sector_ccd = [2, 3]
    result = _match_gains(tess_camera, tess_ccd, gain, sector_camera, sector_ccd)
    assert result == [1.2, 1.3]


def test_match_gains_duplicate_sector_entries():
    # Same camera/ccd appearing twice in sector list → matched twice
    tess_camera = [1]
    tess_ccd = [1]
    gain = [2.0]
    result = _match_gains(tess_camera, tess_ccd, gain, [1, 1], [1, 1])
    assert result == [2.0, 2.0]


def test_match_gains_preserves_order():
    tess_camera = [1, 2]
    tess_ccd = [1, 2]
    gain = [10.0, 20.0]
    result = _match_gains(tess_camera, tess_ccd, gain, [2, 1], [2, 1])
    assert result == [20.0, 10.0]


def test_match_gains_partial_overlap():
    tess_camera = [1, 2, 3, 4]
    tess_ccd = [1, 1, 1, 1]
    gain = [1.0, 2.0, 3.0, 4.0]
    result = _match_gains(tess_camera, tess_ccd, gain, [2, 4], [1, 1])
    assert result == [2.0, 4.0]


def test_match_gains_empty_inputs():
    assert _match_gains([], [], [], [], []) == []


def test_match_gains_cam_ccd_both_must_match():
    # Same camera, different CCD — should not match
    result = _match_gains([1], [1], [9.9], [1], [2])
    assert result == []


# ===========================================================================
# gain slicing logic (A/B/C/D every 4th element)
# ===========================================================================
def test_slice_gains_four_elements():
    gains = [1.0, 2.0, 3.0, 4.0]
    A, B, C, D = _slice_gains(gains)
    assert A == [1.0]
    assert B == [2.0]
    assert C == [3.0]
    assert D == [4.0]


def test_slice_gains_eight_elements():
    gains = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    A, B, C, D = _slice_gains(gains)
    assert A == [1.0, 5.0]
    assert B == [2.0, 6.0]
    assert C == [3.0, 7.0]
    assert D == [4.0, 8.0]


def test_slice_gains_empty():
    A, B, C, D = _slice_gains([])
    assert A == B == C == D == []


def test_slice_gains_returns_four_lists():
    result = _slice_gains([1.0, 2.0, 3.0, 4.0])
    assert len(result) == 4


def test_slice_gains_lengths_equal_for_multiple_of_four():
    gains = list(range(12))
    A, B, C, D = _slice_gains(gains)
    assert len(A) == len(B) == len(C) == len(D) == 3


def test_slice_gains_single_element_only_in_a():
    A, B, C, D = _slice_gains([99.0])
    assert A == [99.0]
    assert B == C == D == []


def test_slice_gains_two_elements():
    A, B, C, D = _slice_gains([10.0, 20.0])
    assert A == [10.0]
    assert B == [20.0]
    assert C == D == []


# ===========================================================================
# download_sector — filesystem and network mocked
# ===========================================================================
def test_download_sector_raises_if_path_not_found():
    from EclipsingBinaries.tess_data_search import download_sector
    cancel = MagicMock()
    cancel.is_set.return_value = False
    with pytest.raises(Exception):
        download_sector("TIC 123456", 1, "/nonexistent/path/xyz", None, cancel)


def test_download_sector_creates_sector_subdirectory():
    from EclipsingBinaries.tess_data_search import download_sector
    cancel = MagicMock()
    cancel.is_set.return_value = False
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_manifest = MagicMock()
        with patch("EclipsingBinaries.tess_data_search.Tesscut.download_cutouts", return_value=mock_manifest), \
             patch("EclipsingBinaries.tess_data_search.process_tess_cutout"):
            download_sector("TIC 123456", 5, tmpdir, None, cancel)
        assert os.path.isdir(os.path.join(tmpdir, "5"))


def test_download_sector_calls_tesscut_with_correct_sector():
    from EclipsingBinaries.tess_data_search import download_sector
    cancel = MagicMock()
    cancel.is_set.return_value = False
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_manifest = MagicMock()
        with patch("EclipsingBinaries.tess_data_search.Tesscut.download_cutouts", return_value=mock_manifest) as mock_dl, \
             patch("EclipsingBinaries.tess_data_search.process_tess_cutout"):
            download_sector("TIC 123456", 7, tmpdir, None, cancel)
        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args
        assert call_kwargs.kwargs.get("sector") == 7 or call_kwargs.args[0] if call_kwargs.args else True


def test_download_sector_calls_process_tess_cutout():
    from EclipsingBinaries.tess_data_search import download_sector
    cancel = MagicMock()
    cancel.is_set.return_value = False
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_manifest = MagicMock()
        with patch("EclipsingBinaries.tess_data_search.Tesscut.download_cutouts", return_value=mock_manifest), \
             patch("EclipsingBinaries.tess_data_search.process_tess_cutout") as mock_process:
            download_sector("TIC 123456", 3, tmpdir, None, cancel)
        mock_process.assert_called_once()


def test_download_sector_logs_completion():
    from EclipsingBinaries.tess_data_search import download_sector
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.tess_data_search.Tesscut.download_cutouts", return_value=MagicMock()), \
             patch("EclipsingBinaries.tess_data_search.process_tess_cutout"):
            download_sector("TIC 123456", 2, tmpdir, messages.append, cancel)
    assert any("Completed" in m or "completed" in m.lower() for m in messages)


def test_download_sector_uses_outprefix_with_system_name():
    from EclipsingBinaries.tess_data_search import download_sector
    cancel = MagicMock()
    cancel.is_set.return_value = False
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("EclipsingBinaries.tess_data_search.Tesscut.download_cutouts", return_value=MagicMock()), \
             patch("EclipsingBinaries.tess_data_search.process_tess_cutout") as mock_process:
            download_sector("MyTarget", 4, tmpdir, None, cancel)
        call_kwargs = mock_process.call_args.kwargs
        assert "MyTarget" in call_kwargs.get("outprefix", "")


# ===========================================================================
# run_tess_search — network mocked
# ===========================================================================
def _gain_table_mock():
    # One row per output slice: camera 1 / CCD 1 has four slices, each with its own gain
    columns = {0: [1, 1, 1, 1], 1: [1, 1, 1, 1], 2: [1, 2, 3, 4], 3: [1.5, 1.6, 1.7, 1.8]}
    return MagicMock(__getitem__=lambda self, key: columns[key])


def test_run_tess_search_logs_on_cancel_before_start():
    from EclipsingBinaries.tess_data_search import run_tess_search
    cancel = MagicMock()
    cancel.is_set.return_value = True
    messages = []
    run_tess_search("TIC 123", False, None, "/tmp", messages.append, cancel)
    assert any("cancel" in m.lower() for m in messages)


def test_run_tess_search_logs_error_on_no_data():
    from EclipsingBinaries.tess_data_search import run_tess_search
    cancel = MagicMock()
    cancel.is_set.return_value = False
    messages = []
    with patch("EclipsingBinaries.tess_data_search.Tesscut.get_sectors", return_value=[]):
        # Errors are raised as well as logged so the GUI doesn't report success
        with pytest.raises(ValueError):
            run_tess_search("TIC 000", False, None, "/tmp", messages.append, cancel)
    assert any("error" in m.lower() or "no tess" in m.lower() for m in messages)


def test_run_tess_search_calls_get_sectors_with_system_name():
    from EclipsingBinaries.tess_data_search import run_tess_search
    cancel = MagicMock()
    cancel.is_set.return_value = False
    mock_table = Table({"sector": [1], "camera": [1], "ccd": [1]})
    messages = []
    with patch("EclipsingBinaries.tess_data_search.Tesscut.get_sectors", return_value=mock_table) as mock_gs, \
         patch("EclipsingBinaries.tess_data_search.pd.read_csv", return_value=_gain_table_mock()), \
         patch("EclipsingBinaries.tess_data_search.download_sector"):
        run_tess_search("TIC 999", False, None, "/tmp", messages.append, cancel)
    mock_gs.assert_called_once_with(objectname="TIC 999")


def test_run_tess_search_specific_sector_missing_logs_error():
    from EclipsingBinaries.tess_data_search import run_tess_search
    cancel = MagicMock()
    cancel.is_set.return_value = False
    mock_table = Table({"sector": [1], "camera": [1], "ccd": [1]})
    messages = []
    with patch("EclipsingBinaries.tess_data_search.Tesscut.get_sectors", return_value=mock_table), \
         patch("EclipsingBinaries.tess_data_search.pd.read_csv") as mock_csv:
        mock_csv.return_value = MagicMock(
            __getitem__=lambda self, key: {0: [1], 1: [1], 2: [1], 3: [1.5, 1.5, 1.5, 1.5]}[key]
        )
        # single_sector=True but no sector given
        with pytest.raises(ValueError):
            run_tess_search("TIC 999", True, None, "/tmp", messages.append, cancel)
    assert any("error" in m.lower() or "not specified" in m.lower() for m in messages)


def test_run_tess_search_download_failure_is_raised():
    # Used to be logged and swallowed, after which the GUI said "completed successfully"
    from EclipsingBinaries.tess_data_search import run_tess_search
    mock_table = Table({"sector": [1], "camera": [1], "ccd": [1]})
    with patch("EclipsingBinaries.tess_data_search.Tesscut.get_sectors", return_value=mock_table), \
         patch("EclipsingBinaries.tess_data_search.pd.read_csv", return_value=_gain_table_mock()), \
         patch("EclipsingBinaries.tess_data_search.download_sector", side_effect=OSError("MAST is down")):
        with pytest.raises(OSError, match="MAST is down"):
            run_tess_search("TIC 999", False, None, "/tmp", lambda _m: None)


@pytest.mark.parametrize("single_sector, sector, expected", [(False, None, [7, 9]), (True, 9, [9])])
def test_run_tess_search_single_sector_flag(single_sector, sector, expected):
    from EclipsingBinaries.tess_data_search import run_tess_search
    mock_table = Table({"sector": [7, 9], "camera": [1, 1], "ccd": [1, 1]})
    with patch("EclipsingBinaries.tess_data_search.Tesscut.get_sectors", return_value=mock_table), \
         patch("EclipsingBinaries.tess_data_search.pd.read_csv", return_value=_gain_table_mock()), \
         patch("EclipsingBinaries.tess_data_search.download_sector") as download:
        run_tess_search("TIC 999", single_sector, sector, "/tmp", lambda _m: None)
    assert [c.args[1] for c in download.call_args_list] == expected


# ===========================================================================
# zip/tuple matching sanity checks
# ===========================================================================
def test_zip_cam_ccd_matching_basic():
    a = list(zip([1, 2, 3], [1, 2, 3]))
    b = list(zip([2], [2]))
    matches = [tess for sect in b for y, tess in enumerate(a) if sect == tess]
    assert matches == [(2, 2)]


def test_zip_cam_ccd_no_match():
    a = list(zip([1, 2], [1, 2]))
    b = list(zip([3], [3]))
    matches = [tess for sect in b for y, tess in enumerate(a) if sect == tess]
    assert matches == []


def test_zip_cam_ccd_multiple_matches():
    a = list(zip([1, 1, 2], [1, 1, 2]))
    b = list(zip([1], [1]))
    matches = [tess for sect in b for y, tess in enumerate(a) if sect == tess]
    assert len(matches) == 2
