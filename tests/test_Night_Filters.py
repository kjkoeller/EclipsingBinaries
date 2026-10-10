# -*- coding: utf-8 -*-
"""
Tests for combining nightly AIJ files with Night Filters.
"""

import pandas as pd
import pytest

from EclipsingBinaries.Night_Filters import NightFileError, combine_nights, main


def _night(path, start, rows=4, flux=True, bjd=True, sep="\t", blank_index=True):
    data = {}
    if blank_index:
        data[" "] = range(1, rows + 1)  # AIJ's unnamed row-number column
    data["HJD"] = [start + 0.01 * i for i in range(rows)]
    if bjd:
        data["BJD_TDB"] = [start + 0.0008 + 0.01 * i for i in range(rows)]
    data["Source_AMag_T1"] = [12.0 + 0.1 * i for i in range(rows)]
    data["Source_AMag_Err_T1"] = [0.01] * rows
    if flux:
        data["rel_flux_T1"] = [0.5 + 0.01 * i for i in range(rows)]
        data["rel_flux_err_T1"] = [0.001] * rows
    pd.DataFrame(data).to_csv(path, sep=sep, index=False)
    return path


def _read(path):
    return pd.read_csv(path, sep="\t", header=None)


def test_combines_nights_in_time_order(tmp_path):
    late = _night(tmp_path / "n2.dat", 2458380.6)
    early = _night(tmp_path / "n1.dat", 2458379.6)
    result = combine_nights([late, early], tmp_path / "out", "T_B", write_callback=lambda m: None)

    mags = _read(result["magnitudes"])
    assert result["rows"] == 8
    assert list(mags[0]) == sorted(mags[0])
    assert mags.shape[1] == 3
    assert mags[1].iloc[0] == pytest.approx(12.0)


def test_magnitude_and_flux_files_hold_the_right_columns(tmp_path):
    """The old version wrote flux into the magnitudes file and the other way round."""
    night = _night(tmp_path / "n1.dat", 2458379.6)
    result = combine_nights([night], tmp_path, "T_V", write_callback=lambda m: None)
    assert result["magnitudes"].endswith("T_V_magnitudes.txt")
    assert _read(result["magnitudes"])[1].iloc[0] == pytest.approx(12.0)
    assert _read(result["flux"])[1].iloc[0] == pytest.approx(0.5)


def test_no_flux_file_without_flux_columns(tmp_path):
    night = _night(tmp_path / "n1.dat", 2458379.6, flux=False)
    result = combine_nights([night], tmp_path, "T", write_callback=lambda m: None)
    assert result["flux"] is None
    assert not (tmp_path / "T_flux.txt").exists()


def test_mixed_flux_writes_magnitudes_only_and_says_so(tmp_path):
    a = _night(tmp_path / "n1.dat", 2458379.6)
    b = _night(tmp_path / "n2.dat", 2458380.6, flux=False)
    messages = []
    result = combine_nights([a, b], tmp_path, "T", write_callback=messages.append)
    assert result["flux"] is None
    assert any("no flux columns in n2.dat" in m for m in messages)


def test_bjd_column_first_when_asked(tmp_path):
    night = _night(tmp_path / "n1.dat", 2458379.6)
    result = combine_nights([night], tmp_path, "T", include_bjd=True, write_callback=lambda m: None)
    mags = _read(result["magnitudes"])
    assert mags.shape[1] == 4
    assert mags[0].iloc[0] == pytest.approx(2458379.6008)
    assert mags[1].iloc[0] == pytest.approx(2458379.6)


def test_bjd_asked_but_missing(tmp_path):
    night = _night(tmp_path / "n1.dat", 2458379.6, bjd=False)
    with pytest.raises(NightFileError, match="no BJD_TDB column"):
        combine_nights([night], tmp_path, "T", include_bjd=True, write_callback=lambda m: None)


def test_missing_magnitude_columns_name_the_file(tmp_path):
    bad = tmp_path / "bad.dat"
    pd.DataFrame({"HJD": [1.0], "Mag": [12.0]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(NightFileError, match="bad.dat is missing the column"):
        combine_nights([bad], tmp_path, "T", write_callback=lambda m: None)


def test_comma_separated_photometry_output_is_accepted(tmp_path):
    """Our own photometry CSV names the error column differently."""
    path = tmp_path / "T_V_data.csv"
    pd.DataFrame({"HJD": [2458379.6, 2458379.61], "BJD": [2458379.6, 2458379.61],
                  "Source_AMag_T1": [12.0, 12.1], "Source_AMag_T1_Error": [0.01, 0.01]}).to_csv(path, index=False)
    result = combine_nights([path], tmp_path, "T", write_callback=lambda m: None)
    assert result["rows"] == 2


def test_bad_rows_are_skipped(tmp_path):
    night = _night(tmp_path / "n1.dat", 2458379.6)
    text = night.read_text().splitlines()
    text[2] = text[2].replace("12.1", "NaN")
    night.write_text("\n".join(text) + "\n")
    messages = []
    result = combine_nights([night], tmp_path, "T", write_callback=messages.append)
    assert result["rows"] == 3
    assert any("skipped 1 row" in m for m in messages)


def test_needs_files_and_a_name(tmp_path):
    with pytest.raises(NightFileError):
        combine_nights([], tmp_path, "T")
    with pytest.raises(NightFileError, match="output name"):
        combine_nights([_night(tmp_path / "n.dat", 1.0)], tmp_path, "  ")
    with pytest.raises(NightFileError, match="doesn't exist"):
        combine_nights([tmp_path / "missing.dat"], tmp_path, "T")


def test_cancel_stops_before_writing(tmp_path):
    import threading

    event = threading.Event()
    event.set()
    night = _night(tmp_path / "n1.dat", 2458379.6)
    assert combine_nights([night], tmp_path, "T", cancel_event=event, write_callback=lambda m: None) is None
    assert not (tmp_path / "T_magnitudes.txt").exists()


def test_terminal_prompts(tmp_path, monkeypatch):
    night = _night(tmp_path / "n1.dat", 2458379.6)
    answers = iter(["abc", "1", str(tmp_path / "nope.dat"), str(night), "combined"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.chdir(tmp_path)
    assert main() == 0
    assert (tmp_path / "combined_magnitudes.txt").exists()
