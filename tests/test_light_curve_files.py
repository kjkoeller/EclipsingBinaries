# -*- coding: utf-8 -*-
"""
Reading light curve files in the formats people actually have: the tab
separated files Night Filters writes, comma separated files, and AIJ
measurement tables saved as CSV with a header row and many columns.
O'Connell Effect and Color Light Curve read through io.importFile_pd;
Find Minimum and Period Search use find_min.read_light_curve.
"""

from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from EclipsingBinaries import find_min
from EclipsingBinaries.vseq_updated import binning, io, light_curve_columns, sniff_text_table

EXAMPLE = files("EclipsingBinaries.examples") / "test_V.txt"
PERIOD, EPOCH = 0.3174245, 2458403.58763


@pytest.fixture
def original():
    return pd.read_csv(EXAMPLE, sep=r"\s+", header=None)


def _aij_table(original, path, time_name="HJD_UTC", sep=","):
    """An AIJ-style measurement table: a header row, text columns and many others."""
    n = len(original)
    table = pd.DataFrame({" ": range(1, n + 1), "Label": [f"V_{i:04d}.fits" for i in range(n)],
                          "J.D.-2400000": original[0] - 2400000, time_name: original[0],
                          "BJD_TDB_x": original[0] + 0.0008, "AIRMASS": 1.2,
                          "Source_AMag_T1": original[1], "Source_AMag_Err_T1": original[2],
                          "rel_flux_T1": 10 ** (-0.4 * original[1])})
    for k in range(40):
        table[f"Source-Sky_C{k + 2}"] = 1000.0 + k
    table.to_csv(path, index=False, sep=sep)
    return str(path)


def _same(columns, original):
    return all(np.allclose(np.asarray(columns[i], float), original[i]) for i in range(3))


@pytest.mark.parametrize("time_name", ["HJD_UTC", "HJD", "BJD_TDB"])
def test_aij_csv_reads_like_the_text_file(tmp_path, original, time_name):
    path = _aij_table(original, tmp_path / "Table_V.csv", time_name=time_name)
    assert _same(io.importFile_pd(path), original)


def test_aij_tab_separated_table(tmp_path, original):
    path = _aij_table(original, tmp_path / "Table_V.txt", sep="\t")
    assert _same(io.importFile_pd(path), original)


def test_plain_comma_separated_file(tmp_path, original):
    path = tmp_path / "V.csv"
    original.to_csv(path, header=False, index=False)
    assert _same(io.importFile_pd(str(path)), original)


def test_night_filters_file_unchanged(original):
    assert _same(io.importFile_pd(str(EXAMPLE)), original)


def test_rows_with_text_in_the_numbers_are_dropped(tmp_path, original):
    path = _aij_table(original, tmp_path / "Table_V.csv")
    text = open(path).read().splitlines()
    cells = text[3].split(",")
    cells[3] = "NaN"   # the HJD_UTC cell
    text[3] = ",".join(cells)
    open(path, "w").write("\n".join(text) + "\n")
    assert len(io.importFile_pd(path)[0]) == len(original) - 1


def test_missing_columns_are_named(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"HJD_UTC": [1.0, 2.0], "Source_AMag_T1": [12.0, 12.1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="bad.csv has a header row but no magnitude error"):
        io.importFile_pd(str(path))


def test_light_curve_columns_prefer_hjd():
    cols = ["BJD_TDB", "HJD_UTC", "HJD", "Source_AMag_T1", "Source_AMag_Err_T1"]
    assert light_curve_columns(cols) == ["HJD", "Source_AMag_T1", "Source_AMag_Err_T1"]


def test_sniffing(tmp_path):
    for name, text, expected in [("a.csv", "HJD,mag,err\n1,2,3\n", (",", True)),
                                 ("b.txt", "1\t2\t3\n", ("\t", False)),
                                 ("c.txt", "2458379.5 12.1 0.01\n", (r"\s+", False)),
                                 ("d.csv", "\n\n1.0,2.0,3.0\n", (",", False))]:
        path = tmp_path / name
        path.write_text(text)
        assert sniff_text_table(str(path)) == expected


def test_phase_binning_runs_on_an_aij_csv(tmp_path, original):
    """The step that failed for O'Connell and Color Light Curve with a CSV table."""
    path = _aij_table(original, tmp_path / "Table_V.csv")
    from_csv = binning.polybinner(path, EPOCH, PERIOD, sections=2, section_order=8)
    from_txt = binning.polybinner(str(EXAMPLE), EPOCH, PERIOD, sections=2, section_order=8)
    assert np.allclose(np.asarray(from_csv[0][1], float), np.asarray(from_txt[0][1], float))


def test_find_minimum_reads_hjd_utc(tmp_path, original):
    path = _aij_table(original, tmp_path / "Table_V.csv")
    t, m, e = find_min.read_light_curve(path)
    assert np.allclose(t, np.sort(original[0]))


# The start of a real AIJ measurement table's header (an ASAS target measured at BSU).
# It has no HJD column, and AIJ puts a space in front of some names.
AIJ_HEADER = (
    "Label,slice,Saturated,J.D.-2400000,JD_UTC,EXPTIME, MOONANGL, MOONPHAS, BJD_TDB, "
    "BJD_UTC,FWHM_Mean,Source_Radius,FWHM_Mult,Source_Rad(base),Sky_Rad(min),Sky_Rad("
    "max),rel_flux_T1,rel_flux_C2,rel_flux_C3,rel_flux_C4,rel_flux_C5,rel_flux_C6,rel"
    "_flux_C7,rel_flux_C8,rel_flux_err_T1,rel_flux_err_C2,rel_flux_err_C3,rel_flux_er"
    "r_C4,rel_flux_err_C5,rel_flux_err_C6,rel_flux_err_C7,rel_flux_err_C8,rel_flux_SN"
    "R_T1,rel_flux_SNR_C2,rel_flux_SNR_C3,rel_flux_SNR_C4,rel_flux_SNR_C5,rel_flux_SN"
    "R_C6,rel_flux_SNR_C7,rel_flux_SNR_C8,tot_C_cnts,tot_C_err,X(IJ)_T1,Y(IJ)_T1,X(FI"
    "TS)_T1,Y(FITS)_T1,RA_T1,DEC_T1,Source-Sky_T1,N_Src_Pixels_T1,Source_Error_T1,Sou"
    "rce_AMag_T1,Source_AMag_Err_T1,Source_SNR_T1,Peak_T1,Mean_T1,Sky/Pixel_T1,N_Sky_"
    "Pixels_T1,FWHM_T1,Width_T1,X-Width_T1,Y-Width_T1,Angle_T1,Roundness_T1,X(IJ)_C2,"
    "Y(IJ)_C2,X(FITS)_C2,Y(FITS)_C2,RA_C2,DEC_C2,Source-Sky_C2,Source_Error_C2,Source"
    "_AMag_C2,Source_AMag_Err_C2,Source_SNR_C2,Peak_C2,Mean_C2,Sky/Pixel_C2,FWHM_C2,W"
    "idth_C2,X-Width_C2,Y-Width_C2,Angle_C2,Roundness_C2"
)


def test_real_aij_header_uses_bjd_tdb(tmp_path, original):
    columns = AIJ_HEADER.split(",")
    assert light_curve_columns(columns) == [" BJD_TDB", "Source_AMag_T1", "Source_AMag_Err_T1"]
    rows = []
    for t, m, e in original.itertuples(index=False):
        row = dict.fromkeys(columns, 1.0)
        row.update({"Label": "ASAS_V_0001.fits", "Saturated": "false", "J.D.-2400000": t - 2400000.0,
                    "JD_UTC": t - 0.0008, " BJD_TDB": t, " BJD_UTC": t - 0.0008,
                    "Source_AMag_T1": m, "Source_AMag_Err_T1": e})
        rows.append(row)
    path = tmp_path / "ASAS J221652+2229.6_Table_V.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    assert _same(io.importFile_pd(str(path)), original)
    assert len(find_min.read_light_curve(path)[0]) == len(original)
