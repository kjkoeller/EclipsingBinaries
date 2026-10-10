# -*- coding: utf-8 -*-
"""
Tests for OC_plot.py

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import Polynomial

from EclipsingBinaries.OC_plot import calculate_oc, data_fit, residuals


# ===========================================================================
# calculate_oc
# ===========================================================================
def test_calculate_oc_t0_zero_uses_first_tom():
    # When T0=0 the first ToM becomes T0, so e=0.5 and OC is negative
    e, OC, OC_err, _, _ = calculate_oc(10, 0.1, 0, 0.1, 1)
    assert e == 0.5
    assert OC == "-0.50000"
    assert OC_err == "0.14142"


def test_calculate_oc_exact_multiple_of_period():
    # 15 - 5 = 10, 10 / 2.0 = 5.0 exactly → OC should be 0
    e, OC, OC_err, _, _ = calculate_oc(15.0, 0.1, 5.0, 0.1, 2.0)
    assert e == 5.0
    assert OC == "0.00000"
    assert OC_err == "0.14142"


def test_calculate_oc_real_tess_value():
    # Line 13 of test_minimums.txt in the examples folder
    e, OC, OC_err, _, _ = calculate_oc(2458843.932122, 0.00058, 2457143.761819, 0.00014, 0.31297)
    assert e == 5432.5
    assert OC == "-0.03922"
    assert OC_err == "0.00060"


def test_calculate_oc_returns_updated_t0_on_first_call():
    # When T0=0, returned T0 should equal the input ToM
    _, _, _, T0_out, T0_err_out = calculate_oc(2457000.0, 0.001, 0, 0, 0.5)
    assert T0_out == pytest.approx(2457000.0)
    assert T0_err_out == pytest.approx(0.001)


def test_calculate_oc_t0_unchanged_on_subsequent_call():
    # When T0 is already set it should not be overwritten
    _, _, _, T0_out, _ = calculate_oc(2457001.0, 0.001, 2457000.0, 0.001, 0.5)
    assert T0_out == pytest.approx(2457000.0)


def test_calculate_oc_primary_eclipse_is_integer():
    # Primary eclipse → e should be a whole number
    e, _, _, _, _ = calculate_oc(15.0, 0.1, 5.0, 0.1, 2.0)
    assert e % 1 == 0.0


def test_calculate_oc_secondary_eclipse_is_half_integer():
    # ToM halfway between two primaries → secondary eclipse at x.5
    e, _, _, _, _ = calculate_oc(10, 0.1, 0, 0.1, 1)
    assert e % 1 == pytest.approx(0.5)


def test_calculate_oc_oc_error_propagates_correctly():
    # OC_err = sqrt(T0_err^2 + err^2) = sqrt(0.0003^2 + 0.0004^2) = 0.0005
    import math
    _, _, OC_err, _, _ = calculate_oc(2457001.0, 0.0004, 2457000.0, 0.0003, 0.5)
    expected = "%.5f" % math.sqrt(0.0003 ** 2 + 0.0004 ** 2)
    assert OC_err == expected


def test_calculate_oc_negative_epoch():
    # ToM before T0 should give a negative epoch number
    e, _, _, _, _ = calculate_oc(2456999.0, 0.001, 2457000.0, 0.001, 0.5)
    assert e < 0


def test_calculate_oc_returns_string_oc_values():
    # OC and OC_err should be formatted strings, not floats
    _, OC, OC_err, _, _ = calculate_oc(15.0, 0.1, 5.0, 0.1, 2.0)
    assert isinstance(OC, str)
    assert isinstance(OC_err, str)


def test_calculate_oc_five_decimal_places():
    # Output strings should always have exactly 5 decimal places
    _, OC, OC_err, _, _ = calculate_oc(15.0, 0.1, 5.0, 0.1, 2.0)
    assert len(OC.split(".")[-1]) == 5
    assert len(OC_err.split(".")[-1]) == 5


# ===========================================================================
# data_fit and residuals
# ===========================================================================
def _write_oc_table(path, slope=2e-5):
    # Primaries on whole epochs, secondaries on half epochs, with a small
    # linear drift so the fit has a period correction to find
    rng = np.random.default_rng(0)
    epochs = np.arange(0, 200, 0.5)
    oc = slope * epochs + rng.normal(0, 1e-5, epochs.size)
    pd.DataFrame({"Epoch": epochs, "O-C": oc, "O-C_Error": np.full(epochs.size, 1e-4)}).to_csv(
        path, sep="\t", index=False)


def test_data_fit_writes_plot_and_tables(tmp_path):
    table = tmp_path / "oc.txt"
    _write_oc_table(table)

    new_period = data_fit(str(table), 0.3, write_callback=lambda _m: None)

    # The linear term of the fit is the period correction
    assert new_period == pytest.approx(0.3 + 2e-5, abs=2e-6)
    assert (tmp_path / "oc.png").stat().st_size > 0
    assert "tabular" in (tmp_path / "oc.tex").read_text()


def test_plotting_leaves_no_pyplot_figures_open(tmp_path):
    # Plots are built on Figure objects, so nothing piles up in pyplot's
    # global figure list when tasks run back to back
    plt.close("all")
    table = tmp_path / "oc.txt"
    _write_oc_table(table)
    data_fit(str(table), 0.3, write_callback=lambda _m: None)

    x = np.arange(10.0)
    model = Polynomial([0.0, 1.0])
    residuals(x, x + 0.01, "Epoch", "O-C", 1, model, x,
              output_path=str(tmp_path / "resid.png"), write_callback=lambda _m: None)

    assert (tmp_path / "resid.png").stat().st_size > 0
    assert plt.get_fignums() == []
