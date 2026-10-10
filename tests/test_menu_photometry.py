# -*- coding: utf-8 -*-
"""
Tests for the Multi-Aperture Photometry panel's filter choices.

These open the real Tk window, so they're skipped where tkinter or a display
isn't available (for example a headless Linux machine without Xvfb).
"""

import time
from unittest.mock import patch

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app():
    import matplotlib
    matplotlib.use("Agg")
    from EclipsingBinaries import menu

    try:
        window = menu.ProgramLauncher()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    window.withdraw()
    window.show_multi_aperture_photometry()
    yield window
    window._task_running = False
    window.destroy()


def _pump(window, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        window.update()
        time.sleep(0.02)


def _set_count(window, count):
    window.photometry_filter_count.set(count)
    window._update_photometry_filter_rows()
    window.update()


def _visible_rows(window):
    window.deiconify()
    window.update()
    rows = [i for i, (_p, entry) in enumerate(window.photometry_filter_rows)
            if entry._field_widgets[1].winfo_ismapped()]
    window.withdraw()
    return rows


def _fill_and_run(window, paths):
    frames = [w for w in window.right_frame.winfo_children() if w.winfo_class() == "Frame"]
    window._set_entry_text(frames[0].winfo_children()[0], "T")
    window._set_entry_text(frames[1].winfo_children()[0], "/data/reduced")
    for (_picker, entry), path in zip(window.photometry_filter_rows, paths):
        window._set_entry_text(entry, path)
    with patch("EclipsingBinaries.menu.multi_ap") as multi_ap:
        window.log_area.delete("1.0", "end")
        window._run_button.invoke()
        _pump(window)
    return multi_ap, window.log_area.get("1.0", "end")


def test_defaults_to_three_filters(app):
    assert app.photometry_filter_count.get() == 3
    assert [p.get() for p, _ in app.photometry_filter_rows] == ["B", "V", "R"]
    assert _visible_rows(app) == [0, 1, 2]


@pytest.mark.parametrize("count, expected", [(1, ["V"]), (2, ["B", "V"]), (3, ["B", "V", "R"])])
def test_count_shows_that_many_rows_with_usual_filters(app, count, expected):
    _set_count(app, count)
    assert _visible_rows(app) == list(range(count))
    assert [p.get() for p, _ in app.photometry_filter_rows[:count]] == expected


def test_single_v_filter_passes_only_v_to_photometry(app):
    _set_count(app, 1)
    multi_ap, _ = _fill_and_run(app, ["/data/V.radec"])
    assert multi_ap.call_args.kwargs["radec_list"] == [None, "/data/V.radec", None]


def test_two_filters_keep_b_v_r_order(app):
    _set_count(app, 2)
    app.photometry_filter_rows[0][0].set("R")
    app.photometry_filter_rows[1][0].set("B")
    app._photometry_filters_picked = True
    multi_ap, _ = _fill_and_run(app, ["/data/R.radec", "/data/B.radec"])
    assert multi_ap.call_args.kwargs["radec_list"] == ["/data/B.radec", None, "/data/R.radec"]


def test_hand_picked_filter_survives_count_changes(app):
    _set_count(app, 1)
    picker, entry = app.photometry_filter_rows[0]
    picker.set("R")
    picker.event_generate("<<ComboboxSelected>>")
    app.update()
    assert entry._placeholder_text.endswith("R.radec")
    _set_count(app, 3)
    _set_count(app, 1)
    assert picker.get() == "R"


def test_typed_paths_survive_hiding_a_row(app):
    _set_count(app, 3)
    app._set_entry_text(app.photometry_filter_rows[2][1], "/data/R.radec")
    _set_count(app, 1)
    _set_count(app, 3)
    assert app._path_value(app.photometry_filter_rows[2][1]) == "/data/R.radec"


def test_same_filter_twice_is_refused(app):
    _set_count(app, 2)
    for picker, _ in app.photometry_filter_rows[:2]:
        picker.set("V")
    app._photometry_filters_picked = True
    multi_ap, log = _fill_and_run(app, ["/data/a.radec", "/data/b.radec"])
    assert not multi_ap.called
    assert "Each filter can only be chosen once" in log


def test_missing_radec_for_a_visible_row_is_refused(app):
    _set_count(app, 2)
    multi_ap, log = _fill_and_run(app, ["/data/B.radec"])
    assert not multi_ap.called
    assert "A RADEC file is required for V" in log
