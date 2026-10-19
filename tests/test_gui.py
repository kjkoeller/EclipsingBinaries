# -*- coding: utf-8 -*-
"""
Tests for the GUI's shared pieces: remembered settings, the status bar, result
buttons, the coloured log, sidebar navigation and keyboard shortcuts.

The helpers in gui_support that don't need a window are tested everywhere; the
rest open the real Tk window and are skipped where no display is available.
"""

import threading
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from EclipsingBinaries.gui_support import SettingsStore, classify_log_line, open_path  # noqa: E402

from .gui_helpers import open_launcher  # noqa: E402


# ----------------------------------------------------------------------
# Helpers that need no window
# ----------------------------------------------------------------------
def test_settings_round_trip(tmp_path):
    path = tmp_path / "sub" / "settings.json"
    store = SettingsStore(path)
    store.update_section("iraf", {"Raw Images Path": "/data/raw", "Location": "", "Use Dark Frames": True})
    store.set("last_panel", "iraf")
    store.save()

    again = SettingsStore(path)
    assert again.get("last_panel") == "iraf"
    assert again.section("iraf") == {"Raw Images Path": "/data/raw", "Use Dark Frames": True}
    assert again.section("missing") == {}


def test_settings_survive_a_damaged_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    store = SettingsStore(path)
    assert store.data == {}
    store.set("last_panel", "gaia")
    store.save()
    assert SettingsStore(path).get("last_panel") == "gaia"


def test_settings_clear_removes_the_file(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.set("last_panel", "oc")
    store.save()
    store.clear()
    assert not path.exists()
    assert store.data == {}


@pytest.mark.parametrize("line, kind", [
    ("An error occurred in Multi-Aperture Photometry: boom", "error"),
    ("Traceback (most recent call last):", "error"),
    ("Warning: no RDNOISE keyword, using 10 e-", "warning"),
    ("[REMOVED] Comparison star #3 is off the image", "warning"),
    ("Task cancellation requested...", "warning"),
    ("Multi-Aperture Photometry completed successfully.", "success"),
    ("Saved light curve data for V to /data/T_V_data.csv", "success"),
    ("Processing 3 images for V filter.", None),
])
def test_log_lines_are_classified(line, kind):
    assert classify_log_line(line) == kind


def test_open_path_refuses_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_path(tmp_path / "nope")


# ----------------------------------------------------------------------
# The real window
# ----------------------------------------------------------------------
@pytest.fixture
def settings_path(tmp_path):
    return tmp_path / "gui_settings.json"


@pytest.fixture
def make_app(settings_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from EclipsingBinaries import menu

    monkeypatch.setattr(menu, "SETTINGS_PATH", settings_path)
    windows = []

    def factory():
        window = open_launcher(menu)
        window.withdraw()
        windows.append(window)
        return window

    yield factory
    for window in windows:
        window._task_running = False
        try:
            window.destroy()
        except tk.TclError:
            pass


@pytest.fixture
def app(make_app):
    return make_app()


def _pump(window, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        window.update()
        time.sleep(0.02)


def _wait_for_task(window, timeout=5):
    end = time.time() + timeout
    while window._task_running and time.time() < end:
        window.update()
        time.sleep(0.02)
    window.update()


def test_opens_on_iraf_the_first_time(app):
    assert app._current_panel == "iraf"
    assert app._status_label.cget("text") == "Ready"


def test_every_enabled_sidebar_item_opens_its_panel(app):
    for key in app._panel_builders:
        app.open_panel(key)
        app.update()
        assert app._current_panel == key
        assert app._run_button is not None


def test_every_sidebar_item_has_a_panel(app):
    assert set(app._nav_items) == set(app._panel_builders)
    assert {"night_filters", "find_min"} <= set(app._panel_builders)


def test_panel_switch_leaves_no_stretched_rows(app):
    app.open_panel("iraf")
    app.open_panel("photometry")
    app.update()
    log_row = app.log_area.master.grid_info()["row"]
    cols, rows = app.right_frame.grid_size()
    stretched = [r for r in range(rows) if int(app.right_frame.grid_rowconfigure(r)["weight"])]
    assert stretched == [log_row]


def test_fields_are_remembered_across_panels_and_restarts(make_app, settings_path):
    first = make_app()
    first.open_panel("gaia")
    key, entry = next(iter(first._panel_fields.items()))
    typed = str(Path("/data/gaia_out"))  # the app keeps paths in the system's own form
    first._set_entry_text(entry, "/data/gaia_out")
    first.open_panel("oc")
    first.open_panel("gaia")
    assert first._path_value(first._panel_fields[key]) == typed
    first._save_panel_settings()

    second = make_app()
    assert second._current_panel == "gaia"
    assert second._path_value(second._panel_fields[key]) == typed


def test_photometry_filter_choices_are_remembered(make_app):
    first = make_app()
    first.open_panel("photometry")
    first.photometry_filter_count.set(1)
    first._update_photometry_filter_rows()
    first.photometry_filter_rows[0][0].set("R")
    first._photometry_filters_picked = True
    first._save_panel_settings()

    second = make_app()
    second.open_panel("photometry")
    assert second.photometry_filter_count.get() == 1
    assert second.photometry_filter_rows[0][0].get() == "R"


def test_clear_saved_settings_forgets_everything(app, settings_path):
    app.open_panel("gaia")
    key, entry = next(iter(app._panel_fields.items()))
    app._set_entry_text(entry, "/data/somewhere")
    app._save_panel_settings()
    with patch("EclipsingBinaries.menu.messagebox.askyesno", return_value=True):
        app.clear_saved_settings()
    assert app._path_value(app._panel_fields[key]) == ""
    assert app.settings.section("gaia") == {}


def test_status_bar_reports_a_finished_task(app):
    app.open_panel("gaia")
    started = threading.Event()
    release = threading.Event()

    def job():
        started.set()
        release.wait(5)

    app.run_task(job)
    assert started.wait(5)
    app.update()
    assert app._status_label.cget("text") == "Running Gaia Search"
    assert app._run_button.instate(["disabled"])
    release.set()
    _wait_for_task(app)
    assert app._status_label.cget("text").startswith("Finished Gaia Search in 0:0")
    assert app._run_button.instate(["!disabled"])


def test_status_bar_reports_an_error(app):
    app.open_panel("gaia")

    def job():
        raise ValueError("bad catalog")

    app.run_task(job)
    _wait_for_task(app)
    assert "stopped with an error" in app._status_label.cget("text")
    assert "error" in app.log_area.tag_names("1.0")


def test_status_bar_reports_a_cancel(app):
    app.open_panel("gaia")
    release = threading.Event()
    app.run_task(lambda: release.wait(5))
    app.cancel_event.set()
    release.set()
    _wait_for_task(app)
    assert app._status_label.cget("text").startswith("Canceled Gaia Search")


def test_result_buttons_appear_after_a_task(app, tmp_path):
    app.open_panel("gaia")
    opened = []

    def job():
        app.offer_results(str(tmp_path), [("Open Results", lambda: opened.append(True))])

    app.run_task(job)
    _wait_for_task(app)
    labels = [b.cget("text") for b in app._results_frame.winfo_children()]
    assert labels == ["Open Output Folder", "Open Results"]
    app._results_frame.winfo_children()[1].invoke()
    assert opened == [True]

    with patch("EclipsingBinaries.menu.open_path") as fake_open:
        app._results_frame.winfo_children()[0].invoke()
    fake_open.assert_called_once_with(str(tmp_path))


def test_result_buttons_skip_a_panel_the_user_left(app, tmp_path):
    app.open_panel("gaia")
    release = threading.Event()

    def job():
        release.wait(5)
        app.offer_results(str(tmp_path))

    app.run_task(job)
    app.open_panel("oc")
    release.set()
    _wait_for_task(app)
    assert app._results_frame.winfo_children() == []


def test_new_run_clears_old_result_buttons(app, tmp_path):
    app.open_panel("gaia")
    app.run_task(lambda: app.offer_results(str(tmp_path)))
    _wait_for_task(app)
    assert app._results_frame.winfo_children()
    release = threading.Event()
    app.run_task(lambda: release.wait(5))
    app.update()
    assert app._results_frame.winfo_children() == []
    release.set()
    _wait_for_task(app)


def test_log_lines_are_coloured(app):
    app._append_log("Processing 3 images for V filter.")
    app._append_log("Error: A RADEC file is required for V.")
    app._append_log("Multi-Aperture Photometry completed successfully.")
    assert app.log_area.tag_names("1.0") == ()
    assert "error" in app.log_area.tag_names("2.0")
    assert "success" in app.log_area.tag_names("3.0")


def test_save_and_clear_log(app, tmp_path):
    app.open_panel("gaia")
    app._append_log("first line")
    target = tmp_path / "log.txt"
    with patch("EclipsingBinaries.menu.filedialog.asksaveasfilename", return_value=str(target)) as ask:
        app._save_log()
    assert ask.call_args.kwargs["initialfile"].startswith("gaia_log_")
    assert target.read_text(encoding="utf-8").startswith("first line")
    app._clear_log()
    assert app.log_area.get("1.0", "end-1c") == ""


def test_enter_runs_the_open_panel(app):
    app.open_panel("gaia")
    entry = next(iter(app._panel_fields.values()))
    with patch.object(app._run_button, "invoke") as run:
        result = app._on_return_key(type("E", (), {"widget": entry})())
    assert result == "break"
    run.assert_called_once()


def test_enter_is_ignored_in_the_log(app):
    app.open_panel("gaia")
    with patch.object(app._run_button, "invoke") as run:
        result = app._on_return_key(type("E", (), {"widget": app.log_area})())
    assert result is None
    run.assert_not_called()


def test_escape_cancels_a_running_task(app):
    app.open_panel("gaia")
    release = threading.Event()
    app.run_task(lambda: release.wait(5))
    with patch("EclipsingBinaries.menu.messagebox.askyesno", return_value=True):
        result = app._on_escape_key(type("E", (), {"widget": app.log_area})())
    assert result == "break"
    assert app.cancel_event.is_set()
    release.set()
    _wait_for_task(app)


def test_help_and_about_open(app):
    app.open_help_window()
    app.show_about()
    app.update()
    dialogs = [w for w in app.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(dialogs) >= 2
    for dialog in dialogs:
        dialog.destroy()


def test_light_curve_viewer_plots_each_filter(app, tmp_path):
    import numpy as np
    import pandas as pd

    hjd = 2460000 + np.linspace(0, 0.3, 40)
    for filt in ("B", "V"):
        pd.DataFrame({"HJD": hjd, "BJD": hjd, "Source_AMag_T1": 12 + 0.3 * np.sin(hjd * 40),
                      "Source_AMag_T1_Error": np.full(40, 0.01)}).to_csv(tmp_path / f"T_{filt}_data.csv", index=False)
    before = {str(w) for w in app.winfo_children()}
    app._show_light_curves(str(tmp_path), "T", ["B", "V", "R"])
    app.update()
    new = [w for w in app.winfo_children() if str(w) not in before and isinstance(w, tk.Toplevel)]
    assert len(new) == 1
    assert new[0].title() == "T light curves"
    new[0].destroy()


def test_light_curve_viewer_reports_missing_files(app, tmp_path):
    app.open_panel("photometry")
    app._show_light_curves(str(tmp_path), "T", ["V"])
    app.update()
    assert "no light curve files" in app.log_area.get("1.0", "end")


# ----------------------------------------------------------------------
# Night Filters and Find Minimum
# ----------------------------------------------------------------------
from importlib.resources import files as _files  # noqa: E402

EXAMPLES = _files("EclipsingBinaries.examples")


def _aij_night(path, start):
    import pandas as pd

    pd.DataFrame({"HJD": [start + 0.01 * i for i in range(5)],
                  "Source_AMag_T1": [12.0 + 0.05 * i for i in range(5)],
                  "Source_AMag_Err_T1": [0.01] * 5}).to_csv(path, sep="\t", index=False)
    return str(path)


def test_night_filters_panel_combines_files(app, tmp_path):
    app.open_panel("night_filters")
    app.night_files_list.insert("end", _aij_night(tmp_path / "n2.dat", 2458380.6))
    app.night_files_list.insert("end", _aij_night(tmp_path / "n1.dat", 2458379.6))
    app._set_entry_text(app._panel_fields["Output Folder"], str(tmp_path / "out"))
    app._set_entry_text(app._panel_fields["Output Name"], "T_B")
    app._run_button.invoke()
    _wait_for_task(app)
    assert (tmp_path / "out" / "T_B_magnitudes.txt").exists()
    assert "Night Filters completed successfully." in app.log_area.get("1.0", "end")
    labels = [b.cget("text") for b in app._results_frame.winfo_children()]
    assert labels == ["Open Output Folder", "Open Magnitude File", "Find Minimum"]

    app._results_frame.winfo_children()[2].invoke()
    assert app._current_panel == "find_min"
    assert app.find_min_filter_count.get() == 1
    assert app._path_value(app.find_min_rows[0][1]) == str(tmp_path / "out" / "T_B_magnitudes.txt")


def test_night_filters_panel_reports_problems(app, tmp_path):
    app.open_panel("night_filters")
    app._run_button.invoke()
    assert "Add at least one night" in app.log_area.get("1.0", "end")

    bad = tmp_path / "bad.dat"
    bad.write_text("HJD\tMag\n1\t2\n")
    app.night_files_list.insert("end", str(bad))
    app._set_entry_text(app._panel_fields["Output Folder"], str(tmp_path))
    app._set_entry_text(app._panel_fields["Output Name"], "T")
    app._run_button.invoke()
    _wait_for_task(app)
    assert "bad.dat is missing the column" in app.log_area.get("1.0", "end")
    assert app._results_frame.winfo_children() == []


def _fill_find_min(app, count, folder=None):
    app.open_panel("find_min")
    app.find_min_filter_count.set(count)
    app._update_find_min_rows()
    app._set_entry_text(app._panel_fields["Object Name"], "NSVS 896797")
    for (picker, entry), filt in zip(app.find_min_rows[:count], "BVR"):
        picker.set(filt)
        app._set_entry_text(entry, str(EXAMPLES / f"test_{filt}.txt"))
    if folder:
        app._set_entry_text(app._panel_fields["Output Folder"], str(folder))
    app.find_min_curves.picked = True


def _finder(app):
    from EclipsingBinaries.find_min_window import MinimumFinderWindow

    windows = [w for w in app.winfo_children() if isinstance(w, MinimumFinderWindow)]
    return windows[-1] if windows else None


def test_find_minimum_panel_has_no_cancel(app):
    app.open_panel("find_min")
    assert app._cancel_button is None
    assert app._run_button.cget("text") == "Open Minimum Finder"


def test_find_minimum_output_folder_is_optional(app):
    app.open_panel("find_min")
    entry = app._panel_fields["Output Folder"]
    entry.event_generate("<FocusIn>")
    entry.event_generate("<FocusOut>")
    app.update()
    assert entry._field_widgets[-1].cget("text") == ""


def test_find_minimum_opens_the_window_and_saves(app, tmp_path):
    _fill_find_min(app, 3, tmp_path)
    app._run_button.invoke()
    app.update()
    window = _finder(app)
    assert window is not None
    assert len(window.nights) == 4
    assert window.minimum is not None
    assert str(window.save_button.state()) != "('disabled',)"

    window.save_minimum()
    saved = tmp_path / "NSVS_896797_BVR_min.txt"
    assert saved.exists()
    assert len(window.table.get_children()) == 1
    assert "Saved minimum HJD" in app.log_area.get("1.0", "end")

    # Fitting one filter saves to that filter's own file
    window.band_vars["R"].set(False)
    window.band_vars["V"].set(False)
    window._bands_changed()
    window.save_minimum()
    assert (tmp_path / "NSVS_896797_B_min.txt").exists()

    window.table.selection_set(window.table.get_children()[0])
    window.remove_selected()
    assert window.minimum_file().entries == []
    window.destroy()


def test_find_minimum_window_navigation_and_boundaries(app, tmp_path):
    _fill_find_min(app, 1, tmp_path)
    app._run_button.invoke()
    app.update()
    window = _finder(app)
    assert window.current == 0
    window.step(1)
    assert window.current == 1
    window.step(-5)
    assert window.current == 0

    lb, rb = window.boundaries[0]
    window.set_boundary("right", lb - 0.01)  # past the left one: they swap
    assert window.boundaries[0] == (lb - 0.01, lb)
    window.set_boundary("left", lb - 0.0105)
    window.set_boundary("right", lb - 0.0104)  # far too narrow
    assert window.minimum is None
    assert "Widen them" in window.result_var.get()
    assert window.save_button.instate(["disabled"])

    for var in window.band_vars.values():
        var.set(False)
    window._bands_changed()
    assert "Tick at least one filter" in window.result_var.get()
    window.destroy()


def test_find_minimum_window_keys(app, tmp_path):
    _fill_find_min(app, 1, tmp_path)
    app._run_button.invoke()
    app.update()
    window = _finder(app)

    class Event:
        def __init__(self, key, x=None, inaxes=None):
            self.key, self.xdata, self.inaxes = key, x, inaxes

    window._on_key(Event("n"))
    assert window.current == 1
    window._on_key(Event("p"))
    assert window.current == 0
    lb, rb = window.boundaries[0]
    window._on_key(Event("a", lb + 0.001, window.ax_lc))
    assert window.boundaries[0][0] == pytest.approx(lb + 0.001)
    window._on_key(Event("enter"))
    assert window.minimum_file().entries
    window.destroy()


def test_find_minimum_validation(app, tmp_path):
    app.open_panel("find_min")
    app._run_button.invoke()
    assert "Object name is required" in app.log_area.get("1.0", "end")

    _fill_find_min(app, 2)
    app.find_min_rows[1][0].set("B")
    app._run_button.invoke()
    assert "Each filter can only be chosen once" in app.log_area.get("1.0", "end")

    _fill_find_min(app, 1)
    app._set_entry_text(app.find_min_rows[0][1], str(tmp_path / "missing.txt"))
    app._run_button.invoke()
    assert "doesn't exist" in app.log_area.get("1.0", "end")
    assert _finder(app) is None


def test_find_minimum_defaults_to_the_light_curve_folder(app):
    _fill_find_min(app, 1)
    app._run_button.invoke()
    app.update()
    window = _finder(app)
    from pathlib import Path

    assert window.output_folder == Path(str(EXAMPLES / "test_B.txt")).parent
    window.destroy()


def test_window_creation_retries_the_windows_tcl_fault(monkeypatch):
    """Windows sometimes can't read init.tcl when many windows are opened in a row."""
    import types

    from . import gui_helpers

    monkeypatch.setattr(gui_helpers.time, "sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise tk.TclError("Can't find a usable init.tcl in the following directories: ...")
        return "window"

    assert gui_helpers.open_launcher(types.SimpleNamespace(ProgramLauncher=flaky)) == "window"
    assert len(calls) == 2

    def broken():
        raise tk.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setenv("EB_REQUIRE_GUI", "1")
    with pytest.raises(tk.TclError):
        gui_helpers.open_launcher(types.SimpleNamespace(ProgramLauncher=broken))


# ----------------------------------------------------------------------
# Period Search
# ----------------------------------------------------------------------
def _open_period(app, tmp_path, count=3, method="ANOVA"):
    app.open_panel("period")
    app.period_curves.count.set(count)
    app.period_curves.update()
    app.period_curves.picked = True
    app._set_entry_text(app._panel_fields["Object Name"], "NSVS 896797")
    for (picker, entry), filt in zip(app.period_curves.rows[:count], "BVR"):
        picker.set(filt)
        app._set_entry_text(entry, str(EXAMPLES / f"test_{filt}.txt"))
    app._set_entry_text(app._panel_fields["Output Folder"], str(tmp_path))
    app.period_method.set(method)
    app._run_button.invoke()
    from EclipsingBinaries.period_window import PeriodWindow

    windows = [w for w in app.winfo_children() if isinstance(w, PeriodWindow)]
    assert windows, app.log_area.get("1.0", "end")
    window = windows[-1]
    _finish_search(window)
    return window


class _PlotEvent:
    """A stand-in for a Matplotlib mouse or key event on the periodogram at a given period."""

    def __init__(self, window, period=None, key=None):
        self.key = key
        self.inaxes = window.ax_power if period is not None else None
        self.xdata = period
        self.x = window.ax_power.transData.transform((period, 0))[0] if period is not None else 0


def _finish_search(window, timeout=30):
    end = time.time() + timeout
    while window._job is not None and time.time() < end:
        window.update()
        time.sleep(0.05)
    window.update()


def test_period_search_finds_the_period_and_sends_it(app, tmp_path):
    window = _open_period(app, tmp_path)
    assert app._cancel_button is None
    assert window.period == pytest.approx(0.31741, abs=3e-4)
    assert "double the peak" in window.status_var.get()
    assert window.save_button.instate(["!disabled"])

    window.save()
    saved = tmp_path / "NSVS_896797_period.txt"
    assert saved.exists()
    from EclipsingBinaries.period import read_period

    assert read_period(saved) == pytest.approx(window.period, abs=1e-7)
    assert "epoch" not in saved.read_text()

    window.send_to("oc")
    assert app._current_panel == "oc"
    assert float(app._entry_value(app._panel_fields["Period (days)"])) == pytest.approx(window.period, abs=1e-6)
    assert app._entry_value(app._panel_fields["Epoch (T0)"]) == ""   # only the period is sent
    window.send_to("oconnell")
    assert app._entry_value(app._panel_fields["Period"]) == f"{window.period:.7f}"
    assert app._entry_value(app._panel_fields["HJD"]) == ""
    window.send_to("color")
    assert app._entry_value(app._panel_fields["Period (days)"]) == f"{window.period:.7f}"
    window.destroy()


def test_period_window_controls(app, tmp_path):
    window = _open_period(app, tmp_path, count=1)
    period = window.period
    window.set_period(period * 2)
    assert window.period == pytest.approx(2 * period)
    window.set_period(period)
    assert not hasattr(window, "epoch")

    window.period_var.set("0.3175")
    window._typed_period()
    assert window.period == pytest.approx(0.3175)
    window.refine_here()
    assert window.period == pytest.approx(0.31741, abs=3e-4)

    before = window.period
    window._on_key(_PlotEvent(window, key="right"))
    assert window.period > before

    # A click without dragging jumps to the nearest peak's top
    click = _PlotEvent(window, 0.1587)
    window._on_press(click)
    window._on_release(click)
    assert window.period == pytest.approx(0.15871, abs=2e-4)
    window.destroy()


def test_period_window_reports_problems(app, tmp_path):
    window = _open_period(app, tmp_path)
    window.max_var.set("500")
    window.start_search()
    _finish_search(window)
    assert "longer than the observations" in window.status_var.get()
    window.min_var.set("abc")
    window.start_search()
    assert "two numbers" in window.status_var.get()
    for var in window.band_vars.values():
        var.set(False)
    window.min_var.set("0.1")
    window.max_var.set("1")
    window.start_search()
    assert "Tick at least one filter" in window.status_var.get()
    window.destroy()


@pytest.mark.parametrize("method", ["Multi-harmonic", "Lomb-Scargle"])
def test_period_search_other_methods(app, tmp_path, method):
    window = _open_period(app, tmp_path, count=1, method=method)
    assert window.period == pytest.approx(0.31741, abs=5e-4)
    window.destroy()


def test_period_search_validation(app, tmp_path):
    app.open_panel("period")
    app._run_button.invoke()
    assert "Object name is required" in app.log_area.get("1.0", "end")
    app._set_entry_text(app._panel_fields["Object Name"], "T")
    app._set_entry_text(app._panel_fields["Shortest Period (d)"], "x")
    app._run_button.invoke()
    assert "must be numbers" in app.log_area.get("1.0", "end")


def test_dragging_the_period_refolds_live(app, tmp_path):
    window = _open_period(app, tmp_path, count=1)
    start = window.period
    line = window._fold_lines["B"]
    before = np.array(line.get_xdata())

    window._on_press(_PlotEvent(window, start))
    window._on_motion(_PlotEvent(window, start + 0.0005))   # a little wobble isn't a drag yet
    window._on_motion(_PlotEvent(window, 0.40))
    assert window.period == pytest.approx(0.40, abs=1e-3)
    assert window.period_var.get() == f"{window.period:.7f}"
    n = int((window.curve.label == "B").sum())   # the points themselves; copies either side follow them
    assert not np.allclose(np.array(line.get_xdata())[:n], before[:n])   # the fold followed
    x, y = window._marker_dot.get_data()
    assert x[0] == pytest.approx(window.period)
    assert y[0] == pytest.approx(window.value_at(window.period))   # the marker rides the curve
    assert window.value_var.get().startswith("F statistic here:")
    assert window.slider_var.get() == pytest.approx(window.period)

    window._on_release(_PlotEvent(window, 0.40))
    assert window.period == pytest.approx(0.40, abs=1e-3)   # releasing a drag doesn't snap
    window.destroy()


def test_slider_moves_the_period_and_follows_the_zoom(app, tmp_path):
    window = _open_period(app, tmp_path, count=1)
    window._on_slider("0.25")
    assert window.period == pytest.approx(0.25)
    assert window._marker_line.get_xdata()[0] == pytest.approx(0.25)

    window.ax_power.set_xlim(0.31, 0.32)
    assert float(window.slider.cget("from")) == pytest.approx(0.31)
    assert float(window.slider.cget("to")) == pytest.approx(0.32)
    window._on_slider("0.5")                  # outside the zoomed range: held to its edge
    assert window.period == pytest.approx(0.32)

    window._syncing = True                    # the program moving the slider doesn't feed back
    window._on_slider("0.315")
    assert window.period == pytest.approx(0.32)
    window.destroy()


# ----------------------------------------------------------------------
# Sidebar fit
# ----------------------------------------------------------------------
def _settle(window, seconds=0.6):
    end = time.time() + seconds
    while time.time() < end:
        window.update()
        time.sleep(0.03)


def _nav_fits(app):
    right_edge = app.left_frame.winfo_rootx() + app.sidebar_width
    return all(item["label"].winfo_rootx() + item["label"].winfo_width() <= right_edge
               for item in app._nav_items.values())


def test_sidebar_names_fit(app):
    app.deiconify()
    app.geometry("1280x860+0+0")
    _settle(app)
    assert _nav_fits(app)
    room = app._nav_frame.winfo_width() - 34   # the bar, padding and focus ring beside the text
    label = app._nav_items["aij"]["label"]
    if app.nav_font.measure(label.cget("text")) <= room:
        # Fits, so it's one line: as tall as a short name that can't wrap. (Comparing with
        # the text height alone breaks on macOS, where the padding and border add more.)
        assert label.winfo_height() == app._nav_items["gaia"]["label"].winfo_height()
    scroller = app._nav_scroller
    if scroller.content.winfo_reqheight() <= scroller.canvas.winfo_height():
        assert not scroller.scrollbar.winfo_ismapped()   # (CI screens can be too short for the list)


def test_sidebar_wraps_and_scrolls_with_large_text(app):
    """Large text, as on a scaled Windows display, wraps rather than being cut off."""
    app.deiconify()
    app.geometry("1280x700+0+0")
    app.nav_font.configure(size=18)
    _settle(app)
    assert _nav_fits(app)
    label = app._nav_items["aij"]["label"]
    one_line = app._nav_items["gaia"]["label"].winfo_height()
    assert label.winfo_height() >= one_line + app.nav_font.metrics("linespace") - 1  # wrapped onto two lines
    assert app._nav_scroller.scrollbar.winfo_ismapped()                  # the list scrolls


def test_sidebar_width_follows_display_scaling(app, monkeypatch):
    monkeypatch.setattr(app, "winfo_fpixels", lambda _unit: 144.0)   # a 150% display
    assert app._scaled(250) == 375
    monkeypatch.setattr(app, "winfo_fpixels", lambda _unit: 96.0)
    assert app._scaled(250) == 250


def test_scrollbar_goes_away_when_contents_shrink(app):
    """A tall panel followed by a short one used to leave the scrollbar up."""
    from EclipsingBinaries.gui_support import ScrollableFrame

    app.deiconify()
    holder = tk.Toplevel(app)
    holder.geometry("300x200+0+0")
    scroller = ScrollableFrame(holder, bg="white")
    scroller.pack(fill="both", expand=True)
    tall = tk.Frame(scroller.content, height=800, width=100)
    tall.pack()
    _settle(app)
    assert scroller.scrollbar.winfo_ismapped()
    tall.destroy()
    tk.Frame(scroller.content, height=50, width=100).pack()
    _settle(app, 3 * ScrollableFrame.WATCH_MS / 1000)
    assert not scroller.scrollbar.winfo_ismapped()
    holder.destroy()
