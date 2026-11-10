# -*- coding: utf-8 -*-
"""
Opening and closing the main window in tests.

On Windows, Tcl now and then fails to read one of its own startup files
when one process creates many Tk windows in a row. It has shown up as
init.tcl ("couldn't read file ... No error") and as tk.tcl, when a Tk
message file couldn't be read. It's a passing fault in Tcl, not the
app, so a fresh attempt is made a few times, with a longer pause each
time, before giving up.

Closing a window has a catch of its own. A closed window leaves tkinter
variables and images behind until Python's garbage collector frees them.
If that happens on a task's worker thread, tkinter waits about a second
for each one, because the tests drive the window with update() rather
than a main loop. With enough of them a short task overran its time
limit (Night Filters on CI). close_launcher frees them on the main
thread straight away instead.
"""

import gc
import os
import time

import pytest

tk = pytest.importorskip("tkinter")

ATTEMPTS = 4

# Wording Tcl uses when it can't read the files it starts up from
STARTUP_FILE_ERRORS = ("Can't find a usable", "couldn't read file")


def is_startup_file_error(exc):
    """True for Tcl's passing failures to read its own startup files."""
    return any(text in str(exc) for text in STARTUP_FILE_ERRORS)


def open_launcher(menu):
    """Create a ProgramLauncher, or skip the test when there's no display (fail on CI)."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return menu.ProgramLauncher()
        except tk.TclError as exc:
            if is_startup_file_error(exc) and attempt < ATTEMPTS:
                gc.collect()  # let windows from earlier tests release their interpreters
                time.sleep(0.5 * attempt)
                continue
            if os.environ.get("EB_REQUIRE_GUI"):
                raise  # CI sets this so a broken display fails instead of skipping
            pytest.skip(f"No display available: {exc}")


def close_launcher(window):
    """Close a window opened with open_launcher and free what it leaves behind."""
    window._task_running = False
    try:
        window.destroy()
    except tk.TclError:
        pass
    gc.collect()
