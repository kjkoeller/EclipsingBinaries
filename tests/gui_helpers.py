# -*- coding: utf-8 -*-
"""
Opening the main window in tests.

On Windows, Tcl now and then fails to read one of its own startup files
when one process creates many Tk windows in a row. It has shown up as
init.tcl ("couldn't read file ... No error") and as tk.tcl, when a Tk
message file couldn't be read. It's a passing fault in Tcl, not the
app, so a fresh attempt is made a few times, with a longer pause each
time, before giving up.
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
