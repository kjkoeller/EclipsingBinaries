# -*- coding: utf-8 -*-
"""
Opening the main window in tests.

On Windows, Tcl now and then fails to read its own init.tcl when one
process creates many Tk windows in a row ("couldn't read file ... No
error"). It's a passing fault in Tcl, not the app, so a fresh attempt is
made a couple of times before giving up.
"""

import gc
import os
import time

import pytest

tk = pytest.importorskip("tkinter")

ATTEMPTS = 3


def open_launcher(menu):
    """Create a ProgramLauncher, or skip the test when there's no display (fail on CI)."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return menu.ProgramLauncher()
        except tk.TclError as exc:
            if "init.tcl" in str(exc) and attempt < ATTEMPTS:
                gc.collect()  # let windows from earlier tests release their interpreters
                time.sleep(0.5 * attempt)
                continue
            if os.environ.get("EB_REQUIRE_GUI"):
                raise  # CI sets this so a broken display fails instead of skipping
            pytest.skip(f"No display available: {exc}")
