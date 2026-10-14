"""
A GUI to centralize all the scripts and capabilities of this package for ease of use for the user and
making it more convenient to use and access than a command line or individual scripts.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 10/10/2026
"""

import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
except ImportError as exc:
    raise ImportError(
        "The EclipsingBinaries GUI needs tkinter, which is missing from this Python install. "
        "Homebrew Python on macOS: run 'brew install python-tk@3.12' (match your Python version). "
        "Ubuntu/Debian: run 'sudo apt install python3-tk'. "
        "The python.org installers for macOS and Windows already include it."
    ) from exc

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astroquery.mast import Tesscut
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from photutils.aperture import CircularAperture, CircularAnnulus

from .IRAF_Reduction import run_reduction, site_config
from .tess_data_search import run_tess_search
from .apass import comparison_selector
from .multi_aperture_photometry import (main as multi_ap, auto_optimize_radii, calculate_target_snr,
                                        DEFAULT_READ_NOISE, read_noise_from_header)
from .gaia import target_star as gaia
from .OC_plot import BSUO, TESS_OC, all_data, data_fit
from .OConnell import main as oconnell
from .color_light_curve import color_plot
from .find_min import MinimumError
from .find_min_window import MinimumFinderWindow
from .Night_Filters import combine_nights, NightFileError
from .gui_support import (ToolTip, SettingsStore, ScrollableFrame, make_app_icon, open_path,
                          classify_log_line)

try:
    __version__ = version("EclipsingBinaries")
except PackageNotFoundError:
    __version__ = "unknown"

IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# The GUI always draws with a light palette. Every color is set explicitly
# because on macOS in Dark Mode Tk's default text color turns white, which
# made labels and entry text vanish against our white backgrounds.
BG_APP = "#f3f5f8"        # behind the panel
BG_PANEL = "#ffffff"
BG_STATUS = "#e8ecf2"
BORDER = "#cfd6df"
FG_TEXT = "#1b2430"
FG_MUTED = "#5b6675"
FG_PLACEHOLDER = "#8a94a3"
FG_ERROR = "#b3261e"
BG_ERROR = "#fdecea"
FG_WARNING = "#8a5a00"
FG_SUCCESS = "#1e6b3a"
ACCENT = "#003366"        # buttons
ACCENT_ACTIVE = "#00509e"
ACCENT_DISABLED = "#8a9bb0"
# Header and sidebar: a night-sky navy, with starlight marking the open panel
NIGHT = "#0e1f3a"
NIGHT_HOVER = "#17305a"
NIGHT_ACTIVE = "#1d3a66"
NIGHT_TEXT = "#e6ecf5"
NIGHT_MUTED = "#8fa3bf"
NIGHT_DISABLED = "#5d6f89"
STARLIGHT = "#f2c14e"

DOCS_URL = "https://eclipsingbinaries.readthedocs.io/"
REPO_URL = "https://github.com/kjkoeller/EclipsingBinaries/"

# Where the GUI remembers paths and settings between sessions. The environment
# variable lets tests and scripts point it somewhere else.
SETTINGS_PATH = Path(os.environ.get("ECLIPSINGBINARIES_SETTINGS",
                                    Path.home() / ".EclipsingBinaries" / "gui_settings.json"))

# Same constant tkinterdnd2 uses for file drops. Kept here so the module
# still imports when tkinterdnd2 can't load.
DND_FILES = "DND_Files"

# Filters the photometry panel offers, in the order photometry's main expects them
PHOTOMETRY_FILTERS = ("B", "V", "R")
# Filters suggested for each number of filters until the user picks their own
PHOTOMETRY_DEFAULTS = {1: ("V",), 2: ("B", "V"), 3: ("B", "V", "R")}

# How often the main thread checks for log lines and other updates queued
# by worker threads, in milliseconds.
UI_POLL_MS = 50


def _example_path(*parts):
    """Build placeholder text for a path field that looks native on the current OS."""
    if IS_WINDOWS:
        return "C:\\Data\\" + "\\".join(parts)
    return "~/Data/" + "/".join(parts)


def _is_float(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _enable_windows_dpi_awareness():
    """
    Tell Windows the app handles its own scaling so text isn't drawn at low
    resolution and stretched on high-DPI displays. Must run before Tk starts.
    """
    if not IS_WINDOWS:
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class ProgramLauncher(tk.Tk):
    """
    Main GUI that the user interacts with for each of the different options for analysis.

    All Tk calls happen on the main thread. Long jobs run on a worker thread and
    talk back to the window through call_in_ui(), write_to_log() and notify(),
    which queue the work for the main loop to pick up. Calling Tk directly from
    a worker thread is what caused the random crashes on macOS.
    """

    def __init__(self):
        super().__init__()

        self._ui_thread_id = threading.get_ident()
        self._ui_queue = queue.Queue()
        self._ui_poll_id = None

        self.cancel_event = threading.Event()
        self.current_task = None
        self._task_running = False
        self._run_button = None
        self._cancel_button = None
        self._results_frame = None
        self._action_buttons = []
        self.log_area = None

        # What the current panel is, and the fields and settings it remembers
        self._current_panel = None
        self._current_panel_title = ""
        self._panel_fields = {}
        self._panel_vars = {}
        self.settings = SettingsStore(SETTINGS_PATH)

        # Bookkeeping for the status bar
        self._task_title = ""
        self._task_panel = None
        self._task_started = 0.0
        self._task_had_error = False
        self._clock_id = None

        self.dnd_enabled = self._try_enable_drag_and_drop()

        # Window size: most of the screen, but not sprawling on very large displays
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.window_width = min(int(self.screen_width * 0.75), 1500)
        self.window_height = min(int(self.screen_height * 0.8), 1000)
        self.minsize(980, 640)
        self.center_window()

        self.title("EclipsingBinaries")
        self.configure(bg=BG_APP)
        self._apply_color_defaults()
        self._setup_fonts()
        self._setup_styles()

        self._icon = make_app_icon(self, 64)
        self._header_icon = self._icon.subsample(2, 2)
        self.iconphoto(True, self._icon)

        # Bind click event to remove focus
        self.bind("<Button-1>", self.remove_focus)

        self.create_header()
        self.create_status_bar()
        self.create_layout()
        self.create_menu()
        self._bind_shortcuts()

        # Closing the window and Cmd-Q on macOS both go through the exit prompt
        self.protocol("WM_DELETE_WINDOW", self.quit_program)
        if IS_MAC:
            self.createcommand("tk::mac::Quit", self.quit_program)

        self._ui_poll_id = self.after(UI_POLL_MS, self._drain_ui_queue)

        # Pick up where the last session left off
        last_panel = self.settings.get("last_panel")
        self.open_panel(last_panel if last_panel in self._panel_builders else "iraf")

    # ------------------------------------------------------------------
    # Platform setup
    # ------------------------------------------------------------------
    def _try_enable_drag_and_drop(self):
        """
        Load the tkdnd extension if possible. When it can't load (older
        tkinterdnd2 with Tk 9, an Intel Mac on Tk 9, or tkinterdnd2 missing)
        the app keeps running and the Browse buttons still work.
        """
        try:
            from tkinterdnd2 import TkinterDnD
            require = getattr(TkinterDnD, "require", None) or TkinterDnD._require
            require(self)
            return True
        except Exception as exc:
            print(f"Drag and drop is unavailable ({exc}). Use the Browse buttons instead.")
            return False

    def _apply_color_defaults(self):
        """Give every classic Tk widget explicit light-theme colors."""
        for widget_class in ("Label", "Checkbutton", "Radiobutton", "Entry", "Text", "Scale", "Listbox"):
            self.option_add(f"*{widget_class}.Foreground", FG_TEXT)
        for widget_class in ("Label", "Checkbutton", "Radiobutton", "Scale", "Frame", "Toplevel"):
            self.option_add(f"*{widget_class}.Background", BG_PANEL)
        for widget_class in ("Entry", "Text", "Listbox"):
            self.option_add(f"*{widget_class}.Background", "white")
            self.option_add(f"*{widget_class}.selectBackground", "#b3d4fc")
            self.option_add(f"*{widget_class}.selectForeground", FG_TEXT)
        self.option_add("*Entry.insertBackground", FG_TEXT)
        self.option_add("*Text.insertBackground", FG_TEXT)
        self.option_add("*Checkbutton.selectColor", "white")
        self.option_add("*Radiobutton.selectColor", "white")
        self.option_add("*Checkbutton.activeForeground", FG_TEXT)
        self.option_add("*Radiobutton.activeForeground", FG_TEXT)
        self.option_add("*Checkbutton.activeBackground", BG_PANEL)
        self.option_add("*Radiobutton.activeBackground", BG_PANEL)
        # No focus box drawn around every radio button and checkbox
        self.option_add("*Checkbutton.highlightThickness", 0)
        self.option_add("*Radiobutton.highlightThickness", 0)
        self.option_add("*Scale.troughColor", "#dddddd")
        self.option_add("*TCombobox*Listbox.background", "white")
        self.option_add("*TCombobox*Listbox.foreground", FG_TEXT)

    def _setup_fonts(self):
        """
        Fonts built on the operating system's own UI font, so the app looks at
        home on each platform. Sizes are in points, which Tk scales for the
        display, so text stays the same physical size on high-DPI screens.
        """
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        bump = 2 if IS_MAC else 0  # macOS point sizes read smaller than Windows and Linux

        def make(size, weight="normal"):
            return tkfont.Font(self, family=family, size=size + bump, weight=weight)

        self.label_font = make(11)
        self.small_font = make(9)
        self.button_font = make(11)
        self.nav_font = make(11)
        self.section_font = make(9, "bold")
        self.header_font = make(17, "bold")   # panel titles
        self.title_font = make(14, "bold")    # app name in the header
        self.strong_font = make(12, "bold")
        self.mono_font = tkfont.nametofont("TkFixedFont").copy()
        self.mono_font.configure(size=10 + bump)

    def _setup_styles(self):
        """
        Button styles. The clam theme is drawn by Tk itself, so background
        colors work the same on macOS, Windows and Linux. That replaces the
        tkmacosx dependency, which only existed because native macOS buttons
        ignore bg.
        """
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure(
            "Accent.TButton", font=self.button_font, padding=(14, 6),
            background=ACCENT, foreground="white", borderwidth=0, relief="flat",
            bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT, focuscolor=ACCENT,
        )
        self.style.map(
            "Accent.TButton",
            background=[("disabled", ACCENT_DISABLED), ("pressed", ACCENT_ACTIVE), ("active", ACCENT_ACTIVE)],
            foreground=[("disabled", "#e6e6e6")],
            bordercolor=[("active", ACCENT_ACTIVE)],
            lightcolor=[("active", ACCENT_ACTIVE)],
            darkcolor=[("active", ACCENT_ACTIVE)],
        )

        self.style.configure(
            "Browse.TButton", font=self.small_font, padding=(8, 2),
            background="#eef1f5", foreground=FG_TEXT, borderwidth=1,
            bordercolor=BORDER, lightcolor="#eef1f5", darkcolor="#eef1f5",
        )
        self.style.map("Browse.TButton", background=[("active", "#e1e6ed")])

        # Quieter buttons for things like Save log and Open output folder
        self.style.configure(
            "Secondary.TButton", font=self.small_font, padding=(10, 3),
            background=BG_PANEL, foreground=ACCENT, borderwidth=1,
            bordercolor=BORDER, lightcolor=BG_PANEL, darkcolor=BG_PANEL, focuscolor=ACCENT,
        )
        self.style.map("Secondary.TButton",
                       background=[("disabled", BG_PANEL), ("active", "#eef3fa")],
                       foreground=[("disabled", FG_PLACEHOLDER)],
                       bordercolor=[("active", ACCENT)])

        self.style.configure("TCombobox", fieldbackground="white", foreground=FG_TEXT)
        self.style.configure("Status.Horizontal.TProgressbar", troughcolor="#d7dee8",
                             background=ACCENT, bordercolor=BG_STATUS,
                             lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)

    # ------------------------------------------------------------------
    # Thread-safe bridge from worker threads to the window
    # ------------------------------------------------------------------
    def call_in_ui(self, func, *args, **kwargs):
        """Run func on the Tk main thread. Safe to call from any thread."""
        if threading.get_ident() == self._ui_thread_id:
            func(*args, **kwargs)
        else:
            self._ui_queue.put((func, args, kwargs))

    def _drain_ui_queue(self):
        # Cap the work per tick so a chatty task can't freeze the window
        for _ in range(500):
            try:
                func, args, kwargs = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.print_exc()
        self._ui_poll_id = self.after(UI_POLL_MS, self._drain_ui_queue)

    def write_to_log(self, message):
        """Add a line to the output log. Safe to call from worker threads."""
        self.call_in_ui(self._append_log, str(message))

    def _append_log(self, message):
        kind = classify_log_line(message)
        if kind == "error" and self._task_running:
            self._task_had_error = True
        if self.log_area is not None and self.log_area.winfo_exists():
            self.log_area.insert("end", message + "\n", (kind,) if kind else ())
            self.log_area.see("end")
        else:
            # The user switched panels mid-task, so there's nowhere to show it
            print(message)

    def notify(self, kind, title, message):
        """Show an info, warning or error dialog. Safe to call from worker threads."""
        dialog = {"info": messagebox.showinfo,
                  "warning": messagebox.showwarning,
                  "error": messagebox.showerror}[kind]
        self.call_in_ui(dialog, title, message, parent=self)

    def run_task(self, target, *args):
        """
        Run target on a worker thread. Only one task runs at a time; the Run
        button greys out and Cancel becomes active until it finishes, and the
        status bar shows what's running and for how long.
        """
        if self._task_running:
            messagebox.showinfo("Task Running",
                                "Another task is still running. Wait for it to finish or cancel it first.",
                                parent=self)
            return
        self.cancel_event.clear()
        self._task_running = True
        self._task_had_error = False
        self._task_title = self._current_panel_title
        self._task_panel = self._current_panel
        self._task_started = time.monotonic()
        self._clear_results()
        self._status_running()

        def worker():
            try:
                target(*args)
            except Exception as exc:
                self.write_to_log(f"Unexpected error: {type(exc).__name__}: {exc}")
                self.write_to_log(traceback.format_exc())
            finally:
                self.call_in_ui(self._on_task_finished)

        self.current_task = threading.Thread(target=worker, daemon=True)
        self.current_task.start()
        self._sync_task_buttons()

    def _on_task_finished(self):
        self._task_running = False
        self.current_task = None
        self._sync_task_buttons()
        self._status_finished()

    def _sync_task_buttons(self):
        busy = self._task_running
        for button in [self._run_button, *self._action_buttons]:
            if button is not None and button.winfo_exists():
                button.state(["disabled"] if busy else ["!disabled"])
        if self._cancel_button is not None and self._cancel_button.winfo_exists():
            self._cancel_button.state(["!disabled"] if busy else ["disabled"])

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def create_status_bar(self):
        bar = tk.Frame(self, bg=BG_STATUS, highlightthickness=1, highlightbackground=BORDER)
        bar.pack(side="bottom", fill="x")
        self._status_label = tk.Label(bar, text="Ready", font=self.small_font, bg=BG_STATUS,
                                      fg=FG_MUTED, anchor="w")
        self._status_label.pack(side="left", padx=12, pady=4)
        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=160,
                                         style="Status.Horizontal.TProgressbar")
        self._elapsed_label = tk.Label(bar, text="", font=self.small_font, bg=BG_STATUS, fg=FG_MUTED)

    @staticmethod
    def _format_elapsed(seconds):
        seconds = int(seconds)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    def _status_running(self):
        self._status_label.config(text=f"Running {self._task_title}", fg=FG_TEXT)
        self._progress.pack(side="right", padx=(6, 12), pady=4)
        self._elapsed_label.pack(side="right")
        self._progress.start(12)
        self._tick_clock()

    def _tick_clock(self):
        if not self._task_running:
            return
        self._elapsed_label.config(text=self._format_elapsed(time.monotonic() - self._task_started))
        self._clock_id = self.after(1000, self._tick_clock)

    def _status_finished(self):
        if self._clock_id is not None:
            self.after_cancel(self._clock_id)
            self._clock_id = None
        self._progress.stop()
        self._progress.pack_forget()
        self._elapsed_label.pack_forget()
        elapsed = self._format_elapsed(time.monotonic() - self._task_started)
        if self.cancel_event.is_set():
            text, colour = f"Canceled {self._task_title} after {elapsed}.", FG_WARNING
        elif self._task_had_error:
            text, colour = f"{self._task_title} stopped with an error after {elapsed}. See the output log.", FG_ERROR
        else:
            text, colour = f"Finished {self._task_title} in {elapsed}.", FG_SUCCESS
        self._status_label.config(text=text, fg=colour)

    # ------------------------------------------------------------------
    # Results shown after a task finishes
    # ------------------------------------------------------------------
    def offer_results(self, folder=None, extras=()):
        """
        Show buttons for a finished task's output next to Run and Cancel: one
        to open the output folder, plus any (label, function) pairs in extras.
        Safe to call from worker threads.
        """
        self.call_in_ui(self._show_results, self._task_panel, folder, list(extras))

    def _show_results(self, panel, folder, extras):
        if panel != self._current_panel or self._results_frame is None or not self._results_frame.winfo_exists():
            return  # the user moved on to another panel
        self._clear_results()
        if folder:
            self._button(self._results_frame, "Open Output Folder",
                         lambda: self._open_in_system(folder), style="Secondary.TButton").pack(side="left", padx=4)
        for label, command in extras:
            self._button(self._results_frame, label, command, style="Secondary.TButton").pack(side="left", padx=4)

    def _clear_results(self):
        if self._results_frame is not None and self._results_frame.winfo_exists():
            for child in self._results_frame.winfo_children():
                child.destroy()

    def _open_in_system(self, path):
        try:
            open_path(path)
        except Exception as exc:
            self.write_to_log(f"Error: couldn't open {path}: {exc}")

    def cancel_task(self):
        """Cancel the currently running task."""
        if self._task_running:
            if messagebox.askyesno("Cancel Task", "Are you sure you want to cancel the current task?",
                                   parent=self):
                self.write_to_log("Task cancellation requested...")
                self.cancel_event.set()
        else:
            messagebox.showinfo("No Task Running", "There is no task currently running.", parent=self)

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------
    def _button(self, parent, text, command, style="Accent.TButton", action=False):
        """
        Create a themed button. Set action=True for buttons that start work so
        they get disabled while a task is running.
        """
        button = ttk.Button(parent, text=text, command=command, style=style, cursor="hand2")
        if action:
            self._action_buttons.append(button)
            if self._task_running:
                button.state(["disabled"])
        return button

    def _entry_value(self, entry):
        """
        Text the user typed, ignoring placeholder hints. Falls back to the
        field's default when it was left empty.
        """
        text = "" if getattr(entry, "_placeholder_active", False) else entry.get().strip()
        if not text and getattr(entry, "_default", None) is not None:
            return entry._default
        return text

    def _path_value(self, entry):
        """Like _entry_value but expands a leading ~ the way a shell would."""
        text = self._entry_value(entry)
        return str(Path(text).expanduser()) if text else ""

    def _set_entry_text(self, entry, text):
        entry.delete(0, "end")
        entry.insert(0, text)
        entry._placeholder_active = False
        entry.config(fg=FG_TEXT)

    def _make_validator(self, entry, error_label, error_message, validation_func, optional=False):
        """Build and return the validate_input function for an entry field. Optional fields may be left empty."""
        def validate_input():
            value = self._entry_value(entry)
            if not value and not optional:
                error_label.config(text=error_message or "This field is required.")
                entry.config(bg=BG_ERROR)
            elif value and validation_func and not validation_func(value):
                error_label.config(text=error_message)
                entry.config(bg=BG_ERROR)
            else:
                error_label.config(text="")
                entry.config(bg="white")
        return validate_input

    def _make_placeholder_bindings(self, entry, placeholder_text, validate_input):
        """Bind focus-in, focus-out, and placeholder behaviour to an entry field."""
        entry._placeholder_text = placeholder_text

        def show_placeholder():
            entry.delete(0, "end")
            entry.insert(0, entry._placeholder_text)
            entry._placeholder_active = True
            entry.config(fg=FG_PLACEHOLDER)

        def on_focus_in(event):
            if entry._placeholder_active:
                entry.delete(0, "end")
                entry._placeholder_active = False
                entry.config(fg=FG_TEXT)

        def on_focus_out(event):
            if not entry.get().strip():
                show_placeholder()
            if not getattr(entry, "_browsing", False):  # Skip validation if browse dialog is open
                validate_input()

        show_placeholder()
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    def _make_browse_button(self, entry_frame, entry, browse_type, validate_input):
        """Add a browse button inside the entry frame if browse_type is specified."""
        if not browse_type:
            return

        def browse_action():
            if browse_type == "file":
                selected_path = filedialog.askopenfilename(title="Select File", parent=self)
            else:
                selected_path = filedialog.askdirectory(title="Select Folder", parent=self)
            entry._browsing = False
            if selected_path:
                self._set_entry_text(entry, selected_path)
                validate_input()

        browse_button = self._button(entry_frame, "Browse", browse_action, style="Browse.TButton")
        browse_button.grid(row=0, column=1, padx=(5, 5), pady=2, sticky="e")
        browse_button.bind("<ButtonPress>", lambda e: setattr(entry, "_browsing", True), add="+")

    def _make_drop_handler(self, entry, validate_input):
        """Build and return the drag-and-drop handler for an entry field."""
        def handle_drop(event):
            # tkdnd hands back a Tcl list, with braces around paths that
            # contain spaces. splitlist undoes that on every platform.
            paths = entry.tk.splitlist(event.data)
            if paths:
                self._set_entry_text(entry, paths[0])
                validate_input()
            return event.action

        return handle_drop

    def create_input_field(self, parent, label_text, placeholder_text, row,
                           validation_func=None, error_message="", browse_type=None, default=None,
                           label_widget=None, tooltip=None, wide=None, remember_as=None, optional=False):
        """
        Create a labeled input field with placeholder functionality, validation, and inline error message.
        Includes an optional browse button integrated into the entry field.

        The placeholder is only a hint and is never read back as a value. Pass
        default for fields where leaving them empty should mean a specific value.
        Pass label_widget (already created inside parent) to use it in place of a
        plain text label, for example a label with a dropdown in it.

        tooltip adds a hover note to the label and the field. Path fields stretch
        with the window; pass wide to choose for any other field. The value is
        remembered between sessions under remember_as, which defaults to the
        label text. Set optional for fields that may be left empty.
        """
        label = label_widget or tk.Label(parent, text=label_text, font=self.label_font, bg=BG_PANEL, fg=FG_TEXT)
        label.grid(row=row, column=0, padx=(16, 10), pady=5, sticky="e")

        if wide is None:
            wide = browse_type is not None
        entry_frame = tk.Frame(parent, bg=BG_PANEL)
        entry_frame.grid(row=row, column=1, padx=(0, 10), pady=5, sticky="ew" if wide else "w")
        entry_frame.columnconfigure(0, weight=1)

        # The entry draws its own border, which turns navy while it has focus
        entry = tk.Entry(entry_frame, width=30 if wide else 22, font=self.label_font, relief="flat",
                         borderwidth=0, highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT)
        entry.grid(row=0, column=0, sticky="ew", ipady=3)
        entry._default = default

        error_label = tk.Label(parent, text="", font=self.small_font, fg=FG_ERROR, bg=BG_PANEL,
                               wraplength=200, justify="left")
        error_label.grid(row=row, column=2, padx=(0, 16), sticky="w")

        # Kept so panels can hide or show the whole row
        entry._field_widgets = (label, entry_frame, error_label)

        validate_input = self._make_validator(entry, error_label, error_message, validation_func, optional)
        self._make_placeholder_bindings(entry, placeholder_text, validate_input)
        self._make_browse_button(entry_frame, entry, browse_type, validate_input)
        handler = self._make_drop_handler(entry, validate_input)
        self.enable_drag_and_drop(entry_frame, handler)
        self.enable_drag_and_drop(entry, handler)

        if tooltip:
            self._tooltip(label, tooltip)
            self._tooltip(entry, tooltip)

        key = remember_as or label_text.strip().rstrip(":")
        if key:
            self._panel_fields[key] = entry
            saved = self.settings.section(self._current_panel).get(key)
            if isinstance(saved, str) and saved:
                self._set_entry_text(entry, saved)

        return entry

    def _tooltip(self, widget, text):
        ToolTip(widget, text, font=self.small_font)

    # ------------------------------------------------------------------
    # Remembered settings
    # ------------------------------------------------------------------
    def _remember_var(self, key, var):
        """Restore a Tk variable from the last session and save it with the panel's other settings."""
        saved = self.settings.section(self._current_panel).get(key)
        if saved is not None:
            try:
                var.set(saved)
            except (tk.TclError, ValueError):
                pass
        self._panel_vars[key] = var
        return var

    def _save_panel_settings(self):
        if not self._current_panel:
            return
        values = {}
        for key, entry in self._panel_fields.items():
            try:
                if entry.winfo_exists() and not getattr(entry, "_placeholder_active", False):
                    values[key] = entry.get().strip()
            except tk.TclError:
                pass
        for key, var in self._panel_vars.items():
            try:
                values[key] = var.get()
            except tk.TclError:
                pass
        self.settings.update_section(self._current_panel, values)
        self.settings.set("last_panel", self._current_panel)
        self.settings.save()

    def clear_saved_settings(self):
        if messagebox.askyesno("Clear Saved Settings",
                               "Forget the folders, files and values remembered in every panel?",
                               parent=self):
            self.settings.clear()
            self._panel_fields = {}
            self._panel_vars = {}
            self.open_panel(self._current_panel or "iraf")
            self.write_to_log("Saved settings cleared.")

    def enable_drag_and_drop(self, widget, callback):
        """
        Enable drag-and-drop functionality for a widget. Does nothing when the
        tkdnd extension couldn't be loaded.

        :param widget: The widget to enable drag-and-drop on.
        :param callback: The function to call when a file/folder is dropped.
        """
        if not self.dnd_enabled:
            return
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", callback)

    @staticmethod
    def _set_placeholder(entry, text):
        """Change a field's placeholder hint, updating it on screen if it's showing."""
        entry._placeholder_text = text
        if getattr(entry, "_placeholder_active", False):
            entry.delete(0, "end")
            entry.insert(0, text)

    @staticmethod
    def _set_field_visible(entry, visible):
        for widget in entry._field_widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def remove_focus(self, event):
        """Remove focus from the currently focused widget unless it's an input widget."""
        clicked_widget = self.winfo_containing(event.x_root, event.y_root)

        # Reset focus only if the clicked widget is not an input widget
        if not isinstance(clicked_widget, (tk.Entry, tk.Text, ttk.Combobox)):
            self.focus_set()

    def center_window(self):
        """Center the window on the screen"""
        x_position = (self.screen_width - self.window_width) // 2
        y_position = (self.screen_height - self.window_height) // 2
        self.geometry(f"{self.window_width}x{self.window_height}+{x_position}+{y_position}")

    def create_header(self):
        """A slim title bar. Links to the docs and GitHub live in the Help menu."""
        header = tk.Frame(self, bg=NIGHT)
        header.pack(side="top", fill="x")
        tk.Label(header, image=self._header_icon, bg=NIGHT).pack(side="left", padx=(14, 8), pady=7)
        tk.Label(header, text="EclipsingBinaries", font=self.title_font, bg=NIGHT, fg=NIGHT_TEXT).pack(side="left")
        tk.Label(header, text=f"v{__version__}", font=self.small_font, bg=NIGHT, fg=NIGHT_MUTED).pack(
            side="left", padx=(8, 0), pady=(4, 0))

    def create_layout(self):
        """Sidebar on the left, the open panel on the right in a scrolling white card."""
        body = tk.Frame(self, bg=BG_APP)
        body.pack(side="top", fill="both", expand=True)

        self.left_frame = tk.Frame(body, bg=NIGHT, width=250)
        self.left_frame.pack(side="left", fill="y")
        self.left_frame.pack_propagate(False)

        holder = tk.Frame(body, bg=BG_APP)
        holder.pack(side="left", fill="both", expand=True, padx=16, pady=16)
        card = tk.Frame(holder, bg=BG_PANEL, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="both", expand=True)
        self._panel_scroller = ScrollableFrame(card, bg=BG_PANEL)
        self._panel_scroller.pack(fill="both", expand=True)
        # Panels build into right_frame, which scrolls when it doesn't fit
        self.right_frame = self._panel_scroller.content

    def create_menu(self):
        """Create the menu bar and the left-side options menu."""
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Clear Saved Settings...", command=self.clear_saved_settings)
        if IS_MAC:
            # macOS keeps About and Quit in the application menu. Quit is
            # wired up through tk::mac::Quit in __init__.
            app_menu = tk.Menu(menubar, name="apple", tearoff=0)
            app_menu.add_command(label="About EclipsingBinaries", command=self.show_about)
            menubar.add_cascade(menu=app_menu)
        else:
            file_menu.add_separator()
            file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.quit_program)
            self.bind_all("<Control-q>", lambda e: self.quit_program())
            self.bind_all("<Control-Q>", lambda e: self.quit_program())  # with Caps Lock on
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, name="help", tearoff=0)
        help_menu.add_command(label="Help Contents", command=self.open_help_window)
        help_menu.add_command(label="Documentation", command=lambda: webbrowser.open(DOCS_URL))
        help_menu.add_command(label="GitHub Repository", command=lambda: webbrowser.open(REPO_URL))
        if not IS_MAC:
            help_menu.add_separator()
            help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self._build_sidebar()

    # ------------------------------------------------------------------
    # Sidebar navigation
    # ------------------------------------------------------------------
    def _nav_sections(self):
        """
        Programs grouped in the order people usually work through them.
        An entry with no builder isn't available in the app yet; its note
        says how to run it instead.
        """
        return [
            ("Reduce", [("iraf", "IRAF Reduction", self.show_iraf_reduction, None)]),
            ("Comparison stars", [("aij", "AIJ Comparison Star Selector", self.show_aij_comparison_selector, None)]),
            ("Photometry", [
                ("photometry", "Multi-Aperture Photometry", self.show_multi_aperture_photometry, None),
                ("night_filters", "Night Filters", self.show_night_filters, None),
            ]),
            ("Analysis", [
                ("find_min", "Find Minimum", self.show_find_minimum, None),
                ("oc", "O-C Plotting", self.show_oc_plot, None),
                ("oconnell", "O'Connell Effect", self.show_oconnell_effect, None),
                ("color", "Color Light Curve", self.show_color_light_curve, None),
            ]),
            ("Archives", [
                ("tess", "TESS Search and Download", self.show_tess_search, None),
                ("gaia", "Gaia Search", self.show_gaia_query, None),
            ]),
        ]

    def _build_sidebar(self):
        self._panel_builders = {}
        self._nav_items = {}
        for section, items in self._nav_sections():
            tk.Label(self.left_frame, text=section, font=self.section_font, bg=NIGHT, fg=NIGHT_MUTED,
                     anchor="w").pack(fill="x", padx=18, pady=(14, 2))
            for key, text, builder, note in items:
                self._nav_items[key] = self._make_nav_item(key, text, enabled=builder is not None, note=note)
                if builder is not None:
                    self._panel_builders[key] = builder

        footer = tk.Label(self.left_frame, text="Developed by Kyle Koeller\nEclipsing Binaries Research",
                          font=self.small_font, bg=NIGHT, fg=NIGHT_MUTED, justify="left", anchor="w")
        footer.pack(side="bottom", fill="x", padx=18, pady=12)

    def _make_nav_item(self, key, text, enabled=True, note=None):
        """One sidebar entry: a starlight bar on the left marks the open panel."""
        row = tk.Frame(self.left_frame, bg=NIGHT)
        row.pack(fill="x")
        bar = tk.Frame(row, bg=NIGHT, width=4)
        bar.pack(side="left", fill="y")
        label = tk.Label(row, text=text, font=self.nav_font, bg=NIGHT,
                         fg=NIGHT_TEXT if enabled else NIGHT_DISABLED, anchor="w",
                         padx=14, pady=6, highlightthickness=1, highlightbackground=NIGHT,
                         highlightcolor=STARLIGHT, takefocus=enabled)
        label.pack(side="left", fill="x", expand=True)
        item = {"row": row, "bar": bar, "label": label, "enabled": enabled}

        if note:
            self._tooltip(label, note)
        if not enabled:
            return item

        def paint(bg, marker=None):
            for widget in (row, label):
                widget.config(bg=bg)
            label.config(highlightbackground=bg)
            bar.config(bg=marker or bg)

        def on_enter(_event):
            if self._current_panel != key:
                paint(NIGHT_HOVER)

        def on_leave(_event):
            if self._current_panel != key:
                paint(NIGHT)

        def activate(_event=None):
            self.open_panel(key)
            return "break"

        for widget in (row, label):
            widget.bind("<Enter>", on_enter, add="+")
            widget.bind("<Leave>", on_leave, add="+")
            widget.bind("<Button-1>", activate, add="+")
            widget.configure(cursor="hand2")
        label.bind("<Return>", activate)
        label.bind("<space>", activate)
        item["paint"] = paint
        return item

    def _set_active_nav(self, key):
        for item_key, item in self._nav_items.items():
            if not item["enabled"]:
                continue
            active = item_key == key
            if active:
                item["paint"](NIGHT_ACTIVE, STARLIGHT)
            else:
                item["paint"](NIGHT)
            item["label"].config(fg="white" if active else NIGHT_TEXT)

    def open_panel(self, key):
        """Open a panel by its sidebar key, for example "photometry"."""
        builder = self._panel_builders.get(key)
        if builder is not None:
            builder()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def _bind_shortcuts(self):
        self.bind_all("<Return>", self._on_return_key, add="+")
        self.bind_all("<KP_Enter>", self._on_return_key, add="+")
        self.bind_all("<Escape>", self._on_escape_key, add="+")

    def _on_return_key(self, event):
        """Enter presses the focused button, or runs the open panel."""
        widget = event.widget
        try:
            if widget.winfo_toplevel() is not self:
                return None  # leave other windows, like the radii optimizer, alone
        except (tk.TclError, AttributeError, KeyError):
            return None
        if isinstance(widget, ttk.Button):
            if widget.instate(["!disabled"]):
                widget.invoke()
            return "break"
        if isinstance(widget, (tk.Text, tk.Label)) or widget.winfo_class() in ("TCombobox", "Listbox"):
            return None
        if self._run_button is not None and self._run_button.winfo_exists() and self._run_button.instate(["!disabled"]):
            self._run_button.invoke()
            return "break"
        return None

    def _on_escape_key(self, event):
        """Escape offers to cancel the running task."""
        try:
            if event.widget.winfo_toplevel() is not self or event.widget.winfo_class() == "Listbox":
                return None
        except (tk.TclError, AttributeError, KeyError):
            return None
        if self._task_running:
            self.cancel_task()
            return "break"
        return None

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    def open_help_window(self):
        """A short guide to the panels and shortcuts."""
        window = tk.Toplevel(self)
        window.title("Help Contents")
        window.configure(bg=BG_PANEL)
        window.transient(self)

        sections = [
            ("Workflow", "Work down the sidebar: reduce the raw images, choose comparison stars, run "
                         "photometry, then analyze the light curves. TESS and Gaia searches are under Archives."),
            ("Fields", "Grey text in a field is only an example. Paths can be typed, chosen with Browse, "
                       "or dragged onto the field. Hover over a field name for a note on what it expects. "
                       "Each panel remembers what you entered last time; File, Clear Saved Settings forgets it."),
            ("Running", "Run starts the program in the background. The status bar shows what's running and "
                        "for how long, and Cancel stops it at the next safe point. When it finishes, buttons "
                        "next to Run open the output folder or the results."),
            ("Output log", "Errors show in red, warnings in amber, and finished steps in green. Save Log keeps "
                           "a copy, which helps when reporting a problem."),
            ("Shortcuts", "Enter runs the open panel. Escape offers to cancel a running task. "
                          + ("Cmd-Q quits." if IS_MAC else "Ctrl-Q quits.")),
        ]
        for heading, text in sections:
            tk.Label(window, text=heading, font=self.strong_font, bg=BG_PANEL, fg=FG_TEXT,
                     anchor="w").pack(fill="x", padx=20, pady=(14, 2))
            tk.Label(window, text=text, font=self.label_font, bg=BG_PANEL, fg=FG_TEXT, justify="left",
                     wraplength=520, anchor="w").pack(fill="x", padx=20)
        links = tk.Frame(window, bg=BG_PANEL)
        links.pack(fill="x", padx=20, pady=16)
        self._link(links, "Full documentation", DOCS_URL).pack(side="left")
        self._place_dialog(window)

    def _place_dialog(self, window):
        """Centre a dialog over the main window and let Escape close it."""
        window.bind("<Escape>", lambda _e: window.destroy())
        window.update_idletasks()
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
        x = self.winfo_rootx() + max((self.winfo_width() - width) // 2, 0)
        y = self.winfo_rooty() + max((self.winfo_height() - height) // 3, 0)
        window.geometry(f"+{x}+{y}")
        window.focus_set()

    def _link(self, parent, text, url):
        link = tk.Label(parent, text=text, font=self.label_font, bg=parent["bg"], fg=ACCENT_ACTIVE, cursor="hand2")
        link.configure(font=tkfont.Font(self, font=self.label_font, underline=True))
        link.bind("<Button-1>", lambda _e: webbrowser.open(url))
        return link

    def show_about(self):
        """About dialog with working links to the docs and the repository."""
        window = tk.Toplevel(self)
        window.title("About EclipsingBinaries")
        window.configure(bg=BG_PANEL)
        window.resizable(False, False)
        window.transient(self)

        tk.Label(window, image=self._icon, bg=BG_PANEL).pack(pady=(20, 6))
        tk.Label(window, text="EclipsingBinaries", font=self.header_font, bg=BG_PANEL, fg=FG_TEXT).pack()
        tk.Label(window, text=f"Version {__version__}", font=self.label_font, bg=BG_PANEL, fg=FG_MUTED).pack(padx=30)
        tk.Label(window, text="Kyle Koeller", font=self.label_font, bg=BG_PANEL, fg=FG_TEXT).pack(pady=(10, 0))
        links = tk.Frame(window, bg=BG_PANEL)
        links.pack(pady=(10, 4))
        self._link(links, "Documentation", DOCS_URL).pack(side="left", padx=8)
        self._link(links, "GitHub", REPO_URL).pack(side="left", padx=8)
        self._button(window, "Close", window.destroy).pack(pady=(10, 18))
        self._place_dialog(window)

    def clear_right_frame(self):
        """
        Clear the panel area for a new panel. The outgoing panel's fields are
        saved first so they're filled in again next time.
        """
        self._save_panel_settings()
        self._panel_fields = {}
        self._panel_vars = {}
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        self.log_area = None
        self._run_button = None
        self._cancel_button = None
        self._results_frame = None
        self._action_buttons = []
        self._panel_scroller.scroll_to_top()
        # Rows the old panel stretched (its output log) would leave gaps in the new one
        cols, rows = self.right_frame.grid_size()
        for r in range(rows):
            self.right_frame.grid_rowconfigure(r, weight=0, minsize=0)
        for c in range(cols):
            self.right_frame.grid_columnconfigure(c, weight=0, minsize=0)

    def create_checkbox(self, parent, text, variable, row, tooltip=None):
        """Create a checkbox lined up with the input fields"""
        box = tk.Checkbutton(parent, text=text, variable=variable, font=self.label_font, bg=BG_PANEL, fg=FG_TEXT)
        box.grid(row=row, column=1, padx=(0, 10), pady=5, sticky="w")
        if tooltip:
            self._tooltip(box, tooltip)
        return box

    def create_scrollbar_and_log(self, row):
        log_frame = tk.Frame(self.right_frame, bg=BG_PANEL, highlightthickness=1, highlightbackground=BORDER)
        log_frame.grid(row=row, column=0, columnspan=3, padx=16, pady=(0, 16), sticky="nsew")

        self.log_area = tk.Text(log_frame, wrap="word", height=12, font=self.mono_font, relief="flat",
                                borderwidth=0, padx=8, pady=6, bg="#fbfcfe", fg=FG_TEXT,
                                highlightthickness=0)
        self.log_area.tag_configure("error", foreground=FG_ERROR)
        self.log_area.tag_configure("warning", foreground=FG_WARNING)
        self.log_area.tag_configure("success", foreground=FG_SUCCESS)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_area.yview)
        self.log_area.configure(yscrollcommand=scrollbar.set)

        self.log_area.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.right_frame.grid_rowconfigure(row, weight=1, minsize=160)

    def create_run_button(self, parent, action, row, text="Run", cancellable=True, **kwargs):
        """
        Create the Run and Cancel buttons for a panel, with room beside them for
        the results buttons that appear when a task finishes. Panels whose
        button opens a window rather than starting a task pass cancellable=False
        to leave out Cancel.
        """
        button_frame = tk.Frame(parent, bg=BG_PANEL)
        button_frame.grid(row=row, column=1, columnspan=2, padx=(0, 10), pady=(14, 10), sticky="w")

        def run():
            self._save_panel_settings()
            action(**kwargs)

        self._run_button = self._button(button_frame, text, run)
        self._run_button.pack(side="left", padx=(0, 8))

        if cancellable:
            self._cancel_button = self._button(button_frame, "Cancel", self.cancel_task)
            self._cancel_button.pack(side="left")

        self._results_frame = tk.Frame(button_frame, bg=BG_PANEL)
        self._results_frame.pack(side="left", padx=(16, 0))

        self._sync_task_buttons()

    def _panel_title(self, text, key):
        """Title the panel, mark it in the sidebar, and remember it as the last one open."""
        self._current_panel = key
        self._current_panel_title = text
        self._set_active_nav(key)
        self.right_frame.grid_columnconfigure(0, weight=0)
        self.right_frame.grid_columnconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(2, weight=0)
        tk.Label(self.right_frame, text=text, font=self.header_font, bg=BG_PANEL, fg=FG_TEXT, anchor="w").grid(
            row=0, column=0, columnspan=3, padx=16, pady=(16, 12), sticky="w"
        )

    def _log_section(self, row):
        header = tk.Frame(self.right_frame, bg=BG_PANEL)
        header.grid(row=row, column=0, columnspan=3, padx=16, pady=(8, 4), sticky="ew")
        tk.Label(header, text="Output log", font=self.strong_font, bg=BG_PANEL, fg=FG_TEXT).pack(side="left")
        self._button(header, "Clear", self._clear_log, style="Secondary.TButton").pack(side="right")
        self._button(header, "Save Log...", self._save_log, style="Secondary.TButton").pack(side="right", padx=8)
        self.create_scrollbar_and_log(row + 1)

    def _clear_log(self):
        if self.log_area is not None:
            self.log_area.delete("1.0", "end")

    def _save_log(self):
        if self.log_area is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        default = f"{(self._current_panel or 'log')}_log_{stamp}.txt"
        path = filedialog.asksaveasfilename(parent=self, title="Save Log", initialfile=default,
                                            defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt"), ("All files", "*")])
        if not path:
            return
        try:
            Path(path).write_text(self.log_area.get("1.0", "end-1c"), encoding="utf-8")
            self.write_to_log(f"Saved the log to {path}")
        except OSError as exc:
            self.write_to_log(f"Error: couldn't save the log: {exc}")

    # ------------------------------------------------------------------
    # IRAF Reduction
    # ------------------------------------------------------------------
    def show_iraf_reduction(self):
        """Display the IRAF reduction panel."""
        self.clear_right_frame()
        self._panel_title("IRAF Reduction", "iraf")

        raw_images_path = self.create_input_field(self.right_frame, "Raw Images Path:",
                                                  _example_path("raw_images"), row=1,
                                                  error_message="File path cannot be empty.",
                                                  browse_type="folder",
                                                  tooltip="Folder with the night's raw bias, dark, flat and "
                                                          "science frames.")

        calibrated_images_path = self.create_input_field(self.right_frame, "Calibrated Images Path:",
                                                         _example_path("calibrated_images"), row=2,
                                                         error_message="File path cannot be empty.",
                                                         browse_type="folder",
                                                         tooltip="Where the reduced frames and the master "
                                                                 "calibration frames are written. Created if "
                                                                 "it doesn't exist.")

        location = self.create_input_field(self.right_frame, "Location:",
                                           "BSUO (default), CTIO, KPNO, etc.", row=3,
                                           error_message="Please enter an observatory.",
                                           default="BSUO",
                                           tooltip="Observatory. BSUO, KPNO, CTIO and LaPalma set the "
                                                   "camera gain and read noise for you; any other site "
                                                   "uses the BSUO camera values.")

        dark_bool_var = self._remember_var("Use Dark Frames", tk.BooleanVar(master=self, value=True))
        self.create_checkbox(self.right_frame, "Use Dark Frames", dark_bool_var, row=4,
                             tooltip="Subtract a master dark. Turn this off if you took no darks.")

        overscan = self.create_input_field(self.right_frame, "Overscan Region:",
                                           "[2073:2115, :]", row=5,
                                           error_message="Please enter at least [:,:].",
                                           default="[2073:2115, :]",
                                           tooltip="FITS section [columns, rows] of the overscan strip, "
                                                   "e.g. [2073:2115, :]. Use all rows (:) here. Enter "
                                                   "none to skip overscan subtraction. Open Bias Image "
                                                   "helps find where it starts.")

        trim = self.create_input_field(self.right_frame, "Trim Region:",
                                       "[20:2060, 12:2057]", row=6,
                                       error_message="Please enter at least [:,:].",
                                       default="[20:2060, 12:2057]",
                                       tooltip="FITS section [columns, rows] of the image to keep after "
                                               "trimming, e.g. [20:2060, 12:2057].")

        bias_button = self._button(self.right_frame, "Open Bias Image", self.open_bias_image,
                                   style="Secondary.TButton")
        bias_button.grid(row=7, column=1, padx=(0, 10), pady=(4, 0), sticky="w")
        self._tooltip(bias_button, "Plot one row of a bias frame to see where the overscan begins.")

        self.create_run_button(self.right_frame, self.run_iraf_reduction, row=8,
                               raw_images_path=raw_images_path,
                               calibrated_images_path=calibrated_images_path,
                               location=location,
                               dark_bool_var=dark_bool_var,
                               overscan_var=overscan,
                               trim_var=trim)

        self._log_section(9)

    def open_bias_image(self):
        """Open a bias image, plot it, and prompt for regions."""
        file_path = filedialog.askopenfilename(
            title="Select Bias Image", parent=self,
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*")],
        )
        if not file_path:
            return
        try:
            ccd = CCDData.read(file_path, unit="adu")
            self.bias_plot(ccd, Path(file_path).name)
            self.write_to_log(f"Successfully loaded and plotted bias image: {file_path}")
        except Exception as e:
            self.write_to_log(f"Failed to load bias image: {e}")

    def bias_plot(self, ccd, title_name=""):
        """
        Plot the counts along one row of a bias frame to help pick overscan and
        trim regions. Drawn into its own window rather than through pyplot so
        it works no matter which matplotlib backend is active.
        """
        n_rows, n_cols = ccd.data.shape
        row = n_rows // 2
        overscan_hint = 2077

        window = tk.Toplevel(self)
        window.title(f"Bias Row {row} {title_name}".strip())

        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot()
        ax.plot(ccd.data[row, :], label="Raw Bias")
        ax.grid()
        if overscan_hint < n_cols:
            ax.axvline(x=overscan_hint, color="black", linewidth=2, linestyle="dashed",
                       label="Suggested Start of Overscan")
        ax.legend()
        ax.set_xlim(-50, n_cols + 50)
        ax.set_xlabel("Pixel Number")
        ax.set_ylabel("Counts")
        ax.set_title(f"Bias Image Analysis: Row {row}")

        canvas = FigureCanvasTkAgg(fig, master=window)
        toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()

    def _show_light_curves(self, folder, obj_name, filters):
        """Plot the light curves photometry just wrote, one panel per filter, in their own window."""
        import pandas as pd

        curves = []
        for filt in filters:
            path = Path(folder) / f"{obj_name}_{filt}_data.csv"
            if path.exists():
                curves.append((filt, pd.read_csv(path)))
        if not curves:
            self.write_to_log(f"Error: no light curve files for {obj_name} were found in {folder}.")
            return

        window = tk.Toplevel(self)
        window.title(f"{obj_name} light curves" if obj_name else "Light curves")
        fig = Figure(figsize=(9, 2.6 * len(curves) + 0.8))
        axes = fig.subplots(len(curves), 1, sharex=True, squeeze=False)[:, 0]
        colours = {"B": "#2f5aa8", "V": "#2d7d46", "R": "#b23a2e"}
        for ax, (filt, data) in zip(axes, curves):
            ax.errorbar(data["HJD"], data["Source_AMag_T1"], yerr=data["Source_AMag_T1_Error"],
                        fmt="o", ms=3, elinewidth=0.6, color=colours.get(filt, ACCENT))
            ax.invert_yaxis()
            ax.set_ylabel(f"{filt} (mag)")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("HJD")
        axes[-1].ticklabel_format(axis="x", useOffset=False, style="plain")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=window)
        toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()

    def run_iraf_reduction(self, raw_images_path, calibrated_images_path, location,
                           dark_bool_var, overscan_var, trim_var):
        """Validate the IRAF inputs, then run the reduction on a worker thread."""
        raw_path = self._path_value(raw_images_path)
        calibrated_path = self._path_value(calibrated_images_path)
        loc = self._entry_value(location)
        use_dark_frames = dark_bool_var.get()
        overscan_region = self._entry_value(overscan_var)
        trim_region = self._entry_value(trim_var)

        if not raw_path or not Path(raw_path).is_dir():
            self.write_to_log("Error: Raw images path does not exist.")
            return
        if not calibrated_path:
            self.write_to_log("Error: Calibrated images path is required.")
            return

        try:
            cfg = site_config(loc, dark_bool=use_dark_frames,
                              overscan_region=overscan_region, trim_region=trim_region)
        except ValueError as e:
            self.write_to_log(f"Error: {e}")
            return

        def reduction_task():
            self.write_to_log("Paths validated successfully.")
            self.write_to_log(f"Raw Images Path: {raw_path}")
            self.write_to_log(f"Calibrated Images Path: {calibrated_path}")
            self.write_to_log(f"Location: {loc}")
            self.write_to_log(f"Use Dark Frames: {'Yes' if use_dark_frames else 'No'}")
            self.write_to_log(f"Trim Region: {trim_region}")
            self.write_to_log(f"Overscan Region: {overscan_region}")
            self.write_to_log("Starting IRAF Reduction...\n")
            try:
                run_reduction(path=raw_path, calibrated=calibrated_path, cfg=cfg,
                              cancel_event=self.cancel_event, write_callback=self.write_to_log)
                if self.cancel_event.is_set():
                    self.write_to_log("IRAF Reduction was canceled.")
                else:
                    self.write_to_log("IRAF Reduction completed successfully.")
                    self.offer_results(calibrated_path)
            except Exception as e:
                self.write_to_log(f"An error occurred during IRAF Reduction: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(reduction_task)

    # ------------------------------------------------------------------
    # TESS search
    # ------------------------------------------------------------------
    def show_tess_search(self):
        """Display the TESS Database Search panel."""
        self.clear_right_frame()
        self._panel_title("TESS Search and Download", "tess")

        system_name = self.create_input_field(self.right_frame, "System Name:",
                                              "e.g. NSVS 896797", row=1,
                                              error_message="Please enter a system name.",
                                              tooltip="TIC ID or any name MAST can resolve, e.g. TIC 12345678 "
                                                      "or NSVS 896797.")

        download_path = self.create_input_field(self.right_frame, "Download Path:",
                                                _example_path("download"), row=2,
                                                error_message="Please enter a file pathway.",
                                                browse_type="folder",
                                                tooltip="Each sector is saved in its own numbered folder here.")

        single_sector_var = tk.BooleanVar(master=self, value=False)

        # Placeholders for widgets that appear when a specific sector is wanted
        self.sector_dropdown = None
        self.retrieve_button = None
        self.sector_label = None

        sector_box = tk.Checkbutton(
            self.right_frame, text="Download Specific Sector", variable=single_sector_var,
            font=self.label_font, bg=BG_PANEL, fg=FG_TEXT,
            command=lambda: self.toggle_sector_options(single_sector_var, system_name)
        )
        sector_box.grid(row=3, column=1, padx=(0, 10), pady=5, sticky="w")
        self._tooltip(sector_box, "Download one sector instead of every sector TESS observed. "
                                  "Retrieve Sectors lists the ones available.")

        self.create_run_button(self.right_frame, self.run_tess_search, row=6,
                               system_name=system_name,
                               download_path=download_path,
                               single_sector_var=single_sector_var)

        self._log_section(7)

    def retrieve_sectors(self, system_name):
        """Look up available sectors for a target without blocking the window."""
        system_name_value = self._entry_value(system_name)
        if not system_name_value:
            self.write_to_log("Error: System name (TIC ID) is required.")
            return

        def lookup():
            self.write_to_log(f"Retrieving sectors for: {system_name_value}")
            try:
                sector_table = Tesscut.get_sectors(objectname=system_name_value)
            except Exception as e:
                self.write_to_log(f"Error retrieving sectors: {e}")
                return
            if not sector_table:
                self.write_to_log(f"No TESS data found for system {system_name_value}.")
                return
            formatted_table = "\n".join(sector_table.pformat(show_name=True, max_width=-1, align="^"))
            self.write_to_log("Available Sectors:\n" + formatted_table)
            self.call_in_ui(self._populate_sectors, [str(s) for s in sector_table["sector"]])

        self.run_task(lookup)

    def _populate_sectors(self, sectors):
        self.available_sectors = sectors
        if self.sector_dropdown is not None and self.sector_dropdown.winfo_exists():
            self.sector_dropdown["values"] = sectors
            self.sector_dropdown.set("Select a Sector")
            self.write_to_log("Sectors successfully retrieved.")

    def toggle_sector_options(self, single_sector_var, system_name):
        """Show or hide the sector dropdown, 'Select Specific Sector' label, and Retrieve Sectors button."""
        if single_sector_var.get():
            if not self.sector_label:
                self.sector_label = tk.Label(self.right_frame, text="Select Specific Sector:",
                                             font=self.label_font, bg=BG_PANEL, fg=FG_TEXT)
                self.sector_label.grid(row=5, column=0, padx=(16, 10), sticky="e")

            if not self.sector_dropdown:
                self.sector_dropdown = ttk.Combobox(self.right_frame, state="readonly",
                                                    values=[], font=self.label_font)
                self.sector_dropdown.grid(row=5, column=1, padx=(0, 10), pady=5, sticky="w")

            if not self.retrieve_button:
                self.retrieve_button = self._button(
                    self.right_frame, "Retrieve Sectors",
                    lambda: self.retrieve_sectors(system_name), action=True, style="Secondary.TButton")
                self.retrieve_button.grid(row=4, column=1, padx=(0, 10), pady=4, sticky="w")
        else:
            if self.sector_label:
                self.sector_label.destroy()
                self.sector_label = None

            if self.sector_dropdown:
                self.sector_dropdown.destroy()
                self.sector_dropdown = None

            if self.retrieve_button:
                self._action_buttons.remove(self.retrieve_button)
                self.retrieve_button.destroy()
                self.retrieve_button = None

    def run_tess_search(self, system_name, download_path, single_sector_var):
        """Validate the TESS inputs, then run the search and download on a worker thread."""
        system_name_value = self._entry_value(system_name)
        download_path_value = self._path_value(download_path)
        single_sector = single_sector_var.get()

        specific_sector_value = None
        if single_sector:
            selected = self.sector_dropdown.get() if self.sector_dropdown else ""
            if not selected.isdigit():
                self.write_to_log("Error: Please select a valid sector.")
                return
            specific_sector_value = int(selected)

        if not system_name_value:
            self.write_to_log("Error: System name (TIC ID) is required.")
            return
        if not download_path_value:
            self.write_to_log("Error: Download path is required.")
            return

        def search_task():
            self.write_to_log(f"System Name: {system_name_value}")
            self.write_to_log(f"Download Path: {download_path_value}")
            if single_sector:
                self.write_to_log(f"Sector: {specific_sector_value}")
            else:
                self.write_to_log("Sectors: all available")
            try:
                run_tess_search(
                    system_name=system_name_value, single_sector=single_sector,
                    specific_sector=specific_sector_value, download_path=download_path_value,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event
                )
                if self.cancel_event.is_set():
                    self.write_to_log("TESS search was canceled.")
                else:
                    self.write_to_log("TESS search completed successfully.")
                    self.offer_results(download_path_value)
            except Exception as e:
                self.write_to_log(f"An error occurred during TESS database search: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(search_task)

    # ------------------------------------------------------------------
    # AIJ comparison selector
    # ------------------------------------------------------------------
    def show_aij_comparison_selector(self):
        """Display the AIJ Comparison Star Selector panel."""
        self.clear_right_frame()
        self._panel_title("AIJ Comparison Star Selector", "aij")

        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):",
                                     "HH:MM:SS.SSSS", row=1,
                                     error_message="Please enter a RA.",
                                     tooltip="Target RA in hours (J2000), e.g. 00:28:27.97.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      error_message="Please enter a DEC.",
                                      tooltip="Target Dec in degrees (J2000), e.g. +78:57:42.66.")

        folder_path = self.create_input_field(self.right_frame, "Data Save Folder Path:",
                                              _example_path("download"), row=3,
                                              error_message="Please enter a file pathway.",
                                              browse_type="folder",
                                              tooltip="Where the APASS catalog, RADEC files and overlay "
                                                      "image are written.")

        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "e.g. NSVS 896797", row=4,
                                           error_message="Please enter the object name.",
                                           tooltip="Used to name the output files.")

        science_image = self.create_input_field(self.right_frame, "Science Image File:",
                                                _example_path("calibrated_images", "image.fits"), row=5,
                                                error_message="Please enter a file pathway.",
                                                browse_type="file",
                                                tooltip="A reduced science frame with a WCS. The comparison "
                                                        "stars are circled on it in the overlay image.")

        self.create_run_button(self.right_frame, self.run_comparison_selector, row=6,
                               text="Run Comparison Selector",
                               ra=ra, dec=dec, folder_path=folder_path, obj_name=obj_name,
                               science_image=science_image)

        self._log_section(7)

    def run_comparison_selector(self, ra, dec, folder_path, obj_name, science_image):
        """Validate the selector inputs, then run it on a worker thread."""
        ra_value = self._entry_value(ra)
        dec_value = self._entry_value(dec)
        folder_value = self._path_value(folder_path)
        obj_value = self._entry_value(obj_name)
        science_image_value = self._path_value(science_image)

        if not all([ra_value, dec_value, folder_value, obj_value, science_image_value]):
            self.write_to_log("Error: All fields are required.")
            return

        def selector_task():
            self.write_to_log(f"Running comparison selector for object: {obj_value}")
            try:
                comparison_selector(ra=ra_value, dec=dec_value, pipeline=False,
                                    folder_path=folder_value, obj_name=obj_value,
                                    science_image=science_image_value,
                                    write_callback=self.write_to_log, cancel_event=self.cancel_event)
                if self.cancel_event.is_set():
                    self.write_to_log("Comparison Selector was canceled.")
                else:
                    self.write_to_log("Comparison Selector completed successfully.")
                    overlay = Path(folder_value) / f"{obj_value}_overlay.png"
                    extras = [("View Overlay", lambda: self._open_in_system(overlay))] if overlay.exists() else []
                    self.offer_results(folder_value, extras)
            except Exception as e:
                self.write_to_log(
                    f"An error occurred during comparison star selection: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(selector_task)

    # ------------------------------------------------------------------
    # Multi-aperture photometry
    # ------------------------------------------------------------------
    def show_multi_aperture_photometry(self):
        """Display the Multi-Aperture Photometry panel."""
        self.clear_right_frame()
        self._panel_title("Multi-Aperture Photometry", "photometry")

        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "e.g. NSVS 896797", row=1,
                                           error_message="Please enter an object name.",
                                           tooltip="Used to name the light curve files.")

        reduced_images_path = self.create_input_field(self.right_frame, "Reduced Images Path:",
                                                      _example_path("reduced_images"), row=2,
                                                      error_message="Please enter a file pathway.",
                                                      browse_type="folder",
                                                      tooltip="Folder of reduced LIGHT frames with a WCS. The "
                                                              "light curves are saved here too.")

        # Number of filters, chosen the same way as in the O'Connell Effect panel
        count_label = tk.Label(self.right_frame, text="Number of Filters:", font=self.label_font,
                               bg=BG_PANEL, fg=FG_TEXT)
        count_label.grid(row=3, column=0, padx=(16, 10), pady=5, sticky="e")
        self._tooltip(count_label, "How many filters you observed in. Each one gets a RADEC row below, "
                                   "and all of them share the same comparison stars.")
        self.photometry_filter_count = self._remember_var("filter_count", tk.IntVar(master=self, value=len(PHOTOMETRY_FILTERS)))
        radio_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        radio_frame.grid(row=3, column=1, padx=(0, 10), pady=5, sticky="w")
        for count in range(1, len(PHOTOMETRY_FILTERS) + 1):
            tk.Radiobutton(radio_frame, text=f"{count} Filter{'s' if count > 1 else ''}",
                           variable=self.photometry_filter_count, value=count,
                           command=self._update_photometry_filter_rows,
                           font=self.label_font, bg=BG_PANEL, fg=FG_TEXT).pack(side="left", padx=(0, 12))

        # One row per filter: a B/V/R picker and that filter's RADEC file
        self.photometry_filter_rows = []
        saved = self.settings.section("photometry")
        self._photometry_filters_picked = any(f"filter_{i}" in saved for i in range(len(PHOTOMETRY_FILTERS)))
        for i, default_filter in enumerate(PHOTOMETRY_FILTERS):
            label_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
            tk.Label(label_frame, text="RADEC File (", font=self.label_font, bg=BG_PANEL,
                     fg=FG_TEXT).pack(side="left")
            picker_var = self._remember_var(f"filter_{i}", tk.StringVar(master=self, value=default_filter))
            picker = ttk.Combobox(label_frame, values=PHOTOMETRY_FILTERS, state="readonly", width=3,
                                  font=self.label_font, textvariable=picker_var)
            picker.pack(side="left")
            tk.Label(label_frame, text=" Filter):", font=self.label_font, bg=BG_PANEL,
                     fg=FG_TEXT).pack(side="left")

            entry = self.create_input_field(self.right_frame, "", _example_path(f"{picker.get()}.radec"),
                                            row=4 + i, error_message="Please enter a RADEC file.",
                                            browse_type="file", label_widget=label_frame,
                                            remember_as=f"radec_{i}",
                                            tooltip="RADEC file from the AIJ Comparison Star Selector for this "
                                                    "filter. The first row is the target, the rest are "
                                                    "comparison stars.")
            picker.bind("<<ComboboxSelected>>",
                        lambda _e, p=picker, ent=entry: self._on_photometry_filter_picked(p, ent))
            self.photometry_filter_rows.append((picker, entry))
        self._update_photometry_filter_rows()

        self.aperture_radius_var = self.create_input_field(self.right_frame, "Aperture Radius (px):",
                                                           "20", row=7, validation_func=_is_float,
                                                           error_message="Please enter a numeric radius.",
                                                           default="20",
                                                           tooltip="Radius of the circle the star's light is "
                                                                   "summed in. Optimize Radii suggests one.")

        self.annulus_inner_var = self.create_input_field(self.right_frame, "Inner Annulus Radius (px):",
                                                         "30", row=8, validation_func=_is_float,
                                                         error_message="Please enter a numeric inner radius.",
                                                         default="30",
                                                         tooltip="Inner edge of the sky background ring. "
                                                                 "Keep it clear of the star's light.")

        self.annulus_outer_var = self.create_input_field(self.right_frame, "Outer Annulus Radius (px):",
                                                         "50", row=9, validation_func=_is_float,
                                                         error_message="Please enter a numeric outer radius.",
                                                         default="50",
                                                         tooltip="Outer edge of the sky background ring. Stars "
                                                                 "whose ring runs off any image are dropped.")

        optimize = self._button(self.right_frame, "Optimize Radii",
                                lambda: self.run_radii_optimizer(reduced_images_path),
                                action=True, style="Secondary.TButton")
        optimize.grid(row=10, column=1, padx=(0, 10), pady=(4, 0), sticky="w")
        self._tooltip(optimize, "Measure the target's size on the first image and suggest radii, with "
                                "sliders to adjust them while watching the signal-to-noise.")

        self.create_run_button(self.right_frame, self.run_multi_aperture_photometry, row=11,
                               obj_name=obj_name,
                               reduced_images_path=reduced_images_path)

        self._log_section(12)

    def _update_photometry_filter_rows(self):
        """Show one RADEC row per chosen filter and hide the rest, keeping what was typed."""
        count = self.photometry_filter_count.get()
        if not self._photometry_filters_picked:
            # Until someone picks filters by hand, suggest the usual ones for this count
            for (picker, entry), filt in zip(self.photometry_filter_rows, PHOTOMETRY_DEFAULTS[count]):
                picker.set(filt)
                self._set_placeholder(entry, _example_path(f"{filt}.radec"))
        for i, (_picker, entry) in enumerate(self.photometry_filter_rows):
            self._set_field_visible(entry, i < count)

    def _on_photometry_filter_picked(self, picker, entry):
        self._photometry_filters_picked = True
        self._set_placeholder(entry, _example_path(f"{picker.get()}.radec"))

    def _photometry_radec_choices(self):
        """
        Filter and RADEC path for each visible row.

        :return: list of (filter, path) pairs, or None after logging the problem
        """
        count = self.photometry_filter_count.get()
        choices = [(picker.get(), self._path_value(entry))
                   for picker, entry in self.photometry_filter_rows[:count]]
        picked = [filt for filt, _ in choices]
        if len(set(picked)) != len(picked):
            self.write_to_log("Error: Each filter can only be chosen once.")
            return None
        missing = [filt for filt, path in choices if not path]
        if missing:
            self.write_to_log(f"Error: A RADEC file is required for {', '.join(missing)}.")
            return None
        return choices

    def run_multi_aperture_photometry(self, obj_name, reduced_images_path):
        """Validate the photometry inputs, then run it on a worker thread."""
        obj_name_value = self._entry_value(obj_name)
        reduced_path_value = self._path_value(reduced_images_path)

        try:
            aperture_radius = float(self._entry_value(self.aperture_radius_var))
            annulus_radii = (float(self._entry_value(self.annulus_inner_var)),
                             float(self._entry_value(self.annulus_outer_var)))
        except ValueError:
            self.write_to_log("Error: Aperture and annulus radii must be numbers.")
            return

        if not reduced_path_value:
            self.write_to_log("Error: Reduced images path is required.")
            return
        choices = self._photometry_radec_choices()
        if choices is None:
            return

        # main takes the RADEC files in B, V, R order, with None for a filter not in use
        chosen = dict(choices)
        radec_list = [chosen.get(filt) for filt in PHOTOMETRY_FILTERS]

        def photometry_task():
            self.write_to_log(f"Object Name: {obj_name_value}")
            self.write_to_log(f"Reduced Images Path: {reduced_path_value}")
            for filt in PHOTOMETRY_FILTERS:
                if filt in chosen:
                    self.write_to_log(f"RADEC File ({filt} Filter): {chosen[filt]}")
            try:
                multi_ap(
                    path=reduced_path_value, pipeline=False,
                    radec_list=radec_list,
                    obj_name=obj_name_value,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event,
                    aperture_radius=aperture_radius,
                    annulus_radii=annulus_radii
                )
                if self.cancel_event.is_set():
                    self.write_to_log("Multi-Aperture Photometry was canceled.")
                else:
                    self.write_to_log("Multi-Aperture Photometry completed successfully.")
                    filters = [f for f in PHOTOMETRY_FILTERS if f in chosen]
                    self.offer_results(reduced_path_value, [(
                        "View Light Curves",
                        lambda: self._show_light_curves(reduced_path_value, obj_name_value, filters))])
            except Exception as e:
                self.write_to_log(
                    f"An error occurred during Multi-Aperture Photometry: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(photometry_task)

    def run_radii_optimizer(self, reduced_images_path):
        """
        Load the first LIGHT frame and find starting radii on a worker thread,
        then open the interactive optimizer window back on the main thread.
        """
        reduced_path = self._path_value(reduced_images_path)

        # Use the first RADEC file given among the filters in use
        count = self.photometry_filter_count.get()
        radec_file = next((path for path in (self._path_value(entry)
                                             for _picker, entry in self.photometry_filter_rows[:count])
                           if path), None)

        if not reduced_path or not radec_file:
            self.write_to_log("Error: Reduced Images Path and at least one RADEC file are required for optimization.")
            return

        def prepare():
            import pandas as pd
            import ccdproc as ccdp
            from astropy.wcs import WCS
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            try:
                images_path = Path(reduced_path)
                if not images_path.exists():
                    self.write_to_log("Error: Reduced Images Path does not exist.")
                    return

                files = ccdp.ImageFileCollection(images_path)
                light_files = files.files_filtered(imagetyp="LIGHT")
                if len(light_files) == 0:
                    self.write_to_log("Error: No LIGHT images found in the given path.")
                    return

                self.write_to_log(f"Estimating starting radii from {light_files[0]}...")
                image_data, header = fits.getdata(images_path / light_files[0], header=True)

                # Read target from RADEC
                df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)
                target_ra = df[0].values[0]
                target_dec = df[1].values[0]

                if ":" in str(target_ra):
                    target_sky = SkyCoord(target_ra, target_dec, unit=(u.hourangle, u.deg), frame="icrs")
                else:
                    target_sky = SkyCoord(target_ra, target_dec, unit=(u.deg, u.deg), frame="icrs")

                target_pixel = WCS(header).world_to_pixel(target_sky)
                target_position = (float(target_pixel[0]), float(target_pixel[1]))

                read_noise = read_noise_from_header(header, DEFAULT_READ_NOISE)
                fwhm, best_aperture, (best_inner, best_outer) = auto_optimize_radii(
                    image_data, target_position, read_noise)
            except Exception as e:
                self.write_to_log(f"Error launching interactive optimizer: {e}")
                self.write_to_log(traceback.format_exc())
                return

            self.call_in_ui(self._open_radii_optimizer, image_data, target_position,
                            fwhm, best_aperture, best_inner, best_outer, read_noise)

        self.run_task(prepare)

    def _open_radii_optimizer(self, image_data, target_position, fwhm, best_aperture, best_inner, best_outer,
                              read_noise=DEFAULT_READ_NOISE):
        """Interactive window for adjusting the aperture and annulus with a live SNR readout."""
        opt_window = tk.Toplevel(self)
        opt_window.title("Interactive Radii Optimizer")
        opt_window.geometry("800x600")

        fig = Figure(figsize=(5, 5))
        ax = fig.add_subplot()

        # Crop image around target
        x_int, y_int = int(np.round(target_position[0])), int(np.round(target_position[1]))
        box_size = 50
        y_start = max(0, y_int - box_size)
        y_end = min(image_data.shape[0], y_int + box_size)
        x_start = max(0, x_int - box_size)
        x_end = min(image_data.shape[1], x_int + box_size)

        cutout = image_data[y_start:y_end, x_start:x_end]
        ax.imshow(cutout, cmap="gray", origin="lower",
                  vmin=np.percentile(cutout, 5), vmax=np.percentile(cutout, 95))

        # Aperture shapes relative to cutout center
        rel_target_position = (target_position[0] - x_start, target_position[1] - y_start)

        canvas = FigureCanvasTkAgg(fig, master=opt_window)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(opt_window, padx=10, pady=10, bg=BG_PANEL)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(controls_frame, text=f"Estimated FWHM: {fwhm:.2f} px", font=self.strong_font).pack(pady=10)

        snr_label = tk.Label(controls_frame, text="SNR: Calculating...", font=self.strong_font)
        snr_label.pack(pady=10)

        area_factor_label = tk.Label(controls_frame, text="Annulus Area: --x Aperture Area", font=self.label_font)
        area_factor_label.pack(pady=5)

        sliders = {}

        def update_snr(*args):
            # Sliders fire while they're still being built
            if len(sliders) < 3:
                return
            ap = float(sliders["ap"].get())
            inn = float(sliders["inn"].get())
            width = float(sliders["width"].get())

            # Keep the annulus outside the aperture
            if inn <= ap:
                inn = ap + 0.1
                sliders["inn"].set(inn)

            out = inn + width

            ap_area = np.pi * ap**2
            ann_area = np.pi * (out**2 - inn**2)
            area_factor = ann_area / ap_area if ap_area > 0 else 0
            area_factor_label.config(text=f"Annulus Area: {area_factor:.1f}x Aperture Area")

            try:
                snr, _, _, _ = calculate_target_snr(image_data, target_position, ap, (inn, out), read_noise)
                snr_label.config(text=f"SNR: {snr:.2f}")
            except Exception:
                snr_label.config(text="SNR: unavailable")

            for patch in list(ax.patches):
                patch.remove()
            CircularAperture(rel_target_position, r=ap).plot(color="red", lw=1.5, ax=ax)
            CircularAnnulus(rel_target_position, inn, out).plot(color="green", lw=1.5, ax=ax)
            canvas.draw_idle()

        def add_slider(key, label, low, high, start):
            tk.Label(controls_frame, text=label).pack()
            slider = tk.Scale(controls_frame, from_=low, to=high, orient=tk.HORIZONTAL,
                              resolution=0.1, command=update_snr)
            slider.set(round(start, 1))
            slider.pack(fill=tk.X, pady=5)
            sliders[key] = slider

        def get_width_for_area(factor, ap, inn):
            return np.sqrt((factor * np.pi * ap**2) / np.pi + inn**2) - inn

        ap_max = max(5.0, float(np.ceil(3 * fwhm)))
        add_slider("ap", "Aperture Radius (px):", 1.0, ap_max, best_aperture)

        add_slider("inn", "Inner Annulus Radius (px):",
                   float(np.floor(3 * fwhm)), float(np.ceil(5 * fwhm)), best_inner)

        width_min = max(1.0, float(np.floor(get_width_for_area(5.0, best_aperture, best_inner))))
        width_max = max(width_min + 1.0, float(np.ceil(get_width_for_area(25.0, best_aperture, best_inner))))
        add_slider("width", "Annulus Width (px):", width_min, width_max, best_outer - best_inner)

        def save_and_close():
            ap = float(sliders["ap"].get())
            inn = float(sliders["inn"].get())
            out = inn + float(sliders["width"].get())

            # The photometry panel may have been closed while this window was open
            for entry, value in ((self.aperture_radius_var, ap), (self.annulus_inner_var, inn),
                                 (self.annulus_outer_var, out)):
                if entry.winfo_exists():
                    self._set_entry_text(entry, str(round(value, 1)))

            self.write_to_log(f"Optimization complete. Saved Aperture: {round(ap, 1)}, "
                              f"Inner: {round(inn, 1)}, Outer: {round(out, 1)}")
            opt_window.destroy()

        self._button(controls_frame, "Save & Close", save_and_close).pack(pady=20)

        update_snr()

    # ------------------------------------------------------------------
    # Night Filters
    # ------------------------------------------------------------------
    def show_night_filters(self):
        """Display the Night Filters panel, which combines nightly AIJ files for one filter."""
        self.clear_right_frame()
        self._panel_title("Night Filters", "night_filters")

        files_label = tk.Label(self.right_frame, text="Night Files:", font=self.label_font,
                               bg=BG_PANEL, fg=FG_TEXT)
        files_label.grid(row=1, column=0, padx=(16, 10), pady=5, sticky="ne")
        self._tooltip(files_label, "One measurements file per night, all for the same filter (for example "
                                   "2018.09.18.APASS.B_datasubset.dat). Order doesn't matter; the rows are "
                                   "sorted by time.")

        list_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        list_frame.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="ew")
        list_frame.columnconfigure(0, weight=1)
        listbox = tk.Listbox(list_frame, height=6, font=self.label_font, selectmode="extended",
                             relief="flat", borderwidth=0, highlightthickness=1,
                             highlightbackground=BORDER, highlightcolor=ACCENT, activestyle="none")
        listbox.grid(row=0, column=0, sticky="ew")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        listbox.configure(yscrollcommand=scroll.set)
        self.night_files_list = listbox

        def add_paths(paths):
            existing = set(listbox.get(0, "end"))
            for path in paths:
                path = str(Path(path).expanduser())
                if path not in existing and Path(path).is_file():
                    listbox.insert("end", path)
                    existing.add(path)

        def add_files():
            # Some Tk builds return one string of brace-quoted paths rather than a tuple
            add_paths(self.tk.splitlist(filedialog.askopenfilenames(title="Select Night Files", parent=self)))

        def remove_selected():
            for index in reversed(listbox.curselection()):
                listbox.delete(index)

        buttons = tk.Frame(list_frame, bg=BG_PANEL)
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._button(buttons, "Add Files...", add_files, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        self._button(buttons, "Remove", remove_selected, style="Secondary.TButton").pack(side="left", padx=(0, 6))
        self._button(buttons, "Clear", lambda: listbox.delete(0, "end"),
                     style="Secondary.TButton").pack(side="left")
        listbox.bind("<Delete>", lambda _e: remove_selected())
        listbox.bind("<BackSpace>", lambda _e: remove_selected())

        if self.dnd_enabled:
            def handle_drop(event):
                add_paths(self.tk.splitlist(event.data))
                return event.action
            self.enable_drag_and_drop(listbox, handle_drop)

        output_folder = self.create_input_field(self.right_frame, "Output Folder:", _example_path("combined"),
                                                row=2, error_message="Please enter a folder.",
                                                browse_type="folder",
                                                tooltip="Where the combined files are saved.")
        output_name = self.create_input_field(self.right_frame, "Output Name:", "e.g. NSVS_896797_B", row=3,
                                              error_message="Please enter a name.",
                                              tooltip="Start of the file names: <name>_magnitudes.txt, and "
                                                      "<name>_flux.txt when the files have flux columns.")
        include_bjd = self._remember_var("include_bjd", tk.BooleanVar(master=self, value=False))
        self.create_checkbox(self.right_frame, "Include BJD_TDB column", include_bjd, row=4,
                             tooltip="Add BJD_TDB as the first column. Leave off for Find Minimum, O'Connell "
                                     "and Color Light Curve, which expect HJD first.")

        self.create_run_button(self.right_frame, self.run_night_filters, row=5, text="Combine",
                               output_folder=output_folder, output_name=output_name, include_bjd=include_bjd)
        self._log_section(6)

    def run_night_filters(self, output_folder, output_name, include_bjd):
        """Validate the Night Filters inputs, then combine the files on a worker thread."""
        files = list(self.night_files_list.get(0, "end"))
        folder = self._path_value(output_folder)
        name = self._entry_value(output_name)
        if not files:
            self.write_to_log("Error: Add at least one night's file.")
            return
        if not folder or not name:
            self.write_to_log("Error: An output folder and name are required.")
            return
        with_bjd = include_bjd.get()

        def combine():
            self.write_to_log(f"Combining {len(files)} night(s)...")
            try:
                result = combine_nights(files, folder, name, include_bjd=with_bjd,
                                        write_callback=self.write_to_log, cancel_event=self.cancel_event)
            except NightFileError as e:
                self.write_to_log(f"Error: {e}")
                return
            if result is None:
                return
            self.write_to_log("Night Filters completed successfully.")
            mag_path = result["magnitudes"]
            extras = [("Open Magnitude File", lambda: self._open_in_system(mag_path))]
            if not with_bjd:
                extras.append(("Find Minimum", lambda: self._send_to_find_minimum(mag_path)))
            self.offer_results(folder, extras)

        self.run_task(combine)

    def _send_to_find_minimum(self, light_curve):
        """Open Find Minimum with a combined light curve already filled in."""
        self.open_panel("find_min")
        self.find_min_filter_count.set(1)
        self._update_find_min_rows()
        self._set_entry_text(self.find_min_rows[0][1], light_curve)

    # ------------------------------------------------------------------
    # Find Minimum
    # ------------------------------------------------------------------
    def show_find_minimum(self):
        """Display the Find Minimum panel."""
        self.clear_right_frame()
        self._panel_title("Find Minimum", "find_min")

        obj_name = self.create_input_field(self.right_frame, "Object Name:", "e.g. NSVS 896797", row=1,
                                           error_message="Please enter an object name.",
                                           tooltip="Used to name the minimum files, e.g. NSVS_896797_B_min.txt.")

        count_label = tk.Label(self.right_frame, text="Number of Filters:", font=self.label_font,
                               bg=BG_PANEL, fg=FG_TEXT)
        count_label.grid(row=2, column=0, padx=(16, 10), pady=5, sticky="e")
        self._tooltip(count_label, "How many filters' light curves to load. In the window you can fit them "
                                   "together or one at a time.")
        self.find_min_filter_count = self._remember_var("filter_count",
                                                        tk.IntVar(master=self, value=len(PHOTOMETRY_FILTERS)))
        radio_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        radio_frame.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="w")
        for count in range(1, len(PHOTOMETRY_FILTERS) + 1):
            tk.Radiobutton(radio_frame, text=f"{count} Filter{'s' if count > 1 else ''}",
                           variable=self.find_min_filter_count, value=count, command=self._update_find_min_rows,
                           font=self.label_font, bg=BG_PANEL, fg=FG_TEXT).pack(side="left", padx=(0, 12))

        self.find_min_rows = []
        saved = self.settings.section("find_min")
        self._find_min_filters_picked = any(f"filter_{i}" in saved for i in range(len(PHOTOMETRY_FILTERS)))
        for i, default_filter in enumerate(PHOTOMETRY_FILTERS):
            label_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
            tk.Label(label_frame, text="Light Curve (", font=self.label_font, bg=BG_PANEL,
                     fg=FG_TEXT).pack(side="left")
            picker_var = self._remember_var(f"filter_{i}", tk.StringVar(master=self, value=default_filter))
            picker = ttk.Combobox(label_frame, values=PHOTOMETRY_FILTERS, state="readonly", width=3,
                                  font=self.label_font, textvariable=picker_var)
            picker.pack(side="left")
            tk.Label(label_frame, text=" Filter):", font=self.label_font, bg=BG_PANEL,
                     fg=FG_TEXT).pack(side="left")
            entry = self.create_input_field(self.right_frame, "", _example_path(f"{picker.get()}_magnitudes.txt"),
                                            row=3 + i, error_message="Please enter a light curve file.",
                                            browse_type="file", label_widget=label_frame,
                                            remember_as=f"curve_{i}",
                                            tooltip="Light curve for this filter: a Night Filters magnitude "
                                                    "file (HJD, magnitude, error) or a light curve CSV from "
                                                    "Multi-Aperture Photometry.")
            picker.bind("<<ComboboxSelected>>", lambda _e, p=picker, ent=entry: self._on_find_min_filter_picked(p, ent))
            self.find_min_rows.append((picker, entry))
        self._update_find_min_rows()

        output_folder = self.create_input_field(self.right_frame, "Output Folder:", "Same folder as the first file",
                                                row=6, browse_type="folder", optional=True,
                                                tooltip="Where the minimum files are saved. Left empty, they go "
                                                        "next to the first light curve. Minima already saved there "
                                                        "are loaded, so you can carry on later.")

        self.create_run_button(self.right_frame, self.run_find_minimum, row=7, text="Open Minimum Finder",
                               cancellable=False, obj_name=obj_name, output_folder=output_folder)
        self._log_section(8)

    def _update_find_min_rows(self):
        count = self.find_min_filter_count.get()
        if not self._find_min_filters_picked:
            for (picker, entry), filt in zip(self.find_min_rows, PHOTOMETRY_DEFAULTS[count]):
                picker.set(filt)
                self._set_placeholder(entry, _example_path(f"{filt}_magnitudes.txt"))
        for i, (_picker, entry) in enumerate(self.find_min_rows):
            self._set_field_visible(entry, i < count)

    def _on_find_min_filter_picked(self, picker, entry):
        self._find_min_filters_picked = True
        self._set_placeholder(entry, _example_path(f"{picker.get()}_magnitudes.txt"))

    def run_find_minimum(self, obj_name, output_folder):
        """Check the inputs and open the interactive Find Minimum window."""
        name = self._entry_value(obj_name)
        count = self.find_min_filter_count.get()
        choices = [(picker.get(), self._path_value(entry)) for picker, entry in self.find_min_rows[:count]]
        labels = [label for label, _ in choices]
        if not name:
            self.write_to_log("Error: Object name is required.")
            return
        if len(set(labels)) != len(labels):
            self.write_to_log("Error: Each filter can only be chosen once.")
            return
        missing = [label for label, path in choices if not path]
        if missing:
            self.write_to_log(f"Error: A light curve file is required for {', '.join(missing)}.")
            return
        not_found = [path for _label, path in choices if not Path(path).is_file()]
        if not_found:
            self.write_to_log(f"Error: {not_found[0]} doesn't exist.")
            return
        folder = self._path_value(output_folder) or str(Path(choices[0][1]).parent)

        try:
            window = MinimumFinderWindow(self, [path for _label, path in choices], labels, folder, name,
                                         log=self.write_to_log)
        except MinimumError as e:
            self.write_to_log(f"Error: {e}")
            return
        except Exception as e:
            self.write_to_log(f"Error: Find Minimum couldn't open: {type(e).__name__}: {e}")
            self.write_to_log(traceback.format_exc())
            return
        self.write_to_log(f"Opened Find Minimum with {len(window.nights)} night(s) in "
                          f"{', '.join(labels)}. Minima are saved in {folder}.")
        self._show_results(self._current_panel, folder, [])

    # ------------------------------------------------------------------
    # O-C plotting
    # ------------------------------------------------------------------
    def show_oc_plot(self):
        """Display the O-C Plotting panel."""
        self.clear_right_frame()
        self._panel_title("O-C Plotting", "oc")

        source_label = tk.Label(self.right_frame, text="Data Source:", font=self.label_font, bg=BG_PANEL,
                                fg=FG_TEXT)
        source_label.grid(row=1, column=0, padx=(16, 10), pady=5, sticky="e")
        self._tooltip(source_label, "BSUO/SARA averages B, V and R times of minimum. TESS uses one file. "
                                    "All Data merges O-C files you've already made.")
        self.oc_mode_var = self._remember_var("data_source", tk.StringVar(master=self, value="BSUO"))
        mode_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        mode_frame.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="w")
        for label, val in [("BSUO/SARA", "BSUO"), ("TESS", "TESS"), ("All Data", "ALL")]:
            tk.Radiobutton(
                mode_frame, text=label, variable=self.oc_mode_var, value=val,
                font=self.label_font, bg=BG_PANEL, fg=FG_TEXT,
                command=self._update_oc_fields
            ).pack(side="left", padx=(0, 12))

        self.oc_has_epoch_var = self._remember_var("has_epoch", tk.BooleanVar(master=self, value=True))
        self.oc_has_epoch_check = tk.Checkbutton(
            self.right_frame, text="I already have an Epoch value",
            variable=self.oc_has_epoch_var, font=self.label_font, bg=BG_PANEL, fg=FG_TEXT
        )
        self.oc_has_epoch_check.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="w")
        self._tooltip(self.oc_has_epoch_check, "Leave this off to use the first time of minimum in the data "
                                               "as the epoch.")

        # Shared fields
        self.oc_epoch_entry = self.create_input_field(
            self.right_frame, "Epoch (T0):", "e.g. 2457143.761819", row=3,
            validation_func=_is_float, error_message="Please enter a valid float.",
            tooltip="Reference time of primary minimum (HJD or BJD) that eclipse numbers count from."
        )
        self.oc_epoch_err_entry = self.create_input_field(
            self.right_frame, "Epoch Error:", "e.g. 0.0002803", row=4,
            validation_func=_is_float, error_message="Please enter a valid float.",
            tooltip="Uncertainty on the epoch, in days."
        )
        self.oc_period_entry = self.create_input_field(
            self.right_frame, "Period (days):", "e.g. 0.31297", row=5,
            validation_func=_is_float, error_message="Please enter a valid period.",
            tooltip="Orbital period in days. The linear fit reports a corrected period."
        )
        self.oc_output_entry = self.create_input_field(
            self.right_frame, "Output Folder:", _example_path("output"), row=6,
            error_message="Please enter an output folder path.", browse_type="folder",
            tooltip="Where the O-C table, plot and LaTeX tables are written."
        )

        # BSUO fields
        self.oc_b_file_entry = self.create_input_field(
            self.right_frame, "B Filter ToM File:", _example_path("B_min.txt"), row=7,
            error_message="Please enter a file path.", browse_type="file"
        )
        self.oc_v_file_entry = self.create_input_field(
            self.right_frame, "V Filter ToM File:", _example_path("V_min.txt"), row=8,
            error_message="Please enter a file path.", browse_type="file"
        )
        self.oc_r_file_entry = self.create_input_field(
            self.right_frame, "R Filter ToM File:", _example_path("R_min.txt"), row=9,
            error_message="Please enter a file path.", browse_type="file"
        )

        # TESS field
        self.oc_tess_file_entry = self.create_input_field(
            self.right_frame, "TESS ToM File:", _example_path("tess_min.txt"), row=7,
            error_message="Please enter a file path.", browse_type="file"
        )

        # All Data fields
        self.oc_nights_entry = self.create_input_field(
            self.right_frame, "Number of Files:", "e.g. 2", row=7,
            validation_func=lambda x: x.isdigit(), error_message="Please enter a whole number."
        )
        self.oc_all_files_entry = self.create_input_field(
            self.right_frame, "O-C Files (comma separated):",
            f"{_example_path('file1.txt')}, {_example_path('file2.txt')}", row=8,
            error_message="Please enter file paths separated by commas.", wide=True,
            tooltip="Paths to O-C files made earlier, separated by commas, in the format of "
                    "example_OC_table.txt."
        )

        self._update_oc_fields()

        self.create_run_button(self.right_frame, self.run_oc_plot, row=10)

        self._log_section(11)

    def _update_oc_fields(self):
        """Show only the inputs that the selected data source uses."""
        mode = self.oc_mode_var.get()
        groups = {
            "BSUO": [self.oc_b_file_entry, self.oc_v_file_entry, self.oc_r_file_entry],
            "TESS": [self.oc_tess_file_entry],
            "ALL": [self.oc_nights_entry, self.oc_all_files_entry],
        }
        for group_mode, entries in groups.items():
            for entry in entries:
                self._set_field_visible(entry, group_mode == mode)

        # All Data mode works the epoch out from the files themselves
        uses_epoch = mode != "ALL"
        for entry in (self.oc_epoch_entry, self.oc_epoch_err_entry):
            self._set_field_visible(entry, uses_epoch)
        if uses_epoch:
            self.oc_has_epoch_check.grid()
        else:
            self.oc_has_epoch_check.grid_remove()

    def run_oc_plot(self):
        """Validate the O-C inputs, then run the calculation and fit on a worker thread."""
        mode = self.oc_mode_var.get()
        period_str = self._entry_value(self.oc_period_entry)
        output = self._path_value(self.oc_output_entry)

        if not period_str:
            self.write_to_log("Error: Period is required.")
            return
        try:
            period = float(period_str)
        except ValueError:
            self.write_to_log("Error: Period must be a number.")
            return
        if not output:
            self.write_to_log("Error: Output folder is required.")
            return

        T0, T0_err = 0, 0
        if self.oc_has_epoch_var.get() and mode != "ALL":
            epoch_str = self._entry_value(self.oc_epoch_entry)
            epoch_err_str = self._entry_value(self.oc_epoch_err_entry)
            if not epoch_str or not epoch_err_str:
                self.write_to_log("Error: Epoch and Epoch Error are required.")
                return
            try:
                T0, T0_err = float(epoch_str), float(epoch_err_str)
            except ValueError:
                self.write_to_log("Error: Epoch and Epoch Error must be numbers.")
                return

        if mode == "BSUO":
            paths = [self._path_value(e) for e in
                     (self.oc_b_file_entry, self.oc_v_file_entry, self.oc_r_file_entry)]
            if not all(paths):
                self.write_to_log("Error: All three filter files are required.")
                return
        elif mode == "TESS":
            paths = [self._path_value(self.oc_tess_file_entry)]
            if not paths[0]:
                self.write_to_log("Error: TESS ToM file is required.")
                return
        else:
            nights_str = self._entry_value(self.oc_nights_entry)
            if not nights_str.isdigit():
                self.write_to_log("Error: Number of files must be a whole number.")
                return
            all_files_str = self._entry_value(self.oc_all_files_entry)
            paths = [str(Path(p.strip()).expanduser()) for p in all_files_str.split(",") if p.strip()]
            if len(paths) != int(nights_str):
                self.write_to_log(f"Error: Expected {nights_str} files, got {len(paths)}.")
                return

        def oc_plot():
            import pandas as pd

            try:
                if mode == "BSUO":
                    db, dv, dr = (pd.read_csv(p, header=None, sep=r"\s+") for p in paths)
                    outfile = BSUO(T0, T0_err, period, db, dv, dr, output,
                                   write_callback=self.write_to_log, cancel_event=self.cancel_event)
                elif mode == "TESS":
                    df = pd.read_csv(paths[0], header=None, sep=r"\s+")
                    outfile = TESS_OC(T0, T0_err, period, df, output,
                                      write_callback=self.write_to_log, cancel_event=self.cancel_event)
                else:
                    outfile = all_data(paths, period, output,
                                       write_callback=self.write_to_log, cancel_event=self.cancel_event)

                if outfile and not self.cancel_event.is_set():
                    data_fit(outfile, period, write_callback=self.write_to_log,
                             cancel_event=self.cancel_event)
                    self.write_to_log("O-C plotting completed successfully.")
                    plot = Path(outfile).with_suffix(".png")
                    extras = [("View O-C Plot", lambda: self._open_in_system(plot))] if plot.exists() else []
                    self.offer_results(output, extras)
            except Exception as e:
                self.write_to_log(f"Error during O-C plotting: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(oc_plot)

    # ------------------------------------------------------------------
    # Gaia query
    # ------------------------------------------------------------------
    def show_gaia_query(self):
        """Display the Gaia query panel."""
        self.clear_right_frame()
        self._panel_title("Gaia Search", "gaia")

        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):",
                                     "HH:MM:SS.SSSS", row=1,
                                     error_message="Please enter a RA.",
                                     tooltip="Target RA in hours (J2000), e.g. 00:28:27.97.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      error_message="Please enter a DEC.",
                                      tooltip="Target Dec in degrees (J2000), e.g. +78:57:42.66.")

        output_folder = self.create_input_field(self.right_frame, "Output Folder:",
                                                _example_path("gaia"), row=3,
                                                error_message="Please enter a folder.",
                                                browse_type="folder",
                                                tooltip="gaia_results.csv is saved here, with the closest "
                                                        "Gaia DR3 source first.")

        self.create_run_button(self.right_frame, self.run_gaia_query, row=4,
                               ra=ra, dec=dec, output_file=output_folder)

        self._log_section(5)

    def run_gaia_query(self, ra, dec, output_file):
        """Validate the Gaia inputs, then run the query on a worker thread."""
        ra_value = self._entry_value(ra)
        dec_value = self._entry_value(dec)
        output_path = self._path_value(output_file)

        if not all([ra_value, dec_value, output_path]):
            self.write_to_log("Error: All fields are required.")
            return

        def gaia_query():
            self.write_to_log(f"Running Gaia Query for RA: {ra_value}, DEC: {dec_value}")
            try:
                gaia(ra_input=ra_value, dec_input=dec_value, output_path=output_path,
                     write_callback=self.write_to_log, cancel_event=self.cancel_event)
                if not self.cancel_event.is_set():
                    results = Path(output_path) / "gaia_results.csv"
                    extras = [("Open Results", lambda: self._open_in_system(results))] if results.exists() else []
                    self.offer_results(output_path, extras)
            except Exception as e:
                self.write_to_log(f"An error occurred during Gaia Query: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(gaia_query)

    # ------------------------------------------------------------------
    # O'Connell effect
    # ------------------------------------------------------------------
    def show_oconnell_effect(self):
        """Display the O'Connell Effect panel."""
        self.clear_right_frame()
        self._panel_title("O'Connell Effect", "oconnell")

        filter_count_var = self._remember_var("filter_count", tk.IntVar(master=self, value=3))  # 3 filters unless changed
        file_path_vars = [self._remember_var(f"file_{i}", tk.StringVar(master=self)) for i in range(3)]
        file_entry_widgets = []

        def update_file_path_fields():
            for widgets in file_entry_widgets:
                for widget in widgets:
                    widget.destroy()
            file_entry_widgets.clear()

            for i in range(filter_count_var.get()):
                label = tk.Label(self.right_frame, text=f"File Path {i + 1}:", font=self.label_font,
                                 bg=BG_PANEL, fg=FG_TEXT)
                label.grid(row=3 + i, column=0, padx=(16, 10), pady=5, sticky="e")
                self._tooltip(label, "Light curve file for one filter (B, V or Cousins R), in the format "
                                     "of examples/test_B.txt.")

                entry_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
                entry_frame.grid(row=3 + i, column=1, padx=(0, 10), pady=5, sticky="ew")
                entry_frame.columnconfigure(0, weight=1)

                entry = tk.Entry(entry_frame, textvariable=file_path_vars[i], font=self.label_font, width=30,
                                 relief="flat", borderwidth=0, highlightthickness=1,
                                 highlightbackground=BORDER, highlightcolor=ACCENT)
                entry.grid(row=0, column=0, sticky="ew", ipady=3)

                def browse(var=file_path_vars[i]):
                    selected = filedialog.askopenfilename(title="Select File", parent=self)
                    if selected:
                        var.set(selected)

                self._button(entry_frame, "Browse", browse, style="Browse.TButton").grid(
                    row=0, column=1, padx=(5, 5), pady=2, sticky="e")

                if self.dnd_enabled:
                    def handle_drop(event, var=file_path_vars[i]):
                        paths = self.tk.splitlist(event.data)
                        if paths:
                            var.set(paths[0])
                        return event.action
                    self.enable_drag_and_drop(entry_frame, handle_drop)
                    self.enable_drag_and_drop(entry, handle_drop)

                file_entry_widgets.append((label, entry_frame))

        count_label = tk.Label(self.right_frame, text="Number of Filters:", font=self.label_font,
                               bg=BG_PANEL, fg=FG_TEXT)
        count_label.grid(row=1, column=0, padx=(16, 10), pady=5, sticky="e")
        self._tooltip(count_label, "How many filters' light curves to analyze. Each gets a file row below.")

        radio_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        radio_frame.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="w")

        for i in range(1, 4):
            tk.Radiobutton(
                radio_frame,
                text=f"{i} Filter{'s' if i > 1 else ''}",
                variable=filter_count_var,
                value=i,
                command=update_file_path_fields,
                font=self.label_font,
                bg=BG_PANEL,
                fg=FG_TEXT,
                anchor="w"
            ).pack(side="left", padx=(0, 12))

        update_file_path_fields()

        hjd_var = self.create_input_field(
            parent=self.right_frame, label_text="HJD:", placeholder_text="e.g. 2458403.58763", row=6,
            validation_func=_is_float, error_message="Please enter a valid HJD.",
            tooltip="Epoch: the HJD of a primary minimum, used to phase the light curves."
        )

        period_var = self.create_input_field(
            parent=self.right_frame, label_text="Period:", placeholder_text="e.g. 0.3175", row=7,
            validation_func=_is_float, error_message="Please enter a valid Period.",
            tooltip="Orbital period in days."
        )

        obj_name_var = self.create_input_field(
            parent=self.right_frame, label_text="System Name:", placeholder_text="e.g. NSVS_896797", row=8,
            error_message="Please enter a System Name.",
            tooltip="Names the output plot (.pdf) and LaTeX table (.txt)."
        )

        output_var = self.create_input_field(
            parent=self.right_frame, label_text="Output Folder:", placeholder_text=_example_path("oconnell"),
            row=9, error_message="Please enter a file pathway.", browse_type="folder",
            tooltip="Where the plot and table are saved."
        )

        self.create_run_button(self.right_frame, self.run_oconnell_effect, row=10,
                               text="Run O'Connell Effect",
                               filter_count_var=filter_count_var, file_path_vars=file_path_vars,
                               hjd=hjd_var, period=period_var, obj_name=obj_name_var, output_file=output_var)

        self._log_section(11)

    def run_oconnell_effect(self, filter_count_var, file_path_vars, hjd, period, obj_name, output_file):
        """Validate the O'Connell inputs, then run the calculation on a worker thread."""
        filter_count = filter_count_var.get()
        file_paths = [str(Path(var.get().strip()).expanduser()) if var.get().strip() else ""
                      for var in file_path_vars[:filter_count]]
        hjd_value = self._entry_value(hjd)
        period_value = self._entry_value(period)
        obj_name_value = self._entry_value(obj_name)
        output_folder = self._path_value(output_file)

        def input_error(log_message, dialog_title, dialog_message):
            self.write_to_log(log_message)
            messagebox.showerror(dialog_title, dialog_message, parent=self)

        if not all(file_paths):
            return input_error("Error: Missing file paths for selected filters.", "Input Error",
                               "Please provide file paths for all selected filters.")
        for file_path in file_paths:
            if not Path(file_path).exists():
                return input_error(f"Error: File does not exist - {file_path}", "File Error",
                                   f"The file {file_path} does not exist.")
        if not _is_float(hjd_value):
            return input_error(f"Error: Invalid HJD value - {hjd_value!r}", "Input Error",
                               "Please enter the Heliocentric Julian Date (HJD) as a number.")
        if not _is_float(period_value):
            return input_error(f"Error: Invalid period - {period_value!r}", "Input Error",
                               "Please enter the period of the system as a number.")
        if not obj_name_value:
            return input_error("Error: System Name is required.", "Input Error",
                               "Please enter the name of the system.")
        if not output_folder:
            return input_error("Error: Output folder is required.", "Input Error",
                               "Please provide an output folder.")

        def oconnell_task():
            self.write_to_log("Starting O'Connell Effect calculation with the following inputs:")
            self.write_to_log(f"Number of Filters: {filter_count}")
            self.write_to_log(f"File Paths: {', '.join(file_paths)}")
            self.write_to_log(f"HJD: {hjd_value}")
            self.write_to_log(f"Period: {period_value}")
            self.write_to_log(f"Output Folder: {output_folder}")
            try:
                oconnell(filepath=output_folder, filter_files=list(file_paths),
                         obj_name=obj_name_value, period=float(period_value),
                         hjd=float(hjd_value), write_callback=self.write_to_log,
                         cancel_event=self.cancel_event)
                if self.cancel_event.is_set():
                    self.write_to_log("O'Connell Effect calculation was canceled.")
                else:
                    self.write_to_log("O'Connell Effect calculation completed successfully.")
                    plot = Path(output_folder) / f"{obj_name_value}.pdf"
                    extras = [("View Plot", lambda: self._open_in_system(plot))] if plot.exists() else []
                    self.offer_results(output_folder, extras)
            except Exception as e:
                self.write_to_log(
                    f"An error occurred during O'Connell Effect calculation: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(oconnell_task)

    # ------------------------------------------------------------------
    # Color light curve
    # ------------------------------------------------------------------
    def show_color_light_curve(self):
        """Display the Color Light Curve panel."""
        self.clear_right_frame()
        self._panel_title("Color Light Curve", "color")

        bfile = self.create_input_field(self.right_frame, "B-band File:",
                                        _example_path("b_band.txt"), row=1,
                                        error_message="Please select a valid B-band file.",
                                        browse_type="file",
                                        tooltip="Johnson B light curve: time, magnitude and error columns.")

        vfile = self.create_input_field(self.right_frame, "V-band File:",
                                        _example_path("v_band.txt"), row=2,
                                        error_message="Please select a valid V-band file.",
                                        browse_type="file",
                                        tooltip="Johnson V light curve in the same format as the B file.")

        period = self.create_input_field(self.right_frame, "Period (days):",
                                         "e.g. 1.234", row=3,
                                         validation_func=_is_float,
                                         error_message="Enter a numeric period.",
                                         tooltip="Orbital period in days.")

        hjd = self.create_input_field(self.right_frame, "HJD (Epoch):",
                                      "e.g. 2459000.123", row=4,
                                      validation_func=_is_float,
                                      error_message="Enter a numeric HJD.",
                                      tooltip="HJD of a primary minimum, used to phase the light curves.")

        outname = self.create_input_field(self.right_frame, "Output Image Name:",
                                          "color_curve.png", row=5,
                                          validation_func=lambda x: x.strip().lower().endswith(".png"),
                                          error_message="Output must be a PNG file.",
                                          default="color_curve.png", wide=True,
                                          tooltip="PNG file for the plot. A bare name is saved in the folder "
                                                  "the app was started from, so give a full path to save it "
                                                  "somewhere specific.")

        self.create_run_button(self.right_frame, self.run_color_light_curve, row=6,
                               bfile=bfile, vfile=vfile, period=period, hjd=hjd, outname=outname)

        self._log_section(7)

    def run_color_light_curve(self, bfile, vfile, period, hjd, outname):
        """Validate the color light curve inputs, then run it on a worker thread."""
        b = self._path_value(bfile)
        v = self._path_value(vfile)
        out = self._path_value(outname)
        try:
            per = float(self._entry_value(period))
            epoch = float(self._entry_value(hjd))
        except ValueError:
            self.write_to_log("Error: Period and HJD must be numbers.")
            return
        if not b or not v:
            self.write_to_log("Error: Both B-band and V-band files are required.")
            return

        def task():
            self.write_to_log("Starting color light curve processing...")
            self.write_to_log(f"B-band file: {b}")
            self.write_to_log(f"V-band file: {v}")
            self.write_to_log(f"Period: {per}")
            self.write_to_log(f"HJD Epoch: {epoch}")
            self.write_to_log(f"Output: {out}")
            try:
                color_plot(Bfile=b, Vfile=v, Epoch=epoch, period=per, outName=out, save=True,
                           write_callback=self.write_to_log, cancel_event=self.cancel_event)
                if self.cancel_event.is_set():
                    self.write_to_log("Color light curve was canceled.")
                else:
                    self.write_to_log("Color light curve completed successfully.")
                    plot = Path(out).resolve()
                    extras = [("View Plot", lambda: self._open_in_system(plot))] if plot.exists() else []
                    self.offer_results(str(plot.parent), extras)
            except Exception as e:
                self.write_to_log(f"Error: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(task)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def quit_program(self):
        """Quit the program after confirming, stopping any running task first."""
        if self._task_running:
            if not messagebox.askyesno("Exit", "A task is still running. Cancel it and exit?", parent=self):
                return
            self.cancel_event.set()
        elif not messagebox.askyesno("Exit", "Are you sure you want to exit?", parent=self):
            return
        self._save_panel_settings()
        if self._ui_poll_id is not None:
            self.after_cancel(self._ui_poll_id)
            self._ui_poll_id = None
        self.destroy()


def main():
    _enable_windows_dpi_awareness()

    # Worker threads save their plots through pyplot. Agg keeps pyplot from
    # ever opening a GUI window off the main thread, which crashes on macOS.
    # Plots shown in the app are embedded Figure canvases, so they still work.
    matplotlib.use("Agg")

    app = ProgramLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
