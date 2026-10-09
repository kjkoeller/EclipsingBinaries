"""
A GUI to centralize all the scripts and capabilities of this package for ease of use for the user and
making it more convenient to use and access than a command line or individual scripts.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 10/09/2026
"""

import queue
import sys
import threading
import traceback
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
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
from .multi_aperture_photometry import main as multi_ap, auto_optimize_radii, calculate_target_snr
from .gaia import target_star as gaia
from .OC_plot import BSUO, TESS_OC, all_data, data_fit
from .OConnell import main as oconnell
from .color_light_curve import color_plot

try:
    __version__ = version("EclipsingBinaries")
except PackageNotFoundError:
    __version__ = "unknown"

IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# The GUI always draws with a light palette. Every color is set explicitly
# because on macOS in Dark Mode Tk's default text color turns white, which
# made labels and entry text vanish against our white backgrounds.
BG_APP = "#f5f5f5"
BG_PANEL = "#ffffff"
FG_TEXT = "#000000"
FG_PLACEHOLDER = "#808080"
FG_ERROR = "#cc0000"
BG_ERROR = "#ffe6e6"
ACCENT = "#003366"
ACCENT_ACTIVE = "#00509e"
ACCENT_DISABLED = "#8a9bb0"

# Same constant tkinterdnd2 uses for file drops. Kept here so the module
# still imports when tkinterdnd2 can't load.
DND_FILES = "DND_Files"

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
        self._action_buttons = []
        self.log_area = None

        self.dnd_enabled = self._try_enable_drag_and_drop()

        # Get screen dimensions
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # Set window size as 75%x75% of screen dimensions
        self.window_width = int(self.screen_width * 0.75)
        self.window_height = int(self.screen_height * 0.75)
        self.minsize(1000, 700)  # Minimum size for usability

        # Center the window
        self.center_window()

        # Window setup
        self.title("EclipsingBinaries")
        self.configure(bg=BG_APP)
        self._apply_color_defaults()

        # Fonts are sized from the window width. On Windows with DPI awareness
        # on, the width comes back in physical pixels while font sizes are in
        # points, so divide out the scaling to keep text from doubling up.
        dpi_factor = self.winfo_fpixels("1i") / 96 if IS_WINDOWS else 1.0
        logical_width = self.window_width / max(dpi_factor, 1.0)
        self.header_font = ("Helvetica", int(min(28, max(16, logical_width // 50))), "bold")
        self.label_font = ("Helvetica", int(min(14, max(10, logical_width // 100))))
        self.button_font = ("Helvetica", int(min(16, max(12, logical_width // 80))))

        self._setup_styles()

        # Bind click event to remove focus
        self.bind("<Button-1>", self.remove_focus)

        # Configure rows and columns for resizing
        self.rowconfigure(0, weight=1)  # Header
        self.rowconfigure(1, weight=8)  # Main content
        self.rowconfigure(2, weight=1)  # Footer
        self.columnconfigure(0, weight=1)  # Full GUI width

        # Create UI components
        self.create_header()
        self.create_layout()
        self.create_menu()

        # Footer
        footer_label = tk.Label(
            self.left_frame,
            text="Developed by Kyle Koeller | Eclipsing Binaries Research",
            font=("Helvetica", 10),
            bg=BG_APP,
        )
        footer_label.pack(side="bottom", pady=5)

        # Closing the window and Cmd-Q on macOS both go through the exit prompt
        self.protocol("WM_DELETE_WINDOW", self.quit_program)
        if IS_MAC:
            self.createcommand("tk::mac::Quit", self.quit_program)

        self._ui_poll_id = self.after(UI_POLL_MS, self._drain_ui_queue)

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
        self.option_add("*Scale.troughColor", "#dddddd")
        self.option_add("*TCombobox*Listbox.background", "white")
        self.option_add("*TCombobox*Listbox.foreground", FG_TEXT)

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
            "Browse.TButton", font=("Helvetica", 9), padding=(6, 1),
            background="#f0f0f0", foreground=FG_TEXT, borderwidth=1,
        )
        self.style.map("Browse.TButton", background=[("active", "#e0e0e0")])

        self.style.configure("TCombobox", fieldbackground="white", foreground=FG_TEXT)

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
        if self.log_area is not None and self.log_area.winfo_exists():
            self.log_area.insert("end", message + "\n")
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
        button greys out and Cancel becomes active until it finishes.
        """
        if self._task_running:
            messagebox.showinfo("Task Running",
                                "Another task is still running. Wait for it to finish or cancel it first.",
                                parent=self)
            return
        self.cancel_event.clear()
        self._task_running = True

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

    def _sync_task_buttons(self):
        busy = self._task_running
        for button in [self._run_button, *self._action_buttons]:
            if button is not None and button.winfo_exists():
                button.state(["disabled"] if busy else ["!disabled"])
        if self._cancel_button is not None and self._cancel_button.winfo_exists():
            self._cancel_button.state(["!disabled"] if busy else ["disabled"])

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

    def _make_validator(self, entry, error_label, error_message, validation_func):
        """Build and return the validate_input function for an entry field."""
        def validate_input():
            value = self._entry_value(entry)
            if not value:
                error_label.config(text=error_message or "This field is required.")
                entry.config(bg=BG_ERROR)
            elif validation_func and not validation_func(value):
                error_label.config(text=error_message)
                entry.config(bg=BG_ERROR)
            else:
                error_label.config(text="")
                entry.config(bg="white")
        return validate_input

    def _make_placeholder_bindings(self, entry, placeholder_text, validate_input):
        """Bind focus-in, focus-out, and placeholder behaviour to an entry field."""
        def show_placeholder():
            entry.delete(0, "end")
            entry.insert(0, placeholder_text)
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
                           validation_func=None, error_message="", browse_type=None, default=None):
        """
        Create a labeled input field with placeholder functionality, validation, and inline error message.
        Includes an optional browse button integrated into the entry field.

        The placeholder is only a hint and is never read back as a value. Pass
        default for fields where leaving them empty should mean a specific value.
        """
        label = tk.Label(parent, text=label_text, font=self.label_font, bg=BG_PANEL)
        label.grid(row=row, column=0, padx=10, pady=5, sticky="e")

        entry_frame = tk.Frame(parent, bg=BG_PANEL, highlightthickness=1, highlightbackground="#cccccc")
        entry_frame.grid(row=row, column=1, padx=10, pady=5, sticky="w")

        entry = tk.Entry(entry_frame, width=25, font=self.label_font, borderwidth=0)
        entry.grid(row=0, column=0, sticky="w", padx=(5, 0))
        entry._default = default

        error_label = tk.Label(parent, text="", font=("Helvetica", 9), fg=FG_ERROR, bg=BG_PANEL)
        error_label.grid(row=row, column=2, padx=(5, 10), sticky="w")

        # Kept so panels can hide or show the whole row
        entry._field_widgets = (label, entry_frame, error_label)

        validate_input = self._make_validator(entry, error_label, error_message, validation_func)
        self._make_placeholder_bindings(entry, placeholder_text, validate_input)
        self._make_browse_button(entry_frame, entry, browse_type, validate_input)
        handler = self._make_drop_handler(entry, validate_input)
        self.enable_drag_and_drop(entry_frame, handler)
        self.enable_drag_and_drop(entry, handler)

        return entry

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
        """Create the header section"""
        header_frame = tk.Frame(self, bg=ACCENT)
        header_frame.pack(fill="x")

        tk.Label(header_frame, text="EclipsingBinaries", font=self.header_font, fg="white", bg=ACCENT).pack(pady=10)
        tk.Label(header_frame, text="Refer to the GitHub README for more details:", font=self.label_font, fg="white",
                 bg=ACCENT).pack()
        tk.Label(header_frame, text="https://github.com/kjkoeller/EclipsingBinaries/", font=self.label_font,
                 fg="#aadfff", bg=ACCENT).pack(pady=5)

    def create_layout(self):
        """Create the main layout with left and right frames"""
        self.left_frame = tk.Frame(self, bg=BG_APP)
        self.left_frame.place(relx=0, rely=0.2, relwidth=0.3, relheight=0.8)

        self.right_frame = tk.Frame(self, bg=BG_PANEL, relief="groove", bd=2)
        self.right_frame.place(relx=0.3, rely=0.2, relwidth=0.7, relheight=0.8)

    def create_menu(self):
        """Create the menu bar and the left-side options menu."""
        menubar = tk.Menu(self)

        if IS_MAC:
            # macOS keeps About and Quit in the application menu. Quit is
            # wired up through tk::mac::Quit in __init__.
            app_menu = tk.Menu(menubar, name="apple", tearoff=0)
            app_menu.add_command(label="About EclipsingBinaries", command=self.show_about)
            menubar.add_cascade(menu=app_menu)
        else:
            file_menu = tk.Menu(menubar, tearoff=0)
            file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.quit_program)
            menubar.add_cascade(label="File", menu=file_menu)
            self.bind_all("<Control-q>", lambda e: self.quit_program())

        help_menu = tk.Menu(menubar, name="help", tearoff=0)
        help_menu.add_command(label="Help Contents", command=self.open_help_window)
        if not IS_MAC:
            help_menu.add_separator()
            help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

        # Left-side options menu
        options = [
            ("IRAF Reduction", self.show_iraf_reduction),
            ("Find Minimum (WIP)", self.dummy_action),
            ("TESS Database Search/Download", self.show_tess_search),
            ("AIJ Comparison Star Selector", self.show_aij_comparison_selector),
            ("Multi-Aperture Calculation", self.show_multi_aperture_photometry),
            ("BSUO or SARA/TESS Night Filters", self.dummy_action),
            ("O-C Plotting", self.show_oc_plot),
            ("Gaia Search", self.show_gaia_query),
            ("O'Connell Effect", self.show_oconnell_effect),
            ("Color Light Curve", self.show_color_light_curve),
            ("Close Program", self.quit_program),
        ]

        for option, command in options:
            self.create_menu_button(option, command)

    def create_menu_button(self, text, command):
        """Create a menu button"""
        self._button(self.left_frame, text, command).pack(pady=5, padx=10, fill="x")

    def open_help_window(self):
        """Open a separate Help window."""
        help_window = tk.Toplevel(self)
        help_window.title("Help Contents")
        help_window.geometry("600x400")
        help_window.configure(bg=BG_APP)

        help_content = (
            "Welcome to the Help Window!\n\n"
            "Here you can find brief descriptions on each of the currently available options.\n"
            "\nFeatures:\n"
            "- IRAF Reduction: Process raw astronomical images.\n"
            "- TESS Search: Retrieve TESS sector data.\n"
            "- AIJ Comparison: Select comparison stars.\n"
            "- Multi Aperture Photometry: Analysis science images with multi aperture photometry.\n"
            "- Gaia Search: Query Gaia for numerous variables related to an eclipsing binary star.\n"
            "- O'Connell Effect: Calculate light curve effects.\n"
            "\nFor more information, visit the GitHub repository:\n"
            "https://github.com/kjkoeller/EclipsingBinaries/"
        )

        help_label = tk.Label(help_window, text=help_content, font=("Helvetica", 10), justify="left",
                              wraplength=550, bg=BG_APP)
        help_label.pack(pady=20, padx=20, anchor="w")

    def show_about(self):
        """Display an About dialog."""
        messagebox.showinfo(
            "About EclipsingBinaries",
            "EclipsingBinaries\n\n"
            f"Version: {__version__}\n"
            "Author: Kyle Koeller\n\n"
            "For support, visit the GitHub repository:\n"
            "https://github.com/kjkoeller/EclipsingBinaries/",
            parent=self,
        )

    def clear_right_frame(self):
        """Clear the right frame by destroying all its widgets and resetting its grid."""
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        self.log_area = None
        self._run_button = None
        self._cancel_button = None
        self._action_buttons = []
        cols, rows = self.right_frame.grid_size()
        for r in range(rows):
            self.right_frame.grid_rowconfigure(r, weight=0)
        for c in range(cols):
            self.right_frame.grid_columnconfigure(c, weight=0)

    def create_checkbox(self, parent, text, variable, row):
        """Create a checkbox with alignment"""
        tk.Checkbutton(parent, text=text, variable=variable, font=self.label_font, bg=BG_PANEL).grid(
            row=row, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
        )

    def create_scrollbar_and_log(self, row):
        # Create a frame to hold the log area and scrollbar
        log_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        log_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        self.log_area = tk.Text(log_frame, wrap="word", height=12, font=("Helvetica", 10))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_area.yview)
        self.log_area.configure(yscrollcommand=scrollbar.set)

        self.log_area.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.right_frame.grid_rowconfigure(row, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

    def create_run_button(self, parent, action, row, text="Run", **kwargs):
        """Create the Run and Cancel buttons for a panel."""
        button_frame = tk.Frame(parent, bg=BG_PANEL)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)

        self._run_button = self._button(button_frame, text, lambda: action(**kwargs))
        self._run_button.pack(side="left", padx=5)

        self._cancel_button = self._button(button_frame, "Cancel", self.cancel_task)
        self._cancel_button.pack(side="left", padx=5)

        self._sync_task_buttons()

    def _panel_title(self, text):
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)
        tk.Label(self.right_frame, text=text, font=self.header_font, bg=BG_PANEL).grid(
            row=0, column=0, columnspan=2, pady=10, sticky="ew"
        )

    def _log_section(self, row):
        tk.Label(self.right_frame, text="Output Log:", font=self.label_font, bg=BG_PANEL).grid(
            row=row, column=0, columnspan=2, pady=5
        )
        self.create_scrollbar_and_log(row + 1)

    # ------------------------------------------------------------------
    # IRAF Reduction
    # ------------------------------------------------------------------
    def show_iraf_reduction(self):
        """Display the IRAF reduction panel."""
        self.clear_right_frame()
        self._panel_title("IRAF Reduction")

        raw_images_path = self.create_input_field(self.right_frame, "Raw Images Path:",
                                                  _example_path("raw_images"), row=1,
                                                  error_message="File path cannot be empty.",
                                                  browse_type="folder")

        calibrated_images_path = self.create_input_field(self.right_frame, "Calibrated Images Path:",
                                                         _example_path("calibrated_images"), row=2,
                                                         error_message="File path cannot be empty.",
                                                         browse_type="folder")

        location = self.create_input_field(self.right_frame, "Location:",
                                           "BSUO (default), CTIO, KPNO, etc.", row=3,
                                           error_message="Please enter an observatory.",
                                           default="BSUO")

        dark_bool_var = tk.BooleanVar(value=True)
        self.create_checkbox(self.right_frame, "Use Dark Frames", dark_bool_var, row=4)

        overscan = self.create_input_field(self.right_frame, "Overscan Region:",
                                           "[2073:2115, :]", row=5,
                                           error_message="Please enter at least [:,:].",
                                           default="[2073:2115, :]")

        trim = self.create_input_field(self.right_frame, "Trim Region:",
                                       "[20:2060, 12:2057]", row=6,
                                       error_message="Please enter at least [:,:].",
                                       default="[20:2060, 12:2057]")

        self._button(self.right_frame, "Open Bias Image", self.open_bias_image).grid(
            row=7, column=0, columnspan=2, pady=10)

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
                    self.notify("info", "Cancelled", "IRAF Reduction was cancelled.")
                else:
                    self.notify("info", "Success", "IRAF Reduction completed successfully!")
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
        self._panel_title("TESS Database Search")

        system_name = self.create_input_field(self.right_frame, "System Name:",
                                              "e.g. NSVS 896797", row=1,
                                              error_message="Please enter a system name.")

        download_path = self.create_input_field(self.right_frame, "Download Path:",
                                                _example_path("download"), row=2,
                                                error_message="Please enter a file pathway.",
                                                browse_type="folder")

        download_all_var = tk.BooleanVar(value=False)

        # Placeholders for widgets that appear when a specific sector is wanted
        self.sector_dropdown = None
        self.retrieve_button = None
        self.sector_label = None

        tk.Checkbutton(
            self.right_frame, text="Download Specific Sector", variable=download_all_var,
            font=self.label_font, bg=BG_PANEL,
            command=lambda: self.toggle_sector_options(download_all_var, system_name)
        ).grid(row=3, column=0, columnspan=2, pady=5)

        self.create_run_button(self.right_frame, self.run_tess_search, row=6,
                               system_name=system_name,
                               download_path=download_path,
                               download_all_var=download_all_var)

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

    def toggle_sector_options(self, download_all_var, system_name):
        """Show or hide the sector dropdown, 'Select Specific Sector' label, and Retrieve Sectors button."""
        if download_all_var.get():
            if not self.sector_label:
                self.sector_label = tk.Label(self.right_frame, text="Select Specific Sector:",
                                             font=self.label_font, bg=BG_PANEL)
                self.sector_label.grid(row=5, column=0, sticky="e")

            if not self.sector_dropdown:
                self.sector_dropdown = ttk.Combobox(self.right_frame, state="readonly",
                                                    values=[], font=self.label_font)
                self.sector_dropdown.grid(row=5, column=1, padx=10, pady=5, sticky="w")

            if not self.retrieve_button:
                self.retrieve_button = self._button(
                    self.right_frame, "Retrieve Sectors",
                    lambda: self.retrieve_sectors(system_name), action=True)
                self.retrieve_button.grid(row=4, column=0, columnspan=2, pady=10)
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

    def run_tess_search(self, system_name, download_path, download_all_var):
        """Validate the TESS inputs, then run the search and download on a worker thread."""
        system_name_value = self._entry_value(system_name)
        download_path_value = self._path_value(download_path)
        download_all = download_all_var.get()

        specific_sector_value = None
        if download_all:  # Specific sector mode
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
            self.write_to_log(f"Download All Sectors: {'Yes' if download_all else 'No'}")
            if specific_sector_value is not None:
                self.write_to_log(f"Specific Sector: {specific_sector_value}")
            try:
                run_tess_search(
                    system_name=system_name_value, download_all=download_all,
                    specific_sector=specific_sector_value, download_path=download_path_value,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event
                )
                if self.cancel_event.is_set():
                    self.notify("info", "Cancelled", "TESS Database Search was canceled.")
                else:
                    self.notify("info", "Success", "TESS Database Search completed successfully.")
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
        self._panel_title("AIJ Comparison Star Selector")

        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):",
                                     "HH:MM:SS.SSSS", row=1,
                                     error_message="Please enter a RA.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      error_message="Please enter a DEC.")

        folder_path = self.create_input_field(self.right_frame, "Data Save Folder Path:",
                                              _example_path("download"), row=3,
                                              error_message="Please enter a file pathway.",
                                              browse_type="folder")

        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "e.g. NSVS 896797", row=4,
                                           error_message="Please enter the object name.")

        science_image = self.create_input_field(self.right_frame, "Science Image File:",
                                                _example_path("calibrated_images", "image.fits"), row=5,
                                                error_message="Please enter a file pathway.",
                                                browse_type="file")

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
        self._panel_title("Multi-Aperture Photometry")

        obj_name = self.create_input_field(self.right_frame, "Object Name:",
                                           "e.g. NSVS 896797", row=1,
                                           error_message="Please enter an object name.")

        reduced_images_path = self.create_input_field(self.right_frame, "Reduced Images Path:",
                                                      _example_path("reduced_images"), row=2,
                                                      error_message="Please enter a file pathway.",
                                                      browse_type="folder")

        radec_b_file = self.create_input_field(self.right_frame, "RADEC File (B Filter):",
                                               _example_path("B.radec"), row=3,
                                               error_message="Please enter a file pathway with file name",
                                               browse_type="file")

        radec_v_file = self.create_input_field(self.right_frame, "RADEC File (V Filter):",
                                               _example_path("V.radec"), row=4,
                                               error_message="Please enter a file pathway with file name.",
                                               browse_type="file")

        radec_r_file = self.create_input_field(self.right_frame, "RADEC File (R Filter):",
                                               _example_path("R.radec"), row=5,
                                               error_message="Please enter a file pathway with file name.",
                                               browse_type="file")

        self.aperture_radius_var = self.create_input_field(self.right_frame, "Aperture Radius (px):",
                                                           "20", row=6, validation_func=_is_float,
                                                           error_message="Please enter a numeric radius.",
                                                           default="20")

        self.annulus_inner_var = self.create_input_field(self.right_frame, "Inner Annulus Radius (px):",
                                                         "30", row=7, validation_func=_is_float,
                                                         error_message="Please enter a numeric inner radius.",
                                                         default="30")

        self.annulus_outer_var = self.create_input_field(self.right_frame, "Outer Annulus Radius (px):",
                                                         "50", row=8, validation_func=_is_float,
                                                         error_message="Please enter a numeric outer radius.",
                                                         default="50")

        self._button(self.right_frame, "Optimize Radii",
                     lambda: self.run_radii_optimizer(reduced_images_path, radec_b_file,
                                                      radec_v_file, radec_r_file),
                     action=True).grid(row=9, column=0, columnspan=2, pady=10)

        self.create_run_button(self.right_frame, self.run_multi_aperture_photometry, row=10,
                               obj_name=obj_name,
                               reduced_images_path=reduced_images_path,
                               radec_b_file=radec_b_file,
                               radec_v_file=radec_v_file,
                               radec_r_file=radec_r_file)

        self._log_section(11)

    def run_multi_aperture_photometry(self, obj_name, reduced_images_path,
                                      radec_b_file, radec_v_file, radec_r_file):
        """Validate the photometry inputs, then run it on a worker thread."""
        obj_name_value = self._entry_value(obj_name)
        reduced_path_value = self._path_value(reduced_images_path)
        radec_b_path = self._path_value(radec_b_file)
        radec_v_path = self._path_value(radec_v_file)
        radec_r_path = self._path_value(radec_r_file)

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
        if not (radec_b_path and radec_v_path and radec_r_path):
            self.write_to_log("Error: RADEC files for all filters are required.")
            return

        def photometry_task():
            self.write_to_log(f"Object Name: {obj_name_value}")
            self.write_to_log(f"Reduced Images Path: {reduced_path_value}")
            self.write_to_log(f"RADEC File (B Filter): {radec_b_path}")
            self.write_to_log(f"RADEC File (V Filter): {radec_v_path}")
            self.write_to_log(f"RADEC File (R Filter): {radec_r_path}")
            try:
                multi_ap(
                    path=reduced_path_value, pipeline=False,
                    radec_list=[radec_b_path, radec_v_path, radec_r_path],
                    obj_name=obj_name_value,
                    write_callback=self.write_to_log, cancel_event=self.cancel_event,
                    aperture_radius=aperture_radius,
                    annulus_radii=annulus_radii
                )
                if self.cancel_event.is_set():
                    self.write_to_log("Multi-Aperture Photometry was canceled.")
                else:
                    self.write_to_log("Multi-Aperture Photometry completed successfully.")
            except Exception as e:
                self.write_to_log(
                    f"An error occurred during Multi-Aperture Photometry: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(photometry_task)

    def run_radii_optimizer(self, reduced_images_path, radec_b_file, radec_v_file, radec_r_file):
        """
        Load the first LIGHT frame and find starting radii on a worker thread,
        then open the interactive optimizer window back on the main thread.
        """
        reduced_path = self._path_value(reduced_images_path)

        # Pick the first provided RADEC file
        radec_file = None
        for entry in (radec_b_file, radec_v_file, radec_r_file):
            value = self._path_value(entry)
            if value:
                radec_file = value
                break

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

                fwhm, best_aperture, (best_inner, best_outer) = auto_optimize_radii(image_data, target_position)
            except Exception as e:
                self.write_to_log(f"Error launching interactive optimizer: {e}")
                self.write_to_log(traceback.format_exc())
                return

            self.call_in_ui(self._open_radii_optimizer, image_data, target_position,
                            fwhm, best_aperture, best_inner, best_outer)

        self.run_task(prepare)

    def _open_radii_optimizer(self, image_data, target_position, fwhm, best_aperture, best_inner, best_outer):
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

        tk.Label(controls_frame, text=f"Estimated FWHM: {fwhm:.2f} px", font=("Helvetica", 12, "bold")).pack(pady=10)

        snr_label = tk.Label(controls_frame, text="SNR: Calculating...", font=("Helvetica", 12, "bold"))
        snr_label.pack(pady=10)

        area_factor_label = tk.Label(controls_frame, text="Annulus Area: --x Aperture Area", font=("Helvetica", 11))
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
                snr, _, _, _ = calculate_target_snr(image_data, target_position, ap, (inn, out))
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
    # O-C plotting
    # ------------------------------------------------------------------
    def show_oc_plot(self):
        """Display the O-C Plotting panel."""
        self.clear_right_frame()
        self._panel_title("O-C Plotting")

        tk.Label(self.right_frame, text="Data Source:", font=self.label_font, bg=BG_PANEL).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.oc_mode_var = tk.StringVar(value="BSUO")
        mode_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        mode_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        for label, val in [("BSUO/SARA", "BSUO"), ("TESS", "TESS"), ("All Data", "ALL")]:
            tk.Radiobutton(
                mode_frame, text=label, variable=self.oc_mode_var, value=val,
                font=self.label_font, bg=BG_PANEL,
                command=self._update_oc_fields
            ).pack(side="left", padx=5)

        self.oc_has_epoch_var = tk.BooleanVar(value=True)
        self.oc_has_epoch_check = tk.Checkbutton(
            self.right_frame, text="I already have an Epoch value",
            variable=self.oc_has_epoch_var, font=self.label_font, bg=BG_PANEL
        )
        self.oc_has_epoch_check.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Shared fields
        self.oc_epoch_entry = self.create_input_field(
            self.right_frame, "Epoch (T0):", "e.g. 2457143.761819", row=3,
            validation_func=_is_float, error_message="Please enter a valid float."
        )
        self.oc_epoch_err_entry = self.create_input_field(
            self.right_frame, "Epoch Error:", "e.g. 0.0002803", row=4,
            validation_func=_is_float, error_message="Please enter a valid float."
        )
        self.oc_period_entry = self.create_input_field(
            self.right_frame, "Period (days):", "e.g. 0.31297", row=5,
            validation_func=_is_float, error_message="Please enter a valid period."
        )
        self.oc_output_entry = self.create_input_field(
            self.right_frame, "Output Folder:", _example_path("output"), row=6,
            error_message="Please enter an output folder path.", browse_type="folder"
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
            error_message="Please enter file paths separated by commas."
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
        self._panel_title("Gaia Query")

        ra = self.create_input_field(self.right_frame, "Right Ascension (RA):",
                                     "HH:MM:SS.SSSS", row=1,
                                     error_message="Please enter a RA.")

        dec = self.create_input_field(self.right_frame, "Declination (DEC):",
                                      "DD:MM:SS.SSSS or -DD:MM:SS.SSSS", row=2,
                                      error_message="Please enter a DEC.")

        output_folder = self.create_input_field(self.right_frame, "Output Folder:",
                                                _example_path("gaia"), row=3,
                                                error_message="Please enter a folder.",
                                                browse_type="folder")

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
        self._panel_title("O'Connell Effect Calculation")

        filter_count_var = tk.IntVar(value=3)  # Default selection is 3 filters
        file_path_vars = [tk.StringVar() for _ in range(3)]
        file_entry_widgets = []

        def update_file_path_fields():
            for widgets in file_entry_widgets:
                for widget in widgets:
                    widget.destroy()
            file_entry_widgets.clear()

            for i in range(filter_count_var.get()):
                label = tk.Label(self.right_frame, text=f"File Path {i + 1}:", font=self.label_font, bg=BG_PANEL)
                label.grid(row=3 + i, column=0, padx=10, pady=2, sticky="e")

                entry_frame = tk.Frame(self.right_frame, bg=BG_PANEL, highlightthickness=1,
                                       highlightbackground="#cccccc")
                entry_frame.grid(row=3 + i, column=1, padx=10, pady=2, sticky="w")

                entry = tk.Entry(entry_frame, textvariable=file_path_vars[i], font=self.label_font, width=30,
                                 borderwidth=0)
                entry.grid(row=0, column=0, sticky="w", padx=(5, 0), ipadx=10, ipady=3)

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

        tk.Label(self.right_frame, text="Select Number of Filters:", font=self.label_font, bg=BG_PANEL).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )

        radio_frame = tk.Frame(self.right_frame, bg=BG_PANEL)
        radio_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        for i in range(1, 4):
            tk.Radiobutton(
                radio_frame,
                text=f"{i} Filter{'s' if i > 1 else ''}",
                variable=filter_count_var,
                value=i,
                command=update_file_path_fields,
                font=self.label_font,
                bg=BG_PANEL,
                anchor="w"
            ).pack(side="left", padx=5)

        update_file_path_fields()

        hjd_var = self.create_input_field(
            parent=self.right_frame, label_text="HJD:", placeholder_text="e.g. 2458403.58763", row=6,
            validation_func=_is_float, error_message="Please enter a valid HJD."
        )

        period_var = self.create_input_field(
            parent=self.right_frame, label_text="Period:", placeholder_text="e.g. 0.3175", row=7,
            validation_func=_is_float, error_message="Please enter a valid Period."
        )

        obj_name_var = self.create_input_field(
            parent=self.right_frame, label_text="System Name:", placeholder_text="e.g. NSVS_896797", row=8,
            error_message="Please enter a System Name."
        )

        output_var = self.create_input_field(
            parent=self.right_frame, label_text="Output Folder:", placeholder_text=_example_path("oconnell"),
            row=9, error_message="Please enter a file pathway.", browse_type="folder"
        )

        self.create_run_button(self.right_frame, self.run_oconnell_effect, row=10,
                               text="Run O'Connell Effect",
                               filter_count_var=filter_count_var, file_path_vars=file_path_vars,
                               hjd=hjd_var, period=period_var, obj_name=obj_name_var, output_file=output_var)

        self.create_scrollbar_and_log(11)

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
                    self.notify("info", "Success", "O'Connell Effect calculation completed successfully!")
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
        self._panel_title("Color Light Curve")

        bfile = self.create_input_field(self.right_frame, "B-band File:",
                                        _example_path("b_band.txt"), row=1,
                                        error_message="Please select a valid B-band file.",
                                        browse_type="file")

        vfile = self.create_input_field(self.right_frame, "V-band File:",
                                        _example_path("v_band.txt"), row=2,
                                        error_message="Please select a valid V-band file.",
                                        browse_type="file")

        period = self.create_input_field(self.right_frame, "Period (days):",
                                         "e.g. 1.234", row=3,
                                         validation_func=_is_float,
                                         error_message="Enter a numeric period.")

        hjd = self.create_input_field(self.right_frame, "HJD (Epoch):",
                                      "e.g. 2459000.123", row=4,
                                      validation_func=_is_float,
                                      error_message="Enter a numeric HJD.")

        outname = self.create_input_field(self.right_frame, "Output Image Name:",
                                          "color_curve.png", row=5,
                                          validation_func=lambda x: x.strip().lower().endswith(".png"),
                                          error_message="Output must be a PNG file.",
                                          default="color_curve.png")

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
                    self.notify("info", "Cancelled", "Color light curve task was cancelled.")
                else:
                    self.notify("info", "Success", "Color light curve generated successfully!")
            except Exception as e:
                self.write_to_log(f"Error: {type(e).__name__}: {e}")
                self.write_to_log(traceback.format_exc())

        self.run_task(task)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def dummy_action(self):
        """Dummy action for unimplemented features"""
        messagebox.showinfo("Action", "This feature is not implemented yet.", parent=self)

    def quit_program(self):
        """Quit the program after confirming, stopping any running task first."""
        if self._task_running:
            if not messagebox.askyesno("Exit", "A task is still running. Cancel it and exit?", parent=self):
                return
            self.cancel_event.set()
        elif not messagebox.askyesno("Exit", "Are you sure you want to exit?", parent=self):
            return
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
