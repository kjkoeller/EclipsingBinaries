# -*- coding: utf-8 -*-
"""
The Find Minimum window: step through the nights of a light curve, set the
boundaries around an eclipse with the mouse, and save each time of minimum
to a file O-C Plotting can read.

The calculations are in find_min.py; this file is only the window.

Author: Kyle Koeller
Created: 10/10/2026
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .find_min import (DEFAULT_NPAIRS, DEFAULT_ORDER, DEFAULT_RESOLUTION, MinimumError, MinimumFile,
                       draw_night, guess_boundaries, load_nights, measure_minimum, minimum_file_name)

HELP_TEXT = ("Click to set the left boundary and right-click (or Shift-click) to set the right one. "
             "Keep both on the sides of the eclipse rather than out on the flat parts. "
             "Keys over the plot: a and d set the boundaries at the pointer, n and p change night, "
             "Enter saves the minimum.")


def _style(widget, name, fallback):
    """Use one of the main window's button styles when it exists (it doesn't when run on its own)."""
    try:
        widget.tk.call("ttk::style", "layout", name)
        return name
    except tk.TclError:
        return fallback


class MinimumFinderWindow(tk.Toplevel):
    """
    Interactive window for measuring times of minimum.

    :param master: parent Tk window
    :param files: light curve files, one per filter
    :param labels: filter name for each file (B, V, R...)
    :param output_folder: where the minimum files are saved
    :param obj_name: used in the minimum file names
    :param log: function for progress messages, such as the main window's log
    """

    def __init__(self, master, files, labels, output_folder, obj_name, log=print,
                 order=DEFAULT_ORDER, resolution=DEFAULT_RESOLUTION, npairs=DEFAULT_NPAIRS):
        super().__init__(master)
        self.log = log
        self.output_folder = Path(output_folder)
        self.obj_name = obj_name
        self.nights = load_nights(files, labels)
        if not self.nights:
            self.destroy()
            raise MinimumError("The light curve files have no observations in them.")
        self.labels = list(dict.fromkeys(b for night in self.nights for b in night.bands))
        self.order, self.resolution, self.npairs = order, resolution, npairs
        self.current = 0
        self.boundaries = {}       # night index: (lb, rb), so each night keeps its own
        self.minimum = None
        self._files = {}           # filter combination: MinimumFile

        self.title(f"Find Minimum: {obj_name}" if obj_name else "Find Minimum")
        self.geometry("1100x760")
        self.minsize(860, 600)
        self._build()
        self.bind("<Escape>", lambda _e: self.destroy())
        self.show_night(0)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self):
        accent = _style(self, "Accent.TButton", "TButton")
        secondary = _style(self, "Secondary.TButton", "TButton")

        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Button(top, text="Previous Night", command=lambda: self.step(-1), style=secondary).pack(side="left")
        self.night_var = tk.StringVar(master=self)
        self.night_picker = ttk.Combobox(top, textvariable=self.night_var, state="readonly", width=46,
                                         values=[n.describe() for n in self.nights])
        self.night_picker.pack(side="left", padx=8)
        self.night_picker.bind("<<ComboboxSelected>>",
                               lambda _e: self.show_night(self.night_picker.current()))
        ttk.Button(top, text="Next Night", command=lambda: self.step(1), style=secondary).pack(side="left")

        ttk.Label(top, text="Filters:").pack(side="left", padx=(24, 4))
        self.band_vars = {}
        for label in self.labels:
            var = tk.BooleanVar(master=self, value=True)
            ttk.Checkbutton(top, text=label, variable=var, command=self._bands_changed).pack(side="left")
            self.band_vars[label] = var

        settings = ttk.Frame(self, padding=(12, 0, 12, 4))
        settings.pack(fill="x")
        self.order_var = self._spin(settings, "Polynomial order", self.order, 2, 9)
        self.pairs_var = self._spin(settings, "Pairs", self.npairs, 5, 60)
        self.resolution_var = self._spin(settings, "Trial times", self.resolution, 50, 1000, step=50)
        ttk.Label(settings, text=HELP_TEXT, wraplength=520, foreground="#5b6675").pack(side="left", padx=(16, 0))

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(body)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        self.figure = Figure(figsize=(7, 5.5))
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.08)
        self.ax_s = self.figure.add_subplot(grid[0])
        self.ax_lc = self.figure.add_subplot(grid[1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        side = ttk.Frame(body, padding=(14, 0, 0, 0), width=320)
        side.grid(row=0, column=1, sticky="ns")
        # A bold copy of the system font; a ("TkDefaultFont", size) tuple would look for a
        # font family by that name and fall back to something else on each platform
        self.heading_font = tkfont.nametofont("TkDefaultFont").copy()
        self.heading_font.configure(weight="bold", size=abs(self.heading_font.cget("size")) + 1)
        ttk.Label(side, text="Time of minimum", font=self.heading_font).pack(anchor="w")
        self.result_var = tk.StringVar(master=self, value="")
        self.result_label = ttk.Label(side, textvariable=self.result_var, wraplength=300, justify="left")
        self.result_label.pack(anchor="w", pady=(4, 8))
        self.save_button = ttk.Button(side, text="Save Minimum", command=self.save_minimum, style=accent)
        self.save_button.pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=12)
        self.file_var = tk.StringVar(master=self)
        ttk.Label(side, textvariable=self.file_var, wraplength=300, justify="left").pack(anchor="w")
        # ttk doesn't size table rows to the font, so they clip on scaled Windows displays
        linespace = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        ttk.Style(self).configure("Minima.Treeview", rowheight=int(linespace * 1.4))
        columns = ("hjd", "points", "error")
        self.table = ttk.Treeview(side, columns=columns, show="headings", height=12, selectmode="browse",
                                  style="Minima.Treeview")
        for col, text, width in (("hjd", "HJD", 130), ("points", "Points", 60), ("error", "Error", 100)):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor="e")
        self.table.pack(fill="both", expand=True, pady=(6, 6))
        buttons = ttk.Frame(side)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Remove Selected", command=self.remove_selected, style=secondary).pack(side="left")
        ttk.Button(buttons, text="Close", command=self.destroy, style=secondary).pack(side="right")

    def _spin(self, parent, text, value, low, high, step=1):
        ttk.Label(parent, text=f"{text}:").pack(side="left")
        var = tk.IntVar(master=self, value=value)
        spin = ttk.Spinbox(parent, from_=low, to=high, increment=step, textvariable=var, width=5,
                           command=self.remeasure)
        spin.pack(side="left", padx=(4, 14))
        spin.bind("<Return>", lambda _e: self.remeasure())
        spin.bind("<FocusOut>", lambda _e: self.remeasure())
        return var

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def selected_labels(self):
        return tuple(label for label, var in self.band_vars.items() if var.get())

    def minimum_file(self):
        """The saved minima for the filters currently ticked."""
        labels = self.selected_labels()
        if labels not in self._files:
            self._files[labels] = MinimumFile(self.output_folder / minimum_file_name(self.obj_name, labels))
        return self._files[labels]

    def _settings(self):
        try:
            return (max(2, int(self.order_var.get())), max(5, int(self.pairs_var.get())),
                    max(20, int(self.resolution_var.get())))
        except (tk.TclError, ValueError):
            return self.order, self.npairs, self.resolution

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def show_night(self, index):
        self.current = max(0, min(index, len(self.nights) - 1))
        night = self.nights[self.current]
        self.night_picker.current(self.current)
        if self.current not in self.boundaries:
            self.boundaries[self.current] = guess_boundaries(night, self.selected_labels() or None)
        self.remeasure()

    def step(self, delta):
        self.show_night(self.current + delta)

    def set_boundary(self, which, x):
        lb, rb = self.boundaries[self.current]
        if which == "left":
            lb = x
        else:
            rb = x
        self.boundaries[self.current] = tuple(sorted((lb, rb)))
        self.remeasure()

    def _bands_changed(self):
        if not self.selected_labels():
            self.result_var.set("Tick at least one filter.")
        self.remeasure()

    def remeasure(self):
        night = self.nights[self.current]
        labels = self.selected_labels()
        lb, rb = self.boundaries[self.current]
        order, npairs, resolution = self._settings()
        self.minimum = None
        message = ""
        if labels:
            try:
                self.minimum = measure_minimum(night, lb, rb, labels=labels, order=order,
                                               resolution=resolution, npairs=npairs)
            except MinimumError as e:
                message = str(e)
        else:
            message = "Tick at least one filter."
        self._show_result(message)
        self._refresh_table()
        draw_night(self.ax_s, self.ax_lc, night, lb, rb, self.minimum, labels=labels)
        self.canvas.draw_idle()

    def _show_result(self, message):
        m = self.minimum
        if m is None:
            self.result_var.set(message)
            self.result_label.configure(foreground="#b3261e")
            self.save_button.state(["disabled"])
            return
        error = f"± {m.error:.6f}" if m.error is not None else "(no error estimate: widen the boundaries)"
        self.result_var.set(f"HJD {m.hjd:.6f} {error}\n"
                            f"{m.n_points} points in {', '.join(m.labels)}, "
                            f"K = {self._settings()[1]}, R² = {m.kwee.R2:.5f}")
        self.result_label.configure(foreground="#1b2430")
        self.save_button.state(["!disabled"] if m.error is not None else ["disabled"])

    def save_minimum(self):
        if self.minimum is None:
            return
        target = self.minimum_file()
        try:
            replaced = target.add(self.minimum)
        except (MinimumError, OSError) as e:
            self.result_var.set(str(e))
            self.result_label.configure(foreground="#b3261e")
            return
        verb = "Replaced the earlier minimum with" if replaced else "Saved minimum"
        self.log(f"{verb} HJD {self.minimum.hjd:.6f} ± {self.minimum.error:.6f} in {target.path}")
        self._refresh_table()

    def remove_selected(self):
        selection = self.table.selection()
        if not selection:
            return
        index = self.table.index(selection[0])
        target = self.minimum_file()
        removed = target.entries[index][0]
        target.remove(index)
        self.log(f"Removed minimum HJD {removed:.6f} from {target.path}")
        self._refresh_table()

    def _refresh_table(self):
        self.table.delete(*self.table.get_children())
        if not self.selected_labels():
            self.file_var.set("")
            return
        target = self.minimum_file()
        self.file_var.set(f"Saved in {target.path.name} ({len(target.entries)}):")
        for t, n, err in target.entries:
            self.table.insert("", "end", values=(f"{t:.6f}", n, f"{err:.6f}"))

    # ------------------------------------------------------------------
    # Mouse and keys on the plot
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes is not self.ax_lc or event.xdata is None:
            return
        if self.toolbar.mode:  # zooming or panning with the toolbar
            return
        right = event.button == 3 or (event.button == 1 and event.key == "shift")
        self.set_boundary("right" if right else "left", event.xdata)

    def _on_key(self, event):
        if event.key in ("a", "d") and event.inaxes is self.ax_lc and event.xdata is not None:
            self.set_boundary("left" if event.key == "a" else "right", event.xdata)
        elif event.key == "n":
            self.step(1)
        elif event.key == "p":
            self.step(-1)
        elif event.key == "enter":
            self.save_minimum()

    def _on_scroll(self, event):
        """Scroll over the light curve to zoom in and out around the pointer."""
        if event.inaxes is not self.ax_lc or event.xdata is None:
            return
        # A trackpad sends many small scroll events, a wheel a few big ones, so
        # zoom by how far it scrolled rather than a fixed amount per event
        step = max(-5.0, min(5.0, float(getattr(event, "step", 1 if event.button == "up" else -1))))
        scale = 0.8 ** step
        x0, x1 = self.ax_lc.get_xlim()
        y0, y1 = self.ax_lc.get_ylim()
        self.ax_lc.set_xlim(event.xdata - (event.xdata - x0) * scale, event.xdata + (x1 - event.xdata) * scale)
        self.ax_lc.set_ylim(event.ydata - (event.ydata - y0) * scale, event.ydata + (y1 - event.ydata) * scale)
        self.canvas.draw_idle()


def run_standalone(files, labels, output_folder, obj_name):
    """Open the window without the main app, for python -m EclipsingBinaries.find_min."""
    root = tk.Tk()
    root.withdraw()
    try:
        window = MinimumFinderWindow(root, files, labels, output_folder, obj_name)
    except MinimumError as e:
        print(f"Error: {e}", file=sys.stderr)
        root.destroy()
        return 1
    # Closing the window ends the event loop; the hidden root is destroyed after it
    window.protocol("WM_DELETE_WINDOW", root.quit)
    window.bind("<Destroy>", lambda e: root.quit() if e.widget is window else None)
    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass
    return 0
