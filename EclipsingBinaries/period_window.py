# -*- coding: utf-8 -*-
"""
The Period Search window: a periodogram above the light curve folded on the
chosen period. Drag along the periodogram (or move the slider) and the fold
follows; click a peak to jump to its top. The period can be saved or sent to
O-C Plotting, O'Connell Effect or Color Light Curve.

The calculations are in period.py; this file is only the window.

Author: Kyle Koeller
Created: 10/10/2026
"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import period as pr

HELP_TEXT = ("Drag along the periodogram, or move the slider under it, and the fold follows the period. "
             "Click a peak without dragging to jump to its top. Over the plots, the left and right arrow keys "
             "nudge the period, and zooming the periodogram makes the slider finer. "
             "An eclipsing binary should show a primary and a secondary eclipse in one cycle: if the fold "
             "shows a single dip and the star has two alike eclipses, use \u00d72.")
COLOURS = {"B": "#2f5aa8", "V": "#2d7d46", "R": "#b23a2e"}
ERROR_FG = "#b3261e"
TEXT_FG = "#1b2430"
MUTED_FG = "#5b6675"
SEND_TARGETS = (("O-C Plotting", "oc"), ("O'Connell Effect", "oconnell"), ("Color Light Curve", "color"))


def _style(widget, name, fallback):
    """Use one of the main window's button styles when it exists (it doesn't when run on its own)."""
    try:
        widget.tk.call("ttk::style", "layout", name)
        return name
    except tk.TclError:
        return fallback


class PeriodWindow(tk.Toplevel):
    """
    Interactive period search.

    :param master: parent Tk window
    :param files: light curve files, one per filter
    :param labels: filter name for each file
    :param output_folder: where <object>_period.txt is saved
    :param obj_name: object name for the file and window title
    :param min_period, max_period: the period range to search, in days
    :param method: one of period.METHODS
    :param log: function for progress messages
    :param send: function(target, period) that fills in another panel's
        period; target is "oc", "oconnell" or "color". Leave out to hide the
        Send buttons.
    """

    def __init__(self, master, files, labels, output_folder, obj_name, min_period=0.1, max_period=1.0,
                 method=pr.DEFAULT_METHOD, log=print, send=None):
        super().__init__(master)
        self.files = [str(f) for f in files]
        self.curve = pr.load_light_curves(files, labels)
        self.output_folder = Path(output_folder)
        self.obj_name = obj_name
        self.log = log
        self.send = send
        self.result = None
        self.period = None
        self.half_width = None
        self._job = None
        self._drag = None           # set while the mouse is pressed on the periodogram
        self._syncing = False       # True while the program, not the user, moves the slider
        self._fold_lines = {}       # filter: the Line2D showing its folded points
        self._marker_line = self._half_line = self._marker_dot = None
        self._fold_text = None
        self._background = None     # snapshot of everything that doesn't move while dragging

        self.title(f"Period Search: {obj_name}" if obj_name else "Period Search")
        self.geometry("1150x800")
        self.minsize(900, 640)
        self._build(min_period, max_period, method)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.start_search()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self, min_period, max_period, method):
        accent = _style(self, "Accent.TButton", "TButton")
        secondary = _style(self, "Secondary.TButton", "TButton")
        self.heading_font = tkfont.nametofont("TkDefaultFont").copy()
        self.heading_font.configure(weight="bold", size=abs(self.heading_font.cget("size")) + 1)

        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Method:").pack(side="left")
        self.method_var = tk.StringVar(master=self, value=method)
        ttk.Combobox(top, textvariable=self.method_var, values=pr.METHODS, state="readonly",
                     width=15).pack(side="left", padx=(4, 14))
        ttk.Label(top, text="Shortest period (d):").pack(side="left")
        self.min_var = tk.StringVar(master=self, value=f"{min_period:g}")
        ttk.Entry(top, textvariable=self.min_var, width=8).pack(side="left", padx=(4, 14))
        ttk.Label(top, text="Longest period (d):").pack(side="left")
        self.max_var = tk.StringVar(master=self, value=f"{max_period:g}")
        ttk.Entry(top, textvariable=self.max_var, width=8).pack(side="left", padx=(4, 14))
        self.search_button = ttk.Button(top, text="Search", command=self.start_search, style=accent)
        self.search_button.pack(side="left")

        self.band_vars = {}
        if len(self.curve.labels) > 1:
            ttk.Label(top, text="Filters:").pack(side="left", padx=(24, 4))
            for label in self.curve.labels:
                var = tk.BooleanVar(master=self, value=True)
                ttk.Checkbutton(top, text=label, variable=var, command=self._bands_changed).pack(side="left")
                self.band_vars[label] = var

        self.status_var = tk.StringVar(master=self)
        self.status_label = ttk.Label(self, textvariable=self.status_var, padding=(12, 0, 12, 4),
                                      wraplength=1000, justify="left")
        self.status_label.pack(fill="x")

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(body)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        self.figure = Figure(figsize=(7.5, 6))
        grid = self.figure.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.35)
        self.ax_power = self.figure.add_subplot(grid[0])
        self.ax_fold = self.figure.add_subplot(grid[1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        # A slider across the periodogram's visible range, the same as dragging the marker
        slider_row = ttk.Frame(plot_frame, padding=(0, 4, 0, 2))
        slider_row.pack(side="bottom", fill="x")
        ttk.Label(slider_row, text="Period:").pack(side="left")
        self.slider_var = tk.DoubleVar(master=self, value=min_period)
        self.slider = ttk.Scale(slider_row, from_=min_period, to=max_period, orient="horizontal",
                                variable=self.slider_var, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("draw_event", self._on_draw)

        side = ttk.Frame(body, padding=(14, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")

        ttk.Label(side, text="Period (days)", font=self.heading_font).pack(anchor="w")
        self.period_var = tk.StringVar(master=self)
        period_entry = ttk.Entry(side, textvariable=self.period_var, width=18)
        period_entry.pack(anchor="w", pady=(4, 4))
        period_entry.bind("<Return>", lambda _e: self._typed_period())
        period_entry.bind("<FocusOut>", lambda _e: self._typed_period())
        row = ttk.Frame(side)
        row.pack(anchor="w")
        ttk.Button(row, text="\u00d72", width=4, command=lambda: self.set_period(self.period * 2),
                   style=secondary).pack(side="left")
        ttk.Button(row, text="÷2", width=4, command=lambda: self.set_period(self.period / 2),
                   style=secondary).pack(side="left", padx=4)
        ttk.Button(row, text="Refine Here", command=self.refine_here, style=secondary).pack(side="left")
        self.value_var = tk.StringVar(master=self)
        ttk.Label(side, textvariable=self.value_var).pack(anchor="w", pady=(6, 0))
        self.width_var = tk.StringVar(master=self)
        ttk.Label(side, textvariable=self.width_var, foreground=MUTED_FG).pack(anchor="w", pady=(2, 12))

        ttk.Separator(side).pack(fill="x", pady=(0, 12))
        self.save_button = ttk.Button(side, text="Save Period", command=self.save, style=accent)
        self.save_button.pack(anchor="w", fill="x")
        self.send_buttons = []
        if self.send is not None:
            for text, target in SEND_TARGETS:
                button = ttk.Button(side, text=f"Send to {text}", style=secondary,
                                    command=lambda t=target: self.send_to(t))
                button.pack(anchor="w", fill="x", pady=(6, 0))
                self.send_buttons.append(button)

        ttk.Label(side, text=HELP_TEXT, wraplength=250, foreground=MUTED_FG,
                  justify="left").pack(anchor="w", pady=(16, 0))
        ttk.Button(side, text="Close", command=self.destroy, style=secondary).pack(side="bottom", anchor="e")
        self._set_actions_enabled(False)

    # ------------------------------------------------------------------
    # Searching (on a worker thread, so the window keeps responding)
    # ------------------------------------------------------------------
    def selected_curve(self):
        if not self.band_vars:
            return self.curve
        chosen = [label for label, var in self.band_vars.items() if var.get()]
        return self.curve.select(chosen) if chosen else None

    def start_search(self):
        if self._job is not None:
            return
        curve = self.selected_curve()
        if curve is None:
            self._status("Tick at least one filter.", error=True)
            return
        try:
            lo, hi = float(self.min_var.get()), float(self.max_var.get())
        except ValueError:
            self._status("The period range needs two numbers, in days.", error=True)
            return
        method = self.method_var.get()
        outcome = {}

        def work():
            try:
                outcome["result"] = pr.search(curve, lo, hi, method)
            except pr.PeriodError as e:
                outcome["error"] = str(e)
            except Exception as e:  # shown in the window rather than lost on a thread
                outcome["error"] = f"{type(e).__name__}: {e}"

        self._job = threading.Thread(target=work, daemon=True)
        self._job.start()
        self.search_button.state(["disabled"])
        self._status(f"Searching {lo:g} to {hi:g} days with {method}...")
        self._wait_for(outcome)

    def _wait_for(self, outcome):
        if not self.winfo_exists():
            return
        if self._job.is_alive():
            self.after(100, lambda: self._wait_for(outcome))
            return
        self._job = None
        self.search_button.state(["!disabled"])
        if "error" in outcome:
            self._status(outcome["error"], error=True)
            return
        self.result = outcome["result"]
        self._report_search()
        self.half_width = self.result.half_width
        self._draw_periodogram()
        self.set_period(self.result.best_period)
        self._set_actions_enabled(True)

    def _report_search(self):
        r = self.result
        text = (f"Best period {r.best_period:.6f} d from {r.method} "
                f"({len(r.periods):,} trial periods).")
        if r.doubled_from is not None:
            text += (f" The highest peak was {r.doubled_from:.6f} d, but folded on twice that the two "
                     "eclipses differ, so the orbital period is double the peak.")
        if r.grid.coarse:
            text += (" The trial periods had to be spaced more widely than usual to keep the search quick; "
                     "narrow the period range for a finer search.")
        self._status(text)
        self.log(f"Period search ({r.method}): best period {r.best_period:.6f} d")

    def _status(self, text, error=False):
        self.status_var.set(text)
        self.status_label.configure(foreground=ERROR_FG if error else TEXT_FG)

    def _bands_changed(self):
        if self.result is not None:
            self._status("Filters changed. Search again to update the periodogram.")
        self._draw_fold()
        self._update_period_artists()

    def _set_actions_enabled(self, enabled):
        for button in [self.save_button, *self.send_buttons]:
            button.state(["!disabled"] if enabled else ["disabled"])

    # ------------------------------------------------------------------
    # Period
    # ------------------------------------------------------------------
    @property
    def fold_origin(self):
        """
        Where phase 0 falls: the first observation, whatever filters are
        ticked, so changing the period is the only thing that moves the fold.
        """
        return float(self.curve.time[0])

    def set_period(self, value):
        if value is None or not np.isfinite(value) or value <= 0:
            return
        self.period = float(value)
        self.period_var.set(f"{self.period:.7f}")
        self._show_width()
        self._update_period_artists()

    def _show_width(self):
        if self.half_width and np.isfinite(self.half_width):
            self.width_var.set(f"Peak half width: \u00b1{self.half_width:.6f} d")
        else:
            self.width_var.set("")

    def _typed_period(self):
        try:
            value = float(self.period_var.get())
        except ValueError:
            if self.period is not None:
                self.period_var.set(f"{self.period:.7f}")
            return
        if self.period is None or abs(value - self.period) > 1e-12:
            self.set_period(value)

    def refine_here(self):
        """Find the top of the peak nearest the current period."""
        if self.period is None:
            return
        curve = self.selected_curve() or self.curve
        best, _power, width = pr.refine(curve, self.period, self.method_var.get())
        self.half_width = width
        self.set_period(best)

    def save(self):
        if self.period is None:
            return
        path = self.output_folder / pr.period_file_name(self.obj_name)
        try:
            pr.save_period(path, self.obj_name, self.period, self.result, self.files)
        except OSError as e:
            self._status(f"Couldn't save {path}: {e}", error=True)
            return
        self.log(f"Saved period {self.period:.7f} d to {path}")
        self._status(f"Saved to {path}")

    def send_to(self, target):
        if self.period is None or self.send is None:
            return
        self.send(target, self.period)

    # ------------------------------------------------------------------
    # Drawing
    #
    # The periodogram and the fold are drawn in full only after a search, a
    # change of filters, a zoom or a resize. The things that move with the
    # period (the markers, the folded points and their caption) are
    # "animated": after each full draw a snapshot of everything else is kept,
    # and moving the period pastes the snapshot back and draws just those on
    # top (blitting), which is quick enough to follow the mouse.
    # ------------------------------------------------------------------
    def redraw(self):
        self._draw_periodogram()
        self._draw_fold()
        self._update_period_artists()

    def _draw_periodogram(self):
        ax = self.ax_power
        ax.clear()
        self._marker_line = self._half_line = self._marker_dot = None
        r = self.result
        if r is not None:
            ax.plot(r.periods, r.power, color="0.25", lw=0.7)
            ax.set_xlim(r.periods[0], r.periods[-1])
            ax.set_xlabel("Trial period (days)", fontsize=9)
            ax.set_ylabel(pr.statistic_name(r.method), fontsize=9)
            ax.grid(alpha=0.3)
            # Semi-transparent so the peak under it still shows at full width
            self._marker_line = ax.axvline(r.best_period, color="#c0392b", lw=1, alpha=0.45, animated=True)
            self._half_line = ax.axvline(r.best_period / 2, color="#c0392b", lw=0.8, ls=":", animated=True)
            (self._marker_dot,) = ax.plot([r.best_period], [r.best_power], "o", color="#c0392b",
                                          ms=7, zorder=5, animated=True)
            # Clearing the axis drops its callbacks, so hook the slider's range to the zoom again
            ax.callbacks.connect("xlim_changed", lambda _ax: self._sync_slider_range())
            self._sync_slider_range()
        ax.tick_params(labelsize=8)
        self._background = None

    def _draw_fold(self):
        ax = self.ax_fold
        ax.clear()
        self._fold_lines = {}
        curve = self.selected_curve()
        if curve is not None:
            for label in curve.labels:
                (self._fold_lines[label],) = ax.plot([], [], "o", ms=2.5, color=COLOURS.get(label, "0.2"),
                                                     label=label if len(curve.labels) > 1 else None,
                                                     animated=True)
            ax.set_ylim(np.nanmax(curve.mag) + 0.05, np.nanmin(curve.mag) - 0.05)  # brighter is up
            ax.set_xlim(-0.25, 1.25)
            ax.set_xlabel("Phase", fontsize=9)
            ax.set_ylabel("Magnitude minus median", fontsize=9)
            self._fold_text = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=8,
                                      animated=True)
            ax.grid(alpha=0.3)
            if len(curve.labels) > 1:
                ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.tick_params(labelsize=8)
        self._background = None

    def _animated_artists(self):
        artists = [self._marker_line, self._half_line, self._marker_dot, self._fold_text,
                   *self._fold_lines.values()]
        return [a for a in artists if a is not None]

    def _on_draw(self, _event):
        """After a full draw: keep a snapshot without the moving parts, then add them."""
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)
        for artist in self._animated_artists():
            artist.axes.draw_artist(artist)

    def _refresh(self):
        """Show the moving parts in their new places, by blitting when a snapshot is ready."""
        if self._background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._background)
        for artist in self._animated_artists():
            artist.axes.draw_artist(artist)
        self.canvas.blit(self.figure.bbox)

    def _update_period_artists(self):
        """Move the period markers and refold the light curve on the current period."""
        if self.period is None:
            self._refresh()
            return
        curve = self.selected_curve()
        if curve is not None and set(self._fold_lines) != set(curve.labels):
            self._draw_fold()
        value = self.value_at(self.period)
        if self._marker_line is not None:
            self._marker_line.set_xdata([self.period, self.period])
            self._half_line.set_xdata([self.period / 2, self.period / 2])
            self._half_line.set_visible(self.result.periods[0] <= self.period / 2)
            self._marker_dot.set_data([self.period], [value])
        name = pr.statistic_name(self.result.method if self.result else self.method_var.get())
        self.value_var.set(f"{name} here: {value:,.1f}" if np.isfinite(value) else "")
        if curve is not None:
            phase = pr.fold(curve.time, self.period, self.fold_origin)
            for label, line in self._fold_lines.items():
                keep = curve.label == label
                ph, mag = phase[keep], curve.mag[keep]
                # Show a little either side of one cycle so eclipses at phase 0 aren't cut in two
                line.set_data(np.concatenate([ph, ph[ph > 0.75] - 1, ph[ph < 0.25] + 1]),
                              np.concatenate([mag, mag[ph > 0.75], mag[ph < 0.25]]))
            if self._fold_text is not None:
                self._fold_text.set_text(f"Period {self.period:.7f} d")
        self._syncing = True
        try:
            self.slider_var.set(self.period)
        finally:
            self._syncing = False
        self._refresh()

    def value_at(self, period):
        """The periodogram's statistic at one period, for the marker and the side panel."""
        curve = self.selected_curve()
        if curve is None:
            return float("nan")
        method = self.result.method if self.result else self.method_var.get()
        try:
            return float(pr.score(curve, [period], method)[0])
        except pr.PeriodError:
            return float("nan")

    def _sync_slider_range(self):
        """Make the slider span what the periodogram shows, so zooming in makes it finer."""
        lo, hi = self.ax_power.get_xlim()
        self.slider.configure(from_=lo, to=hi)

    # ------------------------------------------------------------------
    # Dragging, the slider, clicks and keys
    # ------------------------------------------------------------------
    def drag_period(self, value):
        """Set the period while dragging or sliding, refolding as it goes."""
        if self.result is None or value is None or not np.isfinite(value):
            return
        lo, hi = self.ax_power.get_xlim()
        self.period = float(min(max(value, lo, 1e-6), hi))
        self.period_var.set(f"{self.period:.7f}")
        self.half_width = None   # away from a peak's top the half width doesn't apply
        self._show_width()
        self._update_period_artists()

    def _on_slider(self, value):
        if not self._syncing:
            self.drag_period(float(value))

    def _on_press(self, event):
        if self.result is None or event.inaxes is not self.ax_power or event.xdata is None:
            return
        if self.toolbar.mode:  # zooming or panning with the toolbar
            return
        self._drag = {"x": event.x, "moved": False}

    def _on_motion(self, event):
        if self._drag is None or event.inaxes is not self.ax_power or event.xdata is None:
            return
        if not self._drag["moved"] and abs(event.x - self._drag["x"]) < 4:
            return  # a few pixels of wobble is still a click
        self._drag["moved"] = True
        self.drag_period(event.xdata)

    def _on_release(self, event):
        drag, self._drag = self._drag, None
        if drag is None or drag["moved"] or event.inaxes is not self.ax_power or event.xdata is None:
            return
        # A click without dragging jumps to the top of the nearest peak
        x0, x1 = self.ax_power.get_xlim()
        window = 0.01 * (x1 - x0) / (self.result.periods[-1] - self.result.periods[0])
        picked = pr.nearest_peak(self.result, event.xdata, window=window)
        curve = self.selected_curve() or self.curve
        best, _power, width = pr.refine(curve, picked, self.result.method)
        self.half_width = width
        self.set_period(best)

    def _on_key(self, event):
        if self.period is None or event.key not in ("left", "right"):
            return
        step = (self.half_width or self.period * 1e-4) / 5
        self.set_period(self.period + (step if event.key == "right" else -step))


def run_standalone(files, labels, output_folder, obj_name, **kwargs):
    """Open the window without the main app."""
    root = tk.Tk()
    root.withdraw()
    try:
        window = PeriodWindow(root, files, labels, output_folder, obj_name, **kwargs)
    except pr.PeriodError as e:
        print(f"Error: {e}")
        root.destroy()
        return 1
    window.protocol("WM_DELETE_WINDOW", root.quit)
    window.bind("<Destroy>", lambda e: root.quit() if e.widget is window else None)
    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass
    return 0
