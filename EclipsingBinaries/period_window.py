# -*- coding: utf-8 -*-
"""
The Period Search window: a periodogram above the light curve folded on the
chosen period. Click a peak to fold on it, adjust the period and epoch, then
save them or send them to O-C Plotting, O'Connell Effect or Color Light
Curve.

The calculations are in period.py; this file is only the window.

Author: Kyle Koeller
Created: 10/10/2026
"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import period as pr

HELP_TEXT = ("Click a peak to fold on it. Over the plots, the left and right arrow keys nudge the period. "
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
    :param send: function(target, period, epoch) that fills in another
        panel; target is "oc", "oconnell" or "color". Leave out to hide the
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
        self.epoch = None
        self.half_width = None
        self._job = None

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
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("key_press_event", self._on_key)

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
        self.width_var = tk.StringVar(master=self)
        ttk.Label(side, textvariable=self.width_var, foreground=MUTED_FG).pack(anchor="w", pady=(4, 12))

        ttk.Label(side, text="Epoch (primary minimum)", font=self.heading_font).pack(anchor="w")
        self.epoch_var = tk.StringVar(master=self)
        epoch_entry = ttk.Entry(side, textvariable=self.epoch_var, width=18)
        epoch_entry.pack(anchor="w", pady=(4, 4))
        epoch_entry.bind("<Return>", lambda _e: self._typed_epoch())
        epoch_entry.bind("<FocusOut>", lambda _e: self._typed_epoch())
        row = ttk.Frame(side)
        row.pack(anchor="w")
        ttk.Button(row, text="Faintest Point", command=self.epoch_at_faintest, style=secondary).pack(side="left")
        ttk.Button(row, text="Swap Eclipses", command=self.swap_eclipses, style=secondary).pack(side="left", padx=4)
        ttk.Button(side, text="From Minimum File...", command=self.epoch_from_file,
                   style=secondary).pack(anchor="w", pady=(4, 12))

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
        self.set_period(self.result.best_period, keep_epoch=False)
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
        self.redraw()

    def _set_actions_enabled(self, enabled):
        for button in [self.save_button, *self.send_buttons]:
            button.state(["!disabled"] if enabled else ["disabled"])

    # ------------------------------------------------------------------
    # Period and epoch
    # ------------------------------------------------------------------
    def set_period(self, value, keep_epoch=True):
        if value is None or not np.isfinite(value) or value <= 0:
            return
        self.period = float(value)
        self.period_var.set(f"{self.period:.7f}")
        if not keep_epoch or self.epoch is None:
            curve = self.selected_curve() or self.curve
            self.epoch = pr.faintest_epoch(curve, self.period)
            self.epoch_var.set(f"{self.epoch:.6f}")
        self._show_width()
        self.redraw()

    def _show_width(self):
        if self.half_width and np.isfinite(self.half_width):
            self.width_var.set(f"Peak half width: ±{self.half_width:.6f} d")
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

    def _typed_epoch(self):
        try:
            self.epoch = float(self.epoch_var.get())
        except ValueError:
            if self.epoch is not None:
                self.epoch_var.set(f"{self.epoch:.6f}")
            return
        self.redraw()

    def refine_here(self):
        """Find the top of the peak nearest the current period."""
        if self.period is None:
            return
        curve = self.selected_curve() or self.curve
        best, _power, width = pr.refine(curve, self.period, self.method_var.get())
        self.half_width = width
        self.set_period(best)

    def epoch_at_faintest(self):
        if self.period is None:
            return
        self.epoch = pr.faintest_epoch(self.selected_curve() or self.curve, self.period)
        self.epoch_var.set(f"{self.epoch:.6f}")
        self.redraw()

    def swap_eclipses(self):
        """Move the epoch half a cycle, for when the secondary eclipse was taken as the primary."""
        if self.epoch is None or self.period is None:
            return
        self.epoch += self.period / 2
        self.epoch_var.set(f"{self.epoch:.6f}")
        self.redraw()

    def epoch_from_file(self, path=None):
        """Use the first time of minimum from a Find Minimum file as the epoch."""
        if path is None:
            path = filedialog.askopenfilename(parent=self, title="Choose a Find Minimum file",
                                              initialdir=str(self.output_folder),
                                              filetypes=[("Minimum files", "*_min.txt"), ("All files", "*")])
            if not path:
                return
        try:
            epoch, count = pr.epoch_from_minimum_file(path)
        except pr.PeriodError as e:
            self._status(str(e), error=True)
            return
        self.epoch = epoch
        self.epoch_var.set(f"{epoch:.6f}")
        self._status(f"Epoch {epoch:.6f} is the first of {count} minima in {Path(path).name}.")
        self.redraw()

    def save(self):
        if self.period is None:
            return
        path = self.output_folder / pr.period_file_name(self.obj_name)
        try:
            pr.save_ephemeris(path, self.obj_name, self.period, self.epoch, self.result, self.files)
        except OSError as e:
            self._status(f"Couldn't save {path}: {e}", error=True)
            return
        self.log(f"Saved period {self.period:.7f} d and epoch {self.epoch:.6f} to {path}")
        self._status(f"Saved to {path}")

    def send_to(self, target):
        if self.period is None or self.send is None:
            return
        self.send(target, self.period, self.epoch)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def redraw(self):
        self.ax_power.clear()
        self.ax_fold.clear()
        r = self.result
        if r is not None:
            self.ax_power.plot(r.periods, r.power, color="0.25", lw=0.7)
            self.ax_power.set_xlim(r.periods[0], r.periods[-1])
            self.ax_power.set_xlabel("Trial period (days)", fontsize=9)
            self.ax_power.set_ylabel(pr.statistic_name(r.method), fontsize=9)
            if self.period is not None:
                self.ax_power.axvline(self.period, color="#c0392b", lw=1)
                if r.periods[0] <= self.period / 2:
                    self.ax_power.axvline(self.period / 2, color="#c0392b", lw=0.8, ls=":")
            self.ax_power.grid(alpha=0.3)
        curve = self.selected_curve()
        if curve is not None and self.period is not None and self.epoch is not None:
            phase = pr.fold(curve.time, self.period, self.epoch)
            for label in curve.labels:
                keep = curve.label == label
                ph, mag = phase[keep], curve.mag[keep]
                # Show a little either side of one cycle so eclipses at phase 0 aren't cut in two
                ph = np.concatenate([ph, ph[ph > 0.75] - 1, ph[ph < 0.25] + 1])
                mag = np.concatenate([mag, mag[phase[keep] > 0.75], mag[phase[keep] < 0.25]])
                self.ax_fold.plot(ph, mag, "o", ms=2.5, color=COLOURS.get(label, "0.2"),
                                  label=label if len(curve.labels) > 1 else None)
            self.ax_fold.invert_yaxis()
            self.ax_fold.set_xlim(-0.25, 1.25)
            self.ax_fold.set_xlabel(f"Phase (period {self.period:.7f} d, epoch {self.epoch:.5f})", fontsize=9)
            self.ax_fold.set_ylabel("Magnitude minus median", fontsize=9)
            self.ax_fold.grid(alpha=0.3)
            if len(curve.labels) > 1:
                self.ax_fold.legend(fontsize=7, loc="lower right", frameon=False)
        for ax in (self.ax_power, self.ax_fold):
            ax.tick_params(labelsize=8)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Mouse and keys
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if self.result is None or event.inaxes is not self.ax_power or event.xdata is None:
            return
        if self.toolbar.mode:  # zooming or panning with the toolbar
            return
        x0, x1 = self.ax_power.get_xlim()
        window = 0.01 * (x1 - x0) / (self.result.periods[-1] - self.result.periods[0])
        picked = pr.nearest_peak(self.result, event.xdata, window=window)
        curve = self.selected_curve() or self.curve
        best, _power, width = pr.refine(curve, picked, self.result.method)
        self.half_width = width
        self.set_period(best, keep_epoch=False)

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
