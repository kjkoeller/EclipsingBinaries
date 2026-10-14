"""
Building blocks for the EclipsingBinaries GUI that don't depend on any one
panel: tooltips, remembered settings, a scrolling panel, the window icon,
opening files in the system file browser, and sorting log lines by kind.

Author: Kyle Koeller
Created: 10/10/2026
"""

import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

TIP_BG = "#fffbe6"
TIP_FG = "#1b2430"
TIP_BORDER = "#d9cf9e"


class ToolTip:
    """A short help note that appears after the pointer rests on a widget."""

    DELAY_MS = 500

    def __init__(self, widget, text, font=None):
        self.widget = widget
        self.text = text
        self.font = font
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._tip is not None or not self.text or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        if sys.platform == "darwin":
            # Without this a borderless window on macOS takes focus from the app
            try:
                tip.tk.call("::tk::unsupported::MacWindowStyle", "style", tip._w,
                            "help", "noActivates")
            except tk.TclError:
                pass
        tk.Label(tip, text=self.text, justify="left", wraplength=360, bg=TIP_BG, fg=TIP_FG,
                 font=self.font, padx=8, pady=5, highlightthickness=1,
                 highlightbackground=TIP_BORDER).pack()
        tip.wm_geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


class SettingsStore:
    """
    Values the GUI remembers between sessions, such as the folders last used in
    each panel. Kept in a small JSON file; if it can't be read or written the
    app carries on without it.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data = loaded
        except (OSError, ValueError):
            pass

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def section(self, name):
        value = self.data.get("panels", {}).get(name, {})
        return dict(value) if isinstance(value, dict) else {}

    def update_section(self, name, values):
        """Replace a panel's saved values, leaving out anything empty."""
        panels = self.data.setdefault("panels", {})
        panels[name] = {k: v for k, v in values.items() if v not in ("", None)}

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, self.path)
        except OSError:
            pass

    def clear(self):
        self.data = {}
        try:
            self.path.unlink()
        except OSError:
            pass


class ScrollableFrame(tk.Frame):
    """
    A frame whose contents scroll up and down when they're taller than the
    space available, and stretch to fill it when they're not. Put widgets in
    .content, not in the frame itself.
    """

    def __init__(self, master, bg, **kwargs):
        super().__init__(master, bg=bg, **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", self._fit, add="+")
        self.content.bind("<Configure>", self._fit, add="+")
        self.bind("<Enter>", self._bind_wheel, add="+")
        self.bind("<Leave>", self._unbind_wheel, add="+")
        self._last_needed = None
        self._watch_id = None
        self._watch()

    WATCH_MS = 250

    def _watch(self):
        """
        Refit when the contents want a different height. The contents are
        stretched to fill the canvas, so when they shrink (a panel with fewer
        fields, text that stops wrapping) their actual size doesn't change and
        Tk sends no Configure event; without this a scrollbar could stay up
        after it stopped being needed.
        """
        try:
            needed = self.content.winfo_reqheight()
        except tk.TclError:
            return  # being destroyed
        if needed != self._last_needed:
            self._last_needed = needed
            self._fit()
        self._watch_id = self.after(self.WATCH_MS, self._watch)

    def destroy(self):
        if self._watch_id is not None:
            try:
                self.after_cancel(self._watch_id)
            except tk.TclError:
                pass
            self._watch_id = None
        super().destroy()

    def _fit(self, _event=None):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        needed = self.content.winfo_reqheight()
        self.canvas.itemconfigure(self._window, width=width, height=max(height, needed))
        self.canvas.configure(scrollregion=(0, 0, width, max(height, needed)))
        if needed > height + 1:
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    def scroll_to_top(self):
        self.canvas.yview_moveto(0)

    def _bind_wheel(self, _event=None):
        # Tk 8.6 on Linux reports the wheel as buttons 4 and 5; Tk 9 sends
        # MouseWheel everywhere, so Linux gets both
        if sys.platform.startswith("linux"):
            self.bind_all("<Button-4>", self._on_wheel, add="+")
            self.bind_all("<Button-5>", self._on_wheel, add="+")
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _unbind_wheel(self, _event=None):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.unbind_all(sequence)

    def _on_wheel(self, event):
        # Widgets that scroll themselves (the output log, dropdown lists) keep the wheel
        if event.widget.winfo_class() in ("Text", "Listbox", "TCombobox"):
            return
        if self.content.winfo_reqheight() <= self.canvas.winfo_height():
            return
        if getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        elif sys.platform == "darwin" and abs(event.delta) < 120:
            steps = -event.delta  # Tk 8.6 on macOS: small steps, one per notch or trackpad tick
        else:
            steps = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(steps, "units")


def make_app_icon(master, size=64):
    """
    Draw the window icon: an eclipsing pair, a smaller cool star crossing in
    front of a warm one, on a night-sky disc. Drawn in code so the package
    doesn't need to ship an image file.
    """
    sky = (14, 31, 58)          # night navy
    warm = (242, 193, 78)       # starlight
    cool = (198, 216, 245)
    rim = (233, 240, 250)
    samples = 4                 # supersampling per pixel for smooth edges
    image = tk.PhotoImage(master=master, width=size, height=size)

    def colour_at(px, py):
        x, y = px / size - 0.5, py / size - 0.5
        if x * x + y * y > 0.48 ** 2:
            return None
        bx, by = x - 0.1, y + 0.02          # the star in front
        if bx * bx + by * by <= 0.17 ** 2:
            return cool
        if bx * bx + by * by <= 0.19 ** 2:
            return sky                      # a thin dark gap shows it's in front
        ax, ay = x + 0.1, y - 0.02          # the star behind
        if ax * ax + ay * ay <= 0.22 ** 2:
            return warm
        if x * x + y * y > 0.455 ** 2:
            return rim
        return sky

    rows = []
    transparent = []
    for py in range(size):
        row = []
        for px in range(size):
            hits = []
            for sy in range(samples):
                for sx in range(samples):
                    c = colour_at(px + (sx + 0.5) / samples, py + (sy + 0.5) / samples)
                    if c is not None:
                        hits.append(c)
            if len(hits) * 2 < samples * samples:
                row.append("#000000")
                transparent.append((px, py))
                continue
            r = sum(c[0] for c in hits) // len(hits)
            g = sum(c[1] for c in hits) // len(hits)
            b = sum(c[2] for c in hits) // len(hits)
            row.append(f"#{r:02x}{g:02x}{b:02x}")
        rows.append("{" + " ".join(row) + "}")
    image.put(" ".join(rows))
    for px, py in transparent:
        image.transparency_set(px, py, True)
    return image


def open_path(path):
    """Open a file or folder with the system's default app (Finder, Explorer or a file manager)."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 (opening a local path the user chose)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


_ERROR = re.compile(r"\b(error|errors|failed|traceback|exception)\b", re.IGNORECASE)
_WARNING = re.compile(r"\bwarning\b|^\s*\[(removed|skipped|excluded)\]|\bcancel", re.IGNORECASE)
_SUCCESS = re.compile(r"completed successfully|^\s*finished\b|^\s*saved\b|pipeline complete",
                      re.IGNORECASE)


def classify_log_line(message):
    """Sort a log message into 'error', 'warning' or 'success' for colouring, or None for plain text."""
    if _ERROR.search(message):
        return "error"
    if _WARNING.search(message):
        return "warning"
    if _SUCCESS.search(message):
        return "success"
    return None
