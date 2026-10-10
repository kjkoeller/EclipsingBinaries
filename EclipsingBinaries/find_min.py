# -*- coding: utf-8 -*-
"""
Times of minimum from light curves, with the Kwee-van Woerden method.

A light curve is split into nights, the flux between two boundaries around an
eclipse is used to find the time of minimum, and each minimum can be saved
to a file that O-C Plotting reads. The interactive window that goes with
this is in find_min_window.py; everything here works without a display, so
it can also be used from scripts.

Created:  07/03/2021
Original Author: Alec Neal

Last Edits Done By: Kyle Koeller
Last Edited: 10/10/2026
"""

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.interpolate
from astropy.time import Time

from .vseq_updated import calc, FT

# Starting values for the fit, the same ones the terminal version used
DEFAULT_ORDER = 5          # polynomial drawn through the eclipse (not used for the minimum itself)
DEFAULT_RESOLUTION = 200   # trial times tried across the boundaries
DEFAULT_NPAIRS = 20        # pairs of points compared either side of each trial time
fontsize = 9

# Column names accepted in light curve files that have a header row
TIME_NAMES = ("HJD", "BJD", "BJD_TDB", "JD", "TIME")
MAG_NAMES = ("Source_AMag_T1", "AMag", "MAG", "MAGNITUDE")
ERR_NAMES = ("Source_AMag_Err_T1", "Source_AMag_T1_Error", "AMag Error", "MAG_ERR", "ERROR", "ERR")


class MinimumError(ValueError):
    """A minimum couldn't be measured, with a message saying what to change."""


# ---------------------------------------------------------------------------
# Small helpers kept from the original program
# ---------------------------------------------------------------------------

def percent_to_xy(xy, xlist, ylist, x_rev=False, y_rev=False):
    xrange = max(xlist) - min(xlist)
    yrange = max(ylist) - min(ylist)
    x0 = max(xlist) if x_rev else min(xlist)
    y0 = max(ylist) if y_rev else min(ylist)
    return x0 + xrange * xy[0], y0 + yrange * xy[1]


def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def best_root(coef, lb, rb, filter_files=None):
    """
    The root of the polynomial's derivative between lb and rb closest to the
    middle, which is the polynomial's estimate of the minimum. np.roots
    returns plenty of complex and out-of-range roots, so they're filtered.
    Raises MinimumError if none is left.
    """
    dcoef = [n * coef[n] for n in range(1, len(coef))]
    allroots = np.unique(np.real(np.roots(dcoef[::-1])))
    goodroots = [root for root in allroots
                 if lb < root < rb and -1e-2 < calc.poly.result(dcoef, root) < 1e-2]
    if not goodroots:
        raise MinimumError("The polynomial has no turning point between the boundaries.")
    root_est = (lb + rb) / 2
    return min(goodroots, key=lambda root: abs(root - root_est))


# ---------------------------------------------------------------------------
# Kwee-van Woerden
# ---------------------------------------------------------------------------

def calc_S(fracHJD, flux, guess_min, npairs=None):
    """
    The Kwee-van Woerden S statistic for one trial time: the summed squared
    difference between the light curve reflected about guess_min and itself,
    using npairs evenly spaced pairs. Also returns K, the number of real
    observations that fall in the reflected range.
    """
    interp = scipy.interpolate.interp1d(fracHJD, flux, fill_value="extrapolate")
    max_range = min([guess_min - min(fracHJD), max(fracHJD) - guess_min])
    N = sum(1 for time in fracHJD if guess_min - max_range < time < guess_min + max_range)
    # K is the number of observation pairs, a rough measure of how well the
    # minimum is covered. The pairs themselves are interpolated at even
    # spacing from the trial time, which is the ideal case.
    K = int(np.floor(N / 2))
    if npairs is None:
        npairs = DEFAULT_NPAIRS
    step = max_range / npairs
    klist = np.arange(1, npairs + 1, 1)
    S = sum((interp(guess_min - klist * step) - interp(guess_min + klist * step)) ** 2)
    return S, K


@dataclass
class KweeResult:
    """Everything the Kwee-van Woerden fit produces, for reporting and plotting."""
    t_kw: float                 # time of minimum, in the same units as the input times
    error: float | None         # formal error, or None when it can't be estimated
    Z: float                    # half the number of pairs at t_kw (K / 2)
    guesses: np.ndarray         # trial times
    S: np.ndarray               # S at each trial time
    parabola: np.ndarray        # coefficients (ascending) of the parabola fitted near the lowest S
    para_times: np.ndarray      # trial times used for the parabola
    para_S: np.ndarray
    R2: float                   # how well the parabola fits


def kwee_van_woerden(times, flux, resolution=DEFAULT_RESOLUTION, para_range=None, npairs=DEFAULT_NPAIRS,
                     discard=0.1):
    """
    Find the time of minimum with the Kwee-van Woerden method.

    S is worked out at evenly spaced trial times; a parabola through the
    lowest S values gives the minimum, and its curvature gives the error.
    Raises MinimumError when there aren't enough points to do this.

    Close to either boundary there's hardly any light curve to reflect, so S
    drops towards zero there whatever the shape. The lowest S is therefore
    only looked for away from the outer discard fraction at each end. (The
    original code meant to do this but overwrote the setting, so flat-
    bottomed or wide windows could give the boundary as the minimum.)
    """
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(times) < 5:
        raise MinimumError(f"Only {len(times)} points between the boundaries; widen them.")

    ordered = np.sort(times)
    guesses = np.linspace(ordered[1], ordered[-2], resolution)
    Slist, Klist = [], []
    for guess in guesses:
        S, K = calc_S(times, flux, guess, npairs=npairs)
        Slist.append(S)
        Klist.append(K)
    Slist = np.array(Slist)
    span = times.max() - times.min()
    allowed = np.flatnonzero((guesses >= times.min() + span * discard) &
                             (guesses <= times.max() - span * discard))
    if len(allowed) == 0:
        allowed = np.arange(len(guesses))
    best_index = int(allowed[np.argmin(Slist[allowed])])
    T_BF = guesses[best_index]

    if para_range is None:
        para_range = int(resolution * 0.05)
    para_range = max(int(para_range), 1)
    # Use the trial times either side of the lowest S, without running off
    # either end of the list (a negative index would quietly wrap around)
    lo = max(best_index - para_range, 0)
    hi = min(best_index + para_range, len(guesses) - 1)
    if hi - lo < 2:
        raise MinimumError("The lowest point is at the edge of the boundaries. Move them so the "
                           "eclipse sits in the middle.")
    # Lowest S first, as the original ordering did, so [:3] are the closest three
    order_ = sorted(range(lo, hi + 1), key=lambda i: (abs(i - best_index), i))
    paraHJD = guesses[order_]
    paraS = Slist[order_]

    with np.errstate(divide="ignore", invalid="ignore"):
        coef, _err, R2 = calc.poly.regr_polyfit(paraHJD, paraS, 2)[:3]
    a, b, c = coef[::-1]

    def vertex_ok(a_, b_, c_):
        if a_ <= 0:
            return False
        vertex = -b_ / (2 * a_)
        return c_ - b_ ** 2 / (4 * a_) >= 0 and calc_S(times, flux, vertex, npairs=npairs)[0] <= paraS[0]

    if vertex_ok(a, b, c):
        T_kw = -b / (2 * a)
    else:
        # Fall back to a parabola through the three lowest points, then to the lowest point itself
        with np.errstate(divide="ignore", invalid="ignore"):
            coef2 = calc.poly.regr_polyfit(paraHJD[:3], paraS[:3], 2)[0]
        a2, b2, c2 = coef2[::-1]
        T_kw = -b2 / (2 * a2) if vertex_ok(a2, b2, c2) else T_BF

    S_at_tkw, K = calc_S(times, flux, T_kw, npairs=npairs)
    Z = K / 2
    error = None
    if Z > 1 and a > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(np.sqrt(S_at_tkw / (a * (Z - 1))))
        if np.isfinite(value):
            error = value

    return KweeResult(t_kw=float(T_kw), error=error, Z=Z, guesses=guesses, S=Slist,
                      parabola=np.asarray(coef), para_times=paraHJD, para_S=paraS, R2=float(R2))


def KvW(fracHJD, flux, discard=0.1, resolution=100, para_range=None, plot=False,
        need_error=False, ax=None, npairs=20, entire_S=False):
    """
    The original interface to the Kwee-van Woerden fit, kept for scripts.

    Returns (time of minimum, error, Z). The error is 1 unless need_error is
    set, and NaN when it can't be estimated. Pass a Matplotlib axis as ax
    to draw S against the trial times. plot is ignored.
    """
    result = kwee_van_woerden(fracHJD, flux, resolution=resolution, para_range=para_range, npairs=npairs,
                              discard=discard)
    if ax is not None:
        draw_S(ax, result, npairs=npairs, entire_S=entire_S)
    if not need_error:
        error = 1
    else:
        error = result.error if result.error is not None else float("nan")
    return result.t_kw, error, result.Z


def sim_min(filter_HJD, filter_flux, filter_fluxerr, order, sims, filter_files=None):
    """
    Monte Carlo error on the polynomial minimum: refit simulated light curves
    drawn from each point's error and take the spread of the minima.
    """
    filters = len(filter_HJD)
    all_sims = [[] for _ in range(sims)]
    HJD_inbounds = []
    for band in range(filters):
        results = calc.poly.regr_polyfit(filter_HJD[band], filter_flux[band], order)[3]
        HJD_inbounds += list(filter_HJD[band])
        for sim in range(sims):
            all_sims[sim] += list(FT.sim_ob_flux(results, filter_fluxerr[band], lower=-10, upper=10))
    rootlist = [best_root(calc.poly.regr_polyfit(HJD_inbounds, all_sims[sim], order)[0],
                          min(HJD_inbounds), max(HJD_inbounds))
                for sim in range(sims)]
    return statistics.stdev(rootlist), rootlist, 0


# ---------------------------------------------------------------------------
# Reading light curves and splitting them into nights
# ---------------------------------------------------------------------------

def _pick(columns, names):
    upper = {str(c).strip().upper(): c for c in columns}
    for name in names:
        if name.upper() in upper:
            return upper[name.upper()]
    return None


def read_light_curve(path):
    """
    Read a light curve file and return (time, magnitude, error) arrays sorted
    by time.

    Files without a header use the first three columns, as Night Filters
    writes them (examples/test_B.txt). Files with a header, such as the CSV
    files Multi-Aperture Photometry saves, are read by column name.
    """
    path = Path(path)
    if not path.is_file():
        raise MinimumError(f"{path} doesn't exist.")
    with open(path, encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    tokens = first.replace(",", " ").split()
    try:
        [float(t) for t in tokens[:3]]
        has_header = False
    except ValueError:
        has_header = True

    try:
        if has_header:
            df = pd.read_csv(path, sep=None, engine="python")
            t_col, m_col, e_col = (_pick(df.columns, names) for names in (TIME_NAMES, MAG_NAMES, ERR_NAMES))
            if None in (t_col, m_col, e_col):
                raise MinimumError(f"{path.name} needs time, magnitude and error columns (for example "
                                   "HJD, Source_AMag_T1 and Source_AMag_Err_T1).")
            df = df[[t_col, m_col, e_col]]
        else:
            df = pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python")
            if df.shape[1] < 3:
                raise MinimumError(f"{path.name} needs at least three columns: time, magnitude and error.")
            df = df.iloc[:, :3]
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise MinimumError(f"{path.name} couldn't be read: {e}") from e

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise MinimumError(f"{path.name} has no rows with numbers in them.")
    df = df.sort_values(df.columns[0], kind="stable")
    return tuple(df.iloc[:, i].to_numpy(dtype=float) for i in range(3))


@dataclass
class Band:
    """One filter's points on one night, as relative flux."""
    label: str
    time: np.ndarray            # days after the night's whole-day offset
    flux: np.ndarray
    flux_err: np.ndarray


@dataclass
class Night:
    """
    The observations from one night. Nights run noon to noon UT, so a night
    is every point sharing the same whole Julian day; times are stored as
    days after that whole day (the offset).
    """
    index: int
    offset: int
    bands: dict = field(default_factory=dict)

    @property
    def date(self):
        """Calendar date of the evening the night started (UT)."""
        return Time(self.offset, format="jd").to_value("iso", subfmt="date")

    def points(self, labels=None):
        """Time, flux and error for the chosen filters (all of them by default), sorted by time."""
        chosen = [b for b in self.bands.values() if labels is None or b.label in labels]
        if not chosen:
            return np.array([]), np.array([]), np.array([])
        t = np.concatenate([b.time for b in chosen])
        f = np.concatenate([b.flux for b in chosen])
        e = np.concatenate([b.flux_err for b in chosen])
        order_ = np.argsort(t, kind="stable")
        return t[order_], f[order_], e[order_]

    def count(self, labels=None):
        return len(self.points(labels)[0])

    def span_hours(self, labels=None):
        t = self.points(labels)[0]
        return 24 * (t.max() - t.min()) if len(t) else 0.0

    def describe(self, labels=None):
        return (f"Night {self.index + 1}: {self.date}, {self.count(labels)} points over "
                f"{self.span_hours(labels):.1f} h")


def load_nights(files, labels=None, norm_method="norm"):
    """
    Read one light curve per filter and split them into nights.

    Magnitudes become relative flux, 10^(-0.4 (m - m_ref)), where m_ref is
    the filter's brightest magnitude ("norm") or its mean ("subavg"). The
    filters are kept apart so the window can fit any combination of them.

    :param files: light curve file paths, one per filter
    :param labels: a name for each file, such as "B"; defaults to File 1, File 2...
    :return: list of Night, in time order
    """
    files = list(files)
    if not files:
        raise MinimumError("Add at least one light curve file.")
    labels = list(labels) if labels else [f"File {i + 1}" for i in range(len(files))]
    by_day = {}
    for path, label in zip(files, labels):
        time, mag, err = read_light_curve(path)
        reference = np.mean(mag) if norm_method == "subavg" else np.min(mag)
        flux = 10 ** (-0.4 * (mag - reference))
        flux_err = 0.4 * np.log(10) * err * flux
        days = np.floor(time).astype(int)
        for day in np.unique(days):
            mask = days == day
            by_day.setdefault(int(day), {})[label] = Band(label, time[mask] - day, flux[mask], flux_err[mask])
    return [Night(index=i, offset=day, bands=by_day[day]) for i, day in enumerate(sorted(by_day))]


# ---------------------------------------------------------------------------
# Measuring one minimum
# ---------------------------------------------------------------------------

@dataclass
class Minimum:
    """A measured time of minimum."""
    hjd: float                  # full time of minimum
    error: float | None
    night: int                  # index of the night it came from
    date: str
    lb: float                   # boundaries, in days after the night's offset
    rb: float
    n_points: int               # observations between the boundaries
    labels: tuple               # filters used
    kwee: KweeResult
    poly: np.ndarray            # polynomial through the eclipse, for drawing
    poly_R2: float


def measure_minimum(night, lb, rb, labels=None, order=DEFAULT_ORDER, resolution=DEFAULT_RESOLUTION,
                    npairs=DEFAULT_NPAIRS, para_range=None):
    """
    Measure the time of minimum between two boundaries on one night.

    lb and rb are in days after the night's offset (the x axis of the plot).
    Raises MinimumError with a message to show the user if it can't be done.
    """
    lb, rb = sorted((float(lb), float(rb)))
    t, f, _e = night.points(labels)
    inside = (t > lb) & (t < rb)
    n = int(inside.sum())
    needed = max(order + 2, 5)
    if n < needed:
        raise MinimumError(f"Only {n} points between the boundaries; at least {needed} are needed. "
                           "Widen them or choose more filters.")
    t_in, f_in = t[inside], f[inside]
    with np.errstate(divide="ignore", invalid="ignore"):
        poly, _err, poly_R2 = calc.poly.regr_polyfit(t_in, f_in, order)[:3]
    kwee = kwee_van_woerden(t_in, f_in, resolution=resolution, para_range=para_range, npairs=npairs)
    used = tuple(b for b in night.bands if labels is None or b in labels)
    return Minimum(hjd=night.offset + kwee.t_kw, error=kwee.error, night=night.index, date=night.date,
                   lb=lb, rb=rb, n_points=n, labels=used, kwee=kwee, poly=np.asarray(poly),
                   poly_R2=float(poly_R2))


def guess_boundaries(night, labels=None, half_width=0.04):
    """
    Starting boundaries around the faintest stretch of the night: the lowest
    point of a running median, give or take half_width days (narrower on
    short nights).
    """
    t, f, _e = night.points(labels)
    if len(t) < 3:
        return (t.min(), t.max()) if len(t) else (0.0, 1.0)
    window = max(3, len(f) // 15) | 1
    smooth = pd.Series(f).rolling(window, center=True, min_periods=1).median().to_numpy()
    centre = t[int(np.argmin(smooth))]
    width = min(half_width, (t.max() - t.min()) / 4)
    lb = max(centre - width, t.min())
    rb = min(centre + width, t.max())
    return float(lb), float(rb)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_S(ax, result, npairs=DEFAULT_NPAIRS, entire_S=False):
    """S against trial time, with the fitted parabola and the minimum it gives."""
    ax.plot(result.guesses, result.S, "o", color="0.15", ms=3)
    px, py = calc.poly.polylist(result.parabola, min(result.para_times), max(result.para_times), 100)
    ax.plot(px, py, color="#2f5aa8", lw=1)
    ax.axvline(result.t_kw, color="#2f5aa8", lw=1, ls="--")
    if not entire_S:
        s_range = (max(result.para_S) - min(result.para_S)) or 1.0
        t_range = (max(result.para_times) - min(result.para_times)) or 1.0
        ax.set_ylim(min(result.para_S) - s_range * 0.07, max(result.para_S) + s_range * 0.07)
        ax.set_xlim(min(result.para_times) - t_range * 0.025, max(result.para_times) + t_range * 0.025)
    ax.set_ylabel("S", fontsize=fontsize)
    ax.text(0.99, 0.95, f"$R^2$ = {result.R2:.5f}, K = {npairs} ({int(result.Z * 2)})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)


def draw_night(ax_s, ax_lc, night, lb, rb, minimum=None, labels=None, colours=None):
    """
    The light curve for one night with the boundaries, and, when a minimum
    was measured, the S curve above it and the minimum marked on both.
    """
    colours = colours or {"B": "#2f5aa8", "V": "#2d7d46", "R": "#b23a2e"}
    ax_s.clear()
    ax_lc.clear()
    for label, band in night.bands.items():
        used = labels is None or label in labels
        ax_lc.errorbar(band.time, band.flux, band.flux_err, fmt="o", ms=3, elinewidth=0.6, capsize=0,
                       color=colours.get(label, "0.2"), alpha=1.0 if used else 0.25,
                       label=label if len(night.bands) > 1 else None)
    for x in (lb, rb):
        ax_lc.axvline(x, color="#2e9e48", ls="--", lw=1)
    if minimum is not None:
        px, py = calc.poly.polylist(minimum.poly, minimum.lb, minimum.rb, 200)
        ax_lc.plot(px, py, color="#c0392b", lw=1, zorder=10)
        ax_lc.axvline(minimum.kwee.t_kw, color="#2f5aa8", lw=1)
        draw_S(ax_s, minimum.kwee)
    else:
        ax_s.set_yticks([])
    ax_lc.set_xlabel(f"HJD - {night.offset}", fontsize=fontsize)
    ax_lc.set_ylabel("Relative flux", fontsize=fontsize)
    ax_lc.grid(alpha=0.3)
    if len(night.bands) > 1:
        ax_lc.legend(fontsize=7, loc="lower left", frameon=False)
    ax_s.xaxis.set_label_position("top")
    ax_s.xaxis.tick_top()
    for ax in (ax_s, ax_lc):
        ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Saved minima
# ---------------------------------------------------------------------------

class MinimumFile:
    """
    The times of minimum saved for one object and set of filters.

    Each line is tab separated: time of minimum, number of points used, and
    error. O-C Plotting reads the first and third columns. Measuring the
    same eclipse again replaces the earlier value.
    """

    SAME_ECLIPSE = 0.05  # days; minima closer than this are the same eclipse

    def __init__(self, path):
        self.path = Path(path)
        self.entries = []  # (time, points, error)
        if self.path.is_file():
            try:
                df = pd.read_csv(self.path, sep=r"\s+", header=None)
                for row in df.itertuples(index=False):
                    self.entries.append((float(row[0]), int(row[1]), float(row[2])))
            except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, IndexError):
                pass
        self.entries.sort()

    def add(self, minimum):
        """Add a minimum and save. Returns True if it replaced an earlier one."""
        if minimum.error is None:
            raise MinimumError("This minimum has no error estimate, so it can't be used for O-C. "
                               "Widen the boundaries so more points are paired.")
        replaced = any(abs(t - minimum.hjd) < self.SAME_ECLIPSE for t, _n, _e in self.entries)
        self.entries = [e for e in self.entries if abs(e[0] - minimum.hjd) >= self.SAME_ECLIPSE]
        self.entries.append((minimum.hjd, minimum.n_points, minimum.error))
        self.entries.sort()
        self.save()
        return replaced

    def remove(self, index):
        del self.entries[index]
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            for t, n, err in self.entries:
                handle.write(f"{t:.6f}\t{n}\t{err:.9f}\n")


def minimum_file_name(obj_name, labels):
    """File name for an object's minima in the given filters, e.g. NSVS_B_min.txt."""
    stem = "_".join(str(obj_name).split()) or "target"
    return f"{stem}_{''.join(labels)}_min.txt"


# ---------------------------------------------------------------------------
# Terminal entry point
# ---------------------------------------------------------------------------

def main():
    """
    Ask for one to three light curve files in the terminal, then open the
    Find Minimum window. Minima are saved in the current folder.
    """
    from .find_min_window import run_standalone

    while True:
        answer = input("How many filters do you have (1-3), or 'Close' to stop: ").strip()
        if answer.lower() == "close":
            return 0
        if answer in ("1", "2", "3"):
            break
        print("Enter 1, 2 or 3, or 'Close'.\n")
    files = []
    for i in range(int(answer)):
        while True:
            path = Path(input(f"Light curve file {i + 1}: ").strip().strip('"')).expanduser()
            if path.is_file():
                files.append(path)
                break
            print("That file doesn't exist. Try again.\n")
    name = input("Object name (used to name the minimum file): ").strip() or "target"
    return run_standalone(files, None, Path.cwd(), name)


if __name__ == '__main__':
    main()
