# -*- coding: utf-8 -*-
"""
Finding the orbital period of an eclipsing binary from its light curve.

The light curve is folded on many trial periods and each fold is scored:

+ ANOVA (the default) splits the fold into phase bins and asks how much
  better the bin means describe the magnitudes than one overall mean.
+ Multi-harmonic fits a sum of sines and cosines of the phase and asks how
  much of the scatter it explains.
+ Lomb-Scargle fits a single sine, using the magnitude errors.

Folding two similar eclipses on top of each other scores well with every
method, so the highest peak is often half the orbital period. After the
search the fold on twice the best period is checked: if its two halves show
different eclipses, the orbital period is the doubled one (halves_differ).

The trial periods are spaced evenly in frequency, with the spacing set by
how long the observations span, so narrow peaks from long baselines aren't
stepped over. Nothing here needs a display; the window is period_window.py.

Author: Kyle Koeller
Created: 10/10/2026
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.timeseries import LombScargle

from .find_min import MinimumError, read_light_curve

METHODS = ("ANOVA", "Multi-harmonic", "Lomb-Scargle")
DEFAULT_METHOD = "ANOVA"
DEFAULT_BINS = 20           # phase bins for ANOVA
DEFAULT_MIN_BIN = 3         # bins with fewer points are left out
DEFAULT_HARMONICS = 4       # sharp eclipses need more than one or two
DEFAULT_OVERSAMPLE = 10     # trial frequencies per peak width
MAX_TRIALS = 300_000        # most trial periods in one search
MAX_WORK = 150_000_000      # trial periods times points, to keep a search to seconds


class PeriodError(ValueError):
    """A period search couldn't be run, with a message saying what to change."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class LightCurve:
    """Times, magnitudes and errors from one or more filters, each filter's median taken off."""
    time: np.ndarray
    mag: np.ndarray
    err: np.ndarray
    label: np.ndarray                      # filter of each point
    labels: tuple = ()

    def select(self, labels):
        """The points from the chosen filters only."""
        keep = np.isin(self.label, list(labels))
        return LightCurve(self.time[keep], self.mag[keep], self.err[keep], self.label[keep],
                          tuple(lab for lab in self.labels if lab in labels))

    @property
    def baseline(self):
        return float(self.time.max() - self.time.min()) if len(self.time) else 0.0


def load_light_curves(files, labels=None):
    """
    Read one light curve per filter (the same files Find Minimum takes) into
    one LightCurve. Filters differ in brightness, so each filter's median
    magnitude is subtracted before they're put together.
    """
    files = list(files)
    if not files:
        raise PeriodError("Add at least one light curve file.")
    labels = list(labels) if labels else [f"File {i + 1}" for i in range(len(files))]
    times, mags, errs, which = [], [], [], []
    for path, label in zip(files, labels):
        try:
            t, m, e = read_light_curve(path)
        except MinimumError as exc:
            raise PeriodError(str(exc)) from exc
        times.append(t)
        mags.append(m - np.median(m))
        errs.append(e)
        which.append(np.full(len(t), label, dtype=object))
    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    return LightCurve(t[order], np.concatenate(mags)[order], np.concatenate(errs)[order],
                      np.concatenate(which)[order], tuple(labels))


# ---------------------------------------------------------------------------
# Trial periods
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    periods: np.ndarray        # ascending
    oversample: float          # trial frequencies per peak width actually used
    coarse: bool = False       # True when the grid had to be thinned to keep the search quick


def frequency_grid(time, min_period, max_period, oversample=DEFAULT_OVERSAMPLE, n_points=None):
    """
    Trial periods between min_period and max_period, evenly spaced in
    frequency. A peak is about 1/baseline wide in frequency, so the spacing
    is 1/(baseline x oversample). If that would be too many trials for the
    number of points, the spacing is widened and Grid.coarse is set.
    """
    if not (0 < min_period < max_period):
        raise PeriodError("The shortest period has to be above zero and below the longest.")
    baseline = float(np.max(time) - np.min(time))
    if baseline <= 0:
        raise PeriodError("All the observations are at the same time.")
    if max_period > baseline:
        raise PeriodError(f"The longest period ({max_period:g} d) is longer than the "
                          f"observations span ({baseline:.2f} d). Lower it.")
    f_lo, f_hi = 1.0 / max_period, 1.0 / min_period
    n_points = len(time) if n_points is None else n_points
    wanted = int(np.ceil((f_hi - f_lo) * baseline * oversample)) + 1
    limit = int(min(MAX_TRIALS, MAX_WORK / max(n_points, 1)))
    coarse = wanted > limit
    count = max(min(wanted, limit), 50)
    freqs = np.linspace(f_lo, f_hi, count)
    used = (count - 1) / ((f_hi - f_lo) * baseline)
    return Grid(periods=1.0 / freqs[::-1], oversample=used, coarse=coarse)


# ---------------------------------------------------------------------------
# Scoring folds
# ---------------------------------------------------------------------------

def _chunks(n_periods, n_points, per_chunk):
    size = max(1, int(per_chunk // max(n_points, 1)))
    for start in range(0, n_periods, size):
        yield slice(start, min(start + size, n_periods))


def _phases(time, periods):
    # Subtract the first time before dividing so the phases keep their precision
    return np.mod((time - time[0])[None, :] / periods[:, None], 1.0)


def anova_statistic(time, mag, periods, bins=DEFAULT_BINS, min_bin=DEFAULT_MIN_BIN):
    """
    One-way ANOVA F statistic of the magnitudes grouped into phase bins, for
    each trial period. Bins with fewer than min_bin points are left out, and
    NaN is returned where fewer than two bins are left. Gives the same values
    as scipy.stats.f_oneway on the binned magnitudes.
    """
    time = np.asarray(time, float)
    y = np.asarray(mag, float)
    y = y - y.mean()
    periods = np.asarray(periods, float)
    out = np.full(len(periods), np.nan)
    for part in _chunks(len(periods), len(y), 2_000_000):
        P = periods[part]
        c = len(P)
        idx = np.minimum((_phases(time, P) * bins).astype(int), bins - 1)
        flat = (idx + bins * np.arange(c)[:, None]).ravel()
        counts = np.bincount(flat, minlength=c * bins).reshape(c, bins)
        sums = np.bincount(flat, weights=np.tile(y, c), minlength=c * bins).reshape(c, bins)
        sumsq = np.bincount(flat, weights=np.tile(y * y, c), minlength=c * bins).reshape(c, bins)
        keep = counts >= min_bin
        k = keep.sum(axis=1)
        n = np.where(keep, counts, 0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            between_terms = np.where(keep, sums ** 2 / np.where(counts > 0, counts, 1), 0).sum(axis=1)
            total_sum = np.where(keep, sums, 0).sum(axis=1)
            ssb = between_terms - total_sum ** 2 / n
            ssw = np.where(keep, sumsq, 0).sum(axis=1) - between_terms
            F = (ssb / (k - 1)) / (ssw / (n - k))
        F[(k < 2) | (n <= k)] = np.nan
        out[part] = F
    return out


def multiharmonic_statistic(time, mag, periods, harmonics=DEFAULT_HARMONICS):
    """
    F statistic of a least-squares fit of a constant plus `harmonics` sine and
    cosine terms of the phase, for each trial period: the variance the fit
    explains per term over the variance left per degree of freedom.

    The normal equations only need sums of cos(k phase) and sin(k phase) for
    k up to twice the number of harmonics (products of two harmonics are
    harmonics of their sum and difference), which are built up from powers
    of exp(i phase) rather than forming the full design matrix.
    """
    time = np.asarray(time, float)
    y = np.asarray(mag, float)
    y = y - y.mean()
    N = len(y)
    H = int(harmonics)
    p = 1 + 2 * H
    if N <= p:
        raise PeriodError(f"Multi-harmonic with {H} harmonics needs more than {p} points.")
    periods = np.asarray(periods, float)
    sst = float(np.sum(y * y))
    out = np.full(len(periods), np.nan)
    for part in _chunks(len(periods), N, 1_000_000):
        z = np.exp(2j * np.pi * _phases(time, periods[part]))      # chunk x N
        c = z.shape[0]
        S = np.empty((c, 2 * H + 1), complex)                    # sums of z^k
        Y = np.empty((c, H + 1), complex)                        # sums of y z^k
        zk = np.ones_like(z)
        S[:, 0] = N
        Y[:, 0] = y.sum()
        for k in range(1, 2 * H + 1):
            zk = zk * z
            S[:, k] = zk.sum(axis=1)
            if k <= H:
                Y[:, k] = zk @ y
        C, Sn = S.real, S.imag

        def cos_sum(k):
            return C[:, abs(k)]

        def sin_sum(k):
            return np.sign(k) * Sn[:, abs(k)]

        xtx = np.empty((c, p, p))
        xty = np.empty((c, p))
        xtx[:, 0, 0] = N
        xty[:, 0] = Y[:, 0].real
        for a in range(1, H + 1):
            ca, sa = 2 * a - 1, 2 * a      # columns of cos(a phase) and sin(a phase)
            xtx[:, 0, ca] = xtx[:, ca, 0] = cos_sum(a)
            xtx[:, 0, sa] = xtx[:, sa, 0] = sin_sum(a)
            xty[:, ca] = Y[:, a].real
            xty[:, sa] = Y[:, a].imag
            for b in range(1, H + 1):
                cb, sb = 2 * b - 1, 2 * b
                xtx[:, ca, cb] = (cos_sum(a - b) + cos_sum(a + b)) / 2
                xtx[:, sa, sb] = (cos_sum(a - b) - cos_sum(a + b)) / 2
                xtx[:, ca, sb] = (sin_sum(a + b) - sin_sum(a - b)) / 2
                xtx[:, sb, ca] = xtx[:, ca, sb]
        xtx += np.eye(p) * 1e-12 * np.trace(xtx, axis1=1, axis2=2)[:, None, None]
        beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
        sse = sst - np.einsum("cp,cp->c", beta, xty)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = ((sst - sse) / (p - 1)) / (sse / (N - p))
        F[sse <= 0] = np.nan
        out[part] = F
    return out


def lomb_scargle_statistic(time, mag, err, periods):
    """Lomb-Scargle power at each trial period, weighted by the magnitude errors when they're usable."""
    dy = np.asarray(err, float)
    if not np.all(np.isfinite(dy)) or np.any(dy <= 0):
        dy = None
    frequency = 1.0 / np.asarray(periods, float)
    order = np.argsort(frequency)  # astropy wants the frequencies rising
    power = np.empty_like(frequency)
    power[order] = LombScargle(time, mag, dy).power(frequency[order])
    return power


def score(curve, periods, method=DEFAULT_METHOD, bins=DEFAULT_BINS, harmonics=DEFAULT_HARMONICS):
    """Score every trial period with the chosen method."""
    if method == "ANOVA":
        return anova_statistic(curve.time, curve.mag, periods, bins=bins)
    if method == "Multi-harmonic":
        return multiharmonic_statistic(curve.time, curve.mag, periods, harmonics=harmonics)
    if method == "Lomb-Scargle":
        return lomb_scargle_statistic(curve.time, curve.mag, curve.err, periods)
    raise PeriodError(f"Unknown method {method!r}; choose one of {', '.join(METHODS)}.")


def statistic_name(method):
    return "Power" if method == "Lomb-Scargle" else "F statistic"


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

def _binned_part(curve, period, part, parts, bins):
    """Bin means and their variances for one part of a fold on `parts` times `period`."""
    phase = fold(curve.time, period * parts, curve.time[0]) * parts
    chosen = np.floor(phase).astype(int) == part
    idx = np.minimum(((phase % 1) * bins).astype(int), bins - 1)
    means = np.full(bins, np.nan)
    variances = np.full(bins, np.nan)
    for b in range(bins):
        values = curve.mag[chosen & (idx == b)]
        if len(values) >= 3:
            means[b] = values.mean()
            variances[b] = values.var(ddof=1) / len(values)
    return means, variances


def _mismatch(first, second):
    """Mean squared difference of two binned curves, in units of their combined errors."""
    (m1, v1), (m2, v2) = first, second
    ok = np.isfinite(m1) & np.isfinite(m2) & (v1 + v2 > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.mean((m1[ok] - m2[ok]) ** 2 / (v1[ok] + v2[ok])))


def halves_differ(curve, period, bins=20, ratio_needed=5.0):
    """
    Whether the orbital period is twice `period`. Returns (differ, ratio).

    Folded on twice `period`, the two halves are compared bin by bin. If
    `period` is half the orbit, the halves hold the primary and secondary
    eclipses and differ; if it's the whole orbit, they're the same stretch of
    light curve. Night-to-night offsets make even matching halves differ a
    little, so the difference is measured against the same comparison
    between parts that cover the same stretch of orbit either way (quarters
    one and three, and two and four, of a fold on four times `period`). The
    halves have to differ `ratio_needed` times more than that.

    When the two eclipses are too alike to tell apart, the ratio stays near
    one and only the fold, seen by eye, can settle it.
    """
    halves = _mismatch(_binned_part(curve, period, 0, 2, bins), _binned_part(curve, period, 1, 2, bins))
    same = np.nanmean([_mismatch(_binned_part(curve, period, 0, 4, bins), _binned_part(curve, period, 2, 4, bins)),
                       _mismatch(_binned_part(curve, period, 1, 4, bins), _binned_part(curve, period, 3, 4, bins))])
    if not (np.isfinite(halves) and np.isfinite(same)) or same <= 0:
        return False, float("nan")
    ratio = halves / same
    return bool(ratio > ratio_needed and halves > 3), float(ratio)


@dataclass
class SearchResult:
    """A periodogram and the best period found in it."""
    method: str
    periods: np.ndarray
    power: np.ndarray
    best_period: float
    best_power: float
    half_width: float           # half width of the best peak at half its height, in days
    grid: Grid
    min_period: float
    max_period: float
    settings: dict = field(default_factory=dict)
    doubled_from: float | None = None   # the periodogram's peak, when the orbital period is twice it


def peak_sharpness(method, bins=DEFAULT_BINS, harmonics=DEFAULT_HARMONICS):
    """
    How many times narrower than 1/baseline this method's peaks are. A
    single sine (Lomb-Scargle) gives peaks about 1/baseline wide, but folds
    scored on fine detail (ANOVA's phase bins, the multi-harmonic fit's
    higher harmonics) respond to phase shifts that many times smaller, and
    their peaks narrow to match. The trial periods are packed that much
    closer so the tops of the peaks aren't missed.
    """
    if method == "ANOVA":
        return max(bins // 2, 1)
    if method == "Multi-harmonic":
        return max(2 * int(harmonics), 1)
    return 1


def search(curve, min_period, max_period, method=DEFAULT_METHOD, oversample=DEFAULT_OVERSAMPLE,
           bins=DEFAULT_BINS, harmonics=DEFAULT_HARMONICS):
    """
    Score trial periods between min_period and max_period and find the best,
    refined on a finer grid around the highest peak.
    """
    if len(curve.time) < 10:
        raise PeriodError(f"Only {len(curve.time)} points; a period search needs at least 10.")
    grid = frequency_grid(curve.time, min_period, max_period,
                          oversample=oversample * peak_sharpness(method, bins, harmonics))
    power = score(curve, grid.periods, method, bins=bins, harmonics=harmonics)
    if not np.any(np.isfinite(power)):
        raise PeriodError("No trial period could be scored. Widen the period range or add points.")
    best_index = int(np.nanargmax(power))
    settings = {"bins": bins, "harmonics": harmonics, "oversample": oversample}
    best, best_power, half_width = refine(curve, grid.periods[best_index], method, **_refine_args(settings))
    doubled_from = None
    # Folding two alike eclipses on top of each other scores well, so the
    # highest peak is often half the orbital period. If the fold on twice it
    # shows two different eclipses, the doubled period is the orbit.
    if 4 * best <= curve.baseline and halves_differ(curve, best)[0]:
        # The peak itself is the sharpest measure, so double it rather than
        # searching again around the orbital period, where the score is weaker
        doubled_from = best
        best, half_width = 2 * best, 2 * half_width
        best_power = float(score(curve, [best], method, bins=bins, harmonics=harmonics)[0])
    return SearchResult(method, grid.periods, power, best, best_power, half_width, grid,
                        min_period, max_period, settings, doubled_from)


def _refine_args(settings):
    return {"bins": settings.get("bins", DEFAULT_BINS), "harmonics": settings.get("harmonics", DEFAULT_HARMONICS)}


def refine(curve, period, method=DEFAULT_METHOD, bins=DEFAULT_BINS, harmonics=DEFAULT_HARMONICS):
    """
    Look closely around one period: score 401 trials across one peak width
    either side and return (best period, its score, half width of the peak
    at half height in days). The half width is a rough guide to how well the
    period is pinned down by these observations alone.
    """
    baseline = curve.baseline
    f0 = 1.0 / period
    span = 1.0 / baseline
    freqs = np.linspace(max(f0 - span, f0 * 0.5), f0 + span, 401)
    periods = 1.0 / freqs[::-1]
    power = score(curve, periods, method, bins=bins, harmonics=harmonics)
    if not np.any(np.isfinite(power)):
        return float(period), float("nan"), float("nan")
    i = int(np.nanargmax(power))
    best, best_power = float(periods[i]), float(power[i])
    floor = np.nanmin(power)
    half = floor + (best_power - floor) / 2
    lo = i
    while lo > 0 and power[lo - 1] >= half:
        lo -= 1
    hi = i
    while hi < len(power) - 1 and power[hi + 1] >= half:
        hi += 1
    return best, best_power, float((periods[hi] - periods[lo]) / 2)


def nearest_peak(result, period, window=0.01):
    """
    The highest point of the periodogram within a fraction `window` of the
    plotted range around `period`, for picking a peak with the mouse.
    """
    width = (result.periods[-1] - result.periods[0]) * window
    near = np.flatnonzero(np.abs(result.periods - period) <= width)
    if len(near) == 0:
        near = np.array([int(np.argmin(np.abs(result.periods - period)))])
    values = np.where(np.isfinite(result.power[near]), result.power[near], -np.inf)
    return float(result.periods[near[int(np.argmax(values))]])


# ---------------------------------------------------------------------------
# Folding and saving
# ---------------------------------------------------------------------------

def fold(time, period, origin):
    """Phase from 0 to 1 of each time, counted from `origin` (subtracted first to keep precision)."""
    return np.mod((np.asarray(time, float) - origin) / period, 1.0)


def period_file_name(obj_name):
    stem = "_".join(str(obj_name).split()) or "target"
    return f"{stem}_period.txt"


def save_period(path, obj_name, period, result=None, files=()):
    """
    Write the period to a small text file: one "name<TAB>value" line, with
    the search settings underneath for the record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Period search for {obj_name}", f"period_days\t{period:.9f}"]
    if result is not None:
        lines += [f"period_half_width_days\t{result.half_width:.9f}",
                  f"method\t{result.method}",
                  f"period_range_days\t{result.min_period:g}-{result.max_period:g}"]
    if files:
        lines.append("light_curves\t" + ", ".join(str(f) for f in files))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_period(path):
    """Read back the period from a file written by save_period."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("period_days\t"):
            try:
                return float(line.split("\t", 1)[1])
            except ValueError:
                break
    raise PeriodError(f"{Path(path).name} doesn't hold a period.")


# ---------------------------------------------------------------------------
# Terminal entry point
# ---------------------------------------------------------------------------

def main():
    """
    Ask for one to three light curve files in the terminal, then open the
    Period Search window. Save Period writes to the current folder.
    """
    from .period_window import run_standalone

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
    name = input("Object name (used to name the saved file): ").strip() or "target"
    return run_standalone(files, None, Path.cwd(), name)


if __name__ == "__main__":
    main()
