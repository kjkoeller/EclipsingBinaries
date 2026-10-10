"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 04/27/2026
"""

# Python imports
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import warnings
from tqdm import tqdm

# Astropy imports
import ccdproc as ccdp
from astropy.coordinates import SkyCoord
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.wcs import WCS
import astropy.units as u
import json
from astropy import wcs
from ._log import make_logger

# Suppress FITS header standards-compliance warnings that are not actionable
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


# Read noise used when an image has no RDNOISE keyword (the BSUO camera value)
DEFAULT_READ_NOISE = 10.83

# Filters photometered by main, in the order of its radec_list
FILTER_ORDER = ("B", "V", "R")
# Stars in different filters' RADEC files this close together count as the same star
STAR_MATCH_ARCSEC = 1.0

# Header keywords tried, in order, for each timestamp. AstroImageJ writes the
# -OBS names; this package's own reduction writes HJD and BJD_TDB.
HJD_KEYS = ("HJD-OBS", "HJD", "HJD_UTC")
BJD_KEYS = ("BJD-OBS", "BJD_TDB")


_config_loaded_attempted = False
_loaded_config = None
_config_path_used = None
_config_log_printed = False

def load_filter_config(radec_dir):
    global _loaded_config, _config_path_used, _config_loaded_attempted
    if _config_loaded_attempted:
        return _loaded_config, _config_path_used

    _config_loaded_attempted = True

    paths_to_check = []
    if radec_dir:
        paths_to_check.append(Path(radec_dir) / "filter_config.json")
    paths_to_check.append(Path.cwd() / "filter_config.json")
    paths_to_check.append(Path.home() / ".EclipsingBinaries" / "filter_config.json")

    for p in paths_to_check:
        if p.exists():
            try:
                with open(p, "r") as f:
                    _loaded_config = json.load(f)
                _config_path_used = str(p)
                return _loaded_config, _config_path_used
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
    return None, None

def calculate_target_snr(image_data, target_position, aperture_radius, annulus_radii,
                         read_noise=DEFAULT_READ_NOISE):
    """Calculate the Signal-to-Noise Ratio (SNR) for the target star."""
    target_aperture = CircularAperture(target_position, r=aperture_radius)
    target_annulus = CircularAnnulus(target_position, *annulus_radii)

    target_phot_table = aperture_photometry(image_data, target_aperture)
    target_aperture_sum = float(target_phot_table['aperture_sum'][0])

    target_bkg_mean = ApertureStats(image_data, target_annulus).mean
    if np.isnan(target_bkg_mean) or np.isinf(target_bkg_mean):
        target_bkg_mean = 0.0

    target_bkg = target_bkg_mean * target_aperture.area
    target_flx = target_aperture_sum - float(target_bkg)

    target_flux_err = np.sqrt(target_aperture_sum + target_aperture.area * read_noise ** 2)

    if target_flux_err > 0:
        snr = target_flx / target_flux_err
    else:
        snr = 0.0

    return snr, target_flx, target_flux_err, target_bkg_mean


def auto_optimize_radii(image_data, target_position, read_noise=DEFAULT_READ_NOISE):
    """Auto-suggest optimized aperture and annulus radii based on FWHM and SNR maximization."""
    from astropy.modeling import models, fitting

    x, y = target_position
    x_int, y_int = int(np.round(x)), int(np.round(y))
    box_size = 20

    # Define cutout boundaries, ensuring they stay within the image
    y_start = max(0, y_int - box_size)
    y_end = min(image_data.shape[0], y_int + box_size)
    x_start = max(0, x_int - box_size)
    x_end = min(image_data.shape[1], x_int + box_size)

    cutout = image_data[y_start:y_end, x_start:x_end]

    # If the target is too close to the edge, return safe defaults
    if cutout.size == 0:
        return 4.0, 7.0, (12.0, 19.0)

    yy, xx = np.mgrid[:cutout.shape[0], :cutout.shape[1]]

    # Initialize the 2D Gaussian model
    g_init = models.Gaussian2D(amplitude=np.max(cutout) - np.median(cutout),
                               x_mean=x_int - x_start,
                               y_mean=y_int - y_start,
                               x_stddev=2.0,
                               y_stddev=2.0)

    fitter = fitting.LevMarLSQFitter()
    g_fit = fitter(g_init, xx, yy, cutout - np.median(cutout))

    # Calculate FWHM
    sigma = np.mean([g_fit.x_stddev.value, g_fit.y_stddev.value])
    fwhm = 2.355 * sigma

    if fwhm <= 0 or np.isnan(fwhm):
        fwhm = 4.0 # fallback

    # Starting guess based on FWHM
    suggested_aperture = 1.75 * fwhm
    suggested_inner = np.floor(3 * fwhm)

    # Target 10x area for the suggested outer annulus
    ap_area = np.pi * suggested_aperture**2
    ann_area = 10.0 * ap_area

    # Maximize SNR over a small range of aperture radii
    best_snr = 0
    best_aperture = suggested_aperture

    aperture_range = np.arange(max(1.0, fwhm), min(3 * fwhm, 30.0), 0.5)

    for ap in aperture_range:
        # Dynamic constraints
        inn = max(suggested_inner, ap + 1.0)
        # Approximate outer to maintain 10x annulus area proportion
        ann_area = 10.0 * np.pi * ap**2
        out = np.sqrt(ann_area / np.pi + inn**2)

        snr, _, _, _ = calculate_target_snr(image_data, target_position, ap, (inn, out), read_noise)
        if snr > best_snr:
            best_snr = snr
            best_aperture = ap

    # Calculate final suggested annulus radii based on best aperture
    best_inner = max(np.floor(3 * fwhm), best_aperture + 1.0)
    best_aperture_area = np.pi * best_aperture**2
    suggested_annulus_area = 2 * best_aperture_area
    best_outer = np.ceil(np.sqrt(suggested_annulus_area / np.pi + best_inner**2))

    return fwhm, best_aperture, (best_inner, best_outer)

def resolve_filter(header, radec_dir=None, log=print):
    global _config_log_printed
    filt_candidates = {
        "Empty/B": "B", "B": "B",
        "Empty/V": "V", "V": "V",
        "Empty/R": "R", "R": "R",
    }

    filter_val = None
    if "FILTER" in header:
        filter_val = header["FILTER"]
    elif "FILTERS" in header:
        filter_val = header["FILTERS"]

    if filter_val in filt_candidates:
        return filt_candidates[filter_val]

    config, config_path = load_filter_config(radec_dir)
    if not config:
        return filter_val

    if not _config_log_printed:
        log(f"Using filter configuration from {config_path}")
        _config_log_printed = True

    for telescope in config.get("TELESCOPE", []):
        ident = telescope.get("identification", {})
        fits_key = ident.get("fits_key")
        match_value = ident.get("match_value")

        if fits_key in header and header[fits_key] == match_value:
            for filt in telescope.get("filters", []):
                f_key = filt.get("fits_key")
                f_match = filt.get("match_value")

                if f_key in header and header[f_key] == f_match:
                    return filt.get("processing_symbol")

    return filter_val

def main(path="", pipeline=False, radec_list=None, obj_name="", write_callback=None, cancel_event=None,
         aperture_radius=20, annulus_radii=(30, 50), read_noise=None):
    """
    Entry point for multi-aperture photometry on Johnson B, V and Cousins R.

    All filters share one list of comparison stars: a star is used only if it is
    in every filter's RADEC file with a catalog magnitude, and fits on every image
    of every filter. See run_photometry for how stars that don't fit are handled.

    Parameters
    ----------
    path : str
        Path to the folder containing the reduced images.
    pipeline : bool
        Kept for compatibility. Photometry never prompts.
    radec_list : list
        RADEC file paths for B, V and R, in that order.
    obj_name : str
        Name of the target object, used for output file naming.
    write_callback : callable
        Function to route log messages to a GUI. Falls back to print if None.
    cancel_event : threading.Event
        Event flag that allows external cancellation of the running task.
    aperture_radius : float
        Aperture radius in pixels.
    annulus_radii : tuple of float
        Inner and outer background annulus radii in pixels.
    read_noise : float, optional
        Read noise in electrons for images without an RDNOISE keyword.
    """
    global _config_loaded_attempted, _config_log_printed, _loaded_config
    _config_loaded_attempted = False
    _config_log_printed = False
    _loaded_config = None

    log = make_logger(write_callback)

    try:
        if cancel_event and cancel_event.is_set():
            log("Multi-Aperture Photometry was canceled.")
            return

        images_path = Path(path)
        files = ccdp.ImageFileCollection(images_path)

        valid_radec = [r for r in radec_list if r and Path(r).exists()] if radec_list else []
        radec_dir = Path(valid_radec[0]).parent if valid_radec else None

        # Group LIGHT images by resolved filter symbol
        grouped_images = {}
        for header, file_name in files.headers(imagetyp='LIGHT', return_fname=True):
            sym = resolve_filter(header, radec_dir=radec_dir, log=log)
            if sym is not None:
                grouped_images.setdefault(sym, []).append(file_name)

        # radec_list is in B, V, R order
        jobs = []
        for idx, sym in enumerate(FILTER_ORDER):
            radec_file = radec_list[idx] if radec_list and idx < len(radec_list) else None
            if not radec_file:
                continue
            image_list = grouped_images.get(sym, [])
            if not image_list:
                log(f"No {sym} images found, so {sym} is left out of the photometry.")
                continue
            jobs.append((sym, image_list, radec_file))

        if not jobs:
            raise RuntimeError("No LIGHT images matched a filter that has a RADEC file (B, V or R).")

        return run_photometry(jobs, images_path, obj_name, write_callback=write_callback,
                              cancel_event=cancel_event, aperture_radius=aperture_radius,
                              annulus_radii=annulus_radii, read_noise=read_noise)

    except Exception as e:
        log(f"An error occurred in Multi-Aperture Photometry: {e}")
        raise


def _header_time(header, keys):
    """First of keys present in the header, or None."""
    for key in keys:
        if key in header:
            return header[key]
    return None


def read_noise_from_header(header, fallback):
    """Read noise from the image's RDNOISE keyword, or fallback when it's missing or not a number."""
    try:
        value = float(header["RDNOISE"])
    except (KeyError, TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


def footprint_problem(x, y, radius, image_data):
    """
    Check whether a circle of the given radius around pixel (x, y) can be measured
    on this image. Returns a short reason when it can't, or None when it can.

    A circle can't be measured when any part of it is off the edge of the image,
    or when it covers pixels with no data (NaN or infinite values, such as the
    padded edges of TESS cutouts or aligned frames).
    """
    height, width = image_data.shape
    if not (np.isfinite(x) and np.isfinite(y)):
        return "its position couldn't be worked out from the image WCS"

    # Pixel centres sit on whole numbers, so the image covers -0.5 to width - 0.5
    if (x - radius < -0.5 or x + radius > width - 0.5
            or y - radius < -0.5 or y + radius > height - 0.5):
        return "part of its annulus is off the edge of the image"

    # Any pixel the circle touches, including ones it only clips at a corner
    reach = radius + np.sqrt(0.5)
    x0, x1 = max(int(np.floor(x - reach)), 0), min(int(np.ceil(x + reach)) + 1, width)
    y0, y1 = max(int(np.floor(y - reach)), 0), min(int(np.ceil(y + reach)) + 1, height)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    touched = (xx - x) ** 2 + (yy - y) ** 2 <= reach ** 2
    if not np.all(np.isfinite(image_data[y0:y1, x0:x1][touched])):
        return "its annulus covers pixels with no data (NaN or infinite values)"

    return None


class _IntOffsetFormatter(ScalarFormatter):
    """Show the axis offset as a whole number, e.g. +2460000 rather than 1e6 notation."""

    def get_offset(self):
        if len(self.locs) == 0 or self.offset == 0:
            return ''
        return f"+{int(self.offset)}"


@dataclass
class ComparisonStar:
    """One comparison star, shared by every filter."""
    number: int  # position in the first filter's RADEC file, starting at 1 after the target
    ra: object   # as written in that file: hours, decimal or sexagesimal
    dec: object
    mags: dict = field(default_factory=dict)  # catalog magnitude for each filter symbol


class _StarsDontFit(Exception):
    """Raised as soon as one or more comparison stars don't fit on an image."""

    def __init__(self, filt, image_file, image_size, problems):
        super().__init__(f"{len(problems)} comparison star(s) don't fit on {image_file}")
        self.filt = filt
        self.image_file = image_file
        self.image_size = image_size  # (width, height)
        self.problems = problems      # list of (ComparisonStar, x, y, reason)


def _photometry_logger(write_callback):
    """
    Log function for photometry. In terminal mode messages go through tqdm so
    they appear above the progress bar instead of breaking it up.
    """
    def log(message, pbar=None):
        if write_callback:
            write_callback(message)
        elif pbar is not None:
            pbar.write(message)
        else:
            tqdm.write(message)
    return log


def _sky(ra, dec):
    return SkyCoord(ra, dec, unit=(u.h, u.deg), frame="icrs")


def _has_magnitude(mag):
    # AstroImageJ treats anything above 99 (normally 99.999) as "no magnitude"
    try:
        value = float(mag)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value < 99)


def _read_radec(radec_file):
    """
    Read a RADEC file. Row 0 is the target and every row after it is a
    comparison star. Columns are RA (hours), Dec (degrees), two flags, magnitude.

    :return: target RA, target Dec, then arrays of comparison RA, Dec and magnitude
    """
    df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)
    return df[0].values[0], df[1].values[0], df[0].values[1:], df[1].values[1:], df[4].values[1:]


def shared_comparison_stars(radec_files, log=print):
    """
    Work out which comparison stars every filter can use.

    Stars are matched between RADEC files by position rather than by row, so
    hand-edited files in a different order still line up. A star is kept only if
    it is in every file and has a catalog magnitude in every filter.

    :param radec_files: dict of RADEC file path for each filter symbol, in filter order
    :param log: function that takes one message string
    :return: (target RA and Dec for each filter, kept stars, list of (excluded star, reason))
    """
    filters = list(radec_files)
    parsed = {f: _read_radec(radec_files[f]) for f in filters}
    targets = {f: (parsed[f][0], parsed[f][1]) for f in filters}

    ref = filters[0]
    _, _, ref_ra, ref_dec, ref_mag = parsed[ref]
    if len(ref_ra) == 0:
        raise RuntimeError(f"The {ref} RADEC file has no comparison stars.")

    stars = [ComparisonStar(i + 1, ref_ra[i], ref_dec[i], {ref: ref_mag[i]}) for i in range(len(ref_ra))]
    ref_sky = _sky(ref_ra, ref_dec)
    reasons = {}

    for f in filters[1:]:
        _, _, ra, dec, mag = parsed[f]
        if len(ra) == 0:
            for s in stars:
                reasons.setdefault(s.number, f"isn't in the {f} RADEC file")
            continue
        idx, sep, _ = ref_sky.match_to_catalog_sky(_sky(ra, dec))
        matched = set()
        for s, j, arcsec in zip(stars, idx, sep.arcsec):
            if arcsec <= STAR_MATCH_ARCSEC:
                s.mags[f] = mag[j]
                matched.add(int(j))
            else:
                reasons.setdefault(s.number, f"isn't in the {f} RADEC file")
        extra = len(ra) - len(matched)
        if extra:
            log(f"{extra} star(s) in the {f} RADEC file aren't in the {ref} file, so they aren't used.")

    for s in stars:
        if s.number in reasons:
            continue
        missing = [f for f in filters if not _has_magnitude(s.mags.get(f))]
        if missing:
            reasons[s.number] = f"has no catalog magnitude in {', '.join(missing)}"

    target_ref = _sky(*targets[ref])
    for f in filters[1:]:
        arcsec = target_ref.separation(_sky(*targets[f])).arcsec
        if arcsec > STAR_MATCH_ARCSEC:
            log(f"WARNING: the target in the {f} RADEC file is {arcsec:.1f} arcsec from the target "
                f"in the {ref} file. Check that they are the same star.")

    kept = [s for s in stars if s.number not in reasons]
    excluded = [(s, reasons[s.number]) for s in stars if s.number in reasons]
    return targets, kept, excluded


def _measure_filter(filt, image_list, path, target, stars, aperture_radius, annulus_radii,
                    fallback_read_noise, log, use_tqdm, cancel_event):
    """
    Measure the target in every image of one filter against the given comparison stars.

    :return: dict with the HJD, BJD, magnitude and error lists and the skipped
        images, or None if the task was canceled
    :raises _StarsDontFit: as soon as any comparison star doesn't fit on an image
    """
    # The largest circle drawn around any star; this is what has to fit on the image
    outer_radius = max(float(aperture_radius), float(annulus_radii[1]))
    target_sky = _sky(*target)
    comp_sky = _sky([s.ra for s in stars], [s.dec for s in stars])
    magnitudes_comp = pd.Series([float(s.mags[filt]) for s in stars])

    result = {"hjd": [], "bjd": [], "magnitudes": [], "mag_err": [], "skipped": []}
    warned_missing_time = False

    image_iter = tqdm(image_list, desc=f"Processing {filt} images") if use_tqdm else image_list
    pbar = image_iter if use_tqdm else None
    try:
        for image_file in image_iter:
            if cancel_event and cancel_event.is_set():
                log("Task canceled during photometry.", pbar)
                return None

            log(f"Processing {image_file}", pbar)

            image_data, header = fits.getdata(path / image_file, header=True)
            img_height, img_width = image_data.shape
            read_noise = read_noise_from_header(header, fallback_read_noise)

            # Star positions come from this image's own WCS, so drift during the night is handled
            wcs_ = WCS(header)
            target_pixel = wcs_.world_to_pixel(target_sky)
            target_position = (float(target_pixel[0]), float(target_pixel[1]))
            comp_pixels = wcs_.world_to_pixel(comp_sky)
            comp_x = np.atleast_1d(np.array(comp_pixels[0], dtype=float))
            comp_y = np.atleast_1d(np.array(comp_pixels[1], dtype=float))

            # -----------------------------------------------------------------------
            # Every star's aperture and annulus must sit wholly on real image data,
            # otherwise its flux or background would come from a partial circle.
            # The target is checked first: if it doesn't fit, the image is skipped
            # and the comparison stars aren't judged on it.
            # -----------------------------------------------------------------------
            target_problem = footprint_problem(*target_position, outer_radius, image_data)
            if target_problem:
                log(f"[SKIPPED] {image_file}: the target can't be measured because {target_problem} "
                    f"(x={target_position[0]:.1f}, y={target_position[1]:.1f}, "
                    f"image {img_width}x{img_height}).", pbar)
                result["skipped"].append(image_file)
                continue

            problems = []
            for star, cx, cy in zip(stars, comp_x, comp_y):
                problem = footprint_problem(cx, cy, outer_radius, image_data)
                if problem:
                    problems.append((star, cx, cy, problem))
            if problems:
                raise _StarsDontFit(filt, image_file, (img_width, img_height), problems)

            hjd_value = _header_time(header, HJD_KEYS)
            bjd_value = _header_time(header, BJD_KEYS)
            if (hjd_value is None or bjd_value is None) and not warned_missing_time:
                log(f"WARNING: {image_file} has no HJD or BJD keyword "
                    f"(looked for {', '.join(HJD_KEYS + BJD_KEYS)}); writing NaN for missing times.", pbar)
                warned_missing_time = True
            result["hjd"].append(np.nan if hjd_value is None else hjd_value)
            result["bjd"].append(np.nan if bjd_value is None else bjd_value)

            # -----------------------------------------------------------------------
            # Target flux, then each comparison star's flux and background separately
            # -----------------------------------------------------------------------
            _, target_flx, target_flux_err, _ = calculate_target_snr(
                image_data, target_position, aperture_radius, annulus_radii, read_noise)

            comparison_aperture = [CircularAperture((x, y), r=aperture_radius) for x, y in zip(comp_x, comp_y)]
            comparison_annulus = [CircularAnnulus((x, y), *annulus_radii) for x, y in zip(comp_x, comp_y)]

            comparison_sums = [float(aperture_photometry(image_data, ap)['aperture_sum'][0])
                               for ap in comparison_aperture]

            comparison_bkg_mean = []
            for annulus in comparison_annulus:
                stats = ApertureStats(image_data, annulus)
                comparison_bkg_mean.append(0.0 if not np.isfinite(stats.mean) else stats.mean)

            # Background-subtracted flux for each comparison star
            comparison_flx = [total - float(bkg * ap.area)
                              for total, bkg, ap in zip(comparison_sums, comparison_bkg_mean, comparison_aperture)]

            # Poisson and read-noise flux error for each comparison star
            comp_flux_err = np.array([float(np.sqrt(total + ap.area * read_noise ** 2))
                                      for total, ap in zip(comparison_sums, comparison_aperture)])

            total_comp_flx = sum(comparison_flx)

            # -----------------------------------------------------------------------
            # Ensemble magnitude: the comparison stars' catalog magnitudes combined
            # into one reference magnitude, then the target's flux ratio applied
            # -----------------------------------------------------------------------
            target_magnitude = (
                (-np.log(sum(2.512 ** -magnitudes_comp)) / np.log(2.512)) -
                (2.5 * np.log10(target_flx / total_comp_flx))
            )
            target_magnitude_error = 2.5 * np.log10(1 + np.sqrt(
                (target_flux_err ** 2 / target_flx ** 2) +
                (np.sum(comp_flux_err ** 2) / total_comp_flx ** 2)
            ))

            result["magnitudes"].append(float(target_magnitude))
            result["mag_err"].append(float(target_magnitude_error))
    finally:
        if use_tqdm:
            image_iter.close()

    if not result["magnitudes"]:
        raise RuntimeError(f"The target couldn't be measured in any {filt} image.")
    return result


def _write_filter_outputs(path, obj_name, filt, result, log):
    """Save one filter's light curve as a plot and a CSV file."""
    filter_letter = filt.split("/")[-1]
    target_display_name = f"{obj_name} - {filter_letter}" if obj_name else f"Target - {filter_letter}"

    fig = Figure(figsize=(11, 8))
    ax = fig.subplots()
    ax.errorbar(result["hjd"], result["magnitudes"], yerr=result["mag_err"], fmt='o', label=target_display_name)

    fontsize = 14
    if obj_name:
        ax.set_title(target_display_name, fontsize=fontsize + 2)
    ax.set_xlabel('HJD', fontsize=fontsize)
    ax.set_ylabel(f'Magnitude ({filter_letter})', fontsize=fontsize)
    # Format HJD offset as an integer instead of scientific notation
    ax.xaxis.set_major_formatter(_IntOffsetFormatter(useOffset=True))
    ax.invert_yaxis()  # Magnitudes increase downward by convention
    ax.grid()
    ax.legend(loc="upper right", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig(path / f"{obj_name}_{filter_letter}_figure.jpg")

    light_curve_data = pd.DataFrame({
        'HJD': result["hjd"],
        'BJD': result["bjd"],
        'Source_AMag_T1': result["magnitudes"],
        'Source_AMag_T1_Error': result["mag_err"],
    })
    output_file = path / f"{obj_name}_{filter_letter}_data.csv"
    light_curve_data.to_csv(output_file, index=False)
    log(f"Saved light curve data for {filt} to {output_file}")


def _write_comparison_record(path, obj_name, filters, used, excluded, removed, log):
    """
    Save which comparison stars were used, and why any others weren't, so the
    exact ensemble behind the light curves can be checked later.
    """
    rows = []
    notes = {s.number: reason for s, reason in excluded}
    for star, filt, image_file, reason in removed:
        notes[star.number] = f"removed: {reason} in {filt} image {image_file}"
    used_numbers = {s.number for s in used}
    everyone = {s.number: s for s in used}
    everyone.update({s.number: s for s, _ in excluded})
    everyone.update({star.number: star for star, *_ in removed})
    for number in sorted(everyone):
        star = everyone[number]
        row = {"Star": number, "RA": star.ra, "Dec": star.dec, "Used": number in used_numbers}
        for f in filters:
            mag = star.mags.get(f)
            row[f"{f}_mag"] = float(mag) if _has_magnitude(mag) else np.nan
        row["Note"] = notes.get(number, "")
        rows.append(row)

    output_file = path / f"{obj_name}_comparison_stars.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    log(f"Saved the comparison star list to {output_file}")


def run_photometry(jobs, path, obj_name="", write_callback=None, cancel_event=None,
                   aperture_radius=20, annulus_radii=(30, 50), read_noise=None):
    """
    Aperture photometry for one or more filters using one shared set of comparison stars.

    If any comparison star's aperture or annulus runs off the edge of an image, or
    over pixels with no data, in any filter, every such star in that image is
    dropped from all filters and the whole analysis starts over from the first
    image of the first filter. This repeats until one set of comparison stars fits
    on every image of every filter, so every magnitude in every light curve comes
    from the same ensemble. Nothing is written until that point.

    Images where the target's own annulus doesn't fit are skipped, and the
    comparison stars aren't judged on them.

    :param jobs: list of (filter symbol, image file names, RADEC file), in filter order
    :param path: folder containing the images; outputs are written here too
    :return: the comparison stars used, or None if canceled
    """
    path = Path(path)
    log = _photometry_logger(write_callback)
    fallback_read_noise = DEFAULT_READ_NOISE if read_noise is None else float(read_noise)
    use_tqdm = not write_callback
    filters = [filt for filt, _, _ in jobs]

    targets, stars, excluded = shared_comparison_stars({filt: radec for filt, _, radec in jobs}, log)
    for star, reason in excluded:
        log(f"[EXCLUDED] Comparison star #{star.number} (RA={star.ra}, Dec={star.dec}) {reason}, "
            f"so it isn't used in any filter.")

    removed = []
    pass_number = 0
    while True:
        pass_number += 1
        if not stars:
            raise RuntimeError("All comparison stars have been removed. Cannot continue photometry.")
        if pass_number > 1:
            log(f"Starting photometry over from the first {filters[0]} image with "
                f"{len(stars)} comparison star(s) (pass {pass_number}).")

        results = {}
        try:
            for filt, image_list, _ in jobs:
                log(f"Processing {len(image_list)} images for {filt} filter.")
                result = _measure_filter(filt, image_list, path, targets[filt], stars, aperture_radius,
                                         annulus_radii, fallback_read_noise, log, use_tqdm, cancel_event)
                if result is None:
                    return None
                results[filt] = result
        except _StarsDontFit as misfit:
            width, height = misfit.image_size
            outer_radius = max(float(aperture_radius), float(annulus_radii[1]))
            for star, x, y, reason in misfit.problems:
                removed.append((star, misfit.filt, misfit.image_file, reason))
                log(f"[REMOVED] Comparison star #{star.number} (RA={star.ra}, Dec={star.dec}) "
                    f"in {misfit.filt} image '{misfit.image_file}': {reason}. "
                    f"Pixel position x={x:.1f}, y={y:.1f}; image {width}x{height}; "
                    f"outer radius {outer_radius:g}px. Dropped from every filter.")
            dropped = {star.number for star, *_ in misfit.problems}
            stars = [s for s in stars if s.number not in dropped]
            continue
        break

    for filt in filters:
        _write_filter_outputs(path, obj_name, filt, results[filt], log)

    used = ", ".join(f"#{s.number}" for s in stars)
    log(f"Used the same {len(stars)} comparison star(s) in {', '.join(filters)}: {used}.")
    if removed:
        log(f"Removed {len(removed)} comparison star(s) that didn't fit on every image "
            f"and started over {pass_number - 1} time(s).")
    for filt in filters:
        skipped = results[filt]["skipped"]
        if skipped:
            log(f"{filt}: skipped {len(skipped)} image(s) where the target didn't fit: {', '.join(skipped)}")
    _write_comparison_record(path, obj_name, filters, stars, excluded, removed, log)
    return stars


def multiple_AP(image_list, path, filt, pipeline=False, obj_name="", radec_file="",
                write_callback=None, cancel_event=None, aperture_radius=20, annulus_radii=(30, 50),
                read_noise=None):
    """
    Aperture photometry for a single filter on its own.

    Works the same way as run_photometry with one filter. To photometer several
    filters with the same comparison stars, use main or run_photometry instead.

    Parameters
    ----------
    image_list : list
        List of image filenames to process.
    path : Path
        Path to the folder containing the images.
    filt : str
        Filter label (e.g. 'Empty/B').
    pipeline : bool
        Kept for compatibility. Photometry never prompts.
    obj_name : str
        Target object name used for output file naming.
    radec_file : str
        Path to the RADEC file for this filter.
    write_callback : callable
        Function to route log messages to a GUI.
    cancel_event : threading.Event
        Event flag that allows external cancellation of the running task.
    """
    log = _photometry_logger(write_callback)
    if cancel_event and cancel_event.is_set():
        log("Task canceled during photometry.")
        return None

    try:
        return run_photometry([(filt, image_list, radec_file)], path, obj_name,
                              write_callback=write_callback, cancel_event=cancel_event,
                              aperture_radius=aperture_radius, annulus_radii=annulus_radii,
                              read_noise=read_noise)
    except Exception as e:
        # Raised as well as logged so the GUI and pipeline don't report success
        log(f"Error during photometry for {filt}: {e}")
        raise


def im_plot(image_data, target_aperture, comparison_apertures, target_annulus, comparison_annuli):
    """
    Diagnostic plot of a single image with all apertures and annuli overlaid.
    Useful for visually verifying that aperture positions are correct before
    running the full photometry pipeline.

    Parameters
    ----------
    image_data : array
        Pixel data from the FITS image.
    target_aperture : CircularAperture
        Aperture centred on the target star.
    comparison_apertures : list of CircularAperture
        Apertures centred on each comparison star.
    target_annulus : CircularAnnulus
        Background annulus for the target star.
    comparison_annuli : list of CircularAnnulus
        Background annuli for each comparison star.
    """
    # Interactive window, so this one does go through pyplot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))

    # Display the image with a percentile stretch to avoid saturation dominating the scale
    plt.imshow(
        image_data,
        cmap='gray',
        origin='lower',
        vmin=np.percentile(image_data, 5),
        vmax=np.percentile(image_data, 95)
    )
    plt.colorbar(label='Counts')

    lw = 1.5  # Line width for aperture outlines
    alpha = 1.0  # Opacity for aperture outlines

    # Target star aperture and annulus in green
    target_aperture.plot(color='darkgreen', lw=lw, alpha=alpha)
    target_annulus.plot(color='darkgreen', lw=lw, alpha=alpha)

    # Comparison star apertures and annuli in red
    for comp_ap, comp_an in zip(comparison_apertures, comparison_annuli):
        comp_ap.plot(color='red', lw=lw, alpha=alpha)
        comp_an.plot(color='red', lw=lw, alpha=alpha)

    plt.pause(1)
    plt.show()


if __name__ == '__main__':
    main()
