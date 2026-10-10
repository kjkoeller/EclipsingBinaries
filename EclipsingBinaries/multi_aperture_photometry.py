"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 04/27/2026
"""

# Python imports
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
    Entry point for multi-aperture photometry. Iterates over each filter and its
    corresponding RADEC file, dispatching to multiple_AP for the actual photometry.

    Parameters
    ----------
    path : str
        Path to the folder containing the reduced images.
    pipeline : bool
        If True, suppresses prompts and expects all inputs to be provided programmatically.
    radec_list : list
        List of RADEC file paths, one per filter in the same order as filt_list.
    obj_name : str
        Name of the target object, used for output file naming.
    write_callback : callable
        Function to route log messages to a GUI. Falls back to print if None.
    cancel_event : threading.Event
        Event flag that allows external cancellation of the running task.
    """
    global _config_loaded_attempted, _config_log_printed, _loaded_config
    _config_loaded_attempted = False
    _config_log_printed = False
    _loaded_config = None

    log = make_logger(write_callback)

    try:
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

        # To maintain compatibility with existing radec_list order (B, V, R)
        expected_symbols = ["B", "V", "R"]

        for idx, sym in enumerate(expected_symbols):
            if cancel_event and cancel_event.is_set():
                log("Multi-Aperture Photometry was canceled.")
                return

            if radec_list and idx < len(radec_list):
                radec_file = radec_list[idx]
            else:
                continue

            image_list = grouped_images.get(sym, [])
            if not image_list:
                continue

            log(f"Processing {len(image_list)} images for {sym} filter.")

            multiple_AP(
                image_list=image_list,
                path=images_path,
                filt=sym,
                pipeline=pipeline,
                obj_name=obj_name,
                radec_file=radec_file,
                write_callback=write_callback,
                cancel_event=cancel_event,
                aperture_radius=aperture_radius,
                annulus_radii=annulus_radii,
                read_noise=read_noise,
            )

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


def multiple_AP(image_list, path, filt, pipeline=False, obj_name="", radec_file="",
                write_callback=None, cancel_event=None, aperture_radius=20, annulus_radii=(30, 50),
                read_noise=None):
    """
    Perform aperture photometry on all images for a single filter.

    If any comparison star's aperture or annulus runs off the edge of an image, or
    over pixels with no data, every such star in that image is dropped and the
    whole filter starts over from the first image with the smaller set. This
    repeats until one set of comparison stars works in every image, so all the
    magnitudes in the output come from the same ensemble.

    Images where the target's own annulus doesn't fit are skipped, since there is
    nothing to measure, and the comparison stars are left alone for them.

    Parameters
    ----------
    image_list : list
        List of image filenames to process.
    path : Path
        Path to the folder containing the images.
    filt : str
        Filter label (e.g. 'Empty/B').
    pipeline : bool
        If True, suppresses user prompts.
    obj_name : str
        Target object name used for output file naming.
    radec_file : str
        Path to the RADEC file for the current filter.
    write_callback : callable
        Function to route log messages to a GUI.
    cancel_event : threading.Event
        Event flag that allows external cancellation of the running task.
    """

    def log(message, pbar=None):
        """
        Route a log message appropriately depending on context.
        In terminal mode the message is written through tqdm so it appears
        above the progress bar rather than interleaving with it.
        In GUI mode the write_callback is used directly and tqdm is bypassed.
        """
        if write_callback:
            write_callback(message)
        elif pbar is not None:
            pbar.write(message)
        else:
            tqdm.write(message)

    if cancel_event and cancel_event.is_set():
        log("Task canceled during photometry.")
        return

    # Read noise for the flux errors. Each image's RDNOISE keyword wins when it
    # has one; this value is only the fallback. It used to be fixed at the
    # BSUO value even for KPNO, CTIO and La Palma data.
    fallback_read_noise = DEFAULT_READ_NOISE if read_noise is None else float(read_noise)
    warned_missing_time = False

    try:
        # ---------------------------------------------------------------------------
        # Load the RADEC file
        # Row 0 is the target star; all subsequent rows are comparison stars.
        # Columns: RA (h), Dec (deg), <unused>, <unused>, catalog magnitude
        # ---------------------------------------------------------------------------
        df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)

        all_ra = df[0].values
        all_dec = df[1].values
        all_mag = df[4].values

        # Separate the target star from the comparison stars
        target_ra = all_ra[0]
        target_dec = all_dec[0]
        comp_ra_all = all_ra[1:]
        comp_dec_all = all_dec[1:]
        comp_mag_all = all_mag[1:]

        # ---------------------------------------------------------------------------
        # Build the initial set of valid comparison star indices.
        # Stars with a catalog magnitude of 99.999 (no valid magnitude) are excluded
        # immediately so the magnitude and flux arrays stay in sync throughout all
        # subsequent processing and potential restarts.
        # ---------------------------------------------------------------------------
        valid_mag_mask = np.array([(m != 99.999 and not pd.isna(m)) for m in comp_mag_all])
        valid_comp_indices = list(np.where(valid_mag_mask)[0])

        # Inform the user of any stars excluded due to missing catalog magnitudes
        for idx in np.where(~valid_mag_mask)[0]:
            log(
                f"[EXCLUDED] Comparison star #{idx + 1} "
                f"(RA={comp_ra_all[idx]}, Dec={comp_dec_all[idx]}) "
                f"removed before processing: magnitude is 99.999 (no valid catalog magnitude)."
            )

        # ---------------------------------------------------------------------------
        # Main processing loop. If a comparison star is found to be outside the
        # bounds of any image, it is removed and the loop restarts from image 1
        # with the reduced set. This continues until all remaining comparison stars
        # are within bounds for every image in the list.
        # ---------------------------------------------------------------------------
        # The largest circle drawn around any star; this is what has to fit on the image
        outer_radius = max(float(aperture_radius), float(annulus_radii[1]))
        removed_stars = []
        pass_number = 0

        restart = True
        while restart:
            restart = False
            pass_number += 1

            if not valid_comp_indices:
                raise RuntimeError("All comparison stars have been removed. Cannot continue photometry.")

            if pass_number > 1:
                log(f"Starting {filt} photometry over from the first image with "
                    f"{len(valid_comp_indices)} comparison star(s) (pass {pass_number}).")

            skipped_frames = []

            # Slice coordinate and magnitude arrays down to currently valid stars
            cur_comp_ra = comp_ra_all[valid_comp_indices]
            cur_comp_dec = comp_dec_all[valid_comp_indices]
            cur_comp_mag = comp_mag_all[valid_comp_indices]

            # Build a clean float Series of catalog magnitudes for the active comp stars
            magnitudes_comp = pd.Series(cur_comp_mag.astype(float)).reset_index(drop=True)

            # Accumulators for per-image results
            magnitudes = []
            mag_err = []
            hjd = []
            bjd = []

            # Use tqdm in terminal mode only; the GUI receives progress via write_callback
            use_tqdm = not write_callback
            image_iter = (
                tqdm(image_list, desc=f"Processing {filt} images") if use_tqdm else image_list
            )

            for icount, image_file in enumerate(image_iter):
                pbar = image_iter if use_tqdm else None

                # Check for external cancellation at the start of each image
                if cancel_event and cancel_event.is_set():
                    log("Task canceled during photometry.", pbar)
                    if use_tqdm:
                        image_iter.close()
                    return

                log(f"Processing {image_file}", pbar)

                # Load the image data and header
                image_data, header = fits.getdata(path / image_file, header=True)
                img_height, img_width = image_data.shape
                read_noise = read_noise_from_header(header, fallback_read_noise)

                # Build the WCS from the image header for sky-to-pixel conversion
                wcs_ = WCS(header)

                # Convert target RA/Dec to pixel coordinates
                target_sky = SkyCoord(target_ra, target_dec, unit=(u.h, u.deg), frame='icrs')
                target_pixel = wcs_.world_to_pixel(target_sky)
                target_position = (float(target_pixel[0]), float(target_pixel[1]))

                # Convert all active comparison star RA/Dec values to pixel coordinates
                comp_sky = SkyCoord(cur_comp_ra, cur_comp_dec, unit=(u.h, u.deg), frame='icrs')
                comp_pixels = wcs_.world_to_pixel(comp_sky)
                comp_x = np.array(comp_pixels[0], dtype=float)
                comp_y = np.array(comp_pixels[1], dtype=float)

                # ---------------------------------------------------------------------------
                # Every star's aperture and annulus must sit wholly on real image data,
                # otherwise its flux or background would come from a partial circle.
                # The target is checked first: if it doesn't fit, the image is skipped
                # and the comparison stars aren't judged on it.
                # ---------------------------------------------------------------------------
                target_problem = footprint_problem(*target_position, outer_radius, image_data)
                if target_problem:
                    log(f"[SKIPPED] {image_file}: the target can't be measured because {target_problem} "
                        f"(x={target_position[0]:.1f}, y={target_position[1]:.1f}, "
                        f"image {img_width}x{img_height}).", pbar)
                    skipped_frames.append(image_file)
                    continue

                bad_stars = []
                for i, (cx, cy) in enumerate(zip(comp_x, comp_y)):
                    problem = footprint_problem(cx, cy, outer_radius, image_data)
                    if problem:
                        bad_stars.append((i, problem))

                if bad_stars:
                    # Drop every star that failed in this image at once, then start
                    # the whole filter over so all images use the same ensemble
                    for i, problem in bad_stars:
                        orig_idx = valid_comp_indices[i]
                        removed_stars.append((orig_idx + 1, image_file, problem))
                        log(
                            f"[REMOVED] Comparison star #{orig_idx + 1} "
                            f"(RA={cur_comp_ra[i]}, Dec={cur_comp_dec[i]}) in '{image_file}': {problem}. "
                            f"Pixel position x={comp_x[i]:.1f}, y={comp_y[i]:.1f}; "
                            f"image {img_width}x{img_height}; outer radius {outer_radius:g}px.",
                            pbar
                        )
                    drop = {i for i, _ in bad_stars}
                    valid_comp_indices = [idx for k, idx in enumerate(valid_comp_indices) if k not in drop]
                    if use_tqdm:
                        image_iter.close()
                    restart = True
                    break  # Leave the image loop and start the filter over

                # Build the list of (x, y) positions for all active comparison stars
                comparison_positions = list(zip(comp_x, comp_y))

                # Record the timestamps from the header for this image
                hjd_value = _header_time(header, HJD_KEYS)
                bjd_value = _header_time(header, BJD_KEYS)
                if (hjd_value is None or bjd_value is None) and not warned_missing_time:
                    log(f"WARNING: {image_file} has no HJD or BJD keyword "
                        f"(looked for {', '.join(HJD_KEYS + BJD_KEYS)}); writing NaN for missing times.", pbar)
                    warned_missing_time = True
                hjd.append(np.nan if hjd_value is None else hjd_value)
                bjd.append(np.nan if bjd_value is None else bjd_value)

                # ---------------------------------------------------------------------------
                # Define apertures and annuli for each comparison star
                # ---------------------------------------------------------------------------

                comparison_aperture = [
                    CircularAperture(pos, r=aperture_radius) for pos in comparison_positions
                ]
                comparison_annulus = [
                    CircularAnnulus(pos, *annulus_radii) for pos in comparison_positions
                ]

                # ---------------------------------------------------------------------------
                # Perform aperture photometry for the target star
                # ---------------------------------------------------------------------------
                _, target_flx, target_flux_err, _ = calculate_target_snr(
                    image_data, target_position, aperture_radius, annulus_radii, read_noise)

                # ---------------------------------------------------------------------------
                # Perform aperture photometry for each comparison star individually so
                # background and flux can be computed per-star
                # ---------------------------------------------------------------------------
                comparison_phot_table = []
                for comp_ap, comp_an in zip(comparison_aperture, comparison_annulus):
                    comparison_phot_table.append((
                        aperture_photometry(image_data, comp_ap),
                        aperture_photometry(image_data, comp_an)
                    ))

                # Estimate sky background for each comparison star in the same way
                comparison_bkg_mean = []
                for annulus in comparison_annulus:
                    stats = ApertureStats(image_data, annulus)
                    if np.isnan(stats.mean) or np.isinf(stats.mean):
                        comparison_bkg_mean.append(0.0)
                    else:
                        comparison_bkg_mean.append(stats.mean)

                # ---------------------------------------------------------------------------
                # Extract aperture sums to plain Python floats immediately after photometry.
                # astropy Table columns can carry shape metadata that causes ambiguous
                # behaviour in downstream arithmetic if left as Column objects.
                # ---------------------------------------------------------------------------

                # Background-subtracted flux for each comparison star
                comparison_flx = [
                    float(phot[0]['aperture_sum'][0]) - float(bkg_mean * aperture.area)
                    for phot, bkg_mean, aperture in zip(
                        comparison_phot_table, comparison_bkg_mean, comparison_aperture
                    )
                ]

                # Poisson + read-noise flux error for each comparison star as a clean 1D float array
                comp_flux_err = np.array([
                    float(np.sqrt(float(phot[0]['aperture_sum'][0]) + aperture.area * read_noise ** 2))
                    for phot, aperture in zip(comparison_phot_table, comparison_aperture)
                ])

                # Total flux across all active comparison stars, used in several calculations below
                total_comp_flx = sum(comparison_flx)

                # ---------------------------------------------------------------------------
                # Relative flux for each comparison star: its own flux divided by the sum
                # of all other comparison stars' fluxes (i.e. excluding itself).
                # This gives a check-star light curve for each comparison star.
                # (currently not saved but kept as reference for future check-star analysis)
                # ---------------------------------------------------------------------------
                # rel_flux_comps = [flx / (total_comp_flx - flx) for flx in comparison_flx]

                # ---------------------------------------------------------------------------
                # Target magnitude using the ensemble comparison method.
                # The first term converts the catalog magnitudes of the comparison stars
                # into a reference magnitude for the ensemble. The second term applies
                # the differential flux ratio between target and ensemble.
                # ---------------------------------------------------------------------------
                target_magnitude = (
                    (-np.log(sum(2.512 ** -magnitudes_comp)) / np.log(2.512)) -
                    (2.5 * np.log10(target_flx / total_comp_flx))
                )

                # Propagated magnitude error from flux uncertainties of target and ensemble
                target_magnitude_error = 2.5 * np.log10(1 + np.sqrt(
                    (target_flux_err ** 2 / target_flx ** 2) +
                    (np.sum(comp_flux_err ** 2) / total_comp_flx ** 2)
                ))

                magnitudes.append(float(target_magnitude))
                mag_err.append(float(target_magnitude_error))

        if not magnitudes:
            raise RuntimeError(f"The target couldn't be measured in any {filt} image.")

        used = ", ".join(f"#{idx + 1}" for idx in valid_comp_indices)
        log(f"{filt}: measured {len(magnitudes)} image(s) with {len(valid_comp_indices)} "
            f"comparison star(s) ({used}).")
        if removed_stars:
            log(f"{filt}: removed {len(removed_stars)} comparison star(s) that didn't fit on every image "
                f"and started over {pass_number - 1} time(s).")
        if skipped_frames:
            log(f"{filt}: skipped {len(skipped_frames)} image(s) where the target didn't fit: "
                + ", ".join(skipped_frames))

        # ---------------------------------------------------------------------------
        # Plot the resulting light curve with error bars and save it to disk
        # ---------------------------------------------------------------------------
        fig = Figure(figsize=(11, 8))
        ax = fig.subplots()
        filter_letter = filt.split("/")[-1]
        target_display_name = f"{obj_name} - {filter_letter}" if obj_name else f"Target - {filter_letter}"

        ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o', label=target_display_name)

        fontsize = 14
        if obj_name:
            ax.set_title(target_display_name, fontsize=fontsize+2)

        ax.set_xlabel('HJD', fontsize=fontsize)
        ax.set_ylabel(f'Magnitude ({filter_letter})', fontsize=fontsize)

        # Format HJD offset as an integer instead of scientific notation
        ax.xaxis.set_major_formatter(_IntOffsetFormatter(useOffset=True))
        ax.invert_yaxis()  # Magnitudes increase downward by convention
        ax.grid()
        ax.legend(loc="upper right", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        fig.savefig(path / f"{obj_name}_{filter_letter}_figure.jpg")

        # ---------------------------------------------------------------------------
        # Save the light curve data to a CSV file
        # ---------------------------------------------------------------------------
        light_curve_data = pd.DataFrame({
            'HJD': hjd,
            'BJD': bjd,
            'Source_AMag_T1': magnitudes,
            'Source_AMag_T1_Error': mag_err
        })

        output_file = path / f"{obj_name}_{filter_letter}_data.csv"
        light_curve_data.to_csv(output_file, index=False)
        log(f"Saved light curve data for {filt} to {output_file}")

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
