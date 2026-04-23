"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 04/23/2026

This program is meant to automatically do the data reduction of the raw images from the
Ball State University Observatory (BSUO) and SARA data. The new calibrated images are placed into a new folder as to
not overwrite the original images.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

from astropy import wcs
from astropy.stats import mad_std
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

import ccdproc as ccdp
import numpy as np

# Suppress FITS standard-compliance header warnings
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReductionConfig:
    """
    All tunable parameters for a reduction run in one place.
    Pass a ReductionConfig instance into run_reduction() instead of relying
    on module-level globals.
    """
    gain: float = 1.43                          # e-/ADU  (BSUO default)
    rdnoise: float = 10.83                      # e-      (BSUO default)
    sigclip: int = 5                            # sigma for cosmic-ray removal
    sigma_clip_low_thresh: Optional[int] = None # lower sigma for combine
    sigma_clip_high_thresh: int = 3             # upper sigma for combine
    mem_limit: float = 1600e6                   # bytes (~1.6 GB)
    dark_bool: bool = True                      # whether dark frames exist
    location: str = "bsuo"                      # observing site key
    overwrite: bool = True                      # overwrite existing output files
    overscan_region: str = "[2073:2115, :]"     # FITS section string; set to "none" to skip
    trim_region: str = "[20:2060, 12:2057]"     # FITS section string


def bsuo_config() -> ReductionConfig:
    """Ball State University Observatory defaults (package default)."""
    return ReductionConfig()


def kpno_config() -> ReductionConfig:
    """Kitt Peak National Observatory defaults."""
    return ReductionConfig(gain=2.3, rdnoise=6.0, dark_bool=True, location="kpno")


def ctio_config() -> ReductionConfig:
    """Cerro Tololo Inter-American Observatory defaults."""
    return ReductionConfig(gain=2.0, rdnoise=9.7, dark_bool=True, location="ctio")


def lapalma_config() -> ReductionConfig:
    """La Palma defaults."""
    return ReductionConfig(gain=1.0, rdnoise=6.3, dark_bool=True, location="lapalma")


# ---------------------------------------------------------------------------
# I/O helper — suppresses mask and uncertainty extensions
# ---------------------------------------------------------------------------

def write_image_only(ccd, path, overwrite=True):
    """
    Write a CCDData object to disk as a plain single-extension FITS file.

    ccdproc normally writes mask and uncertainty arrays as additional FITS
    extensions. This helper temporarily clears them on the object, writes,
    then restores them — avoiding an expensive deep copy while keeping the
    live in-memory object intact for downstream operations.

    :param ccd: CCDData object to write
    :param path: Destination path (str or Path)
    :param overwrite: Whether to overwrite an existing file
    """
    mask, uncertainty = ccd.mask, ccd.uncertainty
    try:
        ccd.mask = None
        ccd.uncertainty = None
        ccd.write(str(path), overwrite=overwrite)
    finally:
        ccd.mask = mask
        ccd.uncertainty = uncertainty


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_reduction(
    path,
    calibrated,
    cfg: ReductionConfig = None,
    cancel_event=None,
    write_callback=None,
):
    """
    Run the full CCD reduction pipeline.

    :param path: Path to the directory containing raw images
    :param calibrated: Path to the output directory for calibrated images
    :param cfg: ReductionConfig instance; defaults to bsuo_config() if None
    :param cancel_event: threading.Event; checked before each major step
    :param write_callback: Optional callable(str) for GUI log output
    """
    if cfg is None:
        cfg = bsuo_config()

    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    def canceled():
        return cancel_event is not None and cancel_event.is_set()

    if canceled():
        log("Task canceled before starting.")
        return

    images_path = Path(path)
    calibrated_data = Path(calibrated)

    if not images_path.exists():
        raise FileNotFoundError(f"Raw images path '{path}' does not exist.")
    calibrated_data.mkdir(parents=True, exist_ok=True)

    files = ccdp.ImageFileCollection(images_path)

    # --- Bias ---
    try:
        zero = bias(files, calibrated_data, cfg, log, cancel_event)
    except Exception as e:
        raise RuntimeError("Bias stage failed.") from e
    if zero is None:
        log("Reduction aborted: bias stage did not complete.")
        return

    # --- Dark (optional) ---
    master_dark = None
    if cfg.dark_bool:
        try:
            master_dark = dark(files, zero, calibrated_data, cfg, log, cancel_event)
        except Exception as e:
            raise RuntimeError("Dark stage failed.") from e
        if master_dark is None:
            log("Reduction aborted: dark stage did not complete.")
            return

    # --- Flat ---
    try:
        flat(files, zero, master_dark, calibrated_data, cfg, log, cancel_event)
    except Exception as e:
        raise RuntimeError("Flat stage failed.") from e
    if canceled():
        log("Reduction aborted: flat stage did not complete.")
        return

    # --- Science ---
    try:
        science_images(files, calibrated_data, zero, master_dark, cfg, log, cancel_event)
    except Exception as e:
        raise RuntimeError("Science stage failed.") from e
    if canceled():
        log("Reduction aborted: science stage did not complete.")
        return

    log("\nReduction process completed successfully.\n")


# ---------------------------------------------------------------------------
# Preprocessing helper — shared by all four stage reducers
# ---------------------------------------------------------------------------

def _preprocess(ccd, cfg: ReductionConfig):
    """
    Apply overscan subtraction (if enabled), trimming, and gain correction.
    This is the common first step for every frame type.

    :param ccd: Input CCDData image
    :param cfg: ReductionConfig
    :return: Preprocessed CCDData
    """
    if cfg.overscan_region.lower() != "none":
        ccd = ccdp.subtract_overscan(
            ccd, fits_section=cfg.overscan_region, median=True, overscan_axis=None
        )
    ccd = ccdp.trim_image(ccd, fits_section=cfg.trim_region)
    ccd = ccdp.gain_correct(ccd, gain=cfg.gain * u.electron / u.adu)
    return ccd


# ---------------------------------------------------------------------------
# Stage-specific reducers — each takes only the calibration frames it needs
# ---------------------------------------------------------------------------

def _reduce_bias(ccd, cfg: ReductionConfig):
    """
    Preprocess a single bias frame (overscan, trim, gain).

    :param ccd: Raw bias CCDData
    :param cfg: ReductionConfig
    :return: Preprocessed CCDData
    """
    return _preprocess(ccd, cfg)


def _reduce_dark(ccd, cfg: ReductionConfig, zero):
    """
    Preprocess and bias-subtract a single dark frame.

    :param ccd: Raw dark CCDData
    :param cfg: ReductionConfig
    :param zero: Master bias CCDData
    :return: Bias-subtracted CCDData
    """
    ccd = _preprocess(ccd, cfg)
    return ccdp.subtract_bias(ccd, zero)


def _reduce_flat(ccd, cfg: ReductionConfig, zero, combined_dark):
    """
    Preprocess, bias-subtract, and optionally dark-subtract a single flat frame.

    :param ccd: Raw flat CCDData
    :param cfg: ReductionConfig
    :param zero: Master bias CCDData
    :param combined_dark: Master dark CCDData (ignored if cfg.dark_bool is False)
    :return: Calibrated CCDData
    """
    ccd = _preprocess(ccd, cfg)
    ccd = ccdp.subtract_bias(ccd, zero)
    if cfg.dark_bool:
        ccd = ccdp.subtract_dark(
            ccd, combined_dark, exposure_time="exptime", exposure_unit=u.second, scale=True
        )
    return ccd


def _reduce_science(ccd, cfg: ReductionConfig, zero, combined_dark, good_flat):
    """
    Fully calibrate a single science frame: preprocess, bias, dark, flat-field.

    :param ccd: Raw science CCDData
    :param cfg: ReductionConfig
    :param zero: Master bias CCDData
    :param combined_dark: Master dark CCDData (ignored if cfg.dark_bool is False)
    :param good_flat: Master flat CCDData matched to this frame's filter
    :return: Fully calibrated CCDData
    """
    ccd = _preprocess(ccd, cfg)
    ccd = ccdp.subtract_bias(ccd, zero)
    if cfg.dark_bool:
        ccd = ccdp.subtract_dark(
            ccd, combined_dark, exposure_time="exptime", exposure_unit=u.second, scale=True
        )
    ccd = ccdp.flat_correct(ccd=ccd, flat=good_flat, min_value=1.0)
    return ccd


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def bias(files, calibrated_data, cfg: ReductionConfig, log, cancel_event):
    """
    Overscan-correct, trim, and gain-correct each bias frame, then combine
    them into a master bias.

    Tracks output paths internally to avoid an extra ImageFileCollection
    directory scan before combining.

    :return: Combined master bias CCDData, or None if canceled
    """
    log("\nStarting bias calibration.")
    log(f"Overscan Region: {cfg.overscan_region}")
    log(f"Trim Region:     {cfg.trim_region}")

    # Count frames upfront for progress reporting
    bias_files = files.files_filtered(imagetyp="BIAS")
    n_total = len(bias_files)
    log(f"Found {n_total} bias frame(s).")

    calibrated_bias_paths = []
    for n_done, (ccd, file_name) in enumerate(
        files.ccds(imagetyp="BIAS", return_fname=True, ccd_kwargs={"unit": "adu"}), start=1
    ):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        log(f"Processing bias {n_done}/{n_total}: {file_name}")
        new_ccd = _reduce_bias(ccd, cfg)
        output_path = calibrated_data / f"{file_name.split('.')[0]}.fits"
        write_image_only(new_ccd, output_path, overwrite=cfg.overwrite)
        calibrated_bias_paths.append(str(output_path))

    log(f"\nCombining {n_total} bias frame(s) into master bias.")
    combined_bias = ccdp.combine(
        calibrated_bias_paths,
        method="average",
        sigma_clip=True,
        sigma_clip_low_thresh=cfg.sigma_clip_low_thresh,
        sigma_clip_high_thresh=cfg.sigma_clip_high_thresh,
        sigma_clip_func=np.ma.median,
        mem_limit=cfg.mem_limit,
    )
    combined_bias.meta["combined"] = True
    combined_bias_path = calibrated_data / "zero.fits"
    write_image_only(combined_bias, combined_bias_path, overwrite=cfg.overwrite)
    log(f"Master bias created: {combined_bias_path}")

    return combined_bias


def dark(files, zero, calibrated_path, cfg: ReductionConfig, log, cancel_event):
    """
    Bias-subtract each dark frame, then combine them into a master dark.

    Tracks output paths internally to avoid an extra ImageFileCollection
    directory scan before combining.

    :return: Combined master dark CCDData, or None if canceled
    """
    log("\nStarting dark calibration.")

    dark_files = files.files_filtered(imagetyp="DARK")
    n_total = len(dark_files)
    log(f"Found {n_total} dark frame(s).")

    calibrated_dark_paths = []
    for n_done, (ccd, file_name) in enumerate(
        files.ccds(imagetyp="DARK", return_fname=True, ccd_kwargs={"unit": "adu"}), start=1
    ):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        log(f"Processing dark {n_done}/{n_total}: {file_name}")
        sub_ccd = _reduce_dark(ccd, cfg, zero)
        output_path = calibrated_path / f"{file_name.split('.')[0]}.fits"
        write_image_only(sub_ccd, output_path, overwrite=cfg.overwrite)
        calibrated_dark_paths.append(str(output_path))

    log(f"\nCombining {n_total} dark frame(s) into master dark.")
    combined_dark = ccdp.combine(
        calibrated_dark_paths,
        method="average",
        sigma_clip=True,
        sigma_clip_low_thresh=cfg.sigma_clip_low_thresh,
        sigma_clip_high_thresh=cfg.sigma_clip_high_thresh,
        sigma_clip_func=np.ma.median,
        mem_limit=cfg.mem_limit,
    )
    combined_dark.meta["combined"] = True
    combined_dark_path = calibrated_path / "master_dark.fits"
    write_image_only(combined_dark, combined_dark_path, overwrite=cfg.overwrite)
    log(f"Master dark created: {combined_dark_path}")

    return combined_dark


def flat(files, zero, combined_dark, calibrated_path, cfg: ReductionConfig, log, cancel_event):
    """
    Bias- and dark-subtract each flat frame, then combine per filter into
    normalised master flats.

    Delegates to _process_flats() and _combine_flats() to keep each job
    focused and independently cancellable.
    """
    log("\nStarting flat calibration.")

    paths_by_filter = _process_flats(
        files, zero, combined_dark, calibrated_path, cfg, log, cancel_event
    )
    if paths_by_filter is None:
        return  # canceled during processing

    _combine_flats(paths_by_filter, calibrated_path, cfg, log, cancel_event)


def _process_flats(files, zero, combined_dark, calibrated_path, cfg, log, cancel_event):
    """
    Preprocess individual flat frames (overscan, bias, dark subtraction)
    and group their output paths by filter.

    :return: dict mapping filter name → list of calibrated flat paths,
             or None if canceled
    """
    flat_files = files.files_filtered(imagetyp="FLAT")
    n_total = len(flat_files)
    log(f"Found {n_total} flat frame(s).")

    paths_by_filter: dict[str, list[str]] = {}
    for n_done, (ccd, file_name) in enumerate(
        files.ccds(imagetyp="FLAT", return_fname=True, ccd_kwargs={"unit": "adu"}), start=1
    ):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        filt = ccd.header["FILTER"]
        log(f"Processing flat {n_done}/{n_total} [{filt}]: {file_name}")
        final_ccd = _reduce_flat(ccd, cfg, zero, combined_dark)
        new_fname = f"{file_name.split('.')[0]}.fits"
        output_path = calibrated_path / new_fname
        write_image_only(final_ccd, output_path, overwrite=cfg.overwrite)
        add_header(calibrated_path, new_fname, "FLAT", None, None, None, cfg)
        paths_by_filter.setdefault(filt, []).append(str(output_path))

    log("\nFinished processing individual flat frames.")
    return paths_by_filter


def _combine_flats(paths_by_filter, calibrated_path, cfg, log, cancel_event):
    """
    Combine pre-processed flat frames per filter into normalised master flats.

    :param paths_by_filter: dict mapping filter name → list of calibrated paths
    """
    log("\nStarting flat combination by filter.")
    n_filters = len(paths_by_filter)

    for n_done, (filt, flat_paths) in enumerate(paths_by_filter.items(), start=1):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        n_frames = len(flat_paths)
        log(f"Combining filter {n_done}/{n_filters}: {filt} ({n_frames} frame(s))")
        combined_flats = ccdp.combine(
            flat_paths,
            method="median",
            sigma_clip=True,
            sigma_clip_low_thresh=cfg.sigma_clip_low_thresh,
            sigma_clip_high_thresh=cfg.sigma_clip_high_thresh,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=cfg.mem_limit,
        )
        combined_flats.meta["combined"] = True
        flat_file_name = f"master_flat_{filt.replace('Empty/', '')}.fits"
        write_image_only(combined_flats, calibrated_path / flat_file_name, overwrite=cfg.overwrite)
        add_header(calibrated_path, flat_file_name, "FLAT", None, None, None, cfg)
        log(f"Master flat created: {flat_file_name}")

    log("\nFinished creating master flats by filter.")


def science_images(files, calibrated_data, zero, combined_dark, cfg: ReductionConfig, log, cancel_event):
    """
    Fully calibrate all science (LIGHT) frames: bias, dark, flat-field,
    and write BJD_TDB to the header.
    """
    flat_imagetyp = "FLAT"
    science_imagetyp = "LIGHT"

    # Build the master-flat lookup once — no repeated IFC scans
    ifc_reduced = ccdp.ImageFileCollection(calibrated_data)
    combined_flats = {
        ccd.header["filter"]: ccd
        for ccd in ifc_reduced.ccds(imagetyp=flat_imagetyp, combined=True)
    }

    science_files = files.files_filtered(imagetyp=science_imagetyp)
    n_total = len(science_files)
    log(f"\nFound {n_total} science frame(s). Starting reduction.")

    for n_done, (light, file_name) in enumerate(
        files.ccds(imagetyp=science_imagetyp, return_fname=True, ccd_kwargs={"unit": "adu"}), start=1
    ):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        filt = light.header["filter"]
        log(f"Calibrating science {n_done}/{n_total} [{filt}]: {file_name}")

        good_flat = combined_flats[filt]
        reduced = _reduce_science(light, cfg, zero, combined_dark, good_flat)

        new_fname = f"{file_name.split('.')[0]}.fits"
        write_image_only(reduced, calibrated_data / new_fname, overwrite=cfg.overwrite)

        hjd = light.header["JD-HELIO"]
        ra = light.header["RA"]
        dec = light.header["DEC"]
        add_header(calibrated_data, new_fname, science_imagetyp, hjd, ra, dec, cfg)

    log("\nFinished calibrating all science images.")


# ---------------------------------------------------------------------------
# Header utilities
# ---------------------------------------------------------------------------

def add_header(pathway, fname, imagetyp, hjd, ra, dec, cfg: ReductionConfig):
    """
    Write reduction metadata into a FITS header.

    For LIGHT frames the HJD is converted to BJD_TDB and stored as well.

    :param pathway: Directory containing the file
    :param fname: File name
    :param imagetyp: FITS IMAGETYP value
    :param hjd: Heliocentric Julian Date (LIGHT frames only, else None)
    :param ra: Right ascension string (LIGHT frames only, else None)
    :param dec: Declination string (LIGHT frames only, else None)
    :param cfg: ReductionConfig
    """
    image_name = pathway / fname
    fits.setval(image_name, "GAIN",     value=cfg.gain,     comment="Units of e-/ADU")
    fits.setval(image_name, "RDNOISE",  value=cfg.rdnoise,  comment="Units of e-")
    fits.setval(image_name, "OBSERVAT", value=cfg.location, comment="Observing location")
    fits.setval(image_name, "IMAGETYP", value=imagetyp,     comment="Image type")
    fits.setval(image_name, "DATASEC",  value=cfg.trim_region,     comment="Trim data section")
    fits.setval(image_name, "BIASSEC",  value=cfg.overscan_region, comment="Overscan section")
    fits.setval(image_name, "EPOCH",    value="J2000.0")

    if imagetyp == "LIGHT":
        bjd = BJD_TDB(hjd, cfg.location, ra, dec)
        fits.setval(image_name, "BJD_TDB", value=bjd.value,
                    comment="Bary. Julian Date, Bary. Dynamical Time")


def BJD_TDB(hjd, obs_loc: str, ra, dec):
    """
    Convert a Heliocentric Julian Date to Barycentric Julian Date (TDB).

    :param hjd: HJD of mid-exposure
    :param obs_loc: Site key string (e.g. 'bsuo') or an astropy site name
    :param ra: Right ascension (hms string)
    :param dec: Declination (degrees string)
    :return: Barycentric Julian Date as an astropy Time object (TDB scale)
    """
    if obs_loc.lower() == "bsuo":
        coords = {"lon": -85.411896, "lat": 40.199879, "elevation": 0.2873}
        earth_loc = EarthLocation.from_geodetic(
            coords["lon"], coords["lat"], coords["elevation"]
        )
    else:
        earth_loc = EarthLocation.of_site(obs_loc)

    helio = Time(hjd, scale="utc", format="jd")
    star = SkyCoord(ra, dec, unit=(u.hour, u.deg))

    ltt = helio.light_travel_time(star, "heliocentric", location=earth_loc)
    guess = helio - ltt
    delta = (guess + guess.light_travel_time(star, "heliocentric", earth_loc)).jd - helio.jd
    guess -= delta * u.d

    ltt = guess.light_travel_time(star, "barycentric", earth_loc)
    return guess.tdb + ltt