"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 04/07/2026

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
    overscan_region: str = "[2073:2115, :]"     # FITS section string
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
    extensions. Clearing them before writing ensures only the image data and
    header are saved.

    :param ccd: CCDData object to write
    :param path: Destination path (str or Path)
    :param overwrite: Whether to overwrite an existing file
    """
    ccd.mask = None
    ccd.uncertainty = None
    ccd.write(str(path), overwrite=overwrite)


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

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled before starting.")
            return

        images_path = Path(path)
        calibrated_data = Path(calibrated)

        if not images_path.exists():
            raise FileNotFoundError(f"Raw images path '{path}' does not exist.")
        calibrated_data.mkdir(parents=True, exist_ok=True)

        files = ccdp.ImageFileCollection(images_path)

        zero = bias(files, calibrated_data, cfg, log, cancel_event)
        master_dark = (
            dark(files, zero, calibrated_data, cfg, log, cancel_event)
            if cfg.dark_bool
            else None
        )
        flat(files, zero, master_dark, calibrated_data, cfg, log, cancel_event)
        science_images(files, calibrated_data, zero, master_dark, cfg, log, cancel_event)

        log("\nReduction process completed successfully.\n")

    except Exception as e:
        log(f"An error occurred: {e}")
        raise


# ---------------------------------------------------------------------------
# Reduction core
# ---------------------------------------------------------------------------

def reduce(ccd, cfg: ReductionConfig, num, zero=None, combined_dark=None, good_flat=None):
    """
    Apply overscan subtraction, trimming, gain correction, and the
    stage-appropriate calibration step.

    num codes:
        0 — bias
        1 — dark
        2 — flat
        3 — science

    :param ccd: Input CCDData image
    :param cfg: ReductionConfig
    :param num: Processing stage (0–3)
    :param zero: Master bias CCDData
    :param combined_dark: Master dark CCDData
    :param good_flat: Master flat CCDData for this filter
    :return: Calibrated CCDData
    """
    # --- Overscan + trim + gain ---
    if cfg.overscan_region.lower() != "none":
        ccd = ccdp.subtract_overscan(
            ccd, fits_section=cfg.overscan_region, median=True, overscan_axis=None
        )

    ccd = ccdp.trim_image(ccd, fits_section=cfg.trim_region)
    ccd = ccdp.gain_correct(ccd, gain=cfg.gain * u.electron / u.adu)

    # --- Stage-specific calibration ---
    if num == 0:
        # Bias: overscan/trim/gain only
        return ccd

    elif num == 1:
        # Dark: subtract bias
        return ccdp.subtract_bias(ccd, zero)

    elif num == 2:
        # Flat: subtract bias, optionally subtract dark
        ccd = ccdp.subtract_bias(ccd, zero)
        if cfg.dark_bool:
            ccd = ccdp.subtract_dark(
                ccd, combined_dark, exposure_time="exptime", exposure_unit=u.second, scale=True
            )
        return ccd

    elif num == 3:
        # Science: subtract bias, optionally subtract dark, flat-field correct
        ccd = ccdp.subtract_bias(ccd, zero)
        if cfg.dark_bool:
            ccd = ccdp.subtract_dark(
                ccd, combined_dark, exposure_time="exptime", exposure_unit=u.second, scale=True
            )
        ccd = ccdp.flat_correct(ccd=ccd, flat=good_flat, min_value=1.0)
        return ccd

    else:
        raise ValueError(f"Unknown reduction stage: {num}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def bias(files, calibrated_data, cfg: ReductionConfig, log, cancel_event):
    """
    Overscan-correct, trim, and gain-correct each bias frame, then combine
    them into a master bias.

    :return: Combined master bias CCDData
    """
    log("\nStarting bias calibration.")
    log(f"Overscan Region: {cfg.overscan_region}")
    log(f"Trim Region:     {cfg.trim_region}")

    for ccd, file_name in files.ccds(imagetyp="BIAS", return_fname=True, ccd_kwargs={"unit": "adu"}):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        log(f"Processing bias image: {file_name}")
        new_ccd = reduce(ccd, cfg, num=0)
        output_path = calibrated_data / f"{file_name.split('.')[0]}.fits"
        write_image_only(new_ccd, output_path, overwrite=cfg.overwrite)
        log(f"Saved calibrated bias: {output_path}")

    log("\nCombining bias frames to create master bias.")
    reduced_images = ccdp.ImageFileCollection(calibrated_data)
    calibrated_biases = reduced_images.files_filtered(imagetyp="BIAS", include_path=True)

    combined_bias = ccdp.combine(
        calibrated_biases,
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

    :return: Combined master dark CCDData
    """
    log("\nStarting dark calibration.")

    for ccd, file_name in files.ccds(imagetyp="DARK", return_fname=True, ccd_kwargs={"unit": "adu"}):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        log(f"Processing dark image: {file_name}")
        sub_ccd = reduce(ccd, cfg, num=1, zero=zero)
        output_path = calibrated_path / f"{file_name.split('.')[0]}.fits"
        write_image_only(sub_ccd, output_path, overwrite=cfg.overwrite)
        log(f"Saved calibrated dark: {output_path}")

    log("\nCombining dark frames to create master dark.")
    reduced_images = ccdp.ImageFileCollection(calibrated_path)
    calibrated_darks = reduced_images.files_filtered(imagetyp="DARK", include_path=True)

    combined_dark = ccdp.combine(
        calibrated_darks,
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
    """
    log("\nStarting flat calibration.")

    for ccd, file_name in files.ccds(imagetyp="FLAT", return_fname=True, ccd_kwargs={"unit": "adu"}):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        final_ccd = reduce(ccd, cfg, num=2, zero=zero, combined_dark=combined_dark)
        new_fname = f"{file_name.split('.')[0]}.fits"
        output_path = calibrated_path / new_fname
        write_image_only(final_ccd, output_path, overwrite=cfg.overwrite)
        add_header(calibrated_path, new_fname, "FLAT", "None", None, None, None, cfg)
        log(f"Finished overscan/bias/dark correction for {new_fname}")

    log("\nFinished processing individual flat frames.")
    log("\nStarting flat combination by filter.")

    ifc = ccdp.ImageFileCollection(calibrated_path)
    flat_filters = set(h["FILTER"] for h in ifc.headers(imagetyp="FLAT"))

    for filt in flat_filters:
        to_combine = ifc.files_filtered(imagetyp="flat", filter=filt, include_path=True)
        combined_flats = ccdp.combine(
            to_combine,
            method="median",
            sigma_clip=True,
            sigma_clip_high_thresh=cfg.sigma_clip_high_thresh,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,   # was: signma_clip_dev_func (typo)
            rdnoise=cfg.rdnoise * u.electron,
            gain=cfg.gain * u.electron / u.adu,
            mem_limit=cfg.mem_limit,
        )
        combined_flats.meta["combined"] = True
        flat_file_name = f"master_flat_{filt.replace('Empty/', '')}.fits"
        write_image_only(combined_flats, calibrated_path / flat_file_name, overwrite=cfg.overwrite)
        add_header(calibrated_path, flat_file_name, "FLAT", "None", None, None, None, cfg)
        log(f"Finished combining flat: {flat_file_name}")

    log("\nFinished creating master flats by filter.")


def science_images(files, calibrated_data, zero, combined_dark, cfg: ReductionConfig, log, cancel_event):
    """
    Fully calibrate all science (LIGHT) frames: bias, dark, flat-field,
    and write BJD_TDB to the header.
    """
    science_imagetyp = "LIGHT"
    flat_imagetyp = "FLAT"

    ifc_reduced = ccdp.ImageFileCollection(calibrated_data)
    combined_flats = {
        ccd.header["filter"]: ccd
        for ccd in ifc_reduced.ccds(imagetyp=flat_imagetyp, combined=True)
    }

    log("\nStarting reduction of science images.")

    for light, file_name in files.ccds(imagetyp=science_imagetyp, return_fname=True, ccd_kwargs={"unit": "adu"}):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        good_flat = combined_flats[light.header["filter"]]
        reduced = reduce(light, cfg, num=3, zero=zero, combined_dark=combined_dark, good_flat=good_flat)

        new_fname = f"{file_name.split('.')[0]}.fits"
        write_image_only(reduced, calibrated_data / new_fname, overwrite=cfg.overwrite)

        hjd = light.header["JD-HELIO"]
        ra = light.header["RA"]
        dec = light.header["DEC"]
        add_header(calibrated_data, new_fname, science_imagetyp, "None", hjd, ra, dec, cfg)

        log(f"Finished calibration of {new_fname}")

    log("\nFinished calibrating all science images.")


# ---------------------------------------------------------------------------
# Header utilities
# ---------------------------------------------------------------------------

def add_header(pathway, fname, imagetyp, filter_name, hjd, ra, dec, cfg: ReductionConfig):
    """
    Write reduction metadata into a FITS header.

    For LIGHT frames the HJD is converted to BJD_TDB and stored as well.

    :param pathway: Directory containing the file
    :param fname: File name
    :param imagetyp: FITS IMAGETYP value
    :param filter_name: Filter name (or "None")
    :param hjd: Heliocentric Julian Date (LIGHT frames only)
    :param ra: Right ascension string (LIGHT frames only)
    :param dec: Declination string (LIGHT frames only)
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