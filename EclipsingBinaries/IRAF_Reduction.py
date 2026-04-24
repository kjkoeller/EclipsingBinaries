"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 04/23/2026

This program is meant to automatically do the data reduction of the raw images from the
Ball State University Observatory (BSUO) and SARA data. The new calibrated images are placed into a new folder as to
not overwrite the original images.
"""
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import json
import shutil
import warnings

import astropy
from astropy import wcs
from astropy.stats import mad_std
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.nddata import CCDData

import ccdproc as ccdp
import numpy as np

# Suppress FITS standard-compliance header warnings
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


# Science exposure times more than this factor longer than the longest
# dark produce noisy scaling — warn when exceeded.
DARK_SCALING_WARN_RATIO = 10.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeaderConfig:
    """
    FITS header conventions that vary between observatories and capture
    software. Everything the pipeline reads from or writes to a FITS header
    is driven by this object, so new setups can be supported without code
    changes.

    Invalid values raise ValueError at construction time via __post_init__.
    """
    # --- IMAGETYP values (what this pipeline expects to find in raw frames)
    # Some capture software writes 'Bias Frame' / 'Light Frame' / 'zero' /
    # 'object' etc. — override these to match your data.
    imagetyp_bias: str = "BIAS"
    imagetyp_dark: str = "DARK"
    imagetyp_flat: str = "FLAT"
    imagetyp_light: str = "LIGHT"

    # --- Filter normalization
    # Prefixes stripped from raw FILTER values before uppercasing. Matching
    # is case-insensitive. Example: BSUO writes 'Empty/V' for filter-wheel
    # slot V, so 'Empty/' belongs here.
    filter_prefix_strip: Tuple[str, ...] = ("Empty/",)

    # --- Time / coord header key priority lists (first present wins)
    time_header_keys: Tuple[str, ...] = ("JD-HELIO", "HJD_UTC", "HJD-UTC", "HJD")
    ra_header_keys: Tuple[str, ...] = ("RA", "OBJCTRA", "OBJCT RA", "RA-OBJ")
    dec_header_keys: Tuple[str, ...] = ("DEC", "OBJCTDEC", "OBJCT DEC", "DEC-OBJ")

    # --- Simple keywords the pipeline reads from raw frames
    filter_key: str = "FILTER"
    exptime_key: str = "EXPTIME"

    # --- Keywords the pipeline writes to output frames
    gain_key: str = "GAIN"
    rdnoise_key: str = "RDNOISE"
    observatory_key: str = "OBSERVAT"
    imagetyp_key: str = "IMAGETYP"
    datasec_key: str = "DATASEC"
    biassec_key: str = "BIASSEC"
    epoch_key: str = "EPOCH"
    bjd_tdb_key: str = "BJD_TDB"

    # --- Epoch value written to output headers
    epoch_value: str = "J2000.0"

    def __post_init__(self):
        # Required non-empty strings
        required = {
            "imagetyp_bias": self.imagetyp_bias,
            "imagetyp_dark": self.imagetyp_dark,
            "imagetyp_flat": self.imagetyp_flat,
            "imagetyp_light": self.imagetyp_light,
            "filter_key": self.filter_key,
            "exptime_key": self.exptime_key,
            "gain_key": self.gain_key,
            "rdnoise_key": self.rdnoise_key,
            "observatory_key": self.observatory_key,
            "imagetyp_key": self.imagetyp_key,
            "datasec_key": self.datasec_key,
            "biassec_key": self.biassec_key,
            "epoch_key": self.epoch_key,
            "bjd_tdb_key": self.bjd_tdb_key,
            "epoch_value": self.epoch_value,
        }
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"HeaderConfig.{name} must be a non-empty string, got {value!r}")

        # Tuples must contain at least one non-empty string
        for name in ("time_header_keys", "ra_header_keys", "dec_header_keys"):
            value = getattr(self, name)
            if not value or not all(isinstance(k, str) and k.strip() for k in value):
                raise ValueError(
                    f"HeaderConfig.{name} must be a non-empty sequence of strings, got {value!r}"
                )


@dataclass
class ReductionConfig:
    """
    All tunable parameters for a reduction run in one place.
    Pass a ReductionConfig instance into run_reduction() instead of relying
    on module-level globals.

    Invalid values raise ValueError at construction time via __post_init__.
    """
    gain: float = 1.43                          # e-/ADU  (BSUO default)
    rdnoise: float = 10.83                      # e-      (BSUO default)
    sigclip: int = 5                            # sigma for cosmic-ray removal
    sigma_clip_low_thresh: Optional[int] = None # lower sigma for combine
    sigma_clip_high_thresh: int = 3             # upper sigma for combine
    mem_limit: float = 1600e6                   # bytes (~1.6 GB)
    dark_bool: bool = True                      # whether dark frames exist
    flat_bool: bool = True                      # whether flat frames exist
    location: str = "bsuo"                      # observing site key
    overwrite: bool = True                      # overwrite existing output files
    overscan_region: str = "[2073:2115, :]"     # FITS section string; set to "none" to skip
    trim_region: str = "[20:2060, 12:2057]"     # FITS section string
    reuse_masters: bool = False                 # skip master regeneration if fresh masters exist
    science_only: bool = False                  # load masters from disk, skip bias/dark/flat stages entirely
    # Master filename customisation — default values match what this pipeline
    # generates, but can be overridden to point at existing masters with
    # different names (e.g. Dark.fits or FLAT{filter}.fits). The flat pattern
    # must contain exactly one {filter} placeholder.
    master_bias_name: str = "zero.fits"
    master_dark_name: str = "master_dark.fits"
    master_flat_pattern: str = "master_flat_{filter}.fits"

    # FITS header conventions — IMAGETYP values, keyword names, filter
    # prefix stripping, and time/coord alias lists. Defaults follow the
    # FITS standard / BSUO conventions; override to support other setups.
    headers: HeaderConfig = field(default_factory=HeaderConfig)

    def __post_init__(self):
        if self.gain <= 0:
            raise ValueError(f"gain must be positive, got {self.gain}")
        if self.rdnoise < 0:
            raise ValueError(f"rdnoise must be non-negative, got {self.rdnoise}")
        if self.sigclip <= 0:
            raise ValueError(f"sigclip must be positive, got {self.sigclip}")
        if self.sigma_clip_high_thresh <= 0:
            raise ValueError(
                f"sigma_clip_high_thresh must be positive, got {self.sigma_clip_high_thresh}"
            )
        if self.sigma_clip_low_thresh is not None and self.sigma_clip_low_thresh <= 0:
            raise ValueError(
                f"sigma_clip_low_thresh must be positive or None, got {self.sigma_clip_low_thresh}"
            )
        if self.mem_limit <= 0:
            raise ValueError(f"mem_limit must be positive, got {self.mem_limit}")
        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError(f"location must be a non-empty string, got {self.location!r}")
        if "{filter}" not in self.master_flat_pattern:
            raise ValueError(
                f"master_flat_pattern must contain '{{filter}}' placeholder, "
                f"got {self.master_flat_pattern!r}"
            )


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
# Filter & exposure utilities
# ---------------------------------------------------------------------------

def _normalize_filter(raw_filter, prefixes=("Empty/",)) -> str:
    """
    Normalize a filter name for reliable matching between flats and science.

    Handles:
      - Leading/trailing whitespace
      - Filter-wheel prefixes (case-insensitive) such as ``"Empty/"``
      - Case differences — result is uppercased

    :param raw_filter: Raw filter header value
    :param prefixes: Iterable of prefixes to strip (case-insensitive)
    :return: Normalized filter string
    :raises ValueError: If the filter is empty or not a string
    """
    if raw_filter is None:
        raise ValueError("Filter header value is missing.")
    s = str(raw_filter).strip()
    if not s:
        raise ValueError("Filter header value is empty.")
    s_lower = s.lower()
    for prefix in prefixes:
        if prefix and s_lower.startswith(prefix.lower()):
            s = s[len(prefix):]
            break
    return s.strip().upper()


def _header_get_any(header, *keys) -> Optional[object]:
    """
    Look up the first present keyword from a prioritized list.

    FITS headers from different capture software use different keywords for
    the same conceptual value (e.g. BSUO writes heliocentric time as
    ``HJD_UTC`` rather than the FITS-convention ``JD-HELIO``, and target
    coordinates as ``OBJCTRA`` / ``OBJCT DEC``). This helper returns the
    first matching non-empty value so callers can accept any of them.

    :param header: FITS header
    :param keys: Header keyword aliases, in priority order
    :return: The value of the first present keyword, or None if none match
    """
    for key in keys:
        try:
            value = header.get(key)
        except Exception:
            # Some header implementations raise on invalid keys rather than
            # returning None — treat that as "not present" and move on.
            continue
        if value is None:
            continue
        # Treat empty / whitespace-only strings as missing too
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _get_exposure_time(header, key="EXPTIME") -> Optional[float]:
    """
    Extract exposure time from a FITS header, returning None if missing or invalid.

    :param header: FITS header
    :param key: Header keyword to read (defaults to the FITS standard EXPTIME)
    :return: Exposure time in seconds, or None
    """
    exp = header.get(key)
    if exp is None:
        return None
    try:
        exp_f = float(exp)
    except (TypeError, ValueError):
        return None
    if exp_f <= 0 or not np.isfinite(exp_f):
        return None
    return exp_f


def _max_dark_exposure(dark_paths, exptime_key="EXPTIME") -> Optional[float]:
    """
    Find the maximum exposure time across a list of calibrated dark frames.

    :param dark_paths: List of paths to calibrated darks
    :param exptime_key: Header keyword holding exposure time
    :return: Maximum exposure time in seconds, or None if none readable
    """
    max_exp = None
    for dpath in dark_paths:
        try:
            with fits.open(dpath) as hdul:
                exp = _get_exposure_time(hdul[0].header, key=exptime_key)
                if exp is not None:
                    if max_exp is None or exp > max_exp:
                        max_exp = exp
        except Exception:
            continue
    return max_exp


# ---------------------------------------------------------------------------
# Shape-consistency check
# ---------------------------------------------------------------------------

def _assert_shape_matches(ccd, reference_shape, context):
    """
    Verify that a CCDData array has the expected shape.

    :param ccd: CCDData object to check
    :param reference_shape: Expected (ny, nx) tuple
    :param context: Human-readable description for error messages
    :raises ValueError: If shapes differ
    """
    if ccd.data.shape != reference_shape:
        raise ValueError(
            f"Dimension mismatch in {context}: "
            f"got {ccd.data.shape}, expected {reference_shape}. "
            "Check that binning/windowing is consistent across all frames."
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_image_only(ccd, path, overwrite=True):
    """
    Write a CCDData object to disk as a plain single-extension FITS file,
    atomically.

    The file is first written to a sibling ``<name>.fits.tmp``, then
    atomically renamed into place. If the write or rename fails, the
    destination is never left in a half-written state. Mask and uncertainty
    arrays are stripped before writing and restored afterwards so the live
    in-memory CCDData object is unmodified.

    :param ccd: CCDData object to write
    :param path: Destination path (str or Path)
    :param overwrite: Whether to overwrite an existing file
    :raises IOError: If the destination is not present on disk after rename
    """
    path_obj = Path(path)
    # Keep the .fits extension on the temp name (rather than .fits.tmp) so
    # astropy's format auto-detection recognises it. Insert ".tmp" before the
    # extension: foo.fits -> foo.tmp.fits
    tmp_path = path_obj.with_name(path_obj.stem + ".tmp" + path_obj.suffix)
    if tmp_path.exists():
        tmp_path.unlink()

    mask, uncertainty = ccd.mask, ccd.uncertainty
    try:
        ccd.mask = None
        ccd.uncertainty = None
        try:
            ccd.write(str(tmp_path), overwrite=True, format="fits")
            # Path.replace is atomic on POSIX and near-atomic on Windows
            tmp_path.replace(path_obj)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise
    finally:
        ccd.mask = mask
        ccd.uncertainty = uncertainty

    if not path_obj.exists():
        raise IOError(f"Failed to write FITS file: {path_obj}")


def _prepare_intermediate_dir(calibrated_data: Path) -> Path:
    """
    Create (or clean) a dedicated subfolder for individual calibrated
    bias/dark/flat frames so they don't pollute the main output folder
    or get mixed up with master files from previous runs.

    :param calibrated_data: Main output directory
    :return: Path to the cleaned intermediate directory
    """
    intermediate = calibrated_data / "intermediate"
    if intermediate.exists():
        for f in intermediate.glob("*"):
            if f.is_file():
                f.unlink()
    else:
        intermediate.mkdir(parents=True)
    return intermediate


def _check_disk_space(raw_path: Path, output_path: Path, mem_limit: float, log):
    """
    Estimate disk space needed for the reduction and log/raise if low.

    Rough estimate: calibrated output files are ~same size as raw inputs.
    Allow ~2× raw data size for outputs and intermediates, plus mem_limit
    for combine spillover.

    :raises RuntimeError: If available space is less than the estimated need
    """
    total_raw = 0
    for ext in ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS"):
        total_raw += sum(f.stat().st_size for f in raw_path.glob(ext))

    if total_raw == 0:
        log("Disk space check: no FITS files found in raw path (skipping check).")
        return

    estimated_needed = int(total_raw * 2 + mem_limit)
    available = shutil.disk_usage(output_path).free

    log(
        f"Disk space check: {available / 1e9:.2f} GB available, "
        f"~{estimated_needed / 1e9:.2f} GB estimated."
    )

    if available < estimated_needed * 1.2:
        if available < estimated_needed:
            raise RuntimeError(
                f"Insufficient disk space at {output_path}: "
                f"need ~{estimated_needed / 1e9:.2f} GB, "
                f"have {available / 1e9:.2f} GB."
            )
        log(
            f"Warning: low disk space headroom "
            f"({available / estimated_needed:.1f}× estimated need)."
        )


# ---------------------------------------------------------------------------
# Pre-flight checks and master reuse helpers
# ---------------------------------------------------------------------------

def _list_raw_fits(directory: Path) -> list[Path]:
    """
    List all FITS-like files in a directory (any extension case).

    :param directory: Directory to scan
    :return: Sorted list of FITS-like paths
    """
    extensions = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS")
    result: list[Path] = []
    for ext in extensions:
        result.extend(directory.glob(ext))
    return sorted(set(result))


def _preflight_fits_check(image_paths, log) -> tuple[list[Path], list[Path]]:
    """
    Open each FITS file briefly to verify it can be read.

    Catches truncated files, permission issues, and other low-level
    problems upfront so the user sees a clean list before any processing
    begins.

    :param image_paths: Iterable of paths to check
    :param log: Logging callable
    :return: (readable_paths, unreadable_paths)
    """
    readable: list[Path] = []
    unreadable: list[Path] = []

    for path in image_paths:
        path_obj = Path(path)
        try:
            with fits.open(path_obj, mode="readonly") as hdul:
                hdul.verify("silentfix+warn")
                _ = hdul[0].header  # force header read
            readable.append(path_obj)
        except Exception as e:
            log(f"Warning: unreadable FITS file {path_obj.name}: {e}")
            unreadable.append(path_obj)

    if unreadable:
        log(
            f"Pre-flight check: {len(readable)} readable, "
            f"{len(unreadable)} unreadable FITS file(s) will be skipped."
        )
    else:
        log(f"Pre-flight check: all {len(readable)} FITS file(s) readable.")

    return readable, unreadable


def _master_is_fresh(master_path: Path, raw_paths) -> bool:
    """
    Determine whether an existing master calibration file can be reused.

    A master is considered fresh when it exists and its mtime is at least
    as recent as the newest raw calibration frame. Any raw frame modified
    after the master implies the master is stale.

    :param master_path: Path to the master file (e.g. zero.fits)
    :param raw_paths: Iterable of raw calibration file paths
    :return: True if master is fresh and usable
    """
    if not master_path.exists():
        return False

    master_mtime = master_path.stat().st_mtime
    for raw_path in raw_paths:
        raw_path_obj = Path(raw_path)
        if not raw_path_obj.exists():
            continue
        if raw_path_obj.stat().st_mtime > master_mtime:
            return False
    return True


def _load_master(master_path: Path, unit: str = "electron") -> CCDData:
    """
    Load a master calibration frame from disk.

    Master bias and master dark are gain-corrected in our pipeline, so the
    default unit is electron. Callers can override for special cases.

    :param master_path: Path to the master FITS file
    :param unit: Unit string for CCDData construction
    :return: CCDData object
    """
    return CCDData.read(str(master_path), unit=unit, format="fits")


def _discover_master_flats(calibrated_data: Path, cfg) -> dict:
    """
    Discover all master flats in ``calibrated_data`` matching the filename
    pattern, keyed by normalized filter name.

    The pattern's {filter} placeholder becomes a glob wildcard: e.g.
    ``master_flat_{filter}.fits`` matches ``master_flat_B.fits``,
    ``master_flat_V.fits``, etc. The filter portion of the matched filename
    is extracted and normalized.

    :param calibrated_data: Directory to scan
    :param cfg: ReductionConfig (provides master_flat_pattern)
    :return: dict mapping normalized filter name -> master flat Path
    """
    pattern = cfg.master_flat_pattern
    # Split pattern into prefix + suffix around {filter}
    prefix, suffix = pattern.split("{filter}", 1)
    glob_pattern = f"{prefix}*{suffix}"

    found: dict = {}
    for master_path in calibrated_data.glob(glob_pattern):
        name = master_path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        # Extract the {filter} portion
        raw_filt = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
        if not raw_filt:
            continue
        try:
            filt = _normalize_filter(raw_filt)
        except ValueError:
            continue
        found[filt] = master_path
    return found


def _write_config_snapshot(cfg, output_dir: Path, raw_path: Path, log):
    """
    Write a JSON snapshot of the run's ReductionConfig and environment to
    ``reduction_config.json`` in the output folder.

    This captures exactly which gain, overscan region, thresholds, etc.
    produced the calibrated frames, along with library versions for
    reproducibility.

    :param cfg: ReductionConfig used for this run
    :param output_dir: Output directory to write the snapshot into
    :param raw_path: Input raw-images path (recorded for provenance)
    :param log: Logging callable
    """
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
        "output_path": str(output_dir),
        "versions": {
            "astropy": astropy.__version__,
            "ccdproc": ccdp.__version__,
            "numpy": np.__version__,
        },
        "config": asdict(cfg),
    }
    snapshot_path = output_dir / "reduction_config.json"
    try:
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True, default=str)
        log(f"Config snapshot written: {snapshot_path}")
    except Exception as e:
        # Non-fatal — snapshot is for reproducibility, not correctness
        log(f"Warning: failed to write config snapshot: {e}")


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
    if not images_path.is_dir():
        raise NotADirectoryError(f"Raw images path '{path}' is not a directory.")
    calibrated_data.mkdir(parents=True, exist_ok=True)

    # Clean intermediate subfolder to avoid stale-frame contamination
    intermediate_dir = _prepare_intermediate_dir(calibrated_data)
    log(f"Intermediate frames will be written to: {intermediate_dir}")

    # Config snapshot for reproducibility
    _write_config_snapshot(cfg, calibrated_data, images_path, log)

    # Pre-flight disk space check
    _check_disk_space(images_path, calibrated_data, cfg.mem_limit, log)

    # Pre-flight FITS readability check
    all_raw = _list_raw_fits(images_path)
    _preflight_fits_check(all_raw, log)

    files = ccdp.ImageFileCollection(images_path)

    # --- Science-only mode: load masters from disk, skip calibration stages ---
    if cfg.science_only:
        log("\nScience-only mode: skipping bias/dark/flat calibration stages.")
        try:
            zero, master_dark, max_dark_exp, reference_shape = _load_masters_from_disk(
                calibrated_data, cfg, log
            )
        except Exception as e:
            raise RuntimeError("Failed to load existing master frames.") from e

        try:
            science_images(
                files, calibrated_data, zero, master_dark,
                cfg, log, cancel_event, reference_shape, max_dark_exp,
            )
        except Exception as e:
            raise RuntimeError("Science stage failed.") from e
        if canceled():
            log("Reduction aborted: science stage did not complete.")
            return

        log("\nReduction process completed successfully.\n")
        return

    # --- Bias ---
    try:
        zero = bias(files, calibrated_data, intermediate_dir, cfg, log, cancel_event)
    except Exception as e:
        raise RuntimeError("Bias stage failed.") from e
    if zero is None:
        log("Reduction aborted: bias stage did not complete.")
        return

    reference_shape = zero.data.shape
    log(f"Reference frame shape set from master bias: {reference_shape}")

    # --- Dark (optional) ---
    master_dark = None
    max_dark_exp = None
    if cfg.dark_bool:
        try:
            master_dark, max_dark_exp = dark(
                files, zero, calibrated_data, intermediate_dir,
                cfg, log, cancel_event, reference_shape,
            )
        except Exception as e:
            raise RuntimeError("Dark stage failed.") from e
        if master_dark is None:
            log("Reduction aborted: dark stage did not complete.")
            return

    # --- Flat ---
    try:
        flat(
            files, zero, master_dark, calibrated_data, intermediate_dir,
            cfg, log, cancel_event, reference_shape,
        )
    except Exception as e:
        raise RuntimeError("Flat stage failed.") from e
    if canceled():
        log("Reduction aborted: flat stage did not complete.")
        return

    # --- Science ---
    try:
        science_images(
            files, calibrated_data, zero, master_dark,
            cfg, log, cancel_event, reference_shape, max_dark_exp,
        )
    except Exception as e:
        raise RuntimeError("Science stage failed.") from e
    if canceled():
        log("Reduction aborted: science stage did not complete.")
        return

    log("\nReduction process completed successfully.\n")


def _load_masters_from_disk(calibrated_data: Path, cfg: ReductionConfig, log):
    """
    Load master bias, master dark (if cfg.dark_bool), and discover master
    flats for science-only mode.

    :return: (zero, master_dark, max_dark_exp, reference_shape)
    :raises FileNotFoundError: If any required master is missing
    """
    # Master bias
    bias_path = calibrated_data / cfg.master_bias_name
    if not bias_path.exists():
        raise FileNotFoundError(
            f"Master bias not found at '{bias_path}'. "
            f"Set cfg.master_bias_name or place the file there."
        )
    zero = _load_master(bias_path)
    reference_shape = zero.data.shape
    log(f"Loaded master bias from {bias_path} (shape={reference_shape})")

    # Master dark (optional)
    master_dark = None
    max_dark_exp = None
    if cfg.dark_bool:
        dark_path = calibrated_data / cfg.master_dark_name
        if not dark_path.exists():
            raise FileNotFoundError(
                f"Master dark not found at '{dark_path}'. "
                f"Set cfg.master_dark_name, cfg.dark_bool=False, or place the file there."
            )
        master_dark = _load_master(dark_path)
        if master_dark.data.shape != reference_shape:
            raise ValueError(
                f"Master dark shape {master_dark.data.shape} does not match "
                f"master bias shape {reference_shape}."
            )
        max_dark_exp = _get_exposure_time(master_dark.header)
        log(
            f"Loaded master dark from {dark_path}"
            + (f" (EXPTIME={max_dark_exp:.1f}s)" if max_dark_exp else "")
        )

    # Master flats are discovered in science_images via _discover_master_flats —
    # we don't need to load them here. Just verify at least one exists.
    discovered = _discover_master_flats(calibrated_data, cfg)
    if not discovered:
        raise FileNotFoundError(
            f"No master flats matching pattern '{cfg.master_flat_pattern}' "
            f"found in '{calibrated_data}'."
        )
    log(f"Discovered master flats for filters: {sorted(discovered.keys())}")

    return zero, master_dark, max_dark_exp, reference_shape


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
    """Preprocess a single bias frame (overscan, trim, gain)."""
    return _preprocess(ccd, cfg)


def _reduce_dark(ccd, cfg: ReductionConfig, zero):
    """Preprocess and bias-subtract a single dark frame."""
    ccd = _preprocess(ccd, cfg)
    return ccdp.subtract_bias(ccd, zero)


def _reduce_flat(ccd, cfg: ReductionConfig, zero, combined_dark):
    """Preprocess, bias-subtract, and optionally dark-subtract a single flat frame."""
    ccd = _preprocess(ccd, cfg)
    ccd = ccdp.subtract_bias(ccd, zero)
    if cfg.dark_bool:
        ccd = ccdp.subtract_dark(
            ccd, combined_dark, exposure_time="exptime", exposure_unit=u.second, scale=True
        )
    return ccd


def _reduce_science(ccd, cfg: ReductionConfig, zero, combined_dark, good_flat):
    """Fully calibrate a single science frame: preprocess, bias, dark, flat-field."""
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

def bias(files, calibrated_data, intermediate_dir, cfg: ReductionConfig, log, cancel_event):
    """
    Overscan-correct, trim, and gain-correct each bias frame, then combine
    them into a master bias.

    If ``cfg.reuse_masters`` is True and ``zero.fits`` already exists and is
    fresher than every raw bias frame, the existing master is loaded and
    returned — skipping regeneration entirely.

    :return: Combined master bias CCDData, or None if canceled
    :raises ValueError: If no bias frames are found
    """
    log("\nStarting bias calibration.")
    log(f"Overscan Region: {cfg.overscan_region}")
    log(f"Trim Region:     {cfg.trim_region}")

    bias_paths = files.files_filtered(imagetyp=cfg.headers.imagetyp_bias, include_path=True)
    n_total = len(bias_paths)

    if n_total == 0:
        raise ValueError(
            f"No BIAS frames found in '{files.location}'. "
            "Check that IMAGETYP headers are set to 'BIAS'."
        )
    log(f"Found {n_total} bias frame(s).")

    # Fast path: reuse existing master bias if fresh
    master_bias_path = calibrated_data / cfg.master_bias_name
    if cfg.reuse_masters and _master_is_fresh(master_bias_path, bias_paths):
        log(f"Reusing existing master bias: {master_bias_path}")
        return _load_master(master_bias_path)

    calibrated_bias_paths = []
    n_failed = 0
    reference_shape = None
    for n_done, bias_path in enumerate(bias_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        file_name = Path(bias_path).name
        log(f"Processing bias {n_done}/{n_total}: {file_name}")
        try:
            ccd = CCDData.read(bias_path, unit="adu", format="fits")
            new_ccd = _reduce_bias(ccd, cfg)

            # First successfully reduced bias sets the reference shape
            if reference_shape is None:
                reference_shape = new_ccd.data.shape
                log(f"Reference shape established from first bias: {reference_shape}")
            else:
                _assert_shape_matches(new_ccd, reference_shape, f"bias frame {file_name}")

            output_path = intermediate_dir / f"{file_name.split('.')[0]}.fits"
            write_image_only(new_ccd, output_path, overwrite=cfg.overwrite)
            calibrated_bias_paths.append(str(output_path))
        except Exception as e:
            n_failed += 1
            log(f"Warning: failed to process {file_name}: {e}. Skipping.")
            continue

    if not calibrated_bias_paths:
        raise RuntimeError(
            f"All {n_total} bias frames failed to process. Cannot create master bias."
        )
    if n_failed:
        log(f"Warning: {n_failed}/{n_total} bias frame(s) failed to process.")

    log(f"\nCombining {len(calibrated_bias_paths)} bias frame(s) into master bias.")
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
    write_image_only(combined_bias, master_bias_path, overwrite=cfg.overwrite)
    log(f"Master bias created: {master_bias_path}")

    return combined_bias


def dark(files, zero, calibrated_data, intermediate_dir, cfg: ReductionConfig,
         log, cancel_event, reference_shape):
    """
    Bias-subtract each dark frame, then combine them into a master dark.
    Every frame is checked against reference_shape before writing.

    If ``cfg.reuse_masters`` is True and ``master_dark.fits`` already exists
    and is fresher than every raw dark frame, the existing master is loaded
    and returned — skipping regeneration entirely.

    :return: (Combined master dark CCDData, max dark EXPTIME in seconds)
             or (None, None) if canceled
    :raises ValueError: If no dark frames are found
    """
    log("\nStarting dark calibration.")

    dark_paths = files.files_filtered(imagetyp=cfg.headers.imagetyp_dark, include_path=True)
    n_total = len(dark_paths)

    if n_total == 0:
        raise ValueError(
            f"No DARK frames found in '{files.location}', but cfg.dark_bool=True. "
            "Either add dark frames or set dark_bool=False."
        )
    log(f"Found {n_total} dark frame(s).")

    # Fast path: reuse existing master dark if fresh. Also scan raw darks
    # for max EXPTIME so the later science-scaling warning still works.
    master_dark_path = calibrated_data / cfg.master_dark_name
    if cfg.reuse_masters and _master_is_fresh(master_dark_path, dark_paths):
        log(f"Reusing existing master dark: {master_dark_path}")
        max_dark_exp = _max_dark_exposure(dark_paths)
        if max_dark_exp is not None:
            log(f"Longest dark EXPTIME (from raw headers): {max_dark_exp:.1f} s")
        return _load_master(master_dark_path), max_dark_exp

    calibrated_dark_paths = []
    n_failed = 0
    for n_done, dark_path in enumerate(dark_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None, None

        file_name = Path(dark_path).name
        log(f"Processing dark {n_done}/{n_total}: {file_name}")
        try:
            ccd = CCDData.read(dark_path, unit="adu", format="fits")
            if _get_exposure_time(ccd.header) is None:
                raise ValueError("missing or invalid EXPTIME")

            sub_ccd = _reduce_dark(ccd, cfg, zero)
            _assert_shape_matches(sub_ccd, reference_shape, f"dark frame {file_name}")

            output_path = intermediate_dir / f"{file_name.split('.')[0]}.fits"
            write_image_only(sub_ccd, output_path, overwrite=cfg.overwrite)
            calibrated_dark_paths.append(str(output_path))
        except Exception as e:
            n_failed += 1
            log(f"Warning: failed to process {file_name}: {e}. Skipping.")
            continue

    if not calibrated_dark_paths:
        raise RuntimeError(
            f"All {n_total} dark frames failed to process. Cannot create master dark."
        )
    if n_failed:
        log(f"Warning: {n_failed}/{n_total} dark frame(s) failed to process.")

    log(f"\nCombining {len(calibrated_dark_paths)} dark frame(s) into master dark.")
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
    write_image_only(combined_dark, master_dark_path, overwrite=cfg.overwrite)
    log(f"Master dark created: {master_dark_path}")

    # Track longest dark exposure for later science-exposure scaling check
    max_dark_exp = _max_dark_exposure(calibrated_dark_paths)
    if max_dark_exp is not None:
        log(f"Longest dark EXPTIME: {max_dark_exp:.1f} s")

    return combined_dark, max_dark_exp


def flat(files, zero, combined_dark, calibrated_data, intermediate_dir,
         cfg: ReductionConfig, log, cancel_event, reference_shape):
    """
    Bias- and dark-subtract each flat frame, then combine per filter into
    normalised master flats. Every frame is checked against reference_shape.

    If ``cfg.reuse_masters`` is True, existing master flats are reused when
    (a) every distinct filter present in the raw flats has a corresponding
    ``master_flat_<FILT>.fits``, and (b) every such master is fresher than
    every raw flat frame. If any master is missing or stale, all flats are
    regenerated (partial reuse is not supported to avoid stale/fresh mixes).
    """
    log("\nStarting flat calibration.")

    # Attempt to reuse existing master flats before processing anything
    if cfg.reuse_masters:
        raw_flat_paths = files.files_filtered(imagetyp=cfg.headers.imagetyp_flat, include_path=True)
        if raw_flat_paths and _can_reuse_master_flats(
            raw_flat_paths, calibrated_data, cfg, log
        ):
            log("Reusing existing master flats.")
            return

    paths_by_filter = _process_flats(
        files, zero, combined_dark, intermediate_dir,
        cfg, log, cancel_event, reference_shape,
    )
    if paths_by_filter is None:
        return  # canceled
    if not paths_by_filter:
        raise ValueError(
            "No flat frames were successfully processed. Cannot create master flats."
        )

    _combine_flats(paths_by_filter, calibrated_data, cfg, log, cancel_event)


def _can_reuse_master_flats(raw_flat_paths, calibrated_data, cfg, log) -> bool:
    """
    Check whether every filter present in raw flats has a fresh master.

    Returns True only if a complete, up-to-date set of master flats exists
    covering every filter found in the raw flats. On any missing or stale
    master, returns False so the caller regenerates the whole set.

    :param raw_flat_paths: List of raw flat file paths
    :param calibrated_data: Output directory containing master flats
    :param cfg: ReductionConfig (for master_flat_pattern)
    :param log: Logging callable
    :return: True if all master flats can be reused
    """
    # Determine the set of distinct normalized filters in the raw flats
    filters_needed: set[str] = set()
    for raw_path in raw_flat_paths:
        try:
            with fits.open(raw_path) as hdul:
                filt = _normalize_filter(hdul[0].header.get("FILTER"))
                filters_needed.add(filt)
        except Exception:
            # A single unreadable raw flat forces regeneration
            return False

    # Every filter must have a fresh master
    for filt in filters_needed:
        master_path = calibrated_data / cfg.master_flat_pattern.format(filter=filt)
        if not _master_is_fresh(master_path, raw_flat_paths):
            log(
                f"Master flat for filter '{filt}' is missing or stale "
                f"(expected {master_path.name}). Regenerating all flats."
            )
            return False

    log(f"Found fresh master flats for filters: {sorted(filters_needed)}")
    return True


def _process_flats(files, zero, combined_dark, intermediate_dir,
                   cfg, log, cancel_event, reference_shape):
    """
    Preprocess individual flat frames and group their output paths by
    normalized filter name.

    :return: dict mapping normalized filter name → list of calibrated flat paths,
             or None if canceled
    :raises ValueError: If no flat frames are found in the input directory
    """
    flat_paths = files.files_filtered(imagetyp=cfg.headers.imagetyp_flat, include_path=True)
    n_total = len(flat_paths)

    if n_total == 0:
        raise ValueError(
            f"No FLAT frames found in '{files.location}'. "
            "Check that IMAGETYP headers are set to 'FLAT'."
        )
    log(f"Found {n_total} flat frame(s).")

    paths_by_filter: dict[str, list[str]] = {}
    n_failed = 0
    for n_done, flat_path in enumerate(flat_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return None

        file_name = Path(flat_path).name
        try:
            ccd = CCDData.read(flat_path, unit="adu", format="fits")
        except Exception as e:
            n_failed += 1
            log(f"Warning: failed to read {file_name}: {e}. Skipping.")
            continue

        try:
            filt = _normalize_filter(ccd.header.get("FILTER"))
        except ValueError as e:
            n_failed += 1
            log(f"Warning: {e} in {file_name}. Skipping.")
            continue

        log(f"Processing flat {n_done}/{n_total} [{filt}]: {file_name}")
        try:
            final_ccd = _reduce_flat(ccd, cfg, zero, combined_dark)
            _assert_shape_matches(final_ccd, reference_shape, f"flat frame {file_name}")

            new_fname = f"{file_name.split('.')[0]}.fits"
            output_path = intermediate_dir / new_fname
            write_image_only(final_ccd, output_path, overwrite=cfg.overwrite)
            add_header(intermediate_dir, new_fname, "FLAT", None, None, None, cfg)
        except Exception as e:
            n_failed += 1
            log(f"Warning: flat calibration failed for {file_name}: {e}. Skipping.")
            continue

        paths_by_filter.setdefault(filt, []).append(str(output_path))

    if n_failed:
        log(f"Warning: {n_failed}/{n_total} flat frame(s) failed to process.")
    log("\nFinished processing individual flat frames.")
    return paths_by_filter


def _combine_flats(paths_by_filter, calibrated_data, cfg, log, cancel_event):
    """
    Combine pre-processed flat frames per normalized filter into master flats.
    Warns when a filter has fewer than 3 frames.
    """
    log("\nStarting flat combination by filter.")
    n_filters = len(paths_by_filter)

    for n_done, (filt, flat_paths) in enumerate(paths_by_filter.items(), start=1):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        n_frames = len(flat_paths)
        log(f"Combining filter {n_done}/{n_filters}: {filt} ({n_frames} frame(s))")

        if n_frames < 3:
            log(
                f"Warning: only {n_frames} flat frame(s) for filter '{filt}'. "
                "Sigma clipping may be ineffective."
            )

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
        flat_file_name = cfg.master_flat_pattern.format(filter=filt)
        # Store the normalized name in the header too, so science lookup matches
        combined_flats.meta["FILTER"] = filt
        write_image_only(combined_flats, calibrated_data / flat_file_name, overwrite=cfg.overwrite)
        add_header(calibrated_data, flat_file_name, "FLAT", None, None, None, cfg)
        log(f"Master flat created: {flat_file_name}")

    log("\nFinished creating master flats by filter.")


def science_images(files, calibrated_data, zero, combined_dark,
                   cfg: ReductionConfig, log, cancel_event,
                   reference_shape, max_dark_exp):
    """
    Fully calibrate all science (LIGHT) frames: bias, dark, flat-field,
    and write BJD_TDB to the header when coordinates are present.

    Per-frame errors (missing header keys, missing master flat for the
    filter, read failures, dimension mismatches) are logged as warnings
    and the frame is skipped — they do not abort the entire stage.
    """
    science_imagetyp = cfg.headers.imagetyp_light

    # Build master-flat lookup by filename pattern rather than by IMAGETYP
    # header. This works for both pipeline-generated masters (which have
    # IMAGETYP=<flat> written by add_header) AND externally-produced masters
    # that may not have the same header conventions.
    flat_paths_by_filter = _discover_master_flats(calibrated_data, cfg)
    combined_flats: dict[str, CCDData] = {}
    for filt_key, master_path in flat_paths_by_filter.items():
        try:
            combined_flats[filt_key] = _load_master(master_path)
        except Exception as e:
            log(f"Warning: failed to load master flat {master_path.name}: {e}. Skipping.")
            continue

    if not combined_flats:
        raise RuntimeError(
            f"No master flats found in '{calibrated_data}' "
            f"(pattern '{cfg.master_flat_pattern}'). "
            "Science reduction cannot proceed."
        )
    log(f"Loaded master flats for filter(s): {sorted(combined_flats.keys())}")

    science_paths = files.files_filtered(imagetyp=science_imagetyp, include_path=True)
    n_total = len(science_paths)

    if n_total == 0:
        raise ValueError(
            f"No {science_imagetyp!r} frames found in '{files.location}'. "
            f"Check that IMAGETYP headers match cfg.headers.imagetyp_light."
        )
    log(f"\nFound {n_total} science frame(s). Starting reduction.")

    n_succeeded = 0
    n_failed = 0
    warned_long_exp = False

    for n_done, light_path in enumerate(science_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        file_name = Path(light_path).name

        # Read
        try:
            light = CCDData.read(light_path, unit="adu", format="fits")
        except Exception as e:
            n_failed += 1
            log(f"Warning: failed to read {file_name}: {e}. Skipping.")
            continue

        # EXPTIME validation (needed for dark scaling)
        if cfg.dark_bool:
            exp = _get_exposure_time(light.header)
            if exp is None:
                n_failed += 1
                log(
                    f"Warning: missing or invalid EXPTIME in {file_name} "
                    "and dark scaling requires it. Skipping."
                )
                continue
            if (max_dark_exp is not None and not warned_long_exp
                    and exp > max_dark_exp * DARK_SCALING_WARN_RATIO):
                log(
                    f"Warning: science EXPTIME={exp:.1f}s is more than "
                    f"{DARK_SCALING_WARN_RATIO:.0f}× the longest dark "
                    f"({max_dark_exp:.1f}s). Dark scaling may amplify noise. "
                    "Consider collecting longer darks. (This warning will not repeat.)"
                )
                warned_long_exp = True

        # Filter
        raw_filt = light.header.get("FILTER")
        try:
            filt = _normalize_filter(raw_filt)
        except ValueError as e:
            n_failed += 1
            log(f"Warning: {e} in {file_name}. Skipping.")
            continue

        # Matching master flat
        if filt not in combined_flats:
            n_failed += 1
            log(
                f"Warning: no master flat for filter '{filt}' "
                f"(needed by {file_name}, available: {sorted(combined_flats.keys())}). "
                "Skipping."
            )
            continue

        log(f"Calibrating science {n_done}/{n_total} [{filt}]: {file_name}")

        # Calibrate
        try:
            reduced = _reduce_science(light, cfg, zero, combined_dark, combined_flats[filt])
            _assert_shape_matches(reduced, reference_shape, f"science frame {file_name}")
        except Exception as e:
            n_failed += 1
            log(f"Warning: calibration failed for {file_name}: {e}. Skipping.")
            continue

        new_fname = f"{file_name.split('.')[0]}.fits"
        try:
            write_image_only(reduced, calibrated_data / new_fname, overwrite=cfg.overwrite)
        except Exception as e:
            n_failed += 1
            log(f"Warning: failed to write {new_fname}: {e}. Skipping.")
            continue

        # Header + BJD_TDB (guarded against missing coords/time).
        # Different capture software writes these under different keys —
        # e.g. BSUO frames commonly use HJD_UTC and OBJCTRA / OBJCT DEC
        # instead of the FITS-convention JD-HELIO / RA / DEC.
        hjd = _header_get_any(light.header, "JD-HELIO", "HJD_UTC", "HJD-UTC", "HJD")
        ra = _header_get_any(light.header, "RA", "OBJCTRA", "OBJCT RA", "RA-OBJ")
        dec = _header_get_any(light.header, "DEC", "OBJCTDEC", "OBJCT DEC", "DEC-OBJ")
        if hjd is None or ra is None or dec is None:
            missing = [
                name for name, val in
                (("HJD", hjd), ("RA", ra), ("DEC", dec))
                if val is None
            ]
            log(
                f"Warning: missing {'/'.join(missing)} in {file_name}. "
                "Writing calibrated image without BJD_TDB."
            )
            add_header(calibrated_data, new_fname, science_imagetyp, None, None, None, cfg)
        else:
            try:
                add_header(calibrated_data, new_fname, science_imagetyp, hjd, ra, dec, cfg)
            except Exception as e:
                log(
                    f"Warning: BJD_TDB calculation failed for {file_name}: {e}. "
                    "Writing calibrated image without BJD_TDB."
                )
                add_header(calibrated_data, new_fname, science_imagetyp, None, None, None, cfg)

        n_succeeded += 1

    log(f"\nScience reduction summary: {n_succeeded} succeeded, {n_failed} skipped.")
    if n_succeeded == 0:
        raise RuntimeError(
            f"All {n_total} science frames were skipped. "
            "Check header keywords and master flat filters."
        )


# ---------------------------------------------------------------------------
# Header utilities
# ---------------------------------------------------------------------------

def add_header(pathway, fname, imagetyp, hjd, ra, dec, cfg: ReductionConfig):
    """
    Write reduction metadata into a FITS header.

    For LIGHT frames with all of hjd/ra/dec present, the HJD is converted
    to BJD_TDB and stored. If any of hjd/ra/dec is None, BJD_TDB is
    silently skipped — the caller is responsible for logging.

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

    if imagetyp == "LIGHT" and hjd is not None and ra is not None and dec is not None:
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
