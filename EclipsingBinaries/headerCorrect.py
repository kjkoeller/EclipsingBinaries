"""
FITS header correction utilities for IRAF compatibility and reproducible
photometric reductions.

This module is a refactor of Robert Berrington's standalone header-correct.py
script (rberring@bsu.edu, 08/09/2024). All calculation logic is preserved
verbatim; the structural changes are:

    1. CLI argument parsing is replaced with a HeaderCorrectionOptions
       dataclass plus an ObservatoryRegistry, both populated from the
       pipeline's ReductionConfig (or constructed directly in scripts).
    2. The single monolithic per-file loop is split into focused helpers
       so individual corrections can be enabled or skipped via flags.
    3. ``print()`` debug output is routed through a ``log`` callback so the
       same code drives both the GUI's log pane and the command line.
    4. There is a single public entry point ``correct_headers(image_path,
       cfg, ...)`` returning a small report object describing what changed.

Author:  Kyle Koeller (refactor) - 04/24/2026
Original: Robert Berrington (BSU) - 08/09/2024
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Dict
import re

from astropy.io import fits
from astropy.time import Time
from astropy import coordinates as coords
from astropy import units as u


# ---------------------------------------------------------------------------
# Observatory registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservatorySite:

    """A single observatory site, either by astropy name or explicit coordinates.

    If ``astropy_name`` is set, ``EarthLocation.of_site(astropy_name)`` is
    used. Otherwise ``lat``, ``lon``, ``altitude_m``, and ``ellipsoid`` are
    used to build the EarthLocation. Exactly one of the two must be
    populated.
    """

    name: str
    astropy_name: Optional[str] = None
    lat: Optional[str] = None         # sexagesimal degrees, e.g. "40:11:59.7"
    lon: Optional[str] = None         # sexagesimal degrees, signed; +E -W
    altitude_m: Optional[float] = None
    ellipsoid: str = "WGS84"
    timezone: Optional[int] = None    # hours from UTC; informational only

    def to_earth_location(self) -> coords.EarthLocation:
        if self.astropy_name is not None:
            return coords.EarthLocation.of_site(self.astropy_name)
        if self.lat is None or self.lon is None or self.altitude_m is None:
            raise ValueError(
                f"ObservatorySite {self.name!r} has neither an astropy_name "
                f"nor a complete lat/lon/altitude triple."
            )
        return coords.EarthLocation.from_geodetic(
            lon=coords.Angle(f"{self.lon} degrees"),
            lat=coords.Angle(f"{self.lat} degrees"),
            height=self.altitude_m * u.m,
            ellipsoid=self.ellipsoid,
        )


class ObservatoryRegistry:

    """Lookup table mapping site keys (case-insensitive) to ObservatorySite entries.

    Populated with the BSU/SARA/SFRO defaults from the original
    ``header-correct.py`` script. Custom sites can be registered at
    runtime via :meth:`register`.
    """

    # Defaults — copied verbatim from header-correct.py so behaviour is
    # bit-for-bit identical for these sites.
    _DEFAULT_SITES = (
        ObservatorySite(
            name="BSUO",
            lat="40:11:59.7", lon="-85:24:41.9",
            altitude_m=322.8, ellipsoid="WGS84", timezone=-5,
        ),
        ObservatorySite(
            name="BSU",
            lat="40:11:59.61", lon="-85:24:40.62",
            altitude_m=304.5, ellipsoid="WGS84", timezone=-5,
        ),
        ObservatorySite(
            name="SFRO",
            lat="31:32:49.5", lon="-99:22:56.0",
            altitude_m=464.6, ellipsoid="WGS84", timezone=-6,
        ),
        ObservatorySite(name="SARA-KP", astropy_name="kpno"),
        ObservatorySite(name="SARA-N", astropy_name="kpno"),
        ObservatorySite(name="SARA-CT", astropy_name="ctio"),
        ObservatorySite(name="SARA-S", astropy_name="ctio"),
        ObservatorySite(name="SARA-RM", astropy_name="Roque de los Muchachos"),
    )

    def __init__(self, sites=None):
        """Initialise the registry, optionally seeding with custom sites.

        :param sites: Iterable of :class:`ObservatorySite` to register.
            Defaults to the BSU/BSUO/SARA-*/SFRO set if omitted.
        """
        self._sites: Dict[str, ObservatorySite] = {}
        for site in (sites if sites is not None else self._DEFAULT_SITES):
            self.register(site)

    def register(self, site: ObservatorySite) -> None:
        """Add or replace a site. Lookup is case-insensitive."""
        self._sites[site.name.upper()] = site

    def get(self, name: str) -> Optional[ObservatorySite]:
        """Look up a site by name (case-insensitive). Returns None if absent."""
        if name is None:
            return None
        return self._sites.get(name.upper())

    def resolve(self, name: str, default: Optional[str] = "BSU",
                log: Callable[[str], None] = print) -> coords.EarthLocation:
        """
        Resolve a site name to an EarthLocation, falling back to ``default``
        if the name is not registered. Mirrors the original script's
        "default to BSU on unknown" behaviour.
        """
        site = self.get(name)
        if site is None:
            log(
                f"WARNING: Unknown observatory location {name!r}. "
                f"Defaulting to {default!r}."
            )
            site = self.get(default)
            if site is None:
                raise ValueError(
                    f"Default observatory {default!r} is not registered."
                )
        return site.to_earth_location()


# Module-level default registry. Callers can pass their own to correct_headers.
DEFAULT_REGISTRY = ObservatoryRegistry()


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class HeaderCorrectionOptions:

    """What to compute and how.

    Mirrors the original CLI flags but with sane defaults matching
    ``header-correct.py`` (--JD --HJD --BJD --eairmass --sidereal all on;
    --filter_parse off).
    """

    do_jd: bool = True                  # JD_START / JD_MID / JD_END / JD
    do_hjd: bool = True                 # HJD / HJD_UTC at mid-exposure
    do_bjd: bool = True                 # BJD_UTC and BJD_TDB at mid-exposure
    do_sidereal: bool = True            # SIDEREAL / MEAN_ST / APP_ST / ST
    do_eairmass: bool = True            # EAIRMASS / SECZ
    do_filter_parse: bool = False       # space -> underscore in FILTER
    do_radec_format: bool = True        # rewrite RA/DEC into IRAF sexagesimal

    # Defaults written when the keyword is missing from the FITS header.
    default_epoch: float = 2000.0
    default_equinox: str = "J2000.0"
    default_observatory: str = "BSU"

    # Sexagesimal delimiter for RA/DEC strings.
    delimiter: str = ":"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class HeaderCorrectionReport:

    """Tracks what each call to correct_headers() actually did."""

    file: Path
    observatory: Optional[str] = None
    wrote_keys: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    def note(self, key: str) -> None:
        """Record that ``key`` was written to the FITS header."""
        self.wrote_keys.append(key)

    def warn(self, msg: str) -> None:
        """Record a warning encountered during correction."""
        self.warnings.append(msg)

    def skip(self, what: str) -> None:
        """Record that a section was skipped (e.g. flag disabled)."""
        self.skipped.append(what)


# ---------------------------------------------------------------------------
# Helpers (each owns one section of the original script)
# ---------------------------------------------------------------------------

def _ensure_epoch_equinox(header, opts: HeaderCorrectionOptions, report) -> tuple:
    """Mirror lines 285-295 of header-correct.py."""
    if "EPOCH" in header:
        image_epoch = header["EPOCH"]
    else:
        image_epoch = opts.default_epoch
        header["EPOCH"] = (image_epoch, "Epoch of image coordinates")
        report.note("EPOCH")

    if "EQUINOX" in header:
        image_equinox = header["EQUINOX"]
    else:
        image_equinox = opts.default_equinox
        header["EQUINOX"] = (image_equinox, "Equinox of image coordinates")
        report.note("EQUINOX")

    return image_epoch, image_equinox


def _parse_filter(header, report, log) -> None:
    """Mirror lines 297-327 of header-correct.py."""
    # Pick a filter source the same way the original script does.
    if "FILT_ORG" in header:
        log("Filter already corrected; using FILT_ORG.")
        filter_image = header["FILT_ORG"]
    elif "FILTER" in header:
        filter_image = header["FILTER"]
    elif "FILTERS" in header:
        filter_image = header["FILTERS"]
    else:
        report.warn("No recognized FILTER keyword present in header.")
        return

    # Only rewrite if we haven't already and there's a space to replace.
    if "FILT_ORG" in header:
        return
    if " " not in filter_image:
        return

    header["FILT_ORG"] = (filter_image, "original format of image FILTER keyword")
    reformatted = re.sub(r" ", "_", filter_image, count=2)
    header["FILTER"] = (reformatted, "Image filter")
    header["FILTERS"] = (reformatted, "Image filter")
    report.note("FILTER")
    report.note("FILT_ORG")
    log(f"FILTER changed: {filter_image!r} -> {reformatted!r}")


def _format_radec(header, opts: HeaderCorrectionOptions, report, log) -> None:
    """
    Mirror lines 329-412 of header-correct.py.

    Reads RA / DEC (preferring OBJCTRA / OBJCTDEC if present) and rewrites
    the RA and DEC keywords in IRAF format (HH:MM:SS.s, +DD:MM:SS.s).
    The original values are preserved as RA_ORIG / DEC_ORIG (only on the
    first run; subsequent runs leave the originals untouched).
    """
    # --- RA ---
    ra_image_angle = None
    if "OBJCTRA" in header:
        if "RA_ORIG" not in header and "RA" in header:
            header["RA_ORIG"] = (header["RA"],
                                 "original format of image RA coordinate")
            report.note("RA_ORIG")
        ra_image_angle = coords.Angle(header["OBJCTRA"], u.hour)
    elif "RA" in header:
        if "RA_ORIG" not in header:
            header["RA_ORIG"] = (header["RA"],
                                 "original format of image RA coordinate")
            report.note("RA_ORIG")
        ra_raw = header["RA"]
        if isinstance(ra_raw, str):
            ra_image_angle = coords.Angle(ra_raw, u.hour)
        else:
            ra_image_angle = coords.Angle(ra_raw, u.degree)
    else:
        report.warn("RA or OBJCTRA keyword header does not exist.")

    # --- DEC ---
    dec_image_angle = None
    if "OBJCTDEC" in header:
        if "DEC_ORIG" not in header and "DEC" in header:
            header["DEC_ORIG"] = (header["DEC"],
                                  "original format of image Dec coordinate")
            report.note("DEC_ORIG")
        dec_image_angle = coords.Angle(header["OBJCTDEC"], u.degree)
    elif "DEC" in header:
        if "DEC_ORIG" not in header:
            header["DEC_ORIG"] = (header["DEC"],
                                  "original format of image Dec coordinate")
            report.note("DEC_ORIG")
        dec_image_angle = coords.Angle(header["DEC"], u.degree)
    else:
        report.warn("DEC or OBJCTDEC keyword header does not exist.")

    # Rewrite RA/DEC into IRAF-style sexagesimal.
    if ra_image_angle is not None:
        ra_str = coords.Angle.to_string(
            ra_image_angle, u.hour, sep=opts.delimiter, pad=True
        )
        header["RA"] = (ra_str, "RA of target in correct IRAF format")
        report.note("RA")

    if dec_image_angle is not None:
        dec_str = coords.Angle.to_string(
            dec_image_angle, u.degree, sep=opts.delimiter,
            alwayssign=True, pad=True,
        )
        header["DEC"] = (dec_str, "Dec of target in correct IRAF format")
        report.note("DEC")


def _resolve_observatory(header, opts: HeaderCorrectionOptions,
                         registry, report, log):
    """
    Mirror lines 426-485 of header-correct.py. Returns EarthLocation.

    ``registry`` may be either an :class:`ObservatoryRegistry` (this module)
    or any object exposing ``.get(name) -> EarthLocation | None``. This lets
    the data-reduction pipeline pass its own registry without having to
    rebuild it.
    """
    if "OBSERVAT" in header:
        observer_at = header["OBSERVAT"]
    else:
        observer_at = opts.default_observatory
        header["OBSERVAT"] = observer_at
        report.note("OBSERVAT")
        report.warn(
            f"OBSERVAT keyword was missing; defaulted to {observer_at!r}."
        )

    # Two registry styles supported:
    #   (a) ObservatoryRegistry from this module -- has .resolve()
    #   (b) generic object with .get() returning an EarthLocation or None
    location = None
    if hasattr(registry, "resolve"):
        location = registry.resolve(
            observer_at, default=opts.default_observatory, log=log
        )
    else:
        location = registry.get(observer_at)
        if location is None and opts.default_observatory:
            log(
                f"WARNING: Unknown observatory location {observer_at!r}. "
                f"Falling back to {opts.default_observatory!r}."
            )
            location = registry.get(opts.default_observatory)
        if location is None:
            raise ValueError(
                f"Observatory {observer_at!r} (and default "
                f"{opts.default_observatory!r}) not found in registry."
            )

    report.observatory = observer_at
    return location


def _exposure_time(header, log) -> u.Quantity:
    """Mirror lines 493-501 of header-correct.py."""
    if "EXPTIME" in header:
        return header["EXPTIME"] * u.second
    if "EXP_TIME" in header:
        return header["EXP_TIME"] * u.second
    if "EXPOSURE" in header:
        return header["EXPOSURE"] * u.second
    log("WARNING: no exposure time set. Assuming exposure time = 0 sec.")
    return 0 * u.second


def _write_jd_suite(header, date, date_mid, date_end, report, log) -> None:
    """Mirror lines 523-552 of header-correct.py."""
    if "JD_START" in header:
        jd_original = header["JD_START"]
    elif "JD" in header:
        jd_original = header["JD"]
        if "JD_ORIG" not in header:
            header["JD_ORIG"] = (jd_original, "Original JD value in image.")
            report.note("JD_ORIG")
    else:
        jd_original = date.jd
        if "JD_ORIG" not in header:
            header["JD_ORIG"] = (date.jd, "Original JD from header.")
            report.note("JD_ORIG")

    header["JD"]       = (date_mid.jd,  "Julian Date at mid exposure.")
    header["JD_MID"]   = (date_mid.jd,  "Julian Date at mid exposure.")
    header["JD_START"] = (date.jd,      "Julian Date at exposure start.")
    header["JD_END"]   = (date_end.jd,  "Julian Date at exposure end.")
    for k in ("JD", "JD_MID", "JD_START", "JD_END"):
        report.note(k)


def _build_target(header, image_epoch, image_equinox, date_mid):
    """Mirror lines 563-574 of header-correct.py."""
    if image_epoch == 2000.0 or image_equinox == "J2000.0":
        return coords.SkyCoord(
            header["RA"], header["DEC"],
            unit=(u.hourangle, u.deg),
            obstime=date_mid, frame="icrs",
        )
    return coords.SkyCoord(
        header["RA"], header["DEC"],
        unit=(u.hourangle, u.deg),
        obstime=date_mid, frame=image_equinox,
    )


def _write_hjd(header, date, date_mid, target, report, log) -> Optional[Time]:
    """Mirror lines 576-605 of header-correct.py. Returns the new HJD."""
    hjd_correction = date_mid.light_travel_time(target, "heliocentric")

    # Preserve the original HJD on the first run.
    if "HJD" in header:
        if "HJD_ORIG" not in header:
            header["HJD_ORIG"] = (header["HJD"],
                                  "Original HJD value in header.")
            report.note("HJD_ORIG")
    else:
        log("HJD did not exist in header; computing from exposure start.")
        hjd_orig = date + date.light_travel_time(target, "heliocentric")
        if "HJD_ORIG" not in header:
            header["HJD_ORIG"] = (hjd_orig.jd, "HJD value from exp start.")
            report.note("HJD_ORIG")

    hjd = date_mid + hjd_correction
    header["HJD"]     = (hjd.jd, "HJD_UTC at mid exposure")
    header["HJD_UTC"] = (hjd.jd, "HJD_UTC at mid exposure")
    report.note("HJD")
    report.note("HJD_UTC")
    return hjd


def _write_bjd(header, date_mid, target, report, log) -> None:
    """Mirror lines 616-646 of header-correct.py."""
    bjd_correction = date_mid.light_travel_time(target)
    bjd_utc = date_mid.utc + bjd_correction
    bjd_tdb = date_mid.tdb + bjd_correction

    header["BJD_UTC"] = (bjd_utc.jd, "BJD_UTC at mid exposure")
    header["BJD_TDB"] = (bjd_tdb.jd, "BJD_TDB at mid exposure")
    report.note("BJD_UTC")
    report.note("BJD_TDB")


def _write_sidereal(header, date_mid, location, report, log) -> None:
    """Mirror lines 657-685 of header-correct.py."""
    if "ST_ORIG" in header:
        sidereal_orig = header["ST_ORIG"]
    elif "SIDEREAL" in header:
        sidereal_orig = header["SIDEREAL"]
    elif "ST" in header:
        sidereal_orig = header["ST"]
    else:
        sidereal_orig = None

    mean_st = Time.sidereal_time(date_mid, kind="mean",
                                 longitude=location, model="IAU2006")
    apparent_st = Time.sidereal_time(date_mid, kind="apparent",
                                     longitude=location, model="IAU2006A")

    if sidereal_orig is not None:
        header["ST_ORIG"] = (sidereal_orig,
                             "Original sidereal time at exp start.")
        report.note("ST_ORIG")

    header["SIDEREAL"] = (apparent_st.to_string(sep=":"),
                          "Local app sidereal time at exp midpt [IAU2006A]")
    header["MEAN_ST"]  = (mean_st.to_string(sep=":"),
                          "local mean sidereal time at exp midpt [IAU2006]")
    header["APP_ST"]   = (apparent_st.to_string(sep=":"),
                          "local app sidereal time at exp midpt [IAU2006A]")
    header["ST"]       = (apparent_st.to_string(sep=":"),
                          "local app sidereal time at exp midpt [IAU2006A]")
    for k in ("SIDEREAL", "MEAN_ST", "APP_ST", "ST"):
        report.note(k)


def _write_eairmass(header, date_mid, target, location, report, log) -> None:
    """Mirror lines 693-726 of header-correct.py (the IRAF setairmass formula)."""
    altaz = target.transform_to(
        coords.AltAz(obstime=date_mid, location=location)
    )
    secz = altaz.secz
    s = secz - 1.0
    eairmass = (
        secz
        - 0.0018167 * s
        - 0.002875  * s * s
        - 0.0008083 * s * s * s
    )

    if "SECZ" in header:
        secz_orig = header["SECZ"]
        header["SECZ_ORG"] = (float(secz_orig),
                              "Orignial value of SecZ in image header.")
        report.note("SECZ_ORG")
    header["SECZ"] = (float(secz),
                      "SecZ for airmass estimation at exp midpt.")
    header["EAIRMASS"] = (float(eairmass),
                          "Effective Airmass for exposure.")
    report.note("SECZ")
    report.note("EAIRMASS")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _safe_call(report, label, fn, *args, **kwargs):
    """
    Run a header-writing helper, reporting any failure rather than raising.

    Centralises the try/except/skip/warn pattern that repeats for every
    toggleable correction. Keeps the caller focused on *what* to compute,
    not on how to report when it fails.
    """
    try:
        return fn(*args, **kwargs)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        # We catch a deliberately broad-but-not-bare set of exceptions:
        # malformed coordinates, missing keys, bad header values. We never
        # catch e.g. KeyboardInterrupt or SystemExit.
        report.warn(f"{label} failed: {e}")
        return None


def _read_date_obs(header, location):
    """Build the DATE-OBS Time anchor (returns None if header lacks DATE-OBS)."""
    if "DATE-OBS" not in header:
        return None
    if location is not None:
        return Time(header["DATE-OBS"], scale="utc",
                    format="fits", location=location)
    return Time(header["DATE-OBS"], scale="utc", format="fits")


def _maybe_build_target(opts, header, image_epoch, image_equinox, date_mid, report):
    """Construct a SkyCoord target if any time-correction needs it."""
    if not (opts.do_hjd or opts.do_bjd):
        return None
    try:
        return _build_target(header, image_epoch, image_equinox, date_mid)
    except (ValueError, TypeError, KeyError) as e:
        report.warn(f"Could not build SkyCoord for HJD/BJD: {e}")
        return None


def _apply_time_corrections(opts, header, date, date_mid, date_end,
                            target, location, report, log):
    """
    Run each toggleable time/coord-derived correction. Each one is gated
    by its own opts flag and wrapped in _safe_call so a single failure
    doesn't prevent the others from running.
    """
    if opts.do_jd:
        _safe_call(report, "JD",
                   _write_jd_suite, header, date, date_mid, date_end, report, log)
    else:
        report.skip("JD")

    if opts.do_hjd:
        if target is not None:
            _safe_call(report, "HJD",
                       _write_hjd, header, date, date_mid, target, report, log)
    else:
        report.skip("HJD")

    if opts.do_bjd:
        if target is not None:
            _safe_call(report, "BJD",
                       _write_bjd, header, date_mid, target, report, log)
    else:
        report.skip("BJD")

    if opts.do_sidereal:
        if location is not None:
            _safe_call(report, "Sidereal",
                       _write_sidereal, header, date_mid, location, report, log)
    else:
        report.skip("sidereal")

    if opts.do_eairmass:
        if target is not None and location is not None:
            _safe_call(report, "Effective airmass",
                       _write_eairmass, header, date_mid, target, location, report, log)
    else:
        report.skip("eairmass")


def correct_headers(
    image_path,
    opts: Optional[HeaderCorrectionOptions] = None,
    registry=None,
    log: Callable[[str], None] = print,
) -> HeaderCorrectionReport:
    """
    Apply header corrections in-place to a single FITS file.

    Replicates the entire per-file body of header-correct.py — sections are
    enabled or skipped via flags on ``opts``. The file is opened in update
    mode and closed before this function returns.

    :param image_path: Path to the FITS file
    :param opts: Which corrections to run (defaults to all on)
    :param registry: Observatory lookup. Either an :class:`ObservatoryRegistry`
        from this module or any object with a ``.get(name)`` method returning
        an :class:`astropy.coordinates.EarthLocation` (or None). Defaults to
        the BSU/SARA/SFRO :data:`DEFAULT_REGISTRY`.
    :param log: Callback for diagnostic output (defaults to print)
    :return: HeaderCorrectionReport summarising the changes
    """
    image_path = Path(image_path)
    opts = opts or HeaderCorrectionOptions()
    registry = registry or DEFAULT_REGISTRY
    report = HeaderCorrectionReport(file=image_path)

    with fits.open(str(image_path), mode="update") as hdul:
        header = hdul[0].header

        image_epoch, image_equinox = _ensure_epoch_equinox(header, opts, report)

        if opts.do_filter_parse:
            _parse_filter(header, report, log)
        else:
            report.skip("filter_parse")

        if opts.do_radec_format:
            _format_radec(header, opts, report, log)

        # Everything below needs a time anchor (DATE-OBS) and, for some
        # corrections, an observer location.
        need_observer = (
            opts.do_hjd or opts.do_bjd or opts.do_sidereal or opts.do_eairmass
        )
        location = _resolve_observatory(header, opts, registry, report, log) \
            if need_observer else None

        date = _read_date_obs(header, location)
        if date is None:
            report.warn(
                "DATE-OBS missing; cannot compute JD/HJD/BJD/sidereal/airmass."
            )
            return report

        exp_time = _exposure_time(header, log)
        date_mid = date + exp_time / 2.0
        date_end = date + exp_time

        target = _maybe_build_target(
            opts, header, image_epoch, image_equinox, date_mid, report,
        )

        _apply_time_corrections(
            opts, header, date, date_mid, date_end,
            target, location, report, log,
        )

    return report


def correct_headers_batch(
    image_paths,
    opts: Optional[HeaderCorrectionOptions] = None,
    registry=None,
    log: Callable[[str], None] = print,
):
    """
    Apply correct_headers() to a list of FITS files. Errors on any single
    file are caught, logged, and recorded in that file's report rather than
    aborting the batch.
    """
    opts = opts or HeaderCorrectionOptions()
    registry = registry or DEFAULT_REGISTRY
    reports = []
    for path in image_paths:
        try:
            reports.append(correct_headers(path, opts, registry, log))
        except Exception as e:
            r = HeaderCorrectionReport(file=Path(path))
            r.warn(f"correct_headers raised: {type(e).__name__}: {e}")
            log(f"WARNING: header correction failed for {path}: {e}")
            reports.append(r)
    return reports