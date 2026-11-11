# -*- coding: utf-8 -*-
"""
Download what the public archives hold on one target, from X-ray to radio.

Each database is one entry in ARCHIVES: light curves (ZTF, Gaia, TESS, Kepler,
K2, NEOWISE), catalogue magnitudes from the ultraviolet to the infrared,
spectra, and X-ray and radio detections. download() runs the chosen ones for
a target and puts everything in one folder with a summary file, so a missing
or unreachable database never stops the others.

Light curves are also saved as plain three-column text files (time,
magnitude, error), the format Period Search, Find Minimum and the other
programs read. The time system differs between surveys and is given in the
summary: don't mix files from different surveys in an O-C study without
converting them first.

To add a database, write a function that takes (target, folder, run) and
returns a Result, and add an Archive entry for it. A catalogue that VizieR
hosts only needs the entry.

Author: Kyle Koeller
Created: 11/09/2026
Last Edited: 11/09/2026
"""

import argparse
import contextlib
import io
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from ._log import make_logger
from ._vizier import VizierServers, VizierUnavailable, query_region_tables

LIGHT_CURVES = "Light curves"
MAGNITUDES = "Catalogue magnitudes"
SPECTRA = "Spectra"
HIGH_ENERGY = "X-ray and radio"
KINDS = (LIGHT_CURVES, MAGNITUDES, SPECTRA, HIGH_ENERGY)

SUMMARY_FILE = "archive_summary.csv"
ZTF_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
# Seconds to wait for a connection, then for each piece of the answer
TIMEOUT = (10, 120)
# ZTF images saturate around this magnitude, so brighter stars get few usable points
ZTF_SATURATION = 13.0
# Seconds a SIMBAD query may take before the next server is tried
SIMBAD_TIMEOUT = 60
# Light curve products from these MAST pipelines are downloaded; MAST holds many
# other community light curves for the same stars, which would repeat the data
MAST_PIPELINES = ("SPOC", "TESS-SPOC", "QLP", "Kepler", "K2")

# Gaia's DataLink download unpacks into the current folder, so it's moved into
# the output folder for the length of the call, one caller at a time
_folder_lock = threading.Lock()


class ArchiveError(ValueError):
    """Something the user can fix, with a message saying what."""


@dataclass
class Target:
    """The star being looked up: a name for the folder and its J2000 position."""
    name: str
    coord: SkyCoord

    @property
    def label(self):
        """The name with anything a file system could object to replaced."""
        text = re.sub(r"[^A-Za-z0-9+.-]+", "_", self.name).strip("_")
        return text or "target"


@dataclass
class Result:
    """What one database returned."""
    key: str
    name: str
    kind: str
    status: str                 # "found", "nothing", "failed" or "skipped"
    rows: int = 0
    files: list = field(default_factory=list)
    note: str = ""


@dataclass
class Run:
    """Shared by every database in one download: the log, the cancel flag and a scratch cache."""
    log: object
    cancel_event: object = None
    cache: dict = field(default_factory=dict)

    def cancelled(self):
        return self.cancel_event is not None and self.cancel_event.is_set()


@dataclass(frozen=True)
class Archive:
    """One database: what it is, and the function that fetches from it."""
    key: str
    name: str
    kind: str
    band: str                   # the part of the spectrum, for the panel and the summary
    about: str
    fetch: object


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------

def resolve_target(name="", ra="", dec=""):
    """
    Work out the target's position.

    With ra and dec (J2000, RA in hours as HH:MM:SS or decimal hours, Dec in
    degrees) those are used and the name is only a label. With a name alone,
    the position is looked up through the CDS name resolver, which knows
    SIMBAD, NED and VizieR names.
    """
    name, ra, dec = str(name or "").strip(), str(ra or "").strip(), str(dec or "").strip()
    if ra and dec:
        try:
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
        except (ValueError, TypeError) as error:
            raise ArchiveError(f"The RA and Dec couldn't be read: {error}") from error
        if not name:
            name = coord.to_string("hmsdms", sep="", precision=1).replace(" ", "")
        return Target(name, coord)
    if ra or dec:
        raise ArchiveError("Enter both RA and Dec, or leave both empty to look the name up.")
    if not name:
        raise ArchiveError("Enter a target name, or its RA and Dec.")
    try:
        coord = SkyCoord.from_name(name, cache=False)
    except NameResolveError as error:
        raise ArchiveError(f"'{name}' couldn't be looked up ({error}). Check the spelling, or enter "
                           "the RA and Dec instead.") from error
    return Target(name, coord.icrs)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _first_line(error):
    """An error's first line, or its type when it has no message."""
    text = str(error).strip() if error is not None else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else type(error).__name__


def _save_table(table, path):
    """Write an astropy Table as CSV. Columns holding arrays are saved by astropy instead of pandas."""
    path = Path(path)
    try:
        table.to_pandas().to_csv(path, index=False)
    except (ValueError, TypeError):
        table.write(path, format="ascii.ecsv", overwrite=True)
    return path


def _column(frame, *names):
    """The first of these columns the table has, matched without regard to case, or None."""
    lower = {str(c).lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def save_light_curve(path, time, mag, err):
    """
    Save a light curve as the three-column text file the other programs read
    (time, magnitude, error), sorted by time and without rows that have gaps.
    Returns the number of points written, and writes nothing if there are none.
    """
    data = np.column_stack([np.asarray(time, dtype=float), np.asarray(mag, dtype=float),
                            np.asarray(err, dtype=float)])
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) == 0:
        return 0
    data = data[np.argsort(data[:, 0], kind="stable")]
    np.savetxt(path, data, fmt=["%.6f", "%.4f", "%.4f"], delimiter="\t")
    return len(data)


# ---------------------------------------------------------------------------
# Catalogues on VizieR
# ---------------------------------------------------------------------------

def _vizier_tables(catalog, target, run, radius_arcsec):
    """
    Every table of a VizieR catalogue with rows near the target, sharing the
    run's record of which servers answer. Raises OSError if none can be reached.
    """
    servers = run.cache.setdefault("vizier_servers", VizierServers())
    if servers.all_unreachable:
        raise OSError("No VizieR server could be reached in this run; try again later.")
    try:
        tables = query_region_tables(catalog, target.coord, columns=["*", "+_r"],
                                     radius=radius_arcsec * u.arcsec, servers=servers,
                                     write_callback=run.log, cancel_event=run.cancel_event)
    except VizierUnavailable as error:
        raise OSError(str(error)) from error
    return {name: table for name, table in (tables or {}).items() if len(table)}


def _vizier(catalog, radius_arcsec, note=""):
    """
    A fetch function for a catalogue VizieR hosts. catalog is one table or a
    whole catalogue; every table with rows near the target is saved, nearest
    row first, with VizieR's default columns.
    """
    def fetch(target, folder, run, key):
        # Shared by every VizieR catalogue in the run, so a server that answered is tried
        # first and one that didn't respond isn't waited on again
        servers = run.cache.setdefault("vizier_servers", VizierServers())
        if servers.all_unreachable:
            return "skipped", 0, [], "No VizieR server could be reached in this run; try again later."
        try:
            tables = query_region_tables(catalog, target.coord, columns=["*", "+_r"],
                                         radius=radius_arcsec * u.arcsec, servers=servers,
                                         write_callback=run.log, cancel_event=run.cancel_event)
        except VizierUnavailable as error:
            raise OSError(str(error)) from error
        if not tables:
            return "nothing", 0, [], f"No source within {radius_arcsec:g} arcsec."
        files, rows = [], 0
        for table_name, table in tables.items():
            if len(table) == 0:
                continue
            suffix = "" if len(tables) == 1 else "_" + re.sub(r"[^A-Za-z0-9]+", "_", table_name.split("/")[-1])
            files.append(_save_table(table, folder / f"{key}{suffix}.csv"))
            rows += len(table)
        if not files:
            return "nothing", 0, [], f"No source within {radius_arcsec:g} arcsec."
        return "found", rows, files, note or f"Sources within {radius_arcsec:g} arcsec, nearest first."
    return fetch


def _with_backup(first, second, first_name, second_name):
    """
    A fetch function that uses first, and second only if first raises an error.
    An answer of "nothing here" from first is taken as it is.
    """
    def fetch(target, folder, run, key):
        try:
            return first(target, folder, run, key)
        except Exception as first_error:  # whatever first ran into, second may still work
            run.log(f"{first_name} didn't work ({_first_line(first_error)}); trying {second_name}.")
            try:
                status, rows, files, note = second(target, folder, run, key)
            except Exception as second_error:
                raise OSError(f"{first_name}: {_first_line(first_error)} {second_name}: "
                              f"{_first_line(second_error)}") from second_error
            return status, rows, files, f"From {second_name}, as {first_name} gave an error. {note}"
    return fetch


PANSTARRS_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"


def _fetch_panstarrs(target, folder, run, key, radius_arcsec=3):
    """
    Pan-STARRS1 DR2 mean photometry from MAST's catalogue service. astroquery's
    MAST query couldn't read this table (it fails on the text "None" in empty
    columns), so the CSV is read directly; pandas takes "None" as missing.
    """
    response = requests.get(PANSTARRS_URL, timeout=TIMEOUT, params={
        "ra": f"{target.coord.ra.deg:.7f}", "dec": f"{target.coord.dec.deg:.7f}",
        "radius": f"{radius_arcsec / 3600:.10f}", "nDetections.gte": 1, "pagesize": 5001})
    response.raise_for_status()
    try:
        frame = pd.read_csv(io.StringIO(response.text))
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    except pd.errors.ParserError as error:
        raise OSError(f"MAST's Pan-STARRS answer wasn't a table: {error}") from error
    ra, dec = _column(frame, "raMean"), _column(frame, "decMean")
    if len(frame) == 0 or ra is None or dec is None:
        return "nothing", 0, [], f"No source within {radius_arcsec:g} arcsec."
    found = SkyCoord(pd.to_numeric(frame[ra], errors="coerce").to_numpy() * u.deg,
                     pd.to_numeric(frame[dec], errors="coerce").to_numpy() * u.deg)
    frame.insert(0, "distance_arcsec", target.coord.separation(found).arcsec.round(3))
    frame = frame.sort_values("distance_arcsec", kind="stable")
    path = folder / f"{key}.csv"
    frame.to_csv(path, index=False)
    return "found", len(frame), [path], (f"Sources within {radius_arcsec:g} arcsec from MAST, nearest first. "
                                         "Missing magnitudes are -999.")


def _mast_catalog(catalog, radius_arcsec, **criteria):
    """
    A fetch function for a catalogue MAST hosts (GALEX and Pan-STARRS are MAST
    missions). Every source within radius_arcsec is saved, nearest first.
    """
    def fetch(target, folder, run, key):
        from astroquery.mast import Catalogs
        table = Catalogs.query_region(target.coord, radius=radius_arcsec * u.arcsec, catalog=catalog,
                                      **criteria)
        if table is None or len(table) == 0:
            return "nothing", 0, [], f"No source within {radius_arcsec:g} arcsec."
        if "distance" in table.colnames:
            table.sort("distance")
        files = [_save_table(table, folder / f"{key}.csv")]
        return "found", len(table), files, f"Sources within {radius_arcsec:g} arcsec from MAST, nearest first."
    return fetch


# ---------------------------------------------------------------------------
# Light curves
# ---------------------------------------------------------------------------

def _fetch_ztf(target, folder, run, key):
    """ZTF g, r and i light curves from IRSA, for every ZTF object within 3 arcseconds."""
    radius_deg = 3 / 3600
    response = requests.get(ZTF_URL, timeout=TIMEOUT, params={
        "POS": f"CIRCLE {target.coord.ra.deg:.6f} {target.coord.dec.deg:.6f} {radius_deg:.6f}",
        "BAD_CATFLAGS_MASK": 32768,     # leaves out points ZTF flags as unusable
        "FORMAT": "CSV"})
    response.raise_for_status()
    try:
        frame = pd.read_csv(io.StringIO(response.text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise OSError(f"ZTF's answer wasn't a table: {error}") from error
    mag, err, band = (_column(frame, n) for n in ("mag", "magerr", "filtercode"))
    if len(frame) == 0 or mag is None:
        return "nothing", 0, [], "No ZTF object within 3 arcsec (ZTF covers Dec above about -30)."

    files = [folder / f"{key}_all.csv"]
    frame.to_csv(files[0], index=False)
    hjd, mjd = _column(frame, "hjd"), _column(frame, "mjd")
    if hjd is not None:
        time, system = frame[hjd], "HJD"
    elif mjd is not None:
        time, system = frame[mjd] + 2400000.5, "JD (not heliocentric)"
    else:
        return "found", len(frame), files, "Saved as returned; no time column to make light curve files from."

    bands = frame[band].astype(str) if band is not None else pd.Series("z", index=frame.index)
    for code in sorted(bands.unique()):
        chosen = bands == code
        path = folder / f"{key}_{code.lstrip('z') or code}.txt"
        if save_light_curve(path, time[chosen], frame[mag][chosen], frame[err][chosen] if err else np.zeros(chosen.sum())):
            files.append(path)
    note = f"Time is {system}. One text file per filter, all ZTF fields combined."
    brightest = pd.to_numeric(frame[mag], errors="coerce").median()
    if brightest < ZTF_SATURATION:
        note += (f" The star is about magnitude {brightest:.1f} in ZTF, near where ZTF saturates "
                 f"({ZTF_SATURATION:g}), so expect few points and treat them with care.")
    return "found", len(frame), files, note


def _gaia_source(target, run):
    """
    The Gaia DR3 source for the target, looked up once per run: its source_id
    and which extra products Gaia holds for it. None if nothing matches.
    """
    if "gaia_source" in run.cache:
        return run.cache["gaia_source"]
    from .gaia import TARGET_QUERY_ARCSEC, _query_gaia_archive, _rank_by_separation
    ra, dec = target.coord.ra.deg, target.coord.dec.deg
    g = _query_gaia_archive("""
        SELECT gaia_source.source_id,gaia_source.ra,gaia_source.dec,
        gaia_source.pmra,gaia_source.pmdec,gaia_source.phot_g_mean_mag,
        gaia_source.has_epoch_photometry,gaia_source.has_xp_sampled,gaia_source.has_rvs
        FROM gaiadr3.gaia_source
        WHERE
        CONTAINS(
            POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
            CIRCLE('ICRS',{},{},{})
        )=1""".format(ra, dec, TARGET_QUERY_ARCSEC / 3600), run.log, run.cancel_event)
    source = None
    if g is not None and len(g) > 0:
        order, _separation = _rank_by_separation(g, ra, dec)
        if len(order) > 0:
            row = g.data[int(order[0])]
            source = {"source_id": int(row["source_id"])}
            for flag in ("has_epoch_photometry", "has_xp_sampled", "has_rvs"):
                source[flag] = str(row[flag]).strip().lower() in ("true", "1")
    run.cache["gaia_source"] = source
    return source


# What's asked for from Gaia's data service, in order: (data structure, format).
# On 14 October 2026 ESA renamed the per-source structure from INDIVIDUAL to
# DATAMODEL_STANDARD, and since then INDIVIDUAL gives HTTP 500; astroquery 0.4.11
# still sends INDIVIDUAL unless told otherwise. The old name is kept last in case
# ESA restores it. VOTable is the format ESA documents for every product.
GAIA_REQUESTS = (("DATAMODEL_STANDARD", "votable"), ("DATAMODEL_STANDARD", "csv"), ("INDIVIDUAL", "votable"))


def _gaia_products(source_id, retrieval_type, folder):
    """Gaia DataLink products for one source, as a list of astropy Tables."""
    from astroquery.gaia import Gaia
    products, first_error = None, None
    # load_data unpacks its download into the current folder, which may not be writable
    with _folder_lock:
        before = os.getcwd()
        os.chdir(folder)
        try:
            for structure, data_format in GAIA_REQUESTS:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        products = Gaia.load_data(ids=[source_id], data_release="Gaia DR3",
                                                  retrieval_type=retrieval_type,
                                                  data_structure=structure, format=data_format)
                    break
                except (requests.exceptions.RequestException, OSError) as error:
                    first_error = first_error or error
        finally:
            os.chdir(before)
    if products is None:
        raise OSError(f"Gaia's data service gave an error for {retrieval_type} ({_first_line(first_error)}). "
                      "It may be down for a while; try again later.") from first_error
    tables = []
    for value in (products or {}).values():
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            tables.append(item.to_table() if hasattr(item, "to_table") else item)
    return tables


def _fetch_gaia_light_curve(target, folder, run, key):
    """Gaia DR3 epoch photometry in G, BP and RP. Gaia released it for about 12 million sources."""
    source = _gaia_source(target, run)
    if source is None:
        return "nothing", 0, [], "No Gaia DR3 source at this position."
    if not source["has_epoch_photometry"]:
        return "nothing", 0, [], (f"Gaia DR3 {source['source_id']} has no epoch photometry; DR3 only "
                                  "released it for variable stars and a few other samples.")
    tables = _gaia_products(source["source_id"], "EPOCH_PHOTOMETRY", folder)
    if not tables:
        return "nothing", 0, [], "Gaia sent back no epoch photometry."
    frame = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    files, made = _gaia_epochs_to_files(frame, folder, key)
    note = f"Gaia DR3 {source['source_id']}. "
    if made:
        note += "Time is BJD (TCB). Points Gaia rejected are left out of the text files."
    else:
        note += "Saved as returned; the expected columns weren't there."
    return "found", len(frame), files, note


# Gaia epoch times are days since 2010-01-01 (TCB), i.e. BJD - 2455197.5
GAIA_TIME_ZERO = 2455197.5


def _gaia_epochs_to_files(frame, folder, key):
    """
    Save Gaia epoch photometry, all of it as CSV and each band as a text light
    curve. Works with the ESA layout (one row per measurement, with band, time
    and mag columns) and with tables that give each band its own columns
    (TimeG and Gmag, say). Returns the files and whether light curves were made.
    """
    files = [folder / f"{key}_all.csv"]
    frame.to_csv(files[0], index=False)
    made = False

    time, mag, band = (_column(frame, n) for n in ("time", "mag", "band"))
    if None not in (time, mag, band):
        snr, rejected = _column(frame, "flux_over_error"), _column(frame, "rejected_by_photometry")
        keep = pd.Series(True, index=frame.index)
        if rejected is not None:
            keep &= ~frame[rejected].astype(str).str.lower().isin(["true", "1"])
        with np.errstate(divide="ignore", invalid="ignore"):
            error = (2.5 / np.log(10) / pd.to_numeric(frame[snr], errors="coerce") if snr
                     else pd.Series(0.0, index=frame.index))
        for code in sorted(frame[band].astype(str).unique()):
            chosen = keep & (frame[band].astype(str) == code)
            path = folder / f"{key}_{code}.txt"
            if save_light_curve(path, frame[time][chosen] + GAIA_TIME_ZERO, frame[mag][chosen], error[chosen]):
                files.append(path)
                made = True
        return files, made

    for code in ("G", "BP", "RP"):
        time = _column(frame, f"Time{code}", f"{code}Time", f"t{code}")
        mag = _column(frame, f"{code}mag", f"mag{code}", f"{code}_mag")
        err = _column(frame, f"e_{code}mag", f"{code}magerr", f"e{code}mag")
        if time is None or mag is None:
            continue
        error = frame[err] if err is not None else pd.Series(0.0, index=frame.index)
        path = folder / f"{key}_{code}.txt"
        if save_light_curve(path, pd.to_numeric(frame[time], errors="coerce") + GAIA_TIME_ZERO,
                            pd.to_numeric(frame[mag], errors="coerce"), pd.to_numeric(error, errors="coerce")):
            files.append(path)
            made = True
    return files, made


def _fetch_gaia_light_curve_vizier(target, folder, run, key):
    """
    VizieR's copy of Gaia DR3 epoch photometry (I/355/epphot), for when ESA's
    data service is down. Light curves are made if its columns are recognised;
    the table is saved either way.
    """
    tables = _vizier_tables("I/355/epphot", target, run, 3)
    if not tables:
        return "nothing", 0, [], "VizieR's copy has no epoch photometry within 3 arcsec."
    frame = pd.concat([t.to_pandas() for t in tables.values()], ignore_index=True)
    files, made = _gaia_epochs_to_files(frame, folder, key)
    note = ("Time is BJD (TCB)." if made else
            "Saved as VizieR gives it; the columns weren't recognised, so no text light curves were made.")
    return "found", len(frame), files, note


def _flux_light_curve(path, fits_path):
    """
    Turn a TESS, Kepler or K2 light curve FITS file into a text light curve.
    The flux becomes magnitudes around the star's catalogue magnitude (or
    around zero when the file doesn't give one). Returns the number of points.
    """
    from astropy.io import fits
    with fits.open(fits_path) as hdus:
        data, header = hdus[1].data, hdus[1].header
        names = {n.upper(): n for n in data.columns.names}
        flux_name = next((names[n] for n in ("PDCSAP_FLUX", "KSPSAP_FLUX", "SAP_FLUX", "FLUX") if n in names), None)
        if "TIME" not in names or flux_name is None:
            return 0
        time = np.asarray(data[names["TIME"]], dtype=float)
        flux = np.asarray(data[flux_name], dtype=float)
        err_name = names.get(flux_name.upper() + "_ERR")
        flux_err = np.asarray(data[err_name], dtype=float) if err_name else np.zeros_like(flux)
        good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
        if "QUALITY" in names:
            good &= np.asarray(data[names["QUALITY"]]) == 0
        zero = float(header.get("BJDREFI", 0)) + float(header.get("BJDREFF", 0))
        primary = hdus[0].header
        catalogue_mag = next((float(primary[k]) for k in ("TESSMAG", "KEPMAG")
                              if isinstance(primary.get(k), (int, float))), 0.0)
    if not good.any():
        return 0
    time, flux, flux_err = time[good], flux[good], flux_err[good]
    mag = catalogue_mag - 2.5 * np.log10(flux / np.median(flux))
    return save_light_curve(path, time + zero, mag, 2.5 / np.log(10) * np.abs(flux_err) / flux)


def _fetch_mast(target, folder, run, key):
    """TESS, Kepler and K2 light curves from MAST, as FITS files and as text light curves."""
    from astroquery.mast import Observations
    observations = Observations.query_criteria(coordinates=target.coord, radius=10 * u.arcsec,
                                               obs_collection=["TESS", "Kepler", "K2", "HLSP"],
                                               dataproduct_type="timeseries")
    if len(observations) and "provenance_name" in observations.colnames:
        observations = observations[np.isin(np.asarray(observations["provenance_name"]).astype(str),
                                            MAST_PIPELINES)]
    if len(observations) == 0:
        return "nothing", 0, [], "No TESS, Kepler or K2 light curve within 10 arcsec."
    products = Observations.get_product_list(observations)
    names = np.char.lower(np.asarray(products["productFilename"]).astype(str))
    products = products[np.char.endswith(names, "lc.fits")]
    if len(products) == 0:
        return "nothing", 0, [], "MAST has observations here but no light curve files."
    if run.cancelled():
        return "skipped", 0, [], "Canceled."

    mast_folder = folder / key
    mast_folder.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = Observations.download_products(products, download_dir=str(mast_folder), flat=True)
    files, points = [], 0
    for row in manifest:
        fits_path = Path(str(row["Local Path"]))
        if str(row["Status"]).upper() != "COMPLETE" or not fits_path.is_file():
            continue
        files.append(fits_path)
        text_path = folder / f"{key}_{fits_path.stem}.txt"
        try:
            written = _flux_light_curve(text_path, fits_path)
        except (OSError, ValueError, KeyError, IndexError) as error:
            run.log(f"{fits_path.name} was downloaded but couldn't be turned into a text light curve: {error}")
            continue
        if written:
            files.append(text_path)
            points += written
    if not files:
        return "failed", 0, [], "MAST listed light curves but none could be downloaded."
    return "found", points, files, ("Time is BJD (TDB). Magnitudes are the pipeline flux scaled around "
                                    "the catalogue magnitude; bad-quality points are left out.")


def _fetch_neowise(target, folder, run, key):
    """NEOWISE single exposures at 3.4 and 4.6 microns from IRSA, two visits a year since 2013."""
    from astroquery.ipac.irsa import Irsa
    table = Irsa.query_region(target.coord, catalog="neowiser_p1bs_psd", spatial="Cone",
                              radius=5 * u.arcsec)
    if table is None or len(table) == 0:
        return "nothing", 0, [], "No NEOWISE detection within 5 arcsec."
    files = [_save_table(table, folder / f"{key}_all.csv")]
    frame = table.to_pandas()
    mjd = _column(frame, "mjd")
    for band in ("w1", "w2"):
        mag, err = _column(frame, f"{band}mpro"), _column(frame, f"{band}sigmpro")
        if None in (mjd, mag, err):
            continue
        path = folder / f"{key}_{band.upper()}.txt"
        if save_light_curve(path, frame[mjd] + 2400000.5, frame[mag], frame[err]):
            files.append(path)
    return "found", len(table), files, "Time is JD (UTC), not heliocentric. Exposures come in groups about six months apart."


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def _fetch_sdss_spectra(target, folder, run, key):
    """SDSS optical spectra (DR17) as FITS files."""
    from astroquery.sdss import SDSS
    matches = SDSS.query_region(target.coord, radius=3 * u.arcsec, spectro=True, cache=False)
    if matches is None or len(matches) == 0:
        return "nothing", 0, [], "No SDSS spectrum within 3 arcsec."
    files = [_save_table(matches, folder / f"{key}_list.csv")]
    spectra = SDSS.get_spectra(matches=matches, cache=False, show_progress=False) or []
    for number, (row, hdus) in enumerate(zip(matches, spectra)):
        names = matches.colnames
        tag = "_".join(str(row[c]) for c in ("plate", "mjd", "fiberID") if c in names) or str(number + 1)
        path = folder / f"{key}_{tag}.fits"
        hdus.writeto(path, overwrite=True)
        files.append(path)
    return "found", len(spectra), files, "One FITS file per spectrum, plus the list of matches."


def _fetch_gaia_spectra(target, folder, run, key):
    """Gaia DR3 low-resolution BP/RP spectrum (sampled) and RVS spectrum, where Gaia released them."""
    source = _gaia_source(target, run)
    if source is None:
        return "nothing", 0, [], "No Gaia DR3 source at this position."
    files, missing = [], []
    for flag, retrieval, label in (("has_xp_sampled", "XP_SAMPLED", "xp"), ("has_rvs", "RVS", "rvs")):
        if not source[flag]:
            missing.append(label.upper())
            continue
        for number, table in enumerate(_gaia_products(source["source_id"], retrieval, folder)):
            suffix = "" if number == 0 else f"_{number + 1}"
            files.append(_save_table(table, folder / f"{key}_{label}{suffix}.csv"))
    if not files:
        return "nothing", 0, [], f"Gaia DR3 {source['source_id']} has no released BP/RP or RVS spectrum."
    note = f"Gaia DR3 {source['source_id']}. Wavelength in nm, flux in W m^-2 nm^-1."
    if missing:
        note += f" No {' or '.join(missing)} spectrum was released for it."
    return "found", len(files), files, note


def _fetch_gaia_spectra_vizier(target, folder, run, key):
    """VizieR's copy of the Gaia DR3 sampled BP/RP spectra (I/355/xpsample), for when ESA is down."""
    tables = _vizier_tables("I/355/xpsample", target, run, 3)
    if not tables:
        return "nothing", 0, [], "VizieR's copy has no BP/RP spectrum within 3 arcsec."
    files = [_save_table(table, folder / f"{key}_xp.csv") for table in tables.values()]
    return "found", sum(len(t) for t in tables.values()), files, ("BP/RP sampled spectrum as VizieR gives it. "
                                                                 "RVS spectra aren't in VizieR's copy.")


def _fetch_simbad(target, folder, run, key):
    """
    What SIMBAD lists at the position: identifier and object type. SIMBAD runs at CDS
    with a mirror at Harvard, which is tried when CDS doesn't answer.
    """
    from astroquery.simbad import SimbadClass, conf
    table, errors = None, []
    for server in conf.servers_list:
        # Passing timeout to SimbadClass would ask the CDS server for its limits, which
        # fails when CDS is down; the config value is used without asking
        with conf.set_temp("timeout", SIMBAD_TIMEOUT):
            simbad = SimbadClass()
        simbad.server = server
        try:
            table = simbad.query_region(target.coord, radius=5 * u.arcsec)
            break
        except Exception as error:  # pyvo raises its own errors when a server is down
            errors.append(f"{server}: {_first_line(error)}")
            run.log(f"SIMBAD at {server} didn't answer ({_first_line(error)}).")
    else:
        raise OSError(f"SIMBAD couldn't be reached ({'; '.join(errors)}). Try again later.")
    if table is None or len(table) == 0:
        return "nothing", 0, [], "SIMBAD has no object within 5 arcsec."
    files = [_save_table(table, folder / f"{key}.csv")]
    return "found", len(table), files, "Objects within 5 arcsec."


# ---------------------------------------------------------------------------
# The databases
# ---------------------------------------------------------------------------

ARCHIVES = (
    # Light curves
    Archive("ztf", "ZTF", LIGHT_CURVES, "Optical (g, r, i)",
            "Zwicky Transient Facility, since 2018, northern sky to Dec -30.", _fetch_ztf),
    Archive("mast", "TESS, Kepler and K2", LIGHT_CURVES, "Optical (red)",
            "Space photometry from MAST: SPOC, TESS-SPOC and QLP light curves.", _fetch_mast),
    Archive("gaia_lc", "Gaia DR3 epoch photometry", LIGHT_CURVES, "Optical (G, BP, RP)",
            "Gaia's individual measurements, released for variable stars.",
            _with_backup(_fetch_gaia_light_curve, _fetch_gaia_light_curve_vizier,
                         "ESA's Gaia archive", "VizieR's copy")),
    Archive("neowise", "NEOWISE", LIGHT_CURVES, "Infrared (3.4, 4.6 um)",
            "WISE single exposures, twice a year since 2013.", _fetch_neowise),

    # Catalogue magnitudes, ultraviolet to infrared
    Archive("simbad", "SIMBAD", MAGNITUDES, "Names and type",
            "Identifiers and object type.", _fetch_simbad),
    Archive("vsx", "AAVSO VSX", MAGNITUDES, "Variability",
            "Variable star type, period and epoch.", _vizier("B/vsx/vsx", 5)),
    Archive("galex", "GALEX", MAGNITUDES, "Ultraviolet (FUV, NUV)",
            "GALEX all-sky imaging survey, GR6+7.",
            _with_backup(_mast_catalog("Galex", 5), _vizier("II/335/galex_ais", 5), "MAST", "VizieR")),
    Archive("sdss", "SDSS DR16", MAGNITUDES, "Optical (u, g, r, i, z)",
            "Sloan Digital Sky Survey photometry.", _vizier("V/154/sdss16", 3)),
    Archive("panstarrs", "Pan-STARRS1", MAGNITUDES, "Optical (g, r, i, z, y)",
            "Pan-STARRS1 DR2 mean photometry, Dec above -30.",
            _with_backup(_fetch_panstarrs,
                         _vizier("II/349/ps1", 3), "MAST", "VizieR")),
    Archive("apass", "APASS DR9", MAGNITUDES, "Optical (B, V, g, r, i)",
            "AAVSO Photometric All-Sky Survey.", _vizier("II/336/apass9", 5)),
    Archive("gaia", "Gaia DR3", MAGNITUDES, "Optical (G, BP, RP)",
            "Mean photometry, parallax and proper motion.", _vizier("I/355/gaiadr3", 3)),
    Archive("twomass", "2MASS", MAGNITUDES, "Near infrared (J, H, Ks)",
            "Two Micron All Sky Survey point sources.", _vizier("II/246/out", 3)),
    Archive("allwise", "AllWISE", MAGNITUDES, "Infrared (3.4 to 22 um)",
            "WISE all-sky catalogue.", _vizier("II/328/allwise", 5)),

    # Spectra
    Archive("sdss_spectra", "SDSS spectra", SPECTRA, "Optical",
            "SDSS DR17 optical spectra as FITS files.", _fetch_sdss_spectra),
    Archive("gaia_spectra", "Gaia DR3 spectra", SPECTRA, "Optical (330 to 1050 nm)",
            "BP/RP low-resolution and RVS spectra.",
            _with_backup(_fetch_gaia_spectra, _fetch_gaia_spectra_vizier, "ESA's Gaia archive", "VizieR's copy")),
    Archive("lamost", "LAMOST DR5", SPECTRA, "Optical",
            "Catalogue of LAMOST spectra with their parameters.",
            _vizier("V/164", 3, "LAMOST observations within 3 arcsec. The spectra themselves are "
                                "on lamost.org under the obsid in the table.")),
    Archive("apogee", "APOGEE DR17", SPECTRA, "Near infrared (H band)",
            "Catalogue of APOGEE spectra with stellar parameters and radial velocities.",
            _vizier("III/286", 3, "APOGEE observations within 3 arcsec. The spectra themselves are "
                                  "on the SDSS science archive under the APOGEE ID.")),

    # X-ray and radio
    Archive("rosat", "ROSAT 2RXS", HIGH_ENERGY, "X-ray (0.1 to 2.4 keV)",
            "Second ROSAT all-sky survey source catalogue (Boller et al. 2016).",
            _vizier("J/A+A/588/A103/cat2rxs", 30)),
    Archive("xmm", "XMM-Newton 4XMM-DR13", HIGH_ENERGY, "X-ray (0.2 to 12 keV)",
            "XMM-Newton serendipitous source catalogue.", _vizier("IX/69", 15)),
    Archive("chandra", "Chandra CSC 2", HIGH_ENERGY, "X-ray (0.5 to 7 keV)",
            "Chandra Source Catalog release 2.", _vizier("IX/57/csc2master", 5)),
    Archive("erosita", "eROSITA eRASS1", HIGH_ENERGY, "X-ray (0.2 to 8 keV)",
            "First eROSITA all-sky survey, western Galactic half.", _vizier("J/A+A/682/A34", 15)),
    Archive("nvss", "NVSS", HIGH_ENERGY, "Radio (1.4 GHz)",
            "NRAO VLA Sky Survey, Dec above -40.", _vizier("VIII/65/nvss", 30)),
    Archive("first", "FIRST", HIGH_ENERGY, "Radio (1.4 GHz)",
            "VLA FIRST survey, mostly the northern Galactic cap.", _vizier("VIII/92/first14", 5)),
    Archive("vlass", "VLASS", HIGH_ENERGY, "Radio (3 GHz)",
            "VLA Sky Survey epoch 1 quick look catalogue, Dec above -40.", _vizier("J/ApJS/255/30", 5)),
)


def archive_keys(kinds=None):
    """The keys of every database, or only those of the given kinds."""
    return [a.key for a in ARCHIVES if kinds is None or a.kind in kinds]


# ---------------------------------------------------------------------------
# Running a download
# ---------------------------------------------------------------------------

def download(target, output_folder, keys=None, write_callback=None, cancel_event=None):
    """
    Download from the chosen databases for one target.

    Everything goes in a folder named after the target inside output_folder,
    with archive_summary.csv listing what each database returned. A database
    that fails or has nothing is recorded and the rest carry on.

    :param target: a Target, from resolve_target
    :param keys: database keys to use; all of them by default
    :return: list of Result, one per database tried
    """
    log = make_logger(write_callback)
    chosen = [a for a in ARCHIVES if keys is None or a.key in set(keys)]
    if not chosen:
        raise ArchiveError("Choose at least one database.")
    folder = Path(output_folder) / target.label
    folder.mkdir(parents=True, exist_ok=True)
    run = Run(log=log, cancel_event=cancel_event)
    log(f"{target.name}: RA {target.coord.ra.to_string(u.hourangle, sep=':', precision=2, pad=True)}, "
        f"Dec {target.coord.dec.to_string(sep=':', precision=1, alwayssign=True, pad=True)} (J2000). "
        f"Saving to {folder}")

    results = []
    for archive in chosen:
        if run.cancelled():
            log("Task canceled.")
            break
        try:
            status, rows, files, note = archive.fetch(target, folder, run, archive.key)
        except Exception as error:  # one database failing mustn't stop the others
            status, rows, files = "failed", 0, []
            note = _first_line(error)
        results.append(Result(archive.key, archive.name, archive.kind, status, rows,
                              [Path(f) for f in files], note))
        if status == "found":
            log(f"{archive.name}: {rows} row{'s' if rows != 1 else ''} in {len(files)} "
                f"file{'s' if len(files) != 1 else ''}. {note}")
        elif status == "failed":
            log(f"{archive.name}: failed. {note}")
        else:
            log(f"{archive.name}: {note or status}")

    write_summary(folder, target, results)
    found = sum(r.status == "found" for r in results)
    failed = [r.name for r in results if r.status == "failed"]
    log(f"{found} of {len(results)} databases had data for {target.name}."
        + (f" These couldn't be reached or gave an error: {', '.join(failed)}." if failed else ""))
    return results


def write_summary(folder, target, results):
    """Save archive_summary.csv: one row per database with what it returned."""
    bands = {a.key: a.band for a in ARCHIVES}
    rows = [{"database": r.name, "kind": r.kind, "wavelength": bands.get(r.key, ""), "status": r.status,
             "rows": r.rows, "files": "; ".join(f.name for f in r.files), "note": r.note}
            for r in results]
    frame = pd.DataFrame(rows, columns=["database", "kind", "wavelength", "status", "rows", "files", "note"])
    path = Path(folder) / SUMMARY_FILE
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {target.name}, RA {target.coord.ra.deg:.6f} Dec {target.coord.dec.deg:+.6f} "
                     "(J2000, degrees)\n")
        frame.to_csv(handle, index=False)
    return path


def main(argv=None):
    """Download archive data from the command line: python -m EclipsingBinaries.archives "SW Lac" out"""
    parser = argparse.ArgumentParser(description="Download archive data for one target.")
    parser.add_argument("name", help="target name, looked up through CDS unless --ra and --dec are given")
    parser.add_argument("output_folder")
    parser.add_argument("--ra", default="", help="J2000 RA in hours, HH:MM:SS")
    parser.add_argument("--dec", default="", help="J2000 Dec in degrees, DD:MM:SS")
    parser.add_argument("--only", default="", help="comma-separated database keys; all by default. "
                                                     f"Keys: {', '.join(archive_keys())}")
    args = parser.parse_args(argv)
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or None
    unknown = sorted(set(keys or []) - set(archive_keys()))
    if unknown:
        parser.error(f"unknown database keys: {', '.join(unknown)}")
    try:
        target = resolve_target(args.name, args.ra, args.dec)
        download(target, args.output_folder, keys)
    except ArchiveError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
