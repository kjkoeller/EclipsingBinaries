"""
Author: Kyle Koeller
Date Created: 03/08/2023

Last Edited: 04/19/2026
This program queries Gaia DR3, to gather specific parameters

https://gea.esac.esa.int/archive/
https://iopscience.iop.org/article/10.3847/1538-3881/acaaa7/pdf
https://iopscience.iop.org/article/10.3847/1538-3881/ab3467/pdf
https://arxiv.org/pdf/2012.01916.pdf
"""

from pyia import GaiaData
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
import os
from http.client import HTTPException
from requests import exceptions as request_exceptions
import time

from .vseq_updated import splitter
from ._log import make_logger
from ._vizier import VizierUnavailable, query_region as vizier_query_region

# Gaia sources within this distance of the target (after moving them back to
# J2000 with their own proper motion) count as matches
TARGET_MATCH_ARCSEC = 5.0
# The cone queried around the target is wider than the match radius so that a
# star which has moved since J2000 is still returned by the query
TARGET_QUERY_ARCSEC = 30.0
# Gaia DR3 positions are given at epoch 2016.0; the coordinates entered are J2000
GAIA_DR3_EPOCH = 2016.0
INPUT_EPOCH = 2000.0
MAX_TARGET_ROWS = 4
# Seconds to wait before each retry when the Gaia archive drops the connection
GAIA_RETRY_WAITS = (5, 15)
TIC_CATALOG = "IV/39/tic82"


def _rank_by_separation(g, ra_deg, dec_deg):
    """
    Order Gaia results by distance from the target and keep the close ones.

    Each source is moved from Gaia's 2016 epoch back to J2000 using its own
    proper motion before measuring the separation, so fast-moving stars still
    match the J2000 coordinates the user entered.

    :return: (indices into g, separations in arcsec), closest first
    """
    def finite_or_zero(values):
        values = np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)
        return np.where(np.isfinite(values) & (np.abs(values) < 1e10), values, 0.0)

    years = GAIA_DR3_EPOCH - INPUT_EPOCH
    ra_gaia = np.asarray(g.ra.to_value(u.deg), dtype=float)
    dec_gaia = np.asarray(g.dec.to_value(u.deg), dtype=float)
    pmra = finite_or_zero(g.pmra.to_value(u.mas / u.yr))  # already includes cos(dec)
    pmdec = finite_or_zero(g.pmdec.to_value(u.mas / u.yr))

    dec_j2000 = dec_gaia - pmdec * years / 3.6e6
    ra_j2000 = ra_gaia - pmra * years / 3.6e6 / np.cos(np.radians(dec_gaia))

    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    separation = target.separation(SkyCoord(ra_j2000 * u.deg, dec_j2000 * u.deg)).arcsec

    order = np.argsort(separation)
    order = order[separation[order] <= TARGET_MATCH_ARCSEC][:MAX_TARGET_ROWS]
    return order, separation[order]

def _query_gaia_archive(adql, write_callback=None, cancel_event=None):
    """
    Run an ADQL query on the ESA Gaia archive, trying again a couple of times if
    the connection drops. The archive has been dropping connections now and then
    ("connection reset by peer", "remote end closed connection"), and a second
    try usually works. Returns None if canceled between tries.
    """
    log = make_logger(write_callback)
    for attempt in range(1, len(GAIA_RETRY_WAITS) + 2):
        try:
            return GaiaData.from_query(adql)
        except (request_exceptions.RequestException, OSError, HTTPException) as error:
            if attempt > len(GAIA_RETRY_WAITS):
                raise OSError(f"The Gaia archive at ESA didn't answer after {attempt} tries ({error}). "
                              "It has had outages lately, so please try again later. Updating "
                              "astroquery (pip install -U astroquery) can also help.") from error
            wait = GAIA_RETRY_WAITS[attempt - 1]
            log(f"The Gaia archive didn't answer ({error}); trying again in {wait} seconds.")
            for _ in range(wait):
                if cancel_event is not None and cancel_event.is_set():
                    return None
                time.sleep(1)


def target_star(ra_input, dec_input, output_path, write_callback=None, cancel_event=None):
    """
    This queries Gaia DR3 all the parameters below, but I only outputted the specific parameters that are (at the moment)
    the most important for current research at BSU

    :return: Outputs a file with the specific parameters
    """

    log = make_logger(write_callback)

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        ra_input2 = splitter([ra_input])
        dec_input2 = splitter([dec_input])

        ra = ra_input2[0] * 15
        dec = dec_input2[0]

        # A plain cone around the entered position. This used to call
        # EPOCH_PROP_POS with one particular star's parallax and proper motion
        # written into the query, which shifted every target by that star's motion.
        g = _query_gaia_archive("""
        SELECT TOP 2000
        gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.parallax_error,
        gaia_source.pmra,gaia_source.pmdec,gaia_source.ruwe,gaia_source.phot_g_mean_mag,gaia_source.bp_rp,
        gaia_source.radial_velocity,gaia_source.radial_velocity_error,gaia_source.rv_method_used,
        gaia_source.phot_variable_flag,gaia_source.non_single_star,gaia_source.has_xp_continuous,
        gaia_source.has_xp_sampled,gaia_source.has_rvs,gaia_source.has_epoch_photometry,gaia_source.has_epoch_rv,
        gaia_source.has_mcmc_gspphot,gaia_source.has_mcmc_msc,gaia_source.teff_gspphot,gaia_source.teff_gspphot_lower,
        gaia_source.teff_gspphot_upper,gaia_source.logg_gspphot,
        gaia_source.mh_gspphot,gaia_source.distance_gspphot,gaia_source.distance_gspphot_lower,
        gaia_source.distance_gspphot_upper,gaia_source.azero_gspphot,gaia_source.ag_gspphot,
        gaia_source.ebpminrp_gspphot,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_mag,gaia_source.phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE
        CONTAINS(
            POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
            CIRCLE('ICRS', {}, {}, {})
        )=1""".format(ra, dec, TARGET_QUERY_ARCSEC / 3600), write_callback, cancel_event)

        if g is None:  # canceled while waiting to retry
            log("Task canceled.")
            return

        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        order, separation = _rank_by_separation(g, ra, dec)
        if len(order) == 0:
            log(f"No Gaia DR3 source found within {TARGET_MATCH_ARCSEC:g} arcsec of the target.")
        g = g[order]

        # Closest source first. To add parameters to the output file, add them
        # here as 'g.[param name from the query above]'
        df = pd.DataFrame({
            "Source_ID": np.asarray(g.source_id),
            "Separation(arcsec)": np.round(separation, 3),
            "Parallax(mas)": g.parallax,
            "Parallax_err(mas)": g.parallax_error,
            "Distance_lower(pc)": g.distance_gspphot_lower,
            "Distance(pc)": g.distance_gspphot,
            "Distance_higher(pc)": g.distance_gspphot_upper,
            "T_eff_lower(K)": g.teff_gspphot_lower,
            "T_eff(K)": g.teff_gspphot,
            "T_eff_higher(K)": g.teff_gspphot_upper,
            "G_Mag": g.phot_g_mean_mag,
            "G_BP_Mag": g.phot_bp_mean_mag,
            "G_RP_Mag": g.phot_rp_mean_mag,
            "Radial_velocity(km/s)": g.radial_velocity,
            "Radial_velocity_err(km/s)": g.radial_velocity_error,
        })

        output_file = os.path.join(output_path, "gaia_results.csv")
        df.to_csv(output_file, index=False)
        log(f"Saved {len(df)} Gaia DR3 source(s), closest first, to {output_file}")

        log("\n For more information on each of the output parameters please reference this webpage: "
              "https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html")
        log("If any of the parameters have values of '1e+20', then Gaia does not have data on that specific parameter.")

        log("\nFinished Gaia search.\n")
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


# --- _empty_tic_dataframe ---
# Creates a standardized empty TIC dataframe with expected columns.
def _empty_tic_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["_RAJ2000", "_DEJ2000", "Tmag", "e_Tmag"])


# --- _compute_tmag_limit ---
# Computes a TIC Tmag filter limit from APASS V magnitudes.
def _compute_tmag_limit(apass_vmag) -> float | None:
    if apass_vmag is None:
        return None

    v_series = pd.to_numeric(pd.Series(apass_vmag), errors="coerce").dropna()
    if v_series.empty:
        return None

    return float(v_series.max() + 1.0)


# --- _query_single_tic_region ---
# Queries TIC v8.2 for one sky region, trying the VizieR mirrors in turn.
def _query_single_tic_region(
    center_coord: SkyCoord,
    radius: u.Quantity,
    tmag_limit: float | None = None,
    write_callback=None,
    cancel_event=None
) -> pd.DataFrame:
    required_columns = ["_RAJ2000", "_DEJ2000", "Tmag", "e_Tmag"]
    column_filters = {}
    if tmag_limit is not None:
        column_filters["Tmag"] = f"<{tmag_limit:.3f}"

    table = vizier_query_region(TIC_CATALOG, center_coord, columns=required_columns,
                                column_filters=column_filters, radius=radius,
                                write_callback=write_callback, cancel_event=cancel_event)
    if table is None:
        return _empty_tic_dataframe()

    tic_df = table.to_pandas()
    for column_name in required_columns:
        if column_name not in tic_df.columns:
            tic_df[column_name] = np.nan
        tic_df[column_name] = pd.to_numeric(tic_df[column_name], errors="coerce")
    return tic_df[required_columns]


# --- _query_tic_catalog ---
# Queries TIC v8.2 from VizieR for all input coordinates with fallback behavior.
def _query_tic_catalog(
    ra_hours: np.ndarray,
    dec_degrees: np.ndarray,
    tmag_limit: float | None = None,
    write_callback=None,
    cancel_event=None
) -> pd.DataFrame:
    log = make_logger(write_callback)

    if cancel_event and cancel_event.is_set():
        log("Task canceled.")
        return _empty_tic_dataframe()

    if ra_hours.size == 0:
        return _empty_tic_dataframe()

    input_coords = SkyCoord(ra=ra_hours * 15.0 * u.deg, dec=dec_degrees * u.deg, frame="icrs")
    center_ra = np.median(input_coords.ra.deg)
    center_dec = np.median(input_coords.dec.deg)
    center_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")
    max_sep = input_coords.separation(center_coord).max()
    search_radius = max(max_sep + 30.0 * u.arcsec, 10.0 * u.arcsec)

    try:
        return _query_single_tic_region(
            center_coord,
            search_radius,
            tmag_limit=tmag_limit,
            write_callback=write_callback,
            cancel_event=cancel_event,
        )
    except VizierUnavailable as error:
        # A large field can take too long on every server; smaller cones may still work.
        # Any other failure means VizieR itself is down, and smaller queries won't help.
        if not error.timed_out or ra_hours.size < 2:
            raise

        log("Bulk TIC query timed out. Falling back to per-star TIC queries.")
        star_radius = 5.0 * u.arcsec
        tic_frames = []
        for count, coord in enumerate(input_coords):
            if cancel_event and cancel_event.is_set():
                log("Task canceled.")
                return _empty_tic_dataframe()

            try:
                star_df = _query_single_tic_region(
                    coord,
                    star_radius,
                    tmag_limit=tmag_limit,
                    write_callback=write_callback,
                    cancel_event=cancel_event,
                )
                if not star_df.empty:
                    tic_frames.append(star_df)
            except VizierUnavailable:
                # A 5 arcsecond cone failing on every server means VizieR is down, not
                # that the field is too big, so trying the rest one by one would only wait longer
                log(f"TIC lookup failed for source {count + 1}/{ra_hours.size}, so the rest are skipped.")
                raise

        if not tic_frames:
            log("No TIC entries were retrieved after fallback queries.")
            return _empty_tic_dataframe()

        combined_df = pd.concat(tic_frames, ignore_index=True).drop_duplicates()
        return combined_df[["_RAJ2000", "_DEJ2000", "Tmag", "e_Tmag"]]


# --- _match_tic_magnitudes ---
# Matches input stars to nearest TIC source and formats Tmag outputs.
def _match_tic_magnitudes(
    ra_hours: np.ndarray,
    dec_degrees: np.ndarray,
    tic_df: pd.DataFrame
) -> tuple[list[float], list[float]]:
    if ra_hours.size == 0:
        return [], []

    if tic_df.empty:
        return [99.999] * ra_hours.size, [99.999] * ra_hours.size

    input_coords = SkyCoord(ra=ra_hours * 15.0 * u.deg, dec=dec_degrees * u.deg, frame="icrs")
    catalog_coords = SkyCoord(
        ra=tic_df["_RAJ2000"].to_numpy() * u.deg,
        dec=tic_df["_DEJ2000"].to_numpy() * u.deg,
        frame="icrs",
    )

    nearest_indices, separations, _ = input_coords.match_to_catalog_sky(catalog_coords)
    max_match_sep = 3.0 * u.arcsec

    tmag_list = []
    tmag_error_list = []
    for idx, separation in zip(nearest_indices, separations):
        if separation > max_match_sep:
            tmag_list.append(99.999)
            tmag_error_list.append(99.999)
            continue

        row = tic_df.iloc[int(idx)]
        tmag = row["Tmag"]
        tmag_err = row["e_Tmag"]

        if pd.isna(tmag):
            tmag_list.append(99.999)
        else:
            tmag_list.append(float(format(float(tmag), ".3f")))

        if pd.isna(tmag_err):
            tmag_error_list.append(99.999)
        else:
            tmag_error_list.append(float(format(float(tmag_err), ".3f")))

    return tmag_list, tmag_error_list


def tess_mag(ra, dec, write_callback, cancel_event, apass_vmag=None):
    """
    Calculates TESS magnitudes for comparison stars

    :param ra: List of RA's for comparison stars
    :param dec: List of DEC's for comparison stars
    :param write_callback: Function to write log messages to the GUI
    :param cancel_event:
    :param apass_vmag: APASS V magnitudes for dynamic TIC Tmag filtering

    :return: list of TESS magnitudes and errors
    """
    log = make_logger(write_callback)
    try:
        if cancel_event and cancel_event.is_set():
            log("Task canceled.")
            return

        ra_hours = np.asarray(ra, dtype=float)
        dec_degrees = np.asarray(dec, dtype=float)

        if ra_hours.size != dec_degrees.size:
            raise ValueError("RA and DEC list lengths must match.")

        tmag_limit = _compute_tmag_limit(apass_vmag)

        log("Starting the query for the Vizier catalog.")
        if tmag_limit is not None:
            log(f"Applying TIC Tmag filter: Tmag < {tmag_limit:.3f}")

        try:
            tic_df = _query_tic_catalog(
                ra_hours,
                dec_degrees,
                tmag_limit=tmag_limit,
                write_callback=write_callback,
                cancel_event=cancel_event,
            )
        except VizierUnavailable as error:
            # The APASS magnitudes are already in hand, so keep them and mark Tmag as missing
            log(f"{error} TESS magnitudes are written as 99.999 for this run.")
            tic_df = _empty_tic_dataframe()

        log(f"Finished querying the {TIC_CATALOG} Vizier Catalog.")
        log("Processing data from the Vizier catalog.")

        if cancel_event and cancel_event.is_set():
            log("Task canceled.")
            return

        t_list, t_err_list = _match_tic_magnitudes(ra_hours, dec_degrees, tic_df)

        log("Finished cataloging the Vizier results.")
        log("Finished TIC v8.2 search and extraction of TESS magnitudes.")

        return t_list, t_err_list
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    target_star()

