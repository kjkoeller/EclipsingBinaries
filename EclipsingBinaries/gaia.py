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
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
import os
from requests import exceptions as request_exceptions
import socket
import time

from .vseq_updated import splitter

def target_star(ra_input, dec_input, output_path, write_callback=None, cancel_event=None):
    """
    This queries Gaia DR3 all the parameters below, but I only outputted the specific parameters that are (at the moment)
    the most important for current research at BSU

    :return: Outputs a file with the specific parameters
    """

    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)

    try:
        if cancel_event.is_set():
            log("Task canceled.")
            return

        ra_input2 = splitter([ra_input])
        dec_input2 = splitter([dec_input])

        ra = ra_input2[0] * 15
        dec = dec_input2[0]

        g = GaiaData.from_query("""
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
            POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
            CIRCLE(
                'ICRS',
                COORD1(EPOCH_PROP_POS({}, {},4.7516,48.8840,-24.1470,0,2000,2016.0)),
                COORD2(EPOCH_PROP_POS({}, {},4.7516,48.8840,-24.1470,0,2000,2016.0)),
                0.001388888888888889)
        )=1""".format(ra, dec, ra, dec))

        if cancel_event.is_set():
            log("Task canceled.")
            return

        # to add parameters to the output file, add them here and the format for the parameter is 'g.[param name from above]'
        df = pd.DataFrame({
            "Parallax(mas)": g.parallax[:4],
            "Parallax_err(mas)": g.parallax_error[:4],
            "Distance_lower(pc)": g.distance_gspphot_lower[:4],
            "Distance(pc)": g.distance_gspphot_upper[:4],
            "Distance_higher(pc)": g.distance_gspphot[:4],
            "T_eff_lower(K)": g.teff_gspphot_lower[:4],
            "T_eff(K)": g.teff_gspphot[:4],
            "T_eff_higher(K)": g.teff_gspphot_upper[:4],
            "G_Mag": g.phot_g_mean_mag[:4],
            "G_BP_Mag": g.phot_bp_mean_mag[:4],
            "G_RP_Mag": g.phot_rp_mean_mag[:4],
            "Radial_velocity(km/s)": g.radial_velocity[:4],
            "Radial_velocity_err(km/s)": g.radial_velocity_error[:4],
        })

        output_file = os.path.join(output_path, "gaia_results.csv")

        df.to_csv(output_file, sep="\t")

        log("\n For more information on each of the output parameters please reference this webpage: "
              "https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html")
        log("If any of the parameters have values of '1e+20', then Gaia does not have data on that specific parameter.")

        log("\nFinished Gaia search.\n")
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


# --- _is_timeout_exception ---
# Identifies network timeout errors that should trigger retries or fallback.
def _is_timeout_exception(error: Exception) -> bool:
    timeout_types = (
        TimeoutError,
        socket.timeout,
        request_exceptions.Timeout,
        request_exceptions.ReadTimeout,
    )
    if isinstance(error, timeout_types):
        return True
    return "timed out" in str(error).lower()


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
# Queries TIC v8.2 for one sky region with retries and mirror fallback.
def _query_single_tic_region(
    center_coord: SkyCoord,
    radius: u.Quantity,
    tmag_limit: float | None = None,
    write_callback=None,
    cancel_event=None
) -> pd.DataFrame:
    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)

    required_columns = ["_RAJ2000", "_DEJ2000", "Tmag", "e_Tmag"]
    vizier_servers = (
        "vizier.cds.unistra.fr",
        "vizier.cfa.harvard.edu",
    )
    timeout_seconds = 60
    retries_per_server = 2

    last_timeout_error = None

    for server_name in vizier_servers:
        for attempt in range(1, retries_per_server + 1):
            if cancel_event and cancel_event.is_set():
                log("Task canceled.")
                return _empty_tic_dataframe()

            try:
                column_filters = {}
                if tmag_limit is not None:
                    column_filters["Tmag"] = f"<{tmag_limit:.3f}"

                vizier = Vizier(columns=required_columns, row_limit=-1, column_filters=column_filters)
                vizier.TIMEOUT = timeout_seconds
                vizier.VIZIER_SERVER = server_name
                result = vizier.query_region(center_coord, radius=radius, catalog="IV/39/tic82")

                if len(result) == 0:
                    return _empty_tic_dataframe()

                tic_df = result[0].to_pandas()
                for column_name in required_columns:
                    if column_name not in tic_df.columns:
                        tic_df[column_name] = np.nan
                    tic_df[column_name] = pd.to_numeric(tic_df[column_name], errors="coerce")

                return tic_df[required_columns]

            except Exception as error:
                if _is_timeout_exception(error):
                    last_timeout_error = error
                    log(
                        f"TIC query timeout via {server_name} "
                        f"(attempt {attempt}/{retries_per_server})."
                    )
                    if attempt < retries_per_server:
                        time.sleep(1.0)
                        continue
                    break
                raise

    if last_timeout_error is not None:
        raise request_exceptions.ReadTimeout(last_timeout_error)

    return _empty_tic_dataframe()


# --- _query_tic_catalog ---
# Queries TIC v8.2 from VizieR for all input coordinates with fallback behavior.
def _query_tic_catalog(
    ra_hours: np.ndarray,
    dec_degrees: np.ndarray,
    tmag_limit: float | None = None,
    write_callback=None,
    cancel_event=None
) -> pd.DataFrame:
    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)

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
    except Exception as error:
        if not _is_timeout_exception(error):
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
            except Exception as star_error:
                if _is_timeout_exception(star_error):
                    log(
                        f"Skipping TIC lookup for source {count + 1}/{ra_hours.size} due to timeout."
                    )
                    continue
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
    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)
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

        tic_df = _query_tic_catalog(
            ra_hours,
            dec_degrees,
            tmag_limit=tmag_limit,
            write_callback=write_callback,
            cancel_event=cancel_event,
        )

        log("Finished querying the IV/39/tic82 Vizier Catalog.")
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

