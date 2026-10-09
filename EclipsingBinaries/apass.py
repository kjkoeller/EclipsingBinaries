"""
Combines all APASS programs that were originally separate on GitHub for an easy editing and less to load per file.

Author: Kyle Koeller
Created: 12/26/2022
Last Updated: 04/28/2026
"""

from astroquery.vizier import Vizier
import numpy as np
import pandas as pd
import os

import astropy.units as u
import astropy.coordinates as coord
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from matplotlib.figure import Figure
import warnings
from PyAstronomy import pyasl

from .gaia import tess_mag as ga
from .vseq_updated import isNaN, conversion
from ._log import make_logger

# turn off this warning that just tells the user,
# "The warning raised when the contents of the FITS header have been modified to be standards compliant."
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


def comparison_selector(ra: str = "", dec: str = "", pipeline: bool = False, folder_path: str = "",
                        obj_name: str = "", science_image: str = "", write_callback=None, cancel_event=None):
    """
    This code compares AIJ found stars (given an RA and DEC) to APASS stars to get their respective Johnson B, V, and
    Cousins R values and their respective errors.

    This code is not 100% accurate and will still need the human eye to compare the final list to the AIJ given list. As
    this code can only get down to such an accuracy to be effective in gathering stars to be usable.

    :param ra: The right ascension of the target
    :param dec: The declination of the target
    :param pipeline: The pipeline that is being used
    :param folder_path: The path of the folder where the images are going to
    :param obj_name: The name of the target object
    :param science_image: Science image for the overlay
    :param write_callback: Function to write log messages to the GUI
    :param cancel_event:

    :return: A list of stars that are the most likely to be on the AIJ list of stars
    """

    log = make_logger(write_callback)

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        log("Starting the process for finding APASS stars.")
        apass_file, input_ra, input_dec, input_ra_text, input_dec_text, T_list = cousins_r(
            ra, dec, pipeline, folder_path, obj_name, write_callback, cancel_event
        )
        df = pd.read_csv(apass_file, header=None, skiprows=[0], sep="\t")

        log("Finished Saving")
        # print(
        #     "The output file you have entered has RA and DEC for stars and their B, V, Cousins R, and TESS T magnitudes "
        #     "with their respective errors.\n")

        radec_list = create_radec(
            df,
            input_ra,
            input_dec,
            T_list,
            pipeline,
            folder_path,
            obj_name,
            write_callback,
            cancel_event,
            target_ra_text=input_ra_text,
            target_dec_text=input_dec_text
        )

        overlay_path = overlay(
            df,
            input_ra,
            input_dec,
            science_image,
            folder_path=folder_path,
            obj_name=obj_name,
            write_callback=write_callback,
        )
        if overlay_path:
            log(f"Saved overlay image to: {overlay_path}")

        log("\nReduction process completed successfully.\n")

        return radec_list
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def cousins_r(ra, dec, pipeline, folder_path, obj_name, write_callback, cancel_event):
    """
    Calculates the Cousins R_c value for a given B, V, g', and r' from APASS

    :param ra: The right ascension of the target
    :param dec: The declination of the target
    :param pipeline: The pipeline that is being used
    :param folder_path: The path of the folder where the images are going to
    :param obj_name: The name of the target object
    :param write_callback: Function to write log messages to the GUI
    :param cancel_event:

    :return: Outputs a file to be used for R_c values
    """

    log = make_logger(write_callback)

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled before starting.")
            return

        input_file, input_ra, input_dec, input_ra_text, input_dec_text = catalog_finder(
            ra, dec, pipeline, folder_path, obj_name, write_callback, cancel_event
        )
        df = pd.read_csv(input_file, header=None, skiprows=[0], sep=",")

        # writes the columns from the input file
        ra = df[0]
        dec = df[1]
        B = df[2]
        e_B = df[3]
        V = df[4]
        e_V = df[5]
        g = df[6]
        e_g = df[7]
        r = df[8]
        e_r = df[9]
        i = df[10]
        e_i = df[11]

        Rc = []
        e_Rc = []
        Ic = []
        e_Ic = []
        count = 0

        # loop that goes through each value in B to get the total amount of values to be calculated
        log("Calculating Cousins R and I filter values.")
        for _ in B:
            root_rc, val_rc, root_ic, val_ic = calculations(B, V, g, r, i, e_B, e_V, e_g, e_r, e_i, count)
            if isNaN(val_rc) is True:
                # if the value is nan then append 99.999 to the R_c value and its error to make it obvious that there is
                # no given value
                Rc.append(99.999)
                e_Rc.append(99.999)
            else:
                # if there is a value then format that value with 3 decimal places for RADEC precision consistency
                Rc.append(format(val_rc, ".3f"))
                e_Rc.append(format(root_rc, ".3f"))

            if isNaN(val_ic) is True:
                Ic.append(99.999)
                e_Ic.append(99.999)
            else:
                Ic.append(format(val_ic, ".3f"))
                e_Ic.append(format(root_ic, ".3f"))

            count += 1

        ra_decimal = np.array([_to_decimal_coord(value) for value in ra], dtype=float)
        dec_decimal = np.array([_to_decimal_coord(value) for value in dec], dtype=float)
        log("Starting VizieR Search for TESS Magnitudes")
        log("Tmag is from the TESS Input Catalog (TIC v8.2). Please go to the GitHub page for more information.")
        T_list, T_err_list = ga(ra_decimal, dec_decimal, write_callback, cancel_event, apass_vmag=V)

        # puts all columns into a dataframe for output
        final = pd.DataFrame({
            # need to keep RA and DEC in order to compare with catalog comparison or with the radec file
            "RA": ra,
            "DEC": dec,
            "BMag": B,
            "e_BMag": e_B,
            "VMag": V,
            "e_VMag": e_V,
            "Rc": Rc,
            "e_Rc": e_Rc,
            "Ic": Ic,
            "e_Ic": e_Ic,
            "g": g,
            "e_g": e_g,
            "r": r,
            "e_r": e_r,
            "i": i,
            "e_i": e_i,
            "TMag": T_list,
            "e_TMag": T_err_list
        })

        output_file = os.path.join(folder_path, "APASS_" + obj_name + "_Rc_Ic.txt")
        # noinspection PyTypeChecker
        final.to_csv(output_file, index=True, sep="\t")
        log("Completed Cousins R calculations.")

        return output_file, input_ra, input_dec, input_ra_text, input_dec_text, T_list
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def query_vizier(ra_input, dec_input, write_callback, cancel_event):
    """
    Queries the Vizier database for the APASS catalog

    :param ra_input: Right ascension input
    :param dec_input: Declination input
    :return: table result from Vizier
    """

    log = make_logger(write_callback)

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        log("Starting the query for the Vizier catalog.")

        # Query Vizier here and return result
        result = Vizier(
            columns=['_RAJ2000', '_DEJ2000', 'Vmag', "e_Vmag", 'Bmag', "e_Bmag", "g'mag", "e_g'mag", "r'mag",
                     "e_r'mag", "i'mag", "e_i'mag"],
            row_limit=-1,
            column_filters=({"Vmag": "<14", "Bmag": "<14"})).query_region(
            coord.SkyCoord(ra=ra_input, dec=dec_input, unit=(u.h, u.deg), frame="icrs"),
            width="30m", catalog="APASS")

        # catalog is II/336/apass9
        catalog = "II/336/apass9"
        tb = result[catalog]
        log(f"Finished querying the {catalog} Vizier Catalog.")

        return tb
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def process_data(vizier_result):
    """
    Processes the data from the Vizier query

    :param vizier_result: Table from Vizier
    :return: Panda dataframe
    """
    # converts the table result to a list format for putting values into lists
    table_list = []
    for i in vizier_result:
        table_list.append(i)

    ra = []
    dec = []
    vmag = []
    e_vmag = []
    bmag = []
    e_bmag = []
    gmag = []
    e_gmag = []
    rmag = []
    e_rmag = []
    imag = []
    e_imag = []

    one = 0
    # pastes all variables into a list for future use
    for _ in range(0, len(table_list) - 1):
        two = 0
        ra.append(table_list[one][two])
        dec.append(table_list[one][two + 1])
        vmag.append(table_list[one][two + 2])
        e_vmag.append(table_list[one][two + 3])
        bmag.append(table_list[one][two + 4])
        e_bmag.append(table_list[one][two + 5])
        gmag.append(table_list[one][two + 6])
        e_gmag.append(table_list[one][two + 7])
        rmag.append(table_list[one][two + 8])
        e_rmag.append(table_list[one][two + 9])
        imag.append(table_list[one][two + 10])
        e_imag.append(table_list[one][two + 11])

        one += 1

    # converts degree RA to Hour RA
    ra_new = []
    for i in ra:
        ra_new.append(i / 15)

    # converts all list values to numbers and RA/Dec coordinates and magnitudes to numbers with limited decimal places
    ra_final = conversion(ra_new)
    dec_new = conversion(dec)
    bmag_new = _format_decimal_list(bmag, 3)
    e_bmag_new = _format_decimal_list(e_bmag, 3)
    vmag_new = _format_decimal_list(vmag, 3)
    e_vmag_new = _format_decimal_list(e_vmag, 3)
    gmag_new = _format_decimal_list(gmag, 3)
    e_gmag_new = _format_decimal_list(e_gmag, 3)
    rmag_new = _format_decimal_list(rmag, 3)
    e_rmag_new = _format_decimal_list(e_rmag, 3)
    imag_new = _format_decimal_list(imag, 3)
    e_imag_new = _format_decimal_list(e_imag, 3)

    # places all lists into a DataFrame to paste into a text file for comparison star finder
    df = pd.DataFrame({
        "RA": ra_final,
        "Dec": dec_new,
        "Bmag": bmag_new,
        "e_Bmag": e_bmag_new,
        "Vmag": vmag_new,
        "e_Vmag": e_vmag_new,
        "g'mag": gmag_new,
        "e_g'mag": e_gmag_new,
        "r'mag": rmag_new,
        "e_r'mag": e_rmag_new,
        "i'mag": imag_new,
        "e_i'mag": e_imag_new
    })

    return df


def _format_decimal_list(values, places):
    """
    Formats numeric values as strings with a fixed number of decimal places.

    :param values: Sequence of numeric values
    :param places: Number of decimal places to preserve
    :return: List of formatted numeric strings
    """
    formatted_values = []
    for value in values:
        try:
            numeric_value = float(value)
            formatted_values.append(format(numeric_value, f".{places}f"))
        except (TypeError, ValueError):
            formatted_values.append("nan")
    return formatted_values


def save_to_file(df, filepath):
    """
    Saves the dataframe to a text file

    :param df: Dataframe of the Vizier catalog
    :param filepath: File pathway to where the user wants to save the file
    :return:
    """

    df.to_csv(filepath, index=None)


def catalog_finder(ra, dec, pipeline, folder_path, obj_name, write_callback, cancel_event):
    """
    Finds the APASS catalog for the user to determine comparison stars

    :param ra: Right ascension of the object
    :param dec: Declination of the object
    :param pipeline: Boolean for if pipeline is being used
    :param folder_path: Folder pathway to where files get saved
    :param obj_name: Object name for pipeline
    :param write_callback: Function to write log messages to the GUI
    :param cancel_event:

    :return: Text file pathway, RA, and DEC
    """

    log = make_logger(write_callback)

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return

        ra_input = [_to_decimal_coord(ra)]
        dec_input = [_to_decimal_coord(dec)]

        result = query_vizier(ra_input[0], dec_input[0], write_callback, cancel_event)
        log("Processing data from the Vizier catalog.")
        df = process_data(result)

        text_file = os.path.join(folder_path, "APASS_" + obj_name + "_catalog.txt")

        save_to_file(df, text_file)

        log("Finished cataloging the Vizier results.")
        return text_file, ra_input[0], dec_input[0], ra, dec
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def create_header(ra, dec):
    """
    Creates the header string for the RADEC file.

    :param ra: Right ascension of the object of interest
    :param dec: Declination of the object of interest

    :return: The header string for the RADEC file
    """
    header = "#RA in decimal or sexagesimal HOURS\n" \
             "#Dec in decimal or sexagesimal DEGREES\n" \
             "#Ref Star=0,1,missing (0=target star, 1=ref star, missing->first ap=target, others=ref)\n" \
             "#Centroid=0,1,missing (0=do not centroid, 1=centroid, missing=centroid)\n" \
             "#Apparent Magnitude or missing (value = apparent magnitude, or value > 99 or missing = no mag info)\n" \
             "#Add one comma separated line per aperture in the following format:\n"
    header += "#RA, Dec, Ref Star, Centroid, Magnitude\n"
    header += (
        f"{_format_target_coord(ra, is_ra=True)}, "
        f"{_format_target_coord(dec, is_ra=False)}, 0, 1, 99.999\n"
    )

    return header


# --- _format_target_coord ---
# Formats target coordinates while preserving input sexagesimal precision.
def _format_target_coord(value, is_ra):
    """
    Formats a target coordinate value for RADEC header output.

    :param value: Coordinate value in sexagesimal string or decimal format
    :param is_ra: True if coordinate is right ascension, False for declination
    :return: Formatted coordinate string
    """
    include_plus = isinstance(value, str) and value.strip().startswith("+")
    return _format_sexagesimal_coord(value, is_ra=is_ra, include_plus=include_plus)


# --- _format_sexagesimal_coord ---
# Formats coordinates as HH:MM:SS.sss or +/-DD:MM:SS.sss.
def _format_sexagesimal_coord(value, is_ra, include_plus=False):
    """
    Formats coordinate text for RADEC files with fixed-width sexagesimal components.

    :param value: Coordinate value in sexagesimal string or decimal format
    :param is_ra: True for RA (hours), False for Dec (degrees)
    :param include_plus: Include '+' sign for positive Dec when True
    :return: Normalized sexagesimal coordinate string
    """
    decimal_value = _to_decimal_coord(value)

    sign = ""
    if not is_ra:
        if decimal_value < 0:
            sign = "-"
        elif include_plus:
            sign = "+"

    absolute_value = abs(decimal_value)
    major = int(absolute_value)
    minutes_float = (absolute_value - major) * 60.0
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60.0, 3)

    if seconds >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        major += 1

    return f"{sign}{major:02d}:{minutes:02d}:{seconds:06.3f}"


# --- _to_decimal_coord ---
# Converts coordinate input into decimal hours/degrees.
def _to_decimal_coord(value):
    """
    Converts a coordinate value to decimal representation.

    :param value: Coordinate value in sexagesimal string or decimal format
    :return: Decimal coordinate value
    """
    if isinstance(value, str):
        stripped = value.strip()
        if ":" in stripped:
            parts = stripped.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid sexagesimal coordinate: {value}")

            major = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])

            sign = -1.0 if major < 0 else 1.0
            major_abs = abs(major)
            decimal_value = major_abs + (minutes / 60.0) + (seconds / 3600.0)
            return sign * decimal_value
        return float(stripped)
    return float(value)


def _format_radec_magnitude(value):
    """
    Formats magnitude values for RADEC file output with consistent precision.

    :param value: Magnitude value
    :return: Formatted magnitude string
    """
    if isNaN(value):
        return "99.999"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "99.999"

    if numeric_value >= 99:
        return "99.999"

    return format(numeric_value, ".3f")


def create_lines(ra_list, dec_list, mag_list, ra, dec, filt):
    """
    Creates the data lines string for the RADEC file.

    :param ra_list: The list of right ascensions
    :param dec_list: The list of declinations
    :param mag_list: The list of magnitudes
    :param ra: Right ascension of the object of interest
    :param dec: Declination of the object of interest
    :param filt: The filter for which the RADEC file is being created

    :return: The data lines string for the RADEC file
    """
    lines = ""
    ra_decimal = np.array([_to_decimal_coord(value) for value in ra_list], dtype=float)
    dec_decimal = np.array([_to_decimal_coord(value) for value in dec_list], dtype=float)

    for count, val in enumerate(ra_list):
        next_ra = float(ra_decimal[count])
        next_dec = float(dec_decimal[count])

        # Check where the RA and DEC given by the user at the beginning is in the file to make sure there is no
        # duplication
        angle = angle_dist(float(ra), float(dec), next_ra, next_dec)
        if angle:
            mag_text = _format_radec_magnitude(mag_list[count])
            ra_text = _format_sexagesimal_coord(val, is_ra=True)
            dec_text = _format_sexagesimal_coord(dec_list[count], is_ra=False)
            lines += ra_text + ", " + dec_text + ", " + "1, 1, " + mag_text + "\n"

    return lines


def create_radec(
    df,
    ra,
    dec,
    T_list,
    pipeline,
    folder_path,
    obj_name,
    write_callback,
    cancel_event,
    target_ra_text=None,
    target_dec_text=None
):
    """
    Creates a RADEC file for all 3 filters (Johnson B, V, Cousins R, and T)

    :param df: input catalog DataFrame
    :param ra: user entered RA for system
    :param dec: user entered DEC for system
    :param T_list: TESS magnitudes for comparison stars
    :param pipeline: True if pipeline, False if not
    :param folder_path: folder path for saving RADEC files
    :param obj_name: object name for saving RADEC files
    :param write_callback: Function to write log messages to the GUI
    :param cancel_event:

    :param target_ra_text: Optional original RA input string for header output
    :param target_dec_text: Optional original DEC input string for header output
    :return: None but saves the RADEC files to user specified locations
    """

    log = make_logger(write_callback)

    try:
        if cancel_event is not None and cancel_event.is_set():
            log("Task canceled.")
            return
        filters = ["B", "V", "R", "I", "g", "r", "i", "T"]
        mag_cols = [3, 5, 7, 9, 11, 13, 15, T_list]

        ra_list = df[1]
        dec_list = df[2]

        header_ra = target_ra_text if target_ra_text is not None else ra
        header_dec = target_dec_text if target_dec_text is not None else dec
        header = create_header(header_ra, header_dec)

        # to write lines to the file in order create new RADEC files for each filter
        file_list = []
        for fcount, filt in enumerate(filters):
            if cancel_event is not None and cancel_event.is_set():
                log("Task canceled.")
                return
            if filt != "T":
                mag_list = df[mag_cols[fcount]]
            else:
                mag_list = mag_cols[fcount]

            lines = create_lines(ra_list, dec_list, mag_list, ra, dec, filt)

            output = header + lines

            # Use safe filenames for case-insensitive filesystems (macOS/Windows)
            # to prevent R/r and I/i from overwriting each other
            safe_filt = filt
            if filt == "g":
                safe_filt = "g_prime"
            elif filt == "r":
                safe_filt = "r_prime"
            elif filt == "i":
                safe_filt = "i_prime"

            outputfile = os.path.join(folder_path, obj_name + "_" + safe_filt)

            with open(outputfile + ".radec", "w") as file:
                file.write(output)

            file_list.append(outputfile + ".radec")

        log("Finished writing RADEC files for Johnson B, Johnson V, Cousins R, Cousins I, g', r', i', and TESS magnitudes.")

        return file_list
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def overlay(df, tar_ra, tar_dec, fits_file, folder_path: str = "", obj_name: str = "", write_callback=None):
    """
    Creates an overlay of a science image with APASS objects numbered as seen in the catalog file that
    was saved previously

    :param tar_dec: target declination
    :param tar_ra: target right ascension
    :param df: input catalog DataFrame
    :param folder_path: Directory path to save the overlay image
    :param obj_name: Object name used to generate the overlay filename
    :param write_callback: Optional callback to write log messages
    :return: File path to saved overlay image, or None when no FITS image is available
    """
    log = make_logger(write_callback)

    if not fits_file:
        log("No science image provided. Skipping overlay.")
        return

    if not os.path.exists(fits_file):
        log(f"Science image path does not exist: {fits_file}. Skipping overlay.")
        return

    # If a directory is given, find the first FITS file within it
    if os.path.isdir(fits_file):
        fits_extensions = ('.fits', '.fts', '.fit', '.FITS', '.FTS', '.FIT')
        candidates = sorted(f for f in os.listdir(fits_file) if f.endswith(fits_extensions))
        if not candidates:
            log(f"No FITS files found in {fits_file}. Skipping overlay.")
            return
        fits_file = os.path.join(fits_file, candidates[0])

    # get the image data for plotting purposes
    with fits.open(fits_file) as header_data_unit_list:
        image = header_data_unit_list[0].data
        header = header_data_unit_list[0].header

    # set variables to lists
    index_num = list(df[0])
    ra_catalog = list(df[1])
    dec_catalog = list(df[2])

    # convert the lists to degrees for plotting purposes
    ra_cat_new = (np.array([_to_decimal_coord(value) for value in ra_catalog], dtype=float) * 15) * u.deg
    dec_cat_new = np.array([_to_decimal_coord(value) for value in dec_catalog], dtype=float) * u.deg

    # text for the caption below the graph
    txt = "Number represents index value given in the final output catalog file."

    # Calculate the zscale interval for the image
    zscale = ZScaleInterval()

    # Calculate vmin and vmax for the image display
    vmin, vmax = zscale.get_limits(image)

    # plot the image and the overlays
    wcs = WCS(header)
    fig = Figure(figsize=(12, 8))
    fig.text(.5, 0.02, txt, ha='center')
    ax = fig.add_subplot(projection=wcs)
    ax.imshow(image, origin='lower', cmap='cividis', aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')

    ax.scatter(ra_cat_new, dec_cat_new, transform=ax.get_transform('fk5'), s=200,
               edgecolor='red', facecolor='none', label="Potential Comparison Stars")
    ax.scatter((tar_ra * 15) * u.deg, tar_dec * u.deg, transform=ax.get_transform('fk5'), s=200,
               edgecolor='green', facecolor='none', label="Target Star")

    count = 0
    # annotates onto the image the index number and Johnson V magnitude
    for x, y in zip(ra_cat_new, dec_cat_new):
        px, py = wcs.wcs_world2pix(x, y, 0.)
        ax.annotate(str(index_num[count]), xy=(px + 30, py - 50), color="white", fontsize=12)
        count += 1

    ax.invert_xaxis()
    ax.legend(bbox_to_anchor=(1.45, 1.01), fancybox=False, shadow=False)

    output_dir = folder_path if folder_path else os.path.dirname(os.path.abspath(fits_file))
    overlay_name = f"{obj_name}_overlay.png" if obj_name else "overlay.png"
    overlay_path = os.path.join(output_dir, overlay_name)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(overlay_path, bbox_inches='tight', dpi=256)

    return overlay_path


def calculations(B, V, g, r, i, e_B, e_V, e_g, e_r, e_i, count):
    """
    Calculates (O-C) values

    :param B: Johnson B magnitude
    :param V: Johnson V magnitude
    :param g: g' mag
    :param r: r' mag
    :param i: i' mag
    :param e_B: error of Johnson B mag
    :param e_V: error of Johnson V mag
    :param e_g: error of g' mag
    :param e_r: error of r' mag
    :param e_i: error of i' mag
    :param count: the number that the iteration is on to pick the correct values from lists

    :return: root_rc, val_rc, root_ic, val_ic - mag and error respectively for Rc and Ic
    """
    # predefined values DO NOT change.
    # See Rogers et al 2018, Improved Transformation Equations for Main Sequence Stars
    # See eqn (2):  https://arxiv.org/pdf/astro-ph/0609736
    alpha = 0.278
    e_alpha = 0.016
    beta = 1.321
    e_beta = 0.03
    gamma = 0.219

    # separates the equation out into more easily readable sections
    numerator = alpha * (float(B[count]) - float(V[count])) - gamma - float(g[count]) + float(r[count])
    div = numerator / beta
    val_rc = float(V[count]) + div

    b_v_err = np.sqrt(float(e_B[count]) ** 2 + float(e_V[count]) ** 2)
    b_v_alpha_err = np.abs(alpha * (float(B[count]) - float(V[count]))) * np.sqrt(
        (e_alpha / alpha) ** 2 + (b_v_err / (float(B[count]) - float(V[count]))) ** 2)

    numerator_err = np.sqrt(b_v_alpha_err ** 2 + float(e_g[count]) ** 2 + float(e_r[count]) ** 2)
    div_e = np.abs(div) * np.sqrt((numerator_err / numerator) ** 2 + (e_beta / beta) ** 2)

    root_rc = np.sqrt(div_e ** 2 + float(e_V[count]) ** 2)

    # predefined values DO NOT change.
    # See Rogers et al 2018, Improved Transformation Equations for Main Sequence Stars
    # See eqn (3):  https://arxiv.org/pdf/astro-ph/0609736
    delta = 1.000
    e_delta = 0.006
    epsilon = 0.212

    val_ic = val_rc - (float(r[count]) - float(i[count]) + epsilon) / delta

    e_ic_squared = root_rc ** 2 + (float(e_r[count]) ** 2 + float(e_i[count]) ** 2) / (delta ** 2) + \
                   (((float(r[count]) - float(i[count]) + epsilon) / (delta ** 2)) * e_delta) ** 2
    root_ic = np.sqrt(e_ic_squared)

    return root_rc, val_rc, root_ic, val_ic


def angle_dist(x1, y1, x2, y2):
    """
    Determines whether the two sets of coordinates are the same position or not.
    This is primarily to not output the target star twice in the same RADEC file.

    :param x1: ra coordinate in degrees
    :param y1: dec coordinate in degrees
    :param x2: ra coordinate in degrees
    :param y2: dec coordinate in degrees

    :return: True or False for the equality
    """
    # noinspection PyUnresolvedReferences
    radial = pyasl.getAngDist(x1, y1, x2, y2)
    # print(f"Comparing ({x1}, {y1}) to ({x2}, {y2}), Radial distance: {radial}")
    if 0.15 > radial >= 0.00:  # Exclude exact target and include values within 15 arcminutes (0.25 degrees)
        return True
    else:
        return False

# comparison_selector()
# overlay("test_cat.txt", "00:28:27.9684836736", "78:57:42.657327180")
# find_comp()
