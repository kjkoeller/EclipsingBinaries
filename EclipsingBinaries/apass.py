"""
Combines all APASS programs that were originally separate on GitHub for an easy editing and less to load per file.

Author: Kyle Koeller
Created: 12/26/2022
Last Updated: 05/22/2025
"""

from astroquery.vizier import Vizier
import numpy as np
import pandas as pd

import astropy.units as u
import astropy.coordinates as coord
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from numba import jit
import matplotlib.pyplot as plt
import warnings
from PyAstronomy import pyasl

from .gaia import tess_mag as ga
from .vseq_updated import isNaN, conversion, splitter, decimal_limit

# Helps with MacOS SSL certificates
import ssl
import certifi

ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# turn off this warning that just tells the user,
# "The warning raised when the contents of the FITS header have been modified to be standards compliant."
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


def comparison_selector(ra="", dec="", pipeline=False, folder_path="", obj_name="", science_image="",
                        write_callback=None, cancel_event=None):
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

        log("Starting the process for finding APASS stars.")
        apass_file, input_ra, input_dec, T_list = cousins_r(ra, dec, pipeline, folder_path, obj_name, write_callback,
                                                            cancel_event)
        df = pd.read_csv(apass_file, header=None, skiprows=[0], sep="\t")

        log("Finished Saving")
        # print(
        #     "The output file you have entered has RA and DEC for stars and their B, V, Cousins R, and TESS T magnitudes "
        #     "with their respective errors.\n")

        radec_list = create_radec(df, input_ra, input_dec, T_list, pipeline, folder_path, obj_name, write_callback,
                                  cancel_event)

        overlay(df, input_ra, input_dec, science_image)

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

    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)

    try:
        if cancel_event.is_set():
            log("Task canceled before starting.")
            return

        # predefined values DO NOT change
        alpha = 0.278
        e_alpha = 0.016
        beta = 1.321
        e_beta = 0.03
        gamma = 0.219

        input_file, input_ra, input_dec = catalog_finder(ra, dec, pipeline, folder_path, obj_name, write_callback,
                                                         cancel_event)
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

        Rc = []
        e_Rc = []
        count = 0

        # loop that goes through each value in B to get the total amount of values to be calculated
        log("Calculating Cousins R filter values.")
        for i in B:
            root, val = calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count)
            if isNaN(val) is True:
                # if the value is nan then append 99.999 to the R_c value and its error to make it obvious that there is
                # no given value
                Rc.append(99.999)
                e_Rc.append(99.999)
            else:
                # if there is a value then format that value with only 2 decimal places otherwise there will be like 8
                Rc.append(format(val, ".2f"))
                e_Rc.append(format(root, ".2f"))
            count += 1

        ra_decimal = np.array(splitter(ra))
        dec_decimal = np.array(splitter(dec))
        log("Starting Gaia Search for TESS Magnitudes\n")
        T_list, T_err_list = ga(ra_decimal, dec_decimal, write_callback, cancel_event)

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
            "TMag": T_list,
            "e_TMag": T_err_list
        })

        output_file = folder_path + "\\APASS_" + obj_name + "_Rc.txt"
        # noinspection PyTypeChecker
        final.to_csv(output_file, index=True, sep="\t")
        log("Completed Cousins R calculations.")

        return output_file, input_ra, input_dec, T_list
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

        log("Starting the query for the Vizier catalog.")

        # Query Vizier here and return result
        result = Vizier(
            columns=['_RAJ2000', '_DEJ2000', 'Vmag', "e_Vmag", 'Bmag', "e_Bmag", "g'mag", "e_g'mag", "r'mag",
                     "e_r'mag"],
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

    one = 0
    # pastes all variables into a list for future use
    for i in range(0, len(table_list) - 1):
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

        one += 1

    # converts degree RA to Hour RA
    ra_new = []
    for i in ra:
        ra_new.append(i / 15)

    # converts all list values to numbers and RA/Dec coordinates and magnitudes to numbers with limited decimal places
    ra_final = conversion(ra_new)
    dec_new = conversion(dec)
    bmag_new = decimal_limit(bmag)
    e_bmag_new = decimal_limit(e_bmag)
    vmag_new = decimal_limit(vmag)
    e_vmag_new = decimal_limit(e_vmag)
    gmag_new = decimal_limit(gmag)
    e_gmag_new = decimal_limit(e_gmag)
    rmag_new = decimal_limit(rmag)
    e_rmag_new = decimal_limit(e_rmag)

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
        "e_r'mag": e_rmag_new
    })

    return df


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

        ra_input = splitter([ra])
        dec_input = splitter([dec])

        result = query_vizier(ra_input[0], dec_input[0], write_callback, cancel_event)
        log("Processing data from the Vizier catalog.")
        df = process_data(result)

        text_file = folder_path + "\\APASS_" + obj_name + "_catalog.txt"

        save_to_file(df, text_file)

        log("Finished cataloging the Vizier results.")
        return text_file, ra_input[0], dec_input[0]
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
    header += str(conversion([ra])[0]) + ", " + str(conversion([dec])[0]) + ", 0, 1, 99.999\n"

    return header


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
    ra_decimal = np.array(splitter(ra_list))
    dec_decimal = np.array(splitter(dec_list))

    for count, val in enumerate(ra_list):
        next_ra = float(ra_decimal[count])
        next_dec = float(dec_decimal[count])

        # Check where the RA and DEC given by the user at the beginning is in the file to make sure there is no
        # duplication
        angle = angle_dist(float(ra), float(dec), next_ra, next_dec)
        if angle:
            lines += str(val) + ", " + str(dec_list[count]) + ", " + "1, 1, " + str(mag_list[count]) + "\n"

    return lines


def create_radec(df, ra, dec, T_list, pipeline, folder_path, obj_name, write_callback, cancel_event):
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

    :return: None but saves the RADEC files to user specified locations
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
        filters = ["B", "V", "R", "T"]
        mag_cols = [3, 5, 7, T_list]

        ra_list = df[1]
        dec_list = df[2]

        header = create_header(ra, dec)

        log("The 'T' filter is the calibrated TESS magnitudes calculated from Gaia magnitudes. Please go to the GitHub "
            "page for more information.\n")

        # to write lines to the file in order create new RADEC files for each filter
        file_list = []
        for fcount, filt in enumerate(filters):
            if cancel_event.is_set():
                log("Task canceled.")
                return
            if filt != "T":
                mag_list = df[mag_cols[fcount]]
            else:
                mag_list = mag_cols[fcount]

            lines = create_lines(ra_list, dec_list, mag_list, ra, dec, filt)

            output = header + lines
            outputfile = folder_path + "\\" + obj_name + "_" + filt

            with open(outputfile + ".radec", "w") as file:
                file.write(output)

            file_list.append(outputfile + ".radec")

        log("Finished writing RADEC files for Johnson B, Johnson V, and Cousins R, and T.\n")

        return file_list
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def overlay(df, tar_ra, tar_dec, fits_file):
    """
    Creates an overlay of a science image with APASS objects numbered as seen in the catalog file that
    was saved previously

    :param tar_dec: target declination
    :param tar_ra: target right ascension
    :param df: input catalog DataFrame
    :return: None but displays a science image with over-layed APASS objects
    """
    # NSVS_254037-S001-R004-C001-Empty-R-B2.fts
    # fits_file = input("Enter file pathway to one of your science image files for creating an overlay or "
    #                   "comparison stars: ")

    # get the image data for plotting purposes
    header_data_unit_list = fits.open(fits_file)
    image = header_data_unit_list[0].data
    header = header_data_unit_list[0].header

    # set variables to lists
    index_num = list(df[0])
    ra_catalog = list(df[1])
    dec_catalog = list(df[2])

    # convert the lists to degrees for plotting purposes
    ra_cat_new = (np.array(splitter(ra_catalog)) * 15) * u.deg
    dec_cat_new = np.array(splitter(dec_catalog)) * u.deg

    # text for the caption below the graph
    txt = "Number represents index value given in the final output catalog file."

    # Calculate the zscale interval for the image
    zscale = ZScaleInterval()

    # Calculate vmin and vmax for the image display
    vmin, vmax = zscale.get_limits(image)

    # plot the image and the overlays
    wcs = WCS(header)
    fig = plt.figure(figsize=(12, 8))
    fig.text(.5, 0.02, txt, ha='center')
    ax = plt.subplot(projection=wcs)
    plt.imshow(image, origin='lower', cmap='cividis', aspect='equal', vmin=vmin, vmax=vmax)
    plt.xlabel('RA')
    plt.ylabel('Dec')

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
        plt.annotate(str(index_num[count]), xy=(px + 30, py - 50), color="white", fontsize=12)
        count += 1

    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.45, 1.01), fancybox=False, shadow=False)
    plt.show()


@jit(forceobj=True)
def calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count):
    """
    Calculates (O-C) values

    :param i: i' mag
    :param V: Johnson V magnitude
    :param g: g' mag
    :param r: r' mag
    :param gamma: coefficient from paper
    :param beta: coefficient from paper
    :param e_beta: error of beta
    :param alpha: coefficient from paper
    :param e_alpha: error of alpha
    :param e_B: error of Johnson B mag
    :param e_V: error of Johnson V mag
    :param e_g: error of g' mag
    :param e_r: error of r' mag
    :param count: the number that the iteration is on to pick the correct values from lists

    :return: root, val - mag and error respectively
    """
    # separates the equation out into more easily readable sections
    numerator = alpha * (float(i) - float(V[count])) - gamma - float(g[count]) + float(r[count])
    div = numerator / beta
    val = float(V[count]) + div

    b_v_err = np.sqrt(float(e_B[count]) ** 2 + float(e_V[count]) ** 2)
    b_v_alpha_err = np.abs(alpha * (float(i) - float(V[count]))) * np.sqrt(
        (e_alpha / alpha) ** 2 + (b_v_err / (float(i) - float(V[count]))) ** 2)

    numerator_err = np.sqrt(b_v_alpha_err ** 2 + float(e_g[count]) ** 2 + float(e_r[count]) ** 2)
    div_e = np.abs(div) * np.sqrt((numerator_err / numerator) ** 2 + (e_beta / beta) ** 2)

    root = np.sqrt(div_e ** 2 + float(e_V[count]) ** 2)

    return root, val


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
    if 0.15 > radial > 0.01:  # Exclude exact target and include values within 15 arcminutes (0.25 degrees)
        return True
    else:
        return False

# comparison_selector()
# overlay("test_cat.txt", "00:28:27.9684836736", "78:57:42.657327180")
# find_comp()
