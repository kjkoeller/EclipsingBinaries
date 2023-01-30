"""
Combines all APASS programs that were originally separate on GitHub for an easy editing and less to load per file.

Author: Kyle Koeller
Created: 12/26/2022
Last Updated: 01/30/2023
"""

from astroquery.vizier import Vizier
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
from astropy.wcs import WCS
from astropy.io import fits
from PyAstronomy import pyasl
from numba import jit
import matplotlib.pyplot as plt


def catalog_finder():
    """
    This looks at a region of the sky at the decimal coordinates of an object and gathers the "column" data with
    "column filters"
    You can change the columns or the column filters to whatever you like, I have these set as they are because of the
    telescope that was used (Rooftop)

    I also set the width of the search radius to larger than what you would actually see in the field of view of any
    telescope we use to make sure it gathers all stars within the star field

    Main things to change are:
    "columns"- what factors are actually taken from the online catalog
    "column_filters"- which stars are to actually be extracted from that online data table and narrows list down from a
    couple of hundred to like 50-60
    "ra"/"dec"- must be in decimal notation
    "width"- set to the notation that is currently set as, but you may change the number being used
            30m = 30 arc-minutes
    """
    # catalog is II/336/apass9
    # 00:28:27.9684836736 78:57:42.657327180
    # 78:57:42.657327180
    ra_input = input("Enter the RA of your system (HH:MM:SS.SSSS): ")
    dec_input = input("Enter the DEC of your system (DD:MM:SS.SSSS or -DD:MM:SS.SSSS): ")
    print()

    ra_input2 = splitter([ra_input])
    dec_input2 = splitter([dec_input])

    result = Vizier(
        columns=['_RAJ2000', '_DEJ2000', 'Vmag', "e_Vmag", 'Bmag', "e_Bmag", "g'mag", "e_g'mag", "r'mag", "e_r'mag"],
        row_limit=-1,
        column_filters=({"Vmag": "<14", "Bmag": "<14"})).query_region(
        coord.SkyCoord(ra=ra_input2[0], dec=dec_input2[0], unit=(u.h, u.deg), frame="icrs"),
        width="30m", catalog="APASS")

    tb = result['II/336/apass9']

    # converts the table result to a list format for putting values into lists
    table_list = []
    for i in tb:
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

    # saves the dataframe to a text file and prints that dataframe out to easily see what was copied to the text file
    print("\nThis output file contains all the Vizier magnitudes that will be used to calculate the Cousins R band, and "
          "should not be used for anything else other than calculation confirmation if needed later on.\n")
    text_file = input("Enter a text file pathway/name for the output comparisons (ex: C:\\folder1\\APASS_254037.txt): ")
    df.to_csv(text_file, index=None)
    print("\nCompleted save.\n")

    return text_file


def comparison_selector():
    """
    This code compares AIJ found stars (given an RA and DEC) to APASS stars to get their respective Johnson B, V, and
    Cousins R values and their respective errors.

    This code is not 100% accurate and will still need the human eye to compare the final list to the AIJ given list. As
    this code can only get down to such an accuracy to be effective in gathering stars to be usable.

    :return: A list of stars that are the most likely to be on the AIJ list of stars
    """
    # reads the text files to be analyzed for comparison star matches between APASS and Simbad
    # apass_file = input("Enter the text file name for the generated APASS stars: ")
    print("\nMust have all files in the same folder as the Python code OR type out the full file pathway to the file.\n")
    radec_file = input("Enter the text file name for the RADEC file from AIJ or type 'Close' to exit the program: ")
    if radec_file.lower() == "close":
        exit()
    apass_file = cousins_r()
    while True:
        test = 0
        try:
            df = pd.read_csv(apass_file, header=None, skiprows=[0], sep="\t")
            dh = pd.read_csv(radec_file, header=None, skiprows=7)
        except FileNotFoundError:
            print("\nOne of the files were not found, please enter them again.\n")
            test = -1
        if test == 0:
            break
        else:
            radec_file = input("Enter the text file name for the RADEC file from AIJ: ")
    # noinspection PyUnboundLocalVariable
    duplicate_df = angle_dist(df, dh)

    ra = duplicate_df[0]
    dec = duplicate_df[1]
    B = duplicate_df[2]
    e_B = duplicate_df[3]
    V = duplicate_df[4]
    e_V = duplicate_df[5]
    R_c = duplicate_df[6]
    e_R_c = duplicate_df[7]

    final = pd.DataFrame({
        "RA": ra,
        "DEC": dec,
        "B": B,
        "e_B": e_B,
        "V": V,
        "e_V": e_V,
        "R_c": R_c,
        "e_R_c": e_R_c
    })

    # prints the output and saves the dataframe to the text file with "tab" spacing
    output_file = input("Enter an output file name (ex: APASS_254037_Catalog.txt): ")
    final.to_csv(output_file, index=True, sep="\t")
    print("Finished Saving\n")
    print("This program is not 100% accurate, so the recommendation is to compare what you found in AIJ to what this "
          "code has found and make sure that the two lists are the same and enter in the filter values manually into"
          "the RADEC file for AIJ to use in the photometry.\n")
    print("The output file you have entered has RA and DEC for stars and their B, V, and Cousins R magnitudes with "
          "their respective errors.\n")

    overlay(output_file, radec_file)


def cousins_r():
    """
    Calculates the Cousins R_c value for a given B, V, g', and r' from APASS

    :return: Outputs a file to be used for R_c values
    """
    # predefined values do not change
    alpha = 0.278
    e_alpha = 0.016
    beta = 1.321
    e_beta = 0.03
    gamma = 0.219

    input_file = catalog_finder()
    df = pd.read_csv(input_file, header=None, skiprows=[0], sep=",")

    # writes the columns from the input file
    try:
        # this try except function checks whether there are just enough columns in the file being loaded and tells
        # the user what they need to do in order to get the correct columns
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
    except KeyError:
        # prints off instructions and then closes the program
        print("The file you have loaded does not have the enough columns.")
        print("Must include RA, DEC, B, V, g', r', and their respective errors.")
        print("Please run this program from the beginning first to get these values from the APASS database.\n")
        exit()

    Rc = []
    e_Rc = []
    count = 0

    test = 10.551 + (((0.278 * 0.682) - 0.219 - 10.919 + 10.395) / 1.321)
    total_Rc(test)

    # loop that goes through each value in B to get the total amount of values to be calculated
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
        "e_Rc": e_Rc
    })

    # saves the dataframe to an entered output file
    output_file = input("Enter an output file name (ex: APASS_254037_Rc_values.txt): ")
    # noinspection PyTypeChecker
    final.to_csv(output_file, index=None, sep="\t")
    print("Finished Saving\n")

    return output_file


def overlay(catalog, radec):
    # NSVS_254037-S001-R004-C001-Empty-R-B2.fts
    fits_file = input("Enter file pathway to one of your image files: ")

    # get the image data for plotting purposes
    header_data_unit_list = fits.open(fits_file)
    image = header_data_unit_list[0].data
    header = header_data_unit_list[0].header

    # read in the catalog and radec files
    df = pd.read_csv(catalog, header=None, skiprows=[0], sep="\t")
    dh = pd.read_csv(radec, header=None, skiprows=7)

    # set variables to lists
    index_num = list(df[0])
    ra_catalog = list(df[1])
    dec_catalog = list(df[2])
    ra_radec = list(dh[0])
    dec_radec = list(dh[1])

    # convert the lists to degrees for plotting purposes
    ra_cat_new = (np.array(splitter(ra_catalog)) * 15) * u.deg
    dec_cat_new = np.array(splitter(dec_catalog)) * u.deg
    ra_radec_new = (np.array(splitter(ra_radec)) * 15) * u.deg
    dec_radec_new = np.array(splitter(dec_radec)) * u.deg

    # text for the caption below the graph
    txt = "Number represents index value given in the final output catalog file."

    # plot the image and the overlays
    wcs = WCS(header)
    fig = plt.figure(figsize=(12, 8))
    fig.text(.5, 0.02, txt, ha='center')
    ax = plt.subplot(projection=wcs)
    plt.imshow(image, origin='lower', cmap='cividis', aspect='equal', vmin=300, vmax=1500)
    plt.xlabel('RA')
    plt.ylabel('Dec')

    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')

    ax.scatter(ra_cat_new, dec_cat_new, transform=ax.get_transform('fk5'), s=200,
               edgecolor='red', facecolor='none', label="Potential Comparison Stars")
    ax.scatter(ra_radec_new, dec_radec_new, transform=ax.get_transform('fk5'), s=200,
               edgecolor='green', facecolor='none', label="AIJ Comparison Stars")

    count = 0
    # annotates onto the image the index number and Johnson V magnitude
    for x, y in zip(ra_cat_new, dec_cat_new):
        px, py = wcs.wcs_world2pix(x, y, 0.)
        plt.annotate(str(index_num[count]), xy=(px+30, py-50), color="white", fontsize=12)
        count += 1

    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.45, 1.01), fancybox=False, shadow=False)
    plt.show()


@jit(forceobj=True)
def calculations(i, V, g, r, gamma, beta, e_beta, alpha, e_alpha, e_B, e_V, e_g, e_r, count):
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


def isNaN(num):
    """
    Checks if a value is nan

    :param num: value to be checked
    :return: Boolean True or False
    """

    return num != num


def total_Rc(Rc):
    assert Rc == 10.132071915215745


def new_list(a):
    """
    Converts lists into number format with minimal decimal places
    :param a: list
    :return: new list with floats
    """
    b = []
    for i in a:
        b.append(float(format(i, ".2f")))
    return b


def angle_dist(df, dh):
    """
    Gathers a list of stars that are very close to the ones given in the RADEC file

    :param df: apass dataframe
    :param dh: radec dataframe
    :return: compared list
    """
    # checks specific columns and adds those values to a list variable for comparison in the nested for loops below
    apass_dec = list(df[1])
    apass_ra = list(df[0])
    simbad_dec = list(dh[1])
    simbad_ra = list(dh[0])

    # converts the RA and Dec coordinate format to decimal format
    apass_split_ra = splitter(apass_ra)
    apass_split_dec = splitter(apass_dec)

    simbad_split_ra = splitter(simbad_ra)
    simbad_split_dec = splitter(simbad_dec)

    comp = pd.DataFrame()
    simbad_count = 0
    # finds the comparison star in both APASS text file and RA and Dec files to an output variable with
    # the RA and Dec noted for magnitude finding
    for i in simbad_split_dec:
        apass_count = 0
        for k in apass_split_dec:
            # noinspection PyUnresolvedReferences
            radial = pyasl.getAngDist(float(apass_split_ra[apass_count]), float(k),
                                      float(simbad_split_ra[simbad_count]),
                                      float(i))
            if radial <= 0.025:
                # comp = comp.append(df.loc[apass_count:apass_count], ignore_index=True)
                comp = pd.concat([comp, df.loc[apass_count:apass_count]])
            apass_count += 1
        simbad_count += 1

    # removes all duplicate rows from the dataframe
    duplicate_df = comp.drop_duplicates()
    try:
        list(duplicate_df[0])
    except KeyError:
        print("There were no comparison stars found between APASS and the RADEC file.\n")
        exit()

    return duplicate_df


def conversion(a):
    """
    Converts decimal RA and DEC to standard output with colons

    :param a: decimal RA or DEC
    :return: truncated version using colons
    """
    b = []

    for i in a:
        num1 = float(i)
        if num1 < 0:
            num2 = abs((num1 - int(num1)) * 60)
            num3 = format((num2 - int(num2)) * 60, ".3f")
            num4 = float(num3)
            b.append(str(int(num1)) + ":" + str(int(num2)) + ":" + str(abs(num4)))
        else:
            num2 = (num1 - int(num1)) * 60
            num3 = format((num2 - int(num2)) * 60, ".3f")
            b.append(str(int(num1)) + ":" + str(int(num2)) + ":" + str(num3))

    return b


def splitter(a):
    """
    Splits the truncated colon RA and DEC from simbad into decimal forms

    :param a:
    :return:
    """
    # makes the coordinate string into a decimal number from the text file
    step = []
    final = []
    for i in a:
        new = i.split(":")
        num1 = int(new[0])
        if num1 < 0:
            num2 = -int(new[1])
            num3 = -int(float(new[2]))
            b = num1 + ((num2 + (num3 / 60)) / 60)
        else:
            num2 = int(new[1])
            num3 = int(float(new[2]))
            b = num1 + ((num2 + (num3 / 60)) / 60)
        step.append(format(b, ".7f"))

    for i in step:
        final.append(float(format(i)))
    return final


def decimal_limit(a):
    """
    Limits the amount of decimals that get written out for further code clarity and ease of code use

    :param a: magnitude
    :return: reduced decimal magnitude
    """
    b = list()
    for i in a:
        num = float(i)
        num2 = format(num, ".2f")
        b.append(num2)
    return b

# comparison_selector()
