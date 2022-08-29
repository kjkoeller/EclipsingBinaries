"""
Search the APASS catalog by searching a region of the sky for comparison stars and outputs those comparisons to a file.

Author: Kyle Koeller
Created: 2/8/2021
Last Updated: 8/3/2022
Python Version 3.9
"""

from astroquery.vizier import Vizier
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord


def main():
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
            40m = 40 arc-minutes
    """
    # catalog is II/336/apass9
    # 00:28:27.9684836736
    # 78:57:42.657327180
    ra_input = input("Enter the RA of your system (HH:MM:SS.SSSS): ")
    dec_input = input("Enter the DEC of your system (DD:MM:SS.SS or -DD:MM:SS.SSSS): ")

    ra_input2 = splitter([ra_input])
    dec_input2 = splitter([dec_input])

    result = Vizier(
        columns=['_RAJ2000', '_DEJ2000', 'Vmag', "e_Vmag", 'Bmag', "e_Bmag", "g'mag", "e_g'mag", "r'mag", "e_r'mag"],
        row_limit=-1,
        column_filters=({"Vmag": "<14", "Bmag": "<14"})).query_region(
        coord.SkyCoord(ra=ra_input2[0], dec=dec_input2[0], unit=(u.h, u.deg), frame="icrs"), width="40m", catalog="APASS")

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
    print()
    text_file = input("Enter a text file name for the output comparisons (ex: APASS_3350218.txt): ")
    df.to_csv(text_file, index=None)
    print("Completed save")
    print()

    return text_file


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


if __name__ == '__main__':
    main()
