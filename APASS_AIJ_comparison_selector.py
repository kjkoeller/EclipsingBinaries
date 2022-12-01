"""
Author: Kyle Koeller
Created: 2/8/2022
Last Updated: 12/01/2022

APASS Star comparison finding for the most accurate magnitudes from the list of stars made in AIJ
"""

import pandas as pd
from PyAstronomy import pyasl
import APASS_catalog_finder as APASS_catalog
import cousins_R as cousins
import Overlay_Comparison as overlay


def main():
    """
    This code compares AIJ found stars (given an RA and DEC) to APASS stars to get their respective Johnson B, V, and
    Cousins R values and their respective errors.

    This code is not 100% accurate and will still need the human eye to compare the final list to the AIJ given list. As
    this code can only get down to such an accuracy to be effective in gathering stars to be usable.

    :return: A list of stars that are the most likely to be on the AIJ list of stars
    """
    # reads the text files to be analyzed for comparison star matches between APASS and Simbad
    # apass_file = input("Enter the text file name for the generated APASS stars: ")
    print("Must have all files in the same folder as the Python code OR type out the full file pathway to the file.")
    radec_file = input("Enter the text file name for the RADEC file from AIJ or type 'Close' to exit the program: ")
    if radec_file.lower() == "close":
        exit()
    apass_file = cousins.main()
    while True:
        test = 0
        try:
            df = pd.read_csv(apass_file, header=None, skiprows=[0], sep="\t")
            dh = pd.read_csv(radec_file, header=None, skiprows=7)
        except FileNotFoundError:
            print("File was not found, please enter them again.")
            print()
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
    output_file = input("Enter an output file name (i.e. 'APASS_254037_Catalog.txt): ")
    final.to_csv(output_file, index=True, sep="\t")
    print("Finished Saving")
    print()
    print("This program is not 100% accurate, so the recommendation is to compare what you found in AIJ to what this "
          "code has found and make sure that the two lists are the same and enter in the filter values manually into"
          "the RADEC file for AIJ to use in the photometry.")
    print()
    print("The output file you have entered has RA and DEC for stars and their B, V, and Cousins R magnitudes with "
          "their respective errors.")
    print()

    overlay.main(output_file, radec_file)


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
    apass_split_ra = APASS_catalog.splitter(apass_ra)
    apass_split_dec = APASS_catalog.splitter(apass_dec)

    simbad_split_ra = APASS_catalog.splitter(simbad_ra)
    simbad_split_dec = APASS_catalog.splitter(simbad_dec)

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
        pass
    except KeyError:
        print("There were not comparison stars found between APASS and the RADEC file.")
        exit()

    return duplicate_df


if __name__ == '__main__':
    main()
