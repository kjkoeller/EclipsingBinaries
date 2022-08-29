"""
Created: 2/28/2022
Last Updated: 8/29/2022
Author: Kyle Koeller
Python Version 3.9

This program is meant to make the process of collecting the BJD, rel flux, and the rel flux error from AIJ excel
spreadsheets faster. The user enters however many nights they have and the program goes through and checks those text
files for the different columns for: BJD, rel flux, rel flux error from TESS.

There are error catching statements within the program so if the user mistypes, the program will not crash and
close on them (hopefully).
"""

import pandas as pd
from os import path


def main(c):
    """
    Calls the main functions of the program and uses a counter in order to not re-print so many things over and over
    again in case of a miss enter in file path.

    :param c: Counter to check if this is starting the program or iteration from within the program.
    :return: none
    """
    if c == 0:
        # warning prompts for the user to read to make sure this program works correctly
        print()
        print("Make sure you have turned the output xls files from AIJ into tab delimited text files. "
              "Since these xls files are corrupt for reading directly from.")
        print("You will also need to go into each night and filter and "
              "make the HJD column 6 decimals instead of the output of 3 within Excel.")
        print()
    else:
        print()

    num = 0
    test = True
    while test:
        try:
            num = int(input("Number of nights you have: "))
            test = False
        except ValueError:
            print("Please enter a number.")
            print()

    print()
    get_filters(num)


def get_filters(n):
    """
    Takes a number of nights for a given filter and takes out the HJD, either A_Mag1 or T1_flux, and
    error for mag or flux

    :param n: Number of observation nights
    :return: the output text files for each night in a given filter
    """
    total_bjd = []
    total_rel_flux = []
    total_rel_flux_err = []
    # checks for either the b, v, r filter as either upper or lowercase will work
    for i in range(n):
        while True:
            # makes sure the file pathway is real and points to some file
            # (does not check if that file is the correct one though)
            try:
                # an example pathway for the files
                print(r"Example: E:\Research\Data\NSVS_254037\2018.10.12-reduced\Check\V\2018.10.12.APASS"
                      ".V_measurements.txt")
                file = input("Enter night %d file path: " % (i + 1))
                if path.exists(file):
                    break
                else:
                    continue
            except FileNotFoundError:
                print("Please enter a correct file path")

        # noinspection PyUnboundLocalVariable
        df = pd.read_csv(file, delimiter="\t")

        # set parameters to lists from the file by the column header
        bjd = []
        rel_flux = []
        rel_flux_err = []
        try:
            bjd = list(df["BJD_TDB"])
            rel_flux = list(df["rel_flux_T1"])
            rel_flux_err = list(df["rel_flux_err_T1"])
        except KeyError:
            print("The file you entered does not have the columns of BJD_TDB, rel_flux_T1, or rel_flux_err_T1. Please "
                  "re enter the file path and make sure its the correct file.")
            c = 1
            main(c)

        total_bjd.append(bjd)
        total_rel_flux.append(rel_flux)
        total_rel_flux_err.append(rel_flux_err)

    # converts the Dataframe embedded lists into a normal flat list
    new_bjd = [item for elem in total_bjd for item in elem]
    new_rel_flux = [item for elem in total_rel_flux for item in elem]
    new_rel_flux_err = [item for elem in total_rel_flux_err for item in elem]

    # outputs the new file to dataframe and then into a text file for use in Peranso or PHOEBE
    data = pd.DataFrame({
        "BJD": new_bjd,
        "rel_flux": new_rel_flux,
        "rel_flux_err": new_rel_flux_err
    })
    print("")
    output = input("What is the file output name (with file extension .txt): ")

    data.to_csv(output, index=False, header=False, sep='\t')
    print("")
    print("Fished saving the file to the same location as this program.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    count = 0
    main(count)
