"""
Author: Kyle Koeller
Created: 11/11/2020
Last Updated: 07/11/2023

This program is meant to make the process of collecting the different filters from AIJ Excel spreadsheets faster.
The user enters however many nights they have and the program goes through and checks those text files for the
different columns for,HJD, Amag, and Amag error for the B and V filters.

There are error catching statements within the program so if the user mistypes, the program will not crash and
close on them (hopefully)
"""

import pandas as pd
from os import path


def main(c):
    """
    Runs the whole program but collects the number of nights that the user wants to parse through.

    :param c: Number of whether the user has used the program before or not. Must be manually changed within the code
    """
    # warning prompts for the user to read to make sure this program works correctly
    print("\nFrom each night, yous should have a file that is sort of like this: 2018.09.18.APASS.B_datasubset.dat.\n"
          "This file has 7 or 5 columns and you will only need 5 or 3 of them respectively.\n")
    print("All of the nights that you entire MUST have the exact same column number or this program will not work.\n")
    print(
        "You may also type the word 'Close' in the next prompt to leave this program and return to the main menu.\n")
    num = check_num()
    flux_mag(num, c)


def check_num():
    """
    Checks whether the user enters a real number for the number of nights

    :return: returns the entered number
    """
    while True:
        num = input("Number of nights you have: ")
        if num.isnumeric():
            if int(num) > 0:
                break
            else:
                print("You have entered an invalid number. Please try again.\n")
        elif num.lower() == "close":
            exit()
        else:
            print("You have not entered a number or the word 'Close', please try again.\n")
    return int(num)


def file_path(i):
    """
    Checks whether the user enters a real file path

    :param i: the number of the night

    :return: returns the entered file path
    """
    while True:
        file = input("Enter night %d file path: " % (i + 1))
        if path.exists(file):
            break
        else:
            print("You have entered an invalid file path. Please try again.\n")
    df = pd.read_csv(file, delimiter="\t")

    return df


def flux_mag(night, num):
    """
    Finds the flux and magnitude columns from the file

    :param night: the number of the nights
    :param num: which filter is used

    :return:
    """

    total_hjd = []
    total_bjd = []
    total_amag = []
    total_amag_err = []
    total_flux = []
    total_flux_err = []

    for i in range(night):
        df = file_path(i)
        if len(df.columns) == 7:
            # set parameters to lists from the file by the column header
            hjd = []
            bjd = []
            amag = []
            amag_err = []
            flux = []
            flux_err = []
            try:
                if num == 0:
                    hjd = list(df["HJD"])
                elif num == 1:
                    bjd = list(df["BJD_TDB"])
                    hjd = list(df["HJD"])
                amag = list(df["Source_AMag_T1"])
                amag_err = list(df["Source_AMag_Err_T1"])
                flux = list(df["rel_flux_T1"])
                flux_err = list(df["rel_flux_err_T1"])
            except KeyError:
                print("\nThe file you entered does not have the columns of HJD, Source_AMag_T1, or Source_AMag_Err_T1. "
                      "Please re-enter the file path and make sure its the correct file.\n")
                main(0)

            if num == 0:
                total_hjd.append(hjd)
            elif num == 1:
                total_bjd.append(bjd)
                total_hjd.append(hjd)
                new_bjd = [item for elem in total_bjd for item in elem]

            total_amag.append(amag)
            total_amag_err.append(amag_err)
            total_flux.append(flux)
            total_flux_err.append(flux_err)

            # converts the Dataframe embedded lists into a normal flat list
            new_hjd = [item for elem in total_hjd for item in elem]
            new_amag = [item for elem in total_amag for item in elem]
            new_amag_err = [item for elem in total_amag_err for item in elem]
            new_flux = [item for elem in total_flux for item in elem]
            new_flux_err = [item for elem in total_flux_err for item in elem]

            data_amount = 2
        elif len(df.columns == 5):
            # set parameters to lists from the file by the column header
            hjd = []
            amag = []
            amag_err = []
            try:
                if num == 0:
                    hjd = list(df["HJD"])
                elif num == 1:
                    bjd = list(df["BJD_TDB"])
                    hjd = list(df["HJD"])
                amag = list(df["Source_AMag_T1"])
                amag_err = list(df["Source_AMag_Err_T1"])
            except KeyError:
                print("\nThe file you entered does not have the columns of HJD, Source_AMag_T1, or Source_AMag_Err_T1. "
                      "Please re-enter the file path and make sure its the correct file.\n")
                c = 1
                main(c)

            if num == 0:
                total_hjd.append(hjd)
            elif num == 1:
                total_bjd.append(bjd)
                total_hjd.append(hjd)

            total_amag.append(amag)
            total_amag_err.append(amag_err)

            # converts the Dataframe embedded lists into a normal flat list
            new_bjd = [item for elem in total_bjd for item in elem]
            new_hjd = [item for elem in total_hjd for item in elem]
            new_amag = [item for elem in total_amag for item in elem]
            new_amag_err = [item for elem in total_amag_err for item in elem]
            data_amount = 1
        else:
            print("\nThe file you entered does not have the correct amount of columns.\n")
            main(0)

    if data_amount == 1:
        if num == 0:
            data2 = pd.DataFrame({
                "HJD": new_hjd,
                "AMag": new_amag,
                "AMag Error": new_amag_err
            })
        elif num == 1:
            data2 = pd.DataFrame({
                "BJD_TDB": new_bjd,
                "HJD": new_hjd,
                "AMag": new_amag,
                "AMag Error": new_amag_err
            })

        print("")
        output = input("What is the file output name (WITHOUT any file extension): ")

        # output the text files with a designation of magnitude or flux
        data2.to_csv(output + "_magnitudes.txt", index=False, header=False, sep="\t")
        print("")
        print("Fished saving the file to the same location as this program.\n\n")
    elif data_amount == 2:
        # outputs the new file to dataframe and then into a text file for use in Peranso or PHOEBE
        if num == 0:
            data1 = pd.DataFrame({
                "HJD": new_hjd,
                "rel flux": new_flux,
                "rel flux error": new_flux_err
            })

            data2 = pd.DataFrame({
                "HJD": new_hjd,
                "AMag": new_amag,
                "AMag Error": new_amag_err
            })
        elif num == 1:
            data1 = pd.DataFrame({
                "BJD_TDB": new_bjd,
                "HJD": new_hjd,
                "rel flux": new_flux,
                "rel flux error": new_flux_err
            })

            data2 = pd.DataFrame({
                "BJD_TDB": new_bjd,
                "HJD": new_hjd,
                "AMag": new_amag,
                "AMag Error": new_amag_err
            })

        output = input("What is the file output name (WITHOUT any file extension): ")

        # output both text files with a designation of magnitude or flux
        data1.to_csv(output + "_magnitudes.txt", index=False, header=False, sep="\t")
        data2.to_csv(output + "_flux.txt", index=False, header=False, sep="\t")
        print("")


if __name__ == '__main__':
    main(0)
