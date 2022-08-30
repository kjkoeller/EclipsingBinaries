"""
Author: Kyle Koeller
Created: 4/13/2022
Last Updated: 8/29/2022
Based on this paper: https://arxiv.org/pdf/astro-ph/0609736.pdf

This program calculates a Cousins R (R_c) filter band value from a given Johnson V and B, and g' and r'.
"""

import pandas as pd
import numpy as np
import APASS_catalog_finder as apass


# noinspection PyUnboundLocalVariable
def main():
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
    
    input_file = apass.main()
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
        print("Please run the catalog finder first to get these values from the APASS database before running this "
              "program.")
        exit()

    Rc = []
    e_Rc = []
    count = 0

    test = 10.551 + (((0.278 * 0.682) - 0.219 - 10.919 + 10.395) / 1.321)
    total_Rc(test)

    # loop that goes through each value in B to get the total amount of values to be calculated
    for i in B:
        # separates the equation out into more easily readable sections
        div = (alpha * (float(i) - float(V[count])) - gamma - float(g[count]) + float(r[count])) / beta
        val = float(V[count]) + div

        b_v = ((float(i) - float(V[count])) * e_alpha) ** 2
        v_rc = ((float(V[count]) - val) * e_beta) ** 2
        beta_alpha = ((beta - alpha) * float(e_V[count])) ** 2

        # full equation given in the cited paper at the top of the program file
        root = np.sqrt(b_v + v_rc + float(e_g[count]) ** 2 + float(e_r[count]) ** 2 + (
                    alpha * float(e_B[count])) ** 2 + beta_alpha)

        if isNaN(val) is True:
            # if the value is nan then append 99.999 to the R_c value and its error to make it obvious that there is
            # no given value
            Rc.append(99.999)
            e_Rc.append(99.999)
        else:
            # if there is a value then format that value with only 2 decimal places otherwise there will be like 8
            Rc.append(format(val, ".2f"))
            e_Rc.append(format((1 / beta) * root, ".2f"))
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
    output_file = input("Enter an output file name (i.e. 'APASS_254037_Rc_values.txt): ")
    # noinspection PyTypeChecker
    final.to_csv(output_file, index=None, sep="\t")
    print("Finished Saving")
    print()

    return output_file


def isNaN(num):
    """
    Checks if a value is nan

    :param num: value to be checked
    :return: Boolean True or False
    """

    return num != num


def total_Rc(Rc):
    assert Rc == 10.132071915215745


if __name__ == '__main__':
    main()
