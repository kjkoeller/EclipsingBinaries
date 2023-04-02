"""
Author: Kyle Koeller
Created: 12/19/2022
Last Edited: 04/02/2023

This calculates O-C values and produces an O-C plot.
"""

from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import statsmodels.formula.api as smf
import seaborn as sns
from numba import jit


def main():
    print("\n\nThe format of these input files should be the of the raw form given from Dr. Robert Berrginton's"
          " 'find_minimum' C program.")
    print("Run TESS data by itself through the TESS option and filtered SARA/BSUO data through the BSU option.\n"
          "DO NOT combine them in any way unless you have already run them through to get the (O-C) values and"
          "are about to run the 'All Data' option.")
    print("Enter the corresponding number to what you would like to do.\n")
    while True:
        try:
            num = int(input("Would you like to use BSUO data(1), TESS data(2), All Data(3), or Close Program(4): "))
            if num == 1:
                first_time = input("Do you already have an Epoch value 'Yes' or 'No': ")
                if first_time.lower() == "no":
                    T0 = 0
                    To_err = 0
                    period = float(input("Please enter the period for your system: "))
                else:
                    T0, To_err, period = arguments()
                    print("Example file pathway: C:\\folder1\\folder2\\[file name]")
                while True:
                    inB = input("Please enter your times of minimum file pathway for the Johnson B filter: ")
                    inV = input("Please enter your times of minimum file pathway for the Johnson V filter: ")
                    inR = input("Please enter your times of minimum file pathway for the Cousins R filter: ")
                    try:
                        db = pd.read_csv(inB, header=None, delim_whitespace=True)
                        dv = pd.read_csv(inV, header=None, delim_whitespace=True)
                        dr = pd.read_csv(inR, header=None, delim_whitespace=True)
                        break
                    except FileNotFoundError:
                        print("You have entered in an incorrect file or file pathway. Please try again.\n")
                _ = BSUO(T0, To_err, period, db, dv, dr)
            elif num == 2:
                first_time = input("Do you already have an Epoch value 'Yes' or 'No': ")
                if first_time.lower() == "no":
                    T0 = 0
                    To_err = 0
                    period = float(input("Please enter the period for your system: "))
                else:
                    T0, To_err, period = arguments()
                    print("Example file pathway: C:\\folder1\\folder2\\[file name]")
                while True:
                    infile = input("Please enter your times of minimum file pathway: ")
                    try:
                        df = pd.read_csv(infile, header=None, delim_whitespace=True)
                        break
                    except FileNotFoundError:
                        print("You have entered in an incorrect file or file pathway. Please try again.\n")
                _ = TESS_OC(T0, To_err, period, df)
            elif num == 3:
                while True:
                    try:
                        nights = int(input("How many files will you be using(i.e. if you have BSUO/SARA and TESS data "
                                           "then you have 2 files): "))
                        break
                    except ValueError:
                        print("Please enter a valid whole number.\n")
                total = all_data(nights)
                data_fit(total)
            elif num == 4:
                break
                exit()
            else:
                print("Please enter either 1, 2, 3, or 4.\n")
        except ValueError:
            print("Please enter either 1, 2, 3, or 4.\n")


def TESS_OC(T0, To_err, period, df):
    """
    This function takes ToM data pre-gathered from TESS data and finds corresponding O-C values.

    :return: output file that will be used to plot the O-C data
    """
    # strict Kwee van Woerden method ToM
    min_strict = list(df[0])
    min_strict_err = list(df[2])

    # modified Kwee van Woerdan method ToM for potential use later
    # min_mod = list(df[3])
    # min_mod_err = list(df[4])

    # create the lists that will be used
    E_est = []
    O_C = []
    O_C_err = []

    # this for loop, loops through the min_strict list and calculates a variety of values
    for count, val in enumerate(min_strict):
        # call the function to calculate the O-C values
        e, OC, OC_err, T0, To_err = calculate_oc(val, min_strict_err[count], T0, To_err, period)

        E_est.append(e)
        O_C.append(OC)
        O_C_err.append(OC_err)

    # create a dataframe for all outputs to be places in for easy output
    dp = pd.DataFrame({
        "Minimums": min_strict,
        "Epoch": E_est,
        "O-C": O_C,
        "O-C_Error": O_C_err
    })

    # output file name to place the above dataframe into for saving
    outfile = input("Please enter the output file pathway and file name with extension for the ToM "
                    "(i.e. C:\\folder1\\test.txt): ")
    dp.to_csv(outfile, index=None, sep="\t")
    print("\nFinished saving file to " + outfile + ". This file is in the same folder as this python program.")

    return outfile


def BSUO(T0, To_err, period, db, dv, dr):
    """
    This function uses BSUO filter ToM's to calculate and averaged ToM from the 3 filters used. Then calculates
    O-C values to be plotted later.

    :return: output file that will be used to plot the O-C data
    """
    # strict Kwee van Woerden method ToM for all 3 filters
    strict_B = list(db[0])
    strict_B_err = list(db[2])
    strict_V = list(dv[0])
    strict_V_err = list(dv[2])
    strict_R = list(dr[0])
    strict_R_err = list(dr[2])

    # create the lists that will be used
    E_est = []
    O_C = []
    O_C_err = []
    average_min = []
    average_err = []

    # calculates the minimum by averaging the three filters together and getting the total error for that averaged ToM
    for count, val in enumerate(strict_B):
        # calculate ToM and its error
        minimum = (val + strict_V[count] + strict_R[count]) / 3
        err = sqrt(strict_B_err[count] ** 2 + strict_V_err[count] ** 2 + strict_R_err[count] ** 2) / 3

        average_min.append("%.5f" % minimum)
        average_err.append(err)

        # call the function to calculate the O-C values
        e, OC, OC_err, T0, To_err = calculate_oc(minimum, err, T0, To_err, period)
        E_est.append(e)
        O_C.append(OC)
        O_C_err.append(OC_err)

    # create a dataframe for all outputs to be places in for easy output
    dp = pd.DataFrame({
        "Minimums": average_min,
        "Epoch": E_est,
        "O-C": O_C,
        "O-C_Error": O_C_err
    })

    # output file name to place the above dataframe into for saving
    outfile = input("Please enter the output fil pathway and file name with extension for the ToM "
                    "(i.e. C:\\folder1\\test.txt): ")
    # noinspection PyTypeChecker
    dp.to_csv(outfile, index=None, sep="\t")
    print("\nFinished saving file to " + outfile + ". This file is in the same folder as this python program.")

    return outfile


def all_data(nights):
    count = 1

    minimum_list = []
    e_list = []
    o_c_list = []
    o_c_err_list = []

    while True:
        print("\n\nPlease make sure that the very first line for each and every file that you have starts with the following\n"
              "'Minimums	Epoch	O-C	O-C_Error'\n"
              "With each space entered as a space.\n")
        fname = input("Please enter a file name and pathway (i.e. C:\\folder1\\folder2\\[file name]): ")
        df = pd.read_csv(fname, header=None, skiprows=[0], delim_whitespace=True)
        minimum = np.array(df[0])
        e = np.array(df[1])
        o_c = np.array(df[2])
        o_c_err = np.array(df[3])

        for num, val in enumerate(minimum):
            minimum_list.append("%.5f" % val)
            e_list.append(e[num])
            o_c_list.append("%.5f" % o_c[num])
            o_c_err_list.append("%.5f" % o_c_err[num])
        if count == nights:
            break
        else:
            count += 1
            continue
    dp = pd.DataFrame({
        "Minimums": minimum_list,
        "Epoch": e_list,
        "O-C": o_c_list,
        "O-C_Error": o_c_err_list
    })

    outfile = input("Please enter the output file pathway and file name WITHOUT extension "
                    "for the ToM (i.e. C:\\folder\[file_name]): ")
    dp.to_csv(outfile + ".txt", index=None, sep="\t")
    print("\nFinished saving file to " + outfile + ".txt\n")

    """
    LaTeX table stuff, don't change unless you know what you're doing!
    """
    table_header = "\\renewcommand{\\baselinestretch}{1.00} \small\\normalsize"
    table_header += '\\begin{center}\n' + '\\begin{longtable}{ccc}\n'
    table_header += '$BJD_{\\rm TDB}$ & ' + 'E & ' + 'O-C \\\ \n'
    table_header += '\\hline\n' + '\\endfirsthead\n'
    table_header += '\\multicolumn{3}{c}\n'
    table_header += '{\\tablename\ \\thetable\ -- \\textit{Continued from previous page}} \\\ \n'
    table_header += '$BJD_{\\rm TDB}$ & E & O-C \\\ \n'
    table_header += '\\hline\n' + '\\endhead\n' + '\\hline\n'
    table_header += '\\multicolumn{3}{c}{\\textit{Continued on next page}} \\\ \n'
    table_header += '\\endfoot\n' + '\\endlastfoot\n'

    minimum_lines = []
    for i in range(len(minimum)):
        line = str("%.5f" % minimum[i]) + ' & ' + str(e[i]) + ' & $' + str("%.5f" % o_c[i]) + ' \pm ' + str( "%.5f" %o_c_err[i]) + '$ ' + "\\\ \n"
        minimum_lines.append(line)

    output = table_header
    for count, line in enumerate(minimum_lines):
        output += line

    output += '\\hline\n' + '\\caption{NSVS 896797 O-C. The first column is the \n' \
                            '$BJD_{TDB}$ and column 2 is the epoch number with a whole number \n' \
                            'being a primary eclipse and a half integer value being a secondary \n' \
                            'eclipse. Column 3 is the $(O-C)$ value with the corresponding \n' \
                            '1$\\sigma$ error.}\n' \
              + '\\label{tbl:896797_OC}\n' + '\\end{longtable}\n' + '\\end{center}\n'
    output += '\\renewcommand{\\baselinestretch}{1.66} \\small\\normalsize'
    """
    End LaTeX table stuff.
    """

    # outputfile = input("Please enter an output file name without the extension: ")
    file = open(outfile + ".tex", "w")
    file.write(output)
    file.close()

    outfile += ".txt"

    return outfile


def arguments():
    """
    This function asks the user for the T0

    :return: T0, To_err, and period float values
    """
    while True:
        try:
            T0 = float(input("Please enter your Epoch number (ex. '2457143.761819') : "))  # First primary ToM
            To_err = float(input("Please enter the Epoch error (ex. 0.0002803). : "))  # error associated with the T0
            period = float(input("Please enter the period of your system (ex. 0.31297): "))  # period of a system
            break
        except ValueError:
            print("You have entered an invalid value. Please only enter float values and please try again.\n")
    return T0, To_err, period


@jit(forceobj=True)
def calculate_oc(m, err, T0, T0_err, p):
    """
    Calculates O-C values and errors and find the eclipse number for primary and secondary eclipses
    :param m: ToM
    :param err: ToM error
    :param T0: first ToM
    :param T0_err: error for the T0
    :param p: period of the system

    :return: e (eclipse number), OC (O-C value), OC_err (corresponding O-C error)
    """
    if T0 == 0:
        T0 = m
        T0_err = err
    # get the exact E value
    E_act = (m - T0) / p
    # estimate for the primary or secondary eclipse by rounding to the nearest 0.5
    e = round(E_act * 2) / 2
    # caluclate the calculated ToM and find the O-C value
    T_calc = T0 + (e * p)
    OC = "%.5f" % (m - T_calc)

    # determine the error of the O-C
    OC_err = "%.5f" % sqrt(T0_err ** 2 + err ** 2)

    return e, OC, OC_err, T0, T0_err


def data_fit(input_file):
    """
    Create a linear fit by hand and then use scipy to create a polynomial fit given an equation along with their
    respective residual plots
    :param input_file: input file from either TESS or BSUO

    :return: None
    """
    # read in the text file
    df = pd.read_csv(input_file, header=0, delim_whitespace=True)

    # append values to their respective lists for further and future potential use
    x = df["Epoch"]
    y = df["O-C"]
    y_err = df["O-C_Error"]

    # these next parts are mainly for O-C data as I just want to plot primary minima's and not both primary/secondary
    x1_prim = []
    y1_prim = []
    y_err_new_prim = []

    x1_sec = []
    y1_sec = []
    y_err_new_sec = []

    # collects primary and secondary times of minima separately and its corresponding O-C and O-C error
    for count, val in enumerate(x):
        # noinspection PyUnresolvedReferences
        if val % 1 == 0:  # checks to see if the eclipse number is primary or secondary
            x1_prim.append(float(val))
            y1_prim.append(float(y[count]))
            y_err_new_prim.append(float(y_err[count]))
        else:
            x1_sec.append(float(val))
            y1_sec.append(float(y[count]))
            y_err_new_sec.append(float(y_err[count]))

    # converts the lists to numpy arrays
    x1_prim = np.array(x1_prim)
    y1_prim = np.array(y1_prim)
    y_err_new_prim = np.array(y_err_new_prim)

    x1_sec = np.array(x1_sec)
    y1_sec = np.array(y1_sec)
    y_err_new_sec = np.array(y_err_new_sec)

    # different line styles that can be used
    line_style = [(0, (1, 10)), (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)),
                  (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)),
                  (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
    line_count = 0
    i_string = ""

    # beginning latex to a latex table
    beginningtex = """\\documentclass{report}
            \\usepackage{booktabs}
            \\begin{document}"""
    endtex = "\end{document}"

    # opens a file with this name to begin writing to the file
    output_test = None
    while not output_test:
        output_file = input("What is the output file name and pathway for the regression tables (either .txt or .tex): ")
        if output_file.endswith((".txt", ".tex")):
            output_test = True
        else:
            print("This is not an allowed file output. Please make sure the file has the extension .txt or .tex.\n")

    # noinspection PyUnboundLocalVariable
    f = open(output_file, 'w')
    f.write(beginningtex)

    # sets up the fitting parameters
    xs = np.linspace(x.min(), x.max(), 1000)
    degree = 2
    degree_list = ["Linear", "Quadratic"]
    # noinspection PyUnboundLocalVariable
    for i in range(1, degree+1):
        """
        Inside the model variable:
        'np.polynomial.polynomial.polyfit(x, y, i)' gathers the coefficients of the line fit

        'Polynomial' then finds an array of y values given a set of x data
        """
        model = Polynomial(np.polynomial.polynomial.polyfit(x1_prim, y1_prim, i))

        # plot the main graph with both fits (linear and poly) onto the same graph
        plt.plot(xs, model(xs), color="black", label=degree_list[i-1] + " fit",
                 linestyle=line_style[line_count])
        line_count += 1

        # this if statement adds a string together to be used in the regression analysis
        # pretty much however many degrees in the polynomial there are, there will be that many I values
        if i >= 2:
            i_string = i_string + " + I(x**" + str(i) + ")"
            mod = smf.ols(formula='y ~ x' + i_string, data=df)
            res = mod.fit()
            f.write(res.summary().as_latex())
        elif i == 1:
            mod = smf.ols(formula='y ~ x', data=df)
            res = mod.fit()
            f.write(res.summary().as_latex())

    f.write(endtex)
    # writes to the file the end latex code and then saves the file
    f.close()
    print("\nFinished saving latex/text file.\n\n")

    fontsize = 14
    plt.errorbar(x1_prim, y1_prim, yerr=y_err_new_prim, fmt="o", color="blue", label="Primary")
    plt.errorbar(x1_sec, y1_sec, yerr=y_err_new_sec, fmt="s", color="green", label="Secondary")
    # allows the legend to be moved wherever the user wants the legend to be placed rather than in a fixed location
    print("\n\nNOTE:")
    print("You can drag the legend to move it wherever you would like, the default is the top right. Just click and drag"
          " to move around the figure.\n")
    plt.legend(loc="upper right", fontsize=fontsize).set_draggable(True)

    x_label = "Epoch"
    y_label = "O-C (days)"

    # noinspection PyUnboundLocalVariable
    plt.xlabel(x_label, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # noinspection PyUnboundLocalVariable
    plt.ylabel(y_label, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid()
    plt.show()

    # residuals(x1_prim, y1_prim, x_label, y_label, degree, model, xs)


def residuals(x, y, x_label, y_label, degree, model, xs):
    """
    This plots the residuals of the data from the input file
    :param x: original x data
    :param y: original y data
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param degree: degree of the polynomial fit
    :param model: the last model (equation) that was used from above
    :param xs: numpy x data set

    :return: none
    """
    # appends the y values from the model to a variable
    y_model = model(xs)

    # makes dataframes for both the raw data and the model data
    raw_dat = pd.DataFrame({
        x_label: x,
        y_label: y,
    })

    model_dat = pd.DataFrame({
        x_label: xs,
        y_label: y_model
    })

    # allows for easy change of the format of the subplots
    rows = 2
    cols = 1
    # creates the figure subplot for appending next
    fig, (ax1, ax2) = plt.subplots(rows, cols)
    # adds gridlines to both subplots
    """
    a[0].grid(visible=True, which='major', color='black', linewidth=1.0)
    a[0].grid(visible=True, which='minor', color='black', linewidth=0.5)
    a[1].grid(visible=True, which='major', color='black', linewidth=1.0)
    a[1].grid(visible=True, which='minor', color='black', linewidth=0.5)
    """
    ax1.grid()
    ax2.grid()
    # creates the model line fit
    sns.lineplot(x=x_label, y=y_label, data=model_dat, ax=ax1, color="red")
    # plots the original data to the same subplot as the model fit
    # edge color is removed to any sort of weird visual overlay on the plots as the normal edge color is white
    sns.scatterplot(x=x_label, y=y_label, data=raw_dat, ax=ax1, color="black", edgecolor="none")
    # plots the residuals from the original data to the polynomial degree from  above
    sns.residplot(x=x_label, y=y_label, order=degree, data=raw_dat, ax=ax2, color="black",
                  scatter_kws=dict(edgecolor="none"))
    # adds a horizontal line to the residual plot to represent the model line fit with the same color
    ax2.axhline(y=0, color="red")

    plt.show()


# data_fit('254037_OC.txt')
# data_fit('896797_OC.txt')

if __name__ == '__main__':
    main()
