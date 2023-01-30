"""
Author: Kyle Koeller
Created: 12/19/2022
Last Edited: 01/30/2023

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
    print("The format of these input files should be the of the raw form given from Dr. Robert Berrginton's"
          " 'find_minimum' C program.")
    print("Enter the corresponding number to what you would like to do.\n")
    while True:
        try:
            num = int(input("Would you like to use TESS data(1), BSUO data(2), All Data(3), or Close Program(4): "))
            T0, To_err, period = arguments()
            if num == 1:
                while True:
                    infile = input("Please enter your combined times of minimum file pathway: ")
                    try:
                        df = pd.read_csv(infile, header=None, delim_whitespace=True)
                        break
                    except FileNotFoundError:
                        print("You have entered in an incorrect file or file pathway. Please try again.\n")
                tess = TESS_OC(T0, To_err, period, df)
                data_fit(tess)
            elif num == 2:
                while True:
                    inB = input("Please enter your times of minimum file pathway for the Johnson B filter: ")
                    inV = input("Please enter your times of minimum file pathway for the Johnson V filter: ")
                    inR = input("Please enter your times of minimum file pathway for the Johnson R filter: ")
                    try:
                        db = pd.read_csv(inB, header=None, delim_whitespace=True)
                        dv = pd.read_csv(inV, header=None, delim_whitespace=True)
                        dr = pd.read_csv(inR, header=None, delim_whitespace=True)
                        break
                    except FileNotFoundError:
                        print("You have entered in an incorrect file or file pathway. Please try again.\n")
                bsuo = BSUO(T0, To_err, period, db, dv, dr)
                data_fit(bsuo)
            elif num == 3:
                while True:
                    try:
                        nights = int(input("How many nights of data do you have: "))
                        break
                    except ValueError:
                        print("Please enter a valid number.\n")
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
        e, OC, OC_err = calcualte_oc(val, min_strict_err[count], T0, To_err, period)

        E_est.append(e)
        O_C.append(OC)
        O_C_err.append(OC_err)

    # create a dataframe for all outputs to be places in for easy output
    dp = pd.DataFrame({
        "Minimums": min_strict,
        "Eclipse #": E_est,
        "O-C": O_C,
        "O-C Error": O_C_err
    })

    # output file name to place the above dataframe into for saving
    outfile = input("Please enter the output fil pathway and file name with extension for the ToM (i.e. C:\\test.txt): ")
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

        average_min.append(minimum)
        average_err.append(err)

        # call the function to calculate the O-C values
        e, OC, OC_err = calcualte_oc(minimum, err, T0, To_err, period)
        E_est.append(e)
        O_C.append(OC)
        O_C_err.append(OC_err)

    # create a dataframe for all outputs to be places in for easy output
    dp = pd.DataFrame({
        "Minimums": average_min,
        "Eclipse #": E_est,
        "O-C": O_C,
        "O-C Error": O_C_err
    })

    # output file name to place the above dataframe into for saving
    outfile = input("Please enter the output fil pathway and file name with extension for the ToM (i.e. C:\\test.txt): ")
    dp.to_csv(outfile, index=None, sep="\t")
    print("\nFinished saving file to " + outfile + ". This file is in the same folder as this python program.")

    return outfile


def all_data(nights):
    minimum = []
    e = []
    o_c = []
    o_c_err = []
    count = 1
    while True:
        try:
            fname = input("Please enter a file name (if the file is in the same folder as this program) or the full "
                          "file pathway for all your data: ")
            df = pd.read_csv(fname, header=None, delim_whitespace=True)
            break
        except FileNotFoundError:
            print("You have entered in an incorrect file or file pathway. Please try again.\n")

        minimum.append(df[0])
        e.append(df[1])
        o_c.append(df[2])
        o_c_err.append(df[3])

        if count == nights:
            break
        else:
            count += 1
            continue

    dp = pd.DataFrame({
        "Minimums": minimum,
        "Eclipse N#": e,
        "O-C": o_c,
        "O-C Error": o_c_err
    })

    outfile_path = input("Please enter the JUST the output file pathway (i.e. C:\\folder\[file_name]): ")
    outfile_name = input("Please enter a file name WITHOUT any extension (i.e. 'test' without .txt or anything else): ")
    outfile = outfile_path + outfile_name
    dp.to_csv(outfile + ".txt", index=None, sep="\t")
    print("\nFinished saving file to " + outfile)

    """
    LaTeX table stuff, don't change unless you know what you're doing!
    """
    table_header = "\\renewcommand{\\baselinestretch}{1.00} \small\\normalsize"
    table_header += '\\begin{center}\n' + '\\begin{longtable}{ccc}\n'
    table_header += '$BJD_{TDB}$ & ' + 'E & ' + 'O-C \\\ \n'
    table_header += '\\hline\n' + '\\endfirsthead\n'
    table_header += '\\multicolumn{3}{c}\n'
    table_header += '{\\tablename\ \\thetable\ -- \\textit{Continued from previous page}} \\\ \n'
    table_header += '$BJD_{TDB}$ & E & O-C \\\ \n'
    table_header += '\\hline\n' + '\\endhead\n' + '\\hline\n'
    table_header += '\\multicolumn{3}{c}{\\textit{Continued on next page}} \\\ \n'
    table_header += '\\endfoot\n' + '\\endlastfoot\n'

    minimum_lines = []
    for i in range(len(minimum)):
        line = str(minimum[i]) + ' & ' + str(e[i]) + ' & $' + str(o_c[i]) + ' \pm ' + str(o_c_err[i]) + '$ ' + "\\\ \n"
        minimum_lines.append(line)

    output = table_header
    for count, line in enumerate(minimum_lines):
        output += line

    output += '\\hline\n' + '\caption{NSVS 896797 O-C. The first column is the \n' \
                            '$BJD_{TDB}$ and column 2 is the eclipse number with a whole number \n' \
                            'being a primary eclipse and a half integer value being a secondary \n' \
                            'eclipse. Column 3 is the $(O-C)$ value with the corresponding \n' \
                            'error.}' \
              + '\\label{896797_OC}\n' + '\\end{longtable}\n' + '\\end{center}\n'
    output += '\\renewcommand{\\baselinestretch}{1.66} \small\\normalsize'
    """
    End LaTeX table stuff.
    """
    
    # outputfile = input("Please enter an output file name without the extension: ")
    file = open(outfile + ".tex", "w")
    file.write(output)
    file.close()

    return outfile


def arguments():
    """
    This function asks the user for the T0

    :return: T0, To_err, and period float values
    """
    while True:
        try:
            T0 = float(input("Please enter your T0 (ex. '2457143.761819') : "))  # First ToM
            To_err = float(input("Please enter the T0 error (ex. 0.0002803). : "))  # error associated with the T0
            period = float(input("Please enter the period of your system (ex. 0.31297): "))  # period of a system
            break
        except ValueError:
            print("You have entered an invalid value. Please only enter float values and please try again.\n")
    return T0, To_err, period


@jit(forceobj=True)
def calcualte_oc(m, err, T0, T0_err, p):
    """
    Calculates O-C values and errors and find the eclipse number for primary and secondary eclipses
    :param m: ToM
    :param err: ToM error
    :param T0: first ToM
    :param T0_err: error for the T0
    :param p: period of the system

    :return: e (eclipse number), OC (O-C value), OC_err (corresponding O-C error)
    """
    # get the exact E value
    E_act = (m - T0) / p
    # estimate for the primary or secondary eclipse by rounding to the nearest 0.5
    e = round(E_act * 2) / 2
    # caluclate the calculated ToM and find the O-C value
    T_calc = T0 + (e * p)
    OC = "%.6f" % (m - T_calc)

    # determine the error of the O-C
    OC_err = "%.6f" % sqrt(T0_err ** 2 + err ** 2)

    return e, OC, OC_err


def data_fit(input_file):
    """
    Create a linear fit by hand and then use scipy to create a polynomial fit given an equation along with their
    respective residual plots
    :param input_file: input file from either TESS or BSUO

    :return: None
    """
    # read in the text file
    df = pd.read_csv(input_file, header=0, delimiter="\t")

    # append values to their respective lists for further and future potential use
    x = df["Eclipse #"]
    y = df["O-C"]
    y_err = df["O-C Error"]

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

    # numpy curve fit
    degree_test = None
    while not degree_test:
        # make sure the value entered is actually an integer
        try:
            degree = int(input("How many polynomial degrees do you want to fit (integer values > 0) i.e. 1=linear and 2=qudratic: "))
            degree_test = True
        except ValueError:
            print("This is not an integer, please enter an integer.")
            print()
            degree_test = False

    print("")
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
        output_file = input("What is the output file name for the regression tables (either .txt or .tex): ")
        if output_file.endswith((".txt", ".tex")):
            output_test = True
        else:
            print("This is not an allowed file output. Please make sure the file has the extension .txt or .tex.\n")

    # noinspection PyUnboundLocalVariable
    f = open(output_file, 'w')
    f.write(beginningtex)

    xs = np.linspace(x.min(), x.max(), 1000)
    # noinspection PyUnboundLocalVariable
    for i in range(1, degree + 1):
        """
        Inside the model variable:
        'np.polynomial.polynomial.polyfit(x, y, i)' gathers the coefficients of the line fit

        'Polynomial' then finds an array of y values given a set of x data
        """
        model = Polynomial(np.polynomial.polynomial.polyfit(x1_prim, y1_prim, i))

        # plot the main graph with both fits (linear and poly) onto the same graph
        plt.plot(xs, model(xs), color="black", label="polynomial fit of degree " + str(i),
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
    print("Finished saving latex/text file.")

    plt.errorbar(x1_prim, y1_prim, yerr=y_err_new_prim, fmt="o", color="blue", label="Primary")
    plt.errorbar(x1_sec, y1_sec, yerr=y_err_new_sec, fmt="s", color="green", label="Secondary")
    # make the legend always be in the upper right hand corner of the graph
    plt.legend(loc="upper right")

    empty = None
    while not empty:
        x_label = input("X-Label: ")
        y_label = input("Y-Label: ")
        title = input("Title: ")
        if not x_label:
            print("x label is empty. Please enter a string or value for these variables.\n")
        elif not y_label:
            print("y label is empty. Please enter a string or value for these variables.\n")
        else:
            empty = True

    # noinspection PyUnboundLocalVariable
    plt.xlabel(x_label)
    # noinspection PyUnboundLocalVariable
    plt.ylabel(y_label)
    # noinspection PyUnboundLocalVariable
    plt.title(title)
    plt.grid()
    plt.show()

    residuals(x1_prim, y1_prim, x_label, y_label, degree, model, xs)


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


if __name__ == '__main__':
    main()
