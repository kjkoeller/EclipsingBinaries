"""
Author: Kyle Koeller
Created: 12/19/2022
Last Edited: 12/22/2022

This calculates O-C values and produces an O-C plot.
"""

from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import statsmodels.formula.api as smf
import seaborn as sns


def main():
    file = TESS_OC()
    data_fit(file)


def TESS_OC():
    while True:
        infile = input("Please enter your times of minimum file pathway: ")
        try:
            df = pd.read_csv(infile, header=None, delim_whitespace=True)
            break
        except FileNotFoundError:
            print("You have entered in an incorrect file or file pathway. Please try again.\n")
    while True:
        try:
            T0 = float(input("Please enter your T0 (ex. '2457143.761819') : "))  # First ToM
            To_err = float(input("Please enter the T0 error (ex. 0.0002803). : "))  # error associated with the T0
            period = float(input("Please enter the period of your system (ex. 0.31297): "))  # period of a system
            break
        except ValueError:
            print("You have entered an invalid value. Please only enter float values and please try again.\n")

    # strict Kwee van Woerden method ToM
    min_strict = list(df[0])
    min_strict_err = list(df[2])

    # modified Kwee van Woerdan method ToM for potential use later
    min_mod = list(df[3])
    min_mod_err = list(df[4])

    # create the lists that will be used
    E_est = []
    O_C = []
    O_C_err = []

    # this for loop loops through the min_strict list and calculates a variety of values
    for count, val in enumerate(min_strict):
        # get the exact E value
        E_act = (val-T0)/period
        # estimate for the primary or secondary eclipse by rounding to the nearest 0.5
        e = round(E_act*2)/2
        E_est.append(e)

        # caluclate the calculated ToM and find the O-C value
        T_calc = T0+(e*period)
        O_C.append(format(val - T_calc, ".6f"))

        # determine the error of the O-C
        O_C_err.append(format(sqrt(To_err**2 + min_strict_err[count]**2), ".6f"))

    # create a dataframe for all outputs to be places in for easy output
    dp = pd.DataFrame({
        "Minimums": min_strict,
        "Eclipse #": E_est,
        "O-C": O_C,
        "O-C Error": O_C_err
    })

    # output file name to place the above dataframe into for saving
    outfile = input("Please enter the output file name with extension (i.e. test.txt): ")
    dp.to_csv(outfile, index=None, sep="\t")
    print("\nFinished saving file to " + outfile + ". This file is in the same folder as this python program.")

    return outfile


def data_fit(input_file):
    """
    Create a linear fit by hand and then use scipy to create a polynomial fit given an equation along with their
    respective residual plots.
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
            degree = int(input("How many polynomial degrees do you want to fit (integer values > 0): "))
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
            print("This is not an allowed file output. Please make sure the file has the extension .txt or .tex.")
            print()

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
            print("x label is empty. Please enter a string or value for these variables.")
            print()
        elif not y_label:
            print("y label is empty. Please enter a string or value for these variables.")
            print()
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


main()
