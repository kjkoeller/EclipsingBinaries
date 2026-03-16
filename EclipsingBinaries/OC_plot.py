"""
Author: Kyle Koeller
Created: 12/19/2022
Last Edited: 03/16/2026

This calculates O-C values and produces an O-C plot.
"""

from math import sqrt, floor, ceil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import statsmodels.formula.api as smf
import seaborn as sns
from numba import jit
from pathlib import Path


def TESS_OC(T0, T0_err, period, df, output_path, write_callback=None, cancel_event=None):
    """
    Takes ToM data pre-gathered from TESS data and finds corresponding O-C values.

    Parameters
    ----------
    T0 : float
        Initial epoch. If 0, the first ToM is used as T0.
    T0_err : float
        Error on T0.
    period : float
        Orbital period of the system.
    df : pd.DataFrame
        DataFrame with ToM data (column 0 = ToM, column 2 = error).
    output_path : str
        Base output folder path.
    write_callback : callable, optional
        Function to log messages.
    cancel_event : threading.Event, optional
        Event to check for cancellation.

    Returns
    -------
    str or None
        Path to the saved output file, or None if canceled.
    """
    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    min_strict = list(df[0])
    min_strict_err = list(df[2])

    E_est = []
    O_C = []
    O_C_err = []

    for count, val in enumerate(min_strict):
        if cancel_event and cancel_event.is_set():
            log("Task canceled during TESS O-C calculation.")
            return None

        e, OC, OC_err, T0, T0_err = calculate_oc(val, min_strict_err[count], T0, T0_err, period)
        E_est.append(e)
        O_C.append(OC)
        O_C_err.append(OC_err)

    dp = pd.DataFrame({
        "Minimums": min_strict,
        "Epoch": E_est,
        "O-C": O_C,
        "O-C_Error": O_C_err
    })

    outfile = str(Path(output_path) / "TESS_OC.txt")
    dp.to_csv(outfile, index=None, sep="\t")
    log(f"Finished saving TESS O-C file to {outfile}")
    return outfile


def BSUO(T0, T0_err, period, db, dv, dr, output_path, write_callback=None, cancel_event=None):
    """
    Uses BSUO/SARA filter ToMs to calculate an averaged ToM and O-C values.

    Parameters
    ----------
    T0 : float
        Initial epoch. If 0, the first averaged ToM is used as T0.
    T0_err : float
        Error on T0.
    period : float
        Orbital period of the system.
    db : pd.DataFrame
        B filter ToM data.
    dv : pd.DataFrame
        V filter ToM data.
    dr : pd.DataFrame
        R filter ToM data.
    output_path : str
        Base output folder path.
    write_callback : callable, optional
        Function to log messages.
    cancel_event : threading.Event, optional
        Event to check for cancellation.

    Returns
    -------
    str or None
        Path to the saved output file, or None if canceled.
    """
    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    strict_B = list(db[0])
    strict_B_err = list(db[2])
    strict_V = list(dv[0])
    strict_V_err = list(dv[2])
    strict_R = list(dr[0])
    strict_R_err = list(dr[2])

    E_est = []
    O_C = []
    O_C_err = []
    average_min = []
    average_err = []

    for count, val in enumerate(strict_B):
        if cancel_event and cancel_event.is_set():
            log("Task canceled during BSUO O-C calculation.")
            return None

        minimum = (val + strict_V[count] + strict_R[count]) / 3
        err = sqrt(strict_B_err[count] ** 2 + strict_V_err[count] ** 2 + strict_R_err[count] ** 2) / 3

        average_min.append("%.5f" % minimum)
        average_err.append(err)

        e, OC, OC_err, T0, T0_err = calculate_oc(minimum, err, T0, T0_err, period)
        E_est.append(e)
        O_C.append(OC)
        O_C_err.append(OC_err)

    dp = pd.DataFrame({
        "Minimums": average_min,
        "Epoch": E_est,
        "O-C": O_C,
        "O-C_Error": O_C_err
    })

    outfile = str(Path(output_path) / "BSUO_OC.txt")
    dp.to_csv(outfile, index=None, sep="\t")
    log(f"Finished saving BSUO O-C file to {outfile}")
    return outfile


def all_data(file_paths, period, output_path, write_callback=None, cancel_event=None):
    """
    Merges multiple O-C data files into a single combined output file and
    generates a LaTeX table.

    Parameters
    ----------
    file_paths : list of str
        List of paths to O-C data files to merge.
    period : float
        Orbital period of the system.
    output_path : str
        Base output folder path.
    write_callback : callable, optional
        Function to log messages.
    cancel_event : threading.Event, optional
        Event to check for cancellation.

    Returns
    -------
    str or None
        Path to the saved merged output file, or None if canceled.
    """
    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    minimum_list = []
    e_list = []
    o_c_list = []
    o_c_err_list = []

    for fname in file_paths:
        if cancel_event and cancel_event.is_set():
            log("Task canceled during All Data merge.")
            return None

        df = pd.read_csv(fname, header=None, skiprows=[0], sep=r"\s+")
        minimum = np.array(df[0])
        e = np.array(df[1])
        o_c = np.array(df[2])
        o_c_err = np.array(df[3])

        for num, val in enumerate(minimum):
            minimum_list.append("%.5f" % val)
            e_list.append(e[num])
            o_c_list.append("%.5f" % o_c[num])
            o_c_err_list.append("%.5f" % o_c_err[num])

    dp = pd.DataFrame({
        "Minimums": minimum_list,
        "Epoch": e_list,
        "O-C": o_c_list,
        "O-C_Error": o_c_err_list
    })

    outfile = str(Path(output_path) / "all_data_OC.txt")
    dp.to_csv(outfile, index=None, sep="\t")
    log(f"Finished saving merged O-C file to {outfile}")

    # LaTeX table
    table_header = r"\renewcommand{\baselinestretch}{1.00} \small\normalsize"
    table_header += "\n" + r"\begin{center}" + "\n" + r"\begin{longtable}{ccc}" + "\n"
    table_header += r"$BJD_{\rm TDB}$ & E & O-C \\" + "\n"
    table_header += r"\hline" + "\n" + r"\endfirsthead" + "\n"
    table_header += r"\multicolumn{3}{c}" + "\n"
    table_header += r"{\tablename~\thetable~-- \textit{Continued from previous page}} \\" + "\n"
    table_header += r"$BJD_{\rm TDB}$ & E & O-C \\" + "\n"
    table_header += r"\hline" + "\n" + r"\endhead" + "\n" + r"\hline" + "\n"
    table_header += r"\multicolumn{3}{c}{\textit{Continued on next page}} \\" + "\n"
    table_header += r"\endfoot" + "\n" + r"\endlastfoot" + "\n"

    output = table_header
    for i in range(len(minimum_list)):
        line = (str(minimum_list[i]) + " & " + str(e_list[i]) + " & $" +
                str(o_c_list[i]) + r" \pm " + str(o_c_err_list[i]) + r"$ \\" + "\n")
        output += line

    output += (r"\hline" + "\n" +
               r"\caption{O-C table. Column 1 is $BJD_{TDB}$, column 2 is epoch number, "
               r"column 3 is $(O-C)$ with 1$\sigma$ error.}" + "\n" +
               r"\label{tbl:OC}" + "\n" +
               r"\end{longtable}" + "\n" +
               r"\end{center}" + "\n")
    output += r"\renewcommand{\baselinestretch}{1.66} \small\normalsize"

    tex_file = str(Path(output_path) / "all_data_OC.tex")
    with open(tex_file, "w") as f:
        f.write(output)
    log(f"LaTeX table saved to {tex_file}")

    return outfile


@jit(forceobj=True)
def calculate_oc(m, err, T0, T0_err, p):
    """
    Calculates O-C values and errors and finds the eclipse number.

    Parameters
    ----------
    m : float
        Time of minimum.
    err : float
        Error on the time of minimum.
    T0 : float
        Reference epoch. If 0, set to m on first call.
    T0_err : float
        Error on T0.
    p : float
        Period of the system.

    Returns
    -------
    tuple
        (e, OC, OC_err, T0, T0_err)
    """
    if T0 == 0:
        T0 = m
        T0_err = err

    E_act = (m - T0) / p

    if E_act <= 0:
        e = ceil((E_act * 2) + 0.5) / 2
    else:
        e = floor((E_act * 2) + 0.5) / 2

    T_calc = T0 + (e * p)
    OC = "%.5f" % (m - T_calc)
    OC_err = "%.5f" % sqrt(T0_err ** 2 + err ** 2)

    return e, OC, OC_err, T0, T0_err


def data_fit(input_file, period, write_callback=None, cancel_event=None):
    """
    Creates linear and quadratic fits to O-C data, saves the plot and
    regression tables to the same folder as the input file.

    Parameters
    ----------
    input_file : str
        Path to the O-C data file (tab-separated with header row).
    period : float
        Period of the system.
    write_callback : callable, optional
        Function to log messages.
    cancel_event : threading.Event, optional
        Event to check for cancellation.

    Returns
    -------
    float or None
        The adjusted period from the linear fit, or None if canceled.
    """
    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    df = pd.read_csv(input_file, header=0, sep=r"\s+")

    x = df["Epoch"]
    y = df["O-C"]
    y_err = df["O-C_Error"]
    weights = 1 / (y_err ** 2)

    x1_prim, y1_prim, y_err_prim = [], [], []
    x1_sec, y1_sec, y_err_sec = [], [], []

    for count, val in enumerate(x):
        if val % 1 == 0:
            x1_prim.append(float(val))
            y1_prim.append(float(y[count]))
            y_err_prim.append(float(y_err[count]))
        else:
            x1_sec.append(float(val))
            y1_sec.append(float(y[count]))
            y_err_sec.append(float(y_err[count]))

    x1_prim = np.array(x1_prim)
    y1_prim = np.array(y1_prim)
    y_err_prim = np.array(y_err_prim)
    x1_sec = np.array(x1_sec)
    y1_sec = np.array(y1_sec)
    y_err_sec = np.array(y_err_sec)

    xs = np.linspace(x.min(), x.max(), 1000)
    line_styles = [(0, (5, 5)), (0, (1, 1))]
    degree_list = ["Linear", "Quadratic"]

    # Output paths derived from input file location
    base = Path(input_file)
    tex_path = str(base.with_suffix(".tex"))
    plot_path = str(base.with_suffix(".png"))

    beginningtex = "\\documentclass{report}\n\\usepackage{booktabs}\n\\begin{document}\n"
    endtex = "\\end{document}"

    fig, ax = plt.subplots(figsize=(10, 6))
    i_string = ""
    new_period = period

    with open(tex_path, "w") as f:
        f.write(beginningtex)

        for i in range(1, 3):
            if cancel_event and cancel_event.is_set():
                log("Task canceled during data fitting.")
                plt.close(fig)
                return None

            model = Polynomial(np.polynomial.polynomial.polyfit(x1_prim, y1_prim, i))
            ax.plot(xs, model(xs), color="black", label=degree_list[i - 1] + " fit",
                    linestyle=line_styles[i - 1])

            if i >= 2:
                i_string += f" + I(x**{i})"
                mod = smf.wls(formula="y ~ x" + i_string, data=df, weights=weights)
                res = mod.fit()
                f.write(res.summary().as_latex())
            else:
                mod = smf.wls(formula="y ~ x", data=df, weights=weights)
                res = mod.fit()
                period_add = res.params[1]
                new_period = period + period_add
                log(f"Period correction from linear fit: {period_add:.8f} days")
                log(f"Adjusted period: {new_period:.8f} days")
                f.write(res.summary().as_latex())

        f.write(endtex)

    log(f"Regression tables saved to {tex_path}")

    fontsize = 14
    ax.errorbar(x1_prim, y1_prim, yerr=y_err_prim, fmt="o", color="blue", label="Primary")
    ax.errorbar(x1_sec, y1_sec, yerr=y_err_sec, fmt="s", color="green", label="Secondary")
    ax.legend(loc="upper right", fontsize=fontsize)
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("O-C (days)", fontsize=fontsize)
    ax.grid()

    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"O-C plot saved to {plot_path}")

    return new_period


def residuals(x, y, x_label, y_label, degree, model, xs,
              output_path=None, write_callback=None):
    """
    Plots residuals of the O-C fit and saves to file.

    Parameters
    ----------
    x : array-like
        Epoch values.
    y : array-like
        O-C values.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    degree : int
        Polynomial degree used for residuals.
    model : Polynomial
        Fitted polynomial model.
    xs : array-like
        Dense x values for the model line.
    output_path : str, optional
        Path to save the residuals plot. If None, saves next to input file.
    write_callback : callable, optional
        Function to log messages.
    """
    def log(message):
        if write_callback:
            write_callback(message)
        else:
            print(message)

    y_model = model(xs)

    raw_dat = pd.DataFrame({x_label: x, y_label: y})
    model_dat = pd.DataFrame({x_label: xs, y_label: y_model})

    _, (ax1, ax2) = plt.subplots(2, 1)
    ax1.grid()
    ax2.grid()
    sns.lineplot(x=x_label, y=y_label, data=model_dat, ax=ax1, color="red")
    sns.scatterplot(x=x_label, y=y_label, data=raw_dat, ax=ax1,
                    color="black", edgecolor="none")
    sns.residplot(x=x_label, y=y_label, order=degree, data=raw_dat, ax=ax2,
                  color="black", scatter_kws=dict(edgecolor="none"))
    ax2.axhline(y=0, color="red")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        log(f"Residuals plot saved to {output_path}")
    else:
        plt.savefig("residuals.png", bbox_inches="tight", dpi=150)
        plt.close()
        log("Residuals plot saved to residuals.png")
