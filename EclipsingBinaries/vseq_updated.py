# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:09:28 2020
@author: Alec Neal

Last Updated: 07/20/2024
Last Editor: Kyle Koeller

Collection of functions, coefficients and equations commonly used
with short-period variable stars, but many can be used more
generally
"""
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, AutoLocator, Locator)
import pandas as pd


def isNaN(num):
    """
    Checks if a value is nan

    :param num: value to be checked
    :return: Boolean True or False
    """

    return num != num


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


class io:
    """
    This class provides file input/output functionalities.
    """
    def importFile_pd(inputFile, delimit=None, header=None, file_type='text', engine='python', delim_whitespace=True):
        """
        Imports a file and returns a list of columns.

        Parameters:
        - inputFile: Path to the input file.
        - delimit: Delimiter used in the file. If None, delimiter_whitespace is used for text files.
        - header: Row number(s) to use as the column names.
        - file_type: Type of the input file. Can be 'text' or 'excel'.
        - engine: Parser engine to use. Default is 'python'.
        - delim_whitespace: If True, whitespace is used as the delimiter for text files.

        Returns:
        - columnlist: A list of columns extracted from the file.
        """

        if file_type == 'text':
            if delim_whitespace:
                # Read text file with whitespace delimiter
                file = pd.read_csv(inputFile, delim_whitespace=True, header=header, engine='python')
            else:
                # Read text file with custom delimiter
                file = pd.read_csv(inputFile, sep=delimit, header=header, engine='python')
        elif file_type == 'excel':
            # Read Excel file
            file = pd.read_excel(inputFile, sep=delimit, header=header, engine='python')
        else:
            # Print error message for unsupported file types
            print('File type not currently supported. Choose text or excel type.')

        # Extract columns from the file and append them to columnlist
        columnlist = []
        for column in range(len(list(file))):
            columnlist.append(list(file[column]))

        return columnlist

    def validate_file_path(prompt):
        """
        Prompt the user to enter a file path and validate its existence.

        Parameters
        ----------
        prompt : str
            The message to display when prompting the user for input.

        Returns
        -------
        Path
            A valid Path object representing the entered file path.
        """
        while True:
            path = input(prompt)

            # Exit the program if the user types "Close"
            if path.lower() == "close":
                sys.exit()

            # Check if the entered path is valid
            try:
                images_path = Path(path)
                if images_path.exists():
                    return images_path
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                print("File not found. Please try again.")

    def validate_directory_path(prompt):
        """
        Prompt the user to enter a directory path and validate its existence.

        Parameters
        ----------
        prompt : str
            The message to display when prompting the user for input.

        Returns
        -------
        Path
            A valid Path object representing the entered directory path.
        """
        while True:
            path = input(prompt)

            # Exit the program if the user types "Close"
            if path.lower() == "close":
                sys.exit()

            # Check if the entered path is a valid directory
            try:
                directory_path = Path(path)
                if directory_path.is_dir():
                    return directory_path
                else:
                    raise NotADirectoryError
            except (FileNotFoundError, NotADirectoryError):
                print("Invalid directory. Please try again.")



# =======================================
class calc:  # assortment of functions
    def frac(x):
        """
        Returns the fractional part of a number.

        Parameters:
            x (float): The input number.

        Returns:
            float: The fractional part of the input number.
        """
        return x - np.floor(x)

    def Newton(f, x0, e=1e-8, fprime=None, max_iter=None, dx=1e-8, central_diff=True, print_iters=False):
        """
        Newton-Raphson method for finding roots of a function.

        Parameters:
            f (function): The function whose root is being found.
            x0 (float): Initial guess for the root.
            e (float): Tolerance for convergence.
            fprime (function): The derivative of the function f.
            max_iter (int): Maximum number of iterations allowed.
            dx (float): Step size for numerical differentiation.
            central_diff (bool): Whether to use central differencing for numerical differentiation.
            print_iters (bool): Whether to print the number of iterations.

        Returns:
            float or bool: The estimated root of the function, or False if maximum iterations reached.
        """
        x = x0
        iters = 0
        if fprime is None:
            if not central_diff:
                fprime = lambda x, dx=dx, func=f: (func(x + dx) / dx)

            else:
                fprime = lambda x, dx=dx, func=f: (func(x + dx) - func(x - dx)) / (2 * dx)
        if max_iter is None:
            while abs(f(x)) > e:
                x -= f(x) / fprime(x)
                iters += 1
            if print_iters:
                print('Iterations:', iters)
            return x
        else:
            iters = 0
            while abs(f(x)) > e:
                x -= f(x) / fprime(x)
                iters += 1
                if iters == max_iter:
                    break
            if iters == max_iter:
                return False
            else:
                return x

    class poly:
        def result(coeflist, value, deriv=False):
            """
            Calculates the result of a polynomial given its coefficients and a value.

            Parameters:
                coeflist (list): List of coefficients in ascending order of the polynomial.
                value (float): The value at which the polynomial is evaluated.
                deriv (bool, optional): If True, returns the derivative of the polynomial.

            Returns:
                float: Result of the polynomial at the given value.
            """
            if deriv:
                # Derivative of the polynomial
                return sum(
                    np.array(coeflist[1:]) * np.arange(1, len(coeflist)) * value ** (np.arange(len(coeflist)) - 1))
            else:
                # Evaluate the polynomial
                return sum(np.array(coeflist) * value ** np.arange(len(coeflist)))

        def error(coeflist, value, error):
            """
            Calculates the propagated uncertainty of a polynomial.

            Parameters:
                coeflist (list): List of coefficients in ascending order of the polynomial.
                value (float): The value at which the uncertainty is calculated.
                error (float): The error in the value.

            Returns:
                float: Propagated uncertainty of the polynomial.
            """
            errlist = [n * coef * value ** (n - 1) for n, coef in enumerate(coeflist[1:], start=1)]
            return error * sum(errlist)

        def polylist(coeflist, xmin, xmax, resolution):
            """
            Generates a list of predicted values from a polynomial within specified bounds.

            Parameters:
                coeflist (list): List of coefficients in ascending order of the polynomial.
                xmin (float): Minimum value of the domain.
                xmax (float): Maximum value of the domain.
                resolution (int): Number of points to generate between xmin and xmax.

            Returns:
                tuple: Tuple containing x values and corresponding y values.
            """
            xlist = np.arange(xmin, xmax, (xmax - xmin) / resolution)
            ylist = [calc.poly.result(coeflist, x) for x in xlist]
            return xlist, ylist

        def regr_polyfit(x, y, deg, func=lambda x, n: x ** n, sig_y=None):
            """
            Performs a least squares polynomial fit.

            Parameters:
                x (array_like): Independent variable data.
                y (array_like): Dependent variable data.
                deg (int): Degree of the polynomial.
                func (function, optional): Function to generate features for the polynomial fit.
                sig_y (array_like, optional): Errors associated with dependent variable data.

            Returns:
                tuple: Tuple containing polynomial coefficients, standard errors, coefficient of determination,
                    predicted values, and the original model.
            """
            import statsmodels.api as sm
            x = np.array(x)
            Xlist = [func(x, n) for n in range(1, deg + 1)]
            Xstack = np.column_stack(Xlist)
            Xstack = sm.add_constant(Xstack)
            if sig_y is None:
                ogmodel = sm.OLS(y, Xstack)
            else:
                ogmodel = sm.WLS(y, Xstack, weights=1 / np.array(sig_y) ** 2)
            model = ogmodel.fit()
            return model.params, model.bse, model.rsquared, model.predict(), ogmodel

        def power(coeflist, value, base=10):
            """
            Calculates the result of a power polynomial at a given value.

            Parameters:
                coeflist (list): List of coefficients in ascending order of the power polynomial.
                value (float): The value at which the polynomial is evaluated.
                base (float, optional): Base value of the power polynomial.

            Returns:
                float: Result of the power polynomial at the given value.
            """
            return base ** calc.poly.result(coeflist, value)

        def error_power(coeflist, value, error, base=10):
            """
            Calculates the propagated uncertainty of a power polynomial.

            Parameters:
                coeflist (list): List of coefficients in ascending order of the power polynomial.
                value (float): The value at which the uncertainty is calculated.
                error (float): The error in the value.
                base (float, optional): Base value of the power polynomial.

            Returns:
                float: Propagated uncertainty of the power polynomial.
            """
            return abs(calc.poly.error(coeflist, value, error) * np.log(base) * calc.poly.power(coeflist, value, base))

        def t_eff_err(coeflist, value, error, temp, coeferror=[], base=10):
            """
            Calculates the effective temperature uncertainty.

            Parameters:
                coeflist (list): List of coefficients in ascending order of the polynomial.
                value (float): The value at which the uncertainty is calculated.
                error (float): The error in the value.
                temp (float): The temperature value.
                coeferror (list, optional): List of errors associated with polynomial coefficients.
                base (float, optional): Base value of the power polynomial.

            Returns:
                float: Effective temperature uncertainty.
            """
            if len(coeferror) == 0:
                return temp * np.log(base) *  calc.poly.error(coeflist, value, error)
            else:
                return temp * np.log(base) * np.sqrt( calc.poly.error(coeflist, value, error) ** 2 +
                                                     sum(np.array(coeferror) * (
                                                                 (value ** np.arange(len(coeflist))) ** 2)))

    class error:
        def per_diff(x1, x2):
            """
            Calculates the percentage difference between two values.

            Parameters:
                x1 (float): First value.
                x2 (float): Second value.

            Returns:
                float: Percentage difference between x1 and x2.
            """
            return 100 * (abs(x1 - x2) / np.mean([x1, x2]))

        def SS_residuals(obslist, modellist):
            """
            Calculates the sum of squares of residuals.

            Parameters:
                obslist (array_like): List of observed values.
                modellist (array_like): List of modeled values.

            Returns:
                float: Sum of squares of residuals.
            """
            SS_res = 0
            for n in range(len(obslist)):
                SS_res += (obslist[n] - modellist[n]) ** 2
            return SS_res

        def sig_sum(errorlist):
            """
            Calculates the sum of squared errors.

            Parameters:
                errorlist (array_like): List of errors.

            Returns:
                float: Square root of the sum of squared errors.
            """
            SS = 0
            for n in range(len(errorlist)):
                SS += errorlist[n] ** 2
            return np.sqrt(SS)

        def SS_total(obslist):
            """
            Calculates the total sum of squares.

            Parameters:
                obslist (array_like): List of observed values.

            Returns:
                float: Total sum of squares.
            """
            mean = np.mean(obslist)
            SS_tot = 0
            for n in range(len(obslist)):
                SS_tot += (obslist[n] - mean) ** 2
            return SS_tot

        def CoD(obslist, modellist):
            """
            Calculates the coefficient of determination (R^2).

            Parameters:
                obslist (array_like): List of observed values.
                modellist (array_like): List of modeled values.

            Returns:
                float: Coefficient of determination (R^2).
            """
            return 1 - calc.error.SS_residuals(obslist, modellist) / calc.error.SS_total(obslist)

        def weighted_average(valuelist, errorlist):
            """
            Calculates the weighted average and its uncertainty.

            Parameters:
                valuelist (array_like): List of values.
                errorlist (array_like): List of corresponding errors.

            Returns:
                tuple: Weighted average value, uncertainty in the average, and total weight.
            """
            M = sum(1 / np.array(errorlist) ** 2)
            w_average = 0
            for n in range(len(errorlist)):
                w_average += valuelist[n] / errorlist[n] ** 2
            w_average /= M
            ave_error = 1 / np.sqrt(M)
            return w_average, ave_error, M

        def avg(errorlist):
            """
            Calculates the average of a list of errors.

            Parameters:
                errorlist (array_like): List of errors.

            Returns:
                float: Average error.
            """
            error2list = [error ** 2 for error in errorlist]
            return np.sqrt(sum(error2list)) * (1 / len(errorlist))

        def red_X2(obslist, modellist, obserror):
            """
            Calculates the reduced chi squared.

            Parameters:
                obslist (array_like): List of observed values.
                modellist (array_like): List of modeled values.
                obserror (array_like): List of observed errors.

            Returns:
                float: Reduced chi squared value.
            """
            X2v0 = 0
            X2v = X2v0
            for n in range(len(obslist)):
                X2v += ((obslist[n] - modellist[n]) / obserror[n]) ** 2
            return X2v

        def truncnorm(size, lower=-3.0, upper=3.0, mean=0.0, sigma=1.0):
            """
            Generates truncated Gaussian deviates.

            Parameters:
                size (int): Number of deviates to generate.
                lower (float, optional): Lower bound of the truncated range.
                upper (float, optional): Upper bound of the truncated range.
                mean (float, optional): Mean of the Gaussian distribution.
                sigma (float, optional): Standard deviation of the Gaussian distribution.

            Returns:
                array_like: Array of truncated Gaussian deviates.
            """
            import scipy.stats as sci
            return sci.truncnorm.rvs(lower, upper, loc=mean, scale=sigma, size=size)

    # -------------------
    class astro:
        class convert:
            def HJD_phase(HJDlist, period, Epoch, Pdot=0):
                """
                Converts a list of Heliocentric Julian dates to phase values.

                Parameters:
                    HJDlist (array_like): List of Heliocentric Julian dates.
                    period (float): Period of the cyclic phenomenon.
                    Epoch (float): Reference epoch for phase calculation.
                    Pdot (float, optional): Rate of change of period. Default is 0.

                Returns:
                    array_like: List of corresponding phase values.
                """
                daydiff = np.array(HJDlist) - Epoch
                return (daydiff / (period + Pdot * daydiff)) - np.floor(daydiff / (period + Pdot * daydiff))

            def JD_to_Greg(JD):
                """
                Converts Julian date to Gregorian date format.

                Parameters:
                    JD (float): Julian date.

                Returns:
                    str: Gregorian date in the format 'YYYY MM DD'.
                """
                months = ['none', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                          'October', 'November', 'December']
                f = JD + 1401 + int((int((4 * JD + 274277) / 146097) * 3) / 4) - 38
                e = 4 * f + 3
                g = int((e % 1461) / 4)
                h = 5 * g + 2
                D = int((h % 153) / 5) + 1
                M = (int(h / 153) + 2) % 12 + 1
                Y = int(e / 1461) - 4716 + int((12 + 2 - M) / 12)
                if len(str(D)) < 2:
                    D = '0' + str(D)
                if len(str(M)) < 2:
                    M = '0' + str(M)
                return str(Y) + ' ' + str(M) + ' ' + str(D)

            class magToflux:
                """
                Converts magnitude to flux values.
                """

                def flux(mag):
                    """
                    Converts magnitude to flux.

                    Parameters:
                        mag (float): Magnitude value.

                    Returns:
                        float: Flux value.
                    """
                    return 10 ** (-0.4 * mag)

                def error(mag, magerr):
                    """
                    Calculates flux error from magnitude error.

                    Parameters:
                        mag (float): Magnitude value.
                        magerr (float): Magnitude error value.

                    Returns:
                        float: Flux error value.
                    """
                    return 0.4 * magerr * np.log(10) * 10 ** (-0.4 * mag)

            class fluxTomag:
                """
                Converts flux to magnitude values.
                """

                def mag(flux):
                    """
                    Converts flux to magnitude.

                    Parameters:
                        flux (float): Flux value.

                    Returns:
                        float: Magnitude value.
                    """
                    return -2.5 * np.log10(flux)

                def error(flux, fluxerr):
                    """
                    Calculates magnitude error from flux error.

                    Parameters:
                        flux (float): Flux value.
                        fluxerr (float): Flux error value.

                    Returns:
                        float: Magnitude error value.
                    """
                    return (2.5 * fluxerr) / (flux * np.log(10))


# ======================================
class binning:
    def makebin(phase, bins, phasefluxlist):
        """
        Creates bins and groups phase and flux data into them.

        Parameters:
            phase (float): Phase value to bin around.
            bins (int): Number of bins.
            phasefluxlist (list): List of phase and flux data.

        Returns:
            tuple: Two lists containing phase and flux values grouped into bins.
        """
        halfbin = 0.5 / bins
        binphases = []
        binfluxes = []
        n0 = 0
        n = n0
        while (n < len(phasefluxlist)):
            if phase == 0:
                if 0 < phasefluxlist[n] < halfbin:
                    binphases.append(phasefluxlist[n])
                    binfluxes.append(phasefluxlist[n + 1])
                elif 1 - halfbin < phasefluxlist[n] < 1:
                    binphases.append(phasefluxlist[n])
                    binfluxes.append(phasefluxlist[n + 1])
            else:
                if phase - halfbin < phasefluxlist[n] < phase + halfbin:
                    binphases.append(phasefluxlist[n])
                    binfluxes.append(phasefluxlist[n + 1])
            n += 2
        return binphases, binfluxes

    def binall(bins, phasefluxlist):
        """
        Bins all phases and their corresponding fluxes.

        Parameters:
            bins (int): Number of bins.
            phasefluxlist (list): List of phase and flux data.

        Returns:
            tuple: Lists containing binned phases and averaged binned fluxes.
        """
        binnedphaselist = []
        binnedfluxlist = []
        dphase = 1 / bins
        phase0 = 0
        phase = phase0
        while (phase < 1):
            binnedphaselist.append(phase)
            binnedfluxlist.append(np.mean(binning.makebin(phase, bins, phasefluxlist)[1]))
            phase += dphase
        return binnedphaselist, binnedfluxlist

    def norm_flux(binnedfluxlist, ob_fluxlist, ob_fluxerr, norm_factor='bin'):
        """
        Normalize flux lists based on specified normalization factor.

        Parameters:
            binnedfluxlist (list): List of binned flux values.
            ob_fluxlist (list): List of observed flux values.
            ob_fluxerr (list): List of errors associated with observed flux values.
            norm_factor (str): Normalization factor, either 'bin' or 'ob'. Defaults to 'bin'.

        Returns:
            tuple: Three lists containing normalized binned fluxes, observed fluxes, and their errors.
        """
        if norm_factor == 'ob':
            normf = max(ob_fluxlist)
        else:
            normf = max(binnedfluxlist)
        norm_binned = [binnedflux / normf for binnedflux in binnedfluxlist]
        norm_ob = [ob_flux / normf for ob_flux in ob_fluxlist]
        norm_err = [ob_err / normf for ob_err in ob_fluxerr]
        return norm_binned, norm_ob, norm_err

    def minibinner(phaselist, fluxlist, bins):
        """
        Returns binned lists given a list of phases and fluxes, and specified number of bins.

        Parameters:
            phaselist (list): List of phases.
            fluxlist (list): List of fluxes.
            bins (int): Number of bins.

        Returns:
            tuple: Four lists containing binned phases, binned fluxes, normalized binned fluxes, and normalized observed fluxes.
        """
        obs = len(phaselist)
        phaseflux = [(phaselist[n], fluxlist[n]) for n in range(obs)]
        binned = binning.binall(bins, phaseflux)
        binnedphaselist = binned[0]
        binnedfluxlist = binned[1]
        normed = binning.norm_flux(binnedfluxlist, fluxlist, fluxlist)
        n_binnedfluxlist = normed[0]
        n_ob_fluxlist = normed[1]
        return binnedphaselist, binnedfluxlist, n_binnedfluxlist, n_ob_fluxlist

    def masterbinner(HJD, mag, magerr, Epoch, period, bins=40, weighted=True, norm_factor='alt', centered=True, pdot=0):
        # Convert HJD to phases
        ob_phaselist = calc.astro.convert.HJD_phase(HJD, period, Epoch, Pdot=pdot)

        # Extract observed magnitudes and errors
        ob_maglist = mag
        ob_magerr = magerr
        observations = len(ob_maglist)

        # Convert magnitudes to fluxes
        ob_fluxlist = list(calc.astro.convert.magToflux.flux(np.array(ob_maglist)))
        ob_fluxerr = list(calc.astro.convert.magToflux.error(np.array(ob_maglist), np.array(ob_magerr)))

        # Define bin properties
        halfbin = 0.5 / bins
        dphase = 1 / bins

        # Define function to create bins
        def makebin(phase):
            phases_in_bin = []
            fluxes_in_bin = []
            errors_in_bin = []
            index_in_bin = []
            for n in range(observations):
                if centered == False:
                    if phase < ob_phaselist[n] < phase + dphase:
                        phases_in_bin.append(ob_phaselist[n])
                        fluxes_in_bin.append(ob_fluxlist[n])
                        errors_in_bin.append(ob_fluxerr[n])
                        index_in_bin.append(n)
                else:
                    if phase == 0:
                        if 0 < ob_phaselist[n] < halfbin:
                            phases_in_bin.append(ob_phaselist[n])
                            fluxes_in_bin.append(ob_fluxlist[n])
                            errors_in_bin.append(ob_fluxerr[n])
                            index_in_bin.append(n)
                        elif 1 - halfbin < ob_phaselist[n] < 1:
                            phases_in_bin.append(ob_phaselist[n])
                            fluxes_in_bin.append(ob_fluxlist[n])
                            errors_in_bin.append(ob_fluxerr[n])
                            index_in_bin.append(n)
                    else:
                        if phase - halfbin < ob_phaselist[n] < phase + halfbin:
                            phases_in_bin.append(ob_phaselist[n])
                            fluxes_in_bin.append(ob_fluxlist[n])
                            errors_in_bin.append(ob_fluxerr[n])
                            index_in_bin.append(n)
            return phases_in_bin, fluxes_in_bin, errors_in_bin, index_in_bin

        # Initialize lists for binned data
        binnedfluxlist = []
        binnederrorlist = []
        avgphaselist = []
        master_phases_in_bin = []
        master_fluxes_in_bin = []
        master_errors_in_bin = []
        master_index_in_bin = []

        # Generate binned data
        binnedphaselist = np.arange(0, 1, dphase)
        for phase in binnedphaselist:
            stuff_at_phase = makebin(phase)
            phases_at_phase = stuff_at_phase[0]
            fluxes_at_phase = stuff_at_phase[1]
            errors_at_phase = stuff_at_phase[2]
            index_at_phase = stuff_at_phase[3]
            master_phases_in_bin.append(np.array(phases_at_phase))
            master_fluxes_in_bin.append(np.array(fluxes_at_phase))
            master_errors_in_bin.append(np.array(errors_at_phase))
            master_index_in_bin.append(index_at_phase)
            if weighted == False:
                binnedfluxlist.append(np.mean(fluxes_at_phase))
                binnederrorlist.append(calc.error.avg(errors_at_phase))
            else:
                w_average = calc.error.weighted_average(fluxes_at_phase, errors_at_phase)
                binnedfluxlist.append(w_average[0])
                binnederrorlist.append(w_average[1])
            avgphaselist.append(np.mean(phases_at_phase))

        # Convert binned fluxes to magnitudes
        binnedmaglist = list(calc.astro.convert.fluxTomag.mag(np.array(binnedfluxlist)))
        binnedmagerr = list(calc.astro.convert.fluxTomag.error(np.array(binnedfluxlist), np.array(binnederrorlist)))

        # Determine normalization factor
        if norm_factor == 'obs':
            norm_f = max(ob_fluxlist)
        elif norm_factor == 'bin':
            norm_f = max(binnedfluxlist)
        elif norm_factor == 'avgmag':
            norm_f = 10 ** (-0.4 * np.mean(ob_maglist))
        else:
            offset = 0.025
            quad1 = []
            quad2 = []
            for n in range(len(ob_phaselist)):
                if 0.25 - offset < ob_phaselist[n] < 0.25 + offset:
                    quad1.append(ob_fluxlist[n])
                elif 0.75 - offset < ob_phaselist[n] < 0.75 + offset:
                    quad2.append(ob_fluxlist[n])
            norm_f = max([np.mean(quad1), np.mean(quad2)])

        # Normalize lists
        def normlist(valuelist, norm_f):
            return np.array(valuelist) / norm_f

        n_binnedfluxlist = normlist(binnedfluxlist, norm_f)
        n_ob_fluxlist = normlist(ob_fluxlist, norm_f)
        n_ob_fluxerr = normlist(ob_fluxerr, norm_f)
        n_binnederrorlist = normlist(binnederrorlist, norm_f)
        for n in range(len(master_fluxes_in_bin)):
            master_fluxes_in_bin[n] /= norm_f
            master_errors_in_bin[n] /= norm_f

        # Assemble data into lists
        master_time = [binnedphaselist, ob_phaselist, avgphaselist, HJD]
        master_norm = [n_binnedfluxlist, n_ob_fluxlist, n_ob_fluxerr, n_binnederrorlist]
        master_ob_flux = [binnedfluxlist, ob_fluxlist, ob_fluxerr, binnederrorlist]
        master_ob_mag = [binnedmaglist, ob_maglist, ob_magerr, binnedmagerr]
        master_stuff_in_bin = [master_phases_in_bin, master_fluxes_in_bin, master_errors_in_bin, master_index_in_bin]

        return master_time, master_norm, master_ob_flux, master_ob_mag, norm_f, master_stuff_in_bin

    def masterbinner_FF(fileName, Epoch, period, bins,
                        HJDcol=0, magcol=1, magerrcol=2,
                        weighted=True, norm_factor='bin', sep=None,
                        header=None, file_type='text', centered=True, pdot=0):
        # Import data from file
        file = io.importFile_pd(fileName, delimit=sep, header=header, file_type=file_type)

        # Extract columns from the file
        HJD = file[HJDcol]
        mag = file[magcol]
        magerr = file[magerrcol]

        # Apply masterbinner function to the extracted data
        MB = binning.masterbinner(HJD, mag, magerr, Epoch, period, bins,
                                  weighted=weighted, norm_factor=norm_factor, centered=centered, pdot=pdot)

        return MB

    def minipolybinner(c_master_phases, c_master_fluxes, nc_master_phases, nc_master_fluxes,
                       section_order, section_res=None):
        # Calculate the number of sections
        sections = len(c_master_phases)

        # Determine section resolution if not provided
        if section_res is None:
            section_res = int(128 / sections)

        # Initialize lists to store polynomial phase and flux
        section_polyphase = []
        section_polyflux = []
        lastphase = []
        lastflux = []

        # Calculate parameters
        halfsec = 0.5 / sections
        dphase = 1 / sections
        bound1 = int(section_res * 0.25)
        bound2 = int(section_res * 0.75)

        # Iterate over each section
        for section in range(sections):
            if section == 0:
                # Handle the first section separately to account for phase wrapping
                for n in range(len(c_master_phases[0])):
                    if 1 - halfsec < c_master_phases[0][n] < 1:
                        c_master_phases[0][n] -= 1
                # Fit polynomial to the section's phase and flux data
                c_coef = calc.poly.regr_polyfit(c_master_phases[0], c_master_fluxes[0], section_order)[0]
                c_polylist = calc.poly.polylist(c_coef, -halfsec, halfsec, section_res)
                c_polyphase = c_polylist[0][bound1:bound2:]
                c_polyflux = c_polylist[1][bound1:bound2:]
                # Split the polynomial data into appropriate sections
                for n in range(len(c_polyphase)):
                    if c_polyphase[n] < 0:
                        lastphase.append(c_polyphase[n] + 1)
                        lastflux.append(c_polyflux[n])
                    else:
                        section_polyphase.append(c_polyphase[n])
                        section_polyflux.append(c_polyflux[n])
                # Fit polynomial to the neighboring section's phase and flux data
                nc_coef = calc.poly.regr_polyfit(nc_master_phases[0], nc_master_fluxes[0], section_order)[0]
                nc_polylist = calc.poly.polylist(nc_coef, 0, dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])
            else:
                # Fit polynomial to the current section's phase and flux data
                c_coef = calc.poly.regr_polyfit(c_master_phases[section], c_master_fluxes[section], section_order)[0]
                c_polylist = calc.poly.polylist(c_coef, section / sections - halfsec, section / sections + halfsec,
                                                section_res)
                section_polyphase += list(c_polylist[0][bound1:bound2:])
                section_polyflux += list(c_polylist[1][bound1:bound2:])
                # Fit polynomial to the neighboring section's phase and flux data
                nc_coef = calc.poly.regr_polyfit(nc_master_phases[section], nc_master_fluxes[section], section_order)[0]
                nc_polylist = calc.poly.polylist(nc_coef, section / sections, section / sections + dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])

        # Add the data from the last phase
        section_polyphase += list(lastphase)
        section_polyflux += list(lastflux)

        # Check if the first phase value is zero
        if section_polyphase[0] != 0.0:
            print('WARNING: Unfit for FT, first phase value not zero.')

        return section_polyphase, section_polyflux

    def polybinner(input_file, Epoch, period, sections=4, norm_factor='alt',
                   section_order=8, FT_order=12, section_res=None, HJD_mag_magerr=[],
                   mag_coef=False, pdot=0):
        # Set section resolution if not provided
        if section_res == None:
            section_res = int(128 / sections)

        # Perform master binning based on provided data or input file
        if len(HJD_mag_magerr) == 3:
            c_MB = binning.masterbinner(HJD_mag_magerr[0], HJD_mag_magerr[1], HJD_mag_magerr[2], Epoch, period,
                                        sections, centered=True, norm_factor=norm_factor, pdot=pdot)
            nc_MB = binning.masterbinner(HJD_mag_magerr[0], HJD_mag_magerr[1], HJD_mag_magerr[2], Epoch, period,
                                         sections, centered=False, norm_factor=norm_factor, pdot=pdot)
        else:
            c_MB = binning.masterbinner_FF(input_file, Epoch, period, sections, centered=True, norm_factor=norm_factor,
                                           pdot=pdot)
            nc_MB = binning.masterbinner_FF(input_file, Epoch, period, sections, centered=False,
                                            norm_factor=norm_factor, pdot=pdot)

        # Extract master binning data
        c_master_phases = c_MB[5][0]
        c_master_fluxes = c_MB[5][1]
        nc_master_phases = nc_MB[5][0]
        nc_master_fluxes = nc_MB[5][1]

        ob_phaselist = c_MB[0][1]
        ob_fluxlist = c_MB[1][1]

        # Initialize lists to store polynomial phase and flux
        section_polyphase = []
        section_polyflux = []
        lastphase = []
        lastflux = []
        halfsec = 0.5 / sections
        dphase = 1 / sections
        bound1 = int(section_res * 0.25)
        bound2 = int(section_res * 0.75)

        # Iterate over each section
        for section in range(sections):
            if section == 0:
                # Handle the first section separately to account for phase wrapping
                for n in range(len(c_master_phases[0])):
                    if 1 - halfsec < c_master_phases[0][n] < 1:
                        c_master_phases[0][n] -= 1
                c_coef = calc.poly.regr_polyfit(c_master_phases[0], c_master_fluxes[0], section_order)[0]
                c_polylist = calc.poly.polylist(c_coef, -halfsec, halfsec, section_res)
                c_polyphase = c_polylist[0][bound1:bound2:]
                c_polyflux = c_polylist[1][bound1:bound2:]
                for n in range(len(c_polyphase)):
                    if c_polyphase[n] < 0:
                        lastphase.append(c_polyphase[n] + 1)
                        lastflux.append(c_polyflux[n])
                    else:
                        section_polyphase.append(c_polyphase[n])
                        section_polyflux.append(c_polyflux[n])
                nc_coef = calc.poly.regr_polyfit(nc_master_phases[0], nc_master_fluxes[0], section_order)[0]
                nc_polylist = calc.poly.polylist(nc_coef, 0, dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])
            else:
                c_coef = calc.poly.regr_polyfit(c_master_phases[section], c_master_fluxes[section], section_order)[0]
                c_polylist = calc.poly.polylist(c_coef, section / sections - halfsec, section / sections + halfsec,
                                                section_res)
                section_polyphase += list(c_polylist[0][bound1:bound2:])
                section_polyflux += list(c_polylist[1][bound1:bound2:])

                nc_coef = calc.poly.regr_polyfit(nc_master_phases[section], nc_master_fluxes[section], section_order)[0]
                nc_polylist = calc.poly.polylist(nc_coef, section / sections, section / sections + dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])

        section_polyphase += list(lastphase)
        section_polyflux += list(lastflux)

        # Calculate Fourier transform coefficients
        if mag_coef == True:
            a, b = FT.coefficients(-2.5 * np.log10(np.array(section_polyflux) * c_MB[4]))[1:3]
        else:
            a, b = FT.coefficients(section_polyflux)[1:3]

        return [a, b], [c_MB, nc_MB], [section_polyphase, section_polyflux]


# ======================================
class FT:  # Fourier transform
    def coefficients(binnedvaluelist):
        """
        Calculates Fourier coefficients based on numPy's fft.ifft function

        Parameters
        ----------
        binnedvaluelist : array like
            List of evenly spaced (binned) values

        Returns
        -------
        Fbfli : array
            combines both a & b coefficients, ignore

        coslist : array
            cosine FT coefficients (a)

        sinlist : array
            sine FT coefficients (b)

        """
        Fbfli = np.fft.ifft(binnedvaluelist, norm=None) * 2
        coslist = np.real(Fbfli)
        sinlist = np.imag(Fbfli)
        coslist[0] = np.mean(binnedvaluelist)
        sinlist[0] = 0
        return Fbfli, coslist, sinlist

    def a_sig_fast(a, b, term, aterm, ob_phase, ob_flux, ob_fluxerr, order, dx0=1):
        """
        Estimates the significance of an 'a' coefficient in Fourier fitting.

        Parameters:
            a (list): List of 'a' coefficients.
            b (list): List of 'b' coefficients.
            term (int): Index of the 'a' coefficient being evaluated.
            aterm (float): Initial guess for the value of the 'a' coefficient.
            ob_phase (array-like): Array of observed phases.
            ob_flux (array-like): Array of observed fluxes.
            ob_fluxerr (array-like): Array of observed flux errors.
            order (int): Order of the Fourier fit.
            dx0 (float): Step size for numerical differentiation (default is 1).

        Returns:
            tuple: A tuple containing the estimated significance and the ratio of upper to lower bounds.
        """
        C = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) + 1

        def spec_a(mod_a):
            a[term] = mod_a
            F = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) - C
            return F

        upper = abs(calc.Newton(spec_a, aterm + dx0, 1e-5, dx=1e-10) - aterm)
        lower = abs(calc.Newton(spec_a, aterm - dx0, 1e-5, dx=1e-10) - aterm)
        a[term] = aterm
        return (upper + lower) / 2, upper / lower

    def b_sig_fast(a, b, term, bterm, ob_phase, ob_flux, ob_fluxerr, order, dx0=1):
        """
        Estimates the significance of a 'b' coefficient in Fourier fitting.

        Parameters:
            a (list): List of 'a' coefficients.
            b (list): List of 'b' coefficients.
            term (int): Index of the 'b' coefficient being evaluated.
            bterm (float): Initial guess for the value of the 'b' coefficient.
            ob_phase (array-like): Array of observed phases.
            ob_flux (array-like): Array of observed fluxes.
            ob_fluxerr (array-like): Array of observed flux errors.
            order (int): Order of the Fourier fit.
            dx0 (float): Step size for numerical differentiation (default is 1).

        Returns:
            tuple: A tuple containing the estimated significance and the ratio of upper to lower bounds.
        """
        C = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) + 1

        def spec_b(mod_b):
            b[term] = mod_b
            F = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) - C
            return F

        upper = abs(calc.Newton(spec_b, bterm + dx0, 1e-5, dx=1e-10) - bterm)
        lower = abs(calc.Newton(spec_b, bterm - dx0, 1e-5, dx=1e-10, central_diff=True) - bterm)
        b[term] = bterm
        return (upper + lower) / 2, upper / lower

    def sumatphase(phase, order, a, b):
        """
        Calculates the amplitude (value) of the Fourier Transform at a given phase.

        Parameters:
            phase (float): Phase at which to calculate the amplitude.
            order (int): Order of the Fourier fit.
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.

        Returns:
            float: Amplitude of the Fourier Transform at the given phase.
        """
        orders = np.arange(order + 1)
        return sum(
            a[:order + 1:] * np.cos(2 * np.pi * phase * orders) + b[:order + 1:] * np.sin(2 * np.pi * phase * orders))

    def deriv_sumatphase(phase, order, a, b):
        """
        Calculates the derivative of the Fourier Transform at a given phase.

        Parameters:
            phase (float): Phase at which to calculate the derivative.
            order (int): Order of the Fourier fit.
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.

        Returns:
            float: Derivative of the Fourier Transform at the given phase.
        """
        deriv_atphase = 0
        for k in range(1, order + 1):
            deriv_atphase += (-b[k] * np.sin(2 * np.pi * k * phase) - a[k] * np.cos(
                2 * np.pi * k * phase)) * 4 * np.pi ** 2 * k ** 2
        return deriv_atphase

    def unc_sumatphase(phase, order, a_unc, b_unc):
        """
        Calculates the uncertainty of the Fourier Transform at a given phase,
        given lists of 'a' and 'b' uncertainties.

        Parameters:
            phase (float): Phase at which to calculate the uncertainty.
            order (int): Order of the Fourier fit.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.

        Returns:
            float: Uncertainty of the Fourier Transform at the given phase.
        """
        unc_I = a_unc[0] ** 2
        for k in range(1, order + 1):
            unc_I += (a_unc[k] * np.cos(2 * np.pi * k * phase)) ** 2 + (b_unc[k] * np.sin(2 * np.pi * k * phase)) ** 2
        return np.sqrt(unc_I)

    def FT_plotlist(a, b, order, resolution):
        """
        Generates the results of a Fourier Transform given the coefficients.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.
            resolution (int): Number of points in the resulting plot.

        Returns:
            tuple: A tuple containing lists of phases, Fourier Transform values, and derivative values.
        """
        phase = 0
        FTfluxlist = []
        FTphaselist = []
        FTderivlist = []
        while (phase < 1):
            FTfluxlist.append(FT.sumatphase(phase, order, a, b))
            FTderivlist.append(FT.deriv_sumatphase(phase, order, a, b))
            FTphaselist.append(round(phase, 7))
            phase += (1 / resolution)
        return FTphaselist, FTfluxlist, FTderivlist

    def sim_ob_flux(FT, ob_fluxerr, lower=-3.0, upper=3.0):
        """
        Modifies a pre-generated synthetic Fourier light-curve amplitude list,
        and alters each value by a capped 3 sigma Gaussian deviate of the observational
        error given the index in ob_fluxerr. The two lists should be sorted by the same
        phase, but sim_FT_curve does this naturally.

        This function is mainly for the sim_FT_curve program and not a standalone function.

        Parameters:
            FT (array-like): Synthetic Fourier light-curve amplitude list.
            ob_fluxerr (array-like): Array of observational flux errors.
            lower (float): Lower limit for Gaussian deviate (default is -3.0).
            upper (float): Upper limit for Gaussian deviate (default is 3.0).

        Returns:
            array-like: Modified synthetic Fourier light-curve amplitude list.
        """
        obs = len(ob_fluxerr)
        devlist = calc.error.truncnorm(obs, lower=lower, upper=upper)
        sim_ob_flux = []
        for n in range(obs):
            sim_ob_flux.append(FT[n] + devlist[n] * ob_fluxerr[n])
        return sim_ob_flux

    def synth(a, b, ob_phaselist, order):
        """
        Generates a synthetic light curve using the Fourier coefficients.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            ob_phaselist (array-like): Array of observed phases.
            order (int): Order of the Fourier fit.

        Returns:
            array-like: Synthetic light curve.
        """
        synthlist = []
        for phase in ob_phaselist:
            synthlist.append(FT.sumatphase(phase, order, a, b))
        return synthlist

    def int_sumatphase(a, b, phase, order):
        """
        Calculates the integral of the Fourier Transform at a given phase.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            phase (float): Phase at which to calculate the integral.
            order (int): Order of the Fourier fit.

        Returns:
            float: Integral of the Fourier Transform at the given phase.
        """
        nlist = np.arange(order + 1)[1::]
        return a[0] * phase + (0.5 / np.pi) * sum((a[1:order + 1:] * np.sin(2 * np.pi * nlist * phase) - b[
                                                                                                         1:order + 1:] * np.cos(
            2 * np.pi * nlist * phase)) / nlist)

    def integral(a, b, order, lowerphase, upperphase):
        """
        Calculates the definite integral of the Fourier Transform.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.
            lowerphase (float): Lower bound of the integral.
            upperphase (float): Upper bound of the integral.

        Returns:
            float: Definite integral of the Fourier Transform.
        """
        uppersum = FT.int_sumatphase(a, b, upperphase, order)
        lowersum = FT.int_sumatphase(a, b, lowerphase, order)
        return uppersum - lowersum

    def int_unc_atphase(phase, a_unc, b_unc):
        """
        Calculates the uncertainty of the definite integral at a given phase.

        Parameters:
            phase (float): Phase at which to calculate the uncertainty.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.

        Returns:
            float: Uncertainty of the definite integral at the given phase.
        """
        orderp1 = len(a_unc)
        int_unc = 0
        for k in range(1, orderp1):
            int_unc += ((a_unc[k] / k) * np.sin(2 * np.pi * phase * k)) ** 2 + (
                    (b_unc[k] / k) * np.cos(2 * np.pi * phase * k)) ** 2
        int_unc *= 0.25 / np.pi ** 2
        int_unc += (a_unc[0] * phase) ** 2
        return np.sqrt(int_unc)


# ======================================
class OConnell:  # O'Connell effect
    def OER_FT(a, b, order):
        """
        Calculates the Odd-Even Ratio (OER) of the Fourier Transform.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            float: Odd-Even Ratio of the Fourier Transform.
        """
        nlist = np.arange(order + 1)[1::]
        A = 0.5 * sum(a[nlist])
        B = sum(b[nlist[::2]] / nlist[::2]) / np.pi
        return 1 - (2 / (1 + A / B))

    def OER_FT_error(a, b, a_unc, b_unc, order):
        """
        Calculates both the Odd-Even Ratio (OER) and its error of the Fourier Transform.
        Requires the 'a' and 'b' coefficients, and their respective uncertainties.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            tuple: A tuple containing the Odd-Even Ratio and its error.
        """
        I0_area = sum(a[:order + 1:]) * 0.5
        Top = FT.integral(a, b, order, 0, 0.5) - I0_area
        Bot = FT.integral(a, b, order, 0.5, 1) - I0_area
        OER = Top / Bot
        sigsum = calc.error.sig_sum(a_unc)
        uncTop = np.sqrt((FT.int_unc_atphase(0.5, a_unc, b_unc)) ** 2 + (FT.int_unc_atphase(0, a_unc, b_unc)) ** 2 + (
                0.5 * sigsum) ** 2)
        uncBot = np.sqrt((FT.int_unc_atphase(1, a_unc, b_unc)) ** 2 + (FT.int_unc_atphase(0.5, a_unc, b_unc)) ** 2 + (
                0.5 * sigsum) ** 2)
        OER_unc = OER * np.sqrt((uncTop / Top) ** 2 + (uncBot / Bot) ** 2)
        return OER, OER_unc

    def OER_FT_error_fixed(a, b, a_unc, b_unc, order):
        """
        Calculates the error of the Odd-Even Ratio (OER) of the Fourier Transform with fixed coefficients.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            float: Error of the Odd-Even Ratio of the Fourier Transform.
        """
        a = np.array(a)
        b = np.array(b)
        a_unc = np.array(a_unc)
        b_unc = np.array(b_unc)
        nlist = np.arange(order + 1)[1::]
        A = 0.5 * sum(a[nlist])
        B = sum(b[nlist[::2]] / nlist[::2]) / np.pi
        sig_A2 = 0.25 * sum(a_unc[nlist] ** 2)
        sig_B2 = sum((b_unc[nlist[::2]] / nlist[::2]) ** 2) / (np.pi ** 2)
        return np.sqrt(sig_A2 / A ** 2 + sig_B2 / B ** 2) / abs(1 + A / (2 * B) + B / (2 * A))

    def LCA_FT(a, b, order, resolution):
        """
        Calculates the Light Curve Asymmetry (LCA) given the 'a' and 'b' coefficients.
        Must specify the resolution for numerical integration.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.
            resolution (int): Number of points for integration.

        Returns:
            float: Light Curve Asymmetry (LCA).
        """
        import scipy
        Phi = np.linspace(0, 0.5, resolution)
        no_a = np.zeros(len(a))
        K2 = []
        for phase in Phi:
            K2.append(((2 * FT.sumatphase(phase, order, no_a, b)) / FT.sumatphase(phase, order, a, b)) ** 2)
        LCA = np.sqrt(scipy.integrate.simps(K2, Phi))
        return LCA

    def L_error(phase, order, a, b, a_unc, b_unc):
        """
        Calculates the error in the LCA integrand, which is then used to find the error in the LCA.
        Requires 'a' and 'b' coefficients and their uncertainties.

        Parameters:
            phase (float): Phase value.
            order (int): Order of the Fourier fit.
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.

        Returns:
            tuple: A tuple containing the LCA integrand value and its error.
        """
        I = FT.sumatphase(phase, order, a, b)  # Compute the flux value at the given phase
        J = 2 * FT.sumatphase(phase, order, np.zeros(order + 1), b)  # Compute a component of the LCA integrand
        K = J / I
        L = K ** 2  # Calculate the LCA integrand value

        # Calculate partial derivatives of L with respect to 'a' and 'b' coefficients
        dL_da0 = -2 * L / I
        dL_dak = []
        dL_dbk = []
        for k in range(1, order + 1):
            dL_dak.append(dL_da0 * np.cos(2 * np.pi * k * phase))
            dL_dbk.append(2 * np.sin(2 * np.pi * k * phase) * (2 * J / I ** 2 - J ** 2 / I ** 3))

        # Compute the uncertainty in the LCA integrand using error propagation
        L_err = (dL_da0 * a_unc[0]) ** 2
        for n in range(len(dL_dak)):
            L_err += (dL_dak[n] * a_unc[n + 1]) ** 2 + (dL_dbk[n] * b_unc[n + 1]) ** 2
        L_err = np.sqrt(L_err)

        return L, L_err

    def LCA_FT_error(a, b, a_unc, b_unc, order, resolution):
        """
        Calculates the uncertainty in the Fourier Transform Light Curve Asymmetry (LCA).

        This function computes the uncertainty in the LCA by integrating the error contributions
        at different phases using Simpson's rule. It utilizes the L_error function to calculate
        the LCA integrand at each phase, which provides both the LCA value and its uncertainty.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.
            resolution (int): Number of points for integration.

        Returns:
            tuple: A tuple containing the LCA value and its uncertainty.
        """
        import scipy

        # Generate phase values for integration
        Phi = np.linspace(0, 0.5, resolution)

        # Initialize lists to store LCA integrand values and their uncertainties at each phase
        Llist = []
        Lerrlist = []

        # Compute the LCA integrand and its uncertainties at each phase
        for phase in Phi:
            L_ap = OConnell.L_error(phase, order, a, b, a_unc, b_unc)
            Llist.append(L_ap[0])
            Lerrlist.append(L_ap[1])

        # Integrate the LCA integrand and its uncertainties using Simpson's rule
        int_L = scipy.integrate.simps(Llist, Phi)
        int_L_error = scipy.integrate.simps(Lerrlist, Phi)

        # Compute the LCA value and its uncertainty
        LCA = OConnell.LCA_FT(a, b, order, resolution)
        LCA_error = 0.5 * LCA * (int_L_error / int_L)

        return LCA, LCA_error

    def LCA_FT_error2(a, b, a_unc, b_unc, order, resolution):
        """
        Calculates the uncertainty in the Fourier Transform Light Curve Asymmetry (LCA) using an alternative method.

        This function computes the uncertainty in the LCA by directly evaluating the LCA integrand
        at different phases. It then integrates these values to obtain the total uncertainty in the LCA.
        It is an alternative approach to LCA_FT_error, utilizing a different method for calculating the
        uncertainty contribution at each phase.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.
            resolution (int): Number of points for integration.

        Returns:
            tuple: A tuple containing the LCA value and its uncertainty.

        """
        a, a_unc = np.array(a), np.array(a_unc)
        b, b_unc = np.array(b), np.array(b_unc)

        def L_error2(phase):
            """
            Calculates the error contribution for the Fourier Transform Light Curve Asymmetry (LCA) integrand at a given phase.

            This function computes the error contribution of the LCA integrand at a specified phase.
            It first calculates the derivative of the flux with respect to phase, 'dIphi', and the flux value 'I'
            at the given phase using the provided 'a' and 'b' coefficients. Then, it computes the LCA integrand
            error term 'L', which is the square of the ratio of 'dIphi' to 'I'. Additionally, it calculates the
            uncertainty in 'L' using error propagation, considering the uncertainties in 'a' and 'b' coefficients.

            Parameters:
                phase (float): Phase at which to compute the error contribution.

            Returns:
                tuple: A tuple containing the LCA integrand error term 'L' and its uncertainty.
            """
            nlist = np.arange(order + 1)  # Generate the list of orders
            dIphi = 2 * sum(
                b[nlist] * np.sin(2 * np.pi * nlist * phase))  # Compute the derivative of flux with respect to phase
            I = FT.sumatphase(phase, order, a, b)  # Compute the flux value at the given phase
            L = (dIphi / I) ** 2  # Calculate the LCA integrand error term
            # Calculate the uncertainty in the LCA integrand using error propagation
            L_err = 2 / I * (2 * L ** 0.5 - L) * np.sqrt(sum((b_unc * np.sin(2 * np.pi * phase * nlist)) ** 2))
            return L, L_err

        import scipy
        Phi = np.linspace(0, 0.5, resolution)
        Llist = []
        Lerrlist = []
        for phase in Phi:
            L_ap = L_error2(phase)
            Llist.append(L_ap[0])
            Lerrlist.append(L_ap[1])
        int_L = scipy.integrate.simps(Llist, Phi)
        int_L_error = scipy.integrate.simps(Lerrlist, Phi)
        # print('L_err =',int_L_error,'L =',int_L)
        LCA = OConnell.LCA_FT(a, b, order, resolution)
        LCA_error = 0.5 * LCA * (int_L_error / int_L)
        return LCA, LCA_error

    def Delta_I(a, b, order):
        """
        Directly calculates the difference in flux at the two quadratures (0.25, 0.75),
        given the 'a' and 'b' coefficients.

        This function calculates the difference in flux between two quadrature phases (0.25 and 0.75),
        providing insight into the O'Connell effect. It returns the difference in flux, as well as the
        flux values at the specified phases.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            tuple: A tuple containing the difference in flux, flux at phase 0.25, and flux at phase 0.75.
        """
        Ip = FT.sumatphase(0.25, order, a, b)
        Is = FT.sumatphase(0.75, order, a, b)
        dI = Ip - Is
        return dI, Ip, Is

    def Delta_I_error(a, b, a_unc, b_unc, order):
        """
        Calculates the uncertainty in dI of the Fourier Transform.

        Parameters:
            a (array-like): List of 'a' coefficients.
            b (array-like): List of 'b' coefficients.
            a_unc (array-like): List of uncertainties for 'a' coefficients.
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            tuple: A tuple containing the dI value and its uncertainty.
        """
        # Calculate uncertainties at quadrature phases
        sig_Ip = FT.unc_sumatphase(0.25, order, a_unc, b_unc)
        sig_Is = FT.unc_sumatphase(0.75, order, a_unc, b_unc)

        # Compute total uncertainty in dI
        sig_dI = np.sqrt(sig_Ip ** 2 + sig_Is ** 2)

        # Calculate dI value
        dI = OConnell.Delta_I(a, b, order)[0]

        return dI, sig_dI

    def Delta_I_fixed(b, order):
        """
        Calculates the difference in flux at the two quadratures (0.25, 0.75)
        for a given set of 'b' coefficients.

        Parameters:
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            float: Difference in flux at quadratures.
        """
        mlist = np.arange(int(np.ceil(order / 2)))
        return 2 * sum((-1) ** mlist * b[2 * mlist + 1])

    def Delta_I_error_fixed(b_unc, order):
        """
        Calculates the uncertainty in dI_fixed of the Fourier Transform.

        Parameters:
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.

        Returns:
            float: Uncertainty in dI_fixed.
        """
        # Calculate total uncertainty in dI_fixed
        return 2 * np.sqrt(sum(np.array(b_unc)[1:order + 1:2] ** 2))

    def dI_at_phase(b, order, phase):
        """
        Calculates the difference in flux at a specific phase for a given set of 'b' coefficients.

        Parameters:
            b (array-like): List of 'b' coefficients.
            order (int): Order of the Fourier fit.
            phase (float): Phase value.

        Returns:
            float: Difference in flux at the specified phase.
        """
        nlist = np.arange(order + 1)
        return 2 * sum(b[nlist] * np.sin(2 * np.pi * nlist * phase))

    def dI_at_phase_error(b_unc, order, phase):
        """
        Calculates the uncertainty in dI at a specific phase for a given set of 'b' coefficients.

        Parameters:
            b_unc (array-like): List of uncertainties for 'b' coefficients.
            order (int): Order of the Fourier fit.
            phase (float): Phase value.

        Returns:
            float: Uncertainty in dI at the specified phase.
        """
        nlist = np.arange(order + 1)
        return 2 * np.sqrt(sum((b_unc[nlist] * np.sin(2 * np.pi * nlist * phase)) ** 2))

    def Delta_I_mean_obs(ob_phaselist, ob_fluxlist, ob_fluxerr, phase_range=0.05, weighted=False):
        """
        Calculates the difference of flux at quadrature from observational fluxes
        and the uncertainty of this measure.

        Parameters:
            ob_phaselist (array-like): List of observational phase values.
            ob_fluxlist (array-like): List of observational flux values.
            ob_fluxerr (array-like): List of observational flux errors.
            phase_range (float): Range around quadrature phase to consider.
            weighted (bool): Flag to indicate whether to use weighted averaging.

        Returns:
            tuple: A tuple containing the mean difference in flux at quadrature and its uncertainty.
        """
        Iplist = []
        Iperrors = []
        Islist = []
        Iserrors = []

        # Extract fluxes and errors within specified phase ranges from quadrature
        for n in range(len(ob_phaselist)):
            if 0.25 - phase_range < ob_phaselist[n] < 0.25 + phase_range:
                Iplist.append(ob_fluxlist[n])
                Iperrors.append(ob_fluxerr[n])
            if 0.75 - phase_range < ob_phaselist[n] < 0.75 + phase_range:
                Islist.append(ob_fluxlist[n])
                Iserrors.append(ob_fluxerr[n])

        if weighted:
            # Calculate weighted averages if specified
            Ipmean = calc.error.weighted_average(Iplist, Iperrors)
            Ismean = calc.error.weighted_average(Islist, Iserrors)
            dI_mean_obs = Ipmean[0] - Ismean[0]
            dI_mean_error = np.sqrt(Ipmean[1] ** 2 + Ismean[1] ** 2)
        else:
            # Calculate simple means otherwise
            dI_mean_obs = np.mean(Iplist) - np.mean(Islist)
            dI_mean_error = np.sqrt(calc.error.avg(Iperrors) ** 2 + calc.error.avg(Iserrors) ** 2)

        return dI_mean_obs, dI_mean_error

    def Delta_I_mean_obs_noerror(ob_phaselist, ob_fluxlist, phase_range=0.05):
        """
        Calculates the mean difference in flux at quadrature from observational fluxes
        without considering observational errors.

        Parameters:
            ob_phaselist (array-like): List of observational phase values.
            ob_fluxlist (array-like): List of observational flux values.
            phase_range (float): Range around quadrature phase to consider.

        Returns:
            float: Mean difference in flux at quadrature from observational fluxes.
        """
        Iplist = []
        Islist = []

        # Extract fluxes within specified phase ranges from quadrature
        for n in range(len(ob_phaselist)):
            if 0.25 - phase_range < ob_phaselist[n] < 0.25 + phase_range:
                Iplist.append(ob_fluxlist[n])
            if 0.75 - phase_range < ob_phaselist[n] < 0.75 + phase_range:
                Islist.append(ob_fluxlist[n])

        # Calculate mean difference in flux at quadrature
        dI_mean_obs = np.mean(Iplist) - np.mean(Islist)

        return dI_mean_obs


# Function to calculate the sum of reciprocals of squared errors
def M(errorlist):
    """
    Calculates the sum of reciprocals of squared errors.

    Parameters:
        errorlist (array-like): List of errors.

    Returns:
        float: Sum of reciprocals of squared errors.
    """
    M0 = 0
    M = M0
    for error in errorlist:
        M += 1 / error ** 2
    return M


# Function to calculate the weight factor
def wfactor(errorlist, n, M):
    """
    Calculates the weight factor for a given error.

    Parameters:
        errorlist (array-like): List of errors.
        n (int): Index of the error in the list.
        M (float): Sum of reciprocals of squared errors.

    Returns:
        float: Weight factor for the error at index n.
    """
    return 1 / (errorlist[n] ** 2 * M)


# Class Flower containing functions for calculations related to Flower 1996 and Torres 2010
class Flower:
    class T:
        # Coefficients for Teff calculation
        c = [3.97914510671409, -0.654992268598245, 1.74069004238509, -4.60881515405716, 6.79259977994447,
             -5.39690989132252, 2.19297037652249, -0.359495739295671]

        def Teff(BV, error):
            """
            Calculates the effective temperature (Teff) using the B-V color index.

            Parameters:
                BV (float): B-V color index.
                error (float): Error in B-V color index.

            Returns:
                tuple: A tuple containing the Teff value and its error.
            """
            # Calculate Teff using polynomial approximation
            temp = calc.poly.power(Flower.T.c, BV, 10)
            # Calculate error in Teff using error propagation
            err = calc.poly.t_eff_err(Flower.T.c, BV, error, temp)
            return temp, err



# Class containing functions for calculations related to Harmanec
class Harmanec:
    class mass:
        # Coefficients for M1 calculation
        c = [-121.6782, 88.057, -21.46965, 1.771141]

        def M1(BV):
            """
            Calculates the mass of a star using the B-V color index.

            Parameters:
                BV (float): B-V color index.

            Returns:
                float: Mass of the star.
            """
            # Calculate M1 using polynomial approximation
            M1 = 10 ** (calc.poly.result(Harmanec.mass.c, np.log10(Flower.T.Teff(BV))))
            return M1


# Class containing constants and functions for interstellar reddening
class Red:
    # Constants for color excess
    J_K = 0.17084
    J_H = 0.10554
    V_R = 0.58 / 3.1

    def colorEx(filter1, filter2, Av):
        """
        Calculates the color excess in a specified filter combination.

        Parameters:
            filter1 (str): First filter.
            filter2 (str): Second filter.
            Av (float): V-band extinction.

        Returns:
            float: Color excess.
        """
        excess = str(filter1) + '_' + str(filter2)
        if excess == 'J_K':
            return Av * Red.J_K
        elif excess == 'J_H':
            return Av * Red.J_H
        elif excess == 'V_R':
            return Av * Red.V_R



# Class for plotting functions
class plot:
    def amp(valuelist):
        """
        Calculates the amplitude of a list of values.

        Parameters:
            valuelist (list): List of values.

        Returns:
            float: Amplitude of the values.
        """
        return max(valuelist) - min(valuelist)

    def aliasing2(phaselist, maglist, errorlist, alias=0.6):
        """
        Filters the phase, magnitude, and error lists to keep only data within the specified alias range.

        Parameters:
            phaselist (list): List of phase values.
            maglist (list): List of magnitude values.
            errorlist (list): List of error values.
            alias (float): Alias range.

        Returns:
            tuple: Filtered phase, magnitude, and error lists.
        """
        phase = list(np.array(phaselist) - 1) + list(phaselist)
        mag = list(maglist) + list(maglist)
        error = list(errorlist) + list(errorlist)
        a_phase = []
        a_mag = []
        a_error = []
        for n in range(len(phase)):
            if -alias < phase[n] < alias:
                a_phase.append(phase[n])
                a_mag.append(mag[n])
                a_error.append(error[n])
        return a_phase, a_mag, a_error

    def multiplot(figsize=(8, 8), dpi=256, height_ratios=[3, 1], hspace=0, sharex=True, sharey=False, fig=None):
        """
        Creates a multiplot with specified parameters.

        Parameters:
            figsize (tuple): Figure size.
            dpi (int): Dots per inch.
            height_ratios (list): List of height ratios for subplots.
            hspace (float): Height space between subplots.
            sharex (bool): Share x-axis among subplots.
            sharey (bool): Share y-axis among subplots.
            fig (object): Figure object.

        Returns:
            tuple: Axes and figure objects.
        """
        if fig == None:
            fig = plt.figure(1, figsize=figsize, dpi=dpi)
        axs = fig.subplots(len(height_ratios), sharex=sharex, sharey=sharey,
                           gridspec_kw={'hspace': hspace, 'height_ratios': height_ratios})
        return axs, fig

    def sm_format(ax, X=None, x=None, Y=None, y=None, Xsize=7, xsize=3.5, tickwidth=1,
                  xtop=True, xbottom=True, yright=True, yleft=True, numbersize=12, autoticks=True,
                  topspine=True, bottomspine=True, rightspine=True, leftspine=True, xformatter=True,
                  xdirection='in', ydirection='in', spines=True):
        """
        Formats the plot axis.

        Parameters:
            ax (object): Axis object.
            X (float): Major tick spacing on x-axis.
            x (float): Minor tick spacing on x-axis.
            Y (float): Major tick spacing on y-axis.
            y (float): Minor tick spacing on y-axis.
            Xsize (float): Size of major ticks on x-axis.
            xsize (float): Size of minor ticks on x-axis.
            tickwidth (float): Tick width.
            xtop (bool): Display ticks on top of x-axis.
            xbottom (bool): Display ticks on bottom of x-axis.
            yright (bool): Display ticks on right side of y-axis.
            yleft (bool): Display ticks on left side of y-axis.
            numbersize (int): Size of tick labels.
            autoticks (bool): Use automatic tick spacing.
            topspine (bool): Display top spine.
            bottomspine (bool): Display bottom spine.
            rightspine (bool): Display right spine.
            leftspine (bool): Display left spine.
            xformatter (bool): Use x-axis tick formatter.
            xdirection (str): Tick direction on x-axis.
            ydirection (str): Tick direction on y-axis.
            spines (bool): Display axis spines.

        Returns:
            str: 'DONE' when formatting is complete.
        """
        ax.tick_params(axis='x', which='major', length=Xsize, width=tickwidth, direction=xdirection, top=xtop,
                       bottom=xbottom, labelsize=numbersize)
        ax.tick_params(axis='y', which='major', length=Xsize, width=tickwidth, direction=ydirection, right=yright,
                       labelsize=numbersize, left=yleft)
        ax.tick_params(axis='x', which='minor', length=xsize, width=tickwidth, direction=xdirection, top=xtop,
                       bottom=xbottom)
        ax.tick_params(axis='y', which='minor', length=xsize, width=tickwidth, direction=ydirection, right=yright,
                       left=yleft)
        if spines == False:
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
        else:
            ax.spines['top'].set_linewidth(tickwidth)
            ax.spines['left'].set_linewidth(tickwidth)
            ax.spines['right'].set_linewidth(tickwidth)
            ax.spines['bottom'].set_linewidth(tickwidth)
            ax.spines['bottom'].set_visible(bottomspine)
            ax.spines['top'].set_visible(topspine)
            ax.spines['right'].set_visible(rightspine)
            ax.spines['left'].set_visible(leftspine)
        if X == None:
            ax.xaxis.set_major_locator(AutoLocator())
        else:
            ax.xaxis.set_major_locator(MultipleLocator(X))
        if x == None:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.xaxis.set_minor_locator(MultipleLocator(x))
        if Y == None:
            ax.yaxis.set_major_locator(AutoLocator())
        else:
            ax.yaxis.set_major_locator(MultipleLocator(Y))
        if y == None:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.yaxis.set_minor_locator(MultipleLocator(y))
        if xformatter == True:
            ax.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
        return 'DONE'



# ======================================
class Roche:
    def Kopal_cyl(rho, phi, z, q):
        """
        Calculates the Kopal potential for a cylindrical coordinate system.

        Parameters:
            rho (float): Radial coordinate.
            phi (float): Azimuthal coordinate.
            z (float): Vertical coordinate.
            q (float): Mass ratio.

        Returns:
            float: Value of the Kopal potential.
        """
        return 1 / np.sqrt(rho ** 2 + z ** 2) + q / (
            np.sqrt(1 + rho ** 2 + z ** 2 - 2 * rho * np.cos(phi))) - q * rho * np.cos(phi) + 0.5 * (1 + q) * rho ** 2

    def gen_Kopal_cyl(rho, phi, z, q,
                      xcm=None, ycm=0, zcm=0,
                      potcap=None):
        """
        Generates the Kopal potential for a cylindrical coordinate system.

        Parameters:
            rho (float): Radial coordinate.
            phi (float): Azimuthal coordinate.
            z (float): Vertical coordinate.
            q (float): Mass ratio.
            xcm (float, optional): x-coordinate of the center of mass.
            ycm (float, optional): y-coordinate of the center of mass.
            zcm (float, optional): z-coordinate of the center of mass.
            potcap (float, optional): Potential cap.

        Returns:
            float: Value of the generated Kopal potential.
        """
        if xcm == None:
            xcm = q / (1 + q)
        A1 = -q / (1 + q)
        A2 = 1 / (1 + q)
        B1 = xcm ** 2 + ycm ** 2 + zcm ** 2 + 2 * xcm * A1 + A1 ** 2
        B2 = xcm ** 2 + ycm ** 2 + zcm ** 2 + 2 * xcm * A2 + A2 ** 2
        X = rho * np.cos(phi)
        Y = rho * np.sin(phi)
        s1 = np.sqrt(rho ** 2 + z ** 2 - 2 * (X * (xcm + A1) + Y * ycm + z * zcm) + B1)
        s2 = np.sqrt(rho ** 2 + z ** 2 - 2 * (X * (xcm + A2) + Y * ycm + z * zcm) + B2)
        rw2 = rho ** 2 - 2 * (xcm * X + ycm * Y) + xcm ** 2 + ycm ** 2
        potent = 1 / s1 + q / s2 + 0.5 * (1 + q) * rw2 - 0.5 * q ** 2 / (1 + q)
        return potent

    def Lagrange_123(q, e=1e-8):
        """
        Calculates the Lagrange points L1, L2, and L3 for a given mass ratio.

        Parameters:
            q (float): Mass ratio.
            e (float, optional): Tolerance for the Newton-Raphson method.

        Returns:
            tuple: Values of Lagrange points L1, L2, and L3.
        """
        L1 = lambda x: q / x ** 2 - x * (1 + q) - 1 / (1 - x) ** 2 + 1
        L2 = lambda x: q / x ** 2 - x * (1 + q) + 1 / (1 + x) ** 2 - 1
        L3 = lambda x: 1 / (q * x ** 2) - x * (1 + 1 / q) + 1 / (1 + x) ** 2 - 1
        xL1 = calc.Newton(L1, 0.5, e=e)
        xL2 = calc.Newton(L2, 0.5, e=e)
        xL3 = calc.Newton(L3, 0.5, e=e)
        return xL1, xL2, xL3

    def Kopal_zero(rho, phi, z, q, Kopal, body='M1'):
        """
        Calculates the Kopal potential for zero equilibrium.

        Parameters:
            rho (float): Radial coordinate.
            phi (float): Azimuthal coordinate.
            z (float): Vertical coordinate.
            q (float): Mass ratio.
            Kopal (float): Kopal potential.
            body (str, optional): Body type ('M1', 'M2', 'dM1', 'dM2').

        Returns:
            float: Value of the Kopal potential for zero equilibrium.
        """
        r = rho
        if body == 'M1':
            return 1 / np.sqrt(r ** 2 + z ** 2) + q / np.sqrt(
                1 + r ** 2 + z ** 2 - 2 * r * np.cos(phi)) - q * r * np.cos(phi) + 0.5 * (1 + q) * r ** 2 - Kopal
        elif body == 'M2':
            return q / np.sqrt(r ** 2 + z ** 2) + 1 / np.sqrt(1 + r ** 2 + z ** 2 + 2 * r * np.cos(phi)) + r * np.cos(
                phi) + 0.5 * (1 + q) * r ** 2 + (1 - q) / 2 - Kopal
        elif body == 'dM1':
            return -r / (r ** 2 + z ** 2) ** (3 / 2) - q * (r - np.cos(phi)) / (
                    1 + r ** 2 + z ** 2 - 2 * r * np.cos(phi)) ** (3 / 2) - q * np.cos(phi) + (1 + q) * r
        elif body == 'dM2':
            return -r * q / (r ** 2 + z ** 2) ** (3 / 2) - (r + np.cos(phi)) / (
                    1 + r ** 2 + z ** 2 + 2 * r * np.cos(phi)) ** (3 / 2) + np.cos(phi) + (1 + q) * r
        else:
            return print('Invalid body, choose M1 or M2.')

    def gen_Kopal_zero(rho, phi, z, q, Kopal,
                       xcm=None, ycm=0, zcm=0):
        """
        Generates the Kopal potential for zero equilibrium.

        Parameters:
            rho (float): Radial coordinate.
            phi (float): Azimuthal coordinate.
            z (float): Vertical coordinate.
            q (float): Mass ratio.
            Kopal (float): Kopal potential.
            xcm (float, optional): x-coordinate of the center of mass.
            ycm (float, optional): y-coordinate of the center of mass.
            zcm (float, optional): z-coordinate of the center of mass.

        Returns:
            float: Value of the generated Kopal potential for zero equilibrium.
        """
        return Roche.gen_Kopal_cyl(rho, phi, z, q, xcm=xcm, ycm=ycm, zcm=zcm) - Kopal


class Pecaut:  # V-R effective temperature fit from Pecaut and Mamajek 2013 https://arxiv.org/pdf/1307.2657.pdf
    class T:
        """
        Class T defines the effective temperature (Teff) as a function of color index (V-R).

        Attributes:
            c1 (list): Coefficients for Teff calculation when -0.115 < V-R <= 0.019 (B1V to A1V).
            c2 (list): Coefficients for Teff calculation when 0.019 < V-R < 1.079 (A1V to M3V).
            c1_err (list): Coefficients errors for c1.
            c2_err (list): Coefficients errors for c2.
        """

        c1 = [3.98007, -1.2742, 32.41373, 98.06855]  # -0.115 < V-R < 0.019
        c2 = [3.9764, -0.80319, 0.64832, -0.2651]  # 0.019 < V-R < 1.079

        c1_err = [0.00499, 0.25431, 6.60765, 41.6003]
        c2_err = [0.00179, 0.01409, 0.03136, 0.0197]

        def Teff(VR, error):
            """
            Calculates the effective temperature (Teff) given the color index (V-R).

            Parameters:
                VR (float): Color index (V-R).
                error (float): Error in color index (V-R).

            Returns:
                tuple: A tuple containing the calculated Teff value and its error.
            """
            if -0.115 < VR <= 0.019:  # B1V to A1V
                temp = calc.poly.power(Pecaut.T.c1, VR, 10)
                err = calc.poly.t_eff_err(Pecaut.T.c1, VR, error, temp, coeferror=Pecaut.T.c1_err)
                return temp, err
            elif 0.019 < VR < 1.079:  # A1V to M3V
                temp = calc.poly.power(Pecaut.T.c2, VR, 10)
                err = calc.poly.t_eff_err(Pecaut.T.c2, VR, error, temp, coeferror=Pecaut.T.c2_err)
                return temp, err
            else:
                return 0, 0

