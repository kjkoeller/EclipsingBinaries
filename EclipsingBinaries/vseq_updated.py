# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:09:28 2020
@author: Alec Neal

Last Updated: 04/27/2024
Last Editor: Kyle Koeller

Collection of functions, coefficients and equations commonly used
with short-period variable stars, but many can be used more
generally
"""
import numpy as np
# import csv
# import statistics as st
# import scipy.stats as sci
# import scipy
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


# ======================================
class io:
    def importFile_pd(inputFile, delimit=None, header=None, file_type='text', engine='python', delim_whitespace=True):
        if file_type == 'text':
            if delim_whitespace == True:
                file = pd.read_csv(inputFile, delim_whitespace=True, header=header, engine='python')
            else:
                file = pd.read_csv(inputFile, sep=delimit, header=header, engine='python')
        elif file_type == 'excel':
            file = pd.read_excel(inputFile, sep=delimit, header=header, engine='python')
        else:
            print('File type not currently supported. Choose text or excel type.')
        columnlist = []
        for column in range(len(list(file))):
            columnlist.append(list(file[column]))
            # print(columnlist)
        return columnlist


# =======================================
class calc:  # assortment of functions
    def frac(x):
        return x - np.floor(x)

    def Newton(f, x0, e=1e-8, fprime=None, max_iter=None, dx=1e-8, central_diff=True, print_iters=False):
        x = x0
        iters = 0
        if fprime == None:
            if central_diff == False:
                fprime = lambda x, dx=dx: (f(x + dx) / (dx))
            else:
                fprime = lambda x, dx=dx: (f(x + dx) - f(x - dx)) / (2 * dx)
        if max_iter == None:
            while abs(f(x)) > e:
                x -= f(x) / fprime(x)
                iters += 1
            # print(iters)
            # print(x/x0)
            if print_iters == True:
                print('iters:', iters)
            return x
        else:
            iters = 0
            while abs(f(x)) > e:
                # while abs(f(x)) > e:
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
            Result of a polynomial given an ascending order coefficient list, and
            an x value.
            """
            # n0=0
            # n=n0
            # termlist=[]
            # while(n<len(coeflist)):
            # termlist.append(coeflist[n]*value**n)
            # n+=1
            # return sum(termlist)
            # deg = len(coeflist)-1
            # coeflist=np.array(coeflist)
            # if deriv == True:
            #

            return sum(np.array(coeflist) * value ** np.arange(len(coeflist)))

        def error(coeflist, value, error):
            """
            Propagated uncertainty of a standard polynomial.

            coeflist: ascending order coefficient list

            value: input value

            error: error in value
            """
            n0 = 1
            n = n0
            errlist = []
            while n < len(coeflist):
                errlist.append(n * coeflist[n] * value ** (n - 1))
                n += 1
            return error * sum(errlist)

        def polylist(coeflist, xmin, xmax, resolution):
            """
            Generates a list of predicted values from a given
            ascending coefficient list. Specifiy the bounds, and the
            resolution (number of points). x domain is evenly spaced.

            Useful for plotting or Fourier transforms.
            """
            xrange = abs(xmax - xmin)
            xlist = np.arange(xmin, xmax, xrange / resolution)
            ylist = []
            for x in xlist:
                ylist.append(calc.poly.result(coeflist, x))
            return xlist, ylist

        def regr_polyfit(x, y, deg, func=lambda x, n: x ** n, sig_y=None):
            """
            Performs a least squares fit of the chosen order (deg).
            [0] Returns an ascending coefficient list, [1] the standard error
            of the coefficients, [2] the coefficient of determination (R squared),
            [3] and the predicted values.
            """
            import statsmodels.api as sm
            x = np.array(x)
            """
            Constructing the Vandermonde matrix.
            """
            Xlist = []
            for n in range(1, deg + 1):
                # Xlist.append(x**n)
                Xlist.append(func(x, n))
            Xstack = np.column_stack(tuple(Xlist))
            # print(Xstack)
            Xstack = sm.add_constant(Xstack)
            # H=np.linalg.pinv(np.matmul(np.transpose(Xstack),Xstack))
            # U=np.matmul(np.transpose(Xstack),np.transpose(np.array(y)))
            # b=np.transpose(np.matmul(H,np.transpose(U)))
            # print(b)
            # print()
            if sig_y == None:
                ogmodel = sm.OLS(y, Xstack)
            else:
                ogmodel = sm.WLS(y, Xstack, weights=1 / np.array(sig_y) ** 2)
            model = ogmodel.fit()
            # print(model.summary())
            return model.params, model.bse, model.rsquared, model.predict(), ogmodel  # ,b

        def power(coeflist, value, base=10):
            return base ** calc.poly.result(coeflist, value)

        def error_power(coeflist, value, error, base=10):
            return abs(calc.poly.error(coeflist, value, error) * np.log(base) * calc.poly.power(coeflist, value, base))

        def t_eff_err(coeflist, value, error, temp, coeferror=[], base=10):
            if len(coeferror) == 0:
                return temp * np.log(base) * calc.poly.error(coeflist, value, error)
            else:
                return temp * np.log(base) * np.sqrt(calc.poly.error(coeflist, value, error) ** 2 +
                                                     sum(np.array(coeferror) * (
                                                                 (value ** np.arange(len(coeflist))) ** 2)))
            # return np.log(base) * calc.poly.power(coeflist, value, base) * \
            #        np.sqrt(calc.poly.error(coeflist, value, error) ** 2 + \
            #                sum(np.array(coeferror) * value ** np.arange(len(coeflist)) ** 2))

    class error:
        def per_diff(x1, x2):
            return 100 * (abs(x1 - x2) / np.mean([x1, x2]))

        def SS_residuals(obslist, modellist):
            SS_res = 0
            for n in range(len(obslist)):
                SS_res += (obslist[n] - modellist[n]) ** 2
            return SS_res

        def sig_sum(errorlist):
            SS = 0
            for n in range(len(errorlist)):
                SS += errorlist[n] ** 2
            return np.sqrt(SS)

        def SS_total(obslist):
            mean = np.mean(obslist)
            SS_tot = 0
            for n in range(len(obslist)):
                SS_tot += (obslist[n] - mean) ** 2
            return SS_tot

        def CoD(obslist, modellist):
            """
            Calculates the coefficient of determination (R^2) using the
            actual and modelled data. Lists must be the same length.

            ----------

            R^2 = 1 - RSS/ESS

            RSS: Sum of the square of the residuals.

            ESS: Explained sum of squares. Variance in the observed data
            times N-1
            """
            return 1 - calc.error.SS_residuals(obslist, modellist) / calc.error.SS_total(obslist)

        def weighted_average(valuelist, errorlist):
            """
            Returns the weighted average of a list of values, with
            their corresponding errors. Also returns the uncertainty in the
            weighted average, which is the reciprocal of the square
            root of the sum of the reciprocal squared errors.
            """
            M = sum(1 / np.array(errorlist) ** 2)
            w_average = 0
            for n in range(len(errorlist)):
                w_average += valuelist[n] / errorlist[n] ** 2
            w_average /= M
            ave_error = 1 / np.sqrt(M)
            return w_average, ave_error, M

        def avg(errorlist):
            error2list = []
            for error in errorlist:
                error2list.append(error ** 2)
            return np.sqrt(sum(error2list)) * (1 / len(errorlist))

        def red_X2(obslist, modellist, obserror):
            """
            Calculates the reduced chi squared.
            Requires observed values, expected values (model),
            and ideally the observed error. However, if the errors are not known,
            simply make all values in obserror 1. This will then return the residual sum
            of squares, which isn't really chi squared
            """
            X2v0 = 0
            X2v = X2v0
            for n in range(len(obslist)):
                # print(X2v)
                X2v += ((obslist[n] - modellist[n]) / obserror[n]) ** 2
            return X2v

        def truncnorm(size, lower=-3.0, upper=3.0, mean=0.0, sigma=1.0):
            """
            Returns a list (of specified size) of Gaussian deviates
            from a capped standard deviation range.

            Defaults to the traditional 3 sigma rule.
            """
            import scipy.stats as sci
            return sci.truncnorm.rvs(lower, upper, loc=mean, scale=sigma, size=size)

    # -------------------
    class astro:
        class convert:
            def HJD_phase(HJDlist, period, Epoch, Pdot=0):
                """
                Converts a list of Heliocentric julian dates, and
                returns a corresponding phase list. Requires period and Epoch.
                """
                # ob_phaselist=[]
                # for time in HJDlist:
                # phaseT=(((time-Epoch)/period)-np.floor((time-Epoch)/period))
                # ob_phaselist.append(phaseT)
                # return ob_phaselist
                daydiff = np.array(HJDlist) - Epoch
                return (daydiff / (period + Pdot * daydiff)) - np.floor(daydiff / (period + Pdot * daydiff))

            def JD_to_Greg(JD):
                months = ['none', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                          'October', 'November', 'December', ]
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
                # print(months[M],str(D)+',',Y)
                return str(Y) + ' ' + str(M) + ' ' + str(D)

            class magToflux:
                def flux(mag):
                    return 10 ** (-0.4 * mag)

                def error(mag, magerr):
                    return 0.4 * magerr * np.log(10) * 10 ** (-0.4 * mag)

            class fluxTomag:
                def mag(flux):
                    return -2.5 * np.log10(flux)

                def error(flux, fluxerr):
                    return (2.5 * fluxerr) / (flux * np.log(10))


# ======================================
class binning:
    def makebin(phase, bins, phasefluxlist):
        """
        Component of minibinner().
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
        Component of minibinner().
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
        if norm_factor == 'ob':
            normf = max(ob_fluxlist)
        else:
            normf = max(binnedfluxlist)
        norm_binned = []
        for binnedflux in binnedfluxlist:
            norm_binned.append(binnedflux / normf)
        norm_ob = []
        norm_err = []
        for n in range(len(ob_fluxlist)):
            norm_ob.append(ob_fluxlist[n] / normf)
            norm_err.append(ob_fluxerr[n] / normf)
        return norm_binned, norm_ob, norm_err

    def minibinner(phaselist, fluxlist, bins):
        """
        Returns binned lists given a list of phases, and a list of fluxes,
        and specified number of bins (recommend 40).
        Lite version of masterbinner; doesn't support weighting.
        Used in calculating Monte Carlo Fourier transforms.
        """
        obs = len(phaselist)
        phaseflux = []
        for n in range(obs):
            phaseflux.append(phaselist[n])
            phaseflux.append(fluxlist[n])
        binned = binning.binall(bins, phaseflux)
        binnedphaselist = binned[0]
        binnedfluxlist = binned[1]
        normed = binning.norm_flux(binnedfluxlist, fluxlist, fluxlist)
        n_binnedfluxlist = normed[0]
        n_ob_fluxlist = normed[1]
        return binnedphaselist, binnedfluxlist, n_binnedfluxlist, n_ob_fluxlist

    def masterbinner(HJD, mag, magerr, Epoch, period, bins=40, weighted=True, norm_factor='alt', centered=True, pdot=0):
        ob_phaselist = calc.astro.convert.HJD_phase(HJD, period, Epoch, Pdot=pdot)
        ob_maglist = mag
        ob_magerr = magerr
        observations = len(ob_maglist)

        # ob_fluxlist=[] ; ob_fluxerr=[]
        # for n in range(observations):
        # ob_fluxlist.append(calc.astro.convert.magToflux.flux(ob_maglist[n]))
        # ob_fluxerr.append(calc.astro.convert.magToflux.error(ob_maglist[n],ob_magerr[n]))
        ob_fluxlist = list(calc.astro.convert.magToflux.flux(np.array(ob_maglist)))
        ob_fluxerr = list(calc.astro.convert.magToflux.error(np.array(ob_maglist), np.array(ob_magerr)))

        halfbin = 0.5 / bins
        dphase = 1 / bins

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
            # print(round(phase,10),fluxes_in_bin)
            return phases_in_bin, fluxes_in_bin, errors_in_bin, index_in_bin

        binnedfluxlist = []
        binnederrorlist = []
        avgphaselist = []
        master_phases_in_bin = []
        master_fluxes_in_bin = []
        master_errors_in_bin = []
        master_index_in_bin = []
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

        # binnedmaglist=[] ; binnedmagerr=[]
        # for n in range(bins):
        # binnedmaglist.append(calc.astro.convert.fluxTomag.mag(binnedfluxlist[n]))
        # binnedmagerr.append(calc.astro.convert.fluxTomag.error(binnedfluxlist[n],binnederrorlist[n]))

        binnedmaglist = list(calc.astro.convert.fluxTomag.mag(np.array(binnedfluxlist)))
        binnedmagerr = list(calc.astro.convert.fluxTomag.error(np.array(binnedfluxlist), np.array(binnederrorlist)))

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

        def normlist(valuelist, norm_f):
            return np.array(valuelist) / norm_f

        n_binnedfluxlist = normlist(binnedfluxlist, norm_f)
        n_ob_fluxlist = normlist(ob_fluxlist, norm_f)
        n_ob_fluxerr = normlist(ob_fluxerr, norm_f)
        n_binnederrorlist = normlist(binnederrorlist, norm_f)
        for n in range(len(master_fluxes_in_bin)):
            master_fluxes_in_bin[n] /= norm_f
            master_errors_in_bin[n] /= norm_f

        # ----------------------0----------------1-------------2--------------3------------
        master_time = [binnedphaselist, ob_phaselist, avgphaselist, HJD]  # 0
        master_norm = [n_binnedfluxlist, n_ob_fluxlist, n_ob_fluxerr, n_binnederrorlist]  # 1
        master_ob_flux = [binnedfluxlist, ob_fluxlist, ob_fluxerr, binnederrorlist]  # 2
        master_ob_mag = [binnedmaglist, ob_maglist, ob_magerr, binnedmagerr]  # 3
        master_stuff_in_bin = [master_phases_in_bin, master_fluxes_in_bin, master_errors_in_bin,
                               master_index_in_bin]  # 5

        # ------------0------------1------------2---------------3---------4-------------5--------
        return master_time, master_norm, master_ob_flux, master_ob_mag, norm_f, master_stuff_in_bin

    def masterbinner_FF(fileName, Epoch, period, bins,
                        HJDcol=0, magcol=1, magerrcol=2,
                        weighted=True, norm_factor='bin', sep=None,
                        header=None, file_type='text', centered=True, pdot=0):
        file = io.importFile_pd(fileName, delimit=sep, header=header, file_type=file_type)
        HJD = file[HJDcol]
        mag = file[magcol]
        magerr = file[magerrcol]
        MB = binning.masterbinner(HJD, mag, magerr, Epoch, period, bins,
                                  weighted=weighted, norm_factor=norm_factor, centered=centered, pdot=pdot)
        return MB

    def minipolybinner(c_master_phases, c_master_fluxes, nc_master_phases, nc_master_fluxes,
                       section_order, section_res=None):
        # c_master_phases=c_MB[5][0]   ; c_master_fluxes=c_MB[5][1]
        # nc_master_phases=nc_MB[5][0] ; nc_master_fluxes=nc_MB[5][1]
        # ob_phaselist=c_MB[0][1] ; ob_fluxlist=c_MB[1][1] ; ob_fluxerr=c_MB[1][2]
        sections = len(c_master_phases)
        if section_res == None:
            section_res = int(128 / sections)

        section_polyphase = []
        section_polyflux = []
        lastphase = []
        lastflux = []

        halfsec = 0.5 / sections
        dphase = 1 / sections
        bound1 = int(section_res * 0.25)
        bound2 = int(section_res * 0.75)
        for section in range(sections):
            if section == 0:
                for n in range(len(c_master_phases[0])):
                    if 1 - halfsec < c_master_phases[0][n] < 1:
                        c_master_phases[0][n] -= 1
                c_coef = calc.poly.regr_polyfit(c_master_phases[0], c_master_fluxes[0], section_order)[0]
                # c_coef=calc.poly.polyfit(c_master_phases[0],c_master_fluxes[0],section_order)
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
                # nc_coef=calc.poly.polyfit(nc_master_phases[0],nc_master_fluxes[0],section_order)
                nc_polylist = calc.poly.polylist(nc_coef, 0, dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])
            else:
                c_coef = calc.poly.regr_polyfit(c_master_phases[section], c_master_fluxes[section], section_order)[0]
                # c_coef=calc.poly.polyfit(c_master_phases[section],c_master_fluxes[section],section_order)
                c_polylist = calc.poly.polylist(c_coef, section / sections - halfsec, section / sections + halfsec,
                                                section_res)
                section_polyphase += list(c_polylist[0][bound1:bound2:])
                section_polyflux += list(c_polylist[1][bound1:bound2:])
                nc_coef = calc.poly.regr_polyfit(nc_master_phases[section], nc_master_fluxes[section], section_order)[0]
                # nc_coef=calc.poly.polyfit(nc_master_phases[section],nc_master_fluxes[section],section_order)
                nc_polylist = calc.poly.polylist(nc_coef, section / sections, section / sections + dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])
        section_polyphase += list(lastphase)
        section_polyflux += list(lastflux)

        if section_polyphase[0] != 0.0:
            print('WARNING: Unfit for FT, first phase value not zero.')

        return section_polyphase, section_polyflux

    def polybinner(input_file, Epoch, period, sections=4, norm_factor='alt',
                   section_order=8, FT_order=12, section_res=None, HJD_mag_magerr=[],
                   mag_coef=False, pdot=0):
        if section_res == None:
            section_res = int(128 / sections)
        # section_res=84
        # if is_power_of_two(sections) == False:
        # sections=4
        # print(str(sections)+""" is not a power of two.\nSection counts that aren't powers of two are strongly discouraged.""")
        # if section_res%4 != 0:
        # section_res=32
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

        c_master_phases = c_MB[5][0]
        c_master_fluxes = c_MB[5][1]
        nc_master_phases = nc_MB[5][0]
        nc_master_fluxes = nc_MB[5][1]

        ob_phaselist = c_MB[0][1]
        ob_fluxlist = c_MB[1][1]

        section_polyphase = []
        section_polyflux = []
        lastphase = []
        lastflux = []
        halfsec = 0.5 / sections
        dphase = 1 / sections
        bound1 = int(section_res * 0.25)
        bound2 = int(section_res * 0.75)
        for section in range(sections):
            if section == 0:
                for n in range(len(c_master_phases[0])):
                    if 1 - halfsec < c_master_phases[0][n] < 1:
                        c_master_phases[0][n] -= 1
                c_coef = calc.poly.regr_polyfit(c_master_phases[0], c_master_fluxes[0], section_order)[0]
                # c_coef=np.polyfit(c_master_phases[0],c_master_fluxes[0],section_order)[::-1]

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
                # nc_coef=np.polyfit(nc_master_phases[0],nc_master_fluxes[0],section_order)[::-1]

                nc_polylist = calc.poly.polylist(nc_coef, 0, dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])
            else:
                c_coef = calc.poly.regr_polyfit(c_master_phases[section], c_master_fluxes[section], section_order)[0]
                # c_coef=np.polyfit(c_master_phases[section],c_master_fluxes[section],section_order)[::-1]

                c_polylist = calc.poly.polylist(c_coef, section / sections - halfsec, section / sections + halfsec,
                                                section_res)
                section_polyphase += list(c_polylist[0][bound1:bound2:])
                section_polyflux += list(c_polylist[1][bound1:bound2:])

                nc_coef = calc.poly.regr_polyfit(nc_master_phases[section], nc_master_fluxes[section], section_order)[0]
                # nc_coef=np.polyfit(nc_master_phases[section],nc_master_fluxes[section],section_order)[::-1]

                nc_polylist = calc.poly.polylist(nc_coef, section / sections, section / sections + dphase, section_res)
                section_polyphase += list(nc_polylist[0][bound1:bound2:])
                section_polyflux += list(nc_polylist[1][bound1:bound2:])
        section_polyphase += list(lastphase)
        section_polyflux += list(lastflux)
        # print('phase[0] =',section_polyphase[0])
        if mag_coef == True:
            a, b = FT.coefficients(-2.5 * np.log10(np.array(section_polyflux) * c_MB[4]))[1:3]
        else:
            a, b = FT.coefficients(section_polyflux)[1:3]
        # FTcoef=FT.coefficients(section_polyflux)
        # a=FTcoef[1]
        # b=FTcoef[2]
        # print(len(section_polyphase))
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
        C = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) + 1

        def spec_a(mod_a):
            a[term] = mod_a
            F = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) - C
            # print(F)
            return F

        upper = abs(calc.Newton(spec_a, aterm + dx0, 1e-5, dx=1e-10) - aterm)
        lower = abs(calc.Newton(spec_a, aterm - dx0, 1e-5, dx=1e-10) - aterm)
        a[term] = aterm
        return (upper + lower) / 2, upper / lower

    def b_sig_fast(a, b, term, bterm, ob_phase, ob_flux, ob_fluxerr, order, dx0=1):
        C = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) + 1

        def spec_b(mod_b):
            b[term] = mod_b
            F = calc.error.red_X2(ob_flux, FT.synth(a, b, ob_phase, order), ob_fluxerr) - C
            # print(F)
            return F

        upper = abs(calc.Newton(spec_b, bterm + dx0, 1e-5, dx=1e-10) - bterm)
        lower = abs(calc.Newton(spec_b, bterm - dx0, 1e-5, dx=1e-10, central_diff=True) - bterm)
        b[term] = bterm
        return (upper + lower) / 2, upper / lower

    def sumatphase(phase, order, a, b):
        """
        Calculates the amplitude (value) of the FT at a given phase.
        Needs a and b coefficient lists.
        """
        # atphaselist=0 ; k0=0 ; k=k0
        # while(k <= order):
        # atphaselist+=(a[k]*np.cos(2*np.pi*k*phase)+b[k]*np.sin(2*np.pi*k*phase))
        # k+=1
        # return atphaselist
        orders = np.arange(order + 1)
        return sum(
            a[:order + 1:] * np.cos(2 * np.pi * phase * orders) + b[:order + 1:] * np.sin(2 * np.pi * phase * orders))

    def deriv_sumatphase(phase, order, a, b):
        deriv_atphase = 0
        for k in range(1, order + 1):
            deriv_atphase += (-b[k] * np.sin(2 * np.pi * k * phase) - a[k] * np.cos(
                2 * np.pi * k * phase)) * 4 * np.pi ** 2 * k ** 2
            # deriv_atphase+=(b[k]*np.sin(2*np.pi*k*phase)+a[k]*np.cos(2*np.pi*k*phase))*16*np.pi**4*k**4
            # deriv_atphase*=2*np.pi*k
        return deriv_atphase

    def unc_sumatphase(phase, order, a_unc, b_unc):
        """
        Calculates the uncertainty of the FT at a given phase,
        given a list of a and b uncertainties.
        """
        unc_I = a_unc[0] ** 2
        for k in range(1, order + 1):
            unc_I += (a_unc[k] * np.cos(2 * np.pi * k * phase)) ** 2 + (b_unc[k] * np.sin(2 * np.pi * k * phase)) ** 2
        return np.sqrt(unc_I)

    def FT_plotlist(a, b, order, resolution):
        """
        Generates the results of a FT given the coefficients.
        Resolution: how many points you want.
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

        This is really only for the sim_FT_curve program, and not a standalone fucntion.
        """
        obs = len(ob_fluxerr)
        devlist = calc.error.truncnorm(obs, lower=lower, upper=upper)
        sim_ob_flux = []
        for n in range(obs):
            sim_ob_flux.append(FT[n] + devlist[n] * ob_fluxerr[n])
        return sim_ob_flux

    def synth(a, b, ob_phaselist, order):
        """
        Similar to FT_plotlist, but instead of equal spacing throughout
        the curve, calculates the flux at the observational phases,
        resulting in a synthetic light curve.
        """
        synthlist = []
        for phase in ob_phaselist:
            synthlist.append(FT.sumatphase(phase, order, a, b))
        return synthlist
        # return FT.sumatphase(np.array(ob_phaselist),order,a,b)

    def int_sumatphase(a, b, phase, order):
        """
        Integral of the FT at a phase
        """
        # int_sum=0
        # for k in range(1,order+1):
        # int_sum+=((1/k)*(a[k]*np.sin(2*np.pi*phase*k)-b[k]*np.cos(2*np.pi*phase*k)))
        # int_sum*=(0.5/np.pi)
        # int_sum+=a[0]*phase
        # int_sum=0
        nlist = np.arange(order + 1)[1::]
        # print(nlist)
        return a[0] * phase + (0.5 / np.pi) * sum((a[1:order + 1:] * np.sin(2 * np.pi * nlist * phase) - b[
                                                                                                         1:order + 1:] * np.cos(
            2 * np.pi * nlist * phase)) / nlist)
        # return int_sum

    def integral(a, b, order, lowerphase, upperphase):
        """
        Definite integral of a Fourier transform.
        """
        uppersum = FT.int_sumatphase(a, b, upperphase, order)
        lowersum = FT.int_sumatphase(a, b, lowerphase, order)
        return uppersum - lowersum

    def int_unc_atphase(phase, a_unc, b_unc):
        """
        Returns uncertainty for a given bound, not both.
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
        nlist = np.arange(order + 1)[1::]
        A = 0.5 * sum(a[nlist])
        B = sum(b[nlist[::2]] / nlist[::2]) / np.pi
        return 1 - (2 / (1 + A / B))

    def OER_FT_error(a, b, a_unc, b_unc, order):
        """
        Calculates both the OER and OER error.
        Requires the a and b coefficients, but also their errors.
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

    # ==================================
    def LCA_FT(a, b, order, resolution):
        """
        Calculates the Light Curve Asymmetry (LCA), given
        the a and b coefficients (and order of FT).
        Must specify the resolution because the LCA integral
        is calculated using a Simpson integration (numerical).
        Anything over 200 is fine, but > 1000 is ideal for one time calculations.
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
        Calculates the error in the LCA integrand, which is then used
        to find the error in the LCA. Need a and b coefficients and uncertainties.
        """
        I = FT.sumatphase(phase, order, a, b)
        J = 2 * FT.sumatphase(phase, order, np.zeros(order + 1), b)
        K = J / I
        L = K ** 2

        dL_da0 = -2 * L / I
        dL_dak = []
        dL_dbk = []
        # dL_da0=-2*J**2*I ; dL_dak=[] ; dL_dbk=[]
        # J2I=J**2*I
        for k in range(1, order + 1):
            # dL_dak.append(dL_da0*np.cos(2*np.pi*k*phase))
            dL_dak.append(dL_da0 * np.cos(2 * np.pi * k * phase))
            dL_dbk.append(2 * np.sin(2 * np.pi * k * phase) * (2 * J / I ** 2 - J ** 2 / I ** 3))
        L_err = (dL_da0 * a_unc[0]) ** 2
        for n in range(len(dL_dak)):
            L_err += (dL_dak[n] * a_unc[n + 1]) ** 2 + (dL_dbk[n] * b_unc[n + 1]) ** 2
        L_err = np.sqrt(L_err)
        return L, L_err

    def LCA_FT_error(a, b, a_unc, b_unc, order, resolution):
        """
        Calculates the uncertainty in the FT LCA.
        Monte Carlo deviations should be considered
        for the total error.
        """
        import scipy
        Phi = np.linspace(0, 0.5, resolution)
        Llist = []
        Lerrlist = []
        for phase in Phi:
            L_ap = OConnell.L_error(phase, order, a, b, a_unc, b_unc)
            Llist.append(L_ap[0])
            Lerrlist.append(L_ap[1])
        int_L = scipy.integrate.simps(Llist, Phi)
        int_L_error = scipy.integrate.simps(Lerrlist, Phi)
        # print('L_err =',int_L_error,'L =',int_L)
        LCA = OConnell.LCA_FT(a, b, order, resolution)
        LCA_error = 0.5 * LCA * (int_L_error / int_L)
        return LCA, LCA_error

    def LCA_FT_error2(a, b, a_unc, b_unc, order, resolution):
        a, a_unc = np.array(a), np.array(a_unc)
        b, b_unc = np.array(b), np.array(b_unc)

        def L_error2(phase):
            nlist = np.arange(order + 1)
            dIphi = 2 * sum(b[nlist] * np.sin(2 * np.pi * nlist * phase))
            I = FT.sumatphase(phase, order, a, b)
            # if (dIphi/I)**2 == 0.0:
            # L=1e-16
            # else:
            L = (dIphi / I) ** 2
            print(L)
            # radicand=0
            # radicand+=sum((a_unc*np.cos(2*np.pi*phase*nlist))**2)
            # radicand+=(2/np.sqrt(L)-1)**2*sum((b_unc*np.sin(2*np.pi*phase*nlist))**2)
            # return L, 2*L/I*np.sqrt(radicand)
            return L, 2 / I * (2 * L ** 0.5 - L) * np.sqrt(sum((b_unc * np.sin(2 * np.pi * phase * nlist)) ** 2))
            # return L, 4/I*np.sqrt(L*sum((b_unc*np.sin(2*np.pi*phase*nlist))**2))

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
        Directly calculates the difference in flux at the two
        quadratures (0.25, 0.75), given the a and b coefficients.
        One of the direct measures of the O'Connell effect, along
        with dIave and 2b1.
        """
        Ip = FT.sumatphase(0.25, order, a, b)
        Is = FT.sumatphase(0.75, order, a, b)
        dI = Ip - Is
        return dI, Ip, Is

    def Delta_I_error(a, b, a_unc, b_unc, order):
        """
        Calculates the FT error in dI. Requires the coefficients
        and their uncertainties.
        """
        sig_Ip = FT.unc_sumatphase(0.25, order, a_unc, b_unc)
        sig_Is = FT.unc_sumatphase(0.75, order, a_unc, b_unc)
        sig_dI = np.sqrt(sig_Ip ** 2 + sig_Is ** 2)
        dI = OConnell.Delta_I(a, b, order)[0]
        return dI, sig_dI

    def Delta_I_fixed(b, order):
        mlist = np.arange(int(np.ceil(order / 2)))
        return 2 * sum((-1) ** mlist * b[2 * mlist + 1])

    def Delta_I_error_fixed(b_unc, order):
        # return 2*np.sqrt(sum(b_unc[np.arange(order+1)[1::2]]**2))
        return 2 * np.sqrt(sum(np.array(b_unc)[1:order + 1:2] ** 2))

    def dI_at_phase(b, order, phase):
        nlist = np.arange(order + 1)
        return 2 * sum(b[nlist] * np.sin(2 * np.pi * nlist * phase))

    def dI_at_phase_error(b_unc, order, phase):
        nlist = np.arange(order + 1)
        return 2 * np.sqrt(sum((b_unc[nlist] * np.sin(2 * np.pi * nlist * phase)) ** 2))

    def Delta_I_mean_obs(ob_phaselist, ob_fluxlist, ob_fluxerr, phase_range=0.05, weighted=False):
        """
        Calculates the difference of flux at quadrature from observational fluxes
        and the error the uncertainty of this measure.

        Averages the fluxes in a specified range from quadrature, and subtracts the two
        values. Average can be weighted, although that is probably overkill and/or
        undesireable.
        """
        Iplist = []
        Iperrors = []
        Islist = []
        Iserrors = []
        for n in range(len(ob_phaselist)):
            if 0.25 - phase_range < ob_phaselist[n] < 0.25 + phase_range:
                Iplist.append(ob_fluxlist[n])
                Iperrors.append(ob_fluxerr[n])
            if 0.75 - phase_range < ob_phaselist[n] < 0.75 + phase_range:
                Islist.append(ob_fluxlist[n])
                Iserrors.append(ob_fluxerr[n])

        if weighted == True:
            Ipmean = calc.error.weighted_average(Iplist, Iperrors)
            Ismean = calc.error.weighted_average(Islist, Iserrors)
            dI_mean_obs = Ipmean[0] - Ismean[0]
            dI_mean_error = np.sqrt(Ipmean[1] ** 2 + Ismean[1] ** 2)
        else:
            dI_mean_obs = np.mean(Iplist) - np.mean(Islist)
            dI_mean_error = np.sqrt(calc.error.avg(Iperrors) ** 2 + calc.error.avg(Iserrors) ** 2)
        return dI_mean_obs, dI_mean_error

    def Delta_I_mean_obs_noerror(ob_phaselist, ob_fluxlist, phase_range=0.05):
        """
        Same as Delta_I_mean_obs but just for simulations (errors not used).
        """
        Iplist = []
        Islist = []
        for n in range(len(ob_phaselist)):
            if 0.25 - phase_range < ob_phaselist[n] < 0.25 + phase_range:
                Iplist.append(ob_fluxlist[n])
            if 0.75 - phase_range < ob_phaselist[n] < 0.75 + phase_range:
                Islist.append(ob_fluxlist[n])
        dI_mean_obs = np.mean(Iplist) - np.mean(Islist)
        return dI_mean_obs


# weighted averages
def M(errorlist):  # sum of 1/errors
    M0 = 0
    M = M0
    for error in errorlist:
        M += 1 / error ** 2
    return M


#
def wfactor(errorlist, n, M):
    return 1 / (errorlist[n] ** 2 * M)


# ======================================
class Flower:  # stuff from Flower 1996, Torres 2010
    class T:
        # c=[c0,c1,c2,c3,c4,c5,c6,c7]
        c = [3.97914510671409, -0.654992268598245, 1.74069004238509, -4.60881515405716, 6.79259977994447,
             -5.39690989132252, 2.19297037652249, -0.359495739295671]

        def Teff(BV, error):
            # return 10**((Flower.T.c[0]*BV**0)+(Flower.T.c[1]*BV**1)+(Flower.T.c[2]*BV**2)+(Flower.T.c[3]*BV**3)+(Flower.T.c[4]*BV**4)+(Flower.T.c[5]*BV**5)+(Flower.T.c[6]*BV**6)+(Flower.T.c[7]*BV**7))
            temp = calc.poly.power(Flower.T.c, BV, 10)
            err = calc.poly.t_eff_err(Flower.T.c, BV, error, temp)
            return temp, err


# ======================================
class Harmanec:
    class mass:
        c = [-121.6782, 88.057, -21.46965, 1.771141]

        def M1(BV):
            # M1=10**((Harmanec.mass.c0*np.log10(Flower.T.Teff(BV))**0)+(Harmanec.mass.c1*np.log10(Flower.T.Teff(BV))**1)+(Harmanec.mass.c2*np.log10(Flower.T.Teff(BV))**2)+(Harmanec.mass.c3*np.log10(Flower.T.Teff(BV))**3))
            M1 = 10 ** (calc.poly.result(Harmanec.mass.c, np.log10(Flower.T.Teff(BV))))
            return M1


# ======================================
class Red:  # interstellar reddening
    J_K = 0.17084
    J_H = 0.10554
    V_R = 0.58 / 3.1

    def colorEx(filter1, filter2, Av):
        excess = str(filter1) + '_' + str(filter2)
        if excess == 'J_K':
            return Av * Red.J_K
        elif excess == 'J_H':
            return Av * Red.J_H
        elif excess == 'V_R':
            return Av * Red.V_R


# ======================================
class plot:
    def amp(valuelist):
        # test documentation
        return max(valuelist) - min(valuelist)

    def aliasing2(phaselist, maglist, errorlist, alias=0.6):
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
        if fig == None:
            fig = plt.figure(1, figsize=figsize, dpi=dpi)
        axs = fig.subplots(len(height_ratios), sharex=sharex, sharey=sharey,
                           gridspec_kw={'hspace': hspace, 'height_ratios': height_ratios})
        return axs, fig

    def sm_format(ax, X=None, x=None, Y=None, y=None, Xsize=7, xsize=3.5, tickwidth=1,
                  xtop=True, xbottom=True, yright=True, yleft=True, numbersize=12, autoticks=True,
                  topspine=True, bottomspine=True, rightspine=True, leftspine=True, xformatter=True,
                  xdirection='in', ydirection='in', spines=True):
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
            # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('$%g$'))
        # for label in ax.get_xticklabels():
        # label.set_fontproperties('DejaVu Sans')
        # for label in ax.get_yticklabels():
        # label.set_fontproperties('DejaVu Sans')
        return 'DONE'


# ======================================
class Roche:
    def Kopal_cyl(rho, phi, z, q):
        return 1 / np.sqrt(rho ** 2 + z ** 2) + q / (
            np.sqrt(1 + rho ** 2 + z ** 2 - 2 * rho * np.cos(phi))) - q * rho * np.cos(phi) + 0.5 * (1 + q) * rho ** 2

    def gen_Kopal_cyl(rho, phi, z, q,
                      xcm=None, ycm=0, zcm=0,
                      potcap=None):
        if xcm == None:
            xcm = q / (1 + q)
        A1 = -q / (1 + q)
        A2 = 1 / (1 + q)
        B1 = xcm ** 2 + ycm ** 2 + zcm ** 2 + 2 * xcm * A1 + A1 ** 2
        # print(B1)
        B2 = xcm ** 2 + ycm ** 2 + zcm ** 2 + 2 * xcm * A2 + A2 ** 2
        X = rho * np.cos(phi)
        Y = rho * np.sin(phi)
        s1 = np.sqrt(rho ** 2 + z ** 2 - 2 * (X * (xcm + A1) + Y * ycm + z * zcm) + B1)
        s2 = np.sqrt(rho ** 2 + z ** 2 - 2 * (X * (xcm + A2) + Y * ycm + z * zcm) + B2)
        rw2 = rho ** 2 - 2 * (xcm * X + ycm * Y) + xcm ** 2 + ycm ** 2
        potent = 1 / s1 + q / s2 + 0.5 * (1 + q) * rw2 - 0.5 * q ** 2 / (1 + q)
        return potent

    def Lagrange_123(q, e=1e-8):
        L1 = lambda x: q / x ** 2 - x * (1 + q) - 1 / (1 - x) ** 2 + 1
        L2 = lambda x: q / x ** 2 - x * (1 + q) + 1 / (1 + x) ** 2 - 1
        L3 = lambda x: 1 / (q * x ** 2) - x * (1 + 1 / q) + 1 / (1 + x) ** 2 - 1
        xL1 = calc.Newton(L1, 0.5, e=e)
        xL2 = calc.Newton(L2, 0.5, e=e)
        xL3 = calc.Newton(L3, 0.5, e=e)
        return xL1, xL2, xL3

    def Kopal_zero(rho, phi, z, q, Kopal, body='M1'):
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
        return Roche.gen_Kopal_cyl(rho, phi, z, q, xcm=xcm, ycm=ycm, zcm=zcm) - Kopal


class Pecaut:  # V-R effective temperature fit from Pecaut and Mamajek 2013 https://arxiv.org/pdf/1307.2657.pdf
    class T:
        # c=[c0,c1,c2,c3]
        c1 = [3.98007, -1.2742, 32.41373, 98.06855]  # -0.115 < V-R < 0.019
        c2 = [3.9764, -0.80319, 0.64832, -0.2651]  # 0.019 < V-R < 1.079

        c1_err = [0.00499, 0.25431, 6.60765, 41.6003]
        c2_err = [0.00179, 0.01409, 0.03136, 0.0197]

        def Teff(VR, error):
            if -0.115 < VR <= 0.019: # B1V to A1V
                temp = calc.poly.power(Pecaut.T.c1, VR, 10)
                err = calc.poly.t_eff_err(Pecaut.T.c1, VR, error, temp, coeferror=Pecaut.T.c1_err)
                return temp, err
            elif 0.019 < VR < 1.079: # A1V to M3V
                temp = calc.poly.power(Pecaut.T.c2, VR, 10)
                err = calc.poly.t_eff_err(Pecaut.T.c2, VR, error, temp, coeferror=Pecaut.T.c2_err)
                return temp, err
            else:
                return 0, 0
