# -*- coding: utf-8 -*-
"""
Created:  07/03/2021
Original Author: Alec Neal

Last Edits Done By: Kyle Koeller
Last Edited: 03/03/2023
"""

# import vseq  # testing purposes
from . import vseq_updated as vseq
from os import environ, path
import matplotlib.pyplot as plt
import numpy as np
import scipy
# from tkinter import *
from collections import Counter
import statistics
from tqdm import tqdm
environ['MPLBACKEND'] = 'TkAgg'


day = 1  # starting day
lb = 0.69  # starting left boundary
rb = 0.775  # starting right boundary
last_rb = rb  # defines the last right boundary used for global use
last_lb = lb  # defines the last left boundary used for global use
order = 5  # order for the LSQ fitting (not shown in the figure)
resolution = 200  # resolution for the fitting
npairs = 20  # number of data point pairs
fontsize = 9  # font size of text
import_files = []


def percent_to_xy(xy, xlist, ylist, x_rev=False, y_rev=False):
    xrange = max(xlist) - min(xlist)
    yrange = max(ylist) - min(ylist)
    if x_rev:
        x0 = max(xlist)
    else:
        x0 = min(xlist)
    if y_rev:
        y0 = max(ylist)
    else:
        y0 = min(ylist)
    return x0 + xrange * xy[0], y0 + yrange * xy[1]


def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def best_root(coef, lb, rb, filter_files):
    """
    np.roots is a shitshow (to put it lightly),
    so we have to do this nonsense to get the most
    sensible root.
    """
    dcoef = []
    for n in range(1, len(coef)):
        dcoef.append(n * coef[n])
    allroots = np.unique(np.real(np.roots(dcoef[::-1])))
    # print(allroots)
    goodroots = []
    for root in range(len(allroots)):
        if lb < allroots[root] < rb and -1e-2 < vseq.calc.poly.result(dcoef, allroots[root]) < 1e-2:
            goodroots.append(allroots[root])
    root_est = (lb + rb) / 2
    try:
        bestroot = min(list(zip(abs(np.array(goodroots) - root_est), goodroots)))[1]
    except ValueError:
        print("\nAn error ocurred with your last boundary entry. Please try again.\n")
        plot_obs(filter_files, day=day, lb=last_lb, rb=last_rb, order=order, resolution=resolution, npairs=npairs,
                 para_range=None, norm_method="norm")
    return bestroot


def calc_S(fracHJD, flux, guess_min, npairs=None):
    # if interp == None:
    interp = scipy.interpolate.interp1d(fracHJD, flux)
    max_range = min([guess_min - min(fracHJD), max(fracHJD) - guess_min])
    N = 0  # represents the total number of obs. in the range
    for time in fracHJD:
        if guess_min - max_range < time < guess_min + max_range:
            N += 1
    """
    K is the number of "paired" observations, and is a general measure
    of the minimum's fidelity. The observations are not actually computed
    with each other, but rather linear interpolated based on even
    spacing from the guess min. This replicates the ideal case.
    """
    K = int(np.floor(N / 2))  # whole # of pairs (N may be non-even)
    # S=0
    # for k in range(1,K+1):
    # S+=(interp(guess_min-k*step)-interp(guess_min+k*step))**2
    # step=max_range/K
    # klist=np.arange(1,K+1,1)
    if npairs is None:
        npairs = 20

    # thing=100 # better signal with +10 but cheating?
    step = max_range / npairs
    klist = np.arange(1, npairs + 1, 1)
    S = sum((interp(guess_min - klist * step) - interp(guess_min + klist * step)) ** 2)

    return S, K


def KvW(fracHJD, flux, discard=0.1, resolution=100, para_range=None, plot=False,
        need_error=False, ax=None, npairs=20, entire_S=False):
    guess_range = (max(fracHJD) - min(fracHJD))
    guesses = np.linspace(guess_range * discard + min(fracHJD), guess_range * (1 - discard) + min(fracHJD), resolution)
    guesses = np.linspace(sorted(fracHJD)[1], sorted(fracHJD)[-2], resolution)
    Slist = []
    Klist = []
    for guess in guesses:
        # interp=scipy.interpolate.interp1d(fracHJD,flux)
        kwee = calc_S(fracHJD, flux, guess, npairs=npairs)
        Slist.append(kwee[0])
        Klist.append(kwee[1])
    pack = list(zip(Slist, guesses, np.arange(len(guesses)), Klist))
    best_min = min(pack)
    best_index = min(pack)[2]
    Z = min(pack)[3] / 2  # Z=K/2, K for best best (min) S value
    # print(Z)
    # print(best_min)
    T_BF = best_min[1]
    if para_range is None:  # para = parabola
        para_range = int(resolution * 0.05)
    paraHJD = [best_min[1]]
    paraS = [best_min[0]]
    for n in range(1, para_range + 1):
        paraHJD.append(guesses[best_index + n])
        paraHJD.append(guesses[best_index - n])
        paraS.append(Slist[best_index + n])
        paraS.append(Slist[best_index - n])

    np.seterr(divide='ignore')
    parabola = vseq.calc.poly.regr_polyfit(np.array(paraHJD), np.array(paraS), 2)
    coef, err, R2, results = parabola[:4]
    a, b, c = coef[::-1]
    if c - b ** 2 / (4 * a) < 0 or calc_S(fracHJD, flux, -b / (2 * a), npairs=npairs)[0] > paraS[0]:
        with np.errstate(divide='ignore'):
            p2 = vseq.calc.poly.regr_polyfit(paraHJD[:3], paraS[:3], 2)
        # coef, err, R2, results = parabola[:4]
        a2, b2, c2 = p2[0][::-1]
        if c2 - b2 ** 2 / (4 * a2) < 0 or calc_S(fracHJD, flux, -b2 / (2 * a2), npairs=npairs)[0] > paraS[0]:
            T_kw = T_BF
        else:
            T_kw = -b2 / (2 * a2)
    else:
        T_kw = -b / (2 * a)
    S_at_tkw, Z = calc_S(fracHJD, flux, T_kw, npairs=npairs)
    Z /= 2
    if need_error:
        if Z <= 1:
            print('Too few observations!')
        sigt_kw = np.sqrt(S_at_tkw / (a * (Z - 1)))

        print('S(T_BF) =', paraS[0])
        print('S(T_KW) =', calc_S(fracHJD, flux, T_kw, npairs=npairs))
    else:
        sigt_kw = 1

    if ax is not None:
        Srange = max(paraS) - min(paraS)
        HJDrange = max(paraHJD) - min(paraHJD)
        if not entire_S:
            ax.set_ylim(min(paraS) - Srange * 0.07, max(paraS) + Srange * 0.07)
            ax.set_xlim(min(paraHJD) - HJDrange * 0.025, max(paraHJD) + HJDrange * 0.025)

        ax.plot(guesses, np.array(Slist), 'ok', ms=5)
        polyx, polyy = vseq.calc.poly.polylist(coef, min(paraHJD), max(paraHJD), 100)
        ax.plot(polyx, polyy, 'b', lw=1)
        # plt.plot(paraHJD,results,'b',lw=1,label=r'${S}^{\prime}$')
        ax.axvline(T_kw, color='b', label=r'$T_{\rm KW}=' + str(round(T_kw, 6)) + '$', linewidth=1, ls='--')
        # ax.axvline(best_min[1], color='gray', label=r'$T_{\rm BF}=' + str(round(best_min[1], 6)) + '$', linewidth=1,
        #            ls='-.')

        ax.set_ylabel(r'$S$', fontsize=fontsize, usetex=False)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=[0.1, 1.03], frameon=False).set_draggable(True)
        xt, yt = percent_to_xy((0.6, 0.8), paraHJD, ax.get_ylim())
        # strcoef = (coef / a).round(5)

        if npairs is None:
            Kstr = int(2 * Z)
        else:
            Kstr = npairs
        # ax.text(xt, yt, r'$S_{\rm fit}/a=' + 'T^2' + str(strcoef[1]) + 'T+' + str(strcoef[0]) + '$'
        #                                                                                         '\n $R^2=' + str(
        #     round(R2, 5)) + '$'
        #                     r', $K=' + str(Kstr) + '~(' + str(int(Z * 2)) + ')$', fontsize=7)
        ax.text(xt, yt, '$R^2=' + str(round(R2, 5)) + '$, $K=' + str(Kstr) + '~(' + str(int(Z * 2)) + ')$', fontsize=7)

    return T_kw, sigt_kw


def sim_min(filter_HJD, filter_flux, filter_fluxerr, order, sims, filter_files):
    filters = len(filter_HJD)
    all_sims = []
    for sim in range(sims):
        all_sims.append([])
    filter_results = []
    HJD_inbounds = []
    for band in range(filters):
        poly = vseq.calc.poly.regr_polyfit(filter_HJD[band], filter_flux[band], order)
        filter_results.append(poly[3])
        HJD_inbounds += list(filter_HJD[band])
        for sim in range(sims):
            all_sims[sim] += list(vseq.FT.sim_ob_flux(filter_results[band], filter_fluxerr[band], lower=-10, upper=10))
    rootlist = []
    # kweelist = []
    for sim in tqdm(range(sims), position=0):
        rootlist.append(best_root(vseq.calc.poly.regr_polyfit(HJD_inbounds, all_sims[sim], order)[0], min(HJD_inbounds),
                                  max(HJD_inbounds), filter_files))
        # kweelist.append(KvW(HJD_inbounds,all_sims[sim])[0]) # uncomment to sim kwee
    rooterr = statistics.stdev(rootlist)
    # kweeerr=statistics.stdev(kweelist) # uncomment for siming kwee
    kweeerr = 0  # comment to sim kwee
    return rooterr, rootlist, kweeerr


def plot_obs(filter_files, day=0, lb=None, rb=None, order=5, resolution=200,
             npairs=None, para_range=None, discard=0.1, sims=1000, norm_method='norm', entire_S=False, Xtick=0.02):
    if para_range is None:
        para_range = int(resolution * 0.05)

    filters = len(filter_files)
    master_HJD = []
    master_mag = []
    master_magerr = []
    master_dates = []
    master_ob_night = []
    master_unique_dates = []
    master_HJDnight = []
    master_magnight = []
    master_fluxnight = []
    master_magerrnight = []
    master_fluxerrnight = []
    master_span = []

    for band in range(filters):
        HJD_mag_magerr = vseq.io.importFile_pd(filter_files[band])
        HJD = HJD_mag_magerr[0]
        mag = HJD_mag_magerr[1]
        magerr = HJD_mag_magerr[2]
        HJD, mag, magerr = zip(*sorted(list(zip(HJD, mag, magerr))))
        master_HJD.append(HJD)
        master_mag.append(mag)
        master_magerr.append(magerr)
        master_dates.append([])
        for n in range(len(master_HJD[band])):
            master_dates[band].append(vseq.calc.astro.convert.JD_to_Greg(np.floor(master_HJD[band][n]) + 0.75))
            # master_dates[band].append(np.floor(master_HJD[band][n]))
        count_nights = Counter(master_dates[band])
        count_uniques = np.unique(master_dates[band])
        master_unique_dates.append(list(count_uniques))
        master_ob_night.append([])
        master_HJDnight.append([])
        master_magnight.append([])
        master_fluxnight.append([])
        master_magerrnight.append([])
        master_fluxerrnight.append([])
        master_span.append([])
        index_count = 0
        for night in range(len(count_uniques)):
            master_ob_night[band].append(count_nights[count_uniques[night]])
            master_HJDnight[band].append(master_HJD[band][index_count:index_count + master_ob_night[band][night]:])
            master_magnight[band].append(master_mag[band][index_count:index_count + master_ob_night[band][night]:])
            master_magerrnight[band].append(
                master_magerr[band][index_count:index_count + master_ob_night[band][night]:])

            if norm_method == 'subavg':
                fun = lambda x: np.mean(x)
            else:
                fun = lambda x: min(x)
            master_fluxnight[band].append(
                10 ** (-0.4 * (np.array(master_magnight[band][night]) - fun(master_mag[band]))))
            master_fluxerrnight[band].append(
                0.4 * np.log(10) * np.array(master_magerrnight[band][night]) * master_fluxnight[band][night])
            master_span[band].append(
                round(24 * (max(master_HJDnight[band][night]) - min(master_HJDnight[band][night])), 1))
            index_count += master_ob_night[band][night]
    # end loop

    nights = len(count_uniques)
    all_HJD = []
    all_flux = []
    all_fluxerr = []
    for night in range(nights):
        all_HJD.append([])
        all_flux.append([])
        all_fluxerr.append([])
        for band in range(filters):
            all_HJD[night] += list(master_HJDnight[band][night])
            all_flux[night] += list(master_fluxnight[band][night])
            all_fluxerr[night] += list(master_fluxerrnight[band][night])

    intday = int(all_HJD[day][0])
    fracHJD = vseq.calc.frac(np.array(all_HJD[day]))

    global last_lb
    global last_rb

    if lb is not None and rb is not None:
        # Splot = plt.figure(1, figsize=(9,8), dpi=256)
        # MIN = Splot.subplots(len([1, 3]), sharex=False, sharey=False,
        #                   gridspec_kw={'hspace': 0, 'height_ratios': [1, 3]})

        parameters = {'axes.labelsize': fontsize}
        plt.rcParams.update(parameters)

        [Splot, MIN], fig = vseq.plot.multiplot(figsize=(5, 4), height_ratios=[1, 4], sharex=False, sharey=False,
                                                hspace=0)
        MIN.errorbar(fracHJD, all_flux[day], all_fluxerr[day], fmt='ok', ms=3, capsize=3, elinewidth=0.6, ecolor='gray',
                     capthick=0.6)  # ,'ok',ms=3)

        fig.canvas.mpl_connect('key_press_event', press)
        fig.canvas.get_tk_widget().focus_force()

        HJD_inbounds, flux_inbounds, fluxerr_inbounds = [], [], []
        for n in range(len(all_HJD[day])):
            if lb < fracHJD[n] < rb:
                HJD_inbounds.append(fracHJD[n])
                flux_inbounds.append(all_flux[day][n])
                fluxerr_inbounds.append(all_fluxerr[day][n])
        try:
            poly = vseq.calc.poly.regr_polyfit(HJD_inbounds, flux_inbounds, order)
        except ValueError:
            print("\nThe boundary you last entered was incorrect, please try again. \n")
            plot_obs(filter_files, day=day, lb=last_lb, rb=last_rb, order=order, resolution=resolution, npairs=npairs,
                     para_range=None, norm_method="norm")
        try:
            coef, err, R2, results = poly[:4:]
        except UnboundLocalError:
            print("\nEntered in an incorrect boundary value. Please try again.\n")
            plot_obs(filter_files, day=day, lb=last_lb, rb=last_rb, order=order, resolution=resolution, npairs=npairs,
                     para_range=None, norm_method="norm")
        print('R2 =', R2)
        coef_deriv = []

        for n in range(1, len(coef)):
            coef_deriv.append(n * coef[n])
        HJD_band_in = []
        flux_band_in = []
        fluxerr_band_in = []
        for band in range(filters):
            HJD_band_in.append([])
            flux_band_in.append([])
            fluxerr_band_in.append([])
            for n in range(len(master_HJDnight[band][day])):
                tempfrac = vseq.calc.frac(np.array(master_HJDnight[band][day]))
                if lb < tempfrac[n] < rb:
                    HJD_band_in[band].append(tempfrac[n])
                    flux_band_in[band].append(master_fluxnight[band][day][n])
                    fluxerr_band_in[band].append(master_fluxerrnight[band][day][n])

        # red_chi = vseq.calc.error.red_X2(flux_inbounds, results, fluxerr_inbounds) / len(results)
        # root_error, kweeer = sim_min(HJD_band_in, flux_band_in, fluxerr_band_in, order, sims, filter_files)[0:3:2]

        # ===================================================================================================
        try:
            t_kw, sigt_kw = KvW(HJD_inbounds, flux_inbounds, resolution=resolution, para_range=para_range, discard=discard,
                                need_error=True, ax=Splot, npairs=npairs, entire_S=entire_S)
        except IndexError:
            print("Please try entering in that last key board press you entered. An error ocurred.\n")
            plot_obs(filter_files, day=day, lb=last_lb, rb=last_rb, order=order, resolution=resolution, npairs=npairs,
                     para_range=None, norm_method="norm")

        # bestroot = best_root(coef, min(HJD_inbounds), max(HJD_inbounds), filter_files)

        # from T_LSQ_error_test import idktho
        # idkerror = idktho(HJD_inbounds, flux_inbounds, bestroot, coef)

        MIN.set_xlabel(r'$\mathrm{HJD}-' + str(intday) + '$', fontsize=fontsize)
        MIN.set_ylabel('Flux', fontsize=fontsize)
        MIN.grid(alpha=0.0)
        MIN.set_ylim(None, 0.06 + MIN.get_ylim()[-1])
        MIN.margins(0.02, None)
        MIN.axvline(lb, color='limegreen', linestyle='--', alpha=1, linewidth=None)
        MIN.axvline(rb, color='limegreen', linestyle='--', alpha=1, linewidth=None)
        # MIN.axvline(bestroot, color='r',
        #            label=r'$T_{\rm LSQ^' + str(order) + '}=' + str(round(intday * 0 + bestroot, 6)) + '\pm' + str(
        #                round(idkerror, 6)) + '$')
        poly_x, poly_y = vseq.calc.poly.polylist(coef, min(HJD_inbounds), max(HJD_inbounds), 100)
        MIN.plot(poly_x, poly_y, 'r-', lw=1, zorder=10)
        Splot.xaxis.set_label_position("top")
        Splot.xaxis.tick_top()

        print('============================')
        print('T_KW:     ' + str(intday + t_kw))
        # totalerr = np.sqrt(sigt_kw ** 2 + kweeer ** 2)
        if sigt_kw != float(sigt_kw):
            MIN.axvline(t_kw, color='blue',
                        label=r' $T_{\rm KW}=' + str(round(intday * 0 + t_kw, 6)) + '\pm$' + 'ERROR')
        else:
            MIN.axvline(t_kw, color='blue', label=r' $  T_{\rm KW}=' + str(round(intday * 0 + t_kw, 6)) + '\pm' + str(
                round(sigt_kw, 6)) + '$')
            print('err_kw:   ' + str(round(sigt_kw, 16)))
        print('')
        # print('T_LSQ:    ' + str(bestroot + intday))
        # print('err_form: ' + str(round(idkerror, 16)))
        # print('err_MC:   ' + str(round(root_error, 16)))

        MIN.legend(loc='upper right', framealpha=1, borderaxespad=0, shadow=False, fancybox=False, edgecolor='k', fontsize=5)
        MIN.annotate('' + filter_files[0] + '', xy=(MIN.get_xlim()[0] + 0.001, MIN.get_ylim()[1] - 0.005), va='top',
                     ha='left', fontsize=fontsize)

        vseq.plot.sm_format(MIN, xtop=False, yright=False, ydirection='out', xdirection='out', X=Xtick, numbersize=6)
        vseq.plot.sm_format(Splot, xbottom=False, yright=False, ydirection='out', xdirection='out', bottomspine=False,
                            y=1000000, numbersize=6)
        plt.grid(alpha=0.35)
        plt.savefig('MIN-program_demo.pdf', bbox_inches='tight')
        # print(f"Backend: {plt.get_backend()}")
        f = zoom_factory(MIN, base_scale=2.)
        plt.show()
    else:
        ax = plt.figure(figsize=(9, 7), dpi=256).subplots()
        vseq.plot.sm_format(ax, xformatter=False, numbersize=7, Xsize=0, xsize=0, tickwidth=1)
        scaling = 1
        # fracHJD=vseq.calc.frac(np.array(all_HJD[day]))*scaling
        fracHJD = vseq.calc.frac(np.array(all_HJD[day])) * scaling

        # plt.plot(fracHJD,all_flux[day],'ok',ms=3)
        plt.errorbar(fracHJD, all_flux[day], all_fluxerr[day], fmt='ok', ms=3, capsize=3, elinewidth=0.6, ecolor='gray',
                     capthick=0.6)  # ,'ok',ms=3)
        # plt.errorbar(fracHJD,-2.5*np.log10(np.array(all_flux[day])),all_fluxerr[day],fmt='ok',ms=3,capsize=3,elinewidth=0.6,ecolor='gray',capthick=0.6)#,'ok',ms=3)
        plt.xlabel('+' + str(intday) + ' [HJD]', fontsize=fontsize)
        plt.ylabel('Flux', fontsize=fontsize)

        plt.grid(alpha=0.35)
        plt.show()

    last_lb = lb
    last_rb = rb

    return 'Done'


def press(event):
    global rb
    global lb
    if event.key == "d":
        # right boundary line
        lb = lb
        rb = event.xdata
        plt.close()
        plot_obs(["896797_B.txt"], day=day, lb=lb, rb=rb, order=order, resolution=resolution, npairs=npairs,
                 para_range=None, norm_method="norm")
    elif event.key == "a":
        # left boundary line
        lb = event.xdata
        rb = rb
        plt.close()
        plot_obs(["896797_B.txt"], day=day, lb=lb, rb=rb, order=order, resolution=resolution, npairs=npairs,
                 para_range=None, norm_method="norm")
    elif event.key == "w":
        # writes to a file
        pass
    elif event.key == "escape":
        # exits the program
        exit()
    elif event.key == "f":
        pass
    else:
        print("\nPress 'a' for right boundary, 'd' for left boundary, or the 'ESC' key to close the figure.\n")


def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    # return the function
    return zoom_fun


def main():
    num_filters = input("How many filters do you have (i.e. 1-3) or type 'Close' to close the program: ")
    print()
    if num_filters == '1':
        while True:
            filter1 = input("Please enter a complete file pathway to your data file (i.e. C:\\folder1\\folder2\\data.txt: ")
            if path.exists(filter1):
                break
            else:
                print("\nOne of the files you have entered does not exist, please try all three again.\n")
                continue
        plot_obs([filter1], day=day, lb=lb, rb=rb, order=order, resolution=resolution, npairs=npairs,
                 para_range=None, norm_method='norm')
    elif num_filters == '2':
        while True:
            filter1 = input("Please enter a complete file pathway to data file 1 (i.e. C:\\folder1\\folder2\\data.txt: ")
            filter2 = input("Please enter a complete file pathway to data file 2 (i.e. C:\\folder1\\folder2\\data.txt: ")
            if path.exists(filter1) and path.exists(filter2):
                break
            else:
                print("\nOne of the files you have entered does not exist, please try all three again.\n")
                continue
        plot_obs([filter1, filter2], day=day, lb=lb, rb=rb, order=order, resolution=resolution, npairs=npairs,
                 para_range=None, norm_method='norm')
    elif num_filters == '3':
        while True:
            filter1 = input("Please enter a complete file pathway to data file 1 (i.e. C:\\folder1\\folder2\\data.txt: ")
            filter2 = input("Please enter a complete file pathway to data file 2 (i.e. C:\\folder1\\folder2\\data.txt: ")
            filter3 = input("Please enter a complete file pathway to data file 3 (i.e. C:\\folder1\\folder2\\data.txt: ")
            if path.exists(filter1) and path.exists(filter2) and path.exists(filter3):
                break
            else:
                print("\nOne of the files you have entered does not exist, please try all three again.\n")
                continue
        plot_obs([filter1, filter2, filter3], day=day, lb=lb, rb=rb, order=order, resolution=resolution, npairs=npairs,
                 para_range=None, norm_method='norm')
    elif num_filters.lower() == "close":
        exit()
    else:
        print("Please enter a number between 1-3 or the word 'Close'.\n")
        main()


if __name__ == '__main__':
    main()
