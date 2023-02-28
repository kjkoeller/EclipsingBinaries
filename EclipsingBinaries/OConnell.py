# -*- coding: utf-8 -*-
"""
Calculates the O'Connel Effect based on this paper: https://app.aavso.org/jaavso/article/3511/

Created on Thu Feb 25 00:47:37 2021
Last Edited: 02/27/2023

Original Author: Alec Neal
Last Edits Done By: Kyle Koeller
"""

import matplotlib.pyplot as plt
from .vseq_updated import plot, binning, calc, FT, OConnell
from tqdm import tqdm
import numpy as np
import statistics as st
from os import path

sig_f = lambda f, x, sig_x: abs(f(x + sig_x) - f(x - sig_x)) / 2


def main():
    print("How many filters are you going to use?")
    while True:
        prompt = input("Enter 3 if you are going to use BVR or less than 3 for a combination of them of type 'Close' "
                       "to close the program: ")
        if prompt.isnumeric():
            if 0 < int(prompt) < 4:
                break
            else:
                print("\nYou have entered in a wrong number not between 1-3. Please try again.\n")
        elif prompt.lower() == "close":
            exit()
        else:
            print("\nYou have entered a wrong value. Please try entering a number or the word 'Close'.\n")
    # If statement checks which value was entered for the 'prompt' and corresponds the correct number of files to enter
    # for the O'Connel calculations
    if prompt == "3":
        print("\nPlease enter full file pathways for the following prompt.\n")
        while True:
            infile1 = input("File 1 name: ")
            infile2 = input("File 2 name: ")
            infile3 = input("File 3 name: ")
            if path.exists(infile1) and path.exists(infile2) and path.exists(infile3):
                break
            else:
                print("\nOne of the files you have entered does not exist, please try all three again.\n")
                continue
        hjd = float(input("What is the HJD: "))
        period = float(input("What is the period: "))
        outputile = input("What is the output file name and pathway with .pdf exntension (i.e. C:\\folder1\\test.pdf): ")
        multi_OConnell_total([infile1, infile2, infile3], hjd, period, order=10, sims=1000,
                             sections=4, section_order=7, plot_only=False, save=True, outName=outputile)
    elif prompt == "2":
        print("\nPlease enter full file pathways for the following prompt.\n")
        while True:
            infile1 = input("File 1 name: ")
            infile2 = input("File 2 name: ")
            if path.exists(infile1) and path.exists(infile2):
                break
            else:
                print("\nOne of the files you have entered does not exist, please try all three again.\n")
                continue
        hjd = float(input("What is the HJD: "))
        period = float(input("What is the period: "))
        outputile = input("What is the output file name and pathway with .pdf exntension (i.e. C:\\folder1\\test.pdf): ")
        multi_OConnell_total([infile1, infile2], hjd, period, order=10, sims=1000,
                             sections=4, section_order=7, plot_only=False, save=True, outName=outputile)
    else:
        print("\nPlease enter full file pathways for the following prompt.\n")
        while True:
            infile1 = input("File 1 name: ")
            if path.exists(infile1):
                break
            else:
                print("\nThe file you have entered does not exist, please try all three again.\n")
                continue
        hjd = float(input("What is the HJD: "))
        period = float(input("What is the period: "))
        outputile = input("What is the output file name and pathway with .pdf exntension (i.e. C:\\folder1\\test.pdf): ")
        multi_OConnell_total([infile1], hjd, period, order=10, sims=1000,
                             sections=4, section_order=7, plot_only=False, save=True, outName=outputile)


# print(sig_f(lambda x:1/x,3.4,0.01))
# print(0.01/3.4**2)

def quick_tex(thing):
    plt.rcParams['text.usetex'] = True
    # thing
    plt.rcParams['text.usetex'] = False


dI_phi = lambda b, phase, order: 2 * sum(b[1:order + 1:] * np.sin(2 * np.pi * phase * np.arange(order + 1)[1::]))


def Half_Comp(filter_files, Epoch, period,
              FT_order=10, sections=4, section_order=8,
              resolution=512, offset=0.25, save=False, outName='noname_halfcomp.png',
              title=None, filter_names=None, sans_font=False):
    if sans_font == False:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    bands = len(filter_files)
    axs, fog = plot.multiplot(figsize=(6, 9), dpi=512, height_ratios=[7 / 3 * bands, 3])
    flux = axs[0]
    dI = axs[1]
    colors = ['blue', 'limegreen', 'red', 'm']
    styles = ['--', '-.', ':', '--']

    R2_halves = []
    dI.axhline(0, linestyle='-', color='black', linewidth=1)
    for band in range(bands):
        a, b = binning.polybinner(filter_files[band], Epoch, period, sections=sections,
                                       norm_factor='alt', section_order=section_order)[0][:2:]
        half = int(0.5 * resolution) + 1
        FT1 = FT.FT_plotlist(a, b, FT_order, resolution)
        FT2 = FT.FT_plotlist(a, -1 * b, FT_order, resolution)
        FTphase1, FTflux1 = FT1[0][:half:], FT1[1][:half:]
        FTphase2, FTflux2 = FT2[0][:half:], FT2[1][:half:]

        # R2_halves.append(np.mean([calc.error.CoD(FTflux1,FTflux2),calc.error.CoD(FTflux2,FTflux1)]))
        R2_halves.append(calc.error.CoD(FTflux1, FTflux2))
        R2_halves.append(calc.error.CoD(FTflux2, FTflux1))

        dIlist = []
        for phase in FTphase1:
            dIlist.append(dI_phi(b, phase, FT_order))
        FTflux1 = np.array(FTflux1) + (1 - band) * offset
        FTflux2 = np.array(FTflux2) + (1 - band) * offset
        flux.plot(FTphase1, FTflux1, linestyle=styles[band], color=colors[band])
        flux.plot(FTphase2, FTflux2, '-', color=colors[band])
        # flux.annotate('B',xy=(-0.1,min(FTflux1)))
        if filter_names is not None:
            if len(filter_names) == bands:
                flux.text(-0.12, FTflux1[0], filter_names[band], fontsize=18, rotation=0)
        dI.plot(FTphase1, dIlist, linestyle=styles[band], color=colors[band])

    plt.xlim(-0.025, 0.525)
    if filter_names is None:
        flux.set_ylabel('Flux', fontsize=16)

    # mpl.rcParams['mathtext.fontset'] = 'dejavusans'
    plot.sm_format(flux, numbersize=15, X=0.125, x=0.025, xbottom=False, bottomspine=False, Y=None)
    plot.sm_format(dI, numbersize=15, X=0.125, x=0.025, xtop=False, topspine=False)
    # plt.rcParams['text.usetex'] = True
    # mpl.rcParams['mathtext.fontset'] = 'cm'
    dI.set_xlabel(r'$\Phi$', fontsize=18)
    # plt.rcParams['text.usetex'] = False
    dI.set_ylabel(r'$\Delta I(\Phi)_{\rm FT}$', fontsize=18)
    # dI.set_ylabel(r'$\Delta I(\Phi)_{\rm FT}$',fontsize=14.4)
    if title != '':
        flux.set_title(title, fontsize=14.4, loc='left')
    '''
    print(flux.get_ylim())
    data_int=flux.yaxis.get_data_interval()
    print(data_int)
    allticks=sorted(list(flux.yaxis.get_minorticklocs())+list(flux.yaxis.get_majorticklocs()))
    minorticks=flux.yaxis.get_minorticklocs()
    tick_spacing=min(abs(np.array(minorticks[1::])-np.array(minorticks[:-1:])))
    print(allticks)
    print(tick_spacing)
    flux.set_ylim(top=max(allticks[1:-1])+tick_spacing)#-tick_spacing)
    '''
    # print(flux.yaxis.get_ticks())
    if save:
        plt.savefig(outName, bbox_inches='tight')
        print(outName + ' saved.')
    plt.show()

    # reset to mpl default style
    plt.rcParams['font.family'] = 'sans'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # print('R^2 half =',(calc.error.CoD(FTflux1,FTflux2)+calc.error.CoD(FTflux2,FTflux1))/2)
    # print(R2_halves)
    # print(1-np.mean(R2_halves),'+/-',statistics.stdev(R2_halves))
    # print()

    return print('\nDone.')


def OConnell_total(inputFile, Epoch, period, order, sims=1000,
                   sections=4, section_order=8, norm_factor='alt',
                   starName='', filterName='', FT_order=10, FTres=500):
    """
    This does things.
    Monte Carlo errors are probably underestimates.
    Approximate runtime: ~ sims/1000 minutes.
    glhf
    """

    """
    Generating master parameters from the observational data.
    """
    # ============== DO NOT CHANGE ┬─┬ノ( º _ ºノ) ==============================
    PB = binning.polybinner(inputFile, Epoch, period, sections=sections, norm_factor=norm_factor,
                                 section_order=section_order, FT_order=FT_order)
    c_MB = PB[1][0]
    nc_MB = PB[1][1]
    ob_phaselist = c_MB[0][1]
    ob_fluxlist = c_MB[1][1]
    ob_fluxerr = c_MB[1][2]
    c_sec_phases = c_MB[5][0]
    nc_sec_phases = nc_MB[5][0]
    c_sec_index = c_MB[5][3]
    nc_sec_index = nc_MB[5][3]
    # norm_f = c_MB[4]
    # ob_magerr = c_MB[3][2]
    a = PB[0][0]
    b = PB[0][1]  # Fourier coefficients
    # ==========================================================================

    FTsynth = FT.synth(a, b, ob_phaselist, FT_order)  # synthetic FT curve
    master_simflux = []
    for sim in tqdm(range(sims), desc='Simulating light curves', position=0):
        master_simflux.append(FT.sim_ob_flux(FTsynth, ob_fluxerr))
    # ============

    # ============
    c_master_simflux = []
    nc_master_simflux = []
    master_polyflux = []
    """
    ^List of the resampled polynomial fluxes. Each embedded list is different 
    because the generating data has been Monte Carloed.
    """
    master_FTflux = []  # FT fluxes resulting from the simulations
    master_a = []
    master_b = []  # Lists of the a and b FT coefficients for each sim
    OERlist, LCAlist, dIlist, dIavelist = [], [], [], []

    # = begin sim loop =
    for sim in tqdm(range(sims), desc='Simulation processing', position=0):
        """
        Generating empty lists so that stuff can be inserted into the
        specified [sim] index. Otherwise when trying to append list[sim],
        it would throw an error (there'd be nothing to append to).
        """
        c_master_simflux.append([])
        nc_master_simflux.append([])

        # == begin section loop ===
        for section in range(sections):
            c_master_simflux[sim].append([])  # same as before
            nc_master_simflux[sim].append([])

            """
            'Replacing' the observational fluxes with the Monte Carlo fluxes
            based on the index of each data point when it was in ob_fluxlist
            (the MC fluxes are ordered the same as ob_fluxlist).
            """
            for i in range(len(c_sec_phases[section])):
                c_master_simflux[sim][section].append(master_simflux[sim][c_sec_index[section][i]])
            for i in range(len(nc_sec_phases[section])):
                nc_master_simflux[sim][section].append(master_simflux[sim][nc_sec_index[section][i]])
        # == end section loop == ; resume sim loop

        # Generating polynomial resampled fluxes for each simulation.
        minipoly = binning.minipolybinner(c_sec_phases, c_master_simflux[sim],
                                               nc_sec_phases, nc_master_simflux[sim],
                                               section_order)
        master_polyflux.append(minipoly[1])

        # Calculating various parameters for each simulation.
        FTcoef = FT.coefficients(master_polyflux[sim])
        master_a.append(FTcoef[1])
        master_b.append(FTcoef[2])
        master_FTflux.append(FT.FT_plotlist(master_a[sim], master_b[sim], FT_order, FTres)[1])
        OERlist.append(OConnell.OER_FT(master_a[sim], master_b[sim], FT_order))
        LCAlist.append(OConnell.LCA_FT(master_a[sim], master_b[sim], FT_order, 256))
        dIlist.append(OConnell.Delta_I_fixed(master_b[sim], FT_order))
        dIavelist.append(OConnell.Delta_I_mean_obs_noerror(ob_phaselist, ob_fluxlist, phase_range=0.05))
    # = end sim loop =

    # dIaveerror=st.stdev(dIavelist)#no purpose, compining MC errors with observation errors is double-counting

    # === end Monte Carlo ===

    """
    Calculating FT model errors, as opposed to the errors calculated
    with the simulate light curves.
    """
    a0sig, a0rat = FT.a_sig_fast(a, b, 0, a[0], ob_phaselist, ob_fluxlist, ob_fluxerr, order)
    a_model_err = [a0sig]
    b_model_err = [0]
    a_rat = [a0rat]
    b_rat = [1]
    for n in tqdm(range(1, order + 1), position=0, desc='Calculating FT errors'):
        # TODO, dx0=1 isn't fullproof
        asig, ar = FT.a_sig_fast(a, b, n, a[n], ob_phaselist, ob_fluxlist, ob_fluxerr, order, dx0=1)
        bsig, br = FT.b_sig_fast(a, b, n, b[n], ob_phaselist, ob_fluxlist, ob_fluxerr, order, dx0=1)
        a_model_err.append(asig)
        b_model_err.append(bsig)
        a_rat.append(ar)
        b_rat.append(br)

    # == FT coefficients ==
    a_MC_err = np.array(list(map(st.stdev, zip(*master_a))))
    b_MC_err = np.array(list(map(st.stdev, zip(*master_b))))
    a_total_err = (np.array(a_model_err) ** 2 + a_MC_err[:order + 1:] ** 2) ** 0.5
    b_total_err = (np.array(b_model_err) ** 2 + b_MC_err[:order + 1:] ** 2) ** 0.5

    a2_0125_a2 = a[2] * (0.125 - a[2])
    a2_0125_a2_err = sig_f(lambda x: x * (0.125 - x), a[2], a_total_err[2])

    # == OER ==
    OER = OConnell.OER_FT(a, b, order)
    OER_model_err = OConnell.OER_FT_error_fixed(a, b, a_model_err, b_model_err, order)
    OER_MC_err = st.stdev(OERlist)
    OER_total_err = (OER_model_err ** 2 + OER_MC_err ** 2) ** 0.5

    # == LCA ==
    LCA = OConnell.LCA_FT(a, b, order, 1024)
    LCA_model_err = OConnell.LCA_FT_error(a, b, a_model_err, b_model_err, order, 1024)[1]
    LCA_MC_err = st.stdev(LCAlist)
    LCA_total_err = (LCA_model_err ** 2 + LCA_MC_err ** 2) ** 0.5

    # == Delta_I ==
    Delta_I_025 = OConnell.Delta_I_fixed(b, order)
    Delta_I_025_model_err = OConnell.Delta_I_error_fixed(b_model_err, order)
    Delta_I_025_MC_err = st.stdev(dIlist)
    Delta_I_025_total_err = (Delta_I_025_model_err ** 2 + Delta_I_025_MC_err ** 2) ** 0.5

    # == Delta_I_ave ==
    DIave = OConnell.Delta_I_mean_obs(ob_phaselist, ob_fluxlist, ob_fluxerr, phase_range=0.05, weighted=False)
    Delta_I_ave = DIave[0]
    Delta_I_ave_err = DIave[1]

    # == pizza time ==
    r = lambda x: round(x, 5)

    valerr = lambda x, dx, label, PRECISION=6: print(label + ' =', round(x, PRECISION), '+/-', round(dx, PRECISION))
    # print('\nOER =',r(OER),'+/-',r(OER_total_err))

    print('\n')
    valerr(a[1], a_total_err[1], 'a1')
    valerr(a[2], a_total_err[2], 'a2')
    valerr(a[4], a_total_err[4], 'a4')
    valerr(a[2] * (0.125 - a[2]),
           sig_f(lambda x: x * (0.125 - x), a[2], (a_model_err[2] ** 2 + a_MC_err[2] ** 2) ** 0.5), 'a2(0.125-a2)')
    print('')
    valerr(OER, OER_total_err, 'OER')
    print(r(OER_model_err), r(OER_MC_err), '\n')

    valerr(LCA, LCA_total_err, 'LCA')
    print(r(LCA_model_err), r(LCA_MC_err), '\n')

    valerr(Delta_I_025, Delta_I_025_total_err, 'Delta_I')
    print(r(Delta_I_025_model_err), r(Delta_I_025_MC_err), '\n')

    valerr(Delta_I_ave, Delta_I_ave_err, 'Delta_I_ave')

    # print('LCA =',r(LCA),'+/-',LCA_total_err)
    # print('Delta_I =',Delta_I_025,'+/-',Delta_I_025_total_err)
    # print('')
    return [a, a_total_err], [b, b_total_err], [Delta_I_025, Delta_I_025_total_err], [Delta_I_ave, Delta_I_ave_err], [
        OER, OER_total_err], [LCA, LCA_total_err], [a2_0125_a2, a2_0125_a2_err]


# OConnell_total('NSVS3792718_B_apasscorr.txt',2457288.809054,0.438168,10,1000,sections=2,section_order=8)
# OConnell_total('NSVS3792718_V_apasscorr.txt',2457288.809054,0.438168,10,1000,sections=2,section_order=8)
# OConnell_total('NSVS3792718_R_apasscorr.txt',2457288.809054,0.438168,10,100,sections=2,section_order=8)

# OConnell_total("254037_B.txt", 2458403.58763, 0.317471, 10, 1000, sections=2, section_order=8)

# OConnell_total('NSVS5214334_B3_mags.txt',2459016.769744,0.345551,10,10000,sections=8,section_order=3)
# OConnell_total('NSVS5214334_V4_mags.txt',2459016.769744,0.345551,10,10000,sections=8,section_order=3)
# OConnell_total('NSVS5214334_R3_mags.txt',2459016.769744,0.345551,10,10000,sections=8,section_order=3)

def multi_OConnell_total(filter_files, Epoch, period, order=10,
                         sims=1000, sections=4, section_order=8,
                         norm_factor='alt', starName='', filterNames=[r'$\rm B$', r'$\rm V$', r'$\rm R_C$'],
                         FT_order=10, FTres=500, plot_only=False,
                         plotoff=0.25, save=False, outName='noname.png'):
    """
    does multiple things.
    """

    """
    Half comparison plot. The light curve is mirroed along phase = 0, so that
    similar phases can be compared (should be the in theory). This gives an
    intuitive visual of the O'Connell effect, unlike the barrage of numbers
    this program otherwise spits out.
    """
    Half_Comp(filter_files, Epoch, period, FT_order=order, sections=sections,
              section_order=section_order, offset=plotoff, save=save, outName=outName,
              filter_names=filterNames)

    """
    Actual OConnell stuff, set plot_only=True if you just want the
    half-comparison plot---so you don't have to wait forever just for a plot!
    """
    if plot_only != True:
        filters = len(filter_files)
        a_all = []
        a_err_all = []  # list set up crap
        b_all = []
        b_err_all = []
        dI_FT = []
        dI_FT_err = []
        dI_ave = []
        dI_ave_err = []
        OERs = []
        OERs_err = []
        LCAs = []
        LCAs_err = []
        a22s = []
        a22s_err = []
        """
        Running the O'Connell function on each filter, and placing the resulting
        parameters in corresponding value, value_err lists. E.g. once this completes,
        OERs will contain the OER values for filter 1,2,... and OERs_err contains the errors.
        """
        for band in range(len(filter_files)):
            oc = OConnell_total(filter_files[band], Epoch, period, order, sims=sims,
                                sections=sections, section_order=section_order,
                                norm_factor=norm_factor, FT_order=order, FTres=FTres,
                                filterName='Filter ' + str(band + 1))
            a_all.append(oc[0][0])
            a_err_all.append(oc[0][1])
            b_all.append(oc[1][0])
            b_err_all.append(oc[1][1])
            dI_FT.append(oc[2][0])
            dI_FT_err.append(oc[2][1])
            dI_ave.append(oc[3][0])
            dI_ave_err.append(oc[3][1])
            OERs.append(oc[4][0])
            OERs_err.append(oc[4][1])
            LCAs.append(oc[5][0])
            LCAs_err.append(oc[5][1])
            a22s.append(oc[6][0])
            a22s_err.append(oc[6][1])
            # there's probably an easier way to do this (?), oh well.

        """
        LaTeX table stuff, don't change unless you know what you're doing!
        """
        table_header = '\\begin{table}[H]\n' + '\\begin{center}\n' + '\\begin{tabular}{c|'
        for band in range(filters):
            table_header += 'c'
        table_header += '}\n' + '\\hline\\hline\n' + 'Parameter '
        for band in range(filters):
            table_header += '& Filter ' + str(band + 1)
        table_header += '\\\ \n' + '\\hline\n'

        a1_line = '$a_1$ '
        a2_line = '$a_2$ '
        a4_line = '$a_4$ '
        a22_line = '$a_2(0.125-a_2)$ '
        b1_line = '$2b_1$ '
        dIFT_line = '$\Delta I_{\\rm FT}$ '
        dIave_line = '$\Delta I_{\\rm ave}$ '
        OER_line = 'OER '
        LCA_line = 'LCA '

        strr = lambda x, e=5: str(round(x, e))
        for band in range(filters):
            a1_line += '& $' + strr(a_all[band][1]) + '\pm ' + strr(a_err_all[band][1]) + '$ '
            a2_line += '& $' + strr(a_all[band][2]) + '\pm ' + strr(a_err_all[band][2]) + '$ '
            a4_line += '& $' + strr(a_all[band][4]) + '\pm ' + strr(a_err_all[band][4]) + '$ '
            a22_line += '& $' + strr(a22s[band]) + '\pm ' + strr(a22s_err[band]) + '$ '  # fix
            b1_line += '& $' + strr(2 * b_all[band][1]) + '\pm ' + strr(2 * b_err_all[band][1]) + '$ '
            dIFT_line += '& $' + strr(dI_FT[band]) + '\pm ' + strr(dI_FT_err[band]) + '$ '
            dIave_line += '& $' + strr(dI_ave[band]) + '\pm ' + strr(dI_ave_err[band]) + '$ '
            OER_line += '& $' + strr(OERs[band]) + '\pm ' + strr(OERs_err[band]) + '$ '
            LCA_line += '& $' + strr(LCAs[band]) + '\pm ' + strr(LCAs_err[band]) + '$ '

        lines = [a1_line, a2_line, a4_line, a22_line, b1_line, dIFT_line, dIave_line, OER_line, LCA_line]
        for count, line in enumerate(lines):
            line += '\\\ \n'

        output = table_header
        for count, line in enumerate(lines):
            output += line
        output += '\\hline\n' + '\\end{tabular}\n' + '\\caption{Fourier and O\'Connell stuff (' + str(
            sims) + ' sims)}\n' + '\\label{tbl:OConnell}\n' + '\\end{center}\n' + '\\end{table}\n'
        """
        End LaTeX table stuff.
        """
        print(output)
        outputfile = input("Please enter an output file name without the extension: ")
        file = open(outputfile+".txt", "w")
        file.write(output)
        file.close()
    return 'nada'


# multi_OConnell_total(['NSVS5214334_B3_mags.txt','NSVS5214334_V4_mags.txt','NSVS5214334_R3_mags.txt'],2459016.769744,0.345551,
# order=10,sims=100,sections=8,section_order=3,plot_only=True,plotoff=0.2,save=False,outName='NSVS5214334_BVR_comps.pdf')

# multi_OConnell_total(['NSVS3792718_B_apasscorr.txt','NSVS3792718_V_apasscorr.txt','NSVS3792718_R_apasscorr.txt'],2457288.809054,0.438168,
# order=10,sims=10000,sections=4,section_order=7,plot_only=True)

if __name__ == '__main__':
    main()
