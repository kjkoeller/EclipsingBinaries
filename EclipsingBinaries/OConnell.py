# -*- coding: utf-8 -*-
"""
Calculates the O'Connel Effect based on this paper: https://app.aavso.org/jaavso/article/3511/

Created on Thu Feb 25 00:47:37 2021
Last Edited: 05/29/2024

Original Author: Alec Neal
Last Edits Done By: Kyle Koeller
"""

# Importing necessary libraries
import matplotlib.pyplot as plt
from .vseq_updated import plot, binning, calc, FT, OConnell
# from vseq_updated import plot, binning, calc, FT, OConnell  # testing purposes
from tqdm import tqdm
import numpy as np
import statistics as st
from os import path

# Lambda function to calculate sigma of a function
sig_f = lambda f, x, sig_x: abs(f(x + sig_x) - f(x - sig_x)) / 2


# Main function for calculating O'Connell Effect
def main(filepath="", pipeline=False, radec_list=None, obj_name="", period=0, hjd=0):
    if not pipeline:
        # User interaction for number of filters
        print("How many filters are you going to use?")
        while True:
            prompt = input(
                "Enter 3 if you are going to use BVR or less than 3 for a combination of them of type 'Close' "
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

        # Logic to handle different numbers of filters
        if prompt == "3":
            # For 3 filters (BVR)
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
            outputile = input(
                "What is the output file name and pathway with .pdf extension (i.e. C:\\folder1\\test.pdf): ")
            multi_OConnell_total([infile1, infile2, infile3], hjd, period, order=10, sims=1000,
                                 sections=4, section_order=7, plot_only=False, save=True, outName=outputile,
                                 pipeline=pipeline)
        elif prompt == "2":
            # For 2 filters
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
            outputile = input(
                "What is the output file name and pathway with .pdf exntension (i.e. C:\\folder1\\test.pdf): ")
            multi_OConnell_total([infile1, infile2], hjd, period, order=10, sims=1000,
                                 sections=4, section_order=7, plot_only=False, save=True, outName=outputile,
                                 pipeline=pipeline)
        else:
            # For 1 filter
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
            outputile = input(
                "What is the output file name and pathway with .pdf exntension (i.e. C:\\folder1\\test.pdf): ")
            multi_OConnell_total([infile1], hjd, period, order=10, sims=1000,
                                 sections=4, section_order=7, plot_only=False, save=True, outName=outputile,
                                 pipeline=pipeline)
    else:
        # If running in a pipeline mode
        multi_OConnell_total([radec_list], hjd, period, order=10, sims=1000,
                             sections=4, section_order=7, plot_only=False, save=True,
                             outName=(filepath + "\\" + obj_name + ".pdf"), pipeline=pipeline)


def quick_tex(thing):
    """
    Quick TeX formatting function.
    """
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.usetex'] = False


# Lambda function to calculate dI_phi
dI_phi = lambda b, phase, order: 2 * sum(b[1:order + 1:] * np.sin(2 * np.pi * phase * np.arange(order + 1)[1::]))


def Half_Comp(filter_files, Epoch, period,
              FT_order=10, sections=4, section_order=8,
              resolution=512, offset=0.25, save=False, outName='noname_halfcomp.png',
              title=None, filter_names=None, sans_font=False):
    # Setting font family if not using sans font
    if sans_font == False:
        plt.rcParams['font.family'] = 'serif'  # Set font family to serif if sans_font is False
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # Set math font to serif if sans_font is False

    # Calculate the number of bands
    bands = len(filter_files)

    # Create subplots for flux and dI
    axs, fog = plot.multiplot(figsize=(6, 9), dpi=512, height_ratios=[7 / 3 * bands, 3])
    flux = axs[0]  # Flux subplot
    dI = axs[1]  # dI subplot

    colors = ['blue', 'limegreen', 'red', 'm']  # Color list for plots
    styles = ['--', '-.', ':', '--']  # Line styles for plots

    R2_halves = []  # List to store R^2 values for each band

    # Draw horizontal line at y=0 on dI subplot
    dI.axhline(0, linestyle='-', color='black', linewidth=1)

    # Loop over each band
    for band in range(bands):
        # Binning of data
        a, b = binning.polybinner(filter_files[band], Epoch, period, sections=sections,
                                  norm_factor='alt', section_order=section_order)[0][:2:]

        half = int(0.5 * resolution) + 1  # Calculate half index for resolution

        # Compute Fourier Transform for positive and negative bins
        FT1 = FT.FT_plotlist(a, b, FT_order, resolution)
        FT2 = FT.FT_plotlist(a, -1 * b, FT_order, resolution)
        FTphase1, FTflux1 = FT1[0][:half:], FT1[1][:half:]  # Positive phase and flux
        FTphase2, FTflux2 = FT2[0][:half:], FT2[1][:half:]  # Negative phase and flux

        # Calculate coefficient of determination (R^2) between positive and negative fluxes
        R2_halves.append(calc.error.CoD(FTflux1, FTflux2))
        R2_halves.append(calc.error.CoD(FTflux2, FTflux1))

        dIlist = []  # List to store dI values
        for phase in FTphase1:
            dIlist.append(dI_phi(b, phase, FT_order))  # Calculate dI values

        # Adjust flux values for plotting
        FTflux1 = np.array(FTflux1) + (1 - band) * offset
        FTflux2 = np.array(FTflux2) + (1 - band) * offset

        # Plot flux and dI
        flux.plot(FTphase1, FTflux1, linestyle=styles[band], color=colors[band])
        flux.plot(FTphase2, FTflux2, '-', color=colors[band])

        # Add filter names to flux subplot if provided
        if filter_names is not None:
            if len(filter_names) == bands:
                flux.text(-0.12, FTflux1[0], filter_names[band], fontsize=18, rotation=0)

        # Plot dI
        dI.plot(FTphase1, dIlist, linestyle=styles[band], color=colors[band])

    # Set x-axis limit for flux subplot
    plt.xlim(-0.025, 0.525)

    # Set y-axis label for flux subplot if filter_names is None
    if filter_names is None:
        flux.set_ylabel('Flux', fontsize=16)

    # Format subplots
    plot.sm_format(flux, numbersize=15, X=0.125, x=0.025, xbottom=False, bottomspine=False, Y=None)
    plot.sm_format(dI, numbersize=15, X=0.125, x=0.025, xtop=False, topspine=False)
    dI.set_xlabel(r'$\Phi$', fontsize=18)  # Set x-axis label for dI subplot
    dI.set_ylabel(r'$\Delta I(\Phi)_{\rm FT}$', fontsize=18)  # Set y-axis label for dI subplot

    # Set title for flux subplot if title is not empty
    if title != '':
        flux.set_title(title, fontsize=14.4, loc='left')

    # Save figure if save is True
    if save:
        plt.savefig(outName, bbox_inches='tight')
        print(outName + ' saved.')

    plt.show()  # Show the plot

    # Reset font settings to default
    plt.rcParams['font.family'] = 'sans'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    return print('\nDone.')  # Return "Done" message


def OConnell_total(inputFile, Epoch, period, order, sims=1000,
                   sections=4, section_order=8, norm_factor='alt',
                   starName='', filterName='', FT_order=10, FTres=500):
    """
    This function calculates various parameters related to the O'Connell Effect.
    It performs Monte Carlo simulations to estimate errors.
    Approximate runtime: ~ sims/1000 minutes.
    """

    """
    Generating master parameters from the observational data.
    """
    # ============================== DO NOT CHANGE ==============================
    # Binning the data
    PB = binning.polybinner(inputFile, Epoch, period, sections=sections, norm_factor=norm_factor,
                            section_order=section_order, FT_order=FT_order)
    c_MB = PB[1][0]  # Correct phase bins and fluxes
    nc_MB = PB[1][1]  # Non-corrected phase bins and fluxes
    ob_phaselist = c_MB[0][1]  # Corrected phase list
    ob_fluxlist = c_MB[1][1]  # Corrected flux list
    ob_fluxerr = c_MB[1][2]  # Corrected flux error list
    c_sec_phases = c_MB[5][0]  # Corrected section phases
    nc_sec_phases = nc_MB[5][0]  # Non-corrected section phases
    c_sec_index = c_MB[5][3]  # Corrected section index
    nc_sec_index = nc_MB[5][3]  # Non-corrected section index

    a = PB[0][0]  # Fourier coefficients (cosine terms)
    b = PB[0][1]  # Fourier coefficients (sine terms)
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
    List of the resampled polynomial fluxes. Each embedded list is different 
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

    # == print parameters ==
    r = lambda x: round(x, 5)

    valerr = lambda x, dx, label, PRECISION=6: print(label + ' =', round(x, PRECISION), '+/-', round(dx, PRECISION))

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

    return [a, a_total_err], [b, b_total_err], [Delta_I_025, Delta_I_025_total_err], [Delta_I_ave, Delta_I_ave_err], [
        OER, OER_total_err], [LCA, LCA_total_err], [a2_0125_a2, a2_0125_a2_err]


def multi_OConnell_total(filter_files, Epoch, period, order=10,
                         sims=1000, sections=4, section_order=8,
                         norm_factor='alt', starName='', filterNames=[r'$\rm B$', r'$\rm V$', r'$\rm R_C$'],
                         FT_order=10, FTres=500, plot_only=False,
                         plotoff=0.25, save=False, outName='noname.png', pipeline=False):
    """
    This function generates a half-comparison plot and calculates various parameters related to the O'Connell Effect.
    If plot_only is set to True, only the half-comparison plot will be generated.
    """

    # Generate half-comparison plot
    Half_Comp(filter_files, Epoch, period, FT_order=order, sections=sections,
              section_order=section_order, offset=plotoff, save=save, outName=outName,
              filter_names=filterNames)

    # Perform O'Connell analysis
    if not plot_only:
        filters = len(filter_files)
        a_all = []
        a_err_all = []
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

        # Running the O'Connell function on each filter
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

        # LaTeX table creation
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
            a22_line += '& $' + strr(a22s[band]) + '\pm ' + strr(a22s_err[band]) + '$ '
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
        # End LaTeX table creation

        print(output)  # Output LaTeX table to console
        outputfile = input("Please enter an output file name without the extension: ")
        file = open(outputfile + ".txt", "w")
        file.write(output)  # Write LaTeX table to file
        file.close()

    return 'nada'


if __name__ == '__main__':
    main()
