"""
Author: Kyle Koeller
Date Created: 03/08/2023

Last Edited: 04/05/2023
This program queries Gaia DR3, to gather specific parameters
https://gea.esac.esa.int/archive/
https://iopscience.iop.org/article/10.3847/1538-3881/acaaa7/pdf
https://iopscience.iop.org/article/10.3847/1538-3881/ab3467/pdf
https://arxiv.org/pdf/2012.01916.pdf
"""

from pyia import GaiaData
import pandas as pd
from .vseq_updated import splitter
# from vseq_updated import splitter  # testing
import math as mt


def target_star():
    """
    This queries Gaia DR3 all the parameters below, but I only outputted the specific parameters that are (at the moment)
    the most important for current research at BSU

    :return: Outputs a file with the specific parameters
    """
    # 13:27:50.4728234064 75:39:45.384765984
    # 00:28:27.9684836736 78:57:42.657327180
    ra_input = input("\nEnter the RA of your system (HH:MM:SS.SSSS): ")
    dec_input = input("Enter the DEC of your system (DD:MM:SS.SSSS or -DD:MM:SS.SSSS): ")

    ra_input2 = splitter([ra_input])
    dec_input2 = splitter([dec_input])

    ra = ra_input2[0] * 15
    dec = dec_input2[0]

    # tess_mag(ra, dec)

    g = GaiaData.from_query("""
    SELECT TOP 2000 
    gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.parallax_error,
    gaia_source.pmra,gaia_source.pmdec,gaia_source.ruwe,gaia_source.phot_g_mean_mag,gaia_source.bp_rp,
    gaia_source.radial_velocity,gaia_source.radial_velocity_error,gaia_source.rv_method_used,
    gaia_source.phot_variable_flag,gaia_source.non_single_star,gaia_source.has_xp_continuous,
    gaia_source.has_xp_sampled,gaia_source.has_rvs,gaia_source.has_epoch_photometry,gaia_source.has_epoch_rv,
    gaia_source.has_mcmc_gspphot,gaia_source.has_mcmc_msc,gaia_source.teff_gspphot,gaia_source.teff_gspphot_lower,
    gaia_source.teff_gspphot_upper,gaia_source.logg_gspphot,
    gaia_source.mh_gspphot,gaia_source.distance_gspphot,gaia_source.distance_gspphot_lower,
    gaia_source.distance_gspphot_upper,gaia_source.azero_gspphot,gaia_source.ag_gspphot,
    gaia_source.ebpminrp_gspphot,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_mag,gaia_source.phot_rp_mean_mag
    FROM gaiadr3.gaia_source 
    WHERE 
    CONTAINS(
        POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
        CIRCLE(
            'ICRS',
            COORD1(EPOCH_PROP_POS({}, {},4.7516,48.8840,-24.1470,0,2000,2016.0)),
            COORD2(EPOCH_PROP_POS({}, {},4.7516,48.8840,-24.1470,0,2000,2016.0)),
            0.001388888888888889)
    )=1""".format(ra, dec, ra, dec))

    # to add parameters to the output file, add them here and the format for the parameter is 'g.[param name from above]'
    df = pd.DataFrame({
        "Parallax(mas)": g.parallax[:4],
        "Parallax_err(mas)": g.parallax_error[:4],
        "Distance_lower(pc)": g.distance_gspphot_lower[:4],
        "Distance(pc)": g.distance_gspphot[:4],
        "Distance_higher(pc)": g.distance_gspphot[:4],
        "T_eff_lower(K)": g.teff_gspphot_lower[:4],
        "T_eff(K)": g.teff_gspphot[:4],
        "T_eff_higher(K)": g.teff_gspphot_upper[:4],
        "G_Mag": g.phot_g_mean_mag[:4],
        "G_BP_Mag": g.phot_bp_mean_mag[:4],
        "G_RP_Mag": g.phot_rp_mean_mag[:4],
        "Radial_velocity(km/s)": g.radial_velocity[:4],
        "Radial_velocity_err(km/s)": g.radial_velocity_error[:4],
    })

    text_file = input("\nEnter a text file pathway and name for Gaia data "
                      "(ex: C:\\folder1\\Gaia_254037.txt): ")
    df.to_csv(text_file, index=None, sep="\t")

    print("\n For more information on each of the output parameters please reference this webpage: "
          "https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html")
    print("If any of the parameters have values of '1e+20', then Gaia does not have data on that specific parameter.")

    print("\nCompleted save.\n")


def tess_mag(ra, dec):
    """
    Calculates TESS magnitudes for comparison stars

    :param ra: List of RA's for comparison stars
    :param dec: List of DEC's for comparison stars

    :return: list of TESS magnitudes and errors
    """
    T_list = []
    T_err_list = []
    for count, val in enumerate(ra):
        # the query searching for the magnitudes and fluxes
        g = GaiaData.from_query("""
            SELECT TOP 2000 gaia_source.source_id,gaia_source.ra,gaia_source.dec,
            gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,
            gaia_source.phot_bp_mean_flux_over_error,gaia_source.phot_bp_mean_mag,
            gaia_source.phot_rp_mean_flux_over_error,gaia_source.phot_rp_mean_mag
            FROM gaiadr3.gaia_source 
            WHERE 
            CONTAINS(
                POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
                CIRCLE('ICRS',{},{}, {})
        )=1""".format(val*15, dec[count], 0.0008333333333333334))
        # 0.0002777777777777778 is 1 arcsecond,
        # 0.0005555555555555556 is 2 arcseconds,
        # 0.0008333333333333334 is 3 arcseconds
        # 0.001388888888888889 is 5 arcseconds

        # 13:27:50.4728234064 75:39:45.384765984
        # 00:28:27.9684836736 78:57:42.657327180

        """
        Each mag and flux from Gaia's filters
        Use .value at the end of each Gaia variable to get only the number and not the unit
        The fluxes are coming in as arrays and not quantities, so the .value cannot be applied like for the magnitudes
        """
        G = g.phot_g_mean_mag[:4].value
        G_flux = g.phot_g_mean_flux_over_error[:4]
        BP = g.phot_bp_mean_mag[:4].value
        BP_flux = g.phot_bp_mean_flux_over_error[:4]
        RP = g.phot_rp_mean_mag[:4].value
        RP_flux = g.phot_rp_mean_flux_over_error[:4]

        if len(BP) == 0 or len(RP) == 0:
            # this equation is used if there are no BP or RP magnitudes
            if len(G) == 0 or len(G_flux) == 0:
                T_list.append(99.999)
                T_err_list.append(99.999)
            else:
                T = G - 0.403
                G_err = (2.5 / mt.log(10)) * (G_flux[0] ** -1)
                T_list.append(T[0])
                T_err_list.append(G_err)
        else:
            # calculate the TESS magnitude (T) and corresponding error (T_err)
            T = G - 0.00522555 * (BP - RP) ** 3 + 0.0891337 * (BP - RP) ** 2 - 0.633923 * (BP - RP) + 0.0324473
            T = T[0]

            # error propagation formulae
            G_err = (2.5 / mt.log(10)) * (G_flux[0] ** -1)
            BP_err = (2.5 / mt.log(10)) * (BP_flux[0] ** -1)
            RP_err = (2.5 / mt.log(10)) * (RP_flux[0] ** -1)

            Bp_Rp_err = mt.sqrt(BP_err ** 2 + RP_err ** 2)
            T_err = mt.sqrt(G_err ** 2 + (Bp_Rp_err * 3) ** 2)

            # append to respective lists with only 2 decimal places, instead of ~15
            T_list.append(float(format(T, ".2f")))
            T_err_list.append(float(format(T_err, ".3f")))

    return T_list, T_err_list


if __name__ == '__main__':
    target_star()
