"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 08/24/2023
"""

# Python imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# import time
import warnings
from tqdm import tqdm

# Astropy imports
import ccdproc as ccdp
from astropy.coordinates import SkyCoord
from astropy.io import fits
# from astropy.nddata import CCDData
# from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.wcs import WCS
import astropy.units as u
from astropy import wcs

# from astropy.visualization import ZScaleInterval

# turn off this warning that just tells the user,
# "The warning raised when the contents of the FITS header have been modified to be standards compliant."
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)


def main(path="", pipeline=False, radec_list=None, obj_name=""):
    """
    Main function for aperture photometry

    Parameters
    ----------
    radec_list: list
        RADEC files for each filter
    obj_name: str
        Name of the target
    path : str
        Path to the folder containing the images.
    pipeline : bool
        If True, then the program is being run from the pipeline and will not ask for user input.
    Returns
    -------
    N/A
    """
    filt_list = ["Empty/B", "Empty/V", "Empty/R"]
    if not pipeline:
        # path = "D:\Research\Data\\NSVS_254037\\2018.09.18-reduced"  # For testing purposes
        path = input(
            "Please enter a file pathway (i.e. C:\\folder1\\folder2\\[raw]) to where the reduced images are or type "
            "the word 'Close' to leave: ")
        # allows the user to input where the raw images are and where the calibrated images go to
        radec_file = ""
        while True:
            try:
                images_path = Path(path)
                break
            except FileNotFoundError:
                print("File not found. Please try again.")
                path = input(
                    "Please enter a file pathway (i.e. C:\\folder1\\folder2\\[reduced]) to where the reduced images are or type "
                    "the word 'Close' to leave: ")

        if path.lower() == "close":
            exit()

        science_imagetyp = 'LIGHT'
        files = ccdp.ImageFileCollection(images_path)

        for filt in filt_list:
            if "/B" in filt:
                radec_file = input("Enter the file location for the RADEC file for the B filter: ")
            elif "/V" in filt:
                radec_file = input("Enter the file location for the RADEC file for the V filter: ")
            elif "/R" in filt:
                radec_file = input("Enter the file location for the RADEC file for the R filter: ")

            image_list = files.files_filtered(imagetyp=science_imagetyp, filter=filt)
            multiple_AP(image_list, images_path, filt, pipeline=pipeline, radec_file=radec_file)
    else:
        images_path = Path(path)

        science_imagetyp = 'LIGHT'

        files = ccdp.ImageFileCollection(images_path)

        for filt in filt_list:
            if "/B" in filt:
                radec_file = radec_list[0]
            elif "/V" in filt:
                radec_file = radec_list[1]
            elif "/R" in filt:
                radec_file = radec_list[2]
            image_list = files.files_filtered(imagetyp=science_imagetyp, filter=filt)
            substring_to_match = obj_name
            filtered_image_list = [file for file in image_list if substring_to_match in file]
            multiple_AP(filtered_image_list, images_path, filt, pipeline=pipeline, radec_file=radec_file)


def multiple_AP(image_list, path, filter, pipeline=False, radec_file=""):
    """
    Perform multi-aperture photometry on a list of images for a single target

    Parameters
    ----------
    filter: String
        Filter used for the images
    pipeline: Boolean
        If True, then the program is being run from the pipeline and will not ask for user input.
    radec_file: string
        Location of a radec file. If not given, the user will be prompted to enter one.
    path : pathway
        Path to the folder containing the images.
    image_list : List
        Images to perform multi-aperture photometry on.

    Returns
    -------
    None
    """

    # Define the aperture parameters
    # Define the aperture and annulus radii
    aperture_radius = 20
    annulus_radii = (30, 50)

    read_noise = 10.83  # * u.electron  # gathered from fits headers manually
    # gain = 1.43  # * u.electron / u.adu  # gathered from fits headers manually
    # F_dark = 0.01  # dark current in u.electron / u.pix / u.s

    if not pipeline:
        while True:
            try:
                # df = pd.read_csv('NSVS_254037-B.radec', skiprows=7, sep=",", header=None)
                df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)
                break
            except FileNotFoundError:
                print("File not found. Please try again.")
                radec_file = input("Please enter the RADEC file (i.e. C://folder1//folder2//[file name]: ")
    else:
        df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)
        print("RADEC file found.\n")

    magnitudes_comp = df[4]

    magnitudes_comp = magnitudes_comp.replace(99.999, pd.NA).dropna().reset_index(drop=True)

    ra = df[0]
    dec = df[1]
    # ref_star = df[2]
    # centroid = df[3]  # Not used (I don't think at least)

    magnitudes = []
    mag_err = []
    hjd = []
    bjd = []

    for icount, image_file in tqdm(enumerate(image_list), desc="Performing aperture photometry on {} images".format(len(image_list))):
        image_data, header = fits.getdata(path / image_file, header=True)
        # All the following up till the 'if' statement stays under the for loop due to needing the header information
        wcs_ = WCS(header)

        # ccd = CCDData(image_data, wcs=wcs, unit='adu')

        # Convert RA and DEC to pixel positions
        sky_coords = SkyCoord(ra, dec, unit=(u.h, u.deg), frame='icrs')
        pixel_coords = wcs_.world_to_pixel(sky_coords)

        x_coords, y_coords = pixel_coords

        # target_position = np.array(pixel_coords[0])
        # comparison_positions = np.array(pixel_coords[1:])
        target_position = (x_coords[0], y_coords[0])
        comparison_positions = list(zip(x_coords[1:], y_coords[1:]))

        hjd.append(header['HJD-OBS'])
        bjd.append(header['BJD-OBS'])

        # Create the apertures and annuli
        target_aperture = CircularAperture(target_position, r=aperture_radius)
        target_annulus = CircularAnnulus(target_position, *annulus_radii)

        comparison_aperture = [CircularAperture(pos1, r=aperture_radius) for pos1 in comparison_positions]
        comparison_annulus = [CircularAnnulus(pos2, *annulus_radii) for pos2 in comparison_positions]
        target_phot_table = aperture_photometry(image_data, target_aperture)
        # comparison_phot_table = aperture_photometry(image_data, comparison_aperture)

        if icount == 0:
            im_plot(image_data, target_aperture, comparison_aperture, target_annulus, comparison_annulus)
            # Create a figure and axis
            _, ax = plt.subplots(figsize=(11, 8))

        comparison_phot_table = []
        for comp_aperture, comp_annulus in zip(comparison_aperture, comparison_annulus):
            # Perform aperture photometry on the star
            aperture_phot_table = aperture_photometry(image_data, comp_aperture)

            # Perform aperture photometry on the annulus (background)
            annulus_phot_table = aperture_photometry(image_data, comp_annulus)

            # Store the result in the comparison_phot_table list
            comparison_phot_table.append((aperture_phot_table, annulus_phot_table))

        # Perform annulus photometry to estimate the background
        target_bkg_mean = ApertureStats(image_data, target_annulus).mean
        # comparison_bkg_mean = ApertureStats(image_data, comparison_annulus).mean

        # Calculate the total background for the comparison stars
        comparison_bkg_mean = []
        for annulus in comparison_annulus:
            stats = ApertureStats(image_data, annulus)
            if np.isnan(stats.mean) or np.isinf(stats.mean):
                comparison_bkg_mean.append(0)
            else:
                comparison_bkg_mean.append(stats.mean)

        # Calculate the total background for the target star
        if np.isnan(target_bkg_mean) or np.isinf(target_bkg_mean):
            target_bkg_mean = 0

        # Multiply the background mean by the aperture area to get the total background
        target_bkg = target_bkg_mean * target_aperture.area
        comparison_bkg = [bkg_mean * aperture.area for bkg_mean, aperture in
                          zip(comparison_bkg_mean, comparison_aperture)]

        # target_bkg = ApertureStats(image_data, target_aperture, local_bkg=target_bkg_mean).sum
        # # comparison_bkg = ApertureStats(image_data, comparison_aperture, local_bkg=comparison_bkg_mean).sum
        #
        # comparison_bkg = []
        # for aperture, bkg_mean in zip(comparison_aperture, comparison_bkg_mean):
        #     stats = ApertureStats(image_data, aperture, local_bkg=bkg_mean)
        #     comparison_bkg.append(stats.sum)

        # Calculate the background subtracted counts
        target_flx = target_phot_table['aperture_sum'] - target_bkg
        target_flux_err = np.sqrt(target_phot_table['aperture_sum'] + target_aperture.area * read_noise**2)
        # comparison_flx = comparison_phot_table['aperture_sum'] - comparison_bkg
        # comp_flux_err = np.sqrt(comparison_phot_table['aperture_sum'] + comparison_aperture.area * read_noise ** 2)
        comparison_flx = [phot_table[0]['aperture_sum'] - bkg
                          for phot_table, bkg in zip(comparison_phot_table, comparison_bkg)]

        comp_flux_err = [np.sqrt(phot_table[0]['aperture_sum'] + aperture.area * read_noise ** 2)
                         for phot_table, aperture in zip(comparison_phot_table, comparison_aperture)]
        comp_flux_err = np.array(comp_flux_err)

        # calculate the relative flux for each comparison star and the target star
        # rel_flx_T1 = target_flx / sum(comparison_flx)
        count = 0
        rel_flux_comps = []
        for i in comparison_flx:
            if i == comparison_flx[count]:
                rel_flux_c = i / (sum(comparison_flx) - i)
                rel_flux_comps.append(rel_flux_c)
            count += 1

        # rel_flux_comps = np.array(rel_flux_comps)

        # find the number of pixels used to estimate the sky background
        # n_b = (np.pi * annulus_radii[1]**2) - (np.pi * annulus_radii[0] ** 2)  # main equation
        # n_b_mask_comp = comparison_annulus.to_mask(method="center")
        # n_b_comp = np.sum(n_b_mask_comp)

        # n_b_mask_tar = target_annulus.to_mask(method="center")
        # n_b_tar = np.sum(n_b_mask_tar.data)

        """
        # find the number of pixels used in the aperture if the radius of the apertures is in arcseconds not pixels
        focal_length = 4114  # mm
        pixel_size = 9  # microns
        pixel_size = pixel_size * 10 ** -3  # mm
        ap_area = np.pi * aperture_radius.area**2  # area of the aperture in mm^2
        plate_scale = 1/focal_length  # rad/mm
        plate_scale = plate_scale * 206265  # arcsec/mm
        n_pix = ap_area / (plate_scale * pixel_size)**2  # number of pixels in the aperture
        """
        """
        n_pix = np.pi * aperture_radius**2  # number of pixels in the aperture

        # Calculate the total noise
        sigma_f = 0.289  # quoted from Collins 2017 https://iopscience.iop.org/article/10.3847/1538-3881/153/2/77/pdf
        F_s = 0.01  # number of sky background counts per pixel in ADU

        # N_comp = np.sqrt(gain * comparison_flx + n_pix * (1 + (n_pix / n_b)) *
        #                  (gain * F_s + F_dark + read_noise ** 2 + gain ** 2 + sigma_f ** 2)) / gain
        N_comp = [np.sqrt(gain * flx + n_pix * (1 + (n_pix / n_b)) *
                          (gain * F_s + F_dark + read_noise ** 2 + gain ** 2 + sigma_f ** 2)) / gain
                  for flx in comparison_flx]
        N_tar = np.sqrt(gain * target_flx + n_pix * (1 + (n_pix / n_b)) *
                        (gain * F_s + F_dark + read_noise ** 2 + gain ** 2 + sigma_f ** 2)) / gain

        # calculate the total comparison ensemble noise
        N_e_comp = np.sqrt(np.sum(np.array(N_comp) ** 2))

        rel_flux_err = (rel_flx_T1/rel_flux_comps)*np.sqrt((N_tar**2/target_flx**2) +
                                                          (N_e_comp**2/sum(comparison_flx)**2))
        """
        # calculate the total target magnitude and error
        target_magnitude = (-np.log(sum(2.512**-magnitudes_comp))/np.log(2.512)) - \
                           (2.5*np.log10(target_flx/sum(comparison_flx)))

        target_magnitude_error = 2.5*np.log10(1 + np.sqrt(((target_flux_err**2)/(target_flx**2)) +
                                                          (sum(comp_flux_err**2)/sum(comparison_flx)**2)))

        # comparison_magnitude = -(2.5*np.log10(target_flx/sum(comparison_flx)))

        # Append the calculated magnitude and error to the lists
        magnitudes.append(target_magnitude.value[0])
        mag_err.append(target_magnitude_error.value[0])

    # Plot the magnitudes with error bars
    # noinspection PyUnboundLocalVariable
    ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o', label="Source_AMag_T1")
    # ax.scatter(hjd, magnitudes, marker='o', color='black')

    # Set the labels and parameters
    fontsize = 14
    ax.set_xlabel('HJD', fontsize=fontsize)
    ax.set_ylabel('Source_AMag_T1', fontsize=fontsize)
    ax.invert_yaxis()
    ax.grid()

    ax.legend(loc="upper right", fontsize=fontsize).set_draggable(True)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.show()

    light_curve_data = pd.DataFrame({
        'HJD': hjd,
        'BJD': bjd,
        'Source_AMag_T1': magnitudes,
        'Source_AMag_T1_Error': mag_err
    })

    if not pipeline:
        output_file = input("Enter an output file name and location for the final light curve data in the {} filter "
                            "(ex: C:\\folder1\\folder2\\APASS_254037_B.txt): ".format(filter))
    else:
        output_file = path + "//APASS_254037_" + filter + "_LC_dat.txt"

    light_curve_data.to_csv(output_file, index=False)


def im_plot(image_data, target_aperture, comparison_apertures, target_annulus, comparison_annuli):
    """
    Plot the image with the apertures and annuli overlaid

    Parameters
    ----------
    image_data: array
        Pixel data from the image
    target_aperture: CircularAperture object
        Target aperture location
    comparison_apertures: list of CircularAperture objects
        Comparison aperture locations
    target_annulus: CircularAnnulus object
        Target annulus location
    comparison_annuli: list of CircularAnnulus objects
        Comparison annulus locations

    Returns
    -------
    None
    """
    # First, plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data, cmap='gray', origin='lower', vmin=np.percentile(image_data, 5),
               vmax=np.percentile(image_data, 95))
    plt.colorbar(label='Counts')

    # Now plot the apertures
    lw = 1.5  # line width
    alpha = 1  # line opacity
    target_aperture.plot(color='darkgreen', lw=lw, alpha=alpha)
    for comparison_aperture in comparison_apertures:
        comparison_aperture.plot(color='red', lw=lw, alpha=alpha)

    # Now plot the annuli
    target_annulus.plot(color='darkgreen', lw=lw, alpha=alpha)
    for comparison_annulus in comparison_annuli:
        comparison_annulus.plot(color='red', lw=lw, alpha=alpha)

    plt.pause(1)
    plt.show()


if __name__ == '__main__':
    main()
