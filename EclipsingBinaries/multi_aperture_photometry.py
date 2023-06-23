"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 06/23/2023
"""

# Python imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
import warnings

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


def main(path="", pipeline=False):
    """
    Main function for aperture photometry

    Parameters
    ----------
    path : str
        Path to the folder containing the images.

    pipeline : bool
        If True, then the program is being run from the pipeline and will not ask for user input.
    Returns
    -------
    N/A
    """
    if not pipeline:
        # path = input("Please enter a file pathway (i.e. C:\\folder1\\folder2\\[raw]) to where the reduced images are or type "
        #              "the word 'Close' to leave: ")

        path = "D:\\BSUO data\\2022.09.29-reduced"  # For testing purposes

        if path.lower() == "close":
            exit()
    else:
        pass

    science_imagetyp = 'LIGHT'

    images_path = Path(path)
    files = ccdp.ImageFileCollection(images_path)
    image_list = files.files_filtered(imagetyp=science_imagetyp, filter="Empty/V")

    multiple_AP(image_list, images_path)


def single_AP(image_list, path):
    """
    Perform multi-aperture photometry on a list of images for a single target

    Parameters
    ----------
    path : path
        Path to the folder containing the images.
    image_list : table
        Table of images to perform aperture photometry on.

    Returns
    -------
    None
    """

    # Define the aperture parameters
    # Define the aperture and annulus radii
    aperture_radius = 20
    annulus_radii = (30, 50)

    read_noise = 10.83  # * u.electron  # gathered from fits headers manually
    gain = 1.43  # * u.electron / u.adu  # gathered from fits headers manually
    F_dark = 0.01  # dark current in u.electron / u.pix / u.s

    # Magnitudes of the comparison stars (replace with your values)
    df = pd.read_csv('254037_B.radec', skiprows=7, sep=",", header=None)
    magnitudes_comp = df[4]
    ra = df[0]
    dec = df[1]
    ref_star = df[2]
    # centroid = df[3]  # Not used (I don't think at least)

    magnitudes = []
    mag_err = []
    hjd = []
    bjd = []

    # Start interactive mode
    plt.ion()

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.show()

    for icount, image_file in enumerate(image_list):
        image_data, header = fits.getdata(path / image_file, header=True)
        wcs_ = WCS(header)

        # ccd = CCDData(image_data, wcs=wcs, unit='adu')

        # Convert RA and DEC to pixel positions
        sky_coords = SkyCoord(ra, dec, unit=(u.h, u.deg), frame='icrs')
        pixel_coords = wcs_.world_to_pixel(sky_coords)

        target_position = list(pixel_coords[0])
        comparison_positions = list(pixel_coords[1:])

        hjd.append(header['HJD-OBS'])
        bjd.append(header['BJD-OBS'])

        # Create the apertures and annuli
        target_aperture = CircularAperture(target_position, r=aperture_radius)
        target_annulus = CircularAnnulus(target_position, *annulus_radii)

        target_phot_table = aperture_photometry(image_data, target_aperture)

        # Perform annulus photometry to estimate the background
        target_bkg_mean = ApertureStats(image_data, target_annulus).mean

        # Calculate the total background
        if np.isnan(target_bkg_mean) or np.isinf(target_bkg_mean):
            target_bkg_mean = 0

        target_bkg = ApertureStats(image_data, target_aperture, local_bkg=target_bkg_mean).sum

        # Calculate the background subtracted counts
        target_flx = target_phot_table['aperture_sum'] - target_bkg

        target_flx_err = np.sqrt(target_phot_table['aperture_sum'])

        target_magnitude = 25 - 2.5 * np.log10(target_flx)
        target_magnitude_error = (2.5 / np.log(10)) * (target_flx_err / target_flx)

        # Append the calculated magnitude and error to the lists
        magnitudes.append(target_magnitude[0])
        mag_err.append(target_magnitude_error[0])

        # Clear the axis
        ax.clear()

        # Plot the magnitudes with error bars
        # ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o')
        ax.scatter(hjd, magnitudes, marker='o', color='black')

        # Set the labels
        ax.set_xlabel('HJD')
        ax.set_ylabel('Source_AMag_T1')

        # Draw the figure
        fig.canvas.draw()

        # Pause for a bit to allow the figure to update
        time.sleep(0.1)
        plt.pause(0.0001)

    # Disable interactive mode
    plt.ioff()


def multiple_AP(image_list, path):
    """
    Perform multi-aperture photometry on a list of images for a single target

    Parameters
    ----------
    path : path
        Path to the folder containing the images.
    image_list : table
        Table of images to perform aperture photometry on.

    Returns
    -------
    None
    """

    # Define the aperture parameters
    # Define the aperture and annulus radii
    aperture_radius = 20
    annulus_radii = (30, 50)

    read_noise = 10.83  # * u.electron  # gathered from fits headers manually
    gain = 1.43  # * u.electron / u.adu  # gathered from fits headers manually
    F_dark = 0.01  # dark current in u.electron / u.pix / u.s

    # Magnitudes of the comparison stars (replace with your values)
    df = pd.read_csv('254037_B.radec', skiprows=7, sep=",", header=None)
    magnitudes_comp = df[4]
    ra = df[0]
    dec = df[1]
    # ref_star = df[2]
    # centroid = df[3]  # Not used (I don't think at least)

    magnitudes = []
    mag_err = []
    hjd = []
    bjd = []

    # Start interactive mode
    plt.ion()

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.show()

    for icount, image_file in enumerate(image_list):
        image_data, header = fits.getdata(path / image_file, header=True)
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

        im_plot(image_data, target_aperture, comparison_aperture, target_annulus, comparison_annulus)

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

        target_bkg = ApertureStats(image_data, target_aperture, local_bkg=target_bkg_mean).sum
        # comparison_bkg = ApertureStats(image_data, comparison_aperture, local_bkg=comparison_bkg_mean).sum

        comparison_bkg = []
        for aperture, bkg_mean in zip(comparison_aperture, comparison_bkg_mean):
            stats = ApertureStats(image_data, aperture, local_bkg=bkg_mean)
            comparison_bkg.append(stats.sum)

        # Calculate the background subtracted counts
        target_flx = target_phot_table['aperture_sum'] - target_bkg
        target_flux_err = np.sqrt(target_phot_table['aperture_sum'] + target_aperture.area * read_noise**2)
        # comparison_flx = comparison_phot_table['aperture_sum'] - comparison_bkg
        # comp_flux_err = np.sqrt(comparison_phot_table['aperture_sum'] + comparison_aperture.area * read_noise ** 2)
        comparison_flx = [phot_table[0]['aperture_sum'] - bkg
                          for phot_table, bkg in zip(comparison_phot_table, comparison_bkg)]

        comp_flux_err = [np.sqrt(phot_table[0]['aperture_sum'] + aperture.area * read_noise ** 2)
                         for phot_table, aperture in zip(comparison_phot_table, comparison_aperture)]

        # calculate the relative flux for each comparison star and the target star
        rel_flx_T1 = target_flx / sum(comparison_flx)
        count = 0
        for i in comparison_flx:
            if i == comparison_flx[count]:
                rel_flux_comp = i / (sum(comparison_flx) - i)
            count += 1

        # find the number of pixels used to estimate the sky background
        n_b = (np.pi * annulus_radii[1]**2) - (np.pi * annulus_radii[0] ** 2)
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

        rel_flux_err = (rel_flx_T1/rel_flux_comp)*np.sqrt((N_tar**2/target_flx**2) +
                                                          (N_e_comp**2/sum(comparison_flx)**2))

        # calculate the total target magnitude and error
        target_magnitude = -np.log(sum(2.512**(magnitudes_comp)))/np.log(2.512) - \
                           (2.5*np.log10(target_bkg/sum(comparison_bkg)))
        target_magnitude_error = 2.5*np.log10(1 + np.sqrt(target_flux_err**2/target_bkg**2) +
                                              (sum(comp_flux_err)**2/sum(comparison_bkg)**2))

        # Append the calculated magnitude and error to the lists
        magnitudes.append(target_magnitude)
        mag_err.append(target_magnitude_error)

        # Clear the axis
        ax.clear()

        # Plot the magnitudes with error bars
        # ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o')
        ax.scatter(hjd, magnitudes, marker='o', color='black')

        # Set the labels
        ax.set_xlabel('HJD')
        ax.set_ylabel('Source_AMag_T1')

        # Draw the figure
        fig.canvas.draw()

        # Pause for a bit to allow the figure to update
        time.sleep(0.1)
        plt.pause(0.0001)

    # Disable interactive mode
    plt.ioff()
    plt.show()


def im_plot(image_data, target_aperture, comparison_apertures, target_annulus, comparison_annuli):
    # First, plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray', origin='lower', vmin=np.percentile(image_data, 5),
               vmax=np.percentile(image_data, 95))
    plt.colorbar(label='Counts')

    # Now plot the apertures
    target_aperture.plot(color='blue', lw=1.5, alpha=0.5)
    for comparison_aperture in comparison_apertures:
        comparison_aperture.plot(color='red', lw=1.5, alpha=0.5)

    # Now plot the annuli
    target_annulus.plot(color='blue', lw=1.5, alpha=0.5, linestyle='dashed')
    for comparison_annulus in comparison_annuli:
        comparison_annulus.plot(color='red', lw=1.5, alpha=0.5, linestyle='dashed')

    plt.pause(1000)
    plt.show()


if __name__ == '__main__':
    main()
