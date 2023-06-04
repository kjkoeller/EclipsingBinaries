"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 06/03/2023
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


def main():
    # path = input("Please enter a file pathway (i.e. C:\\folder1\\folder2\\[raw]) to where the reduced images are or type "
    #             "the word 'Close' to leave: ")

    path = "D:\\BSUO data\\2022.09.23-reduced"  # For testing purposes

    if path.lower() == "close":
        exit()

    science_imagetyp = 'LIGHT'

    images_path = Path(path)
    files = ccdp.ImageFileCollection(images_path)
    image_list = files.files_filtered(imagetyp=science_imagetyp, filter="Empty/V")
    print(type(image_list))

    multi_aperture_photometry(image_list, images_path)


def multi_aperture_photometry(image_list, path):
    """
    Perform aperture photometry on a list of images.

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

    r_in, r_out = annulus_radii

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
        comparison_apertures = CircularAperture(comparison_positions, r=aperture_radius)
        target_annulus = CircularAnnulus(target_position, *annulus_radii)
        comparison_annuli = CircularAnnulus(comparison_positions, *annulus_radii)

        target_phot_table = aperture_photometry(image_data, target_aperture)
        comparison_phot_tables = aperture_photometry(image_data, comparison_apertures)

        # Perform annulus photometry to estimate the background
        target_bkg_mean = ApertureStats(image_data, target_annulus).mean
        comparison_bkg_mean = ApertureStats(image_data, comparison_annuli).mean

        # Calculate the total background
        if np.isnan(target_bkg_mean) or np.isinf(target_bkg_mean) or np.isnan(comparison_bkg_mean) or np.isinf(comparison_bkg_mean):
            target_bkg_mean = 0
            comparison_bkg_mean = 0

        target_bkg = ApertureStats(image_data, target_aperture, local_bkg=target_bkg_mean).sum
        comparison_bkg = ApertureStats(image_data, comparison_apertures, local_bkg=comparison_bkg_mean).sum
        # target_bkg = target_bkg_mean * target_area
        # comparison_bkg = comparison_bkg_mean * comparison_area

        # Calculate the background subtracted counts
        target_flx = target_phot_table['aperture_sum'] - target_bkg
        comparisons_flx = comparison_phot_tables['aperture_sum'] - comparison_bkg

        target_flx_err = np.sqrt(target_phot_table['aperture_sum'])

        target_magnitude = 25 - 2.5*np.log10(target_flx)
        target_magnitude_error = (2.5/np.log(10)) * (target_flx_err/target_flx)

        # target_magnitude = (np.log(sum(2.512**(-magnitudes_comp)))/np.log(2.512)) + 2.5*np.log10(target_sum/comparison_sums)
        # target_flux_rel = target_sum / comparison_sums

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
        time.sleep(0.2)

    # Disable interactive mode
    plt.ioff()

    # Show the final figure
    plt.show()


if __name__ == '__main__':
    main()
