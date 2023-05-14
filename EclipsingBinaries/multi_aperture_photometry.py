"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 05/14/2023
"""

# Python imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Astropy imports
import ccdproc as ccdp
from astropy.coordinates import SkyCoord
from astropy.io import fits
# from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.wcs import WCS


def main():
    path = input("Please enter a file pathway (i.e. C:\\folder1\\folder2\\[raw]) to where the reduced images are or type "
                 "the word 'Close' to leave: ")

    if path.lower() == "close":
        exit()

    science_imagetyp = 'LIGHT'
    images_path = Path(path)
    files = ccdp.ImageFileCollection(images_path)
    image_list = files.ccds(imagetyp=science_imagetyp, return_fname=True)

    multi_aperture_photometry(image_list)


def multi_aperture_photometry(image_list):
    """
    Perform aperture photometry on a list of images.

    Parameters
    ----------
    image_list : list
        List of images to perform aperture photometry on.

    Returns
    -------
    None
    """

    # Define the aperture parameters
    # Define the aperture and annulus radii
    aperture_radius = 25
    annulus_radii = (40, 60)

    read_noise = 10.83  # * u.electron  # gathered from fits headers manually
    gain = 1.43  # * u.electron / u.adu  # gathered from fits headers manually

    # Magnitudes of the comparison stars (replace with your values)
    df = pd.read_csv('254037_B.radec', skiprows=1, delim_whitespace=True)
    magnitudes_comp = df[4]
    ra = df[0]
    dec = df[1]
    ref_star = df[2]
    # centroid = df[3]  # Not used (I don't think at least)

    # Convert RA and DEC to pixel positions
    sky_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    pixel_coords = WCS.all_world2pix(sky_coords.ra, sky_coords.dec, 1)

    target_position = pixel_coords[0]
    comparison_positions = pixel_coords[1:]

    magnitudes = []
    mag_err = []
    hjd = []
    bjd = []

    # Start interactive mode
    plt.ion()

    # Create a figure and axis
    fig, ax = plt.subplots()

    for _, image_file in enumerate(image_list):
        image_data, header = fits.getdata(image_file, header=True)

        hjd.append(header['HJD'])
        bjd.append(header['BJD'])

        # Create the apertures and annuli
        target_aperture = CircularAperture(target_position, r=aperture_radius)
        comparison_apertures = CircularAperture(comparison_positions, r=aperture_radius)
        target_annulus = CircularAnnulus(target_position, *annulus_radii)
        comparison_annuli = CircularAnnulus(comparison_positions, *annulus_radii)

        # Perform aperture photometry
        target_phot_table = aperture_photometry(image_data, target_aperture)
        comparison_phot_tables = aperture_photometry(image_data, comparison_apertures)

        # Perform annulus photometry to estimate the background
        target_annulus_table = aperture_photometry(image_data, target_annulus)
        comparison_annuli_tables = aperture_photometry(image_data, comparison_annuli)

        # Calculate the background mean and standard deviation
        target_background_mean = target_annulus_table['aperture_sum'][0] / target_annulus.area()
        target_background_stddev = np.sqrt(target_background_mean)

        comparison_background_means = [table['aperture_sum'][0] / annulus.area()
                                       for table, annulus in zip(comparison_annuli_tables, comparison_annuli)]
        comparison_background_stddevs = [np.sqrt(mean) for mean in comparison_background_means]

        # Subtract the background: Calculate net integrated counts
        target_sum = target_phot_table['aperture_sum'][0] - target_aperture.area() * target_background_mean
        comparison_sums = [table['aperture_sum'][0] - aperture.area() * mean
                           for table, aperture, mean in
                           zip(comparison_phot_tables, comparison_apertures, comparison_background_means)]

        # Calculate the errors
        target_error = np.sqrt(
            target_sum / gain + target_aperture.area() * target_background_stddev ** 2 + target_aperture.area() ** 2 * read_noise ** 2)
        comparison_errors = [
            np.sqrt(comp_sums / gain + aperture.area() * stddev ** 2 + aperture.area() ** 2 * read_noise ** 2)
            for comp_sums, aperture, stddev in zip(comparison_sums, comparison_apertures, comparison_background_stddevs)]

        # Calculate the magnitude and error of the target star
        target_magnitude = -2.5 * np.log10(target_sum / np.mean(comparison_sums))

        # Calculate the error in the magnitude
        target_magnitude_error = (2.5 / np.log(10)) * np.sqrt((target_error / target_sum) ** 2 + np.mean(
            [error ** 2 / star_sum ** 2 for error, star_sum in zip(comparison_errors, comparison_sums)]) / len(comparison_sums))

        print('Target magnitude: ', target_magnitude, '+/-', target_magnitude_error)

        # Append the calculated magnitude and error to the lists
        magnitudes.append(target_magnitude)
        mag_err.append(target_magnitude_error)

        # Clear the axis
        ax.clear()

        # Plot the magnitudes with error bars
        ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o')

        # Set the labels
        ax.set_xlabel('HJD')
        ax.set_ylabel('Source_AMag_T1')

        # Draw the figure
        fig.canvas.draw()

        # Pause for a bit to allow the figure to update
        time.sleep(0.1)

    # Disable interactive mode
    plt.ioff()

    # Show the final figure
    plt.show()


if __name__ == '__main__':
    main()
