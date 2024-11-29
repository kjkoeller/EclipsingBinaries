"""
Analyze images using aperture photometry within Python and not with Astro ImageJ (AIJ)

Author: Kyle Koeller
Created: 05/07/2023
Last Updated: 07/20/2024
"""

# Python imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from pathlib import Path

# from .vseq_updated import io
from vseq_updated import io # testing purposes

# Astropy imports
import ccdproc as ccdp
from astropy.coordinates import SkyCoord
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.wcs import WCS
import astropy.units as u
from astropy import wcs

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
    #filt_list = ["Bessel B", "Bessel V", "Bessel R"]
    if not pipeline:
        prompt_message = "Please enter a file pathway (i.e. C:\\folder1\\folder2\\[reduced]) to where the reduced images are or type 'Close' to leave: "
        images_path = io.validate_directory_path(prompt_message)

        science_imagetyp = 'LIGHT'
        files = ccdp.ImageFileCollection(images_path)

        for filt in filt_list:
            if "/B" in filt:
                prompt_message = "Enter the file location for the RADEC file for the B filter: "
                radec_file = io.validate_file_path(prompt_message)
            elif "/V" in filt:
                prompt_message = "Enter the file location for the RADEC file for the V filter: "
                radec_file = io.validate_file_path(prompt_message)
            elif "/R" in filt:
                prompt_message = "Enter the file location for the RADEC file for the R filter: "
                radec_file = io.validate_file_path(prompt_message)

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


def load_radec_file(radec_file, df):
    """
    Load RADEC file either from input or DataFrame.

    Parameters:
        radec_file (str): Location of the RADEC file.
        df (DataFrame): DataFrame to populate if pipeline mode is True.

    Returns:
        DataFrame: DataFrame containing RADEC data.
    """
    if not df.empty:
        return df

    while True:
        try:
            df = pd.read_csv(radec_file, skiprows=7, sep=",", header=None)
            break
        except FileNotFoundError:
            print("File not found. Please try again.")
            radec_file = input("Please enter the RADEC file (i.e. C://folder1//folder2//[file name]: ")

    return df


def process_image(path, image_file, ra, dec, aperture_radius, annulus_radii, hjd, bjd):
    """
    Process an image to extract target and comparison positions, and perform aperture photometry.

    Parameters:
        path (str): Path to the folder containing the images.
        image_file (str): Name of the image file.
        ra (array-like): Right ascension coordinates.
        dec (array-like): Declination coordinates.
        aperture_radius (float): Radius of the aperture.
        annulus_radii (tuple): Radii for the annulus.
        hjd (list): List to append HJD values.
        bjd (list): List to append BJD values.

    Returns:
        tuple: Tuple containing the target aperture photometry table and image header.
    """
    image_data, header = fits.getdata(path / image_file, header=True)
    wcs_ = WCS(header)

    sky_coords = SkyCoord(ra, dec, unit=(u.h, u.deg), frame='icrs')
    pixel_coords = wcs_.world_to_pixel(sky_coords)
    x_coords, y_coords = pixel_coords

    target_position = (x_coords[0], y_coords[0])
    comparison_positions = list(zip(x_coords[1:], y_coords[1:]))

    hjd.append(header['HJD-OBS'])
    bjd.append(header['BJD-OBS'])

    target_aperture = CircularAperture(target_position, r=aperture_radius)
    target_annulus = CircularAnnulus(target_position, *annulus_radii)

    comparison_aperture = [CircularAperture(pos1, r=aperture_radius) for pos1 in comparison_positions]
    comparison_annulus = [CircularAnnulus(pos2, *annulus_radii) for pos2 in comparison_positions]

    target_phot_table = aperture_photometry(image_data, target_aperture)

    return image_data, target_phot_table, target_annulus, target_aperture, comparison_annulus, comparison_aperture


def calculate_background_and_flux(image_data, target_bkg_mean, comparison_annulus, comparison_aperture,
                                  target_aperture, target_phot_table, comparison_phot_table, read_noise):
    """
    Calculate background and flux for target and comparison stars.

    Parameters:
        target_bkg_mean (float): Mean background for the target star.
        comparison_annulus (list): List of CircularAnnulus objects for comparison stars.
        comparison_aperture (list): List of CircularAperture objects for comparison stars.
        target_aperture (CircularAperture): CircularAperture object for the target star.
        target_phot_table (QTable): Photometry table for the target star.
        comparison_phot_table (list): List of photometry tables for comparison stars.
        read_noise (float): Read noise value.

    Returns:
        tuple: Tuple containing background subtracted counts for the target star (target_flx),
               target flux error (target_flux_err), and background subtracted counts for comparison stars (comparison_flx).
    """
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

    # Calculate the background subtracted counts
    target_flx = target_phot_table['aperture_sum'] - target_bkg
    target_flux_err = np.sqrt(target_phot_table['aperture_sum'] + target_aperture.area * read_noise ** 2)
    comparison_flx = [phot_table[0]['aperture_sum'] - bkg
                      for phot_table, bkg in zip(comparison_phot_table, comparison_bkg)]

    comp_flux_err = [np.sqrt(phot_table[0]['aperture_sum'] + aperture.area * read_noise ** 2)
                     for phot_table, aperture in zip(comparison_phot_table, comparison_aperture)]
    comp_flux_err = np.array(comp_flux_err)

    return target_flx, target_flux_err, comparison_flx, comp_flux_err


def calculate_magnitude_and_error(comparison_flx, magnitudes_comp, target_flx, target_flux_err, comp_flux_err):
    """
    Calculate relative flux, target magnitude, and target magnitude error.

    Parameters:
        comparison_flx (list): List of background subtracted counts for comparison stars.
        magnitudes_comp (pd.Series): Series containing magnitudes of comparison stars.
        target_flx (float): Background subtracted counts for the target star.
        target_flux_err (float): Flux error for the target star.
        comp_flux_err (np.array): Array containing flux errors for comparison stars.

    Returns:
        tuple: Tuple containing relative flux for each comparison star (rel_flux_comps),
               target magnitude (target_magnitude), and target magnitude error (target_magnitude_error).
    """
    # Calculate the relative flux for each comparison star and the target star
    # rel_flux_comps = [comp_flux / (sum(comparison_flx) - comp_flux) for comp_flux in comparison_flx]

    # Calculate the total target magnitude
    target_magnitude = (-np.log(sum(2.512 ** -magnitudes_comp)) / np.log(2.512)) - \
                       (2.5 * np.log10(target_flx / sum(comparison_flx)))

    # Calculate the target magnitude error
    target_magnitude_error = 2.5 * np.log10(1 + np.sqrt(((target_flux_err ** 2) / (target_flx ** 2)) +
                                                        (sum(comp_flux_err ** 2) / sum(comparison_flx) ** 2)))

    return target_magnitude, target_magnitude_error


def multiple_AP(image_list, path, filter_list, pipeline=False, radec_file="", df=pd.DataFrame({})):
    """
    Perform multi-aperture photometry on a list of images for a single target

    Parameters
    ----------
    filter_list: String
        Filter used for the images
    pipeline: Boolean
        If True, then the program is being run from the pipeline and will not ask for user input.
    radec_file: string
        Location of a radec file. If not given, the user will be prompted to enter one.
    path : pathway
        Path to the folder containing the images.
    image_list : List
        Images to perform multi-aperture photometry on.
    df : DataFrame
        DataFrame containing the RA, DEC, and magnitudes of the target and comparison stars.

    Returns
    -------
    None
    :param df:
    """
    aperture_radius = 20
    annulus_radii = (30, 50)
    read_noise = 10.83  # Example value

    df = load_radec_file(radec_file, df)

    magnitudes_comp = df[4].replace(99.999, pd.NA).dropna().reset_index(drop=True)
    ra = df[0]
    dec = df[1]

    magnitudes = []
    mag_err = []
    hjd = []
    bjd = []

    _, ax = plt.subplots(figsize=(11, 8))

    def are_apertures_valid(image_shape, apertures):
        """
        Check if apertures are within the boundaries of the image.

        Parameters
        ----------
        image_shape : tuple
            A tuple representing the shape of the image (height, width).
        apertures : list
            A list of aperture objects, each with positions and radius.

        Returns
        -------
        valid : list
            A list of booleans indicating whether each aperture is valid (True) or not (False).
        """
        # Unpack the image dimensions
        image_height, image_width = image_shape

        # Initialize a list to store validity of each aperture
        valid = []

        # Iterate through each aperture to check its validity
        for aperture in apertures:
            # Get the x, y positions and radius of the aperture
            x, y, r = aperture.positions[0], aperture.positions[1], aperture.r

            # Check if the aperture is within the image boundaries
            if x - r >= 0 and x + r < image_width and y - r >= 0 and y + r < image_height:
                valid.append(True)
            else:
                valid.append(False)

        # Return the list of validity flags
        return valid

    for _, image_file in tqdm(enumerate(image_list), desc="Performing aperture photometry on {} images in the {} filter.".format(len(image_list), filter)):
        [image_data, target_phot_table, target_annulus, target_aperture, comparison_annulus, comparison_aperture] = (
            process_image(path, image_file, ra, dec, aperture_radius, annulus_radii, hjd, bjd))

        valid_apertures = are_apertures_valid(image_data.shape, comparison_aperture)

        while not all(valid_apertures):
            comparison_aperture = [ap for ap, valid in zip(comparison_aperture, valid_apertures) if valid]
            comparison_annulus = [an for an, valid in zip(comparison_annulus, valid_apertures) if valid]
            valid_apertures = are_apertures_valid(image_data.shape, comparison_aperture)

        comparison_phot_table = []
        for comp_aperture, comp_annulus in zip(comparison_aperture, comparison_annulus):
            aperture_phot_table = aperture_photometry(image_data, comp_aperture)
            annulus_phot_table = aperture_photometry(image_data, comp_annulus)
            comparison_phot_table.append((aperture_phot_table, annulus_phot_table))

        target_bkg_mean = ApertureStats(image_data, target_annulus).mean

        [target_flx, target_flux_err, comparison_flx, comp_flux_err] = (
            calculate_background_and_flux(image_data, target_bkg_mean, comparison_annulus, comparison_aperture, target_aperture,
                                          target_phot_table, comparison_phot_table, read_noise))

        [target_magnitude, target_magnitude_error] = (
            calculate_magnitude_and_error(comparison_flx, magnitudes_comp, target_flx, target_flux_err, comp_flux_err))

        magnitudes.append(target_magnitude.value[0])
        mag_err.append(target_magnitude_error.value[0])

    ax.errorbar(hjd, magnitudes, yerr=mag_err, fmt='o', label="Source_AMag_T1")
    ax.set_xlabel('HJD', fontsize=14)
    ax.set_ylabel('Source_AMag_T1', fontsize=14)
    ax.invert_yaxis()
    ax.grid()
    ax.legend(loc="upper right", fontsize=14).set_draggable(True)
    ax.tick_params(axis='both', which='major', labelsize=14)

    filter_sanitized = filter_list.replace("Empty/", "")

    if not pipeline:
        plt.show()
    else:
        plt.savefig(f"{path}/APASS_254037_{filter_sanitized}_LC_dat.jpg")

    light_curve_data = pd.DataFrame({
        'HJD': hjd,
        'BJD': bjd,
        'Source_AMag_T1': magnitudes,
        'Source_AMag_T1_Error': mag_err
    })

    im_plot(image_data, target_aperture, comparison_aperture, target_annulus, comparison_annulus)

    if not pipeline:
        output_file = input(f"Enter an output file name and location for the final light curve data in the {filter_sanitized} filter (ex: C:\\folder1\\folder2\\APASS_254037_B.txt): ")
    else:
        output_file = f"{path}/APASS_254037_{filter_sanitized}_LC_dat.txt"

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


# D:\Research\Data\NSVS_254037\2018.09.18-reduced\NSVS_254037-B.radec

if __name__ == '__main__':
    main()

