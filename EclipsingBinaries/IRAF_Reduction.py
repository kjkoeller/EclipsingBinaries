"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 12/16/2024

This program is meant to automatically do the data reduction of the raw images from the
Ball State University Observatory (BSUO) and SARA data. The new calibrated images are placed into a new folder as to
not overwrite the original images.
"""
# import sys
from pathlib import Path
import time
import warnings

from astropy import wcs
from astropy.stats import mad_std
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

import ccdproc as ccdp
import numpy as np

# turn off this warning that just tells the user,
# "The warning raised when the contents of the FITS header have been modified to be standards compliant."
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)

# global variables for combining, to either play with or set depending on the run
sigma_clip_low_thresh = None
sigma_clip_high_thresh = 3
sigclip = 5  # sigclip for cosmic ray removal
rdnoise = 10.83  # * u.electron  # gathered from fits headers manually
gain = 1.43  # * u.electron / u.adu  # gathered from fits headers manually
overwrite = True  # if the user wants to overwrite already existing files or not, by default it is set to True
mem_limit = 1600e6  # maximum memory limit 4.5 Gb is the recommended which is 450e6 (i.e. 8.0 Gb would be 800e6)
dark_bool = "True"
location = "bsuo"


def run_reduction(path, calibrated, location, cancel_event, dark_bool=True, overscan_region="[2073:2115, :]", trim_region="[20:2060, 12:2057]", write_callback=None):
    """
    Main function to run the reduction process. It replaces user input with programmatic arguments.

        :param path: The path to the raw images directory
    :param calibrated: The path to the output directory for calibrated images
    :param location: The observing location (e.g., 'bsuo', 'kpno', etc.)
    :param cancel_event:
    :param dark_bool: Boolean to indicate whether to use dark frames
    :param overscan_region: String for the overscan region of the CCD
    :param trim_region: String for the trim region of the CCD
    :param write_callback: Function to write log messages to the GUI
    """
    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)

    try:
        if cancel_event.is_set():
            log("Task canceled before starting.")
            return

        # Convert paths to Path objects
        images_path = Path(path)
        calibrated_data = Path(calibrated)

        if not images_path.exists():
            raise FileNotFoundError(f"Raw images path '{path}' does not exist.")
        if not calibrated_data.exists():
            calibrated_data.mkdir(parents=True)

        # Configure location-specific defaults
        if location.lower() == "kpno":
            configure_kpno()
        elif location.lower() == "ctio":
            configure_ctio()
        elif location.lower() == "lapalma":
            configure_lapalma()

        # Process files
        files = ccdp.ImageFileCollection(images_path)
        # zero, overscan_region, trim_region = bias(files, calibrated_data, log)
        zero = bias(files, calibrated_data, overscan_region, trim_region, log, cancel_event)
        master_dark = dark(files, zero, calibrated_data, overscan_region, trim_region, log, cancel_event) if dark_bool else None
        flat(files, zero, master_dark, calibrated_data, overscan_region, trim_region, log, cancel_event)
        science_images(files, calibrated_data, zero, master_dark, trim_region, overscan_region, log, cancel_event)

        log("\nReduction process completed successfully.\n")
    except Exception as e:
        log(f"An error occurred: {e}")
        raise


def kpno():
    """
    Kitt Peak National Observatory default values

    :return: None
    """
    global gain, rdnoise, dark_bool

    gain = 2.3
    rdnoise = 6.0
    dark_bool = True


def ctio():
    """
    Cerro Tololo Inter-American Observatory default values

    :return: None
    """
    global gain, rdnoise, dark_bool

    gain = 2.0
    rdnoise = 9.7
    dark_bool = True


def lapalma():
    """
    La Palma default values

    :return: None
    """
    global gain, rdnoise, dark_bool

    gain = 1.0
    rdnoise = 6.3
    dark_bool = True


def new_default():
    """
    Generates new values that the user can enter

    :return: newly entered default values
    """
    global sigma_clip_high_thresh, sigma_clip_low_thresh, gain, rdnoise, sigclip, dark_bool, location

    sigma_clip_low_thresh = (input("\nEnter a sigma clip low threshold, default is 'None': "))
    if sigma_clip_low_thresh.lower() == "none":
        sigma_clip_low_thresh = None
    else:
        sigma_clip_low_thresh = int(sigma_clip_high_thresh)
    sigma_clip_high_thresh = int(input("Enter a sigma clip high threshold, default is '3': "))
    gain = float(input("Enter a gain value, default is '1.43' electron/adu: "))
    rdnoise = float(input("Enter a readnoise value, default is '10.83' electrons: "))
    sigclip = int(input("Enter a sigma clip value for cosmic ray removal, default is '5': "))
    dark_bool = input("Are you using Dark Frames, default is True (enter 'True' or 'False'): ")
    if dark_bool.lower == "false":
        dark_bool = False
    elif dark_bool.lower == "true":
        dark_bool = True
    location = input("What is your observation location, default is bsuo (ctio, kpno, lapalma")


def reduce(ccd, overscan_region, trim_region, num, zero, combined_dark, good_flat):
    """
    This function takes the information for each section of the reduction process into a singular function for
    limits in duplication of the code.

    :param ccd: individual image
    :param overscan_region: the overscan region of the image
    :param trim_region: region of the image to trim
    :param num: tells the program what stage of the process it is in
    :param zero: master bias
    :param combined_dark: master dark
    :param good_flat: master flat
    :return: depends on the stage of process, but will return a final image for each process
    """

    # subtract overscan, trims image, and gain corrects
    if overscan_region.lower() == "none":
        pass
    else:
        # new_ccd = ccdp.ccd_process(ccd, fits_section=overscan_region, trim=trim_region, gain=gain * u.electron / u.adu,
        #                            readnoise=rdnoise * u.electron, error=False)
        ccd = ccdp.subtract_overscan(ccd, fits_section=overscan_region, median=True, overscan_axis=None)

    trim_ccd = ccdp.trim_image(ccd, fits_section=trim_region)
    new_ccd = ccdp.gain_correct(trim_ccd, gain=gain * u.electron / u.adu)  # gain correct the image

    # this if statement checks whether the input is bias, dark, flat, or science reduction
    # bias combining
    if num == 0:
        return new_ccd
    # dark combining
    elif num == 1:
        # print(new_ccd.unit, zero.unit)
        # Subtract bias
        sub_ccd = ccdp.subtract_bias(new_ccd, zero)

        return sub_ccd
    # flat combining
    elif num == 2:
        # print(new_ccd.unit, zero.unit)
        # Subtract bias
        sub_ccd = ccdp.subtract_bias(new_ccd, zero)

        # Subtract the dark current
        if not dark_bool:
            final_ccd = sub_ccd
        else:
            final_ccd = ccdp.subtract_dark(sub_ccd, combined_dark, exposure_time='exptime', exposure_unit=u.second,
                                           scale=True)
        return final_ccd
    # science calibration
    elif num == 3:
        # Subtract bias
        if location.lower() == "bsuo":
            sub_ccd = ccdp.subtract_bias(new_ccd, zero)
        else:
            sub_ccd = ccdp.subtract_bias(new_ccd, zero)
        # Subtract the dark current
        if not dark_bool:
            reduced = sub_ccd
        else:
            reduced = ccdp.subtract_dark(sub_ccd, combined_dark, exposure_time='exptime', exposure_unit=u.second,
                                         scale=True)
        # flat field correct the science image based on filter
        reduced = ccdp.flat_correct(ccd=reduced, flat=good_flat, min_value=1.0)

        # cosmic ray reject above 5 sigmas and gain_apply is set to false because it changes the units of the image
        # new_reduced = ccdp.cosmicray_lacosmic(reduced, gain_apply=False, readnoise=rdnoise, gain=gain, sigclip=sigclip)

        return reduced


def bias(files, calibrated_data, overscan_region, trim_region, log, cancel_event):
    """
    Calibrates bias images and creates a master bias.

    :param files: Image file collection
    :param calibrated_data: Path to save calibrated images
    :param overscan_region: String for the overscan region of the CCD
    :param trim_region: String for the trim region of the CCD
    :param log: Logging function
    :return: Master bias
    """
    log("\nStarting bias calibration.")
    # Simulate overscan and trim region determination
    # overscan_region = "[2073:2115, :]"
    # trim_region = "[20:2060, 12:2057]"

    log(f"Overscan Region: {overscan_region}")
    log(f"Trim Region: {trim_region}")

    for ccd, file_name in files.ccds(imagetyp='BIAS', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        if cancel_event.is_set():
            log("Task canceled.")
            return
        log(f"Processing bias image: {file_name}")
        new_ccd = reduce(ccd, overscan_region, trim_region, 0, zero=None, combined_dark=None, good_flat=None)
        output_path = calibrated_data / f"{file_name.split('.')[0]}.fits"
        new_ccd.write(output_path, overwrite=overwrite)
        log(f"Saved calibrated bias image: {output_path}")

    log("\nCombining bias frames to create master bias.")
    reduced_images = ccdp.ImageFileCollection(calibrated_data)
    calibrated_biases = reduced_images.files_filtered(imagetyp='BIAS', include_path=True)

    combined_bias = ccdp.combine(
        calibrated_biases,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=sigma_clip_low_thresh,
        sigma_clip_high_thresh=sigma_clip_high_thresh,
        sigma_clip_func=np.ma.median,
        mem_limit=mem_limit
    )
    combined_bias.meta['combined'] = True
    combined_bias_path = calibrated_data / "zero.fits"
    combined_bias.write(combined_bias_path, overwrite=overwrite)
    log(f"Master bias created: {combined_bias_path}")

    return combined_bias # , overscan_region, trim_region


def dark(files, zero, calibrated_path, overscan_region, trim_region, log, cancel_event):
    """
    Calibrates dark frames and creates a master dark.

    :param files: Image file collection
    :param zero: Master bias
    :param calibrated_path: Path to save calibrated images
    :param overscan_region: Overscan region for dark frames
    :param trim_region: Trim region for dark frames
    :param log: Logging function
    :return: Master dark frame
    """
    log("\nStarting dark calibration.")
    for ccd, file_name in files.ccds(imagetyp='DARK', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        if cancel_event.is_set():
            log("Task canceled.")
            return

        log(f"Processing dark image: {file_name}")
        sub_ccd = reduce(ccd, overscan_region, trim_region, 1, zero, combined_dark=None, good_flat=None)
        output_path = calibrated_path / f"{file_name.split('.')[0]}.fits"
        sub_ccd.write(output_path, overwrite=overwrite)
        log(f"Saved calibrated dark image: {output_path}")

    log("\nCombining dark frames to create master dark.")
    reduced_images = ccdp.ImageFileCollection(calibrated_path)
    calibrated_darks = reduced_images.files_filtered(imagetyp='DARK', include_path=True)

    combined_dark = ccdp.combine(
        calibrated_darks,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=sigma_clip_low_thresh,
        sigma_clip_high_thresh=sigma_clip_high_thresh,
        sigma_clip_func=np.ma.median,
        mem_limit=mem_limit
    )
    combined_dark.meta['combined'] = True
    combined_dark_path = calibrated_path / "master_dark.fits"
    combined_dark.write(combined_dark_path, overwrite=overwrite)
    log(f"Master dark created: {combined_dark_path}")

    return combined_dark


def flat(files, zero, combined_dark, calibrated_path, overscan_region, trim_region, log, cancel_event):
    """
    Calibrate flat images.

    :param files: file location for raw images
    :param zero: combined bias image
    :param combined_dark: combined bias image
    :param calibrated_path: file location for new images
    :param trim_region: trim region for images
    :param overscan_region: overscan region for images
    :return: master flat files in each filter
    """
    log("\nStarting flat calibration.")

    # calibrating and combining the flat frames
    for ccd, file_name in files.ccds(imagetyp='FLAT', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        if cancel_event.is_set():
            log("Task canceled.")
            return

        final_ccd = reduce(ccd, overscan_region, trim_region, 2, zero, combined_dark, good_flat=None)

        # new file name with the filter and number from the original file
        list_of_words = file_name.split(".")
        new_fname = "{}.fits".format(list_of_words[0])

        # Save the result
        final_ccd.write(calibrated_path / new_fname, overwrite=overwrite)

        add_header(calibrated_path, new_fname, "FLAT", "None", None, None, None, trim_region, overscan_region)

        log("Finished overscan correction, bias subtraction, and dark subtraction for " + str(new_fname))

    log("\nFinished overscan, bias subtracting, and dark subtracting of flat frames.")
    log("\nStarting flat combination.")
    time.sleep(10)

    ifc = ccdp.ImageFileCollection(calibrated_path)
    flat_filters = set(h['FILTER'] for h in ifc.headers(imagetyp="FLAT"))
    for filt in flat_filters:
        to_combine = ifc.files_filtered(imagetyp="flat", filter=filt, include_path=True)
        combined_flats = ccdp.combine(to_combine,
                                      method='median',
                                      sigma_clip_high_thresh=sigma_clip_high_thresh,
                                      sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                      rdnoise=rdnoise * u.electron, gain=gain * u.electron / u.adu,
                                      mem_limit=mem_limit
                                      )

        combined_flats.meta['combined'] = True
        flat_file_name = 'master_flat_{}.fits'.format(filt.replace("Empty/", ""))

        combined_flats.write(calibrated_path / flat_file_name, overwrite=overwrite)

        add_header(calibrated_path, flat_file_name, "FLAT", "None", None, None, None, trim_region, overscan_region)

        log("Finished combining flat " + str(flat_file_name))

    log("\nFinished creating the master flats by filter.")


def science_images(files, calibrated_data, zero, combined_dark, trim_region, overscan_region, log, cancel_event):
    all_reds = []
    science_imagetyp = 'LIGHT'
    flat_imagetyp = 'FLAT'

    ifc_reduced = ccdp.ImageFileCollection(calibrated_data)
    combined_flats = {ccd.header['filter']: ccd for ccd in ifc_reduced.ccds(imagetyp=flat_imagetyp, combined=True)}

    log("\nStarting reduction of science images.")

    for light, file_name in files.ccds(imagetyp=science_imagetyp, return_fname=True, ccd_kwargs={'unit': 'adu'}):
        if cancel_event.is_set():
            log("Task canceled.")
            return

        good_flat = combined_flats[light.header['filter']]
        reduced = reduce(light, overscan_region, trim_region, 3, zero, combined_dark, good_flat)

        list_of_words = file_name.split(".")
        new_fname = "{}.fits".format(list_of_words[0])

        all_reds.append(reduced)
        reduced.write(calibrated_data / new_fname, overwrite=overwrite)

        hjd = light.header["JD-HELIO"]
        ra = light.header["RA"]
        dec = light.header["DEC"]

        add_header(calibrated_data, new_fname, science_imagetyp, "None", hjd, ra, dec, trim_region, overscan_region)

        log("Finished calibration of " + str(new_fname))
    log("\nFinished calibrating all science images.")


def add_header(pathway, fname, imagetyp, filter_name, hjd, ra, dec, trim_region, overscan_region):
    """
    Adds values to the header of each image reduced

    :param overscan_region: BIASSEC for the header
    :param trim_region: DATASEC for the header
    :param dec: declination of target object
    :param ra: right ascension of target object
    :param hjd: time of mid-exposure for each image
    :param pathway: pathway to the new file
    :param fname: file name
    :param imagetyp: image type (bias, flat, dark, light)
    :param filter_name: filter type (B, V, R)
    :return: None
    """
    # bias_image = get_pkg_data_filename(pathway + "\\" + fname)
    image_name = pathway / fname
    fits.setval(image_name, "GAIN", value=gain, comment="Units of e-/ADU")
    fits.setval(image_name, "RDNOISE", value=rdnoise, comment="UNits of e-")
    fits.setval(image_name, "OBSERVAT", value=location, comment="Obs. Loc.")
    fits.setval(image_name, "IMAGETYP", value=imagetyp, comment="Image type")
    fits.setval(image_name, "DATASEC", value=trim_region, comment="Trim data section")
    fits.setval(image_name, "BIASSEC", value=overscan_region, comment="Overscan section")
    fits.setval(image_name, "EPOCH", value="J2000.0")
    # fits.setval(bias_image, "FILTER", value=filter_name)
    if imagetyp == "LIGHT":
        if location.lower() == "bsuo":
            # location of BSUO
            obs_location = {
                'lon': -85.411896,  # degrees
                'lat': 40.199879,  # degrees
                'elevation': 0.2873  # kilometers
            }
            bjd = BJD_TDB(hjd, obs_location, ra, dec)
            fits.setval(image_name, "BJD_TDB", value=bjd.value, comment="Bary. Julian Date, Bary. Dynamical Time")
        else:
            bjd = BJD_TDB(hjd, location, ra, dec)
            fits.setval(image_name, "BJD_TDB", value=bjd.value, comment="Bary. Julian Date, Bary. Dynamical Time")


def BJD_TDB(hjd, loc, ra, dec):
    """
    Converts HJD to BJD_TDB

    :param ra: right ascension of target object
    :param dec: declination of target object
    :param hjd: HJD time for the mid-exposure image
    :param loc: location of the observations
    :return: Newly calculated BJD_TDB that is light time corrected
    """
    helio = Time(hjd, scale='utc', format='jd')
    obs = EarthLocation.from_geodetic(loc["lon"], loc["lat"], loc["elevation"])
    star = SkyCoord(ra, dec, unit=(u.hour, u.deg))
    ltt = helio.light_travel_time(star, 'heliocentric', location=obs)
    guess = helio - ltt
    # if we assume guess is correct - how far is heliocentric time away from true value?
    delta = (guess + guess.light_travel_time(star, 'heliocentric', obs)).jd - helio.jd
    # apply this correction
    guess -= delta * u.d

    ltt = guess.light_travel_time(star, 'barycentric', obs)
    return guess.tdb + ltt
