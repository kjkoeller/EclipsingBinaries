"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 04/05/2023

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
import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData
import matplotlib.pyplot as plt

# turn off this warning that just tells the user,
# "The warning raised when the contents of the FITS header have been modified to be standards compliant."
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)

# global variables for combining, to either play with or set depending on the run
sigma_clip_low_thresh = None
sigma_clip_high_thresh = 3
sigclip = 5  # sigclip for cosmic ray removal
rdnoise = 10.83  # * u.electron  # gathered from fits headers manually
gain = 1.43  # * (u.electron / u.adu)  # gathered from fits headers manually
overwrite = True  # if the user wants to overwrite already existing files or not, by default it is set to True
mem_limit = 450e6  # maximum memory limit 4.5 Gb is the recommended which is 450e6 (i.e. 8.0 Gb would be 800e6)
dark_bool = "True"


def main():
    """
    This function calls all other functions in order of the calibration.

    :return: outputs all calibration images into a new reduced folder designated by the user.
    """

    # allows the user to input where the raw images are and where the calibrated images go to
    path = input("Please enter a file pathway (i.e. C:\\folder1\\folder2\\[raw]) to where the raw images are or type "
                 "the word 'Close' to leave: ")
    if path.lower() == "close":
        exit()
    # path = "Calibration2"
    calibrated = input("Please enter a file pathway for a new calibrated folder to not overwrite the original images "
                       "(C:\\folder1\\folder2\\[calibrated]): ")

    # checks whether the file paths from above are real
    while True:
        try:
            images_path = Path(path)
            calibrated_data = Path(calibrated)
            break
        except FileNotFoundError:
            print("Files were not found. Please try again.\n")
            path = input("Please enter a file path or folder name (if this code is in the same main folder): ")
            calibrated = input("Please enter a name for a new calibrated folder to not overwrite the original images: ")

    print("\nDo you want to load default options like gain and readnoise? The defaults are for BSUO")
    while True:
        default_ans = input("To load defaults type 'Default' otherwise type 'New' to enter values: ")
        if default_ans.lower() == "default":
            break
        elif default_ans.lower() == "new":
            default()
            break
        else:
            print("Please either enter 'Default' or 'New'\n.")

    calibrated_data.mkdir(exist_ok=True)
    files = ccdp.ImageFileCollection(images_path)

    zero, overscan_region, trim_region = bias(files, calibrated_data, path)
    if not dark_bool:
        master_dark = None
    else:
        master_dark = dark(files, zero, calibrated_data, overscan_region, trim_region)
    flat(files, zero, master_dark, calibrated_data, overscan_region, trim_region)
    science_images(files, calibrated_data, zero, master_dark, trim_region, overscan_region)


def default():
    """
    Generates new values that the user can enter

    :return: newly entered default values
    """
    global sigma_clip_high_thresh
    global sigma_clip_low_thresh
    global gain
    global rdnoise
    global sigclip
    global dark_bool

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
    # Subtract the overscan, ccd[columns, rows] I think?
    if overscan_region.lower() == "none":
        pass
    else:
        ccd = ccdp.subtract_overscan(ccd, fits_section=overscan_region, median=True, overscan_axis=1)

    # Trim the overscan and gain correct the image
    ccd = ccdp.trim_image(ccd, fits_section=trim_region)
    new_ccd = ccdp.gain_correct(ccd, gain=gain)  # gain correct the image

    # this if statement checks whether the input is bias, dark, flat, or science reduction
    # bias combining
    if num == 0:
        return new_ccd
    # dark combining
    elif num == 1:
        # Subtract bias
        sub_ccd = ccdp.subtract_bias(new_ccd, zero)

        return sub_ccd
    # flat combining
    elif num == 2:
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
        new_reduced = ccdp.cosmicray_lacosmic(reduced, gain_apply=False, readnoise=rdnoise, gain=gain, sigclip=sigclip)

        return new_reduced


def bias(files, calibrated_data, path):
    """
    Calibrates the bias images

    :param path: the raw images folder path
    :param files: file location where all raw images are
    :param calibrated_data: file location where the new images go
    :return: the combined bias image and the trim and overscan regions
    """

    # plots one of the flat image mean count values across all columns to find the trim and overscan regions
    print("\n\nThe flat image that you enter next should be inside the " + "\033[1m" + "\033[93m" + "FIRST" +
          "\033[00m" + " folder that you entered above or this will crash.\n")
    while True:
        try:
            image = input(
                "Please enter the name of one of the flat image to be looked at for overscan and trim regions: ")
            cryo_path = Path(path)
            bias_1 = CCDData.read(cryo_path / image, unit='adu')
            break
        except FileNotFoundError:
            print("\nThe file you entered could not be found, please try entering " 
                  "\033[1m" + "\033[93m" + "JUST" + "\033[00m" + " the file name only.\n")
    # bias_1 = CCDData.read(cryo_path / 'bias-0001.fits', unit='adu')  # testing

    print("\n\nFor the overscan region, [rows, columns], and if you want all the columns then you want would enter, \n"
          "[1234:5678, 1234:5678] and this would say rows between those values and all the columns. \n"
          "This would also work if you wanted all the rows ([: , 1234:5678]).\n")
    bias_plot(bias_1)

    overscan_region = input("Please enter the overscan region you determined from the figure.\n"
                            "Example '[2073:2115, :]' or if you do not have an overscan region enter 'None': ")
    trim_region = input("Please enter the trim region. Example '[20:2060, 12:2057]': ")
    print()

    print("\nStarting overscan on bias.\n")
    for ccd, file_name in files.ccds(imagetyp='BIAS', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        new_ccd = reduce(ccd, overscan_region, trim_region, 0, zero=None, combined_dark=None, good_flat=None)

        list_of_words = file_name.split(".")
        new_fname = "{}.fits".format(list_of_words[0])
        # new_fname = "bias_o_{}.fits".format(list_of_words[3])
        # new_fname = "bias_o_{}.fits".format(list_of_words[1])  # testing
        # Save the result
        new_ccd.write(calibrated_data / new_fname, overwrite=overwrite)

        # output that an image is finished for updates to the user
        print("Finished overscan correction for " + str(new_fname))

    print("\nFinished overscan correcting bias frames.")
    # combine all the output bias images into a master bias
    reduced_images = ccdp.ImageFileCollection(calibrated_data)
    calibrated_biases = reduced_images.files_filtered(imagetyp='BIAS', include_path=True)

    combined_bias = ccdp.combine(calibrated_biases,
                                 method='average',
                                 sigma_clip=True, sigma_clip_low_thresh=sigma_clip_low_thresh,
                                 sigma_clip_high_thresh=sigma_clip_high_thresh,
                                 sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                 mem_limit=450e6
                                 )

    combined_bias.meta['combined'] = True
    combined_bias.write(calibrated_data / 'zero.fits', overwrite=overwrite)

    print("\nFinished creating zero.fits\n")

    return combined_bias, overscan_region, trim_region


def bias_plot(ccd):
    """
    Plots the count values for row 1000 to find the overscan and trim regions

    :param ccd: bias image to be looked at
    :return: None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(ccd.data[1000][:], label='Raw Flat')
    plt.grid()
    plt.axvline(x=2077, color='black', linewidth=2, linestyle='dashed', label='Suggested Start of Overscan')
    plt.legend()
    # plt.ylim(0, 60000)
    plt.xlim(-50, 2130)
    plt.xlabel('pixel number')
    plt.ylabel('Counts')
    # plt.title('Count Values for Row 1000')
    plt.show()


def dark(files, zero, calibrated_path, overscan_region, trim_region):
    """
    Calibrates the dark frames.

    :param files: file location of raw images
    :param zero: master bias image
    :param calibrated_path: file location for the new images
    :param trim_region: trim region for images
    :param overscan_region: overscan region for images
    :return: combined master dark
    """
    print("Starting dark calibration.\n")
    # calibrating a combining the dark frames
    for ccd, file_name in files.ccds(imagetyp='DARK', ccd_kwargs={'unit': 'adu'}, return_fname=True):
        sub_ccd = reduce(ccd, overscan_region, trim_region, 1, zero, combined_dark=None, good_flat=None)

        # new file name that uses the number from the original image
        list_of_words = file_name.split(".")
        new_fname = "{}.fits".format(list_of_words[0])
        # new_fname = "dark_o_b_{}.fits".format(list_of_words[3])
        # Save the result
        sub_ccd.write(calibrated_path / new_fname, overwrite=overwrite)

        print("Finished overscan correction and bias subtraction for " + str(new_fname))

    print("\nFinished overscan correcting and bias subtracting all dark frames.")
    print("\nStarting combining dark frames.\n")
    time.sleep(10)
    reduced_images = ccdp.ImageFileCollection(calibrated_path)
    calibrated_darks = reduced_images.files_filtered(imagetyp='dark', include_path=True)

    combined_dark = ccdp.combine(calibrated_darks,
                                 method='average',
                                 sigma_clip=True, sigma_clip_low_thresh=sigma_clip_low_thresh,
                                 sigma_clip_high_thresh=sigma_clip_high_thresh,
                                 sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                 rdnoise=rdnoise, gain=gain, mem_limit=450e6
                                 )

    combined_dark.meta['combined'] = True
    combined_dark.write(calibrated_path / 'master_dark.fits', overwrite=overwrite)
    print("Finished creating a master dark.\n")
    return combined_dark


def flat(files, zero, combined_dark, calibrated_path, overscan_region, trim_region):
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
    print("Starting flat calibration.\n")
    print()

    # calibrating and combining the flat frames
    for ccd, file_name in files.ccds(imagetyp='FLAT', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        final_ccd = reduce(ccd, overscan_region, trim_region, 2, zero, combined_dark, good_flat=None)

        # new file name with the filter and number from the original file
        list_of_words = file_name.split(".")
        new_fname = "{}.fits".format(list_of_words[0])
        # new_fname = "flat_o_b_d_{}{}.fits".format(list_of_words[2], list_of_words[4])
        # new_fname = "flat_o_b_d_{}_{}.fits".format(list_of_words[1], list_of_words[0])  # testing

        # Save the result
        final_ccd.write(calibrated_path / new_fname, overwrite=overwrite)

        print("Finished overscan correction, bias subtraction, and dark subtraction for " + str(new_fname))

    print("\nFinished overscan, bias subtracting, and dark subtracting of flat frames.\n")
    print("Starting flat combination.\n")
    time.sleep(10)

    ifc = ccdp.ImageFileCollection(calibrated_path)
    flat_filters = set(h['FILTER'] for h in ifc.headers(imagetyp="FLAT"))
    for filt in flat_filters:
        to_combine = ifc.files_filtered(imagetyp="flat", filter=filt, include_path=True)
        combined_flats = ccdp.combine(to_combine,
                                      method='average',
                                      sigma_clip_high_thresh=sigma_clip_high_thresh,
                                      sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                      rdnoise=rdnoise, gain=gain, mem_limit=450e6
                                      )

        combined_flats.meta['combined'] = True
        flat_file_name = 'master_flat_{}.fits'.format(filt.replace("Empty/", ""))

        combined_flats.write(calibrated_path / flat_file_name, overwrite=overwrite)

        print("Finished combining flat " + str(flat_file_name))

    print("\nFinished creating the master flats by filter.\n")

    return combined_flats


def science_images(files, calibrated_data, zero, combined_dark, trim_region, overscan_region):
    all_reds = []
    science_imagetyp = 'LIGHT'
    flat_imagetyp = 'FLAT'

    ifc_reduced = ccdp.ImageFileCollection(calibrated_data)
    combined_flats = {ccd.header['filter']: ccd for ccd in ifc_reduced.ccds(imagetyp=flat_imagetyp, combined=True)}

    print("Starting reduction of science images.\n")

    for light, file_name in files.ccds(imagetyp=science_imagetyp, return_fname=True, ccd_kwargs={'unit': 'electron'}):
        good_flat = combined_flats[light.header['filter']]
        reduced = reduce(light, overscan_region, trim_region, 3, zero, combined_dark, good_flat)

        list_of_words = file_name.split(".")
        new_fname = "{}.fits".format(list_of_words[0])
        # new_fname = "{}_o_b_d_f_{}_{}_{}.fits".format(list_of_words[0], list_of_words[2], list_of_words[5], list_of_words[6].replace(".fts", ""))
        # new_fname = "{}_o_b_d_{}.fits".format(list_of_words[0], list_of_words[1])  # testing

        all_reds.append(reduced)
        reduced.write(calibrated_data / new_fname, overwrite=overwrite)

        print("Finished calibration of " + str(new_fname))
    print("\nFinished calibrating all science images.")


if __name__ == '__main__':
    main()
