"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 11/11/2022

This program is meant to automatically do the data reduction of the raw images from the
Ball State University Observatory (BSUO). The new calibrated images are placed into a new folder
"""

from pathlib import Path
import time

from astropy.stats import mad_std
from astropy import units as u
import ccdproc as ccdp
import numpy as np


def main():
    """
    This function calls all other functions in order of the calibration.

    :return: outputs all calibration images into a new reduced folder designated by the user.
    """

    # allows the user to input where the raw images are and where the calibrated images go to
    path = input(
        "Please enter a file path or folder name (if this code is in the same main folder) or type the word 'Close' to leave: ")
    if path.lower() == "close":
        exit()
    calibrated = input("Please enter a name for a new calibrated folder to not overwrite the original images: ")

    # checks whether the file paths from above are real
    while True:
        test = 0
        try:
            images_path = Path(path)
            calibrated_data = Path(calibrated)
        except FileNotFoundError:
            print("Files were not found")
            print()
            test = -1
        if test == 0:
            break
        else:
            path = input("Please enter a file path or folder name (if this code is in the same main folder): ")
            calibrated = input("Please enter a name for a new calibrated folder to not overwrite the original images: ")

    calibrated_data.mkdir(exist_ok=True)
    rdnoise = 10.83  # gathered from fits headers manually
    gain = 1.43  # gathered from fits headers manually
    files = ccdp.ImageFileCollection(images_path)

    zero = bias(files, calibrated_data)
    master_dark = dark(files, zero, calibrated_data, rdnoise, gain)
    flat(files, zero, master_dark, calibrated_data, rdnoise, gain)


def bias(files, calibrated_data):
    """
    Calibrates the bias images

    :param files: file location where all raw images are
    :param calibrated_data: file location where the new images go
    :return: the combined bias image
    """
    # raw_biases = files.files_filtered(include_path=True, imagetyp='BIAS')
    for ccd, file_name in files.ccds(imagetyp='BIAS', ccd_kwargs={'unit': 'adu'}, return_fname=True):
        # Just get the bias frames
        # CCDData requires a unit for the image if 
        # it is not in the header # Provide the file name too.

        # Subtract the overscan
        ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, 13:2057], median=True)

        # Trim the overscan
        ccd = ccdp.trim_image(ccd[20:2065, 13:2057])
        list_of_words = file_name.split("-")

        new_fname = "bias_osc_{}.fits".format(list_of_words[3])
        # Save the result
        ccd.write(calibrated_data / new_fname, overwrite=True)

    print("Finished overscan correcting bias frames.")
    print()
    # combine all the output bias images into a master bias
    reduced_images = ccdp.ImageFileCollection(calibrated_data)
    calibrated_biases = reduced_images.files_filtered(imagetyp='bias', include_path=True)

    combined_bias = ccdp.combine(calibrated_biases,
                                 method='average',
                                 minmax_clip=True, minmax_clip_min=0, minmax_clip_max=1,
                                 sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
                                 sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std, mem_limit=450e6
                                 )

    combined_bias.meta['combined'] = True
    combined_bias.write(calibrated_data / 'zero.fits', overwrite=True)

    print("Finished creating zero.fits")
    print()
    return combined_bias


def dark(files, combined_bias, calibrated_path, rdnoise, gain):
    """
    Calibrates the dark frames.

    :param files: file location of raw images
    :param combined_bias: master bias image
    :param calibrated_path: file location for the new images
    :param rdnoise: readout noise for BSUO
    :param gain: gain for BSUO
    :return:
    """
    # reduced_images = ccdp.ImageFileCollection(calibrated_path)
    print("Starting dark calibration.")
    print()

    # calibrating a combining the dark frames
    for ccd, file_name in files.ccds(imagetyp='DARK', ccd_kwargs={'unit': 'adu'}, return_fname=True):
        # Just get the dark frames
        # CCDData requires a unit for the image if 
        # it is not in the header # Provide the file name too.):

        # Subtract the overscan
        ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, 13:2057], median=True)

        # Trim the overscan
        ccd = ccdp.trim_image(ccd[20:2065, 13:2057])

        # Subtract bias
        ccd = ccdp.subtract_bias(ccd, combined_bias)
        list_of_words = file_name.split("-")

        new_fname = "dark_osc_bs_{}.fits".format(list_of_words[3])
        # Save the result
        ccd.write(calibrated_path / new_fname, overwrite=True)

    print("Finished overscan correcting and bias subtracting dark frames.")
    print()
    print("Starting combining dark frames.")
    print()
    time.sleep(10)
    reduced_images = ccdp.ImageFileCollection(calibrated_path)
    calibrated_darks = reduced_images.files_filtered(imagetyp='dark', include_path=True)

    combined_darks = ccdp.combine(calibrated_darks,
                                  method='average',
                                  minmax_clip=True, minmax_clip_min=0, minmax_clip_max=1,
                                  sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
                                  sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                  rdnoise=rdnoise, gain=gain, mem_limit=450e6
                                  )

    combined_darks.meta['combined'] = True
    combined_darks.write(calibrated_path / 'master_dark.fits', overwrite=True)

    print("Finished creating a master dark.")
    print()
    return combined_darks


def flat(files, combined_bias, combined_darks, calibrated_path, rdnoise, gain):
    """
    Calibrate flat images.

    :param files: file location for raw images
    :param combined_bias: combined bias image
    :param combined_darks: combined bias image
    :param calibrated_path: file location for new images
    :param rdnoise: readout noise for BSUO
    :param gain: gain for BSUO
    :return:
    """
    print("Starting flat calibration.")
    print()
    
    # calibrating and combining the flat frames
    for ccd, file_name in files.ccds(imagetyp='FLAT', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        # Just get the bias frames
        # Provide the file name too.

        # Subtract the overscan
        ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, 13:2057], median=True)

        # Trim the overscan
        ccd = ccdp.trim_image(ccd[20:2065, 13:2057])

        # Subtract bias
        ccd = ccdp.subtract_bias(ccd, combined_bias)

        # Subtract the dark current
        ccd = ccdp.subtract_dark(ccd, combined_darks, exposure_time='exptime', exposure_unit=u.second, scale=True)
        list_of_words = file_name.split("-")

        new_fname = "flat_osc_bs_ds_{}_{}.fits".format(list_of_words[2], list_of_words[4])
        # Save the result
        ccd.write(calibrated_path / new_fname, overwrite=True)

    print("Finished overscan, bias subtracting, and dark subtracting of flat frames.")
    print()
    print("Starting flat combination.")
    print()
    time.sleep(10)

    ifc = ccdp.ImageFileCollection(calibrated_path)
    flat_filters = set(h['FILTER'] for h in ifc.headers(imagetyp="FLAT"))
    for filt in flat_filters:
        to_combine = ifc.files_filtered(imagetyp="flat", filter=filt, include_path=True)
        combined_flat = ccdp.combine(to_combine,
                                     method='average',
                                     minmax_clip=True, minmax_clip_min=0, minmax_clip_max=1,
                                     sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
                                     sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                     rdnoise=rdnoise, gain=gain, mem_limit=450e6
                                     )

        combined_flat.meta['combined'] = True
        dark_file_name = 'master_flat_{}.fit'.format(filt.replace("Empty/", ""))

        combined_flat.write(calibrated_path / dark_file_name, overwrite=True)

    print("Finished creating the master flats by filter.")
    print()
    print("Done")


if __name__ == '__main__':
    main()
