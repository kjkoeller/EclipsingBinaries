"""
Author: Kyle Koeller
Created: 11/08/2022
Last Edited: 11/16/2022

This program is meant to automatically do the data reduction of the raw images from the
Ball State University Observatory (BSUO). The new calibrated images are placed into a new folder
"""

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
rdnoise = 10.83  # gathered from fits headers manually
gain = 1.43  # gathered from fits headers manually


def main():
    """
    This function calls all other functions in order of the calibration.

    :return: outputs all calibration images into a new reduced folder designated by the user.
    """

    # allows the user to input where the raw images are and where the calibrated images go to
    # path = input("Please enter a file path or folder name (if this code is in the same main folder) or type the word "
    #              "'Close' to leave: ")
    path = "Calibration"
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
    files = ccdp.ImageFileCollection(images_path)

    zero, overscan_region, trim_region = bias(files, calibrated_data, path)
    master_dark = dark(files, zero, calibrated_data, overscan_region, trim_region)
    flat(files, zero, master_dark, calibrated_data, overscan_region, trim_region)


def bias(files, calibrated_data, path):
    """
    Calibrates the bias images

    :param path: the raw images folder path
    :param files: file location where all raw images are
    :param calibrated_data: file location where the new images go
    :return: the combined bias image and the trim and overscan regions
    """

    # plots one of the bias image mean count values across all columns to find the trim and overscan regions
    cryo_path = Path(path)
    bias_1 = CCDData.read(cryo_path / 'Bias-S008-R003-C001-B2.fts', unit='adu')
    plt.figure(figsize=(10, 5))
    plt.plot(bias_1.data[1000][:], label='Raw Bias')
    plt.grid()
    plt.axvline(x=2077, color='black', linewidth=3, linestyle='dashed', label='Suggested Start of Overscan')
    plt.legend()
    plt.ylim(0, 2000)
    plt.xlim(-50, 2130)
    plt.xlabel('pixel number')
    plt.ylabel('Counts')
    plt.title('Overscan region, averaged over all rows')
    # plt.show()

    print("For the overscan region, [rows, columns], and if you want all the columns then you want would enter, "
          "[1234:5678, 1234:5678] and this would say rows between those values and all the columns. This would also work if you"
          "wanted all the rows ([: , 1234:5678]).")
    print()
    overscan_region = input("Please enter the overscan region you determined from the figure. Example [2073:2115, :]: ")
    # [20:2060, 12:2057]
    trim_region = input("Please enter the trim region. Example [20:2060, 12:2057]: ")
    print()

    print("Starting overscan on bias.")
    print()
    # raw_biases = files.files_filtered(include_path=True, imagetyp='BIAS')
    # ccd_kwargs={'unit': 'adu'},
    for ccd, file_name in files.ccds(imagetyp='BIAS', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        # Just get the bias frames
        # CCDData requires a unit for the image if
        # it is not in the header # Provide the file name too.

        # Subtract the overscan, ccd[columns, rows] I think?
        ccd = ccdp.subtract_overscan(ccd, fits_section=overscan_region, median=True)

        # Trim the overscan
        ccd = ccdp.trim_image(ccd, fits_section=trim_region)

        # cosmic ray reject above 5 sigmas and gain_apply is set to false because it changes the units of the image
        # and applies a gain to the image as well
        new_ccd = ccdp.cosmicray_lacosmic(ccd, gain_apply=False, readnoise=rdnoise, gain=gain, sigclip=sigclip)

        list_of_words = file_name.split("-")
        new_fname = "bias_o_{}.fits".format(list_of_words[3])
        # Save the result
        new_ccd.write(calibrated_data / new_fname, overwrite=True)

        # output that an image is finished for updates to the user
        print("Finished overscan correction for " + str(new_fname))

    print()
    print("Finished overscan correcting bias frames.")
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
    combined_bias.write(calibrated_data / 'zero.fits', overwrite=True)

    print()
    print("Finished creating zero.fits")
    print()

    return combined_bias, overscan_region, trim_region


def dark(files, combined_bias, calibrated_path, overscan_region, trim_region):
    """
    Calibrates the dark frames.

    :param files: file location of raw images
    :param combined_bias: master bias image
    :param calibrated_path: file location for the new images
    :param trim_region: trim region for images
    :param overscan_region: overscan region for images
    :return: combinedmaster dark
    """
    # reduced_images = ccdp.ImageFileCollection(calibrated_path)
    print("Starting dark calibration.")
    print()

    # calibrating a combining the dark frames
    for ccd, file_name in files.ccds(imagetyp='DARK', ccd_kwargs={'unit': 'adu'}, return_fname=True):
        # Just get the dark frames
        # CCDData requires a unit for the image if
        # it is not in the header # Provide the file name too.):

        # Subtract the overscan, ccd[columns, rows] I think?
        ccd = ccdp.subtract_overscan(ccd, fits_section=overscan_region, median=True)

        # Trim the overscan
        ccd = ccdp.trim_image(ccd, fits_section=trim_region)

        # cosmic ray reject above 5 sigmas and gain_apply is set to false because it changes the units of the image
        # and applies a gain to the image as well
        new_ccd = ccdp.cosmicray_lacosmic(ccd, gain_apply=False, readnoise=rdnoise, gain=gain, sigclip=5)

        # Subtract bias
        new_ccd = ccdp.subtract_bias(new_ccd, combined_bias)
        list_of_words = file_name.split("-")

        new_fname = "dark_o_b_{}.fits".format(list_of_words[3])
        # Save the result
        new_ccd.write(calibrated_path / new_fname, overwrite=True)

        print("Finished overscan correction for " + str(new_fname))

    print("Finished overscan correcting and bias subtracting dark frames.")
    print()
    print("Starting combining dark frames.")
    print()
    time.sleep(10)
    reduced_images = ccdp.ImageFileCollection(calibrated_path)
    calibrated_darks = reduced_images.files_filtered(imagetyp='dark', include_path=True)

    combined_darks = ccdp.combine(calibrated_darks,
                                  method='average',
                                  sigma_clip=True, sigma_clip_low_thresh=sigma_clip_low_thresh,
                                  sigma_clip_high_thresh=sigma_clip_high_thresh,
                                  sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                  rdnoise=rdnoise, gain=gain, mem_limit=450e6
                                  )

    combined_darks.meta['combined'] = True
    combined_darks.write(calibrated_path / 'master_dark.fits', overwrite=True)

    print("Finished creating a master dark.")
    print()
    return combined_darks


def flat(files, combined_bias, combined_darks, calibrated_path, overscan_region, trim_region):
    """
    Calibrate flat images.

    :param files: file location for raw images
    :param combined_bias: combined bias image
    :param combined_darks: combined bias image
    :param calibrated_path: file location for new images
    :return: master flat files in each filter
    """
    print("Starting flat calibration.")
    print()

    # calibrating and combining the flat frames
    for ccd, file_name in files.ccds(imagetyp='FLAT', return_fname=True, ccd_kwargs={'unit': 'adu'}):
        # Just get the bias frames
        # Provide the file name too.

        # Subtract the overscan, ccd[columns, rows] I think?
        ccd = ccdp.subtract_overscan(ccd, fits_section=overscan_region, median=True)

        # Trim the overscan
        ccd = ccdp.trim_image(ccd, fits_section=trim_region)

        # cosmic ray reject above 5 sigmas and gain_apply is set to false because it changes the units of the image
        # and applies a gain to the image as well
        cosmic_ccd = ccdp.cosmicray_lacosmic(ccd, gain_apply=False, readnoise=rdnoise, gain=gain, sigclip=5)

        # Subtract bias
        new_ccd = ccdp.subtract_bias(cosmic_ccd, combined_bias)

        # Subtract the dark current
        final_ccd = ccdp.subtract_dark(new_ccd, combined_darks, exposure_time='exptime', exposure_unit=u.second, scale=True)
        list_of_words = file_name.split("-")

        new_fname = "flat_o_b_d_{}_{}.fits".format(list_of_words[2], list_of_words[4])
        # Save the result
        final_ccd.write(calibrated_path / new_fname, overwrite=True)

        print("Finished overscan correction for " + str(new_fname))

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
                                     sigma_clip=True, sigma_clip_low_thresh=sigma_clip_low_thresh,
                                     sigma_clip_high_thresh=sigma_clip_high_thresh,
                                     sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                     rdnoise=rdnoise, gain=gain, mem_limit=450e6
                                     )

        combined_flat.meta['combined'] = True
        flat_file_name = 'master_flat_{}.fits'.format(filt.replace("Empty/", ""))

        combined_flat.write(calibrated_path / flat_file_name, overwrite=True)

        print("Finished combing flat " + str(new_fname))

    print("Finished creating the master flats by filter.")
    print()
    print("Done")


if __name__ == '__main__':
    main()
