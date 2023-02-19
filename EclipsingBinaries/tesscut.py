# -*- coding: utf-8 -*-
"""
Author: John Kielkopf (University of Louisville)
Created: Unknown

Editor: Kyle Koeller
Last Edited: 02/19/2023

Spyder Editor
This is a temporary script file.

Paper is: https://ui.adsabs.harvard.edu/abs/2019ascl.soft05007B/abstract
"""

# !/usr/local/bin/python3

import numpy as np
import astropy.io.fits as pyfits
from time import gmtime, strftime  # for utc
from astropy.time import Time
from PyAstronomy import pyasl

"""
Extract all images from a TESS pixel BINTABLE file
Removes extended headers
Selects only the image data
Detects and does not convert low quality
"""


def main(search_file):
    print(str(search_file))
    infile = "tess" + str(search_file).split("tess")[1]  # gets the actual sector file
    print(infile)
    pathway = str(search_file).split("tess")[0]  # gets the file pathway
    print("\nThe program will use the file pathway that you entered previously and will now ask for a prefix "
          "to each file name.")

    print("Example prefix might be 'NSVS_896797_[program adds stuff here]'\n")
    outprefix = input("Please enter what you want each file to always have in its name: ")

    # Set an overwrite flag True so that images can be overwritten
    # Otherwise set it False for safety

    overwriteflag = True

    # Open the fits file readonly by default and create an input hdulist
    inlist = pyfits.open(search_file)
    # Assign the input headers

    # Master

    # inhdr0 = inlist[0].header

    # Sector information including range of dates
    # For more information inspect the entries in inhdr1
    # Is there a DQUALITY flag?

    # inhdr1 = inlist[1].header

    # Coordinates

    inhdr2 = inlist[2].header

    # Create a new fits header for the pixel file images

    newhdr = inhdr2

    # Clear the instrument and telescope keywords
    # AIJ may use them in development software that could break processing

    del newhdr['INSTRUME']
    del newhdr['TELESCOP']
    del newhdr['CHECKSUM']

    # Clear the extension name which will not apply to the cutout slices

    del newhdr['EXTNAME']

    imagedata = inlist[1].data
    nimages = np.size(imagedata)

    # Diagnostics

    # print(np.size(imagedata))
    # print(len(imagedata[0]))

    print("Starting to check images. This may take a few minutes. \n")
    for i in range(nimages):
        # Get image data

        inimage = imagedata[i][4]

        # Get BJD -2457000

        bjd0 = 2457000.
        bjd1 = imagedata[i][0]
        quality_flag = imagedata[i][8]
        tess_ffi = imagedata[i][11]

        if np.isnan(bjd1):
            print('Image ', i + 1, 'skipped: lacks valid BJD timestamp.')

        elif quality_flag != 0:
            print('Image ', i + 1, 'skipped: poor quality.')

        else:
            # calculate the mid-exposure in BJD
            bjd = bjd0 + bjd1
            outimage = inimage

            # convert the BJD to HJD
            time_inp = Time(bjd, format='jd', scale='tdb')
            split = infile.split("_")
            ra = split[1]
            dec = split[2]

            jd_t = time_inp.jd
            # noinspection PyUnresolvedReferences
            hjd = pyasl.helio_jd(jd_t-2.4e6, float(ra), float(dec))

            # Create the fits object for this image using the header of the bintable image
            # Use float32 for output type

            outlist = pyfits.PrimaryHDU(outimage.astype('float32'), newhdr)

            # Provide a new date stamp

            file_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

            # Update the header
            # Append data to the header from the other header blocks of the bintable file

            outhdr = outlist.header
            outhdr['LST_UPDT'] = file_time
            outhdr['BJD_TDb'] = bjd
            outhdr['HJD'] = hjd
            outhdr['COMMENT'] = tess_ffi
            # outhdr['history'] = 'Image from ' + infile

            # Write the fits file

            outfile = pathway + outprefix + 'tess_%05d.fits' % (i,)
            outlist.writeto(outfile, overwrite=overwriteflag)

    print("Finished checking all images.\n")
    # Close the input  and exit
    inlist.close()


# main("D:\\NSVS_11868841\\tess-s0056-1-1_349.489832_19.284108_87x88_astrocut.fits")
