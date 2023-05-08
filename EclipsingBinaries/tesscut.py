# -*- coding: utf-8 -*-
"""
Author: John Kielkopf (University of Louisville)
Created: Unknown

Editor: Kyle Koeller
Last Edited: 05/07/2023

Paper is: https://ui.adsabs.harvard.edu/abs/2019ascl.soft05007B/abstract
"""

# !/usr/local/bin/python3

import numpy as np
import astropy.io.fits as pyfits
from time import gmtime, strftime  # for utc
from .vseq_updated import conversion
# from vseq_updated import conversion  # testing purposes

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u

"""
Extract all images from a TESS pixel BINTABLE file
Removes extended headers
Selects only the image data
Detects and does not convert low quality
"""


def main(search_file, pathway):
    # print(str(search_file))
    infile = "tess" + str(search_file).split("tess")[1]  # gets the actual sector file
    # print(infile)
    # pathway = str(search_file).split("tess")[0]  # gets the file pathway
    print("\nThe program will use the file pathway that you entered previously and will now ask for a prefix "
          "to each file name.")

    print("Example prefix might be 'NSVS_896797_[program adds stuff here]'\n")
    outprefix = input("Please enter what you want each file to always have in its name: ")

    # Set an overwrite flag True so that images can be overwritten
    # Otherwise set it False for safety

    overwriteflag = True

    # Open the fits file readonly by default and create an input hdulist
    inlist = pyfits.open(pathway + "\\" + infile)
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

            split = infile.split("_")
            ra = conversion([str(float(split[1]) / 15)])[0]
            dec = conversion([split[2]])[0]

            hjd = bary_to_helio(ra, dec, bjd, "greenwich")

            outlist = pyfits.PrimaryHDU(outimage.astype('float32'), newhdr)

            # Provide a new date stamp

            file_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

            # Update the header
            # Append data to the header from the other header blocks of the bintable file

            outhdr = outlist.header
            outhdr['LST_UPDT'] = file_time
            outhdr['BJD_TDB'] = bjd
            outhdr['HJD'] = hjd.value
            outhdr['COMMENT'] = tess_ffi
            # outhdr['history'] = 'Image from ' + infile

            # Write the fits file

            outfile = pathway + "/" + outprefix + 'tess_%05d.fits' % (i,)
            # print(outfile)
            outlist.writeto(r"" + outfile, overwrite=overwriteflag)

    print("Finished checking all images.\n")
    # Close the input  and exit
    inlist.close()


def bary_to_helio(ra, dec, bjd, obs_name):
    bary = Time(bjd, scale='tdb', format='jd')
    obs = EarthLocation.of_site(obs_name)
    star = SkyCoord(ra, dec, unit=(u.hour, u.deg))
    ltt = bary.light_travel_time(star, 'barycentric', location=obs)
    guess = bary - ltt
    delta = (guess + guess.light_travel_time(star, 'barycentric', obs)).jd - bary.jd
    guess -= delta * u.d

    ltt = guess.light_travel_time(star, 'heliocentric', obs)
    return guess.utc + ltt


# main("tess-s0059-3-1_7.116535_78.961849_88x88_astrocut.fits", "C:\\New folder")
