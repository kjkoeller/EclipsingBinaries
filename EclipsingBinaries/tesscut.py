"""
Author: John Kielkopf (University of Louisville)
Created: Unknown

Editor: Kyle Koeller
Last Edited: 12/12/2024

Paper is: https://ui.adsabs.harvard.edu/abs/2019ascl.soft05007B/abstract
"""

import numpy as np
import astropy.io.fits as pyfits
from time import gmtime, strftime
from .vseq_updated import conversion
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u


def process_tess_cutout(search_file, pathway, sector, outprefix, write_callback, cancel_event):
    """
    Process TESS pixel data from a BINTABLE file and save individual images.

    Parameters
    ----------
    search_file : str
        The path to the TESS BINTABLE file.
    pathway : str
        The directory to save output FITS files.
    outprefix : str
        The prefix for the output file names.
    write_callback : function, optional
        A function to log messages to the GUI.
    sector: int
        The sector number being downloaded
    """
    def log(message):
        """Log messages to the GUI or print to the console."""
        if write_callback:
            write_callback(message)
        else:
            print(message)

    try:
        if cancel_event.is_set():
            log(f"Task canceled while processing Sector {sector}.")
            return

        # Extract file name and verify file paths
        infile = "tess" + str(search_file).split("tess")[1]
        log(f"Processing file: {infile}")
        log(f"Output directory: {pathway}")

        # Open FITS file
        inlist = pyfits.open(f"{pathway}/{infile}")
        inhdr2 = inlist[2].header
        newhdr = inhdr2

        # Remove problematic header fields
        for field in ["INSTRUME", "TELESCOP", "CHECKSUM", "EXTNAME"]:
            newhdr.pop(field, None)

        imagedata = inlist[1].data
        nimages = len(imagedata)

        log("Starting to process images. This may take a few minutes.")
        for i in range(nimages):
            inimage = imagedata[i][4]
            bjd1 = imagedata[i][0]
            quality_flag = imagedata[i][8]
            tess_ffi = imagedata[i][11]

            if np.isnan(bjd1):
                log(f"Image {i + 1} skipped: lacks valid BJD timestamp.")
                continue
            elif quality_flag != 0:
                log(f"Image {i + 1} skipped: poor quality.")
                continue

            # Calculate mid-exposure BJD and HJD
            bjd = 2457000. + bjd1
            split = infile.split("_")
            ra = conversion([str(float(split[1]) / 15)])[0]
            dec = conversion([split[2]])[0]
            hjd = bary_to_helio(ra, dec, bjd, "greenwich")

            # Create output FITS file
            outimage = inimage
            outlist = pyfits.PrimaryHDU(outimage.astype("float32"), newhdr)
            outhdr = outlist.header

            # Update header metadata
            outhdr["LST_UPDT"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            outhdr["BJD_TDB"] = bjd
            outhdr["HJD"] = hjd.value
            outhdr["COMMENT"] = tess_ffi

            outfile = f"{pathway}/{outprefix}_tess_{i:05d}.fits"
            outlist.writeto(outfile, overwrite=True)
            log(f"Saved image {i + 1}: {outfile}")

        log("Finished processing all images.")
        inlist.close()
    except Exception as e:
        log(f"An error occurred during processing: {e}")
        raise


def bary_to_helio(ra, dec, bjd, obs_name):
    """Convert BJD to HJD for the given coordinates."""
    bary = Time(bjd, scale="tdb", format="jd")
    obs = EarthLocation.of_site(obs_name)
    star = SkyCoord(ra, dec, unit=(u.hour, u.deg))
    ltt = bary.light_travel_time(star, "barycentric", location=obs)
    guess = bary - ltt
    delta = (guess + guess.light_travel_time(star, "barycentric", obs)).jd - bary.jd
    guess -= delta * u.d
    ltt = guess.light_travel_time(star, "heliocentric", obs)
    return guess.utc + ltt
