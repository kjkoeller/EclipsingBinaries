"""
Look up the TESS data and download that data onto a local drive.
Author: Kyle Koeller
Created: 2/19/2022
Last Updated: 12/09/2024
"""

# import required packages
import astroquery.exceptions
from astroquery.mast import Tesscut
# from .tesscut import process_tess_cutout
from tesscut import process_tess_cutout  # testing purposes
from os.path import exists
import pandas as pd
import os

from astropy import units as u
import pkg_resources


def run_tess_search(system_name, download_all, specific_sector, download_path, write_callback, cancel_event):
    """
    Search for TESS data and download the specified sectors with cancel functionality.

    Parameters
    ----------
    system_name : str
        The TIC ID or system name.
    download_all : bool
        Whether to download all available sectors.
    specific_sector : int, optional
        The specific sector to download if `download_all` is False.
    download_path : str, optional
        The directory to save downloaded files.
    write_callback : function, optional
        Callback to log progress or errors to the GUI.
    """

    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)
    try:
        # Check for cancellation
        if cancel_event.is_set():
            log("Task canceled before starting.")
            return

        # Validate system name
        log(f"Searching for sectors for system: {system_name}")
        sector_table = Tesscut.get_sectors(objectname=system_name)

        if not sector_table:
            raise ValueError(f"No TESS data found for system {system_name}.")

        # The ccd info comes from this paper:
        # https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/active-missions/tess/_documents/TESS_Instrument_Handbook_v0.1.pdf
        filename = pkg_resources.resource_filename(__name__, 'tess_ccd_info.txt')
        dc = pd.read_csv(filename, header=None, sep="\t", skiprows=[0])

        gain = dc[3]  # videoscale, gain for the individual camera/ccd
        tess_camera = dc[0]  # camera number
        tess_ccd = dc[1]  # ccd number
        # slice = dc[2]  # could be used in the future

        sector_camera = []
        sector_ccd = []
        for count, val in enumerate(sector_table["camera"]):
            sector_camera.append(val)
            sector_ccd.append(sector_table["ccd"][count])

        # create a tuple form of the sector and TESS camera/ccd data
        list_gain = []
        a = list(zip(tess_camera, tess_ccd))  # tess cam and ccd's
        b = list(zip(sector_camera, sector_ccd))  # sect cam and ccd's

        # compare the TESS and sector information
        for _, sect in enumerate(b):
            for y, tess in enumerate(a):
                if sect == tess:
                    list_gain.append(gain[y])

        # splits up list_gain into lists by every 1, 2, 3, or 4th value since there are 4 slices for each camera and ccd
        A = list_gain[::4]
        B = list_gain[1::4]
        C = list_gain[2::4]
        D = list_gain[3::4]

        # append the gain values to the sector table list
        sector_table.add_column(A, name="A gain")
        sector_table.add_column(B, name="B gain")
        sector_table.add_column(C, name="C gain")
        sector_table.add_column(D, name="D gain")

        # prints off the sector table to let the user know what sectors TESS has observed the object
        log("\nThe read noise for the cameras is between 7-11 electrons/pixel.")

        # Log the sector table
        formatted_table = "\n".join(sector_table.pformat(show_name=True, max_width=-1, align="^"))
        log("Sector Table:\n" + formatted_table)

        # Check specific_sector logic
        if not download_all:
            log("Downloading all available sectors.")
            for sector in sector_table["sector"]:
                if cancel_event.is_set():
                    log(f"Task canceled while processing Sector {sector}.")
                    return
                download_sector(system_name, sector, download_path, write_callback, cancel_event)
        else:
            if specific_sector:
                log(f"Downloading specific sector: {specific_sector}.")
                download_sector(system_name, specific_sector, download_path, write_callback, cancel_event)
            else:
                log("Error: Specific sector is not specified.")
                raise ValueError("Specific sector is not specified.")

        log("Finished downloading all sector data related to " + system_name + "\n")
    except Exception as e:
        log(f"An error occurred during TESS Database Search: {e}")


def download_sector(system_name, sector, download_path, write_callback, cancel_event):
    """
    Download TESS sector data for a given system.

    Parameters
    ----------
    system_name : str
        The TIC ID or system name.
    sector : int
        The sector number to download.
    download_path : str
        The path to save downloaded data.
    write_callback: function
        to write log messages to the GUI
    """

    def log(message):
        """Log messages to the GUI if callback provided, otherwise print"""
        if write_callback:
            write_callback(message)
        else:
            print(message)

    try:
        if not exists(download_path):
            raise FileNotFoundError(f"The path '{download_path}' does not exist.")

        # Create sector-specific directory if it doesn't exist
        sector_path = os.path.join(download_path, str(sector))
        os.makedirs(sector_path, exist_ok=True)  # Create the directory if it doesn't exist
        log(f"Directory created or already exists: {sector_path}")

        log(f"Starting download for Sector {sector}.")
        manifest = Tesscut.download_cutouts(
            objectname=system_name, size=[30, 30] * u.arcmin, sector=sector, path=sector_path)
        process_tess_cutout(
            search_file=manifest,  # Replace with actual file
            pathway=sector_path,
            sector=sector,
            outprefix=f"{system_name}_S{sector}_",
            write_callback=write_callback,
            cancel_event=cancel_event
        )
        log(f"Completed download for Sector {sector}.")
    except Exception as e:
        log(f"Failed to download Sector {sector}: {e}")
        raise
