"""
Look up the TESS data and download that data onto a local drive.
Author: Kyle Koeller
Created: 2/19/2022
Last Updated: 01/30/2022
"""

# import required packages
import astroquery.exceptions
from astroquery.mast import Tesscut
import astropy.units as u
from .tesscut import main as tCut


def main():
    """
    This function allows the user to enter a TIC ID to be entered, and it also makes sure that number is valid or exists.
    This program will also list off the sector data to be downloaded for cross-referencing if needed.
    :return: Downloaded pixel data in the form of .fits files to be extracted later
    """

    # While loops checks to make sure that the user has entered a valid TIC number that can be found.
    print("If you want to close this program type 'Close' in the following prompt.\n")
    while True:
        try:
            system_name = input("Enter in the TIC-ID given in SIMBAD (TIC 468293391): ")
            sector_table = Tesscut.get_sectors(objectname=system_name)
            break
        except astroquery.exceptions.ResolverError:
            print("\nThe TIC number you entered is invalid or there is no data for this given system.\n")

    if system_name.lower() == "close":
        exit()

    # prints off the sector table to let the user know what sectors TESS has observed the object
    print(sector_table)
    for i in sector_table["sector"]:
        # downloads the pixel file data that can then be analyzed with AIJ
        print()
        print("Starting download of Sector " + str(i))
        manifest = Tesscut.download_cutouts(objectname=system_name, size=[30, 30] * u.arcmin, sector=i)
        tCut(manifest)
        print("Finished downloading Sector " + str(i))
    print("Finished downloading all sector data related to " + system_name + "\n")


if __name__ == '__main__':
    main()
