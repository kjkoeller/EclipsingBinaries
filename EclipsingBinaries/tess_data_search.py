"""
Look up the TESS data and download that data onto a local drive.
Author: Kyle Koeller
Created: 2/19/2022
Last Updated: 04/26/2023
"""

# import required packages
import astroquery.exceptions
from astroquery.mast import Tesscut
from .tesscut import main as tCut
# from tesscut import main as tCut  # testing purposes
from os.path import exists
import pandas as pd

from astropy import units as u
import pkg_resources


def main():
    """
    This function allows the user to enter a TIC ID to be entered, and it also makes sure that number is valid or exists.
    This program will also list off the sector data to be downloaded for cross-referencing if needed.
    :return: Downloaded pixel data in the form of .fits files to be extracted later
    """

    # While loops checks to make sure that the user has entered a valid TIC number that can be found.
    print("\nIf you want to close this program type 'Close' in the following prompt.\n")
    while True:
        try:
            system_name = input("Enter in the TIC-ID given in SIMBAD (TIC 468293391) or the word 'Close' "
                                "to close the program: ")
            sector_table = Tesscut.get_sectors(objectname=system_name)
            break
        except astroquery.exceptions.ResolverError:
            print("\nThe TIC number you entered is invalid or there is no data for this given system.\n")

    if system_name.lower() == "close":
        exit()

    filename = pkg_resources.resource_filename(__name__, 'tess_ccd_info.txt')
    dc = pd.read_csv(filename, header=None, sep="\t", skiprows=[0])

    gain = dc[6]  # gain for the individual camera/ccd
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

    # sector_table.add_columns(["Gain"])
    # prints off the sector table to let the user know what sectors TESS has observed the object
    while True:
        print("\n The read noise for the cameras is between 7-11 electrons/pixel.")
        print(sector_table)
        print("\nDo you want to download " + "\033[1m" + "\033[93m" + "ALL" + "\033[00m" +
              " the TESS data?")
        download_ans = input("Type 'Yes' to download all Sector data or type 'No' to specify a sector that you want. "
                             "Or type 'Close' to leave: ")
        # prints the word 'ALL' in bold '\033[1m' and in yellow '\033[93m' must return to normal with '\033[0m'
        if download_ans.lower() == "yes":
            print("\nWhen TESS data starts the initial download, it downloads, essentially, a big ZIP file with "
                  "all the individual images inside. Below, please enter the entire file path.")
            print("Example output file path: 'C:\\folder1\\TESS_data\n'")

            for i in sector_table["sector"]:
                download(system_name, i)
                print("\nFinished downloading Sector " + str(i))
        elif download_ans.lower() == "no":
            sector_num = int(input("Which Sector would you like to download: "))
            download(system_name, sector_num)
        elif download_ans.lower() == 'close':
            print("The program will now exit.\n")
            break
        else:
            print("\nPlease enter 'Yes' or 'No' only.\n")

    print("Finished downloading all sector data related to " + system_name + "\n")


def download(system_name, i):
    """
    Download the sector data

    Parameters
    ----------
    system_name - name of the system to download sector data for
    i - sector number

    Returns
    -------
    None
    """
    while True:
        download_path = input("Please enter a file pathway where you want the sector ZIP file to go: ")
        if exists(download_path):
            break
        else:
            print("\nThe file pathway you entered does not exist. Please try again.\n")
    # downloads the pixel file data that can then be analyzed with AIJ
    print("\n\nStarting download of Sector " + str(i))
    manifest = Tesscut.download_cutouts(objectname=system_name, size=[30, 30] * u.arcmin, sector=i,
                                        path=download_path)
    tCut(manifest, download_path)


if __name__ == '__main__':
    main()
