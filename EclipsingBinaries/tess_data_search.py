"""
Look up the TESS data and download that data onto a local drive.
Author: Kyle Koeller
Created: 2/19/2022
Last Updated: 03/23/2023
"""

# import required packages
import astroquery.exceptions
from astroquery.mast import Tesscut
import astropy.units as u
from .tesscut import main as tCut
# from tesscut import main as tCut  # testing purposes
from os.path import exists


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

    # prints off the sector table to let the user know what sectors TESS has observed the object
    while True:
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
            print("The program will not exit.\n")
            exit()
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
