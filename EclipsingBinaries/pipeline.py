"""
This script checks for new files in a directory every second for the start of a data pipeline.

Author: Kyle Koeller
Created: 06/15/2023
Last Edited: 06/17/2023
"""

from os import path, listdir
from time import time, sleep
import argparse

from .apass import comparison_selector
from .IRAF_Reduction import main


def monitor_directory():
    # create the parser
    parser = argparse.ArgumentParser(description="Monitor a directory for new files and start a data pipeline.")

    # add the arguments
    parser.add_argument("input", metavar="Input File", type=str, help="The path of the folder where the images are going to.")
    parser.add_argument("output", metavar="Output File", type=str, help="The path of the folder where the reduced images "
                                                                        "and all files will go.")
    parser.add_argument("--time", metavar="Time Threshold", type=int, default=3600,
                        help="The time threshold in seconds. If no new file is added within this time, an alert is "
                             "raised. Default is 3600 seconds (1 hour).")
    parser.add_argument("--loc", metavar="Location", type=str, default="BSUO",
                        help="The location of the telescope (BSUO, CTIO, LaPalma, KPNO). Default is BSUO.")
    parser.add_argument("--ra", type=str, default="00:00:00",
                        help="The right ascension of the target. Default is 00:00:00.", required=True)
    parser.add_argument("--dec", type=str, default="00:00:00",
                        help="The declination of the target (if negative -00:00:00). Default is 00:00:00.", required=True)
    parser.add_argument("--name", metavar="Object Name", type=str, default="target",
                        help="The name of the target. Default is target.")

    # parse the arguments
    args = parser.parse_args()

    def get_latest_file(folder_path):
        """
        Get the latest file in the directory

        :param folder_path: folder path entered by the user
        :return:
        """
        files = [path.join(folder_path, f) for f in listdir(folder_path) if
                 path.isfile(path.join(folder_path, f))]
        if files:
            latest_file = max(files, key=path.getctime)
            return latest_file
        else:
            return None

    # store the current latest file
    current_latest_file = get_latest_file(folder_path=args.input)

    start_time = time()

    print("\n\nMonitoring directory for new files...\n")
    while True:
        sleep(1)  # pause for 1 second
        latest_file = get_latest_file(args.input)

        if latest_file == current_latest_file:
            # if no new file has been added
            elapsed_time = time() - start_time
            if elapsed_time > args.time:
                print("No new file has been added for the past " + str(args.time) + " seconds!\n")
                break
        else:
            # if a new file has been added
            print("Latest file: " + latest_file)
            start_time = time()
            current_latest_file = latest_file

    print("Starting data reduction.\n")
    main(path=args.input, calibrated=args.output, pipeline=True, location=args.loc)
    print("\n\nStarting comparison star selection.")
    comparison_selector(ra=args.ra, dec=args.dec, pipeline=True, folder_path=args.output, obj_name=args.name)
