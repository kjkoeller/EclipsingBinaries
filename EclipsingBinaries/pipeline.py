"""
This script checks for new files in a directory every second for the start of a data pipeline.

Author: Kyle Koeller
Created: 06/15/2023
Last Edited: 06/16/2023
"""

from os import path, listdir
from time import time, sleep
import argparse

# from IRAF_Reduction import main  # testing purposes
from .IRAF_Reduction import main


def monitor_directory():
    # create the parser
    parser = argparse.ArgumentParser(description="Monitor a directory for new files.")

    # add the arguments
    parser.add_argument("raw_folder_path", type=str, help="The path of the folder where the images are going to.")
    parser.add_argument("new_folder_path", type=str, help="The path of the folder where the reduced images "
                                                            "and all files will go.")
    parser.add_argument("--time_threshold", type=int, default=10,
                        help="The time threshold in seconds. If no new file is added within this time, an alert is "
                             "raised. Default is 3600 seconds (1 hour).")
    parser.add_argument("--location", type=str, default="BSUO",
                        help="The location of the telescope (BSUO, SARA-KP, SARA-RM, SARA-CT). Default is BSUO.")
    parser.add_argument("--ra", type=str, default="00:00:00",
                        help="The right ascension of the target. Default is 00:00:00.")
    parser.add_argument("--dec", type=str, default="00:00:00",
                        help="The declination of the target (if negative -00:00:00). Default is 00:00:00.")

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
    current_latest_file = get_latest_file(folder_path=args.raw_folder_path)

    start_time = time()

    print("\n\nMonitoring directory for new files...\n")
    while True:
        sleep(1)  # pause for 1 second
        latest_file = get_latest_file(args.raw_folder_path)

        print("Latest file: " + latest_file)

        if latest_file == current_latest_file:
            # if no new file has been added
            elapsed_time = time() - start_time
            if elapsed_time > args.time_threshold:
                print("No new file has been added for the past " + str(args.time_threshold) + " seconds!\n")
                break
        else:
            # if a new file has been added
            start_time = time()
            current_latest_file = latest_file

    main(path=args.raw_folder_path, calibrated=args.new_folder_path, pipeline=True, location=args.location)
