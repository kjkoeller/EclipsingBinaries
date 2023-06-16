"""
This script checks for new files in a directory every second for the start of a data pipeline.

Author: Kyle Koeller
Created: 06/15/2023
Last Edited: 06/15/2023
"""
from os import path, listdir
from time import time, sleep

from IRAF_Reduction import main  # testing purposes
# from .IRAF_Reduction import main


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


def pipeline():
    """
    The pipeline function checking for new files in a directory

    :return: None
    """
    # specify the directory you want to traverse
    folder_path = "your_folder_path"
    # time threshold in seconds (1 hour in this example)
    time_threshold = 3600
    # store the current latest file
    current_latest_file = get_latest_file(folder_path)

    start_time = time()

    while True:
        sleep(1)  # pause for 1 second
        latest_file = get_latest_file(folder_path)

        if latest_file == current_latest_file:
            # if no new file has been added
            elapsed_time = time() - start_time
            if elapsed_time > time_threshold:
                print("No new file has been added for the past hour!\n")
        else:
            # if a new file has been added
            start_time = time()
            current_latest_file = latest_file

    main(path=folder_path, pipeline=True)
