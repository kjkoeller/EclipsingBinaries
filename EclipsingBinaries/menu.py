"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 06/15/2023
"""

import pandas as pd

"""
# testing imports
from tess_data_search import main as data_search
from Night_Filters import main as night
from apass import comparison_selector as comp_select
from OConnell import main as oconnell
from color_light_curve import color_gui as gui
from IRAF_Reduction import main as IRAF
from OC_plot import main as data_fit
from gaia import target_star as gaia
from find_min import main as find_min
"""
from .tess_data_search import main as data_search
from .Night_Filters import main as night
from .apass import comparison_selector as comp_select
from .OConnell import main as oconnell
from .color_light_curve import color_gui as gui
from .IRAF_Reduction import main as IRAF
from .OC_plot import main as data_fit
from .gaia import target_star as gaia
from .find_min import main as find_min


def main():
    print("If you need a description of what each option does, please refer to the README for this packages GitHub page"
          " https://github.com/kjkoeller/EclipsingBinaries/")
    print("\nWhich program do you want to run?\n\n")

    options = ["IRAF Reduction", "Find Minimum (WIP)", "TESS Database Search/Download", "AIJ Comparison Star Selector",
               "BSUO or SARA/TESS Night Filters", "O-C Plotting", "Gaia Search", "O'Connel Effect", "Color Light Curve",
               "Close Program"]
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    total = pd.DataFrame({"Options": options,
                          "Numbers": numbers})

    while True:
        print(total.to_string(index=False))
        print()
        prompt = int(input("Please type out the number corresponding to the corresponding action: "))
        if prompt == 3:
            # tess data
            data_search()
        elif prompt == 2:
            # find minimum
            find_min()
        elif prompt == 5:
            # night filters for AIJ and TESS
            aij = input("BSUO/SARA or TESS or 'Go Back': ")
            if aij.lower() == "aij":
                night(0)
            elif aij.lower() == "tess":
                night(1)
            elif aij.lower() == "go back":
                pass
            else:
                print("\nYou did not enter AIJ or TESS please go back through the prompts again and enter AIJ or TESS.\n")
        elif prompt == 4:
            # comparison finder
            comp_select()
        elif prompt == 8:
            # o'connell effect
            oconnell()
        elif prompt == 9:
            # color light curve
            gui(False)
        elif prompt == 1:
            # iraf reduction
            IRAF()
        elif prompt == 6:
            # O-C plotting
            data_fit()
        elif prompt == 10:
            # close program
            break
        elif prompt == 7:
            gaia()
        else:
            print("\nYou have not entered any of the allowed entries, please try again.\n")


if __name__ == '__main__':
    main()
