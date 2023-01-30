"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.
Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 01/30/2023
"""

from .tess_data_search import main as data_search
from .Night_Filters import main as night
from .apass import comparison_selector as comp_select
from.OConnell import main as oconnell
from .color_light_curve import color_gui as gui
from .IRAF_Reduction import main as IRAF
from .OC_plot import main as data_fit


def main():
    print("If you need a description of what each option does, please refer to the README for this packages GitHub page"
          " https://github.com/kjkoeller/Binary_Star_Research_Package")
    print("\nWhich program do you want to run?\n")

    while True:
        print(
            "TESS Database Search/Download(1), AIJ/TESS Night Filters(2), AIJ Comparison Star Selector(3), O'Connel Effect(4), "
            "Color Light Curve(5), IRAF Reduction(6), O-C Plotting(7), Close Program(8)\n")
        print("")
        prompt = int(input("Please type out the number corresponding to the corresponding action: "))
        if prompt == 1:
            data_search()
        elif prompt == 2:
            aij = input("AIJ or TESS or 'Go Back': ")
            if aij.lower() == "aij":
                night(0)
            elif aij.lower() == "tess":
                night(1)
            elif aij.lower() == "go back":
                pass
            else:
                print("\nYou did not enter AIJ or TESS please go back through the prompts again and enter AIJ or TESS.\n")
        elif prompt == 3:
            comp_select()
        elif prompt == 4:
            oconnell()
        elif prompt == 5:
            gui(False)
        elif prompt == 6:
            IRAF()
        elif prompt == 7:
            data_fit()
        elif prompt == 8:
            break
        else:
            print("\nYou have not entered any of the allowed entries, please try again.\n")


if __name__ == '__main__':
    main()
