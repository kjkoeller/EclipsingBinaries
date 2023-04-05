"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.
Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 04/05/2023
"""

"""
# testing imports
from tess_data_search import main as data_search
from Night_Filters import main as night
from apass import comparison_selector as comp_select
from OConnell import main as oconnell
from color_light_curve import color_gui as gui
from IRAF_Reduction import main as IRAF
from OC_plot import main as data_fit
from gaia import main
"""
from .tess_data_search import main as data_search
from .Night_Filters import main as night
from .apass import comparison_selector as comp_select
from .OConnell import main as oconnell
from .color_light_curve import color_gui as gui
from .IRAF_Reduction import main as IRAF
from .OC_plot import main as data_fit
from .gaia import target_star as gaia


def main():
    print("If you need a description of what each option does, please refer to the README for this packages GitHub page"
          " https://github.com/kjkoeller/Binary_Star_Research_Package")
    print("\nWhich program do you want to run?\n\n")

    while True:
        print(
            "IRAF Reduction(1), TESS Database Search/Download(2), AIJ Comparison Star Selector(3), "
            "BSUO or SARA/TESS Night Filters(4), O-C Plotting(5), Gaia Search(6), O'Connel Effect(7), "
            "Color Light Curve(8), Close Program(9)\n")
        prompt = int(input("Please type out the number corresponding to the corresponding action: "))
        if prompt == 2:
            # tess data
            data_search()
        elif prompt == 4:
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
        elif prompt == 3:
            # comparison finder
            comp_select()
        elif prompt == 7:
            # o'connell effect
            oconnell()
        elif prompt == 8:
            # color light curve
            gui(False)
        elif prompt == 1:
            # iraf reduction
            IRAF()
        elif prompt == 5:
            # O-C plotting
            data_fit()
        elif prompt == 9:
            # close program
            break
        elif prompt == 6:
            gaia()
        else:
            print("\nYou have not entered any of the allowed entries, please try again.\n")


if __name__ == '__main__':
    main()
