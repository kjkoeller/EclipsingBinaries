"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 11/02/2022
"""

import tess_data_search
import AIJ_Night_Filters
import TESS_Night_Filters
import APASS_AIJ_comparison_selector
import OConnell
import color_light_curve


def main():
    print()
    print("Which program do you want to run?")

    while True:
        print("TESS Database Search/Download(1), AIJ/TESS Night Filters(2), AIJ Comparison Star Selector(3), O'Connel Effect(4), "
              "Color Light Curve(5), Close Program(6)")
        print("")
        prompt = int(input("Please type out the number corresponding to the corresponding action: "))
        if prompt == 1:
            tess_data_search.main()
        elif prompt == 2:
            aij = input("AIJ or TESS or 'Go Back': ")
            if aij.lower() == "aij":
                AIJ_Night_Filters.main(0)
            elif aij.lower() == "tess":
                TESS_Night_Filters.main(0)
            elif aij.lower() == "go back":
                pass
            else:
                print("You did not enter AIJ or TESS please go back through the prompts again and enter AIJ or TESS.")
        elif prompt == 3:
            APASS_AIJ_comparison_selector.main()
        elif prompt == 4:
            OConnell.main()
        elif prompt == 5:
            color_light_curve.color_gui(False)
        elif prompt ==  6:
            break
        else:
            print("You have not entered any of the allowed entries, please try again.")
            print()


if __name__ == '__main__':
    main()
