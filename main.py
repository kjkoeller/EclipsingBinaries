"""
The main program helps centralize all the other programs into one selection routine that can be run and call all
other programs.

Author: Kyle Koeller
Created: 8/29/2022
Last Updated: 8/30/2022
"""

import tess_data_search
import AIJ_Night_Filters
import TESS_Night_Filters
import APASS_AIJ_comparison_selector
import OConnell


def main():
    print()
    print("Which program do you want to run?")
    print("TESS database search(1), AIJ/TESS Night Filters(2), AIJ Comp Selector(3), or O'Connel Effect(4)")
    print("")

    while True:
        prompt = input("Please type out the number corresponding to the corresponding action: ")
        if prompt == 1:
            tess_data_search.main()
            break
        elif prompt == 2:
            aij = input("AIJ or TESS: ")
            if aij.lower() == "aij":
                AIJ_Night_Filters.main(0)
                break
            elif aij.lower() == "tess":
                TESS_Night_Filters.main(0)
                break
            else:
                print("You did not enter AIJ or TESS please go back through the prompts again and enter AIJ or TESS.")
        elif prompt == 3:
            APASS_AIJ_comparison_selector.main()
            break
        elif prompt == 4:
            OConnell.main()
            break
        else:
            print("You have not entered any of the allowed entries, please try again.")
            print()


if __name__ == '__main__':
    main()
