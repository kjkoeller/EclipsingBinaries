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
    print("TESS database search, AIJ/TESS Night Filters, AIJ Comp Selector, or O'Connel Effect")
    print("")

    while True:
        prompt = input("Please type out exactly the option you want as stated above: ")
        if prompt.lower() == "tess database search":
            tess_data_search.main()
            break
        elif prompt.lower() == "aij/tess night filters":
            aij = input("AIJ or TESS: ")
            if aij.lower() == "aij":
                AIJ_Night_Filters.main(0)
                break
            elif aij.lower() == "tess":
                TESS_Night_Filters.main(0)
                break
            else:
                print("You did not enter AIJ or TESS please go back through the prompts again and enter AIJ or TESS.")
        elif prompt.lower() == "aij comp selector":
            APASS_AIJ_comparison_selector.main()
            break
        elif prompt.lower() == "o'connel effect":
            OConnell.main()
            break
        else:
            print("You have not entered any of the allowed entries, please try again.")
            print()


if __name__ == '__main__':
    main()
