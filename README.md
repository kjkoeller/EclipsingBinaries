[![Python OS 3.8, 3.9, 3.10](https://github.com/kjkoeller/Binary_Star_Research_Package/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/kjkoeller/Binary_Star_Research_Package/actions/workflows/ci_tests.yml)
[![Documentation Status](https://readthedocs.org/projects/eclipsingbinaries/badge/?version=latest)](https://eclipsingbinaries.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/EclipsingBinaries.svg)](https://badge.fury.io/py/EclipsingBinaries)
[![GitHub release](https://img.shields.io/github/v/release/kjkoeller/Variable_Star_Research_Package)](https://github.com/kjkoeller/Variable_Star_Research_Package/releases/)
![GitHub](https://img.shields.io/github/license/kjkoeller/Variable_Star_Research_Package)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9cd9a15e47ab4ed7b78071d096ea099d)](https://www.codacy.com/gh/kjkoeller/EclipsingBinaries/dashboard?utm_source=github.com\&utm_medium=referral\&utm_content=kjkoeller/EclipsingBinaries\&utm_campaign=Badge_Grade)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

# Binary Star Research Package

A Python project for Ball State University's Variable Star Research Group. Check the [Wiki](https://github.com/kjkoeller/EclipsingBinaries/wiki) for more detailed descriptions than the ones below and for usage details of each program.

***

## Sponsor

If you would like to sponsor this project go [here](https://github.com/sponsors/kjkoeller) and click the sponsor button.

***

## Installation and Usage

To install type the following,

    pip install EclipsingBinaires

Once installed, in the command line type the following:

    EclipsingBinaries

This will run the `menu.py` file and will initiate all other programs for usage.
Once installed using pip, you can just go to a command line and type `EclipsingBinaries` to start the program each time.

To check the version you have,

    pip show EclipsingBinaries

this will show numerous things, but you want to look at the version and make sure it is up to date.

If your version is not the most recent version then in order to update type the following,

    pip install --upgrade EclipsingBinaries

### Pipeline

To use the pipeline functionality type the following:

    EB_pipeline -h

This will print out all the options that are available to edit and change. The `-i` and the `-o` are required for the script to run. Otherwise, the script will crash.

***

## Dependencies

*   python >=3.7
*   astropy>=5.1.1
*   astroquery>=0.4.6
*   ccdproc>=2.4.0
*   matplotlib>=3.3.1
*   numpy>=1.19.1
*   pandas>=1.1.0
*   PyAstronomy>=0.18.0
*   scipy>=1.5.2
*   statsmodels>=0.13.5
*   tqdm>=4.64.1
*   numba>=0.56.3
*   seaborn>=0.12.2
*   pyia>=1.3

***

## Documentation

You can find the documentation at this site https://eclipsingbinaries.readthedocs.io/en/latest/?badge=latest# and any questions can be talked about in the discussions page or in an issue.
