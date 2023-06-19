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

## Descriptions

Check the wiki for some help with specific programs. More will be added at a later date.

### menu.py

The main program that can call all the other programs given a user's prompts. When using this package, always run the main.py first as this will connect to all the following programs.

### Night\_Filters.py

This program is meant to make the process of collecting the different filters from AIJ and TESS excel spreadsheets faster.
The user enters however many nights they have and the program goes through and checks those text files for the
different columns for,HJD, Amag, and Amag error for the B and V filters.
The program will also calculate the R magnitude from the rel flux of T1.
There are error catching statements within the program so if the user mistypes, the program will not crash and
close on them (hopefully).

### apass.py

With this program you compile a list of stars that are 30 arcmin box around a set of coordinates that the user enters and output a file that gives RA, DEC, B, V, R\_c, and the respective band pass errors.

From the paper listed in the program (https://arxiv.org/pdf/astro-ph/0609736.pdf) this program finds the Cousins R value from the band passes found and gives this in an output file.

This program then overlays all found stars onto a real science image to show the user what the field will look like and then will create a RADEC file for each of the filters (Johnson B, Johnson V, and Cousins R) that can be used in AIJ.

### OConnell.py

Applies analysis of the O'Connel effect presented by this paper: https://app.aavso.org/jaavso/article/3511/

### color\_light\_curve.py

Allows the user to create a color light curve for both B-V and V-R bands and gather the values automatically by way of a GUI. The R file is optional and not required but B and V band-pass files are.

### IRAF\_Reduction.py

This program automatiaclly reduces calibration images without the need for the user to do any sort of input except where the images are and where they will go. This program, at the moment, does technically reduce science images all the way with bias, dark, and flats, BUT is not fully tested to make sure it is comparable to IRAF legacy.

### OC\_plot.py

Creates a plot of O-C data that differentiates between primary and secondary eclipses. This program also creates a residual plot from a given fit that the user enters. What this program does not do, is actually solve for Times of Minimum and only utlizes already found values from Dr. Robert C. Berrington's C program.

### tess\_data\_search.py

Searches the TESS database for sector data for a specific TIC system. Lets the user know if there are no sectors available and if there are, automatically downloads the data to a specified folder that user designates via `tesscut.py`

### gaia.py

Searches Gaia data for parameters for a given system entered by a user such as: parallax, distance, and effective temperature of the hotter star.
