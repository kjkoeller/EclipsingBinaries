[![GitHub release](https://img.shields.io/github/v/release/kjkoeller/Variable_Star_Research_Package)](https://github.com/kjkoeller/Variable_Star_Research_Package/releases/)
![GitHub](https://img.shields.io/github/license/kjkoeller/Variable_Star_Research_Package)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9cd9a15e47ab4ed7b78071d096ea099d)](https://www.codacy.com/gh/kjkoeller/Binary_Star_Research_Package/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kjkoeller/Binary_Star_Research_Package&amp;utm_campaign=Badge_Grade)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

# Binary Star Research Package
A Python project for Ball State University's Variable Star Research Group

--------------------------

## Dependencies
https://github.com/kjkoeller/Variable_Star_Research_Package/wiki/Installation

Once inside the main directory type the following to automatically install the required packages below and check for a minimum python version.

```
python setup.py install
```
* python >=3.5
* astropy>=5.1.1
* astroquery>=0.4.6
* ccdproc>=2.4.0
* matplotlib>=3.3.1
* numpy>=1.19.1
* pandas>=1.1.0
* PyAstronomy>=0.18.0
* scipy>=1.5.2
* statsmodels>=0.13.5
* tqdm>=4.64.1

--------------------------

## Descriptions (v0.7.2)

### main.py
The main program that can call all the other programs given a user's prompts. When using this package, always run the main.py first as this will connect to all the following programs.

### AIJ_Night_Filters.py
This program is meant to make the process of collecting the different filters from AIJ excel spreadsheets faster.
The user enters however many nights they have and the program goes through and checks those text files for the
different columns for,HJD, Amag, and Amag error for the B and V filters.
The program will also calculate the R magnitude from the rel flux of T1.
There are error catching statements within the program so if the user mistypes, the program will not crash and
close on them (hopefully)

### TESS_Night_Filters.py
Same thing as AIJ filters but finds the appropriate values given TESS data rather than SARA or BSUO data.

### apass.py
With this program you call upon the cousins_r.py and APASS_catalog_finder.py programs to compile a list of stars that are close to what AIJ found off Simbad and output a file that gives RA, DEC, B, V, R_c, and the respective band pass errors.

From the paper listed in the program (https://arxiv.org/pdf/astro-ph/0609736.pdf) this program finds the Cousins R value from the band passes found in the APASS_catalog_finder and gives this output file to the APASS_AIJ_comparison_selector program.

Given an RA and DEC from simbad, finds stars 40 arc min box around those coordinates from the APASS catalog. This program then outputs these stars and numerous band passes to a text file that will be saved and used by the cousins_r.py program.

This program overlays both APASS and AIJ comparison stars onto a real science image. This allows for the best comparison of comparison stars between AIJ and APASS.

### tess_data_search.py
Look up the TESS data via TIC ID number and output the sector and camera number. Then download all the sector data for that TIC ID number to obtain pixel data that can then be used in Astro ImageJ to get flux data.

### OConnell.py
Applies analysis of the O'Connel effect presented by this paper: https://app.aavso.org/jaavso/article/3511/

### color_light_curve.py
Allows the user to create a color light curve for both B-V and V-R bands and gather the values automatically by way of a GUI. The R file is optional and not required but B and V band-pass files are.

### IRAF_Reduction.py
This program automatiaclly reduces calibration images without the need for the user to do any sort of input except where the images are and where they will go. This program, at the moment, does technically reduce science images all the way with bias, dark, and flats, BUT is not fully tested to make sure it is comparable to IRAF legacy.

### OC_plot.py
Creates a plot of O-C data that differentiates between primary and secondary eclipses. This program also creates a residual plot from a given fit that the user enters. What this program does not do, is actually solve for Times of Minimum and only utlizes already found values from Dr. Robert C. Berrington's C program.
