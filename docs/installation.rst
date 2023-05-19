EclipsingBinaries Package
=========================

.. contents:: Table of Contents
    :depth: 2

Introduction
------------

EclipsingBinaries is a binary star research package for Ball State University's Variable Star Research Group. The purpose of this package is to increase the efficiency of the group's analysis process of more basic steps. The package also adds some new statistical analysis that used to not be common for the group until this package.

Installation
------------

.. code-block:: bash

    pip install EclipsingBinaries

Then to upgrade to the next version

.. code-block:: bash

    pip install --upgrade EclipsingBinaries

Dependencies
------------

- python >=3.7
- astropy>=5.1.1
- astroquery>=0.4.6
- ccdproc>=2.4.0
- matplotlib>=3.3.1
- numpy>=1.19.1
- pandas>=1.1.0
- PyAstronomy>=0.18.0
- scipy>=1.5.2
- statsmodels>=0.13.5
- tqdm>=4.64.1
- numba>=0.56.3
- seaborn>=0.12.2
- pyia>=1.3

Modules
-------

menu.py
~~~~~~~

The main program that can call all the other programs given a user's prompts.

Night_Filters.py
~~~~~~~~~~~~~~~~

This program is meant to make the process of collecting the different filters from AIJ and TESS excel spreadsheets faster.

apass.py
~~~~~~~~

With this program, you compile a list of stars that are 30 arcmin box around a set of coordinates.
