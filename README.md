[![Python 3.12](https://github.com/kjkoeller/EclipsingBinaries/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/kjkoeller/EclipsingBinaries/actions/workflows/ci_tests.yml)
[![Documentation Status](https://readthedocs.org/projects/eclipsingbinaries/badge/?version=latest)](https://eclipsingbinaries.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/EclipsingBinaries.svg)](https://badge.fury.io/py/EclipsingBinaries)
[![GitHub release](https://img.shields.io/github/v/release/kjkoeller/EclipsingBinaries)](https://github.com/kjkoeller/EclipsingBinaries/releases/)
![GitHub](https://img.shields.io/github/license/kjkoeller/EclipsingBinaries)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9cd9a15e47ab4ed7b78071d096ea099d)](https://www.codacy.com/gh/kjkoeller/EclipsingBinaries/dashboard?utm_source=github.com&utm_medium=referral&utm_content=kjkoeller/EclipsingBinaries&utm_campaign=Badge_Grade)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

# EclipsingBinaries

EclipsingBinaries is a Python project for faster analysis of eclipsing binary star systems. The package can currently reduce data, find comparison stars from the APASS catalog, calculate and plot O-C values, find the color index and effective temperature, O’Connell effect parameters, download TESS data, and perform interactive multi-aperture photometry with auto-suggested, optimized radii.

-----

## Documentation

You can find the documentation at this [site](https://eclipsingbinaries.readthedocs.io/en/latest/?badge=latest#) and any questions can be talked about in the discussions page or in an issue.

-----

## OS and Python Versions Stable On

The list of OS’s and Python versions listed below have been tested to be able to build the package on.

- Macos- 3.12
- Ubuntu- 3.12
- Windows- 3.12

The minimum working Python version is 3.12 and as of right now no versions lower work with all aspects of this package.

For MacOS HomeBrew users, you will need to be built against TK 8.6 since TK 9.0 breaks drag and drop extension.
For Ubuntu/Debian users, install the system tkinter package before starting the app:

```
sudo apt install python3-tk
```

-----

## Installation and Usage

To install type the following,

```
pip install EclipsingBinaries
```

Once installed, in the command line type the following:

```
EclipsingBinaries
```

This will run the `menu.py` file and will initiate all other programs for usage.
Once installed using pip, you can just go to a command line and type `EclipsingBinaries` to start the program each time.

To check the version you have,

```
pip show EclipsingBinaries
```

this will show numerous things, but you want to look at the version and make sure it is up to date.

If your version is not the most recent version then in order to update type the following,

```
pip install --upgrade EclipsingBinaries
```

### Pipeline

To use the pipeline functionality type the following:

```
EB_pipeline -h
```

This will print out all the options that are available to edit and change. The `-i` and the `-o` are required for the script to run. Otherwise, the script will crash.

-----

## Dependencies

- python >=3.12
- astropy>=6.0
- astroquery>=0.4.6
- ccdproc>=2.4.0
- matplotlib>=3.7.1
- numpy>=1.26
- pandas>=2.1.0
- PyAstronomy>=0.18.1
- scipy>=1.11.2
- statsmodels>=0.14
- tqdm>=4.64.1
- numba>=0.59.0
- seaborn>=0.12.2
- pyia>=1.4
- photutils>=1.8.0
- tkinterdnd2>=0.4.3
- tkmacosx>=1.0.4 (macOS only)
- python3-tk (Ubuntu/Debian system package)

-----
