************
Installation
************

Requirements
============

EclipsingBinaries has the following requirements:

- Python>=3.12
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

Installing EclipsingBinaries
============================

To install EclipsingBinaries with `pip <https://pip.pypa.io/en/latest/>`_, simply run::

    pip install EclipsingBinaries

Updating
--------

To update to the latest version, simply run::

    pip install --upgrade EclipsingBinaries

To install a specific version, run::

    pip install EclipsingBinaries==[version]

Development Installation
------------------------

To install the development version directly from GitHub::

    git clone https://github.com/kjkoeller/EclipsingBinaries.git
    cd EclipsingBinaries
    pip install -e .
