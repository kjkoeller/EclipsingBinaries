************
Installation
************

Requirements
============

EclipsingBinaries needs Python 3.12 or newer. pip installs these packages automatically:

- astropy>=6.0
- astroquery>=0.4.6
- ccdproc>=2.4.0
- matplotlib>=3.8.0
- numpy>=1.26
- pandas>=2.1.1
- PyAstronomy>=0.18.1
- scipy>=1.11.2
- statsmodels>=0.14
- tqdm>=4.64.1
- pyia>=1.4
- photutils>=1.8.0
- tkinterdnd2>=0.6.1

The list in ``pyproject.toml`` is the authoritative one.

tkinter
-------

The GUI also needs tkinter, which comes with Python but is left out of some installs:

- **macOS with Homebrew Python** — install the formula that matches your Python version::

    brew install python-tk@3.12

- **Ubuntu/Debian** — install the system package::

    sudo apt install python3-tk

- **Windows and macOS python.org installers** — already included. On Windows, keep
  "tcl/tk and IDLE" checked in the installer.

Everything except the GUI works without tkinter, so scripts and the ``EB_pipeline``
command run on headless machines too.

Installing EclipsingBinaries
============================

To install EclipsingBinaries with `pip <https://pip.pypa.io/en/latest/>`_, simply run::

    pip install EclipsingBinaries

This installs two commands: ``EclipsingBinaries`` for the GUI (see :ref:`EB`) and
``EB_pipeline`` for automated reductions (see :ref:`pipeline`).

To check which version you have::

    pip show EclipsingBinaries

Updating
--------

To update to the latest version, simply run::

    pip install --upgrade EclipsingBinaries

To install a specific version, run::

    pip install EclipsingBinaries==[version]

Development Installation
------------------------

To install the development version directly from GitHub, along with the test and
documentation tools::

    git clone https://github.com/kjkoeller/EclipsingBinaries.git
    cd EclipsingBinaries
    pip install -e ".[test,docs]"

Run the tests with::

    pytest

The tests don't need an internet connection. To run them the way CI does, including the
oldest supported dependency versions, use ``tox``.

Build these docs with::

    sphinx-build -b html docs docs/_build/html
