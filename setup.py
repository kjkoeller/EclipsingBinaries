#!/usr/bin/env python

"""Setup script for the package."""

from setuptools import setup, find_packages  # Always prefer setuptools over distutils
import sys
import codecs
import os.path

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

MINIMUM_PYTHON_VERSION = 3, 8


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()
version = get_version("EclipsingBinaries/__init__.py")

setup(
    version=version,
    name="EclipsingBinaries",
    description="Binary Star Package for Ball State University's Astronomy Research Group",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kjkoeller/EclipsingBinaries',
    author='Kyle Koeller',

    packages=find_packages(),

    entry_points={'console_scripts': [
        'EclipsingBinaries = EclipsingBinaries.menu:main'
    ],
    },

    license="MIT",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'astropy>=5.1.1',
        'astroquery>=0.4.6',
        'ccdproc>=2.4.0',
        'matplotlib>=3.3.1',
        'numpy>=1.19.1',
        'pandas>=1.1.0',
        'PyAstronomy>=0.18.0',
        'scipy>=1.5.2',
        'statsmodels>=0.13.5',
        'tqdm>=4.64.1',
        'numba>=0.56.3',
        'seaborn>=0.12.2',
        'pyia>=1.3',
    ],
    
    extras_require={
        'testing': [
            'pytest>=7.2.2',
        ],
    },

    include_package_data=True,
    package_data={
        'EclipsingBinaries': ['tests/*'],
    },

)
