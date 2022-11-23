#!/usr/bin/env python

"""Setup script for the package."""

import setuptools
import sys


MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setuptools.setup(
    description="Binary Star Package for Ball State University's Astronomy Research Group",
    url='https://github.com/kjkoeller/Binary_Star_Research_Package',
    author='Kyle Koeller',

    packages=setuptools.find_packages(),

    entry_points={'console_scripts': []},

    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
    ],

    install_requires=open("requirements.txt").readlines(),
)
