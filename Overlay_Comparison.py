"""
Author: Kyle Koeller
Created: 11/18/2022
Last Updated: 11/22/2022

Creates an overlay of potential comparison stars on a science image to easily compare between RADEC AIJ file and what
APASS_catalog_finder.py finds
"""

import matplotlib.pyplot as plt
import APASS_catalog_finder as apass

from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
import pandas as pd
import numpy as np


def main(catalog, radec):
    # NSVS_254037-S001-R004-C001-Empty-R-B2.fts
    fits_file = input("Enter file pathway to one of your image files: ")

    header_data_unit_list = fits.open(fits_file)
    image = header_data_unit_list[0].data
    header = header_data_unit_list[0].header

    df = pd.read_csv(catalog, header=None, skiprows=[0], sep="\t")
    dh = pd.read_csv(radec, header=None, skiprows=7)

    index_num = list(df[0])
    ra_catalog = list(df[1])
    dec_catalog = list(df[2])
    ra_radec = list(dh[0])
    dec_radec = list(dh[1])

    ra_cat_new = (np.array(apass.splitter(ra_catalog)) * 15) * u.deg
    dec_cat_new = np.array(apass.splitter(dec_catalog)) * u.deg
    ra_radec_new = (np.array(apass.splitter(ra_radec)) * 15) * u.deg
    dec_radec_new = np.array(apass.splitter(dec_radec)) * u.deg

    wcs = WCS(header)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection=wcs)
    plt.imshow(image, origin='lower', cmap='cividis', aspect='equal', vmin=300, vmax=1500)
    plt.xlabel('RA')
    plt.ylabel('Dec')

    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted')

    offset = 0.1
    ax.scatter(ra_cat_new, dec_cat_new, transform=ax.get_transform('fk5'), s=200,
               edgecolor='red', facecolor='none', label="Potential Comparison Stars")
    ax.scatter(ra_radec_new, dec_radec_new, transform=ax.get_transform('fk5'), s=200,
               edgecolor='green', facecolor='none', label="AIJ Comparison Stars")

    # plt.text(100, 100, "Hello")
    count = 0
    for x, y in zip(ra_cat_new, dec_cat_new):
        px, py = wcs.wcs_world2pix(x, y, 0.)
        plt.annotate(str(index_num[count]), xy=(px+30, py-50), color="white")
        count += 1

    plt.gca().invert_xaxis()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=False, ncol=2)
    plt.show()


# main("APASS_254037_Catalog.txt", "NSVS_254037-B.radec")
