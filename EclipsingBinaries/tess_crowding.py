"""
How much light from nearby stars falls on a TESS target.

TESS pixels are 21 arcseconds across, so stars well apart on the sky can share
a pixel, and their light adds to the target's. That makes eclipses look
shallower than they are, and an eclipsing neighbour can look like the
target's own eclipse. SPOC's PDCSAP flux has this taken out using the TIC;
other light curves and photometry on TESS cutouts may not.

The estimate here adds up the TIC stars around the target, weighted by
brightness, within one and two pixels. It ignores where the pixel edges fall
and the shape of the PSF, so it's a guide to whether to check, not a
correction to apply.
"""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from ._log import make_logger

PIXEL_ARCSEC = 21.0
# Stars this close to the position are taken to be the target itself
TARGET_MATCH_ARCSEC = 5.0
# Above this share of the target's light from neighbours within two pixels, the log warns
WARN_FRACTION = 0.10
# Stars fainter than the target by more than this add too little to matter
FAINT_LIMIT = 7.0


@dataclass
class Crowding:
    """
    Light from other stars, as a fraction of the target's own: 0.25 means the
    neighbours add a quarter as much light again.
    """
    target_tmag: float | None
    within_one: float           # within one pixel
    within_two: float           # within two pixels
    neighbours: int             # stars counted within two pixels

    @property
    def crowded(self):
        return self.within_two >= WARN_FRACTION

    def message(self):
        if self.target_tmag is None:
            return ("No TIC star was found at the target's position, so how much light nearby "
                    "stars add couldn't be worked out.")
        if self.neighbours == 0:
            return (f"No TIC star brighter than Tmag {self.target_tmag + FAINT_LIMIT:.1f} lies within "
                    f"two TESS pixels ({2 * PIXEL_ARCSEC:.0f} arcsec) of the target.")
        text = (f"Other stars add about {100 * self.within_one:.0f}% to the target's light within one "
                f"TESS pixel ({PIXEL_ARCSEC:.0f} arcsec) and {100 * self.within_two:.0f}% within two "
                f"({self.neighbours} TIC star{'s' if self.neighbours != 1 else ''}, target Tmag "
                f"{self.target_tmag:.2f}).")
        if self.crowded:
            text += (" Warning: eclipses will look shallower in TESS data that isn't corrected for "
                     "this, such as photometry on TESS cutouts, and an eclipse could belong to a "
                     "neighbour. SPOC's PDCSAP flux is corrected; check the others against "
                     "ground-based data.")
        return text


def neighbour_flux(target, stars, tmags):
    """
    Work out the Crowding for a target from the stars around it.

    :param target: SkyCoord of the target
    :param stars: SkyCoord of the catalogue stars around it, the target included
    :param tmags: their TESS magnitudes
    """
    tmags = np.asarray(tmags, dtype=float)
    if len(tmags) == 0:
        return Crowding(None, 0.0, 0.0, 0)
    separation = target.separation(stars).arcsec
    usable = np.isfinite(tmags)
    candidates = np.flatnonzero(usable & (separation <= TARGET_MATCH_ARCSEC))
    if len(candidates) == 0:
        return Crowding(None, 0.0, 0.0, 0)
    # The brightest star at the position is the one TESS sees as the target
    target_index = candidates[np.argmin(tmags[candidates])]
    target_tmag = float(tmags[target_index])

    others = usable & (np.arange(len(tmags)) != target_index) & (tmags < target_tmag + FAINT_LIMIT)
    ratio = np.where(others, 10 ** (-0.4 * (np.where(usable, tmags, 99.0) - target_tmag)), 0.0)
    one = separation <= PIXEL_ARCSEC
    two = separation <= 2 * PIXEL_ARCSEC
    return Crowding(target_tmag, float(ratio[one].sum()), float(ratio[two].sum()),
                    int((others & two).sum()))


def tic_crowding(coord=None, name=None, write_callback=None, cancel_event=None):
    """
    Look up the TIC stars around a target at MAST and work out its Crowding.
    Give either its coordinates (a SkyCoord) or a name MAST can resolve,
    such as "TIC 12345678". Returns None if canceled or MAST can't be
    reached; the reason goes to the log, since this is only advice.
    """
    log = make_logger(write_callback)
    if cancel_event is not None and cancel_event.is_set():
        return None
    try:
        from astroquery.mast import Catalogs
        radius = 3 * PIXEL_ARCSEC * u.arcsec
        if coord is not None:
            table = Catalogs.query_region(coord, radius=radius, catalog="TIC")
        else:
            table = Catalogs.query_object(name, radius=radius, catalog="TIC")
    except Exception as error:  # network and service errors come in many types
        log(f"The TIC couldn't be searched for stars near the target ({error}), so how much light "
            "they add to the TESS data isn't known.")
        return None
    if table is None or len(table) == 0:
        return Crowding(None, 0.0, 0.0, 0)
    stars = SkyCoord(np.asarray(table["ra"], dtype=float) * u.deg, np.asarray(table["dec"], dtype=float) * u.deg)
    if coord is None:
        # query_object centres the search on the name's position; dstArcSec is from there
        nearest = int(np.nanargmin(np.asarray(table["dstArcSec"], dtype=float)))
        coord = stars[nearest]
    tmags = np.ma.filled(np.ma.asarray(table["Tmag"], dtype=float), np.nan)
    return neighbour_flux(coord, stars, tmags)


def crowdsap(fits_path):
    """
    SPOC's own estimate from a light curve file's header: the fraction of the
    flux in its aperture that comes from the target (CROWDSAP), or None if
    the file doesn't have one.
    """
    from astropy.io import fits
    try:
        with fits.open(fits_path) as hdus:
            for hdu in hdus[1:3]:
                value = hdu.header.get("CROWDSAP")
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return float(value)
    except (OSError, IndexError):
        pass
    return None


def describe_crowdsap(values):
    """A sentence about the CROWDSAP values from several sectors, or "" when there are none."""
    values = [v for v in values if v is not None]
    if not values:
        return ""
    low, high = min(values), max(values)
    share = f"{100 * low:.0f}%" if abs(high - low) < 0.005 else f"{100 * low:.0f}% to {100 * high:.0f}%"
    text = (f"SPOC puts {share} of the light in its apertures down to the target (CROWDSAP); "
            "its PDCSAP flux, used here, has the rest taken out.")
    if low < 1 - WARN_FRACTION:
        text += " SAP flux and other light curves of this star may still include it."
    return text
