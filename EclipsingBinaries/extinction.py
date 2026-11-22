"""
Looking up a star's reddening, E(B-V).

The reddening along the whole line of sight out of the Galaxy comes from the
IRSA Galactic Dust Reddening and Extinction service, using the Schlafly &
Finkbeiner (2011) recalibration of the Schlegel, Finkbeiner & Davis (1998)
dust map. That's the same value NED's extinction calculator gives.

Those maps count all the dust in that direction, and most eclipsing binaries
studied from the ground are only a few hundred parsecs away, in front of much
of it. So when Gaia DR3 has a good parallax for the star, the total is scaled
down for its distance with a dust layer 125 pc thick (Bonifacio, Caffau &
Molaro 2000):

    E(B-V)(d) = E(B-V)(total) * (1 - exp(-d |sin b| / 125 pc))

where b is the Galactic latitude. That's an approximation: dust is clumpy,
and near the Galactic plane a 3D dust map does better. The log gives both
values so either can be used.
"""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from ._log import make_logger

DUST_SCALE_HEIGHT_PC = 125.0
# A parallax has to be this many times its error to give a distance worth using
MIN_PARALLAX_OVER_ERROR = 5.0
GAIA_CONE_ARCSEC = 5.0


@dataclass
class Reddening:
    """
    total: E(B-V) along the whole line of sight (Schlafly & Finkbeiner 2011)
    total_error: its standard deviation over the area IRSA averages
    distance_pc: the Gaia distance used to scale it, or None
    scaled: E(B-V) in front of the star, or None without a distance
    latitude: Galactic latitude in degrees
    """
    total: float
    total_error: float
    latitude: float
    distance_pc: float | None = None
    scaled: float | None = None

    @property
    def best(self):
        """The value to use: scaled for distance when that could be done, otherwise the total."""
        return self.scaled if self.scaled is not None else self.total

    def message(self):
        text = (f"E(B-V) along the whole line of sight is {self.total:.4f} +/- {self.total_error:.4f} "
                f"(Schlafly & Finkbeiner 2011, from IRSA; Galactic latitude {self.latitude:.1f} deg).")
        if self.scaled is not None:
            text += (f" At the Gaia distance of {self.distance_pc:.0f} pc, with a dust layer "
                     f"{DUST_SCALE_HEIGHT_PC:.0f} pc thick, that's about {self.scaled:.4f}, which is used.")
        else:
            text += (" Gaia has no good distance for the star, so the whole line of sight is used; for a "
                     "nearby star that's more reddening than it really has.")
        return text


def scale_for_distance(total, distance_pc, latitude_deg, scale_height=DUST_SCALE_HEIGHT_PC):
    """The share of a line of sight's reddening in front of a star at distance_pc."""
    height = distance_pc * abs(np.sin(np.radians(latitude_deg)))
    if height == 0:
        # In the plane the formula gives nothing; all the dust out to the star counts, but
        # there's no way to say how much that is here, so the total is returned
        return float(total)
    return float(total * (1 - np.exp(-height / scale_height)))


def resolve(name=None, ra=None, dec=None):
    """
    A SkyCoord from RA and Dec (hours and degrees, as text such as "00:28:27.96"
    or numbers), or else from a name the CDS name resolver knows.
    """
    if ra not in (None, "") and dec not in (None, ""):
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    if name:
        return SkyCoord.from_name(name)
    raise ValueError("Give the star's RA and Dec, or a name the name resolver knows.")


def total_reddening(coord):
    """E(B-V) and its error along the whole line of sight, from IRSA's dust service."""
    from astroquery.ipac.irsa.irsa_dust import IrsaDust
    table = IrsaDust.get_query_table(coord, section="ebv")
    return float(table["ext SandF mean"][0]), float(table["ext SandF std"][0])


def gaia_distance(coord, write_callback=None, cancel_event=None):
    """
    The distance in parsecs from the Gaia DR3 parallax of the star at coord
    (the brightest source within 5 arcsec), or None if there's no parallax
    good enough to use.
    """
    from .gaia import _query_gaia_archive, _rank_by_separation
    radius = GAIA_CONE_ARCSEC / 3600
    g = _query_gaia_archive(f"""
        SELECT TOP 20 ra, dec, pmra, pmdec, parallax, parallax_error, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius}))
        """, write_callback=write_callback, cancel_event=cancel_event)
    if g is None or len(g) == 0:
        return None
    order, _separations = _rank_by_separation(g, coord.ra.deg, coord.dec.deg)
    if len(order) == 0:
        return None
    close = g[order]
    star = int(np.nanargmin(close.phot_g_mean_mag.value))
    parallax = float(close.parallax.to_value(u.mas)[star])
    error = float(close.parallax_error.to_value(u.mas)[star])
    if not (np.isfinite(parallax) and np.isfinite(error) and error > 0
            and parallax / error >= MIN_PARALLAX_OVER_ERROR):
        return None
    return 1000.0 / parallax


def look_up(name=None, ra=None, dec=None, write_callback=None, cancel_event=None):
    """
    Look up a star's reddening (see the module notes). Returns a Reddening, or
    None if canceled. Raises OSError if IRSA can't be reached, and ValueError
    if the star can't be found. A Gaia problem only means the total is used.
    """
    log = make_logger(write_callback)
    coord = resolve(name, ra, dec)
    if cancel_event is not None and cancel_event.is_set():
        return None
    try:
        total, error = total_reddening(coord)
    except Exception as e:  # astroquery raises several kinds for a service that's down
        raise OSError(f"IRSA's dust service couldn't be reached ({e}). Try again later, or enter E(B-V) "
                      "from NED's extinction calculator.") from e
    latitude = float(coord.galactic.b.deg)
    result = Reddening(total, error, latitude)
    if cancel_event is not None and cancel_event.is_set():
        return None
    try:
        distance = gaia_distance(coord, write_callback=write_callback, cancel_event=cancel_event)
    except Exception as e:
        log(f"Gaia couldn't be searched for the star's distance ({e}).")
        distance = None
    if distance is not None:
        result.distance_pc = distance
        result.scaled = scale_for_distance(total, distance, latitude)
    return result
