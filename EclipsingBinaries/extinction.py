"""
Looking up a star's reddening, E(B-V).

Everything here uses R = A_V / E(B-V) = 3.1, so E(B-V) is always A_V / 3.1.

The first choice is the Bayestar19 3D dust map (Green, Schlafly, Finkbeiner et
al. 2019), which gives the reddening in front of a star at a given distance.
The distance comes from the star's Gaia DR3 parallax. Bayestar19 covers the
sky north of declination -30 degrees, and is asked over the web, so nothing
needs downloading. Its values are on the scale of the Schlegel, Finkbeiner &
Davis (1998) map, and are turned into E(B-V) with the factor of 0.884 the
authors give. That's the same as A_V / 3.1 with A_V worked out as for the 2D
map below (2.742 / 3.1 = 0.8845).

When Bayestar19 can't be used (a star south of -30 degrees, no good parallax,
or the service being down), the 2D map is used instead: the reddening along
the whole line of sight out of the Galaxy, from the IRSA Galactic Dust
Reddening and Extinction service. A_V for the Landolt V band is the map's
value at the star times 2.742, from Table 6 of Schlafly & Finkbeiner (2011)
for R_V = 3.1, which is the A_V NED's extinction calculator gives, and E(B-V)
is that divided by 3.1. It counts all the dust in that direction, so for a
nearby star it can be more than the star really has.

Bayestar19 is reached the same way the dustmaps package does it, but without
that package, because it needs healpy, which can't be installed on Windows.
"""

import base64
import json
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from ._log import make_logger

# A parallax has to be this many times its error to give a distance worth using
MIN_PARALLAX_OVER_ERROR = 5.0
GAIA_CONE_ARCSEC = 5.0

R_V = 3.1
# A_V (Landolt V) per unit of the Schlegel et al. (1998) map, Schlafly & Finkbeiner (2011) Table 6, R_V = 3.1
SFD_TO_AV = 2.742

BAYESTAR_URL = "http://argonaut.skymaps.info/api/v2/bayestar2019/query"
# Bayestar19 units to E(B-V), from the g - r relation in Green et al. (2019)
BAYESTAR_TO_EBV = 0.884
BAYESTAR_TIMEOUT = 30  # seconds
BAYESTAR_MIN_DEC = -30.0

BAYESTAR19 = "Bayestar19"
TOTAL_2D = "Schlafly & Finkbeiner (2011), whole line of sight"


@dataclass
class Reddening:
    """
    What was found for a star. value is the E(B-V) to use and method says
    where it came from (BAYESTAR19 or TOTAL_2D). The rest is kept
    for the log: the 2D map's value at the star and how much the map varies
    within 5 arcmin of it, the Gaia distance, and Bayestar19's 16th to 84th percentile
    range and whether the star is within the distances the map is reliable for
    along that sightline.
    """
    value: float
    method: str
    latitude: float
    total: float | None = None
    total_error: float | None = None
    distance_pc: float | None = None
    bayestar_range: tuple | None = None
    bayestar_reliable: bool | None = None

    @property
    def best(self):
        return self.value

    def message(self):
        lines = []
        if self.method == BAYESTAR19:
            low, high = self.bayestar_range
            lines.append(f"E(B-V) = {self.value:.4f} (16th to 84th percentile {low:.4f} to {high:.4f}), so "
                         f"A_V = {R_V * self.value:.4f}, in front of the star at its Gaia distance of "
                         f"{self.distance_pc:.0f} pc, from the Bayestar19 3D dust map. This is used.")
            if self.bayestar_reliable is False:
                lines.append("Warning: the star is outside the distances Bayestar19 is reliable for along this "
                             "sightline (usually too near for it to have many stars in front), so treat this "
                             "value with care.")
        if self.total is not None:
            text = (f"Along the whole line of sight, A_V at the star's position is {R_V * self.total:.4f} "
                    f"(Landolt V, Schlafly & Finkbeiner 2011, as NED gives it; Galactic latitude "
                    f"{self.latitude:.1f} deg), so E(B-V) = A_V / {R_V} = {self.total:.4f}. The map varies "
                    f"by {self.total_error:.4f} in E(B-V) (standard deviation) within 5 arcmin of the star; "
                    "that's how patchy the dust is nearby, not an uncertainty on the value.")
            if self.method == TOTAL_2D:
                text += (" This is used. It counts all the dust in that direction, so for a nearby star it "
                         "can be more reddening than the star really has.")
            lines.append(text)
        return " ".join(lines)


def resolve(name=None, ra=None, dec=None):
    """
    A SkyCoord from RA and Dec (hours and degrees, as text such as
    "00:28:27.96"), or else from a name the CDS name resolver knows.
    """
    if ra not in (None, "") and dec not in (None, ""):
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    if name:
        return SkyCoord.from_name(name)
    raise ValueError("Give the star's RA and Dec, or a name the name resolver knows.")


def total_reddening(coord):
    """
    E(B-V) along the whole line of sight from IRSA's dust service, as
    A_V / 3.1. A_V is NED's: the Schlegel et al. (1998) map's value in the
    pixel at the star times 2.742. Also returns the standard deviation of the
    map within 5 arcmin, in the same units, which shows how much the dust
    varies around the star rather than how well the value is known.
    """
    from astroquery.ipac.irsa.irsa_dust import IrsaDust
    table = IrsaDust.get_query_table(coord, section="ebv")
    to_ebv = SFD_TO_AV / R_V
    return float(table["ext SFD ref"][0]) * to_ebv, float(table["ext SFD std"][0]) * to_ebv


def _quantity(value, unit):
    return {"_type": "astropy.units.Quantity", "value": float(value), "unit": unit}


def _decode(d):
    """Turn the typed objects in a dustmaps web reply back into numpy values."""
    kind = d.get("_type")
    if kind == "np.dtype":
        descr = d["descr"]
        if isinstance(descr, str):
            return np.dtype(descr)
        return np.dtype([tuple(tuple(c) if isinstance(c, list) else c for c in col) for col in descr])
    if kind == "np.ndarray":
        if "data" in d:
            values = np.frombuffer(base64.b64decode(d["data"]), dtype=d["dtype"])
            return values.reshape(d["shape"])
        if "value" in d:
            return np.array(d["value"], dtype=d["dtype"])
        raise ValueError("Bayestar19 sent an array in a form this doesn't read.")
    if kind == "tuple":
        return tuple(d["items"])
    if kind == "astropy.units.Quantity":
        return u.Quantity(d["value"], unit=d["unit"])
    return d


def bayestar_reddening(coord, distance_pc, timeout=BAYESTAR_TIMEOUT):
    """
    Bayestar19's E(B-V) in front of a star at coord and distance_pc, as
    (median, (16th percentile, 84th percentile), reliable), where reliable says
    whether the distance is within the range the map is reliable for along
    that sightline. Returns None for stars the map doesn't cover. Raises
    OSError if the service can't be reached.
    """
    import requests
    galactic = coord.galactic
    request = {
        "coords": {"_type": "astropy.coordinates.SkyCoord", "frame": "galactic", "representation": "spherical",
                   "lon": _quantity(galactic.l.deg, "deg"), "lat": _quantity(galactic.b.deg, "deg"),
                   "distance": _quantity(distance_pc / 1000, "kpc")},
        "mode": "percentile", "pct": [16, 50, 84], "return_flags": True,
    }
    try:
        reply = requests.post(BAYESTAR_URL, data=json.dumps(request),
                              headers={"content-type": "application/json"}, timeout=timeout)
        reply.raise_for_status()
    except requests.RequestException as e:
        raise OSError(f"The Bayestar19 service couldn't be reached ({e}).") from e
    answer = json.loads(reply.text, object_hook=_decode)
    values, flags = answer if isinstance(answer, (tuple, list)) and len(answer) == 2 else (answer, None)
    values = np.asarray(values, dtype=float).ravel()
    if values.size != 3 or not np.all(np.isfinite(values)):
        return None
    low, median, high = values * BAYESTAR_TO_EBV
    reliable = None
    if flags is not None and getattr(flags, "dtype", None) is not None and flags.dtype.names:
        if "reliable_dist" in flags.dtype.names:
            reliable = bool(np.asarray(flags["reliable_dist"]).ravel()[0])
    return float(median), (float(low), float(high)), reliable


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
    None if canceled. Raises ValueError if the star can't be found, and
    OSError if neither Bayestar19 nor IRSA gave an answer.
    """
    log = make_logger(write_callback)

    def canceled():
        return cancel_event is not None and cancel_event.is_set()

    coord = resolve(name, ra, dec)
    latitude = float(coord.galactic.b.deg)
    if canceled():
        return None

    try:
        distance = gaia_distance(coord, write_callback=write_callback, cancel_event=cancel_event)
    except Exception as e:  # the archive raises several kinds when it's down
        log(f"Gaia couldn't be searched for the star's distance ({e}).")
        distance = None
    if canceled():
        return None
    if distance is None:
        log("Gaia has no parallax good enough to give the star's distance, so the 3D map can't be used.")

    result = None
    if distance is not None:
        if coord.dec.deg < BAYESTAR_MIN_DEC:
            log(f"The star is south of declination {BAYESTAR_MIN_DEC:.0f}, outside Bayestar19, so the 2D map "
                "is used.")
        else:
            try:
                found = bayestar_reddening(coord, distance)
            except (OSError, ValueError) as e:
                log(f"{e} The 2D map is used instead.")
            else:
                if found is not None:
                    median, spread, reliable = found
                    result = Reddening(median, BAYESTAR19, latitude, distance_pc=distance,
                                       bayestar_range=spread, bayestar_reliable=reliable)
                elif not canceled():
                    log("Bayestar19 has no value at the star's position, so the 2D map is used.")
    if canceled():
        return None

    # The 2D map: the answer when Bayestar19 couldn't give one, and a comparison when it did
    try:
        total, error = total_reddening(coord)
    except Exception as e:  # astroquery raises several kinds for a service that's down
        if result is not None:
            log(f"IRSA's dust service couldn't be reached for the 2D comparison ({e}).")
            return result
        raise OSError(f"Neither Bayestar19 nor IRSA's dust service gave an answer ({e}). Try again later, or "
                      "enter E(B-V) from NED's extinction calculator.") from e
    if result is not None:
        result.total, result.total_error = total, error
        return result
    return Reddening(total, TOTAL_2D, latitude, total=total, total_error=error, distance_pc=distance)
