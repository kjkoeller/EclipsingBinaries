"""
Keeping track of which time system a set of times is in.

Light curves come with times in different systems: AIJ's HJD column is
heliocentric JD in UTC, its BJD_TDB column is barycentric JD in TDB, and
other programs write other things. The two differ by about a minute (mostly
the TT - UTC offset), which matters for O-C work, so a set of times of minimum
shouldn't be labelled as one system when it's really in another.

Nothing here converts between systems; tesscut.bary_to_helio does that for
TESS data. This only names the systems, works out which one a file's column
is in, and tags files of minima so O-C Plotting can label its tables
correctly. When the system isn't known, the labels just say "Time".
"""

from pathlib import Path

HJD_UTC = "HJD_UTC"
BJD_TDB = "BJD_TDB"
BJD_UTC = "BJD_UTC"
JD_UTC = "JD_UTC"
SYSTEMS = (HJD_UTC, BJD_TDB, BJD_UTC, JD_UTC)

# Column names, upper case, and the system each one is in. AIJ's plain "HJD"
# column is in UTC. Plain "JD", "BJD" and "TIME" don't say enough to know.
_COLUMNS = {
    "HJD": HJD_UTC, "HJD_UTC": HJD_UTC, "HJD-UTC": HJD_UTC, "JD-HELIO": HJD_UTC,
    "BJD_TDB": BJD_TDB, "BJD-TDB": BJD_TDB,
    "BJD_UTC": BJD_UTC, "BJD-UTC": BJD_UTC,
    "JD_UTC": JD_UTC, "JD-UTC": JD_UTC,
}

_NAMES = {HJD_UTC: "HJD (UTC)", BJD_TDB: "BJD (TDB)", BJD_UTC: "BJD (UTC)", JD_UTC: "JD (UTC)"}
_LATEX = {HJD_UTC: r"$HJD_{\rm UTC}$", BJD_TDB: r"$BJD_{\rm TDB}$",
          BJD_UTC: r"$BJD_{\rm UTC}$", JD_UTC: r"$JD_{\rm UTC}$"}

# The first line of a file of minima, when its time system is known
_TAG = "# Times of minimum in "


def from_column(name):
    """The system a column of this name is in, or None if the name doesn't say."""
    return _COLUMNS.get(str(name).strip().upper())


def describe(system):
    """A short name for labels, such as "HJD (UTC)", or "Time" when unknown."""
    return _NAMES.get(system, "Time")


def latex(system):
    """A LaTeX column heading for times in this system."""
    return _LATEX.get(system, "Time of minimum")


def check(system):
    """Return system if it's one of SYSTEMS (or None), otherwise raise ValueError."""
    if system is not None and system not in SYSTEMS:
        raise ValueError(f"Unknown time system {system!r}; use one of {', '.join(SYSTEMS)}.")
    return system


def tag_line(system):
    """The line that starts a file of minima in this system, or "" when it isn't known."""
    return f"{_TAG}{system}\n" if system in SYSTEMS else ""


def system_in_file(path):
    """
    The time system a file of minima or O-C values says it's in, from its
    first line, or None if the file doesn't say (files from before this was
    recorded, or times whose system wasn't known).
    """
    try:
        with open(Path(path), encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip()
    except OSError:
        return None
    if first.startswith(_TAG.strip()):
        system = first[len(_TAG.strip()):].strip()
        return system if system in SYSTEMS else None
    return None


def combined(systems, names=None):
    """
    The one system a set of files is in, ignoring files that don't say.
    Raises ValueError if two of them say different things, because times of
    minimum in different systems can't go in the same O-C diagram.

    :param systems: a system (or None) for each file
    :param names: optional file names, used in the error message
    """
    systems = list(systems)
    names = list(names) if names else [f"file {i + 1}" for i in range(len(systems))]
    known = [(name, system) for name, system in zip(names, systems) if system is not None]
    if len({system for _name, system in known}) > 1:
        listed = ", ".join(f"{name} is {describe(system)}" for name, system in known)
        raise ValueError(f"These times are in different time systems ({listed}). "
                         "Convert them to one system before using them together.")
    return known[0][1] if known else None
