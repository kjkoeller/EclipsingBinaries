"""
One small file per star, holding what's been worked out about it so far.

Every tool asks for some of the same things (the star's name, position,
period, epoch), and each one's results are written somewhere different. The
star file keeps them together: each tool that knows a star's name adds what
it found, with where it saved it, and panels can fill in what's already
known instead of asking again. It's also a record of how each number was
arrived at, for writing up.

The files are JSON, one per star, in ~/.EclipsingBinaries/stars (or the
folder the ECLIPSINGBINARIES_STARS environment variable names), so they
follow the star whichever folders the data ends up in. "NSVS 896797" and
"nsvs_896797" are the same star.

Keeping the file up to date is never allowed to stop the real work: if it
can't be written, the reason goes to the log and the tool carries on.

From a terminal, ``python -m EclipsingBinaries.star_file "NSVS 896797"``
prints what's saved for a star, and with no name lists the stars.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ._log import make_logger

# The values a star file holds, and what each one is
FIELDS = {
    "ra": "right ascension, J2000, as entered (hours)",
    "dec": "declination, J2000, as entered (degrees)",
    "period": "orbital period in days",
    "period_error": "1 sigma error on the period, in days",
    "epoch": "a time of primary minimum (T0)",
    "epoch_error": "error on the epoch, in days",
    "time_system": "time system of the epoch and the minima",
    "ebv": "E(B-V) reddening",
}
HISTORY_LENGTH = 200  # older history lines are dropped past this many


def stars_folder():
    """Where the star files are kept."""
    return Path(os.environ.get("ECLIPSINGBINARIES_STARS",
                               Path.home() / ".EclipsingBinaries" / "stars")).expanduser()


def star_key(name):
    """The file name stem for a star: lower case, with runs of anything else made one underscore."""
    key = re.sub(r"[^a-z0-9+.-]+", "_", str(name).strip().lower()).strip("_")
    if not key:
        raise ValueError("A star needs a name.")
    return key


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StarFile:
    """
    What's known about one star. values holds the FIELDS that have been
    found, files maps each step (such as "period" or "minima") to the files
    it wrote, and history lists what was added, when, and by which step.
    """
    name: str
    values: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @property
    def path(self):
        return stars_folder() / f"{star_key(self.name)}.json"

    @classmethod
    def load(cls, name):
        """
        The star's file, or an empty one if nothing has been saved for it yet.
        The name it was first saved under is kept, however it's spelled this time.
        """
        star = cls(str(name).strip())
        if star.path.is_file():
            data = json.loads(star.path.read_text(encoding="utf-8"))
            star.name = data.get("name") or star.name
            star.values = dict(data.get("values", {}))
            star.files = {step: list(paths) for step, paths in data.get("files", {}).items()}
            star.history = list(data.get("history", []))
        return star

    def get(self, key, default=None):
        return self.values.get(key, default)

    def update(self, step, files=(), **values):
        """
        Add what one step found and save. None values are ignored, so a step
        can pass everything it might know. Files are kept per step, newest
        last, without repeats.
        """
        unknown = set(values) - set(FIELDS)
        if unknown:
            raise ValueError(f"Star files don't hold {', '.join(sorted(unknown))}.")
        changed = {key: value for key, value in values.items()
                   if value is not None and self.values.get(key) != value}
        self.values.update(changed)
        new_files = [str(Path(f)) for f in files if f]
        added = []
        if new_files:
            listed = self.files.setdefault(step, [])
            added = [f for f in new_files if f not in listed]
            for path in new_files:
                if path in listed:
                    listed.remove(path)
                listed.append(path)
        if changed or added:
            entry = {"time": _now(), "step": step}
            if changed:
                entry["values"] = changed
            if added:
                entry["files"] = added
            self.history = (self.history + [entry])[-HISTORY_LENGTH:]
        self.save()
        return changed

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "values": self.values, "files": self.files, "history": self.history}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def summary(self):
        """A few lines saying what's known, for the log or a terminal."""
        lines = [self.name]
        for key in FIELDS:
            if key in self.values:
                lines.append(f"  {key}: {self.values[key]}")
        for step, paths in self.files.items():
            lines.append(f"  {step}: {paths[-1]}" + (f" (and {len(paths) - 1} earlier)" if len(paths) > 1 else ""))
        if len(lines) == 1:
            lines.append("  nothing saved yet")
        return "\n".join(lines)


def record(name, step, files=(), write_callback=None, **values):
    """
    Add a step's results to a star's file, if the star has a name. Never
    raises: a problem with the star file is logged and the work goes on.
    Returns the StarFile, or None if nothing was recorded.
    """
    if not name or not str(name).strip():
        return None
    try:
        star = StarFile.load(name)
        star.update(step, files=files, **values)
        return star
    except (OSError, ValueError, TypeError) as error:
        make_logger(write_callback)(f"The star file for {name} couldn't be updated ({error}); "
                                    "nothing else is affected.")
        return None


def known(name):
    """The values saved for a star, or {} if there are none or the file can't be read."""
    if not name or not str(name).strip():
        return {}
    try:
        return dict(StarFile.load(name).values)
    except (OSError, ValueError):
        return {}


def all_stars():
    """The names of every star with a file, sorted."""
    names = []
    for path in sorted(stars_folder().glob("*.json")):
        try:
            names.append(json.loads(path.read_text(encoding="utf-8")).get("name") or path.stem)
        except (OSError, ValueError):
            continue
    return sorted(names, key=str.lower)


def main(argv=None):
    """Print a star's file, or the list of stars."""
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        names = all_stars()
        print("\n".join(names) if names else f"No star files yet in {stars_folder()}.")
        return 0
    star = StarFile.load(" ".join(args))
    print(star.summary())
    print(f"\nSaved in {star.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
