"""
Author: Kyle Koeller
Created: 11/11/2020
Last Updated: 10/10/2026

Combines the nightly files Astro ImageJ (AIJ) writes for one filter into a
single file covering every night, ready for Find Minimum, O'Connell, Peranso
or PHOEBE.

Each night's file needs HJD, Source_AMag_T1 and Source_AMag_Err_T1 columns.
If every night also has rel_flux_T1 and rel_flux_err_T1, a flux file is
written alongside the magnitude file, and BJD_TDB can be included as an extra
first column.
"""

from pathlib import Path

import pandas as pd

from ._log import make_logger

# Column names in AIJ's measurement tables, and the names used in the output
TIME_COLUMN = "HJD"
BJD_COLUMN = "BJD_TDB"
MAG_COLUMNS = {"Source_AMag_T1": "AMag", "Source_AMag_Err_T1": "AMag Error"}
FLUX_COLUMNS = {"rel_flux_T1": "rel flux", "rel_flux_err_T1": "rel flux error"}

# Our own photometry writes the magnitude error under this name
COLUMN_ALIASES = {"Source_AMag_T1_Error": "Source_AMag_Err_T1"}


class NightFileError(ValueError):
    """A night's file can't be used, with a message saying which and why."""


def read_night(path):
    """
    Read one night's AIJ file (tab or comma separated, with a header row)
    and return it with the column names tidied up.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise NightFileError(f"{path.name} couldn't be read: {e}") from e
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    required = [TIME_COLUMN, *MAG_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise NightFileError(f"{path.name} is missing the column(s) {', '.join(missing)}. "
                             "Use the measurements file AIJ saves for each night.")
    return df


def combine_nights(files, output_folder, output_name, include_bjd=False,
                   write_callback=None, cancel_event=None):
    """
    Combine nightly AIJ files for one filter into a single file.

    Parameters
    ----------
    files : list of str or Path
        One file per night, all for the same filter.
    output_folder : str or Path
        Where to save the combined files. Created if needed.
    output_name : str
        Start of the output file names, for example "NSVS_B".
    include_bjd : bool
        Add BJD_TDB as the first column. Every night needs a BJD_TDB column.
    write_callback : callable, optional
        Where log messages go (the GUI's log, or print).
    cancel_event : threading.Event, optional
        Stops the work between files when set.

    Returns
    -------
    dict or None
        "magnitudes" and "flux" (None when not every night has flux) file
        paths and the number of "rows" written, or None if canceled.

    The files are tab separated without a header: (BJD_TDB,) HJD, magnitude,
    error in <output_name>_magnitudes.txt, and (BJD_TDB,) HJD, relative flux,
    error in <output_name>_flux.txt. Rows are sorted by time.
    """
    log = make_logger(write_callback)
    files = [Path(f) for f in files]
    if not files:
        raise NightFileError("Add at least one night's file.")
    output_name = str(output_name).strip()
    if not output_name:
        raise NightFileError("An output name is required.")

    nights = []
    for path in files:
        if cancel_event is not None and cancel_event.is_set():
            log("Night Filters was canceled.")
            return None
        if not path.is_file():
            raise NightFileError(f"{path} doesn't exist.")
        df = read_night(path)
        if include_bjd and BJD_COLUMN not in df.columns:
            raise NightFileError(f"{path.name} has no {BJD_COLUMN} column. "
                                 "Turn off the BJD option or re-export it from AIJ with BJD.")
        log(f"Read {len(df)} rows from {path.name}")
        nights.append(df)

    with_flux = [all(c in df.columns for c in FLUX_COLUMNS) for df in nights]
    write_flux = all(with_flux)
    if any(with_flux) and not write_flux:
        lacking = ", ".join(p.name for p, has in zip(files, with_flux) if not has)
        log(f"Warning: no flux columns in {lacking}, so only the magnitude file is written.")

    time_cols = ([BJD_COLUMN] if include_bjd else []) + [TIME_COLUMN]
    keep = time_cols + list(MAG_COLUMNS) + (list(FLUX_COLUMNS) if write_flux else [])
    combined = pd.concat([df[keep] for df in nights], ignore_index=True)
    combined = combined.apply(pd.to_numeric, errors="coerce")

    bad = combined.isna().any(axis=1)
    if bad.any():
        log(f"Warning: skipped {int(bad.sum())} row(s) with missing or non-numeric values.")
        combined = combined[~bad]
    combined = combined.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    if combined.empty:
        raise NightFileError("None of the rows had usable values.")

    folder = Path(output_folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)

    mag_path = folder / f"{output_name}_magnitudes.txt"
    combined[time_cols + list(MAG_COLUMNS)].rename(columns=MAG_COLUMNS).to_csv(
        mag_path, index=False, header=False, sep="\t")
    log(f"Saved {len(combined)} rows of magnitudes from {len(files)} night(s) to {mag_path}")

    flux_path = None
    if write_flux:
        flux_path = folder / f"{output_name}_flux.txt"
        combined[time_cols + list(FLUX_COLUMNS)].rename(columns=FLUX_COLUMNS).to_csv(
            flux_path, index=False, header=False, sep="\t")
        log(f"Saved {len(combined)} rows of relative flux to {flux_path}")

    return {"magnitudes": str(mag_path), "flux": str(flux_path) if flux_path else None,
            "rows": len(combined)}


def _ask(prompt):
    answer = input(prompt).strip()
    if answer.lower() == "close":
        raise SystemExit(0)
    return answer


def main(include_bjd=False):
    """
    Ask for the nightly files in the terminal and combine them. The combined
    files are saved in the current folder. Type 'Close' at any prompt to stop.
    """
    print("\nEach night should have a file like 2018.09.18.APASS.B_datasubset.dat, all for the "
          "same filter.\nType 'Close' at any prompt to stop.\n")
    while True:
        count = _ask("Number of nights you have: ")
        if count.isdigit() and int(count) > 0:
            break
        print("Enter a whole number above zero, or 'Close'.\n")

    files = []
    for night in range(int(count)):
        while True:
            path = Path(_ask(f"Night {night + 1} file path: ").strip('"')).expanduser()
            if path.is_file():
                files.append(path)
                break
            print("That file doesn't exist. Try again.\n")

    name = _ask("Output name, without a file extension: ")
    try:
        combine_nights(files, Path.cwd(), name, include_bjd=bool(include_bjd))
    except NightFileError as e:
        print(f"\nError: {e}\n")
        return 1
    return 0


if __name__ == '__main__':
    main()
