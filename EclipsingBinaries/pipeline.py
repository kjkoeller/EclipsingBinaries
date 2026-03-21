"""
This script checks for new files in a directory and starts a data pipeline
once the observation session ends (no new files for a configurable timeout).

Author: Kyle Koeller
Created: 06/15/2023
Last Edited: 03/20/2026
"""

import logging
import signal
import sys
from os import path, listdir
from pathlib import Path
from time import time, sleep
import argparse

from .apass import comparison_selector
from .IRAF_Reduction import main as IRAF
from .multi_aperture_photometry import main as multiple_AP


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(log_file=None):
    """Configure console and optional file logging with timestamps."""
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def get_latest_file(folder_path):
    """
    Return the most recently modified file in folder_path, or None if empty.

    Uses mtime (modification time) rather than ctime because ctime is not
    reliable as a creation time on Linux.

    Parameters
    ----------
    folder_path : str
        Path to the directory to scan.

    Returns
    -------
    str or None
    """
    files = [
        path.join(folder_path, f)
        for f in listdir(folder_path)
        if path.isfile(path.join(folder_path, f))
    ]
    if files:
        return max(files, key=path.getmtime)
    return None


def count_files(folder_path):
    """Return the number of files currently in folder_path."""
    return sum(
        1 for f in listdir(folder_path)
        if path.isfile(path.join(folder_path, f))
    )


# ---------------------------------------------------------------------------
# Directory monitor
# ---------------------------------------------------------------------------
def monitor_directory(
    input_dir,
    timeout,
    poll_interval=1,
    log_interval=60,
):
    """
    Block until no new file has appeared in input_dir for `timeout` seconds.

    Parameters
    ----------
    input_dir : str
        Directory to watch for new files.
    timeout : int
        Seconds of inactivity before monitoring stops.
    poll_interval : int
        How often (seconds) to check for new files. Default 1.
    log_interval : int
        How often (seconds) to log a "still waiting" heartbeat. Default 60.

    Returns
    -------
    bool
        True if monitoring ended due to timeout (pipeline should start),
        False if interrupted by signal.
    """
    log = logging.getLogger(__name__)
    current_latest = get_latest_file(input_dir)
    start_time = time()
    last_log_time = time()

    log.info("Monitoring %s for new files (timeout: %ds)...", input_dir, timeout)

    while True:
        sleep(poll_interval)
        latest = get_latest_file(input_dir)

        if latest != current_latest:
            # A new file arrived — reset the idle clock
            log.info("New file detected: %s", latest)
            current_latest = latest
            start_time = time()
            last_log_time = time()
        else:
            elapsed = time() - start_time
            remaining = timeout - elapsed

            # Periodic heartbeat so the user knows the script is still alive
            if time() - last_log_time >= log_interval:
                log.info(
                    "Still waiting... %.0fs elapsed, %.0fs until timeout "
                    "(%d files in directory)",
                    elapsed, remaining, count_files(input_dir)
                )
                last_log_time = time()

            if elapsed >= timeout:
                log.info(
                    "No new file for %ds — observation session appears complete.",
                    timeout
                )
                return True

    return False  # unreachable but explicit


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(
        description="Monitor a directory for new files and start a data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", metavar="INPUT_DIR",
        help="Directory where incoming raw images will appear."
    )
    parser.add_argument(
        "output", metavar="OUTPUT_DIR",
        help="Directory for reduced images and pipeline output files."
    )
    parser.add_argument(
        "--time", metavar="SECONDS", type=int, default=3600,
        help="Idle timeout in seconds. Pipeline starts after this many seconds "
             "with no new files."
    )
    parser.add_argument(
        "--poll", metavar="SECONDS", type=int, default=1,
        help="How often to poll the directory for new files."
    )
    parser.add_argument(
        "--log-interval", metavar="SECONDS", type=int, default=60,
        help="How often to print a heartbeat log while waiting."
    )
    parser.add_argument(
        "--loc", metavar="LOCATION", type=str, default="None",
        help="Telescope location (BSUO, CTIO, LaPalma, KPNO)."
    )
    parser.add_argument(
        "--ra", type=str, required=True,
        help="Right ascension of the target, e.g. 12:34:56.78"
    )
    parser.add_argument(
        "--dec", type=str, required=True,
        help="Declination of the target, e.g. -12:34:56.78"
    )
    parser.add_argument(
        "--name", metavar="OBJECT_NAME", type=str, default="NSVS_254037",
        help="Target name (use underscores instead of spaces)."
    )
    parser.add_argument(
        "--mem", metavar="BYTES", type=float, default=450e6,
        help="Memory limit for IRAF reduction in bytes (e.g. 450e6 = 450 MB)."
    )
    parser.add_argument(
        "--gain", metavar="GAIN", type=float, default=1.43,
        help="Camera gain (e/ADU)."
    )
    parser.add_argument(
        "--rdnoise", metavar="RDNOISE", type=float, default=10.83,
        help="Camera readout noise (e-)."
    )
    parser.add_argument(
        "--log-file", metavar="PATH", type=str, default=None,
        help="Optional path to write log output to a file."
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def monitor_and_run():
    args = _build_parser().parse_args()
    _setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    # --- Validate directories upfront ---
    for label, directory in [("Input", args.input), ("Output", args.output)]:
        if not path.isdir(directory):
            log.error("%s directory does not exist: %s", label, directory)
            sys.exit(1)

    # --- Graceful shutdown on Ctrl+C ---
    def _handle_interrupt(sig, frame):
        log.warning("Interrupted — pipeline will NOT start.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    # --- Monitor ---
    timed_out = monitor_directory(
        input_dir=args.input,
        timeout=args.time,
        poll_interval=args.poll,
        log_interval=args.log_interval,
    )

    if not timed_out:
        log.warning("Monitoring ended without timeout — pipeline not started.")
        sys.exit(0)

    # --- Pipeline ---
    log.info("Starting IRAF reduction...")
    IRAF(
        path=args.input,
        calibrated=args.output,
        pipeline=True,
        location=args.loc,
        gain=args.gain,
        rdnoise=args.rdnoise,
        mem_limit=args.mem,
    )

    log.info("Starting comparison star selection...")
    radec_files = comparison_selector(
        ra=args.ra,
        dec=args.dec,
        pipeline=True,
        folder_path=args.output,
        obj_name=args.name,
    )

    log.info("Starting aperture photometry...")
    multiple_AP(
        path=args.output,
        pipeline=True,
        radec_list=radec_files,
        obj_name=args.name,
    )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    monitor_and_run()
