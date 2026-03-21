"""
This script checks for new files in a directory every second for the start of a data pipeline.

Author: Kyle Koeller
Created: 06/15/2023
Last Edited: 03/20/2026
"""

import logging
import signal
import sys
import fcntl
from os import path, listdir
from pathlib import Path
from time import time, sleep
from datetime import timedelta
import argparse

from .apass import comparison_selector
from .IRAF_Reduction import main as IRAF
from .multi_aperture_photometry import main as multiple_AP


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(log_file=None):
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers)


# ---------------------------------------------------------------------------
# Process lock
# ---------------------------------------------------------------------------
class ProcessLock:
    """
    Prevents two instances of the pipeline from running on the same directory.
    Uses a .lock file with an exclusive flock so the lock is automatically
    released if the process dies unexpectedly.
    """

    def __init__(self, lock_dir):
        self.lock_path = path.join(lock_dir, ".pipeline.lock")
        self._lock_file = None
        self.log = logging.getLogger(__name__)

    def acquire(self):
        try:
            self._lock_file = open(self.lock_path, "w")
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_file.write(str(sys.argv) + "\n")
            self._lock_file.flush()
            self.log.info("Process lock acquired: %s", self.lock_path)
            return True
        except BlockingIOError:
            self.log.error(
                "Another pipeline instance is already running on this directory. "
                "If this is wrong, delete %s and retry.", self.lock_path
            )
            return False

    def release(self):
        if self._lock_file:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            try:
                Path(self.lock_path).unlink()
            except FileNotFoundError:
                pass
            self.log.info("Process lock released.")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
class PipelineSummary:
    """
    Tracks pipeline stage timings and warnings, writes a summary report
    to disk at the end of the run.
    """

    def __init__(self, output_dir, obj_name):
        self.output_dir = output_dir
        self.obj_name = obj_name
        self.start_time = time()
        self.stages = []
        self.warnings = []
        self.log = logging.getLogger(__name__)

    def record_stage(self, name, duration, status="ok"):
        self.stages.append((name, duration, status))

    def add_warning(self, message):
        self.warnings.append(message)
        self.log.warning(message)

    def write(self):
        total = time() - self.start_time
        report_path = path.join(
            self.output_dir, f"{self.obj_name}_pipeline_summary.txt"
        )

        lines = [
            "=" * 60,
            f"Pipeline Summary — {self.obj_name}",
            "=" * 60,
            f"Total runtime : {str(timedelta(seconds=int(total)))}",
            "",
            "Stages:",
        ]
        for name, duration, status in self.stages:
            lines.append(
                f"  {name:<30} {str(timedelta(seconds=int(duration))):<12} [{status}]"
            )

        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("")
            lines.append("No warnings.")

        lines.append("=" * 60)
        report = "\n".join(lines) + "\n"

        with open(report_path, "w") as f:
            f.write(report)

        self.log.info("Summary report written to %s", report_path)
        return report


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def get_latest_file(folder_path):
    files = [
        path.join(folder_path, f)
        for f in listdir(folder_path)
        if path.isfile(path.join(folder_path, f))
    ]
    return max(files, key=path.getmtime) if files else None


def count_files(folder_path):
    return sum(
        1 for f in listdir(folder_path)
        if path.isfile(path.join(folder_path, f))
    )


# ---------------------------------------------------------------------------
# Directory monitor
# ---------------------------------------------------------------------------
def monitor_directory(input_dir, timeout, poll_interval=1, log_interval=60):
    log = logging.getLogger(__name__)
    current_latest = get_latest_file(input_dir)
    start_time = time()
    last_log_time = time()

    log.info("Monitoring %s for new files (timeout: %ds)...", input_dir, timeout)

    while True:
        sleep(poll_interval)
        latest = get_latest_file(input_dir)

        if latest != current_latest:
            log.info("New file detected: %s", latest)
            current_latest = latest
            start_time = time()
            last_log_time = time()
        else:
            elapsed = time() - start_time
            if time() - last_log_time >= log_interval:
                log.info(
                    "Still waiting... %.0fs elapsed, %.0fs until timeout (%d files)",
                    elapsed, timeout - elapsed, count_files(input_dir)
                )
                last_log_time = time()

            if elapsed >= timeout:
                log.info("No new file for %ds — session complete.", timeout)
                return True

    return False


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
        help="Idle timeout in seconds before the pipeline starts."
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
        help="Memory limit for IRAF reduction in bytes."
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
def monitor_directory_cli():
    args = _build_parser().parse_args()
    _setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    # Validate directories
    for label, directory in [("Input", args.input), ("Output", args.output)]:
        if not path.isdir(directory):
            log.error("%s directory does not exist: %s", label, directory)
            sys.exit(1)

    # Process lock — prevent duplicate pipeline runs on the same directory
    lock = ProcessLock(args.output)
    if not lock.acquire():
        sys.exit(1)

    # Summary report tracker
    summary = PipelineSummary(args.output, args.name)

    # Graceful shutdown
    def _handle_interrupt(sig, frame):
        log.warning("Interrupted — pipeline will NOT start.")
        summary.add_warning("Pipeline interrupted by user before completion.")
        summary.write()
        lock.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    try:
        # Monitor
        timed_out = monitor_directory(
            input_dir=args.input,
            timeout=args.time,
        )

        if not timed_out:
            log.warning("Monitoring ended without timeout — pipeline not started.")
            summary.add_warning("Monitoring ended without timeout.")
            summary.write()
            lock.release()
            sys.exit(0)

        # IRAF reduction
        log.info("Starting IRAF reduction...")
        stage_start = time()
        try:
            IRAF(
                path=args.input,
                calibrated=args.output,
                pipeline=True,
                location=args.loc,
                gain=args.gain,
                rdnoise=args.rdnoise,
                mem_limit=args.mem,
            )
            summary.record_stage("IRAF Reduction", time() - stage_start, "ok")
        except Exception as e:
            summary.record_stage("IRAF Reduction", time() - stage_start, "FAILED")
            summary.add_warning(f"IRAF Reduction failed: {e}")
            log.error("IRAF Reduction failed: %s", e)
            summary.write()
            lock.release()
            sys.exit(1)

        # Comparison star selection
        log.info("Starting comparison star selection...")
        stage_start = time()
        try:
            radec_files = comparison_selector(
                ra=args.ra,
                dec=args.dec,
                pipeline=True,
                folder_path=args.output,
                obj_name=args.name,
            )
            summary.record_stage("Comparison Star Selection", time() - stage_start, "ok")
        except Exception as e:
            summary.record_stage("Comparison Star Selection", time() - stage_start, "FAILED")
            summary.add_warning(f"Comparison star selection failed: {e}")
            log.error("Comparison star selection failed: %s", e)
            summary.write()
            lock.release()
            sys.exit(1)

        # Aperture photometry
        log.info("Starting aperture photometry...")
        stage_start = time()
        try:
            multiple_AP(
                path=args.output,
                pipeline=True,
                radec_list=radec_files,
                obj_name=args.name,
            )
            summary.record_stage("Aperture Photometry", time() - stage_start, "ok")
        except Exception as e:
            summary.record_stage("Aperture Photometry", time() - stage_start, "FAILED")
            summary.add_warning(f"Aperture photometry failed: {e}")
            log.error("Aperture photometry failed: %s", e)
            summary.write()
            lock.release()
            sys.exit(1)

        # All done
        log.info("Pipeline complete.")
        summary.write()

    finally:
        lock.release()


if __name__ == "__main__":
    monitor_directory_cli()
