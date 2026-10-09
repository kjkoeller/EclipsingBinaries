"""
Watches a directory for incoming images during an observing session, then runs
the data pipeline once no new file has arrived for a set time:
IRAF-style reduction, comparison star selection, and aperture photometry.

Run it with the EB_pipeline command, e.g.

    EB_pipeline raw/ reduced/ --ra 00:28:27.97 --dec +78:57:42.66 --name NSVS_896797 --loc BSUO

Author: Kyle Koeller
Created: 06/15/2023
Last Edited: 10/09/2026
"""

import argparse
import logging
import signal
import sys
import threading
from datetime import timedelta
from os import path, listdir
from pathlib import Path
from time import time, sleep

from .apass import comparison_selector
from .IRAF_Reduction import run_reduction, site_config
from .multi_aperture_photometry import main as multiple_AP

# File locking differs by platform. fcntl doesn't exist on Windows, and
# importing it unconditionally kept this module from loading there at all.
if sys.platform == "win32":
    import msvcrt

    def _try_lock(lock_file):
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(lock_file):
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(lock_file):
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(lock_file):
        fcntl.flock(lock_file, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(log_file=None):
    """
    Send pipeline and analysis messages to the console, and to log_file if
    given. Uses the package logger, which the analysis modules also write to
    when they have no GUI to report to.
    """
    logger = logging.getLogger("EclipsingBinaries")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Process lock
# ---------------------------------------------------------------------------
class ProcessLock:
    """
    Prevents two instances of the pipeline from running on the same directory.
    The OS releases the lock automatically if the process dies, so a lock file
    left behind by a crash doesn't block the next run.
    """

    def __init__(self, lock_dir):
        self.lock_path = path.join(lock_dir, ".pipeline.lock")
        self._lock_file = None
        self.log = logging.getLogger(__name__)

    def acquire(self):
        # Append mode so a second instance doesn't wipe the running one's
        # details before finding out the file is locked
        lock_file = open(self.lock_path, "a+")
        try:
            _try_lock(lock_file)
        except OSError:
            lock_file.close()
            self.log.error("Another pipeline instance is already running on this directory (%s).",
                           self.lock_path)
            return False

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(" ".join(sys.argv) + "\n")
        lock_file.flush()
        self._lock_file = lock_file
        self.log.info("Process lock acquired: %s", self.lock_path)
        return True

    def release(self):
        if self._lock_file is None:
            return
        try:
            _unlock(self._lock_file)
        except OSError:
            pass
        self._lock_file.close()
        self._lock_file = None
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

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        self.log.info("Summary report written to %s", report_path)
        return report


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
def monitor_directory(input_dir, timeout, poll_interval=1, log_interval=60, cancel_event=None):
    """
    Wait until no new file has shown up in input_dir for timeout seconds.

    :return: True once the idle timeout is reached, False if cancel_event was set first
    """
    log = logging.getLogger(__name__)
    current_latest = get_latest_file(input_dir)
    start_time = time()
    last_log_time = time()

    log.info("Monitoring %s for new files (timeout: %ds)...", input_dir, timeout)

    while True:
        if cancel_event is not None and cancel_event.is_set():
            return False

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(
        prog="EB_pipeline",
        description="Monitor a directory for new files and start a data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", metavar="INPUT_DIR",
        help="Directory where incoming raw images will appear."
    )
    parser.add_argument(
        "output", metavar="OUTPUT_DIR",
        help="Directory for reduced images and pipeline output files. Created if missing."
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
        "--name", metavar="OBJECT_NAME", type=str, required=True,
        help="Target name used in output file names (use underscores instead of spaces)."
    )
    parser.add_argument(
        "--time", metavar="SECONDS", type=int, default=3600,
        help="Idle timeout in seconds before the pipeline starts."
    )
    parser.add_argument(
        "--loc", metavar="LOCATION", type=str, default="BSUO",
        help="Telescope location. BSUO, KPNO, CTIO and LaPalma have gain and read noise presets."
    )
    parser.add_argument(
        "--gain", metavar="GAIN", type=float, default=None,
        help="Camera gain (e/ADU). Defaults to the --loc preset."
    )
    parser.add_argument(
        "--rdnoise", metavar="RDNOISE", type=float, default=None,
        help="Camera readout noise (e-). Defaults to the --loc preset."
    )
    parser.add_argument(
        "--mem", metavar="BYTES", type=float, default=450e6,
        help="Memory limit for IRAF reduction in bytes."
    )
    parser.add_argument(
        "--log-file", metavar="PATH", type=str, default=None,
        help="Optional path to write log output to a file."
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
class _StageFailed(Exception):
    pass


def _run_stage(name, func, summary, log, cancel_event, /, **kwargs):
    """
    Run one pipeline stage, recording its time and outcome in the summary.
    The leading arguments are positional-only so a stage can still be handed
    its own cancel_event keyword.
    """
    log.info("Starting %s...", name)
    start = time()
    try:
        result = func(**kwargs)
    except Exception as exc:
        summary.record_stage(name, time() - start, "FAILED")
        summary.add_warning(f"{name} failed: {exc}")
        log.error("%s failed: %s", name, exc, exc_info=True)
        raise _StageFailed(name) from exc
    summary.record_stage(name, time() - start, "canceled" if cancel_event.is_set() else "ok")
    return result


def monitor_directory_cli(argv=None):
    """
    Entry point for the EB_pipeline command.

    :param argv: Argument list, for calling from Python. Defaults to sys.argv.
    :return: Process exit code: 0 on success or a clean stop, 1 on failure
    """
    args = _build_parser().parse_args(argv)
    log = _setup_logging(args.log_file).getChild("pipeline")

    if not path.isdir(args.input):
        log.error("Input directory does not exist: %s", args.input)
        return 1
    if Path(args.input).resolve() == Path(args.output).resolve():
        # Reduced files landing in the watched folder would keep resetting
        # the idle timer and get mixed in with the raw frames
        log.error("INPUT_DIR and OUTPUT_DIR must be different folders.")
        return 1
    Path(args.output).mkdir(parents=True, exist_ok=True)

    try:
        cfg = site_config(args.loc, gain=args.gain, rdnoise=args.rdnoise, mem_limit=args.mem)
    except ValueError as exc:
        log.error("Invalid reduction settings: %s", exc)
        return 1

    lock = ProcessLock(args.output)
    if not lock.acquire():
        return 1

    summary = PipelineSummary(args.output, args.name)
    cancel_event = threading.Event()

    # The first Ctrl-C asks every stage to stop at its next checkpoint so the
    # summary and lock get cleaned up. A second one quits immediately.
    def _handle_interrupt(sig, frame):
        if cancel_event.is_set():
            log.warning("Second interrupt, quitting now.")
            raise KeyboardInterrupt
        log.warning("Interrupt received. Stopping after the current step (Ctrl-C again to quit now).")
        cancel_event.set()

    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, _handle_interrupt)

    stage_log = logging.getLogger("EclipsingBinaries").info
    exit_code = 0
    try:
        if not monitor_directory(input_dir=args.input, timeout=args.time, cancel_event=cancel_event):
            summary.add_warning("Stopped before the idle timeout, so the pipeline was not started.")
            return 0

        log.info("Reduction settings: location=%s gain=%s rdnoise=%s mem_limit=%.0f",
                 cfg.location, cfg.gain, cfg.rdnoise, cfg.mem_limit)
        _run_stage("IRAF Reduction", run_reduction, summary, log, cancel_event,
                   path=args.input, calibrated=args.output, cfg=cfg,
                   cancel_event=cancel_event, write_callback=stage_log)
        if cancel_event.is_set():
            summary.add_warning("Stopped during IRAF Reduction.")
            return 0

        radec_files = _run_stage("Comparison Star Selection", comparison_selector, summary, log, cancel_event,
                                 ra=args.ra, dec=args.dec, pipeline=True, folder_path=args.output,
                                 obj_name=args.name, write_callback=stage_log, cancel_event=cancel_event)
        if cancel_event.is_set():
            summary.add_warning("Stopped during Comparison Star Selection.")
            return 0
        if not radec_files:
            summary.add_warning("Comparison Star Selection produced no RADEC files; skipping photometry.")
            return 1

        _run_stage("Aperture Photometry", multiple_AP, summary, log, cancel_event,
                   path=args.output, pipeline=True, radec_list=radec_files, obj_name=args.name,
                   write_callback=stage_log, cancel_event=cancel_event)
        if cancel_event.is_set():
            summary.add_warning("Stopped during Aperture Photometry.")
            return 0

        log.info("Pipeline complete.")
    except _StageFailed:
        exit_code = 1
    except KeyboardInterrupt:
        summary.add_warning("Pipeline interrupted by user before completion.")
        exit_code = 130
    finally:
        summary.write()
        lock.release()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    return exit_code


if __name__ == "__main__":
    sys.exit(monitor_directory_cli())
