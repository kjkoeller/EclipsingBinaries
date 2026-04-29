"""End-of-run summary tracking and report writing for the data reduction pipeline.

Every run of :func:`EclipsingBinaries.IRAF_Reduction.run_reduction`
records into a :class:`RunSummary` instance. On completion (success,
failure, or cancellation) the summary is flushed to two files alongside
the existing ``reduction_config.json`` snapshot:

* ``reduction_summary.json`` -- the full structured record, suitable for
  downstream tooling and diff-based regression checks
* ``reduction_summary.txt`` -- a concise human-readable digest with
  stage-by-stage counts, durations, and any failures

The summary is written even when the run is cancelled or raises so the
artifact directory is always self-describing.

Author: Kyle Koeller
Created: 04/27/2026
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Callable
import json


# Possible terminal statuses for a stage. "pending" is the initial value
# and indicates the stage never ran.
STAGE_STATUSES = ("pending", "ok", "skipped", "failed", "cancelled")

# Stage names emitted by run_reduction. Listed here so the txt report
# preserves a stable ordering even if dict iteration order changes.
STAGE_ORDER = ("bias", "dark", "flat", "science")

# Cap on stored failure-reason strings. A pathological exception message
# (e.g. a 50-line traceback fragment) shouldn't bloat the JSON report or
# wreck the txt formatting.
FAILURE_REASON_MAX = 200


@dataclass
class StageSummary:

    """Per-stage tally of frames processed and how the stage finished."""

    name: str
    status: str = "pending"
    detail: str = ""
    duration_sec: Optional[float] = None
    n_total: int = 0
    n_succeeded: int = 0
    n_failed: int = 0


@dataclass
class FrameFailure:

    """A single frame that failed during a stage, with the reason."""

    stage: str
    file: str
    reason: str


@dataclass
class RunSummary:

    """Aggregate record of a reduction run.

    All fields are populated incrementally as the pipeline executes;
    :func:`finalize` is called once at the end to add the terminal
    status, finished timestamp, and total duration before writing to disk.
    """

    # --- run identification ---
    raw_path: str
    calibrated_path: str
    location: str
    started_utc: str
    finished_utc: Optional[str] = None
    overall_status: str = "pending"       # ok / failed / cancelled / pending
    error_message: Optional[str] = None
    duration_sec: Optional[float] = None

    # --- shared state captured during the run ---
    reference_shape: Optional[List[int]] = None
    longest_dark_exposure_sec: Optional[float] = None
    longest_science_exposure_sec: Optional[float] = None
    masters_reused: List[str] = field(default_factory=list)

    # --- stages and failures ---
    stages: Dict[str, StageSummary] = field(default_factory=dict)
    failed_frames: List[FrameFailure] = field(default_factory=list)

    # --------- mutation API used by run_reduction ----------------------

    def stage(self, name: str) -> StageSummary:
        """
        Return the StageSummary for ``name``, creating it on first access.
        Lets callers do ``summary.stage("bias").n_total = 30`` without
        any boilerplate setup.
        """
        if name not in self.stages:
            self.stages[name] = StageSummary(name=name)
        return self.stages[name]

    def record_master_reused(self, key: str) -> None:
        """Note that a master frame was reused rather than regenerated."""
        if key not in self.masters_reused:
            self.masters_reused.append(key)

    def record_failure(self, stage: str, file: str, reason: str) -> None:
        """
        Append a single frame failure with the reason.

        The reason is truncated to :data:`FAILURE_REASON_MAX` characters
        so a runaway exception message can't bloat the report.
        """
        text = str(reason)
        if len(text) > FAILURE_REASON_MAX:
            text = text[: FAILURE_REASON_MAX - 3] + "..."
        self.failed_frames.append(FrameFailure(stage=stage, file=file, reason=text))


# ----------------------------------------------------------------------
# Construction / finalization
# ----------------------------------------------------------------------

def new_summary(raw_path, calibrated_path, location: str) -> RunSummary:
    """Create a fresh RunSummary stamped with the current UTC time."""
    return RunSummary(
        raw_path=str(raw_path),
        calibrated_path=str(calibrated_path),
        location=str(location),
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def finalize(summary: RunSummary, status: str,
             error_message: Optional[str] = None) -> None:
    """
    Stamp the run as complete and write both the JSON and TXT reports.

    Idempotent: a second call after the first does not change the
    finished_utc timestamp or recompute the duration. This matters
    because :func:`run_reduction` may call finalize from both a success
    branch and a finally-block on the same path.

    Writing is best-effort: if the artifact directory is gone or full,
    the function logs to stderr but does not raise (the run has already
    finished, and we don't want to mask the real error with a
    logging-of-the-summary error).

    :param summary: The RunSummary populated during the run
    :param status: Terminal status (one of "ok", "failed", "cancelled")
    :param error_message: Optional error string when status != "ok"
    """
    if summary.finished_utc is not None:
        # Already finalized; do nothing
        return

    summary.overall_status = status
    summary.error_message = error_message
    summary.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Compute total duration from started/finished timestamps
    try:
        start = datetime.fromisoformat(summary.started_utc)
        end = datetime.fromisoformat(summary.finished_utc)
        summary.duration_sec = (end - start).total_seconds()
    except Exception:
        summary.duration_sec = None

    # Mark any stage still in "pending" as cancelled if the whole run was
    # cancelled, or skipped if the run completed (so the report reflects
    # what actually ran)
    fallback = "cancelled" if status == "cancelled" else "skipped"
    for st in summary.stages.values():
        if st.status == "pending":
            st.status = fallback

    # Best-effort write to both JSON and TXT
    try:
        write_summary(summary, Path(summary.calibrated_path))
    except Exception as e:
        # Don't propagate -- the run itself is over.
        import sys
        print(
            f"Warning: failed to write reduction summary to "
            f"{summary.calibrated_path}: {e}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def write_summary(summary: RunSummary, output_dir: Path,
                  log: Optional[Callable[[str], None]] = None) -> List[Path]:
    """
    Write reduction_summary.json and reduction_summary.txt.

    :param summary: The completed RunSummary
    :param output_dir: Destination directory (created if absent)
    :param log: Optional callable for diagnostic output
    :return: List of paths actually written
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "reduction_summary.json"
    txt_path = output_dir / "reduction_summary.txt"

    # JSON: structured, easy to diff, friendly to downstream tooling
    json_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=False, default=str),
        encoding="utf-8",
    )

    # TXT: human-readable digest
    txt_path.write_text(_format_txt(summary), encoding="utf-8")

    if log is not None:
        log(f"Run summary written: {json_path.name}, {txt_path.name}")

    return [json_path, txt_path]


def _format_header(s: RunSummary) -> List[str]:
    """Title bar + run identification block."""
    lines = [
        "=" * 70,
        "EclipsingBinaries Data Reduction Summary",
        "=" * 70,
        f"Status:      {s.overall_status.upper()}",
    ]
    if s.error_message:
        lines.append(f"Error:       {s.error_message}")
    lines.append(f"Started:     {s.started_utc}")
    if s.finished_utc:
        lines.append(f"Finished:    {s.finished_utc}")
    if s.duration_sec is not None:
        lines.append(f"Duration:    {_fmt_duration(s.duration_sec)}")
    return lines


def _format_run_metadata(s: RunSummary) -> List[str]:
    """Path / location / shape / reused-masters block."""
    lines = [
        "",
        f"Raw input:   {s.raw_path}",
        f"Output:      {s.calibrated_path}",
        f"Location:    {s.location}",
    ]
    if s.reference_shape:
        lines.append(f"Frame shape: {tuple(s.reference_shape)}")
    if s.masters_reused:
        lines.append(f"Reused:      {', '.join(s.masters_reused)}")
    return lines


def _format_exposure_check(s: RunSummary) -> List[str]:
    """Optional dark/science exposure-ratio sanity check."""
    if s.longest_dark_exposure_sec is None and s.longest_science_exposure_sec is None:
        return []
    lines = ["", "Exposure-time check:"]
    if s.longest_dark_exposure_sec is not None:
        lines.append(f"  Longest dark:    {s.longest_dark_exposure_sec:.1f} s")
    if s.longest_science_exposure_sec is not None:
        lines.append(f"  Longest science: {s.longest_science_exposure_sec:.1f} s")
    if (s.longest_dark_exposure_sec is not None
            and s.longest_science_exposure_sec is not None
            and s.longest_dark_exposure_sec > 0):
        ratio = s.longest_science_exposure_sec / s.longest_dark_exposure_sec
        lines.append(f"  Science/dark ratio: {ratio:.1f}x")
    return lines


def _format_stage_table(s: RunSummary) -> List[str]:
    """Per-stage tabular breakdown ordered by STAGE_ORDER."""
    lines = [
        "",
        "-" * 70,
        f"{'Stage':<10} {'Status':<10} {'Total':>6} {'OK':>6} {'Fail':>6} "
        f"{'Time':>10}  Detail",
        "-" * 70,
    ]
    # Known stages in canonical order, then any custom stages we don't know about
    seen = set()
    for name in STAGE_ORDER:
        if name in s.stages:
            lines.append(_format_stage_line(s.stages[name]))
            seen.add(name)
    for name, stage in s.stages.items():
        if name not in seen:
            lines.append(_format_stage_line(stage))
    return lines


def _format_failures(s: RunSummary) -> List[str]:
    """Per-frame failure list, or 'no failures' line."""
    if not s.failed_frames:
        return ["", "No per-frame failures."]
    lines = ["", "-" * 70, f"Failed frames ({len(s.failed_frames)}):", "-" * 70]
    for f in s.failed_frames:
        lines.append(f"  [{f.stage}] {f.file}")
        for line in str(f.reason).splitlines() or [""]:
            lines.append(f"      {line}")
    return lines


def _format_txt(s: RunSummary) -> str:
    """Render a RunSummary as a plain-text report."""
    sections = (
        _format_header(s)
        + _format_run_metadata(s)
        + _format_exposure_check(s)
        + _format_stage_table(s)
        + _format_failures(s)
        + [""]   # trailing blank line before final newline
    )
    return "\n".join(sections) + "\n"


def _format_stage_line(st: StageSummary) -> str:
    """One row of the per-stage table."""
    duration = _fmt_duration(st.duration_sec) if st.duration_sec is not None else "--"
    return (
        f"{st.name:<10} {st.status:<10} "
        f"{st.n_total:>6} {st.n_succeeded:>6} {st.n_failed:>6} "
        f"{duration:>10}  {st.detail}"
    ).rstrip()


def _fmt_duration(seconds: Optional[float]) -> str:
    """Format a duration as e.g. '4.2s', '1m 23s', or '1h 02m 15s'."""
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = seconds - 60 * m
        return f"{m}m {s:04.1f}s"
    h = int(seconds // 3600)
    rem = seconds - 3600 * h
    m = int(rem // 60)
    s = rem - 60 * m
    return f"{h}h {m:02d}m {s:04.1f}s"