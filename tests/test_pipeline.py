# -*- coding: utf-8 -*-
"""
Tests for pipeline.py and the EB_pipeline command

"""

import os
import signal
import sys

import pytest

from EclipsingBinaries import pipeline
from EclipsingBinaries.IRAF_Reduction import site_config


ARGS = ["--ra", "00:28:27.97", "--dec", "+78:57:42.66", "--name", "TEST_STAR", "--time", "0"]


@pytest.fixture
def dirs(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    (raw / "frame1.fits").write_bytes(b"")
    return raw, out


@pytest.fixture
def stages(monkeypatch):
    """Replace the three stages with recorders and skip the polling delay."""
    calls = []

    def reduction(**kwargs):
        calls.append(("reduction", kwargs))

    def selector(**kwargs):
        calls.append(("selector", kwargs))
        return ["b.radec", "v.radec", "r.radec"]

    def photometry(**kwargs):
        calls.append(("photometry", kwargs))

    monkeypatch.setattr(pipeline, "run_reduction", reduction)
    monkeypatch.setattr(pipeline, "comparison_selector", selector)
    monkeypatch.setattr(pipeline, "multiple_AP", photometry)
    monkeypatch.setattr(pipeline, "sleep", lambda _s: None)
    return calls


def _summary(out):
    return (out / "TEST_STAR_pipeline_summary.txt").read_text(encoding="utf-8")


# ===========================================================================
# Entry point wiring
# ===========================================================================
def test_console_script_points_at_cli():
    from importlib.metadata import entry_points
    (ep,) = [e for e in entry_points(group="console_scripts") if e.name == "EB_pipeline"]
    assert ep.value == "EclipsingBinaries.pipeline:monitor_directory_cli"
    assert ep.load() is pipeline.monitor_directory_cli


def test_name_is_required(dirs, capsys):
    raw, out = dirs
    with pytest.raises(SystemExit):
        pipeline.monitor_directory_cli([str(raw), str(out), "--ra", "1", "--dec", "2"])
    assert "--name" in capsys.readouterr().err


# ===========================================================================
# Full runs with the stages stubbed out
# ===========================================================================
def test_runs_all_stages_in_order(dirs, stages):
    raw, out = dirs
    assert pipeline.monitor_directory_cli([str(raw), str(out), *ARGS]) == 0

    assert [name for name, _ in stages] == ["reduction", "selector", "photometry"]
    reduction_kwargs = stages[0][1]
    assert reduction_kwargs["path"] == str(raw)
    assert reduction_kwargs["calibrated"] == str(out)
    assert reduction_kwargs["cfg"].location == "bsuo"
    assert stages[2][1]["radec_list"] == ["b.radec", "v.radec", "r.radec"]

    text = _summary(out)
    assert text.count("[ok]") == 3
    assert "No warnings." in text
    assert not (out / ".pipeline.lock").exists()


def test_loc_preset_and_overrides_reach_reduction(dirs, stages):
    raw, out = dirs
    pipeline.monitor_directory_cli([str(raw), str(out), *ARGS, "--loc", "KPNO", "--rdnoise", "4.5"])
    cfg = stages[0][1]["cfg"]
    assert cfg.location == "kpno"
    assert cfg.gain == 2.3      # KPNO preset
    assert cfg.rdnoise == 4.5   # explicit override


def test_creates_missing_output_folder(dirs, stages):
    raw, out = dirs
    assert not out.exists()
    pipeline.monitor_directory_cli([str(raw), str(out), *ARGS])
    assert out.is_dir()


def test_missing_input_folder_fails(tmp_path, stages):
    assert pipeline.monitor_directory_cli([str(tmp_path / "nope"), str(tmp_path / "out"), *ARGS]) == 1
    assert stages == []


def test_same_input_and_output_folder_fails(dirs, stages):
    raw, _ = dirs
    assert pipeline.monitor_directory_cli([str(raw), str(raw), *ARGS]) == 1
    assert stages == []


def test_stage_failure_stops_pipeline(dirs, stages, monkeypatch):
    raw, out = dirs

    def broken_selector(**kwargs):
        raise RuntimeError("Vizier is down")

    monkeypatch.setattr(pipeline, "comparison_selector", broken_selector)
    assert pipeline.monitor_directory_cli([str(raw), str(out), *ARGS]) == 1

    assert [name for name, _ in stages] == ["reduction"]
    text = _summary(out)
    assert "[FAILED]" in text and "Vizier is down" in text
    assert not (out / ".pipeline.lock").exists()


def test_no_radec_files_skips_photometry(dirs, stages, monkeypatch):
    raw, out = dirs
    monkeypatch.setattr(pipeline, "comparison_selector", lambda **kw: None)
    assert pipeline.monitor_directory_cli([str(raw), str(out), *ARGS]) == 1
    assert [name for name, _ in stages] == ["reduction"]
    assert "no RADEC files" in _summary(out)


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to the test process")
def test_ctrl_c_stops_cleanly_after_current_stage(dirs, stages, monkeypatch):
    raw, out = dirs

    def reduction(**kwargs):
        stages.append(("reduction", kwargs))
        os.kill(os.getpid(), signal.SIGINT)
        # Real stages check this between files
        assert kwargs["cancel_event"].is_set()

    monkeypatch.setattr(pipeline, "run_reduction", reduction)
    assert pipeline.monitor_directory_cli([str(raw), str(out), *ARGS]) == 0

    assert [name for name, _ in stages] == ["reduction"]
    text = _summary(out)
    assert "[canceled]" in text and "Stopped during IRAF Reduction" in text
    assert not (out / ".pipeline.lock").exists()
    # The handler is put back so later code sees normal Ctrl-C behaviour
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# ===========================================================================
# Process lock
# ===========================================================================
def test_lock_blocks_second_instance(tmp_path):
    first = pipeline.ProcessLock(str(tmp_path))
    second = pipeline.ProcessLock(str(tmp_path))
    assert first.acquire()
    try:
        assert not second.acquire()
    finally:
        first.release()
    assert second.acquire()
    second.release()
    assert not (tmp_path / ".pipeline.lock").exists()


def test_release_twice_is_harmless(tmp_path):
    lock = pipeline.ProcessLock(str(tmp_path))
    assert lock.acquire()
    lock.release()
    lock.release()


def test_leftover_lock_file_does_not_block(tmp_path):
    # A crash leaves the file behind but the OS has dropped the lock
    (tmp_path / ".pipeline.lock").write_text("stale\n")
    lock = pipeline.ProcessLock(str(tmp_path))
    assert lock.acquire()
    lock.release()


# ===========================================================================
# Directory monitor
# ===========================================================================
def test_monitor_returns_false_when_cancelled(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(pipeline, "sleep", lambda _s: None)
    event = threading.Event()
    event.set()
    assert pipeline.monitor_directory(str(tmp_path), timeout=10, cancel_event=event) is False


# ===========================================================================
# site_config
# ===========================================================================
@pytest.mark.parametrize("name, gain", [("BSUO", 1.43), ("kpno", 2.3), ("CTIO", 2.0),
                                        ("LaPalma", 1.0), ("La Palma", 1.0)])
def test_site_config_presets(name, gain):
    assert site_config(name).gain == gain


def test_site_config_unknown_site_keeps_name_and_defaults():
    cfg = site_config("SFRO")
    assert cfg.location == "SFRO"
    assert cfg.gain == 1.43


def test_site_config_ignores_none_overrides():
    cfg = site_config("kpno", gain=None, rdnoise=3.0)
    assert cfg.gain == 2.3 and cfg.rdnoise == 3.0


def test_site_config_validates():
    with pytest.raises(ValueError):
        site_config("bsuo", gain=-1)
