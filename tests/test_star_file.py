# -*- coding: utf-8 -*-
"""Tests for star_file.py, the file per star that the tools share."""

import json

import pytest

from EclipsingBinaries import star_file as sf


def test_names_that_differ_only_in_spacing_and_case_are_the_same_star():
    assert sf.star_key("NSVS 896797") == sf.star_key("  nsvs_896797 ") == "nsvs_896797"
    assert sf.star_key("V* SW Lac") == "v_sw_lac"
    with pytest.raises(ValueError):
        sf.star_key("  ")


def test_a_star_file_gathers_values_and_files_from_each_step(tmp_path):
    sf.record("NSVS 896797", "archives", files=[tmp_path / "archives"], ra="00:28:27.97", dec="+78:57:42.7")
    sf.record("nsvs_896797", "period", files=[tmp_path / "p.txt"], period=0.3700)
    star = sf.StarFile.load("NSVS 896797")
    assert star.values == {"ra": "00:28:27.97", "dec": "+78:57:42.7", "period": 0.37}
    assert star.files["period"] == [str(tmp_path / "p.txt")]
    assert [entry["step"] for entry in star.history] == ["archives", "period"]
    assert sf.known("NSVS 896797")["period"] == 0.37
    assert sf.all_stars() == ["NSVS 896797"]
    assert json.loads(star.path.read_text())["name"] == "NSVS 896797"


def test_saving_the_same_thing_again_adds_no_history(tmp_path):
    sf.record("SW Lac", "minima", files=[tmp_path / "a_min.txt"], time_system="HJD_UTC")
    sf.record("SW Lac", "minima", files=[tmp_path / "a_min.txt"], time_system="HJD_UTC")
    sf.record("SW Lac", "minima", files=[tmp_path / "b_min.txt"])
    star = sf.StarFile.load("SW Lac")
    assert len(star.history) == 2
    # Newest last, no repeats; None values don't clear what's there
    assert star.files["minima"] == [str(tmp_path / "a_min.txt"), str(tmp_path / "b_min.txt")]
    assert star.get("time_system") == "HJD_UTC"


def test_problems_are_logged_not_raised(monkeypatch, tmp_path):
    blocker = tmp_path / "a file"
    blocker.write_text("")
    monkeypatch.setenv("ECLIPSINGBINARIES_STARS", str(blocker / "stars"))
    messages = []
    assert sf.record("SW Lac", "period", write_callback=messages.append, period=0.32) is None
    assert "couldn't be updated" in messages[0]
    assert sf.record("", "period", period=0.32) is None
    assert sf.known("") == {}


def test_unknown_values_are_refused():
    with pytest.raises(ValueError, match="don't hold colour"):
        sf.StarFile.load("SW Lac").update("test", colour="blue")


def test_terminal_summary(capsys):
    assert sf.main([]) == 0
    assert "No star files yet" in capsys.readouterr().out
    sf.record("SW Lac", "period", period=0.3207)
    sf.main(["SW", "Lac"])
    out = capsys.readouterr().out
    assert "period: 0.3207" in out and "SW Lac" in out
