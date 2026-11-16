# -*- coding: utf-8 -*-
"""Settings shared by every test."""

import pytest


@pytest.fixture(autouse=True)
def star_files_in_a_temporary_folder(tmp_path_factory, monkeypatch):
    """Keep the star files the tools write (star_file.py) out of the real home folder."""
    monkeypatch.setenv("ECLIPSINGBINARIES_STARS", str(tmp_path_factory.mktemp("stars")))
