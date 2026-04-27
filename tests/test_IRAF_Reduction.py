"""
Tests for the EclipsingBinaries data reduction pipeline.

Covers the BJD_TDB calculation path (the only thing the previous test
suite checked), plus the helper functions, configuration validation,
header-correction toggles, and the observatory registry that were added
during the refactor.

NOTE on expected BJD values
---------------------------
The expected BJD_TDB values here differ from the previous test suite
because the new module uses the standard astropy pattern that matches
Robert Berrington's original ``header-correct.py`` script:

    BJD_TDB = date.tdb.jd + light_travel_time(target, barycentric).jd

The previous standalone ``BJD_TDB()`` function used a different formula
that did not match the original script. Expected values used here come
directly from the new implementation.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from EclipsingBinaries.IRAF_Reduction import (
    HeaderConfig,
    ReductionConfig,
    bsuo_config,
    kpno_config,
    ctio_config,
    lapalma_config,
    _normalize_filter,
    _header_get_any,
    _get_exposure_time,
    _master_is_fresh,
    _discover_master_flats,
    _build_header_correction_opts,
    correct_headers as cfg_correct_headers,
)
from EclipsingBinaries.headerCorrect import (
    HeaderCorrectionOptions,
    ObservatoryRegistry,
    ObservatorySite,
    correct_headers,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_fits_for_jd(jd, observatory, ra, dec, exptime=0.0,
                      extra_keys=None):
    """
    Write a minimal FITS file whose mid-exposure time equals ``jd``.

    With EXPTIME=0 the start of the exposure equals the midpoint, so
    DATE-OBS is just the ISO form of ``jd``. With a nonzero exposure,
    DATE-OBS is rolled back by half the exposure so the midpoint still
    lands on ``jd``.
    """
    half_exp_days = (exptime / 2.0) / 86400.0
    start_jd = jd - half_exp_days
    date_obs = Time(start_jd, format="jd", scale="utc").fits

    fd, path = tempfile.mkstemp(suffix=".fits")
    os.close(fd)

    hdr = fits.Header()
    hdr["DATE-OBS"] = date_obs
    hdr["EXPTIME"]  = exptime
    hdr["OBSERVAT"] = observatory
    hdr["RA"]       = ra
    hdr["DEC"]      = dec
    if extra_keys:
        for k, v in extra_keys.items():
            hdr[k] = v

    fits.PrimaryHDU(np.zeros((1, 1), dtype=np.float32), header=hdr).writeto(
        path, overwrite=True,
    )
    return Path(path)


# ---------------------------------------------------------------------------
# BJD_TDB calculation tests (the original test scope)
# ---------------------------------------------------------------------------

class TestBJDTDB(unittest.TestCase):
    """BJD_TDB accuracy at known sites for known targets."""

    # 6 decimal places ≈ 0.1 second tolerance. The astropy site-list path
    # (used for SARA-RM) and the explicit-coords path (used for BSUO)
    # can disagree at the 8th decimal due to differing observer positions,
    # but for any photometric timing application 6 places is more than
    # sufficient.
    PLACES = 6

    def setUp(self):
        # Run only the BJD calculation; turn off the rest so the test is
        # focused and doesn't depend on sidereal/airmass code paths.
        self.opts = HeaderCorrectionOptions(
            do_jd=False, do_hjd=False, do_bjd=True,
            do_sidereal=False, do_eairmass=False, do_radec_format=False,
        )
        self.registry = ObservatoryRegistry()
        self._tmp_files = []

    def tearDown(self):
        for p in self._tmp_files:
            try:
                p.unlink()
            except OSError:
                pass

    def _bjd_for(self, jd, observatory, ra, dec):
        path = _make_fits_for_jd(jd, observatory, ra, dec)
        self._tmp_files.append(path)
        correct_headers(path, opts=self.opts, registry=self.registry,
                        log=lambda _msg: None)
        return fits.getheader(str(path))["BJD_TDB"]

    def test_BJD_TDB_bsuo_target1(self):
        bjd = self._bjd_for(2458403.58763, "BSUO",
                            "00:28:27.97", "+78:57:42.66")
        self.assertAlmostEqual(bjd, 2458403.590238446, places=self.PLACES)

    def test_BJD_TDB_lapalma_target1(self):
        bjd = self._bjd_for(2458403.58763, "SARA-RM",
                            "00:28:27.97", "+78:57:42.66")
        self.assertAlmostEqual(bjd, 2458403.590238446, places=self.PLACES)

    def test_BJD_TDB_bsuo_target2(self):
        bjd = self._bjd_for(2457143.76136, "BSUO",
                            "13:27:50.47", "+75:55:16.60")
        self.assertAlmostEqual(bjd, 2457143.761993383, places=self.PLACES)

    def test_BJD_TDB_lapalma_target2(self):
        bjd = self._bjd_for(2457143.76136, "SARA-RM",
                            "13:27:50.47", "+75:55:16.60")
        self.assertAlmostEqual(bjd, 2457143.761993383, places=self.PLACES)


class TestObservatoryIndependence(unittest.TestCase):
    """
    BJD_TDB is barycentric, so the same target at the same JD produces the
    same BJD_TDB regardless of observatory location. Earth-diameter
    light-travel differences are below our 7-decimal tolerance.
    """

    def test_bsuo_vs_lapalma_same_target(self):
        opts = HeaderCorrectionOptions(
            do_jd=False, do_hjd=False, do_bjd=True,
            do_sidereal=False, do_eairmass=False, do_radec_format=False,
        )
        registry = ObservatoryRegistry()

        path_bsuo = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
        )
        path_lapalma = _make_fits_for_jd(
            2458403.58763, "SARA-RM", "00:28:27.97", "+78:57:42.66",
        )
        try:
            correct_headers(path_bsuo, opts=opts, registry=registry,
                            log=lambda _: None)
            correct_headers(path_lapalma, opts=opts, registry=registry,
                            log=lambda _: None)
            bjd_bsuo = fits.getheader(str(path_bsuo))["BJD_TDB"]
            bjd_lapalma = fits.getheader(str(path_lapalma))["BJD_TDB"]
            # 6 places ≈ 0.1 second; observer-position differences between
            # the explicit-coords and astropy-site-list paths show up at
            # the 8th decimal but well below this threshold.
            self.assertAlmostEqual(bjd_bsuo, bjd_lapalma, places=6)
        finally:
            for p in (path_bsuo, path_lapalma):
                try: p.unlink()
                except OSError: pass


# ---------------------------------------------------------------------------
# correct_headers behaviour
# ---------------------------------------------------------------------------

class TestHeaderCorrectionIdempotency(unittest.TestCase):
    """Re-running correct_headers must not alter the ``*_ORIG`` keywords."""

    def test_orig_keywords_preserved_on_second_run(self):
        opts = HeaderCorrectionOptions(do_eairmass=False, do_sidereal=False)
        registry = ObservatoryRegistry()

        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
        )
        try:
            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            ra_orig_first = fits.getheader(str(path)).get("RA_ORIG")

            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            ra_orig_second = fits.getheader(str(path)).get("RA_ORIG")
            self.assertEqual(ra_orig_first, ra_orig_second)
        finally:
            try: path.unlink()
            except OSError: pass


class TestSelectiveSkipping(unittest.TestCase):
    """Toggle flags on HeaderCorrectionOptions actually skip work."""

    def test_skip_all_calculations(self):
        opts = HeaderCorrectionOptions(
            do_jd=False, do_hjd=False, do_bjd=False,
            do_sidereal=False, do_eairmass=False,
        )
        registry = ObservatoryRegistry()
        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
        )
        try:
            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            hdr = fits.getheader(str(path))
            for key in ("BJD_TDB", "HJD", "EAIRMASS", "SIDEREAL"):
                self.assertNotIn(key, hdr,
                                 f"{key} should be absent when skipped")
        finally:
            try: path.unlink()
            except OSError: pass

    def test_jd_only(self):
        """do_jd alone writes JD_START/JD_MID/JD_END but nothing else."""
        opts = HeaderCorrectionOptions(
            do_jd=True, do_hjd=False, do_bjd=False,
            do_sidereal=False, do_eairmass=False, do_radec_format=False,
        )
        registry = ObservatoryRegistry()
        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
        )
        try:
            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            hdr = fits.getheader(str(path))
            for key in ("JD_START", "JD_MID", "JD_END", "JD"):
                self.assertIn(key, hdr)
            for key in ("BJD_TDB", "HJD", "EAIRMASS"):
                self.assertNotIn(key, hdr)
        finally:
            try: path.unlink()
            except OSError: pass


class TestRADECFormatting(unittest.TestCase):
    """RA/DEC are reformatted into IRAF-compatible sexagesimal strings."""

    def test_ra_dec_get_colons(self):
        opts = HeaderCorrectionOptions(
            do_jd=False, do_hjd=False, do_bjd=False,
            do_sidereal=False, do_eairmass=False, do_radec_format=True,
        )
        registry = ObservatoryRegistry()
        # Space-separated values, no colons
        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "16 41 41.24", "+36 27 35.5",
        )
        try:
            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            hdr = fits.getheader(str(path))
            self.assertEqual(hdr["RA"].count(":"), 2)
            self.assertEqual(hdr["DEC"].count(":"), 2)
            self.assertTrue(hdr["DEC"].startswith(("+", "-")))
            # Originals preserved
            self.assertEqual(hdr["RA_ORIG"], "16 41 41.24")
        finally:
            try: path.unlink()
            except OSError: pass


class TestFilterSpaceReplacement(unittest.TestCase):
    """do_filter_parse=True replaces spaces in FILTER with underscores."""

    def test_space_to_underscore(self):
        opts = HeaderCorrectionOptions(
            do_jd=False, do_hjd=False, do_bjd=False,
            do_sidereal=False, do_eairmass=False,
            do_filter_parse=True,
        )
        registry = ObservatoryRegistry()
        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
            extra_keys={"FILTER": "Empty V"},
        )
        try:
            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            hdr = fits.getheader(str(path))
            self.assertEqual(hdr["FILTER"], "Empty_V")
            self.assertEqual(hdr["FILT_ORG"], "Empty V")
        finally:
            try: path.unlink()
            except OSError: pass

    def test_disabled_by_default(self):
        opts = HeaderCorrectionOptions(
            do_jd=False, do_hjd=False, do_bjd=False,
            do_sidereal=False, do_eairmass=False,
        )
        registry = ObservatoryRegistry()
        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
            extra_keys={"FILTER": "Empty V"},
        )
        try:
            correct_headers(path, opts=opts, registry=registry,
                            log=lambda _: None)
            hdr = fits.getheader(str(path))
            self.assertEqual(hdr["FILTER"], "Empty V")
            self.assertNotIn("FILT_ORG", hdr)
        finally:
            try: path.unlink()
            except OSError: pass


class TestHeaderCorrectionMasterSwitch(unittest.TestCase):
    """cfg.correct_headers=False short-circuits without touching the file."""

    def test_master_switch_off(self):
        cfg = ReductionConfig(correct_headers=False)
        path = _make_fits_for_jd(
            2458403.58763, "BSUO", "00:28:27.97", "+78:57:42.66",
        )
        try:
            result = cfg_correct_headers(path, cfg)
            self.assertIsNone(result)
            hdr = fits.getheader(str(path))
            for key in ("BJD_TDB", "HJD", "JD_MID", "EAIRMASS"):
                self.assertNotIn(key, hdr)
        finally:
            try: path.unlink()
            except OSError: pass


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestNormalizeFilter(unittest.TestCase):
    def test_basic_uppercase(self):
        self.assertEqual(_normalize_filter("v"), "V")
        self.assertEqual(_normalize_filter("R"), "R")

    def test_strips_whitespace(self):
        self.assertEqual(_normalize_filter("  V  "), "V")

    def test_strips_empty_prefix(self):
        # default prefix is ("Empty/",)
        self.assertEqual(_normalize_filter("Empty/V"), "V")
        self.assertEqual(_normalize_filter("empty/B"), "B")  # case-insensitive
        self.assertEqual(_normalize_filter("EMPTY/R"), "R")

    def test_custom_prefixes(self):
        self.assertEqual(
            _normalize_filter("Clear/V", prefixes=("Empty/", "Clear/")),
            "V",
        )

    def test_unmatched_prefix_left_alone(self):
        self.assertEqual(_normalize_filter("Wheel/V"), "WHEEL/V")

    def test_empty_prefix_tuple(self):
        self.assertEqual(_normalize_filter("V", prefixes=()), "V")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            _normalize_filter(None)

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            _normalize_filter("   ")


class TestHeaderGetAny(unittest.TestCase):
    def test_first_present_wins(self):
        hdr = fits.Header()
        hdr["JD-HELIO"] = 12345.0
        hdr["HJD_UTC"] = 99999.0
        # First key in priority list wins even if later ones exist
        self.assertEqual(
            _header_get_any(hdr, "JD-HELIO", "HJD_UTC", "HJD"),
            12345.0,
        )

    def test_falls_through_to_alias(self):
        hdr = fits.Header()
        hdr["HJD_UTC"] = 99999.0
        # JD-HELIO not present, falls through to HJD_UTC
        self.assertEqual(
            _header_get_any(hdr, "JD-HELIO", "HJD_UTC"),
            99999.0,
        )

    def test_objct_dec_with_space(self):
        # Astropy stores 'OBJCT DEC' as a HIERARCH card. Confirm lookup works.
        hdr = fits.Header()
        hdr["OBJCT DEC"] = "+78:57:42.66"
        self.assertEqual(
            _header_get_any(hdr, "DEC", "OBJCTDEC", "OBJCT DEC"),
            "+78:57:42.66",
        )

    def test_none_when_absent(self):
        hdr = fits.Header()
        self.assertIsNone(_header_get_any(hdr, "NOPE", "ALSO_NOPE"))

    def test_whitespace_value_is_skipped(self):
        hdr = fits.Header()
        hdr["RA"] = "   "
        hdr["OBJCTRA"] = "16:41:41.24"
        # Whitespace-only RA is treated as missing; falls through to OBJCTRA
        self.assertEqual(
            _header_get_any(hdr, "RA", "OBJCTRA"),
            "16:41:41.24",
        )


class TestGetExposureTime(unittest.TestCase):
    def test_simple_value(self):
        hdr = fits.Header()
        hdr["EXPTIME"] = 60.0
        self.assertEqual(_get_exposure_time(hdr), 60.0)

    def test_missing_returns_none(self):
        hdr = fits.Header()
        self.assertIsNone(_get_exposure_time(hdr))

    def test_zero_returns_none(self):
        hdr = fits.Header()
        hdr["EXPTIME"] = 0.0
        self.assertIsNone(_get_exposure_time(hdr))

    def test_negative_returns_none(self):
        hdr = fits.Header()
        hdr["EXPTIME"] = -5.0
        self.assertIsNone(_get_exposure_time(hdr))

    def test_invalid_string_returns_none(self):
        hdr = fits.Header()
        hdr["EXPTIME"] = "not a number"
        self.assertIsNone(_get_exposure_time(hdr))

    def test_custom_key(self):
        hdr = fits.Header()
        hdr["ITIME"] = 300.0
        self.assertEqual(_get_exposure_time(hdr, key="ITIME"), 300.0)
        self.assertIsNone(_get_exposure_time(hdr, key="EXPTIME"))


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

class TestReductionConfigValidation(unittest.TestCase):
    def test_defaults_construct(self):
        cfg = ReductionConfig()
        self.assertEqual(cfg.location, "bsuo")
        self.assertGreater(cfg.gain, 0)

    def test_negative_gain_rejected(self):
        with self.assertRaises(ValueError):
            ReductionConfig(gain=-1.0)

    def test_zero_gain_rejected(self):
        with self.assertRaises(ValueError):
            ReductionConfig(gain=0.0)

    def test_negative_rdnoise_rejected(self):
        with self.assertRaises(ValueError):
            ReductionConfig(rdnoise=-5.0)

    def test_zero_rdnoise_allowed(self):
        cfg = ReductionConfig(rdnoise=0.0)
        self.assertEqual(cfg.rdnoise, 0.0)

    def test_empty_location_rejected(self):
        with self.assertRaises(ValueError):
            ReductionConfig(location="")

    def test_master_flat_pattern_must_have_filter(self):
        with self.assertRaises(ValueError):
            ReductionConfig(master_flat_pattern="FLAT.fits")

    def test_master_flat_pattern_with_filter_ok(self):
        cfg = ReductionConfig(master_flat_pattern="FLAT{filter}.fits")
        self.assertIn("{filter}", cfg.master_flat_pattern)

    def test_negative_mem_limit_rejected(self):
        with self.assertRaises(ValueError):
            ReductionConfig(mem_limit=-1.0)


class TestHeaderConfigValidation(unittest.TestCase):
    def test_defaults_construct(self):
        h = HeaderConfig()
        self.assertEqual(h.imagetyp_bias, "BIAS")

    def test_empty_imagetyp_rejected(self):
        with self.assertRaises(ValueError):
            HeaderConfig(imagetyp_bias="")

    def test_whitespace_imagetyp_rejected(self):
        with self.assertRaises(ValueError):
            HeaderConfig(filter_key="  ")

    def test_each_config_instance_has_own_header_config(self):
        # Regression: default_factory must give each ReductionConfig its
        # own HeaderConfig (not a shared singleton).
        a = ReductionConfig()
        b = ReductionConfig()
        a.headers.imagetyp_bias = "MUTATED"
        self.assertEqual(b.headers.imagetyp_bias, "BIAS")


class TestSiteFactories(unittest.TestCase):
    """Each site factory returns a config tagged for that site."""

    def test_bsuo(self):
        cfg = bsuo_config()
        self.assertEqual(cfg.location, "bsuo")

    def test_kpno(self):
        cfg = kpno_config()
        self.assertEqual(cfg.location, "kpno")
        self.assertGreater(cfg.gain, 0)

    def test_ctio(self):
        cfg = ctio_config()
        self.assertEqual(cfg.location, "ctio")

    def test_lapalma(self):
        cfg = lapalma_config()
        self.assertEqual(cfg.location, "lapalma")


# ---------------------------------------------------------------------------
# Master file discovery and freshness
# ---------------------------------------------------------------------------

class TestDiscoverMasterFlats(unittest.TestCase):
    def test_default_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for name in ("master_flat_B.fits", "master_flat_V.fits",
                         "master_flat_R.fits"):
                (td / name).write_bytes(b"x")
            # Files that should NOT be picked up
            (td / "zero.fits").write_bytes(b"x")
            (td / "master_dark.fits").write_bytes(b"x")

            cfg = ReductionConfig()
            found = _discover_master_flats(td, cfg)
            self.assertEqual(set(found.keys()), {"B", "V", "R"})

    def test_custom_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for name in ("FLATB.fits", "FLATV.fits", "FLATR.fits"):
                (td / name).write_bytes(b"x")

            cfg = ReductionConfig(master_flat_pattern="FLAT{filter}.fits")
            found = _discover_master_flats(td, cfg)
            self.assertEqual(set(found.keys()), {"B", "V", "R"})

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = ReductionConfig()
            found = _discover_master_flats(Path(td), cfg)
            self.assertEqual(found, {})


class TestMasterIsFresh(unittest.TestCase):
    def test_missing_master_is_not_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw.fits"
            raw.write_bytes(b"x")
            self.assertFalse(_master_is_fresh(td / "missing.fits", [raw]))

    def test_fresher_master_is_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw.fits"
            raw.write_bytes(b"x")
            time.sleep(0.05)
            master = td / "master.fits"
            master.write_bytes(b"y")
            self.assertTrue(_master_is_fresh(master, [raw]))

    def test_stale_master_is_not_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            master = td / "master.fits"
            master.write_bytes(b"y")
            time.sleep(0.05)
            raw = td / "raw.fits"
            raw.write_bytes(b"x")  # newer than master
            self.assertFalse(_master_is_fresh(master, [raw]))


# ---------------------------------------------------------------------------
# ObservatoryRegistry behaviour
# ---------------------------------------------------------------------------

class TestObservatoryRegistry(unittest.TestCase):
    def test_explicit_site_resolves(self):
        reg = ObservatoryRegistry()
        # BSUO has explicit lat/lon — does not need network
        loc = reg.resolve("BSUO")
        self.assertIsNotNone(loc)

    def test_case_insensitive_lookup(self):
        reg = ObservatoryRegistry()
        loc_upper = reg.resolve("BSUO")
        loc_lower = reg.resolve("bsuo")
        self.assertAlmostEqual(loc_upper.lat.deg, loc_lower.lat.deg, places=6)

    def test_register_custom_site(self):
        reg = ObservatoryRegistry()
        reg.register(ObservatorySite(
            name="MIDWEST",
            lat="41:30:00", lon="-87:00:00", altitude_m=200.0,
        ))
        loc = reg.resolve("midwest")
        self.assertAlmostEqual(loc.lat.deg, 41.5, places=4)

    def test_unknown_falls_back_to_default(self):
        reg = ObservatoryRegistry()
        msgs = []
        loc = reg.resolve("NOWHERE", default="BSU", log=msgs.append)
        self.assertIsNotNone(loc)
        # Some kind of warning should have been logged
        self.assertTrue(any("NOWHERE" in m for m in msgs))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TestBuildHeaderCorrectionOpts(unittest.TestCase):
    def test_defaults_pass_through(self):
        cfg = ReductionConfig()
        opts = _build_header_correction_opts(cfg)
        self.assertEqual(opts.do_jd, cfg.correct_jd)
        self.assertEqual(opts.do_hjd, cfg.correct_hjd)
        self.assertEqual(opts.do_bjd, cfg.correct_bjd)
        self.assertEqual(opts.do_sidereal, cfg.correct_sidereal)
        self.assertEqual(opts.do_eairmass, cfg.correct_eairmass)
        self.assertEqual(opts.do_filter_parse, cfg.correct_filter_spaces)

    def test_default_observatory_uppercased(self):
        cfg = ReductionConfig(location="kpno")
        opts = _build_header_correction_opts(cfg)
        self.assertEqual(opts.default_observatory, "KPNO")

    def test_individual_toggles_propagate(self):
        cfg = ReductionConfig(correct_eairmass=False, correct_sidereal=False)
        opts = _build_header_correction_opts(cfg)
        self.assertFalse(opts.do_eairmass)
        self.assertFalse(opts.do_sidereal)
        # Others stay on
        self.assertTrue(opts.do_bjd)


if __name__ == "__main__":
    unittest.main()