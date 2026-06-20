"""Unit tests for synthetic/generator.py."""
from __future__ import annotations

import numpy as np
import pytest

from synthetic.generator import SyntheticGenerator
from synthetic.cameras import CAMERAS


class TestPreviewMode:
    def test_returns_ndarray(self, synth_preview):
        assert isinstance(synth_preview, np.ndarray)

    def test_shape_matches_camera_quarter(self, synth_params, synth_preview):
        cam = CAMERAS[synth_params["camera"]]
        expected_w = max(64, cam["width_px"]  // 4)
        expected_h = max(64, cam["height_px"] // 4)
        assert synth_preview.shape == (expected_h, expected_w)

    def test_dtype_float32(self, synth_preview):
        assert synth_preview.dtype == np.float32

    def test_values_nonnegative(self, synth_preview):
        assert float(synth_preview.min()) >= 0.0

    def test_values_within_adu_ceiling(self, synth_preview):
        # Stars plus sky should stay well below 3× the 16-bit ADU ceiling
        assert float(synth_preview.max()) < 65535.0 * 4.0

    def test_deterministic_same_seed(self, synth_params):
        gen = SyntheticGenerator()
        a = gen.generate(synth_params, preview=True)
        b = gen.generate(synth_params, preview=True)
        np.testing.assert_array_equal(a, b)

    def test_different_noise_seeds_differ(self, synth_params):
        gen = SyntheticGenerator()
        params_42 = {**synth_params, "seed": 42}
        params_99 = {**synth_params, "seed": 99}
        a = gen.generate(params_42, preview=True)
        b = gen.generate(params_99, preview=True)
        assert not np.array_equal(a, b)

    def test_has_nonzero_variance(self, synth_preview):
        assert float(synth_preview.std()) > 0.0

    def test_n_stars_affects_output(self, synth_params):
        gen = SyntheticGenerator()
        few  = gen.generate({**synth_params, "n_stars": 5},  preview=True)
        many = gen.generate({**synth_params, "n_stars": 40}, preview=True)
        # More stars means more total flux
        assert float(many.sum()) > float(few.sum())

    def test_nebula_enabled_increases_flux(self, synth_params):
        gen = SyntheticGenerator()
        no_neb = gen.generate({**synth_params, "nebula_enabled": False}, preview=True)
        neb    = gen.generate({**synth_params, "nebula_enabled": True,
                               "nebula_brightness": 2.0,
                               "nebula_size": 0.15},  preview=True)
        assert float(neb.sum()) > float(no_neb.sum())


@pytest.mark.slow
class TestFitsOutput:
    def test_returns_two_paths(self, synth_fits_path):
        main, starless = synth_fits_path
        assert main is not None
        assert starless is not None

    def test_main_fits_exists(self, synth_fits_path):
        from pathlib import Path
        assert Path(synth_fits_path[0]).exists()

    def test_starless_fits_exists(self, synth_fits_path):
        from pathlib import Path
        assert Path(synth_fits_path[1]).exists()

    def test_main_has_syn_keywords(self, synth_fits_path):
        from astropy.io import fits
        hdr = fits.getheader(synth_fits_path[0])
        for kw in ("SYN_FWHM", "SYN_NSED", "SYN_SSED"):
            assert kw in hdr, f"Missing keyword {kw}"

    def test_main_strl_absent_or_false(self, synth_fits_path):
        from astropy.io import fits
        hdr = fits.getheader(synth_fits_path[0])
        # SYN_STRL is written only on the starless companion; main file omits it
        assert hdr.get("SYN_STRL", False) == False

    def test_starless_strl_is_true(self, synth_fits_path):
        from astropy.io import fits
        hdr = fits.getheader(synth_fits_path[1])
        assert hdr["SYN_STRL"] == True

    def test_syn_seed_matches_params(self, synth_fits_path, synth_params):
        from astropy.io import fits
        hdr = fits.getheader(synth_fits_path[0])
        assert int(hdr["SYN_NSED"]) == int(synth_params["seed"])

    def test_starless_stem_suffix(self, synth_fits_path):
        from pathlib import Path
        _, starless = synth_fits_path
        assert "_starless" in Path(starless).stem
