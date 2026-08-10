"""Unit tests for synthetic/generator.py."""
from __future__ import annotations

import numpy as np
import pytest

from astropy.stats import SigmaClip
from photutils.background import (Background2D, MADStdBackgroundRMS,
                                  SExtractorBackground)

from analysis.background_fit import fit_background_surface
from synthetic.generator import SyntheticGenerator, _sky_gradient_map
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


class TestSkyGradient:
    """The additive light-pollution / moon-glow ramp added for Section 3g testing."""

    def test_zero_gradient_is_exactly_uniform(self):
        """The default must reproduce the pre-gradient uniform sky bit-for-bit."""
        m = _sky_gradient_map(500.0, 100, 200, 0.0, 0.0)
        assert m.shape == (100, 200)
        assert np.ptp(m) == 0.0
        assert m[0, 0] == pytest.approx(500.0)

    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 215.0])
    def test_swing_and_mean_hold_at_every_angle(self, angle):
        """A 'fraction of sky' swing must mean the same thing in any direction.

        Without the span renormalisation a diagonal ramp would span sqrt(2)
        times more than an axis-aligned one for the same setting.
        """
        sky = 500.0
        m = _sky_gradient_map(sky, 400, 600, 0.30, angle)
        assert float(np.ptp(m)) / sky == pytest.approx(0.30, rel=1e-6)
        assert float(m.mean()) == pytest.approx(sky, rel=1e-6)

    def test_gradient_is_never_negative(self):
        """The Poisson expectation must stay positive at the full slider range."""
        m = _sky_gradient_map(10.0, 64, 64, 0.5, 137.0)
        assert float(m.min()) > 0.0

    def test_direction_points_where_the_sky_brightens(self):
        m = _sky_gradient_map(500.0, 200, 200, 0.3, 0.0)     # 0 deg = +x
        assert m[:, -1].mean() > m[:, 0].mean()
        m90 = _sky_gradient_map(500.0, 200, 200, 0.3, 90.0)  # 90 deg = +y
        assert m90[-1, :].mean() > m90[0, :].mean()

    def test_injected_gradient_is_recovered_from_a_generated_frame(self):
        """Round-trip: the ramp must be measurable by the Section 3f fit."""
        rng = np.random.default_rng(0)
        h = w = 512
        sky_e = _sky_gradient_map(1000.0, h, w, 0.30, 20.0)
        data = rng.poisson(sky_e).astype(np.float32)
        bkg = Background2D(
            data, box_size=64, filter_size=3,
            sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
            bkg_estimator=SExtractorBackground(),
            bkg_rms_estimator=MADStdBackgroundRMS())
        fit = fit_background_surface(bkg.background_mesh, bkg.background_rms_mesh,
                                     64, (h, w))
        assert fit["gradient"] is not None
        assert fit["gradient"]["adu_range"] == pytest.approx(0.30 * 1000.0, rel=0.15)
        assert fit["gradient"]["direction_deg"] == pytest.approx(20.0, abs=5.0)

    def test_shot_noise_scales_with_local_sky(self):
        """Applied to the Poisson expectation, not added afterwards.

        A gradient added post-hoc would leave a flat noise level under a
        sloping sky, which no real light-pollution gradient does.
        """
        rng = np.random.default_rng(1)
        sky_e = _sky_gradient_map(1000.0, 512, 512, 0.5, 0.0)
        data = rng.poisson(sky_e).astype(np.float64)
        dark = float(data[:, :64].std())
        bright = float(data[:, -64:].std())
        assert bright > dark
