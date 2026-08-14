"""Unit tests for core/astro_image.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from core.astro_image import AstroImage, decimation_step
from core.models import DEFAULT_PIXEL_SCALE


class TestLoad:
    def test_data_float32(self, astro_image_a):
        assert astro_image_a.data.dtype == np.float32

    def test_correct_shape(self, astro_image_a):
        assert astro_image_a.data.shape == (512, 512)

    def test_label_explicit(self, astro_image_a):
        assert astro_image_a.label == "TestA"

    def test_label_defaults_to_stem(self, test_fits_path):
        img = AstroImage(str(test_fits_path))
        img.load()
        assert img.label == test_fits_path.stem

    def test_header_not_none(self, astro_image_a):
        assert astro_image_a.header is not None

    def test_data_nonnegative(self, astro_image_a):
        # Our synthetic sky is positive; clip at 0 in fixture
        assert float(astro_image_a.data.min()) >= 0.0

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "image.xyz"
        p.write_bytes(b"dummy")
        img = AstroImage(str(p))
        with pytest.raises(ValueError, match="Unsupported"):
            img.load()


class TestPixelScale:
    def test_scale_from_focallen_xpixsz(self, astro_image_a):
        # FOCALLEN=500mm, XPIXSZ=3.76um -> 206.265*3.76/500 ~ 1.55 arcsec/px
        assert 1.0 < astro_image_a.pixel_scale < 3.0

    def test_cdelt1_extraction(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 100
        hdr = ap_fits.Header()
        hdr["CDELT1"] = -0.0004277778   # deg/px -> ~1.54 arcsec/px
        ap_fits.writeto(str(tmp_path / "cdelt.fits"), data, hdr, overwrite=True)
        img = AstroImage(str(tmp_path / "cdelt.fits"))
        img.load()
        assert img.pixel_scale == pytest.approx(abs(-0.0004277778) * 3600, rel=0.01)

    def test_pixscale_keyword(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 100
        hdr = ap_fits.Header()
        hdr["PIXSCALE"] = 2.5
        ap_fits.writeto(str(tmp_path / "ps.fits"), data, hdr, overwrite=True)
        img = AstroImage(str(tmp_path / "ps.fits"))
        img.load()
        assert img.pixel_scale == pytest.approx(2.5)

    def test_default_scale_without_wcs(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 100
        ap_fits.writeto(str(tmp_path / "nowcs.fits"), data, overwrite=True)
        img = AstroImage(str(tmp_path / "nowcs.fits"))
        img.load()
        assert img.pixel_scale == DEFAULT_PIXEL_SCALE
        assert img.pixel_scale_is_estimated

    def test_focal_ratio_computed(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 100
        hdr = ap_fits.Header()
        hdr["FOCALLEN"] = 800.0
        hdr["APTDIA"]   = 100.0   # -> f/8
        ap_fits.writeto(str(tmp_path / "fratio.fits"), data, hdr, overwrite=True)
        img = AstroImage(str(tmp_path / "fratio.fits"))
        img.load()
        assert "Focal ratio" in img.meta
        assert "f/8" in img.meta["Focal ratio"]


class TestBandwidth:
    def test_none_without_keyword(self, astro_image_a):
        assert astro_image_a.bandwidth_nm is None

    def test_bandwid_extraction(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 100
        hdr = ap_fits.Header()
        hdr["BANDWID"] = 3.0
        ap_fits.writeto(str(tmp_path / "bw.fits"), data, hdr, overwrite=True)
        img = AstroImage(str(tmp_path / "bw.fits"))
        img.load()
        assert img.bandwidth_nm == pytest.approx(3.0)


class TestBackground:
    def test_background_set_after_estimate(self, astro_image_a):
        assert astro_image_a.background is not None

    def test_background_rms_shape(self, astro_image_a):
        assert astro_image_a.background_rms.shape == astro_image_a.data.shape

    def test_background_subtracted_shape(self, astro_image_a):
        bs = astro_image_a.background_subtracted()
        assert bs.shape == astro_image_a.data.shape

    def test_background_subtracted_dtype(self, astro_image_a):
        bs = astro_image_a.background_subtracted()
        assert bs.dtype == np.float32

    def test_background_subtracted_without_background(self, test_fits_path):
        img = AstroImage(str(test_fits_path), label="NoBkg")
        img.load()
        # No estimate_background() call: should return data as-is
        bs = img.background_subtracted()
        np.testing.assert_array_equal(bs, img.data)

    def test_background_display_model_shape_dtype(self, astro_image_a):
        disp = astro_image_a.background_display(kind="model")
        # 512x512 fixture is well under the default max_dim=800, so step=1
        # and the array should be untouched apart from the dtype cast.
        assert disp.shape == astro_image_a.background.background.shape
        assert disp.dtype == np.float32

    def test_background_display_rms_matches_attribute(self, astro_image_a):
        disp = astro_image_a.background_display(kind="rms")
        np.testing.assert_array_equal(disp, astro_image_a.background_rms.astype(np.float32))

    def test_background_display_downsamples_large_image(self, astro_image_a):
        max_dim = 100
        disp = astro_image_a.background_display(kind="model", max_dim=max_dim)
        h, w = astro_image_a.background_model.shape
        step = decimation_step(h, w, max_dim)
        expected = astro_image_a.background_model[::step, ::step].astype(np.float32)
        assert disp.shape == expected.shape
        assert disp.shape[0] < astro_image_a.background_model.shape[0]
        np.testing.assert_array_equal(disp, expected)

    def test_background_display_unmasked_model_is_the_phase_one_estimate(self, astro_image_a):
        """kind="unmasked_model" must stay pinned to Background2D's own surface.

        Section 3g's before/after comparison is built on it, so if it ever started
        tracking the adopted model the difference map would silently go to zero
        rather than error.
        """
        disp = astro_image_a.background_display(kind="unmasked_model")
        np.testing.assert_array_equal(
            disp, astro_image_a.background.background.astype(np.float32))

    def test_background_display_unknown_kind_raises(self, astro_image_a):
        with pytest.raises(ValueError):
            astro_image_a.background_display(kind="bogus")

    def test_background_display_without_background_raises(self, test_fits_path):
        img = AstroImage(str(test_fits_path), label="NoBkg")
        img.load()
        with pytest.raises(RuntimeError):
            img.background_display()


class TestSourceMaskedAdoption:
    """estimate_background()'s phase 2 — adopting the source-masked estimate."""

    def test_adopted_model_is_not_the_unmasked_one(self, astro_image_a):
        assert astro_image_a.source_mask_result is not None
        assert astro_image_a.source_mask_result.ok
        assert not np.array_equal(astro_image_a.background_model,
                                  astro_image_a.background.background)

    def test_background_subtracted_uses_the_adopted_model(self, astro_image_a):
        np.testing.assert_allclose(
            astro_image_a.background_subtracted(),
            (astro_image_a.data - astro_image_a.background_model).astype(np.float32),
            rtol=0, atol=0)

    def test_rms_is_the_masked_map_not_the_unmasked_one(self, astro_image_a):
        """Pairing a masked background with an unmasked RMS would mix two
        different estimates of the same sky, so the RMS must move too."""
        res = astro_image_a.source_mask_result
        assert res.rms_map is not None
        np.testing.assert_array_equal(astro_image_a.background_rms, res.rms_map)

    def test_flag_off_keeps_the_plain_estimate(self, test_fits_path):
        img = AstroImage(str(test_fits_path), label="Plain")
        img.load()
        img.use_source_masked_background = False
        img.estimate_background()
        assert img.source_mask_result is None
        np.testing.assert_array_equal(img.background_model,
                                      img.background.background)
        np.testing.assert_array_equal(img.background_rms,
                                      img.background.background_rms)

    def test_setting_the_exclusion_mask_invalidates_a_computed_background(
            self, test_fits_path):
        """The guard keys on background_model, so a mask attached afterwards
        would otherwise be silently ignored — a wrong answer, not an error."""
        img = AstroImage(str(test_fits_path), label="Excl")
        img.load()
        img.use_source_masked_background = False
        img.estimate_background()
        assert img.background_model is not None

        mask = np.zeros(img.data.shape[:2], dtype=bool)
        mask[:64, :64] = True
        img.set_background_exclusion_mask(mask)
        assert img.background is None
        assert img.background_model is None
        assert img.background_rms is None
        assert img.source_mask_result is None

    def test_drawn_region_reaches_the_source_mask(self, test_fits_path):
        img = AstroImage(str(test_fits_path), label="Excl")
        img.load()
        mask = np.zeros(img.data.shape[:2], dtype=bool)
        mask[100:200, 100:200] = True
        img.set_background_exclusion_mask(mask)
        img.estimate_background()
        res = img.source_mask_result
        assert res is not None and res.source_mask is not None
        assert res.source_mask.user_mask is not None
        # Every drawn pixel is excluded, whatever the detector concluded.
        assert res.source_mask.mask[mask].all()

    def test_second_call_is_a_no_op(self, test_fits_path):
        img = AstroImage(str(test_fits_path), label="Idem")
        img.load()
        img.estimate_background()
        first_model = img.background_model
        first_res = img.source_mask_result
        img.estimate_background()
        assert img.background_model is first_model
        assert img.source_mask_result is first_res


class TestDecimationStep:
    def test_no_downsample_needed(self):
        assert decimation_step(512, 512, max_dim=800) == 1

    def test_downsample_ratio(self):
        assert decimation_step(2000, 1000, max_dim=500) == 4

    def test_never_zero(self):
        assert decimation_step(10, 10, max_dim=800) == 1


class TestSaturation:
    def test_saturation_threshold_positive(self, astro_image_a):
        assert astro_image_a.saturation_threshold() > 0

    def test_datamax_keyword(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 1000
        hdr = ap_fits.Header()
        hdr["DATAMAX"] = 60000.0
        ap_fits.writeto(str(tmp_path / "sat.fits"), data, hdr, overwrite=True)
        img = AstroImage(str(tmp_path / "sat.fits"))
        img.load()
        assert img.saturation_threshold() == pytest.approx(60000.0 * 0.90)

    def test_fallback_to_max_data(self, astro_image_a):
        # No DATAMAX in our test FITS -> uses max(data)*0.9
        thresh = astro_image_a.saturation_threshold()
        expected = float(np.max(astro_image_a.data)) * 0.90
        assert thresh == pytest.approx(expected, rel=0.01)


class TestDisplay:
    def test_display_image_uint8(self, astro_image_a):
        out = astro_image_a.display_image()
        assert out.dtype == np.uint8

    def test_display_image_range(self, astro_image_a):
        out = astro_image_a.display_image()
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_display_image_shape(self, astro_image_a):
        out = astro_image_a.display_image()
        assert out.shape == astro_image_a.data.shape


class TestMetadata:
    def test_instrume_mapped_to_camera(self, astro_image_a):
        assert "Camera" in astro_image_a.meta
        assert astro_image_a.meta["Camera"] == "TestCam"

    def test_bit_depth_in_meta(self, astro_image_a):
        assert "Bit depth" in astro_image_a.meta

    def test_repr_contains_label(self, astro_image_a):
        r = repr(astro_image_a)
        assert "TestA" in r
