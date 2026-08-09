"""Tests for the Section 3e (Background Model) / 3f (Background Gradient
Analysis) report wiring in report/report_builder.py.

Covers the new figure builders, the _section_snr HTML block (gated
independently of 3e/3f per CLAUDE.md's "gate on the right condition, not a
sibling's" pitfall), and the _write_inspector_file npz additions.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from core.astro_image import AstroImage
from core.models import AnalysisResult
from report.report_builder import ReportBuilder


def _make_image_pair(tmp_path_factory, size=256) -> tuple[AstroImage, AstroImage]:
    out = tmp_path_factory.mktemp("bgsection_fits")
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(12)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    for path, rng in ((out / "a.fits", rng_a), (out / "b.fits", rng_b)):
        data = rng.normal(1000.0, 20.0, (h, w)).astype(np.float32)
        # Deliberate gradient so the fit has real structure to recover.
        data += (0.05 * xx).astype(np.float32)
        blob = 500.0 * np.exp(-0.5 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / 20.0 ** 2))
        data += blob.astype(np.float32)
        hdr = ap_fits.Header()
        hdr["EGAIN"] = 1.0
        ap_fits.writeto(str(path), data, hdr, overwrite=True)
    img_a = AstroImage(str(out / "a.fits"), label="A")
    img_a.load()
    img_a.estimate_background()
    img_b = AstroImage(str(out / "b.fits"), label="B")
    img_b.load()
    img_b.estimate_background()
    return img_a, img_b


@pytest.fixture(scope="module")
def image_pair(tmp_path_factory):
    return _make_image_pair(tmp_path_factory)


@pytest.fixture(scope="module")
def no_background_image(tmp_path_factory):
    out = tmp_path_factory.mktemp("bgsection_nobg")
    data = np.random.default_rng(13).normal(500.0, 10.0, (256, 256)).astype(np.float32)
    path = out / "nobg.fits"
    ap_fits.writeto(str(path), data, overwrite=True)
    img = AstroImage(str(path), label="NoBg")
    img.load()
    return img


class TestPlotBackgroundMeshPair:
    def test_both_images_two_panels(self, image_pair):
        img_a, img_b = image_pair
        fig = ReportBuilder()._plot_background_mesh_pair(img_a, "A", img_b, "B")
        assert fig is not None
        assert len(fig.axes) == 2

    def test_single_image_one_panel(self, image_pair):
        img_a, _ = image_pair
        fig = ReportBuilder()._plot_background_mesh_pair(img_a, "A", None, "B")
        assert fig is not None
        assert len(fig.axes) == 1

    def test_no_background_returns_none(self, no_background_image):
        fig = ReportBuilder()._plot_background_mesh_pair(no_background_image, "A", None, "B")
        assert fig is None

    def test_both_none_returns_none(self):
        assert ReportBuilder()._plot_background_mesh_pair(None, "A", None, "B") is None


class TestPlotBackgroundMapPair:
    def _disp(self):
        return np.random.default_rng(0).normal(100.0, 5.0, (32, 32)).astype(np.float32)

    def test_two_panels_include_colorbars(self):
        fig = ReportBuilder()._plot_background_map_pair(
            self._disp(), "A", self._disp(), "B", 90.0, 110.0, "ADU", "Background model")
        assert fig is not None
        # One image axes + one colorbar axes per panel.
        assert len(fig.axes) == 4

    def test_single_panel(self):
        fig = ReportBuilder()._plot_background_map_pair(
            self._disp(), "A", None, "B", 90.0, 110.0, "ADU", "Background model")
        assert fig is not None
        assert len(fig.axes) == 2

    def test_both_none_returns_none(self):
        fig = ReportBuilder()._plot_background_map_pair(
            None, "A", None, "B", 0.0, 1.0, "ADU", "Background model")
        assert fig is None

    def test_cmap_override_used_for_residual(self):
        fig = ReportBuilder()._plot_background_map_pair(
            self._disp(), "A", None, "B", -5.0, 5.0, "ADU", "Fit residual", cmap="bwr")
        assert fig.axes[0].images[0].get_cmap().name == "bwr"


class TestPlotBgPixelHistogram:
    def _bgsub_and_mask(self, seed=0):
        rng = np.random.default_rng(seed)
        bgsub = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        mask = np.abs(bgsub) < 3.0
        return bgsub, mask

    def test_two_panels(self):
        bs_a, m_a = self._bgsub_and_mask(1)
        bs_b, m_b = self._bgsub_and_mask(2)
        fig = ReportBuilder()._plot_bg_pixel_histogram(bs_a, m_a, "A", bs_b, m_b, "B")
        assert fig is not None
        assert len(fig.axes) == 2

    def test_single_panel(self):
        bs_a, m_a = self._bgsub_and_mask(1)
        fig = ReportBuilder()._plot_bg_pixel_histogram(bs_a, m_a, "A", None, None, "B")
        assert fig is not None
        assert len(fig.axes) == 1

    def test_both_none_returns_none(self):
        assert ReportBuilder()._plot_bg_pixel_histogram(None, None, "A", None, None, "B") is None


class TestPlotBgPixelOverlay:
    def _disp_and_mask(self, seed=0):
        rng = np.random.default_rng(seed)
        disp = rng.integers(0, 255, (48, 48)).astype(np.uint8)
        mask = rng.random((48, 48)) > 0.1
        return disp, mask

    def test_two_panels(self):
        d_a, m_a = self._disp_and_mask(1)
        d_b, m_b = self._disp_and_mask(2)
        fig = ReportBuilder()._plot_bg_pixel_overlay(d_a, m_a, "A", d_b, m_b, "B")
        assert fig is not None
        assert len(fig.axes) == 2

    def test_single_panel(self):
        d_a, m_a = self._disp_and_mask(1)
        fig = ReportBuilder()._plot_bg_pixel_overlay(d_a, m_a, "A", None, None, "B")
        assert fig is not None
        assert len(fig.axes) == 1

    def test_both_none_returns_none(self):
        assert ReportBuilder()._plot_bg_pixel_overlay(None, None, "A", None, None, "B") is None


class TestSectionSnrBackgroundBlocks:
    @pytest.fixture(scope="class")
    @classmethod
    def section_html_both(cls, image_pair):
        img_a, img_b = image_pair
        ra = AnalysisResult(label="A")
        rb = AnalysisResult(label="B")
        return ReportBuilder()._section_snr(ra, rb, img_a, img_b)

    def test_background_model_heading_present(self, section_html_both):
        assert "<h3>3e. Background Model</h3>" in section_html_both

    def test_gradient_analysis_heading_present(self, section_html_both):
        assert "<h3>3f. Background Gradient Analysis</h3>" in section_html_both

    def test_background_model_caption_present(self, section_html_both):
        assert "Interpolated background model" in section_html_both

    def test_pixel_classification_caption_present(self, section_html_both):
        assert "sigma-clip" in section_html_both

    def test_no_background_omits_both_headings(self, no_background_image):
        # No estimate_background() call: 3e/3f must be entirely absent, not
        # present-with-broken-figures (CLAUDE.md's "gate on the right
        # condition" pitfall).
        ra = AnalysisResult(label="A")
        rb = AnalysisResult(label="B")
        html = ReportBuilder()._section_snr(ra, rb, no_background_image, None)
        assert "<h3>3e. Background Model</h3>" not in html
        assert "<h3>3f. Background Gradient Analysis</h3>" not in html

    def test_single_image_mode_no_crash_and_headings_present(self, image_pair):
        img_a, _ = image_pair
        ra = AnalysisResult(label="A")
        rb = AnalysisResult(label="B")
        html = ReportBuilder()._section_snr(ra, rb, img_a, None)
        assert "<h3>3e. Background Model</h3>" in html
        assert "<h3>3f. Background Gradient Analysis</h3>" in html


class TestWriteInspectorFileBackgroundEntries:
    @pytest.fixture(scope="class")
    @classmethod
    def npz_and_catalog(cls, image_pair, tmp_path_factory):
        img_a, img_b = image_pair
        ra = AnalysisResult(label="A")
        rb = AnalysisResult(label="B")
        out_path = tmp_path_factory.mktemp("bgsection_npz") / "report_inspector.npz"
        ReportBuilder()._write_inspector_file(out_path, img_a, img_b, ra, rb)
        npz = np.load(str(out_path), allow_pickle=False)
        catalog = json.loads(npz["catalog_json"].tobytes().decode("utf-8"))
        return npz, catalog

    @pytest.mark.parametrize("key", [
        "bg_model_a", "bg_model_b", "bg_rms_a", "bg_rms_b",
        "bgfit_surface_a", "bgfit_surface_b", "bgfit_residual_a", "bgfit_residual_b",
    ])
    def test_array_present_and_float32(self, npz_and_catalog, key):
        npz, _ = npz_and_catalog
        assert key in npz.files
        assert npz[key].dtype == np.float32

    def test_background_model_section_has_two_entries(self, npz_and_catalog):
        _, catalog = npz_and_catalog
        entries = catalog["sections"].get("Background Model", [])
        names = {e["name"] for e in entries}
        assert "Background / RMS" in names
        assert "Fitted surface / Residual" in names

    def test_entries_have_nonempty_concept(self, npz_and_catalog):
        _, catalog = npz_and_catalog
        entries = catalog["sections"]["Background Model"]
        for entry in entries:
            assert entry.get("concept")

    def test_background_rms_options_match_expected_keys(self, npz_and_catalog):
        _, catalog = npz_and_catalog
        entries = {e["name"]: e for e in catalog["sections"]["Background Model"]}
        opts = entries["Background / RMS"]["options"]
        assert opts == {
            "Background A": "bg_model_a", "RMS A": "bg_rms_a",
            "Background B": "bg_model_b", "RMS B": "bg_rms_b",
        }

    def test_fitted_surface_aligned_with_background_model(self, npz_and_catalog):
        # Load-bearing alignment check: the residual figure computed in
        # _section_snr subtracts these two arrays element-wise, so they must
        # be exactly the same shape.
        npz, _ = npz_and_catalog
        assert npz["bgfit_surface_a"].shape == npz["bg_model_a"].shape
        assert npz["bgfit_residual_a"].shape == npz["bg_model_a"].shape

    def test_single_image_mode_only_a_entries(self, image_pair, tmp_path_factory):
        img_a, _ = image_pair
        ra = AnalysisResult(label="A")
        rb = AnalysisResult(label="B")
        out_path = tmp_path_factory.mktemp("bgsection_npz_single") / "report_inspector.npz"
        ReportBuilder()._write_inspector_file(out_path, img_a, None, ra, rb)
        npz = np.load(str(out_path), allow_pickle=False)
        assert "bg_model_a" in npz.files
        assert "bg_model_b" not in npz.files

        catalog = json.loads(npz["catalog_json"].tobytes().decode("utf-8"))
        entries = {e["name"]: e for e in catalog["sections"]["Background Model"]}
        opts = entries["Background / RMS"]["options"]
        assert "Background A" in opts
        assert "Background B" not in opts
