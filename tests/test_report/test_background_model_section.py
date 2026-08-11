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


class TestSectionSnrSourceMaskBlock:
    """Section 3g — the source-masked background diagnostic."""

    @pytest.fixture(scope="class")
    @classmethod
    def section_html(cls, image_pair):
        img_a, img_b = image_pair
        return ReportBuilder()._section_snr(
            AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)

    def test_heading_present(self, section_html):
        assert "<h3>3g. Source-Masked Background Check</h3>" in section_html

    def test_appended_after_3f_so_no_reletter_was_needed(self, section_html):
        """3g must follow 3e/3f, never displace them."""
        assert (section_html.index("3e. Background Model")
                < section_html.index("3f. Background Gradient Analysis")
                < section_html.index("3g. Source-Masked Background Check"))

    def test_comparison_table_columns_present(self, section_html):
        for token in ("Current", "Source-masked", "Background median (ADU)",
                      "Background RMS median (ADU)", "Mesh cells used",
                      "Source mask coverage"):
            assert token in section_html, token

    def test_detection_thresholds_are_reported(self, section_html):
        """The auditability requirement: thresholds visible, not hidden in a mask."""
        assert "Detection thresholds" in section_html
        assert "Threshold (ADU, smoothed)" in section_html

    def test_states_it_is_diagnostic_only(self, section_html):
        """The section must not imply it changed any reported SNR."""
        assert "diagnostic only" in section_html.lower()

    def test_methodology_box_present(self, section_html):
        assert "Understanding the source-masked check" in section_html

    def test_captions_present(self, section_html):
        # Caption text is literal HTML; matplotlib-drawn titles are pixels only
        # and can never be asserted against the document (CLAUDE.md pitfall).
        assert "Which pixels were excluded" in section_html
        assert "Fraction of each mesh cell left unmasked" in section_html

    def test_no_background_omits_heading(self, no_background_image):
        html = ReportBuilder()._section_snr(
            AnalysisResult(label="A"), AnalysisResult(label="B"),
            no_background_image, None)
        assert "3g. Source-Masked Background Check" not in html

    def test_single_image_mode(self, image_pair):
        img_a, _ = image_pair
        html = ReportBuilder()._section_snr(
            AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, None)
        assert "<h3>3g. Source-Masked Background Check</h3>" in html

    def test_3e_limitation_text_points_at_3g(self, section_html):
        """Removing a documented limitation must update the text describing it."""
        assert "see <strong>3g</strong> below" in section_html


class TestSourceMaskFallbackReporting:
    def test_fallback_reason_renders_as_a_warning(self, image_pair, monkeypatch):
        """An aborted estimate must surface visibly, not vanish silently."""
        import report.report_builder as rb
        from analysis.source_mask import MaskedBackgroundResult

        def _fake(*_args, **_kwargs):
            return MaskedBackgroundResult(
                surface=None, rms_median=None, background_median=None, fit=None,
                n_cells=0, n_cells_total=64, order_cap=None, coverage=0.99,
                fallback_reason="source mask covers 99.0% of the frame",
                source_mask=None, scaffold=None)

        monkeypatch.setattr(rb, "source_masked_background", _fake)
        img_a, img_b = image_pair
        html = ReportBuilder()._section_snr(
            AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)
        assert "<h3>3g. Source-Masked Background Check</h3>" in html
        assert "source mask covers 99.0% of the frame" in html
        assert "warn-box" in html


class TestWriteInspectorFileSourceMaskEntries:
    @pytest.fixture(scope="class")
    @classmethod
    def npz_and_catalog(cls, image_pair, tmp_path_factory):
        img_a, img_b = image_pair
        out_path = tmp_path_factory.mktemp("bgmask_npz") / "report_inspector.npz"
        ReportBuilder()._write_inspector_file(
            out_path, img_a, img_b, AnalysisResult(label="A"), AnalysisResult(label="B"))
        npz = np.load(str(out_path), allow_pickle=False)
        catalog = json.loads(npz["catalog_json"].tobytes().decode("utf-8"))
        return npz, catalog

    @pytest.mark.parametrize("key,dtype", [
        ("bgmask_surface_a", np.float32), ("bgmask_surface_b", np.float32),
        ("bgmask_delta_a", np.float32), ("bgmask_delta_b", np.float32),
        ("bgmask_mask_a", np.uint8), ("bgmask_mask_b", np.uint8),
    ])
    def test_array_present_with_expected_dtype(self, npz_and_catalog, key, dtype):
        npz, _ = npz_and_catalog
        assert key in npz.files
        assert npz[key].dtype == dtype

    def test_masked_surface_aligned_with_background_model(self, npz_and_catalog):
        """The delta is an element-wise subtraction, so the grids must match."""
        npz, _ = npz_and_catalog
        assert npz["bgmask_surface_a"].shape == npz["bg_model_a"].shape
        assert npz["bgmask_delta_a"].shape == npz["bg_model_a"].shape
        assert npz["bgmask_mask_a"].shape == npz["bg_model_a"].shape

    def test_delta_equals_masked_minus_current(self, npz_and_catalog):
        npz, _ = npz_and_catalog
        assert np.allclose(npz["bgmask_delta_a"],
                           npz["bgmask_surface_a"] - npz["bg_model_a"], atol=1e-4)

    def test_catalog_entry_present_with_concept(self, npz_and_catalog):
        _, catalog = npz_and_catalog
        entries = {e["name"]: e for e in catalog["sections"]["Background Model"]}
        assert "Source-masked background" in entries
        entry = entries["Source-masked background"]
        assert entry.get("concept")
        assert "diagnostic only" in entry["concept"].lower()
        assert set(entry["options"]) == {
            "Masked background A", "Change vs current A", "Source mask A",
            "Masked background B", "Change vs current B", "Source mask B"}

    def test_mask_is_binary(self, npz_and_catalog):
        npz, _ = npz_and_catalog
        assert set(np.unique(npz["bgmask_mask_a"]).tolist()) <= {0, 1}


class TestSectionSnrUserExclusionRegions:
    """Section 3g must report user-drawn regions as the user's, not the detector's."""

    REGION = [{"kind": "polygon",
               "points": [(0.30, 0.30), (0.70, 0.30), (0.70, 0.70), (0.30, 0.70)]}]

    @pytest.fixture(scope="class")
    @classmethod
    def html_with_regions(cls, image_pair):
        img_a, img_b = image_pair
        # The transport the GUI uses: AnalysisThread assigns onto the images,
        # because the settings dict never reaches ReportBuilder.
        for img in (img_a, img_b):
            img.bg_exclusion_regions = cls.REGION
        try:
            yield ReportBuilder()._section_snr(
                AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)
        finally:
            for img in (img_a, img_b):
                img.bg_exclusion_regions = []

    def test_regions_are_reported_as_the_users(self, html_with_regions):
        assert "of which you excluded by hand" in html_with_regions
        assert "Your exclusion regions are in use" in html_with_regions

    def test_states_the_scaffold_also_used_them(self, html_with_regions):
        """Stage-1 injection is the non-obvious half — it must be explained."""
        assert "dropped from step 1" in html_with_regions

    def test_absent_when_no_regions_drawn(self, image_pair):
        # Sets its own state rather than relying on the sibling fixture's
        # teardown: `image_pair` is class-scoped and shared, so a generator
        # fixture that mutates it has not torn down yet when this test runs.
        img_a, img_b = image_pair
        saved = [getattr(i, "bg_exclusion_regions", []) for i in (img_a, img_b)]
        try:
            for img in (img_a, img_b):
                img.bg_exclusion_regions = []
            html = ReportBuilder()._section_snr(
                AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)
            assert "<h3>3g. Source-Masked Background Check</h3>" in html
            assert "Your exclusion regions are in use" not in html
        finally:
            for img, prev in zip((img_a, img_b), saved):
                img.bg_exclusion_regions = prev

    def test_missing_attribute_is_tolerated(self, image_pair):
        """AstroImages built outside the GUI never have the attribute set."""
        img_a, _ = image_pair
        saved = img_a.bg_exclusion_regions
        try:
            del img_a.bg_exclusion_regions
            html = ReportBuilder()._section_snr(
                AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, None)
            assert "<h3>3g. Source-Masked Background Check</h3>" in html
        finally:
            img_a.bg_exclusion_regions = saved


class TestSectionSnrThresholdProvenance:
    def test_threshold_table_reports_which_bound_won(self, image_pair):
        img_a, img_b = image_pair
        html = ReportBuilder()._section_snr(
            AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)
        assert "<th>Set by</th>" in html
        assert "surface brightness" in html
