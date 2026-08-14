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
        # Source masking is on by default, so 3e presents the fitted surface
        # rather than photutils' interpolated mesh.
        assert "fitted to the mesh cells that survived the source mask" in section_html_both

    def test_states_which_model_is_in_use(self, section_html_both):
        """3e's numbers change materially with the flag, so the report must say
        which estimate produced them rather than leaving it to be inferred."""
        assert "Model in use: <strong>source-masked</strong>" in section_html_both

    def test_warns_that_global_snr_is_not_comparable_across_the_setting(
            self, section_html_both):
        """The most confusing consequence of adopting the masked background, and
        the one most likely to be read as a catastrophe.

        Global SNR is median(pixels >3 sigma)/sigma. Once the background stops
        absorbing the nebula, the nebula joins that population — measured on a
        gradient+nebula frame the qualifying pixels went 1.5% -> 11.4% and the
        median fell from 30.3 to 3.7, while recovered signal rose from 46% to 84%
        of truth. Without this paragraph a user flipping the checkbox sees the
        headline number drop 8x and concludes the feature is broken.
        """
        assert "Expect the global SNR number to move" in section_html_both
        assert "never across this setting" in section_html_both

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
            # Present only because source masking was applied — the plain estimate
            # that would otherwise have been used, so the wipe has something to
            # compare the adopted model against.
            "Background A (unmasked)": "bg_model_unmasked_a",
            "Background B (unmasked)": "bg_model_unmasked_b",
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

    def test_states_that_this_is_the_background_in_use(self, section_html):
        """The inverse of the old claim, and the reason the old one had to go:
        this estimate now feeds every SNR above it, so saying otherwise would be
        false rather than merely stale."""
        assert "This is the background in use" in section_html
        assert "diagnostic only" not in section_html.lower()

    def test_plane_quadric_agreement_is_reported(self, section_html):
        """Whether a fitted quadric is extrapolating cannot be judged from the
        cell count alone at every image scale, so the divergence is a number the
        reader gets rather than a decision made silently."""
        assert "Plane vs quadric — max divergence" in section_html
        assert "of which beyond the measured cells" in section_html

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

    def test_3e_points_at_3g_for_the_audit_trail(self, section_html):
        """Removing a documented limitation must update the text describing it.

        3e used to describe source masking as a limitation it did not address and
        send the reader to 3g to see what it would be worth; now 3e *is* masked,
        so the pointer has to be to the evidence rather than to the shortfall.
        """
        assert "see <strong>3g</strong> for the detection thresholds" in section_html
        assert "Known limitation" not in section_html


class TestSourceMaskFallbackReporting:
    def test_fallback_reason_renders_as_a_warning(self, image_pair):
        """An aborted estimate must surface visibly, not vanish silently.

        Patched onto the image rather than onto report_builder's import of
        source_masked_background: estimate_background() stores its result there
        even when it failed, precisely so a fallback stays reportable, and the
        report reads that attribute rather than recomputing.
        """
        from analysis.source_mask import MaskedBackgroundResult

        failed = MaskedBackgroundResult(
            surface=None, rms_median=None, background_median=None, fit=None,
            n_cells=0, n_cells_total=64, order_cap=None, coverage=0.99,
            fallback_reason="source mask covers 99.0% of the frame",
            source_mask=None, scaffold=None)

        img_a, img_b = image_pair
        saved = [img.source_mask_result for img in (img_a, img_b)]
        try:
            for img in (img_a, img_b):
                img.source_mask_result = failed
            html = ReportBuilder()._section_snr(
                AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)
        finally:
            for img, prev in zip((img_a, img_b), saved):
                img.source_mask_result = prev
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

    def test_delta_equals_masked_minus_unmasked(self, npz_and_catalog):
        """Against the unmasked model, not bg_model_a. Once the masked estimate
        is adopted the two are the same array, so a delta computed against
        bg_model_a would be identically zero and the panel useless."""
        npz, _ = npz_and_catalog
        assert np.allclose(npz["bgmask_delta_a"],
                           npz["bgmask_surface_a"] - npz["bg_model_unmasked_a"],
                           atol=1e-4)
        assert not np.allclose(npz["bgmask_delta_a"], 0.0)

    def test_catalog_entry_present_with_concept(self, npz_and_catalog):
        _, catalog = npz_and_catalog
        entries = {e["name"]: e for e in catalog["sections"]["Background Model"]}
        assert "Source-masked background" in entries
        entry = entries["Source-masked background"]
        assert entry.get("concept")
        assert "is</b> the background used" in entry["concept"]
        assert set(entry["options"]) == {
            "Masked background A", "Change vs unmasked A", "Source mask A",
            "Masked background B", "Change vs unmasked B", "Source mask B"}

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
        """Rebuilds the background with the regions attached, which is the only
        transport that works now that the masked estimate IS the background.

        AnalysisThread does exactly this, and specifically does it *before* the
        pre-pass: attaching regions to an image whose background already exists
        cannot retroactively change it, so set_background_exclusion_mask()
        invalidates rather than letting the report show a mask the subtracted
        model never saw.
        """
        from analysis.inspector_regions import exclusion_mask
        img_a, img_b = image_pair
        for img in (img_a, img_b):
            img.bg_exclusion_regions = cls.REGION
            img.set_background_exclusion_mask(
                exclusion_mask(img.data.shape[:2], cls.REGION))
            img.estimate_background()
        try:
            yield ReportBuilder()._section_snr(
                AnalysisResult(label="A"), AnalysisResult(label="B"), img_a, img_b)
        finally:
            for img in (img_a, img_b):
                img.bg_exclusion_regions = []
                img.set_background_exclusion_mask(None)
                img.estimate_background()

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
        #
        # Clearing bg_exclusion_regions is no longer enough on its own — the
        # sibling fixture rebuilds the *background* with the regions applied, and
        # the report reads the mask off that result. The background has to be
        # rebuilt too, which is exactly the invalidation contract.
        img_a, img_b = image_pair
        saved = [getattr(i, "bg_exclusion_regions", []) for i in (img_a, img_b)]
        try:
            for img in (img_a, img_b):
                img.bg_exclusion_regions = []
                img.set_background_exclusion_mask(None)
                img.estimate_background()
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
