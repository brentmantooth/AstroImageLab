"""Regression tests for the Report Inspector's .npz panel catalog.

Covers the bug where _write_inspector_file used a hardcoded, stale panel-name
map (std_15px/std_31px from an old STD_KERNEL_SIZES, no Weber entries at all)
that silently omitted panels SpatialDetailAnalyzer actually computes. The fix
derives the catalog dynamically from panels_a.keys() via _panel_display_name.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from analysis.image_filters import SpatialDetailAnalyzer
from core.astro_image import AstroImage
from core.models import AnalysisResult
from report.report_builder import ReportBuilder


def _make_two_image_pair(tmp_path_factory) -> tuple[AstroImage, AstroImage]:
    out = tmp_path_factory.mktemp("inspector_fits")
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    h, w = 256, 256
    for path, rng in ((out / "a.fits", rng_a), (out / "b.fits", rng_b)):
        data = rng.normal(1000.0, 20.0, (h, w)).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = 500.0 * np.exp(-0.5 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / 30.0 ** 2))
        data += blob.astype(np.float32)
        hdr = ap_fits.Header()
        hdr["EGAIN"] = 1.0
        hdr["GAIN"] = 1.0
        ap_fits.writeto(str(path), data, hdr, overwrite=True)
    img_a = AstroImage(str(out / "a.fits"), label="A")
    img_a.load()
    img_a.estimate_background()
    img_b = AstroImage(str(out / "b.fits"), label="B")
    img_b.load()
    img_b.estimate_background()
    return img_a, img_b


@pytest.fixture(scope="module")
def inspector_npz_path(tmp_path_factory):
    img_a, img_b = _make_two_image_pair(tmp_path_factory)
    spatial = SpatialDetailAnalyzer().analyze(img_a, img_b)
    result_a = AnalysisResult(label="A", spatial_metrics=spatial)
    result_b = AnalysisResult(label="B", spatial_metrics=spatial)

    out_path = tmp_path_factory.mktemp("inspector_out") / "report_inspector.npz"
    ReportBuilder()._write_inspector_file(out_path, img_a, img_b, result_a, result_b)
    return out_path, spatial


class TestInspectorCatalogDynamicPanels:
    def test_every_computed_panel_is_cataloged(self, inspector_npz_path):
        path, spatial = inspector_npz_path
        npz = np.load(str(path), allow_pickle=False)
        catalog = json.loads(npz["catalog_json"].tobytes().decode("utf-8"))
        spatial_entries = catalog["sections"].get("Spatial Detail", [])
        cataloged_names = {e["name"] for e in spatial_entries}

        panels = spatial["panels"]
        assert len(spatial_entries) == len(panels)
        for pkey in panels:
            from report.report_builder import _panel_display_name
            assert _panel_display_name(pkey) in cataloged_names

    def test_weber_panels_present(self, inspector_npz_path):
        # Regression: the old hardcoded _PANEL_IMAGE_SETS never included Weber at all.
        _, spatial = inspector_npz_path
        weber_keys = [k for k in spatial["panels"] if k.startswith("weber_")]
        assert weber_keys, "fixture should have produced Weber panels"

    def test_all_std_kernel_sizes_present(self, inspector_npz_path):
        # Regression: the old map referenced std_15px/std_31px from a stale
        # STD_KERNEL_SIZES and dropped std_3px/std_10px from the current one.
        from core.models import STD_KERNEL_SIZES
        _, spatial = inspector_npz_path
        for ks in STD_KERNEL_SIZES:
            assert f"std_{ks}px" in spatial["panels"]

    def test_noise_normalized_panels_cataloged(self, inspector_npz_path):
        path, spatial = inspector_npz_path
        npz = np.load(str(path), allow_pickle=False)
        catalog = json.loads(npz["catalog_json"].tobytes().decode("utf-8"))
        spatial_entries = catalog["sections"].get("Spatial Detail", [])
        cataloged_names = {e["name"] for e in spatial_entries}
        nrm_keys = [k for k in spatial["panels"] if k.startswith("nrm_")]
        assert nrm_keys
        assert any("(noise-normalized)" in n for n in cataloged_names)
