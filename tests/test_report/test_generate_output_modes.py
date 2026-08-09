"""Tests for ReportBuilder.generate()'s report_format modes.

Covers the two report_format values: REPORT_OUTPUT_SUBFOLDER (default —
images saved as separate PNGs in a sibling <stem>/ folder, referenced by
relative path) and REPORT_OUTPUT_SINGLE_FILE (legacy — every image embedded
as a base64 data URI, no files on disk besides the HTML). This is the first
end-to-end test of generate() in the suite; existing tests only exercise
individual helpers or _write_inspector_file directly.
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")

from core.models import REPORT_OUTPUT_SINGLE_FILE, REPORT_OUTPUT_SUBFOLDER
from report.report_builder import ReportBuilder, _slugify, _export_ctx

_SRC_RE = re.compile(r'src="([^"]+)"')


def _img_srcs(html: str) -> list[str]:
    return _SRC_RE.findall(html)


class TestFolderMode:
    def test_creates_sibling_images_directory_named_like_the_stem(self, astro_image_a, tmp_path):
        out = ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SUBFOLDER,
        )
        images_dir = tmp_path / out.stem
        assert images_dir.is_dir()
        assert list(images_dir.glob("*.png"))

    def test_html_has_no_base64_and_srcs_resolve_to_real_files(self, astro_image_a, tmp_path):
        out = ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SUBFOLDER,
        )
        html = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," not in html

        srcs = [s for s in _img_srcs(html) if s.startswith(f"{out.stem}/")]
        assert srcs
        for src in srcs:
            assert (tmp_path / src).is_file()

    def test_filenames_are_numbered_and_descriptive(self, astro_image_a, tmp_path):
        out = ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SUBFOLDER,
        )
        names = sorted(p.name for p in (tmp_path / out.stem).glob("*.png"))
        assert names
        assert all(re.match(r"^fig\d{3}_[a-z0-9_]+\.png$", n) for n in names)
        # Sequence starts at 1 and is contiguous with document order.
        assert names[0].startswith("fig001_")

    def test_export_context_is_reset_after_generate(self, astro_image_a, tmp_path):
        ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SUBFOLDER,
        )
        assert _export_ctx.enabled is False
        assert _export_ctx.images_dir is None
        assert _export_ctx.counter == 0


class TestSingleFileMode:
    def test_no_sibling_images_directory_created(self, astro_image_a, tmp_path):
        out = ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SINGLE_FILE,
        )
        assert not (tmp_path / out.stem).exists()

    def test_html_embeds_images_as_base64(self, astro_image_a, tmp_path):
        out = ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SINGLE_FILE,
        )
        html = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in html
        assert html.count("<img") > 0

    def test_export_context_stays_disabled(self, astro_image_a, tmp_path):
        ReportBuilder().generate(
            astro_image_a, output_dir=tmp_path, open_browser=False,
            report_format=REPORT_OUTPUT_SINGLE_FILE,
        )
        assert _export_ctx.enabled is False


class TestSlugify:
    def test_lowercases_and_replaces_punctuation(self):
        assert _slugify("Edge #3 ESF + LSF") == "edge_3_esf_lsf"

    def test_already_clean_slug_unchanged(self):
        assert _slugify("corr_std_10px") == "corr_std_10px"

    def test_collapses_whitespace_and_strips_edges(self):
        assert _slugify("  Background   model  ") == "background_model"

    def test_empty_input_falls_back(self):
        assert _slugify("") == "img"
        assert _slugify("###") == "img"

    def test_length_is_capped(self):
        long_text = "x" * 200
        slug = _slugify(long_text, max_len=60)
        assert len(slug) <= 60


class TestImageExportContextCounter:
    def test_next_paths_increments_and_formats(self, tmp_path):
        from report.report_builder import _ImageExportContext
        ctx = _ImageExportContext()
        ctx.enabled = True
        ctx.images_dir = tmp_path
        ctx.html_stem = "report_test"

        path1, rel1 = ctx.next_paths("Background model")
        path2, rel2 = ctx.next_paths("Background model")

        assert path1.name == "fig001_background_model.png"
        assert rel1 == "report_test/fig001_background_model.png"
        assert path2.name == "fig002_background_model.png"
        assert rel2 == "report_test/fig002_background_model.png"
