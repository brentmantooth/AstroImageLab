"""Unit tests for core/inspector_catalog.py — the shared _inspector.npz vocabulary.

Both inspectors read the catalog through this module, and report_builder writes it
through the same panel-naming functions, so a gap here silently degrades the GUI in a
way no other test would catch (gui/ is untested by design — see CLAUDE.md).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.inspector_catalog import (
    Entry,
    InspectorData,
    _concept_from_npz_key,
    _pkey_from_npz_key,
    load_inspector_data,
    panel_concept,
    panel_display_name,
)
from core.models import (
    ENTROPY_KERNEL_SIZES,
    LOG_SIGMAS,
    STD_KERNEL_SIZES,
    WAVELET_LEVELS,
)


def _all_panel_keys() -> list[str]:
    """Every panel key SpatialDetailAnalyzer.analyze() can put in result["panels"].

    Derived from the same constants the analyzer uses, so adding a scale to
    core/models.py automatically widens these tests rather than leaving a blind spot.
    """
    keys = ["original"]
    keys += [f"std_{ks}px" for ks in STD_KERNEL_SIZES]
    keys += [f"entropy_{ks}px" for ks in ENTROPY_KERNEL_SIZES]
    for sigma in LOG_SIGMAS:
        keys += [f"log_{sigma}", f"gradient_{sigma}",
                 f"localgrad_{sigma}", f"loclap_{sigma}"]
    keys += [f"wavelet_{lvl}" for lvl in range(1, WAVELET_LEVELS + 1)]
    return keys + [f"nrm_{k}" for k in keys if k != "original"]


class TestPanelDisplayName:
    @pytest.mark.parametrize("pkey", _all_panel_keys())
    def test_never_returns_a_bare_key(self, pkey):
        # Regression: localgrad_*/loclap_* fell through to `else: name = base`, so the
        # inspector's Image-set combo listed raw keys like "localgrad_1.5".
        name = panel_display_name(pkey)
        assert name, pkey
        assert name.split(" (noise-normalized)")[0] != pkey.removeprefix("nrm_"), (
            f"{pkey} has no display name of its own")

    def test_known_names(self):
        assert panel_display_name("original") == "Original Image (ROI)"
        assert panel_display_name("std_3px") == "Std Dev 3 px"
        assert panel_display_name("log_1.5") == "LoG σ 1.5 px"
        assert panel_display_name("wavelet_2") == "Wavelet level 2"
        assert panel_display_name("entropy_9px") == "Entropy 9 px"
        assert panel_display_name("gradient_3.0") == "Gradient σ 3.0 px"
        assert panel_display_name("localgrad_1.5") == "Local Grad. Energy σ 1.5 px"
        assert panel_display_name("loclap_6.0") == "Local Var. of Laplacian σ 6.0 px"

    def test_nrm_prefix_is_annotated(self):
        assert panel_display_name("nrm_std_3px") == "Std Dev 3 px (noise-normalized)"
        assert panel_display_name("nrm_loclap_1.5").endswith("(noise-normalized)")

    def test_localgrad_and_gradient_are_distinct(self):
        # Same sigma, different families — they must not collapse to one label.
        assert panel_display_name("gradient_1.5") != panel_display_name("localgrad_1.5")
        assert panel_display_name("log_1.5") != panel_display_name("loclap_1.5")


class TestPanelConcept:
    @pytest.mark.parametrize("pkey", _all_panel_keys())
    def test_every_panel_family_is_described(self, pkey):
        # The guard that would have caught the missing 8i/8j descriptions.
        assert panel_concept(pkey).strip(), f"{pkey} has no description"

    def test_unknown_family_returns_empty(self):
        assert panel_concept("something_new_9px") == ""

    def test_nrm_appends_noise_normalised_note(self):
        plain = panel_concept("std_3px")
        nrm = panel_concept("nrm_std_3px")
        assert nrm.startswith(plain)
        assert "Noise-normalised" in nrm

    def test_localgrad_and_loclap_describe_different_things(self):
        lg, ll = panel_concept("localgrad_1.5"), panel_concept("loclap_1.5")
        assert lg != ll
        assert "Acutance" in lg
        assert "Focus" in ll


class TestNpzKeyRecovery:
    @pytest.mark.parametrize("key,expected", [
        ("sp_std_3px_a", "std_3px"),
        ("sp_std_3px_b", "std_3px"),
        ("sp_std_3px_diff", "std_3px"),
        ("sp_nrm_loclap_6.0_a", "nrm_loclap_6.0"),
        ("display_a", ""),
        ("catalog_json", ""),
    ])
    def test_pkey_from_npz_key(self, key, expected):
        assert _pkey_from_npz_key(key) == expected

    def test_concept_from_npz_key(self):
        assert _concept_from_npz_key("sp_localgrad_1.5_a") == panel_concept("localgrad_1.5")
        assert _concept_from_npz_key("display_a") == ""


def _write_npz(tmp_path, catalog: dict, array_keys: list[str]):
    arrays = {k: np.zeros((4, 4), dtype=np.float32) for k in array_keys}
    blob = json.dumps(catalog, ensure_ascii=False).encode("utf-8")
    arrays["catalog_json"] = np.frombuffer(blob, dtype=np.uint8).copy()
    path = tmp_path / "x_inspector.npz"
    np.savez_compressed(str(path), **arrays)
    return np.load(str(path), allow_pickle=False)


class TestLoadInspectorData:
    def test_parses_labels_and_options(self, tmp_path):
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B",
            "filename_a": "a.fits", "filename_b": "b.fits",
            "single_image": False,
            "sections": {"S": [{"name": "E",
                                 "options": {"Image A": "k_a", "Image B": "k_b"},
                                 "concept": "hello"}]},
        }, ["k_a", "k_b"])
        data = load_inspector_data(npz)
        assert isinstance(data, InspectorData)
        assert (data.label_a, data.label_b) == ("A", "B")
        assert data.filename_a == "a.fits"
        assert data.single_image is False
        entry = data.sections["S"][0]
        assert isinstance(entry, Entry)
        assert entry.options == {"Image A": "k_a", "Image B": "k_b"}
        assert entry.concept == "hello"

    def test_options_missing_from_npz_are_dropped(self, tmp_path):
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B", "sections": {
                "S": [{"name": "E", "options": {"Image A": "k_a", "Image B": "gone"}}]},
        }, ["k_a"])
        entry = load_inspector_data(npz).sections["S"][0]
        assert entry.options == {"Image A": "k_a"}

    def test_entry_with_no_surviving_option_is_dropped(self, tmp_path):
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B",
            "sections": {"S": [{"name": "E", "options": {"Image A": "gone"}}]},
        }, ["k_a"])
        assert "S" not in load_inspector_data(npz).sections

    def test_legacy_key_a_key_b_format(self, tmp_path):
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B",
            "sections": {"S": [{"name": "E", "key_a": "k_a", "key_b": "k_b"}]},
        }, ["k_a", "k_b"])
        entry = load_inspector_data(npz).sections["S"][0]
        assert entry.options == {"Image A": "k_a", "Image B": "k_b"}

    def test_missing_concept_is_backfilled_from_the_npz_key(self, tmp_path):
        # An .npz written before localgrad_* had a description: the catalog carries the
        # raw key as its name and no concept at all.  Both must be recovered on read so
        # existing report files gain the description without being regenerated.
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B",
            "sections": {"Spatial Detail": [
                {"name": "localgrad_1.5",
                 "options": {"Image A": "sp_localgrad_1.5_a"}}]},
        }, ["sp_localgrad_1.5_a"])
        entry = load_inspector_data(npz).sections["Spatial Detail"][0]
        assert entry.name == "Local Grad. Energy σ 1.5 px"
        assert entry.concept == panel_concept("localgrad_1.5")

    def test_backfill_leaves_non_spatial_entries_alone(self, tmp_path):
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B",
            "sections": {"SNR": [{"name": "SNR map",
                                   "options": {"Image A": "snr_display_a"}}]},
        }, ["snr_display_a"])
        entry = load_inspector_data(npz).sections["SNR"][0]
        assert entry.name == "SNR map"
        assert entry.concept == ""

    def test_existing_concept_is_not_overwritten(self, tmp_path):
        npz = _write_npz(tmp_path, {
            "label_a": "A", "label_b": "B",
            "sections": {"Spatial Detail": [
                {"name": "Custom", "concept": "custom text",
                 "options": {"Image A": "sp_std_3px_a"}}]},
        }, ["sp_std_3px_a"])
        entry = load_inspector_data(npz).sections["Spatial Detail"][0]
        assert entry.name == "Custom"
        assert entry.concept == "custom text"
