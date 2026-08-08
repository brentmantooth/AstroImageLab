"""Shared reader/vocabulary for the `<stem>_inspector.npz` companion file.

`report/report_builder.py` writes the file; `gui/report_inspector.py` and
`gui/data_inspector.py` both read it.  Everything all three need to agree on lives
here so there is exactly one definition of the catalog format and one source of
truth for panel display names and descriptions.

Headless by design (numpy + stdlib only, no PyQt6, no matplotlib) so the panel
naming/description logic is covered by the normal test suite — the GUI modules that
consume it are not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Panel vocabulary — display names and descriptions
# ---------------------------------------------------------------------------

def panel_display_name(pkey: str) -> str:
    """Derive a human-readable label from a SpatialDetailAnalyzer panels dict key.
    Dynamic (not a hardcoded map) so every scale it computes — including future
    additions — automatically appears in the inspector catalog."""
    nrm = pkey.startswith("nrm_")
    base = pkey[4:] if nrm else pkey
    if base == "original":
        name = "Original Image (ROI)"
    elif base.startswith("std_") and base.endswith("px"):
        name = f"Std Dev {base[4:-2]} px"
    elif base.startswith("log_"):
        name = f"LoG σ {base[4:]} px"
    elif base.startswith("wavelet_"):
        name = f"Wavelet level {base[8:]}"
    elif base.startswith("entropy_") and base.endswith("px"):
        name = f"Entropy {base[8:-2]} px"
    elif base.startswith("gradient_"):
        name = f"Gradient σ {base[9:]} px"
    elif base.startswith("localgrad_"):
        name = f"Local Grad. Energy σ {base[10:]} px"
    elif base.startswith("loclap_"):
        name = f"Local Var. of Laplacian σ {base[7:]} px"
    else:
        name = base
    return f"{name} (noise-normalized)" if nrm else name


def panel_concept(pkey: str) -> str:
    """Short explanatory text for the inspector's concept box: what concept a
    Spatial Detail panel measures, what it responds to, and whether higher or lower
    indicates a stronger response. Dynamic (not a hardcoded map), mirroring
    panel_display_name's key-family parsing."""
    nrm = pkey.startswith("nrm_")
    base = pkey[4:] if nrm else pkey
    if base == "original":
        return ("Source image (mean-signal-normalised), not a detail metric itself — "
                 "shown for reference alongside the derived maps below.")
    elif base.startswith("std_") and base.endswith("px"):
        concept = ("Measures <b>Detail</b> (local texture / variability) within a square "
                   "window. Responds to any local brightness variation — nebula filaments, "
                   "star halos, and pixel noise all raise this value; it cannot distinguish "
                   "real structure from noise on its own. <b>Higher = more local variation</b> "
                   "(more preserved detail in nebula regions, but also more noise in blank sky).")
    elif base.startswith("log_"):
        concept = ("Measures <b>Detail</b> as edge/curvature strength (2nd derivative) at a "
                   "Gaussian smoothing scale σ. Responds to intensity boundaries — filament "
                   "edges, shell rims — that survive smoothing at this scale. "
                   "<b>Higher = sharper, more defined edges</b> at this scale (also amplifies "
                   "noise at small σ).")
    elif base.startswith("wavelet_"):
        concept = ("Measures <b>Detail</b> at a specific spatial scale (~2<sup>level</sup> px) "
                   "via wavelet decomposition. Responds to structure whose size matches this "
                   "level — fine filaments/star cores at low levels, broader emission knots/"
                   "shell edges at higher levels. <b>Higher coefficient magnitude = more "
                   "structure at this scale</b>; see the wavelet SNR chart for whether it's "
                   "signal- or noise-dominated.")
    elif base.startswith("entropy_") and base.endswith("px"):
        concept = ("Measures <b>Detail</b> as texture complexity (Shannon entropy, bits, log&#8322;) "
                   "of the local gray-level histogram within a square window, using data quantized "
                   "into a fixed number of levels beforehand (see 8h methodology). Responds to how "
                   "rich/unpredictable local tonal structure is — tangled nebulosity, mottled dust, "
                   "unresolved star fields — but also to noise, even more readily than Local σ. "
                   "<b>Higher = richer/more unpredictable local tonal distribution</b> (cannot "
                   "distinguish real structure from noise on its own — see the noise-corrected "
                   "score).")
    elif base.startswith("gradient_"):
        concept = ("Measures <b>Detail</b> as edge sharpness (1st derivative, slope magnitude) "
                   "at Gaussian scale σ. Responds to how abrupt an intensity transition is at "
                   "this scale. <b>Higher = crisper, more abrupt boundaries</b> (also amplifies "
                   "noise at small σ).")
    elif base.startswith("localgrad_"):
        concept = ("Measures <b>Acutance concentration</b> — the windowed mean of the squared "
                   "gradient magnitude, mean(|G|&sup2;), at Gaussian scale σ (Section 8i). Unlike "
                   "the raw per-pixel gradient map (8f), which marks where <i>an</i> edge is, this "
                   "asks how much edge energy is packed into each neighbourhood, so broad regions "
                   "of consistently steep transitions score higher than a single isolated edge of "
                   "the same peak height. Squaring weights strong edges far above weak ones. "
                   "<b>Higher = sharpness energy more strongly concentrated here</b> (also "
                   "amplifies noise at small σ).")
    elif base.startswith("loclap_"):
        concept = ("Measures <b>Focus concentration</b> — the windowed variance of the signed "
                   "Laplacian-of-Gaussian, var(LoG), at scale σ (Section 8j). Unlike the raw "
                   "per-pixel |LoG| map (8d), which marks individual curvature features, this asks "
                   "how much second-derivative energy varies within each neighbourhood — the "
                   "classic autofocus criterion. Variance rather than mean-of-squares (8i's form) "
                   "because the Laplacian is signed, so its local mean is near zero and would "
                   "cancel. <b>Higher = focus energy more strongly concentrated here</b>; a "
                   "defocused or smoothed image collapses toward zero.")
    else:
        return ""
    if nrm:
        concept += (" <b>Noise-normalised:</b> this map is divided by this image's own "
                    "empirical noise floor at this scale/operator, putting A and B on a fair "
                    "shared colour scale. <b>Higher = stronger real signal relative to this "
                    "image's own noise</b>; values at or below ~1 indicate the response here "
                    "is no stronger than this image's own noise floor.")
    return concept


def _concept_from_npz_key(npz_key: str) -> str:
    """Recover a panel description from a Spatial Detail npz key (`sp_<pkey>_a`).

    Back-fill for `.npz` files written before a description existed for that panel
    family — the concept string lives in the file's own catalog_json, so without this
    an older file would keep showing a blank description box forever.  Returns "" for
    any key that is not a Spatial Detail panel.
    """
    if not npz_key.startswith("sp_"):
        return ""
    base = npz_key[3:]
    for suffix in ("_a", "_b", "_diff"):
        if base.endswith(suffix):
            return panel_concept(base[: -len(suffix)])
    return ""


# ---------------------------------------------------------------------------
# Catalog data structures
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    name:    str
    options: dict[str, str]   # display_label → npz_key; ordered
    concept: str = ""         # optional explanatory text (Spatial Detail section)


@dataclass
class InspectorData:
    label_a: str
    label_b: str
    sim_label_ref: str
    filename_a: str = ""
    filename_b: str = ""
    single_image: bool = False
    sections: dict[str, list[Entry]] = field(default_factory=dict)


def load_inspector_data(npz: "np.lib.npyio.NpzFile") -> InspectorData:
    json_bytes = npz["catalog_json"].tobytes()
    cat = json.loads(json_bytes.decode("utf-8"))
    data = InspectorData(
        label_a=cat.get("label_a", "Image A"),
        label_b=cat.get("label_b", "Image B"),
        sim_label_ref=cat.get("sim_label_ref", ""),
        filename_a=cat.get("filename_a", ""),
        filename_b=cat.get("filename_b", ""),
        single_image=cat.get("single_image", False),
    )
    npz_keys = set(npz.files)
    for section, entries in cat.get("sections", {}).items():
        parsed = []
        for e in entries:
            # Support both new {options: {...}} and legacy {key_a, key_b} formats
            if "options" in e:
                opts = {k: v for k, v in e["options"].items() if v in npz_keys}
            else:
                opts = {}
                if e.get("key_a") in npz_keys:
                    opts["Image A"] = e["key_a"]
                if e.get("key_b") in npz_keys:
                    opts["Image B"] = e["key_b"]
            if not opts:
                continue
            name = e["name"]
            concept = e.get("concept", "")
            if not concept:
                # File predates this panel family having a description; recover both
                # the name and the text from the npz key it still carries.
                first_key = next(iter(opts.values()))
                concept = _concept_from_npz_key(first_key)
                if concept and first_key.startswith("sp_"):
                    name = panel_display_name(_pkey_from_npz_key(first_key)) or name
            parsed.append(Entry(name=name, options=opts, concept=concept))
        if parsed:
            data.sections[section] = parsed
    return data


def _pkey_from_npz_key(npz_key: str) -> str:
    """`sp_<pkey>_a` → `<pkey>`; "" when the key is not a Spatial Detail panel."""
    if not npz_key.startswith("sp_"):
        return ""
    base = npz_key[3:]
    for suffix in ("_a", "_b", "_diff"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return ""
