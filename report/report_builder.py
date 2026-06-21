from __future__ import annotations

import base64
import io
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import fftconvolve
from scipy.ndimage import zoom as _ndimage_zoom, gaussian_filter as _gaussian_filter
from scipy.interpolate import griddata as _griddata
from PIL import Image as _PILImage

from core.models import (AnalysisResult, HALO_FIT_RADIUS_PX, XS_LINE_ALPHA, GLASS_REFRACTIVE_INDEX,
                          PSF_SPATIAL_MAP_SIZE, PSF_SPATIAL_MAP_SMOOTH_SIGMA, EDGE_ROI_MAP_INDICATOR_PX,
                          LABEL_MAX_LEN, REF_SEEING_ARCSEC, REF_SEEING_BETA,
                          ABERRATION_MIN_STARS, ABERRATION_OUTER_RADIUS_FRAC)
from core.astro_image import AstroImage

_TEST_IMAGE_PATH = Path(__file__).parent.parent / "resources" / "ContrastTestImage.png"


def _inspector_display(img: AstroImage, max_dim: int = 2048) -> np.ndarray:
    """Return uint8 stretched display image, downsampled so max dimension ≤ max_dim."""
    arr = img.display_image(stretch=True)
    h, w = arr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        pil = _PILImage.fromarray(arr)
        arr = np.array(pil.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS))
    return arr



def _make_moffat_kernel(fwhm_px: float, beta: float = REF_SEEING_BETA,
                         size: int | None = None) -> np.ndarray:
    """2D Moffat PSF kernel at native pixel scale, normalized to unit sum."""
    if size is None:
        size = max(25, int(fwhm_px * 6))
        if size % 2 == 0:
            size += 1
    gamma = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    kern = 1.0 / (1.0 + ((x - cx) ** 2 + (y - cy) ** 2) / gamma ** 2) ** beta
    return kern / kern.sum()


def _ref_fwhm_px(pa: dict, pb: dict,
                  ref_seeing_arcsec: float = REF_SEEING_ARCSEC) -> float | None:
    """Derive reference PSF FWHM in pixels from the plate scale encoded in psf_metrics.
    Returns None when plate scale is unknown (both images report 1.0 arcsec/px default)."""
    for p in (pa, pb):
        fwhm_px = p.get("fwhm_px") or 0.0
        fwhm_as = p.get("fwhm_arcsec") or 0.0
        if fwhm_px > 0 and fwhm_as > 0:
            pixel_scale = fwhm_as / fwhm_px   # arcsec / px
            return ref_seeing_arcsec / pixel_scale
    return None


def _psf_make_map(pts: list, img_h: int, img_w: int) -> "np.ndarray | None":
    """Interpolate per-star (x, y, val) triplets onto a smoothed float32 grid.

    Grid long-axis is PSF_SPATIAL_MAP_SIZE px with aspect ratio preserved.
    Returns None when pts is empty.
    """
    if not pts:
        return None
    if img_w >= img_h:
        gw = PSF_SPATIAL_MAP_SIZE
        gh = max(1, int(PSF_SPATIAL_MAP_SIZE * img_h / img_w))
    else:
        gh = PSF_SPATIAL_MAP_SIZE
        gw = max(1, int(PSF_SPATIAL_MAP_SIZE * img_w / img_h))
    gx, gy = np.meshgrid(np.linspace(0, img_w, gw), np.linspace(0, img_h, gh))
    coords = np.array([(p[0], p[1]) for p in pts])
    vals   = np.array([p[2]         for p in pts])
    m  = _griddata(coords, vals, (gx, gy), method="linear")
    nn = _griddata(coords, vals, (gx, gy), method="nearest")
    m  = np.where(np.isnan(m), nn, m)
    return _gaussian_filter(m, sigma=PSF_SPATIAL_MAP_SMOOTH_SIGMA).astype(np.float32)


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; max-width: 960px;
       margin: 0 auto; padding: 20px; color: #222; background: #fafafa; }
h1 { background: #1a3a5c; color: white; padding: 14px 18px;
     border-radius: 6px; margin-bottom: 4px; }
h2 { background: #2d6da3; color: white; padding: 8px 14px;
     border-radius: 4px; margin-top: 28px; }
h3 { color: #1a3a5c; border-bottom: 2px solid #2d6da3;
     padding-bottom: 4px; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th { background: #2d6da3; color: white; padding: 8px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #dde; }
tr:nth-child(even) { background: #f0f4fa; }
.better { background: #d4edda !important; font-weight: bold; }
.worse  { background: #f8d7da !important; }
.warn-box { background: #fff3cd; border: 1px solid #ffc107;
            border-radius: 4px; padding: 10px 14px; margin: 10px 0; }
details.info-box, .info-box { background: #d1ecf1; border: 1px solid #bee5eb;
            border-radius: 4px; padding: 10px 14px; margin: 10px 0; }
details.info-box > summary { cursor: pointer; font-weight: bold;
            user-select: none; padding: 2px 0; list-style: none; }
details.info-box > summary::-webkit-details-marker { display: none; }
details.info-box > summary::before { content: "▶ "; font-size: 0.85em; }
details.info-box[open] > summary::before { content: "▼ "; font-size: 0.85em; }
.bw-warn { background: #f8d7da; border: 1px solid #f5c6cb;
           border-radius: 4px; padding: 12px 16px; margin: 14px 0;
           font-size: 1.05em; }
.metric-label-ok   { color: #155724; font-weight: bold; }
.metric-label-warn { color: #856404; font-weight: bold; }
img { max-width: 100%; height: auto; border: 1px solid #ccc;
      border-radius: 4px; margin: 8px 0; }
.caption { font-style: italic; color: #555; font-size: 0.92em;
           margin-top: -4px; margin-bottom: 12px; }
.error-box { background: #f8d7da; border: 1px solid #f5c6cb;
             border-radius: 4px; padding: 10px 14px; margin: 10px 0;
             font-family: monospace; white-space: pre-wrap; font-size: 0.9em; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def _img_tag(fig: "plt.Figure | str | None", alt: str = "") -> str:
    if fig is None:
        return ""
    if isinstance(fig, str):
        return f'<img src="data:image/png;base64,{fig}" alt="{alt}">'
    return f'<img src="data:image/png;base64,{_fig_to_b64(fig)}" alt="{alt}">'


def _hires_img_tag(fig: "plt.Figure | str | None", alt: str = "") -> str:
    """Like _img_tag but saved at 150 dpi for detail-heavy maps."""
    if fig is None:
        return ""
    if isinstance(fig, str):
        return f'<img src="data:image/png;base64,{fig}" alt="{alt}">'
    return f'<img src="data:image/png;base64,{_fig_to_b64(fig, dpi=150)}" alt="{alt}">'


def _arr_to_b64_png(arr: np.ndarray) -> str:
    """Convert a uint8 H×W (gray) or H×W×3 (RGB) array to base64 PNG at native resolution."""
    if arr.ndim == 2:
        img = _PILImage.fromarray(arr, mode="L")
    else:
        img = _PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _info_box(body: str, title: str = "More information",
              open: bool = False, style: str = "") -> str:
    """Collapsible info panel using HTML5 <details>/<summary>."""
    open_attr  = " open" if open else ""
    style_attr = f' style="{style}"' if style else ""
    return (f'<details class="info-box"{open_attr}{style_attr}>'
            f'<summary>{title}</summary>{body}</details>')


def _arr_img_tag(arr: np.ndarray, alt: str = "") -> str:
    """Inline <img> at native (1:1) pixel resolution from a uint8 numpy array."""
    return f'<img src="data:image/png;base64,{_arr_to_b64_png(arr)}" alt="{alt}" style="max-width:100%;display:block;">'


def _val(v, fmt=".3f", fallback="—") -> str:
    if v is None:
        return fallback
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def _val_pm(v, spread, fmt=".3f", fallback="—") -> str:
    """Format 'value ± MAD'; omits the ± term when spread is None or zero."""
    if v is None:
        return fallback
    s = format(v, fmt)
    if spread is not None and spread > 0:
        s += f" ± {format(spread, fmt)}"
    return s


def _psf_stat_test(va: list, vb: list) -> tuple[str, float | None]:
    """Mann-Whitney U + Cliff's delta for two per-star metric distributions.

    Returns (html, p_value). html is a compact two-line string: effect rating + stars
    on line 1, p-value and delta on line 2. Returns ("", None) if either list < 3 values.
    d > 0 means A values tend to be higher than B.
    """
    from scipy.stats import mannwhitneyu

    if len(va) < 3 or len(vb) < 3:
        return "", None

    _, p = mannwhitneyu(va, vb, alternative="two-sided")

    arr_a = np.array(va)
    arr_b = np.array(vb)
    delta = float(np.sign(arr_a[:, None] - arr_b[None, :]).sum()) / (len(va) * len(vb))
    abs_d = abs(delta)

    if p >= 0.05:
        rating, stars = "n.s.", "~"
    elif abs_d >= 0.474:
        rating, stars = "large", "&#9733;&#9733;&#9733;"
    elif abs_d >= 0.33:
        rating, stars = "medium", "&#9733;&#9733;"
    elif abs_d >= 0.147:
        rating, stars = "small", "&#9733;"
    else:
        rating, stars = "trivial", "~"

    p_str = "p&lt;0.001" if p < 0.001 else f"p={p:.3f}"
    return f"{stars}&nbsp;{rating}<br><small>{p_str},&nbsp;&delta;={delta:+.2f}</small>", float(p)


def _psf_distributions_figure(sd_a: list, sd_b: list,
                               label_a: str, label_b: str) -> tuple[str, str]:
    """Five-panel horizontal distribution figure, one subplot per PSF metric.

    Returns (img_html, caption_html), or ("", "") if no metric has enough data.
    Adapts: stripplot (N<30), swarmplot (30-250), violinplot (N>250).
    Violin: median drawn in magenta, Q1/Q3 in cyan (inner=None + manual lines).
    Single-image mode (sd_b empty): plots Image A row only.
    """
    import seaborn as sns
    import pandas as pd

    metrics = [
        ("fwhm",         "FWHM (px)"),
        ("fwhm_arcsec",  "FWHM (arcsec)"),
        ("beta",         "Moffat β"),
        ("ellipticity",  "Ellipticity"),
        ("eccentricity", "Eccentricity"),
        ("snr",          "Star SNR (ePSF sample)"),
    ]
    # Theoretically ideal or physically expected reference values per metric.
    # FWHM has no universal ideal (seeing-dependent), so it is omitted.
    _IDEAL_REF = {
        "beta":         4.77,   # Kolmogorov atmospheric turbulence
        "ellipticity":  0.0,    # perfectly round star
        "eccentricity": 0.0,    # perfectly round star
    }

    has_b = bool(sd_b)

    has_data = any(
        sum(1 for s in sd_a if s.get(k) is not None) >= 3 and
        (not has_b or sum(1 for s in sd_b if s.get(k) is not None) >= 3)
        for k, _ in metrics
    )
    if not has_data:
        return "", ""

    fig, axes = plt.subplots(6, 1, figsize=(7, 8.5 if has_b else 6))
    fig.subplots_adjust(hspace=0.55, left=0.18, right=0.97, top=0.95, bottom=0.06)
    palette = {label_a: "steelblue", label_b: "tomato"}
    order = [label_a, label_b] if has_b else [label_a]
    plot_types_used: set[str] = set()

    def _draw_boxwhisker(ax, vals_list):
        for i, vals in enumerate(vals_list):
            ax.boxplot(
                [vals], positions=[i], vert=False,
                widths=0.45, zorder=5,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor="none", edgecolor="#00e5ff", linewidth=1.5, alpha=0.9),
                medianprops=dict(color="magenta", linewidth=2.0, alpha=0.9),
                whiskerprops=dict(color="#00e5ff", linewidth=1.5, alpha=0.9),
                capprops=dict(color="#00e5ff", linewidth=1.5, alpha=0.9),
                flierprops=dict(marker="", visible=False),
            )

    for ax, (key, title) in zip(axes, metrics):
        va = [s[key] for s in sd_a if s.get(key) is not None]
        vb = [s[key] for s in sd_b if s.get(key) is not None] if has_b else []

        min_b_ok = not has_b or len(vb) >= 3
        if len(va) < 3 or not min_b_ok:
            ax.set_visible(False)
            continue

        n_max = max(len(va), len(vb)) if has_b else len(va)
        rows = va + vb
        labels = ([label_a] * len(va)) + ([label_b] * len(vb))
        df = pd.DataFrame({"value": rows, "image": labels})
        bw_data = [va, vb] if has_b else [va]

        if n_max < 30:
            plot_types_used.add("strip")
            sns.stripplot(data=df, x="value", y="image",
                          order=order, palette=palette,
                          size=3, jitter=True, ax=ax)
            _draw_boxwhisker(ax, bw_data)
        elif n_max <= 250:
            plot_types_used.add("swarm")
            sns.swarmplot(data=df, x="value", y="image",
                          order=order, palette=palette,
                          size=3, ax=ax)
            _draw_boxwhisker(ax, bw_data)
        else:
            plot_types_used.add("violin")
            # inner=None; box-whisker overlay provides quartile/median markers
            sns.violinplot(data=df, x="value", y="image",
                           order=order, palette=palette,
                           inner=None, linewidth=0.8, ax=ax)
            _draw_boxwhisker(ax, bw_data)

        if key in _IDEAL_REF:
            ax.axvline(_IDEAL_REF[key], color="red", linestyle="--",
                       linewidth=1.0, alpha=0.8, zorder=3)

        ax.set_title(title, fontsize=8, loc="left", pad=2)
        ax.set_xlabel("", fontsize=7)
        ax.set_ylabel("", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7, pad=1)
        ax.spines[["top", "right"]].set_visible(False)

    img_html = _img_tag(fig, "psf_distributions")

    # Build an adaptive caption describing only the plot type(s) actually rendered
    type_desc: list[str] = []
    _bw_desc = (
        "a <span style='color:#00e5ff'><b>cyan box</b></span> spanning Q1–Q3 (IQR), "
        "a <span style='color:magenta'><b>magenta centre line</b></span> at the median, and "
        "<span style='color:#00e5ff'><b>cyan whiskers</b></span> extending to "
        "1.5&nbsp;&times;&nbsp;IQR — all rendered at 90&nbsp;% opacity on top of the distribution"
    )
    if "strip" in plot_types_used:
        type_desc.append(
            "<b>strip plot</b> (N&nbsp;&lt;&nbsp;30): each dot is one star with random "
            f"y-jitter to reduce overlap; {_bw_desc}"
        )
    if "swarm" in plot_types_used:
        type_desc.append(
            "<b>beeswarm / swarm plot</b> (30&nbsp;&le;&nbsp;N&nbsp;&le;&nbsp;250): dots are placed "
            "at their exact measured value with collision-avoidance spread in y so no two dots "
            f"overlap; {_bw_desc}"
        )
    if "violin" in plot_types_used:
        type_desc.append(
            "<b>violin plot</b> (N&nbsp;&gt;&nbsp;250): the filled shape is a kernel density "
            "estimate (KDE) of the distribution &mdash; wider = more stars at that value; "
            f"{_bw_desc}"
        )

    type_sentences = "; ".join(type_desc) + "."

    if has_b:
        intro = (
            f"Each row shows <span style='color:steelblue'><b>Image A (blue)</b></span> above "
            f"<span style='color:tomato'><b>Image B (red)</b></span>. "
        )
        compare_note = (
            f"<b>How to read statistical differences:</b> if the distributions for A and B are "
            f"well-separated (little or no overlap), the two filters produce measurably different "
            f"values for that metric. Overlapping distributions indicate consistent measurements. "
            f"A shift in the magenta median line is the clearest single-value indicator of a "
            f"systematic difference; non-overlapping IQR boxes provide stronger evidence of a real "
            f"separation. "
        )
    else:
        intro = (
            f"Each row shows <span style='color:steelblue'><b>Image A (blue)</b></span> — "
            f"single-image mode, no Image B. "
        )
        compare_note = ""

    caption_html = (
        f'<p class="caption">'
        f"<b>Per-star metric distributions.</b> "
        f"{intro}"
        f"Plot type adapts to the number of stars measured (N): {type_sentences} "
        f"{compare_note}"
        f"A <span style='color:red'><b>red dashed line</b></span> marks the theoretically ideal "
        f"or physically expected reference value where applicable: Moffat&nbsp;&beta;&nbsp;=&nbsp;4.77 "
        f"(Kolmogorov atmospheric turbulence); Ellipticity&nbsp;=&nbsp;Eccentricity&nbsp;=&nbsp;0 "
        f"(perfectly round stars). "
        f"<b>Star SNR</b> is peak&nbsp;ADU&nbsp;/&nbsp;median&nbsp;background&nbsp;RMS for each "
        f"ePSF-fitted star; compare with the aggregate median&nbsp;&plusmn;&nbsp;IQR in Section&nbsp;3."
        f"</p>"
    )

    return img_html, caption_html


def _epsf_stars_cell(psf_metrics: dict) -> str:
    """Format the number of stars used to build the ePSF."""
    n = psf_metrics.get("epsf_n_stars")
    if n is None:
        return "—"
    # If newer photutils exposes convergence, flag non-convergence in red.
    conv = psf_metrics.get("epsf_converged")
    iters = psf_metrics.get("epsf_iterations")
    suffix = ""
    if conv is False:
        suffix = (f' &nbsp;<span style="color:#c0392b;font-weight:bold;">'
                  f'⚠ not converged ({iters} iters)</span>')
    elif iters is not None:
        suffix = f" ({iters} iters)"
    return f"{n}{suffix}"


def _error_box(metric_key: str, ra: AnalysisResult, rb: AnalysisResult) -> str:
    """Return an error box HTML if the metric failed, else empty string."""
    err = ra.errors.get(metric_key) or rb.errors.get(metric_key)
    if not err:
        return ""
    return f'<div class="error-box">⚠ <strong>Analysis failed:</strong> {err}</div>'


def _better_worse_class(val_a, val_b, higher_is_better: bool = True) -> tuple[str, str]:
    if val_a is None or val_b is None:
        return "", ""
    if higher_is_better:
        return ("better", "worse") if val_a >= val_b else ("worse", "better")
    return ("better", "worse") if val_a <= val_b else ("worse", "better")


def _focal_ratio(img: AstroImage) -> float | None:
    hdr = img.header
    if hdr is None:
        return None
    for kw in ("FOCRATIO", "FRATIO", "FNUMBER"):
        try:
            return float(hdr[kw])
        except (KeyError, ValueError, TypeError):
            pass
    try:
        return float(hdr["FOCALLEN"]) / float(hdr["APTDIA"])
    except (KeyError, ValueError, TypeError):
        return None


def _pixel_size_mm(img: AstroImage) -> float | None:
    hdr = img.header
    if hdr is None:
        return None
    try:
        return float(hdr["XPIXSZ"]) / 1000.0
    except (KeyError, ValueError, TypeError):
        return None


# ── Main class ────────────────────────────────────────────────────────────────

class ReportBuilder:
    """Generate a self-contained HTML comparison report."""

    def generate(self, image_a: AstroImage, image_b: AstroImage | None = None,
                  result_a: AnalysisResult | None = None, result_b: AnalysisResult | None = None,
                  output_dir: str | Path = ".",
                  open_browser: bool = True,
                  ref_seeing_arcsec: float = REF_SEEING_ARCSEC) -> Path:

        self._ref_seeing_arcsec = ref_seeing_arcsec
        self._single_image = image_b is None
        if result_a is None:
            result_a = AnalysisResult(label="Image A")
        if result_b is None:
            result_b = AnalysisResult(label="—")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._single_image:
            stem = f"report_{result_a.label}_{ts}".replace(" ", "_")
        else:
            stem = f"report_{result_a.label}_{result_b.label}_{ts}".replace(" ", "_")
        filename = stem + ".html"

        bw_a = image_a.bandwidth_nm
        bw_b = image_b.bandwidth_nm if image_b is not None else None
        bw_differ = (bw_a is not None and bw_b is not None and
                     abs(bw_a - bw_b) > 0.1)

        # Label substitution was applied in the analysis thread before any figures were
        # rendered.  Read the stored original labels from the result objects here so the
        # Section 1 info box can map "Image A/B" back to the full filenames.
        _substituted = (result_a.original_label is not None
                        or result_b.original_label is not None)
        _orig_label_a = result_a.original_label or result_a.label
        _orig_label_b = result_b.original_label or result_b.label

        single_image_banner = ""
        if self._single_image:
            single_image_banner = _info_box(
                'Image B was not loaded. '
                'Comparison colour-coded tables and differential metrics are not available.',
                title="Single Image Analysis",
                open=True,
                style="background:#2a3a2a;color:#c8e6c9;border-color:#4caf50",
            )

        sections = [
            single_image_banner,
            self._section_header(image_a, image_b, result_a, result_b, bw_differ,
                                 substituted=_substituted,
                                 orig_label_a=_orig_label_a,
                                 orig_label_b=_orig_label_b),
            self._section_observation(result_a, result_b),
            self._section_snr(result_a, result_b),
            self._section_psf(result_a, result_b, image_a, image_b),
            self._section_halo(result_a, result_b, image_a, image_b),
            self._section_edge(result_a, result_b, bw_differ),
            self._section_power(result_a, result_b),
            self._section_spatial(result_a, result_b),
            self._section_summary(result_a, result_b, bw_differ),
        ]

        title = (f"Filter Analysis: {result_a.label}" if self._single_image
                 else f"Filter Comparison: {result_a.label} vs {result_b.label}")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
{"".join(sections)}
<p style="color:#999;font-size:0.85em;margin-top:40px;">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by FilterImageCompare
</p>
</body>
</html>"""

        out_path = output_dir / filename
        out_path.write_text(html, encoding="utf-8")

        if open_browser:
            webbrowser.open(out_path.as_uri())

        inspector_path = output_dir / (stem + "_inspector.npz")
        try:
            self._write_inspector_file(
                inspector_path, image_a, image_b, result_a, result_b
            )
        except Exception:
            pass   # never block report delivery for inspector failures

        return out_path

    # ── Inspector file writer ─────────────────────────────────────────────────

    def _write_inspector_file(self,
                               path: Path,
                               image_a: AstroImage, image_b: AstroImage | None,
                               result_a: AnalysisResult,
                               result_b: AnalysisResult) -> None:
        """Write a companion .npz file for the Report Inspector."""
        import json as _json
        arrays: dict[str, np.ndarray] = {}
        catalog_sections: dict[str, list] = {}

        def _add(key: str, arr: np.ndarray) -> None:
            arrays[key] = arr

        def _add_options_entry(section: str, name: str, options: dict) -> None:
            valid = {k: v for k, v in options.items() if v in arrays}
            if valid:
                catalog_sections.setdefault(section, []).append(
                    {"name": name, "options": valid}
                )

        # ── Input images ──────────────────────────────────────────────────────
        _add("display_a", _inspector_display(image_a))
        orig_opts: dict[str, str] = {"Image A": "display_a"}
        if image_b is not None:
            _add("display_b", _inspector_display(image_b))
            orig_opts["Image B"] = "display_b"
        _add_options_entry("Input Images", "Original", orig_opts)

        sl_opts: dict[str, str] = {}
        if image_a.starless_image is not None:
            _add("display_sl_a", _inspector_display(image_a.starless_image))
            sl_opts["Image A"] = "display_sl_a"
        if image_b is not None and image_b.starless_image is not None:
            _add("display_sl_b", _inspector_display(image_b.starless_image))
            sl_opts["Image B"] = "display_sl_b"
        if sl_opts:
            _add_options_entry("Input Images", "Starless", sl_opts)

        # ── PSF / MTF ─────────────────────────────────────────────────────────
        pa = result_a.psf_metrics or {}
        pb = result_b.psf_metrics or {}
        if pa.get("epsf_data") is not None and pb.get("epsf_data") is not None:
            ea = pa["epsf_data"].astype(np.float64)
            eb = pb["epsf_data"].astype(np.float64)
            _add("epsf_a", np.log1p(ea - ea.min()).astype(np.float32))
            _add("epsf_b", np.log1p(eb - eb.min()).astype(np.float32))
            ref_fwhm = _ref_fwhm_px(pa, pb, self._ref_seeing_arcsec)
            if ref_fwhm is not None:
                # Match the measured ePSF size so cross-sections sample the same pixel grid
                epsf_size = int(ea.shape[0])
                kern_ref = _make_moffat_kernel(ref_fwhm, size=epsf_size)
                kr = np.log1p(kern_ref - kern_ref.min()).astype(np.float32)
                _add("epsf_ref", kr)
            epsf_opts: dict[str, str] = {"Image A": "epsf_a", "Image B": "epsf_b"}
            if "epsf_ref" in arrays:
                epsf_opts["Reference"] = "epsf_ref"
            _add_options_entry("PSF / MTF", "ePSF", epsf_opts)

        # ── PSF spatial maps (FWHM gradient, eccentricity) ────────────────────
        stars_a = pa.get("star_data", [])
        stars_b = pb.get("star_data", [])
        img_h_a, img_w_a = image_a.data.shape[:2]
        img_h_b, img_w_b = (image_b.data.shape[:2] if image_b is not None else (0, 0))

        fwhm_pts_a = [(s["x"], s["y"], s["fwhm"]) for s in stars_a if s.get("fwhm") is not None]
        fwhm_map_a = _psf_make_map(fwhm_pts_a, img_h_a, img_w_a)
        fwhm_map_b = None
        if image_b is not None:
            fwhm_pts_b = [(s["x"], s["y"], s["fwhm"]) for s in stars_b if s.get("fwhm") is not None]
            fwhm_map_b = _psf_make_map(fwhm_pts_b, img_h_b, img_w_b)
        fwhm_opts: dict[str, str] = {}
        if fwhm_map_a is not None:
            _add("psf_fwhm_map_a", fwhm_map_a)
            fwhm_opts["Image A"] = "psf_fwhm_map_a"
        if fwhm_map_b is not None:
            _add("psf_fwhm_map_b", fwhm_map_b)
            fwhm_opts["Image B"] = "psf_fwhm_map_b"
        if fwhm_opts:
            _add_options_entry("PSF / MTF", "FWHM gradient map", fwhm_opts)

        ecc_pts_a = [(s["x"], s["y"], s["eccentricity"]) for s in stars_a if s.get("eccentricity") is not None]
        ecc_map_a = _psf_make_map(ecc_pts_a, img_h_a, img_w_a)
        ecc_map_b = None
        if image_b is not None:
            ecc_pts_b = [(s["x"], s["y"], s["eccentricity"]) for s in stars_b if s.get("eccentricity") is not None]
            ecc_map_b = _psf_make_map(ecc_pts_b, img_h_b, img_w_b)
        ecc_opts: dict[str, str] = {}
        if ecc_map_a is not None:
            _add("psf_ecc_map_a", ecc_map_a)
            ecc_opts["Image A"] = "psf_ecc_map_a"
        if ecc_map_b is not None:
            _add("psf_ecc_map_b", ecc_map_b)
            ecc_opts["Image B"] = "psf_ecc_map_b"
        if ecc_opts:
            _add_options_entry("PSF / MTF", "Eccentricity map", ecc_opts)

        ab_a = pa.get("aberration", {})
        ab_b = pb.get("aberration", {})
        er_pts_a = list(zip(ab_a.get("star_xs", []), ab_a.get("star_ys", []), ab_a.get("star_er", [])))
        er_map_a = _psf_make_map(er_pts_a, img_h_a, img_w_a)
        er_map_b = None
        if image_b is not None:
            er_pts_b = list(zip(ab_b.get("star_xs", []), ab_b.get("star_ys", []), ab_b.get("star_er", [])))
            er_map_b = _psf_make_map(er_pts_b, img_h_b, img_w_b)
        er_opts: dict[str, str] = {}
        if er_map_a is not None:
            _add("psf_er_map_a", er_map_a)
            er_opts["Image A"] = "psf_er_map_a"
        if er_map_b is not None:
            _add("psf_er_map_b", er_map_b)
            er_opts["Image B"] = "psf_er_map_b"
        if er_opts:
            _add_options_entry("PSF / MTF", "Radial elongation map", er_opts)

        # ── PSF Simulation ────────────────────────────────────────────────────
        sim = self._plot_psf_simulation(result_a, result_b)
        sim_ref_label = ""
        if sim is not None:
            for skey in ("original", "conv_a", "conv_b", "conv_ref", "diff"):
                arr = sim.get(skey)
                if arr is not None:
                    _add(f"sim_{skey}", arr)
            sim_ref_label = sim.get("label_ref") or ""
            chart_opts: dict[str, str] = {}
            for lbl, k in [("Original", "sim_original"), ("Image A", "sim_conv_a"),
                            ("Image B", "sim_conv_b"), ("Reference", "sim_conv_ref")]:
                if k in arrays:
                    chart_opts[lbl] = k
            _add_options_entry("PSF Simulation", "Convolved test chart", chart_opts)
            _add_options_entry("PSF Simulation", "A−B difference",
                               {"Diff (A−B)": "sim_diff"})

        # ── SNR ───────────────────────────────────────────────────────────────
        sa = result_a.snr_metrics or {}
        sb = result_b.snr_metrics or {}
        if sa.get("snr_display") is not None and sb.get("snr_display") is not None:
            _add("snr_display_a", sa["snr_display"].astype(np.float32))
            _add("snr_display_b", sb["snr_display"].astype(np.float32))
            _add_options_entry("SNR", "SNR map",
                               {"Image A": "snr_display_a", "Image B": "snr_display_b"})
        sl_a = sa.get("starless") or {}
        sl_b = sb.get("starless") or {}
        if sl_a.get("snr_display") is not None and sl_b.get("snr_display") is not None:
            _add("snr_display_sl_a", sl_a["snr_display"].astype(np.float32))
            _add("snr_display_sl_b", sl_b["snr_display"].astype(np.float32))
            _add_options_entry("SNR", "Starless SNR map",
                               {"Image A": "snr_display_sl_a",
                                "Image B": "snr_display_sl_b"})

        # ── Edge Detection ────────────────────────────────────────────────────
        ea_m = result_a.edge_metrics or {}
        eb_m = result_b.edge_metrics or {}
        if ea_m.get("gm_display") is not None and eb_m.get("gm_display") is not None:
            _add("gm_display_a", ea_m["gm_display"].astype(np.float32))
            _add("gm_display_b", eb_m["gm_display"].astype(np.float32))
            _add_options_entry("Edge Detection", "Gradient map",
                               {"Image A": "gm_display_a", "Image B": "gm_display_b"})

        # ── Spatial Detail subsections ────────────────────────────────────────
        _PANEL_IMAGE_SETS = {
            "std_5px":    "Std Dev 5 px",
            "std_15px":   "Std Dev 15 px",
            "std_31px":   "Std Dev 31 px",
            "log_1.5":    "LoG σ 1.5 px",
            "log_3.0":    "LoG σ 3.0 px",
            "log_6.0":    "LoG σ 6.0 px",
            "wavelet_2":  "Wavelet level 2",
            "wavelet_3":  "Wavelet level 3",
        }
        panels_a = (result_a.spatial_metrics or {}).get("panels", {})
        panels_b = (result_b.spatial_metrics or {}).get("panels", {})
        for pkey, img_set_name in _PANEL_IMAGE_SETS.items():
            pa_panel = panels_a.get(pkey)
            if pa_panel is None:
                continue
            pb_panel = panels_b.get(pkey)
            sp_opts: dict[str, str] = {}
            npz_a = f"sp_{pkey}_a"
            _add(npz_a, pa_panel["a"])
            sp_opts["Image A"] = npz_a
            if pb_panel is not None and pb_panel.get("b") is not None:
                npz_b = f"sp_{pkey}_b"
                _add(npz_b, pb_panel["b"])
                sp_opts["Image B"] = npz_b
            if pa_panel.get("diff") is not None:
                npz_diff = f"sp_{pkey}_diff"
                _add(npz_diff, pa_panel["diff"])
                sp_opts["Diff (A−B)"] = npz_diff
            _add_options_entry("Spatial Detail", img_set_name, sp_opts)

        # ── Catalog JSON ──────────────────────────────────────────────────────
        catalog = {
            "label_a": result_a.label,
            "label_b": result_b.label,
            "filename_a": image_a.path.name,
            "filename_b": image_b.path.name if image_b is not None else "",
            "sim_label_ref": sim_ref_label,
            "single_image": image_b is None,
            "sections": catalog_sections,
        }
        json_bytes = _json.dumps(catalog, ensure_ascii=False).encode("utf-8")
        arrays["catalog_json"] = np.frombuffer(json_bytes, dtype=np.uint8).copy()

        np.savez_compressed(str(path), **arrays)

    # ── Section 1: Header ─────────────────────────────────────────────────────

    def _section_header(self, img_a: AstroImage, img_b: AstroImage | None,
                         result_a: AnalysisResult, result_b: AnalysisResult,
                         bw_differ: bool,
                         substituted: bool = False,
                         orig_label_a: str = "",
                         orig_label_b: str = "") -> str:
        bw_warn = ""
        if bw_differ and img_b is not None:
            bw_warn = (f'<div class="bw-warn">⚠ <strong>Bandwidth warning:</strong> '
                       f'Filters have different bandwidths '
                       f'({img_a.bandwidth_nm:.1f} nm vs {img_b.bandwidth_nm:.1f} nm). '
                       f'Metrics marked <span class="metric-label-warn">⚠</span> are '
                       f'sensitive to this difference and should be interpreted with caution. '
                       f'Metrics marked <span class="metric-label-ok">✓</span> are '
                       f'bandwidth-independent.</div>')
        label_sub_box = ""
        if substituted:
            label_sub_box = _info_box(
                f'One or more input filenames exceed {LABEL_MAX_LEN} characters and have been '
                f'abbreviated in all plots and legends throughout this report.<br>'
                f'&nbsp;&nbsp;<strong>Image A</strong> = {orig_label_a}<br>'
                + (f'&nbsp;&nbsp;<strong>Image B</strong> = {orig_label_b}' if img_b is not None else ''),
                title="Label substitution",
                open=True,
            )

        def meta_rows(img: AstroImage, result: AnalysisResult) -> str:
            rows = ""
            for key, val in img.meta.items():
                rows += f"<tr><td><strong>{key}</strong></td><td>{val}</td></tr>"
            if img.pixel_scale_is_estimated:
                rows += ("<tr><td><strong>Pixel scale</strong></td>"
                         f"<td>{img.pixel_scale:.3f} \"/px (estimated)</td></tr>")
            else:
                rows += ("<tr><td><strong>Pixel scale</strong></td>"
                         f"<td>{img.pixel_scale:.3f} \"/px</td></tr>")
            if img.bandwidth_nm:
                rows += ("<tr><td><strong>Bandwidth</strong></td>"
                         f"<td>{img.bandwidth_nm:.1f} nm</td></tr>")
            n_total = (result.psf_metrics or {}).get("n_stars_total")
            if n_total is not None:
                rows += (f"<tr><td><strong>Stars detected (raw)</strong></td>"
                         f"<td>{n_total}</td></tr>")
            sl = getattr(img, "starless_image", None)
            if sl is not None:
                rows += (f"<tr><td><strong>Starless</strong></td>"
                         f"<td>{sl.path.name}</td></tr>")
            return rows

        sl_a = getattr(img_a, "starless_image", None)
        sl_b = getattr(img_b, "starless_image", None) if img_b is not None else None

        thumb_a = _img_tag(self._thumbnail_fig(img_a), f"Preview {img_a.label}")
        thumb_sl_a = _img_tag(self._thumbnail_fig(sl_a, ref_data=img_a.data),
                               f"Starless {img_a.label}") if sl_a else ""
        sl_cap_a = '<p class="caption">Starless (STF-matched stretch)</p>' if sl_a else ""

        hist_tag = _img_tag(self._plot_image_histograms(img_a, img_b), "Pixel histograms")

        title_line = (f"<p><strong>{img_a.label}</strong> vs <strong>{img_b.label}</strong></p>"
                      if img_b is not None
                      else f"<p><strong>{img_a.label}</strong></p>")
        h1_text = ("Filter Image Comparison Report" if img_b is not None
                   else "Filter Image Analysis Report")

        if img_b is not None:
            thumb_b = _img_tag(self._thumbnail_fig(img_b), f"Preview {img_b.label}")
            thumb_sl_b = _img_tag(self._thumbnail_fig(sl_b, ref_data=img_b.data),
                                   f"Starless {img_b.label}") if sl_b else ""
            sl_cap_b = '<p class="caption">Starless (STF-matched stretch)</p>' if sl_b else ""
            b_col = f"""
  <div style="flex:1;">
    <h3>{img_b.label}</h3>
    {thumb_b}
    {thumb_sl_b}{sl_cap_b}
    <table><tbody>{meta_rows(img_b, result_b)}</tbody></table>
  </div>"""
        else:
            b_col = ""

        return f"""
<h1>{h1_text}</h1>
{title_line}
{label_sub_box}
{bw_warn}
<h2>1. Image Metadata</h2>
<div style="display:flex;gap:20px;">
  <div style="flex:1;">
    <h3>{img_a.label}</h3>
    {thumb_a}
    {thumb_sl_a}{sl_cap_a}
    <table><tbody>{meta_rows(img_a, result_a)}</tbody></table>
  </div>{b_col}
</div>
<h3>Pixel Histograms</h3>
{hist_tag}
<p class="caption">Log-scale pixel value distributions. Dotted vertical lines mark the median of each image. Dashed lower-opacity lines show the starless version where available.</p>"""

    def _plot_image_histograms(self, img_a: AstroImage, img_b: AstroImage | None) -> plt.Figure | None:
        """Combined log-scale histogram of both images with median markers."""
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = {"a": "steelblue", "b": "tomato"}

            img_list = [(img_a, "a", img_a.label)]
            if img_b is not None:
                img_list.append((img_b, "b", img_b.label))
            for img, key, label in img_list:
                color = colors[key]

                pixels = img.data.ravel().astype(float)
                positive = pixels[pixels > 0]
                if len(positive) == 0:
                    continue
                lo, hi = np.percentile(positive, [0.01, 99.99])
                if lo <= 0:
                    lo = positive.min()
                if hi <= lo:
                    hi = lo * 10
                bins = np.geomspace(lo, hi, 256)
                counts, edges = np.histogram(positive, bins=bins)
                centers = np.sqrt(edges[:-1] * edges[1:])
                ax.step(centers, counts, where="mid", color=color,
                        alpha=0.90, linewidth=1.5, label=label)
                ax.axvline(float(np.median(positive)), color=color, linestyle="-", linewidth=1.5, alpha=0.75)

                sl = getattr(img, "starless_image", None)
                if sl is not None and getattr(sl, "data", None) is not None:
                    sl_pix = sl.data.ravel().astype(float)
                    sl_pos = sl_pix[sl_pix > 0]
                    if len(sl_pos) > 0:
                        lo2, hi2 = np.percentile(sl_pos, [0.01, 99.99])
                        if lo2 <= 0:
                            lo2 = sl_pos.min()
                        if hi2 <= lo2:
                            hi2 = lo2 * 10
                        bins2 = np.geomspace(lo2, hi2, 256)
                        counts2, edges2 = np.histogram(sl_pos, bins=bins2)
                        centers2 = np.sqrt(edges2[:-1] * edges2[1:])
                        ax.step(centers2, counts2, where="mid", color=color,
                                alpha=0.35, linewidth=1.0, linestyle="--",
                                label=f"{label} (starless)")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Count")
            ax.set_title("Pixel Intensity histogram")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25, which="both")
            fig.tight_layout()
            return fig
        except Exception:
            return None

    def _thumbnail_fig(self, img: AstroImage,
                        ref_data: np.ndarray | None = None) -> plt.Figure | None:
        """Return a small matplotlib figure with a stretched preview of the image.

        If ref_data is provided, apply STF stretch parameters derived from ref_data
        to img (used for starless thumbnails so the tonal curve matches the star image).
        """
        if img is None or img.data is None:
            return None
        try:
            if ref_data is not None:
                from core.stretch import stf_stretch_matched
                arr_f = stf_stretch_matched(img.data, ref_data)
                max_dim = max(arr_f.shape[:2])
                if max_dim > 400:
                    step = max_dim // 400 + 1
                    arr_f = arr_f[::step, ::step]
                arr = (arr_f * 255).astype(np.uint8)
            else:
                arr = img.display_image(stretch=True)
                max_dim = max(arr.shape[:2])
                if max_dim > 400:
                    step = max_dim // 400 + 1
                    arr = arr[::step, ::step]
            fig, ax = plt.subplots(figsize=(4, 4 * arr.shape[0] / arr.shape[1]))
            ax.imshow(arr, origin="upper", cmap="gray", interpolation="bilinear",
                      aspect="auto")
            ax.axis("off")
            fig.tight_layout(pad=0)
            return fig
        except Exception:
            return None

    # ── Section 2: Observation context ────────────────────────────────────────

    def _section_observation(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        psf_a = ra.psf_metrics or {}
        psf_b = rb.psf_metrics or {}
        seeing_warn = ""
        if psf_a.get("seeing_dominated") or psf_b.get("seeing_dominated"):
            seeing_warn = (
                '<div class="warn-box">⚠ <strong>Seeing warning:</strong> '
                'FWHM exceeds 3″ in one or both images. PSF and MTF differences '
                'between filters may reflect atmospheric seeing variation rather than '
                'filter optical quality. For the most valid comparisons, images should '
                'be taken on the same night under similar conditions.</div>'
            )

        all_warnings = list(set(ra.warnings + rb.warnings))
        warn_html = ""
        if all_warnings:
            items = "".join(f"<li>{w}</li>" for w in all_warnings)
            warn_html = f'<div class="warn-box"><ul>{items}</ul></div>'

        return f"""
<h2>2. Observation Context</h2>
{seeing_warn}
{warn_html}
{_info_box('PSF/MTF comparisons are most meaningful when both images were '
           'captured on the same night under similar atmospheric conditions. DATE-OBS values are '
           'shown in the metadata table above.',
           title="Comparison conditions")}"""

    # ── Section 3: PSF / MTF ──────────────────────────────────────────────────

    def _section_psf(self, ra: AnalysisResult, rb: AnalysisResult,
                      img_a: AstroImage, img_b: AstroImage | None) -> str:
        err = _error_box("psf", ra, rb)
        pa = ra.psf_metrics or {}
        pb = rb.psf_metrics or {}
        ca, cb = _better_worse_class(pa.get("fwhm_px"), pb.get("fwhm_px"), higher_is_better=False)
        ma, mb = _better_worse_class(pa.get("mtf50_cycles_per_px"), pb.get("mtf50_cycles_per_px"))

        fig_mtf = None
        freq_a = pa.get("mtf_freq")
        mtf_a  = pa.get("mtf_curve")
        freq_b = pb.get("mtf_freq")
        mtf_b  = pb.get("mtf_curve")

        # Reference PSF: Moffat with Kolmogorov β at 2.0" seeing
        ref_fwhm = _ref_fwhm_px(pa, pb, self._ref_seeing_arcsec)
        if ref_fwhm is not None:
            kern_ref = _make_moffat_kernel(ref_fwhm)
            freq_ref, mtf_ref = self._compute_mtf_from_kernel(kern_ref)
            ref_label = (f"Reference ({self._ref_seeing_arcsec:.1f}″ seeing, "
                         f"β = {REF_SEEING_BETA})")
            img_epsf_ref = _img_tag(self._plot_epsf(kern_ref, ref_label), ref_label)
        else:
            freq_ref = mtf_ref = ref_label = None
            img_epsf_ref = ""

        if freq_a is not None or freq_b is not None:
            fig_mtf = self._overlay_mtf(freq_a, mtf_a, freq_b, mtf_b,
                                        ra.label, rb.label,
                                        freq_ref, mtf_ref, ref_label)

        img_mtf = _img_tag(fig_mtf, "MTF comparison")
        img_epsf_a = _img_tag((pa.get("figures") or {}).get("epsf"), f"ePSF {ra.label}")
        img_epsf_b = _img_tag((pb.get("figures") or {}).get("epsf"), f"ePSF {rb.label}")
        img_scatter = _img_tag(self._plot_fwhm_scatter(ra, rb), "FWHM scatter")

        # Spatial maps and histograms
        img_h_a, img_w_a = img_a.data.shape[:2]
        img_h_b, img_w_b = (img_b.data.shape[:2] if img_b is not None else (0, 0))
        stars_a = pa.get("star_data", [])
        stars_b = pb.get("star_data", [])
        fwhm_vals_a = [s["fwhm"] for s in stars_a if s.get("fwhm") is not None]
        fwhm_vals_b = [s["fwhm"] for s in stars_b if s.get("fwhm") is not None]
        ecc_vals_a  = [s["eccentricity"] for s in stars_a if s.get("eccentricity") is not None]
        ecc_vals_b  = [s["eccentricity"] for s in stars_b if s.get("eccentricity") is not None]

        img_fwhm_map  = _img_tag(self._plot_psf_spatial_map(
            stars_a, stars_b, "fwhm", ra.label, rb.label,
            img_h_a, img_w_a, img_h_b, img_w_b,
            "FWHM spatial map (px)", "viridis"), "FWHM spatial map")
        img_fwhm_hist = _img_tag(self._plot_psf_histogram(
            fwhm_vals_a, fwhm_vals_b, ra.label, rb.label,
            "FWHM (px)", "FWHM distribution"), "FWHM histogram")
        img_ecc_map   = _img_tag(self._plot_psf_spatial_map(
            stars_a, stars_b, "eccentricity", ra.label, rb.label,
            img_h_a, img_w_a, img_h_b, img_w_b,
            "Eccentricity spatial map", "plasma"), "Eccentricity spatial map")
        img_ecc_hist  = _img_tag(self._plot_psf_histogram(
            ecc_vals_a, ecc_vals_b, ra.label, rb.label,
            "Eccentricity", "Eccentricity distribution"), "Eccentricity histogram")

        dist_fig, dist_caption = _psf_distributions_figure(stars_a, stars_b, ra.label, rb.label)
        dist_html = (
            "<h4>Per-star metric distributions</h4>" + dist_fig + dist_caption
            if dist_fig else ""
        )

        # Statistical difference column (Mann-Whitney U + Cliff's delta per metric)
        def _sig(key):
            va = [s[key] for s in stars_a if s.get(key) is not None]
            vb = [s[key] for s in stars_b if s.get(key) is not None]
            return _psf_stat_test(va, vb)   # (html, p | None)

        def _sig_td(html, p):
            if p is None:
                return f"<td>{html}</td>"
            style = 'style="background:#b3e5fc"' if p < 0.05 else 'style="background:#e0e0e0"'
            return f"<td {style}>{html}</td>"

        (sig_fwhm_px,     p_fwhm_px)     = _sig("fwhm")
        (sig_fwhm_arcsec, p_fwhm_arcsec) = _sig("fwhm_arcsec")
        (sig_beta,        p_beta)         = _sig("beta")
        (sig_ell,         p_ell)          = _sig("ellipticity")
        (sig_ecc,         p_ecc)          = _sig("eccentricity")

        _ref_seeing_note = (
            f'<strong>Reference curve:</strong> the green dashed line is a <em>synthetic</em> Moffat '
            f'profile at {self._ref_seeing_arcsec:.1f}&Prime; seeing (&beta;&thinsp;=&thinsp;{REF_SEEING_BETA}, '
            f'Kolmogorov turbulence). It is a benchmark for typical good atmospheric conditions, not a '
            f'theoretical maximum. FWHM converts to pixels via the plate scale from the measured ePSFs; '
            f'if the plate scale is unknown the reference is omitted.<br><br>'
        ) if img_epsf_ref else ""

        _compare_box = _info_box(
            'A smaller FWHM (arcsec) and higher MTF50 indicate sharper resolution &mdash; '
            'these are the primary quality indicators for filter comparison. A higher '
            'Moffat &beta; indicates less scattered light in the wings. Ellipticity should be '
            'similar between filters; a large difference suggests filter tilt, substrate '
            'wedge, or different seeing conditions between sessions. If the ePSFs show '
            'the same asymmetric tail in both images, the cause is common to both (optics '
            'or tracking) and does not reflect a filter quality difference &mdash; what matters '
            'for comparison is whether the tail is <em>more pronounced</em> in one image.',
            title="How to compare these images", open=True,
        )

        return f"""
<h2>4. PSF / MTF &nbsp;<span class="metric-label-ok">✓ bandwidth-independent</span></h2>
{err}
{_compare_box}
{_info_box('<strong>Star quality pipeline &mdash; how the table counts are derived</strong><br><br>'
           'Stars pass through a two-stage quality pipeline before any metric or ePSF is computed. '
           'All five metrics (FWHM, &beta;, ellipticity, eccentricity) and the ePSF are computed from '
           'the same final clean set, so the statistics are self-consistent.<br><br>'
           '<b>Stage 1 &mdash; spatial and photometric filters (Candidate PSF stars):</b> '
           'DAOStarFinder detects all sources above a 5&sigma; threshold. Candidates are then kept only '
           'if they are: (a) unsaturated (peak &lt; 90&nbsp;% of the data range); (b) high signal-to-noise '
           '(peak&nbsp;/&nbsp;background&nbsp;RMS &ge; 30); (c) at least 50&nbsp;px from any image border '
           '(to avoid edge-truncated PSFs); and (d) isolated (no neighbour within 5&times;FWHM, which would '
           'contaminate the Moffat fit or the ePSF cutout).<br><br>'
           '<b>Stage 2 &mdash; Moffat fit quality gates (Outliers rejected):</b> '
           'A Moffat&nbsp;2D profile is fitted to each candidate. Stars are excluded if: '
           'the fit fails to converge; the fitted &gamma; &lt; 0.1 (unphysically narrow); '
           'the fitted &beta; falls outside [1.0,&nbsp;10.0] (unphysical wing-falloff exponent &mdash; '
           'pure Kolmogorov turbulence predicts &beta;&nbsp;&asymp;&nbsp;4.77, so values below 1 or above 10 indicate '
           'a failed or pathological fit); or the resulting FWHM deviates by more than 3&times;MAD '
           'from the median FWHM of the full candidate set (sigma-clipping to remove occasional '
           'saturated, trailed, or double stars that survived stage 1).<br><br>'
           'A large ratio of detected to used stars is normal; deep narrowband frames with rich nebulosity '
           'can contain thousands of faint and crowded sources that are correctly excluded.',
           title="Star quality pipeline")}

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th><th>Difference</th></tr>
  <tr><td>Stars detected (raw)</td><td>{_val(pa.get("n_stars_total"), "d")}</td><td>{_val(pb.get("n_stars_total"), "d")}</td><td></td></tr>
  <tr><td>Candidate PSF stars</td><td>{_val(pa.get("n_psf_candidates"), "d")}</td><td>{_val(pb.get("n_psf_candidates"), "d")}</td><td></td></tr>
  <tr><td>Outliers rejected</td><td>{_val(pa.get("n_outliers_rejected"), "d")}</td><td>{_val(pb.get("n_outliers_rejected"), "d")}</td><td></td></tr>
  <tr><td>Stars used for PSF</td><td>{_val(pa.get("n_stars_used"), "d")}</td><td>{_val(pb.get("n_stars_used"), "d")}</td><td></td></tr>
  <tr><td>FWHM (px) <small style="color:#555">↓ smaller = sharper</small></td><td class="{ca}">{_val_pm(pa.get("fwhm_px"), pa.get("fwhm_px_mad"))}</td><td class="{cb}">{_val_pm(pb.get("fwhm_px"), pb.get("fwhm_px_mad"))}</td>{_sig_td(sig_fwhm_px, p_fwhm_px)}</tr>
  <tr><td>FWHM (arcsec) <small style="color:#555">↓ smaller = sharper</small></td><td class="{ca}">{_val_pm(pa.get("fwhm_arcsec"), pa.get("fwhm_arcsec_mad"))}</td><td class="{cb}">{_val_pm(pb.get("fwhm_arcsec"), pb.get("fwhm_arcsec_mad"))}</td>{_sig_td(sig_fwhm_arcsec, p_fwhm_arcsec)}</tr>
  <tr><td>Moffat &beta; <small style="color:#555">↑ higher = tighter wings</small></td><td>{_val_pm(pa.get("beta"), pa.get("beta_mad"))}</td><td>{_val_pm(pb.get("beta"), pb.get("beta_mad"))}</td>{_sig_td(sig_beta, p_beta)}</tr>
  <tr><td>Ellipticity <small style="color:#555">↓ lower = rounder stars</small></td><td>{_val_pm(pa.get("ellipticity"), pa.get("ellipticity_mad"))}</td><td>{_val_pm(pb.get("ellipticity"), pb.get("ellipticity_mad"))}</td>{_sig_td(sig_ell, p_ell)}</tr>
  <tr><td>Eccentricity <small style="color:#555">↓ lower = rounder stars</small></td><td>{_val_pm(pa.get("eccentricity"), pa.get("eccentricity_mad"))}</td><td>{_val_pm(pb.get("eccentricity"), pb.get("eccentricity_mad"))}</td>{_sig_td(sig_ecc, p_ecc)}</tr>
  <tr><td>MTF50 (cyc/px) <small style="color:#555">↑ higher = sharper</small></td><td class="{ma}">{_val(pa.get("mtf50_cycles_per_px"), ".4f")}</td><td class="{mb}">{_val(pb.get("mtf50_cycles_per_px"), ".4f")}</td><td></td></tr>
  <tr><td>MTF @ Nyquist <small style="color:#555">ideal ≈ 0</small></td><td>{_val(pa.get("mtf_nyquist"), ".4f")}</td><td>{_val(pb.get("mtf_nyquist"), ".4f")}</td><td></td></tr>
  <tr><td>Stars used in ePSF</td><td>{_epsf_stars_cell(pa)}</td><td>{_epsf_stars_cell(pb)}</td><td></td></tr>
</table>
<p class="footnote">&#9733;&nbsp;small (|&delta;|&ge;0.147),&nbsp;&nbsp;&#9733;&#9733;&nbsp;medium (|&delta;|&ge;0.33),&nbsp;&nbsp;&#9733;&#9733;&#9733;&nbsp;large (|&delta;|&ge;0.474);&nbsp;&nbsp;n.s.&nbsp;= p&ge;0.05;&nbsp;&nbsp;&delta;&gt;0 means {ra.label} values tend higher.</p>

{dist_html}

{_info_box(
  '<strong>FWHM (Full Width at Half Maximum)</strong> &mdash; The diameter of a star '
  'image at half its peak brightness, derived from a Moffat&nbsp;2D fit. Smaller = sharper. '
  'Ground-based imaging is typically seeing-limited (1&ndash;3 arcsec); the best sites '
  'achieve sub-arcsecond FWHM. For filter comparison the arcsec value is the primary metric '
  '(scale-independent). A larger FWHM in one image may indicate worse seeing during that '
  'session, or additional softening introduced by the filter (e.g. substrate wedge or '
  'coating scatter). The reported value is the <em>median</em> of the clean per-star FWHM '
  'distribution; the &plusmn; figure is the Median Absolute Deviation (MAD), a robust spread '
  'measure that, like the median, is insensitive to the outlier stars removed by the quality '
  'pipeline. Stars whose FWHM deviates by more than 3&times;MAD from the median are excluded '
  'before the aggregate is computed.<br><br>'
  '<strong>Moffat &beta; (beta)</strong> &mdash; The wing-falloff exponent of the Moffat '
  'profile I(r)&nbsp;=&nbsp;A&nbsp;&times;&nbsp;(1&nbsp;+&nbsp;(r/&gamma;)&sup2;)<sup>&minus;&beta;</sup> '
  'fitted to each star. Higher &beta; means the stellar wings fall off more steeply, '
  'leaving less scattered light outside the core. Pure Kolmogorov atmospheric turbulence '
  'predicts &beta;&nbsp;&asymp;&nbsp;4.77; in practice values of 2&ndash;6 are typical. '
  '<strong>Ideal: &beta;&nbsp;&gt;&nbsp;3.</strong> '
  'Low &beta; (1&ndash;2) indicates extended wings from vibration, wind shake, or poor tracking; '
  'very high &beta; (&gt;&nbsp;6) suggests an unusually compact PSF or thin atmosphere. '
  'A consistently lower &beta; for one filter implies it scatters more light into the '
  'halo/wing region &mdash; compare with the Halo Analysis section. '
  'Only fits with &beta; in [1.0,&nbsp;10.0] are accepted; values outside this range are '
  'considered pathological fits and are excluded from the statistics and ePSF.<br><br>'
  '<strong>Ellipticity</strong> &mdash; How non-circular the average star shape is, '
  'measured from second-order image moments (0 = perfectly round, 1 = infinitely '
  'elongated). <strong>Ideal: &lt; 0.05.</strong> Values of 0.05&ndash;0.10 are '
  'borderline; &gt; 0.10 indicates a significant elongation that may reduce '
  'effective resolution in one axis. Common causes: tracking drift, autoguider '
  'lag, astigmatism, or filter substrate wedge. A large difference in ellipticity '
  'between the two filters is a specific indicator of filter tilt or wedge.<br><br>'
  '<strong>Eccentricity</strong> &mdash; A complementary measure of star elongation '
  'derived from the ratio of semi-minor to semi-major axis: e = &radic;(1 &minus; (b/a)&sup2;). '
  '<strong>Ideal: &lt; 0.10.</strong> Unlike ellipticity, eccentricity weights '
  'extreme elongation more strongly.<br><br>'
  '<strong>MTF50 (cycles/pixel)</strong> &mdash; The spatial frequency at which the '
  'Modulation Transfer Function falls to 50% of its peak. Higher MTF50 = the '
  'system preserves contrast at finer scales. The maximum physically possible '
  'value is 0.5 cyc/px (Nyquist limit for fully-sampled images). '
  '<strong>Ideal: as high as possible; typical ground-based: 0.1&ndash;0.3 cyc/px.</strong> '
  'MTF50 is the single most useful number for ranking overall sharpness.<br><br>'
  '<strong>MTF @ Nyquist</strong> &mdash; The residual MTF at exactly 0.5 cyc/px. '
  'For a well-sampled, diffraction-limited system this should approach 0. '
  '<strong>Ideal: close to 0.</strong> A notably non-zero value at Nyquist can '
  'indicate undersampling (FWHM &lt; ~2 px) or aliasing from a very sharp PSF.',
  title="Understanding the PSF metrics")}

{img_fwhm_map}
<p class="caption">Smoothed FWHM map (px) across the field. Shared colour scale between both images. Dots mark individual star measurements.</p>
{img_fwhm_hist}
<p class="caption">Distribution of per-star FWHM values.</p>

{img_ecc_map}
<p class="caption">Smoothed eccentricity map across the field. 0 = circular star, 1 = fully elongated.</p>
{img_ecc_hist}
<p class="caption">Distribution of per-star eccentricity values.</p>

{img_scatter}
<p class="caption">Per-star FWHM correlation. Points near the slope = 1 line indicate
consistent star size between filters. Systematic offset reveals which filter produces
tighter stars. Points far from the line indicate individual star measurement scatter.</p>

{_info_box(
  'The ePSF is constructed from all stars that passed the quality filters '
  '(unsaturated, adequate SNR, isolated from neighbours). Cutouts of each star '
  'are extracted from the background-subtracted image with a box size of '
  'max(25 px, 6 &times; FWHM) to capture the full wing extent. '
  'The <a href="https://photutils.readthedocs.io/en/stable/api/photutils.psf.EPSFBuilder.html" '
  'target="_blank"><em>photutils</em> <code>EPSFBuilder</code></a> '
  '(<a href="https://photutils.readthedocs.io/en/stable/user_guide/epsf_building.html" '
  'target="_blank">building guide</a>) iteratively stacks these cutouts at '
  '<strong>2&times; oversampling</strong>, aligning each star to its fitted '
  'sub-pixel centroid position to fill in the finer spatial grid. At each iteration '
  'the ePSF model is updated and each star is re-centred; the process repeats until '
  'the maximum centroid shift across all stars falls below '
  '<strong>0.001 px</strong> (the <code>center_accuracy</code> convergence '
  'criterion), or until the hard limit of <strong>15 iterations</strong> is '
  'reached. The number of stars actually used in the fit is shown in the table above. '
  'The result is displayed on a logarithmic scale to reveal structure spanning '
  'several orders of magnitude in brightness. '
  '<em>Note:</em> the current version of photutils does not expose the actual '
  'iteration count or a per-star residual metric through its public API; these '
  'fields will be populated automatically if a future photutils release returns '
  'them.<br><br>'
  '<strong>What to look for:</strong>'
  '<ul style="margin:0.4em 0 0 1.2em;padding:0;">'
  '<li><strong>Circular, compact core</strong> &mdash; ideal outcome: good focus, '
  'stable atmosphere, no significant aberrations.</li>'
  '<li><strong>Elliptical core</strong> &mdash; the elongation direction indicates '
  'the dominant cause: tracking drift (RA or Dec axis), astigmatism '
  '(diagonal elongation), or field rotation (curved smear on Alt-Az mounts). '
  'The eccentricity and position angle metrics in the table quantify this.</li>'
  '<li><strong>Asymmetric tails extending to one side</strong> &mdash; most commonly '
  'tracking or guiding drift in one axis, coma from the optical system '
  '(particularly if stars across the whole field share the same tail direction), '
  'or wind-induced mount vibration. If both filters show the same tail '
  'direction and magnitude, the cause is common to both capture sessions '
  '(optical or tracking), not filter-specific.</li>'
  '<li><strong>Extended, diffuse wings</strong> &mdash; poor seeing, thermal '
  'currents in the optical path, or vibration broadening the PSF without '
  'a directional bias.</li>'
  '<li><strong>Steep, clean falloff</strong> &mdash; the flux drops 2&ndash;3 orders of '
  'magnitude within a few FWHM. This is ideal: most of the star&rsquo;s light '
  'is in the core, minimising contamination of adjacent nebula structure. '
  'A steeper falloff (higher Moffat &beta;) is always better for contrast on '
  'fine detail next to bright stars.</li>'
  '<li><strong>Airy-ring structure</strong> &mdash; concentric rings around the '
  'core indicate near-diffraction-limited performance (exceptional seeing '
  'and optics, rarely seen in long-exposure deep-sky imaging).</li>'
  '</ul>',
  title="How the Empirical PSF (ePSF) was built")}

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{img_epsf_a}</div>
  {f'<div style="flex:1;">{img_epsf_b}</div>' if img_b is not None else ''}
  {f'<div style="flex:1;">{img_epsf_ref}</div>' if img_epsf_ref else ''}
</div>
<p class="caption">Empirical PSFs (log&#x2081;&#x208a; scale, viridis colormap). The ePSF is
built at 2&times; oversampling from all quality-filtered stars in the field. A circular,
compact core with rapid falloff is ideal. Asymmetric tails indicate tracking,
guiding, or optical aberrations &mdash; compare tail direction and magnitude between the
two images to distinguish session-specific from system-wide causes.
{f"The third panel shows the synthetic reference PSF ({self._ref_seeing_arcsec:.1f}&Prime; seeing, Moffat &beta;&thinsp;=&thinsp;{REF_SEEING_BETA}) for comparison." if img_epsf_ref else ""}</p>

{img_mtf}
<p class="caption">MTF curves for both filters overlaid, derived from the ePSF shown above.
Higher curve = better contrast preservation at fine scales.</p>
{_info_box(
    _ref_seeing_note +
    '<strong>From ePSF to MTF:</strong> '
    'The ePSF is normalised to unit sum, then a 2-D FFT produces the Optical Transfer Function (OTF). '
    'The MTF is the magnitude of the OTF, normalised to 1.0 at zero frequency. Because the ePSF is '
    'built at 2&times; oversampling, the frequency axis is scaled so the MTF is expressed in '
    '<strong>cycles per native image pixel</strong> with Nyquist at 0.5 cyc/px. '
    'The 2-D MTF is then radially averaged into the 1-D curve shown; if the ePSF is elliptical the '
    'curve represents the geometric mean of the two axes.<br><br>'
    '<strong>Interpreting the curve:</strong> '
    'An ideal MTF starts at 1.0 and falls monotonically to 0 at the Nyquist limit. '
    'MTF50 (the frequency where contrast drops to 50%) is the single most useful sharpness number &mdash; '
    'higher is sharper. If one filter&rsquo;s curve lies consistently above the other it delivers better '
    'contrast at all spatial scales. If the curves cross, one filter is sharper at fine detail while '
    'the other preserves broader structures better. '
    '<strong>Common causes of a lower MTF curve:</strong> poor seeing, focus offset, filter tilt, or '
    'optical aberrations in the filter glass.',
    title="How the MTF is derived")}

{self._psf_simulation_html(ra, rb)}""" + self._section_psf_aberration(ra, rb, img_a, img_b)

    def _section_psf_aberration(self, ra: AnalysisResult, rb: AnalysisResult,
                                  img_a: AstroImage, img_b: AstroImage | None) -> str:
        """Aberration analysis subsection appended to the PSF section."""
        pa = ra.psf_metrics or {}
        pb = rb.psf_metrics or {}
        ab_a = pa.get("aberration", {})
        ab_b = pb.get("aberration", {})

        if not ab_a and not ab_b:
            return ""

        warn_a = ab_a.get("warning", "")
        warn_b = ab_b.get("warning", "")
        n_a = ab_a.get("n_stars_used", 0)
        n_b = ab_b.get("n_stars_used", 0)

        def _interpret(ab: dict, label: str) -> str:
            ci = ab.get("coma_index")
            rf = ab.get("radial_frac")
            cs = ab.get("collimation_circstd_deg")
            cc = ab.get("corner_centre_ratio")
            if ci is None:
                return ""
            findings = []
            if ci is not None and rf is not None and ci > 0.5 and rf > 0.5:
                findings.append(
                    "pattern consistent with <strong>off-axis coma</strong> "
                    "(elongation grows with field radius and points radially outward)")
            if cs is not None and rf is not None and cs < 20.0 and rf > 0.4:
                findings.append(
                    "pattern consistent with <strong>collimation error / tilt coma</strong> "
                    "(stars elongated in a near-uniform direction across the field)")
            if cc is not None and cc > 1.3 and (ci is None or ci < 0.3):
                findings.append(
                    "pattern consistent with <strong>field curvature or defocus</strong> "
                    "(FWHM grows toward corners without radial elongation direction)")
            if not findings:
                return f"<p><strong>{label}:</strong> No dominant aberration signature detected.</p>\n"
            bullets = "".join(f"<li>{f}</li>" for f in findings)
            return (f"<p><strong>{label}:</strong></p>"
                    f"<ul style='margin:0.3em 0 0.8em 1.2em;padding:0;'>{bullets}</ul>\n")

        interp_a = _interpret(ab_a, ra.label)
        interp_b = _interpret(ab_b, rb.label)

        html = "\n<h3>Field Aberration Analysis</h3>\n"

        if warn_a and warn_b:
            return html + _info_box(
                f'{warn_a}<br>{warn_b}',
                title="Insufficient star coverage for aberration analysis",
                open=True,
            )

        if interp_a or interp_b:
            html += "<h4>Interpretation</h4>\n" + interp_a + interp_b

        def _score_td(val, lo, hi, higher_is_bad: bool = True, fmt: str = ".3f") -> str:
            if val is None:
                return "<td>—</td>"
            if higher_is_bad:
                bg = ("#c8e6c9" if val < lo else "#fff9c4" if val < hi else "#ffcdd2")
            else:
                bg = ("#c8e6c9" if val > hi else "#fff9c4" if val > lo else "#ffcdd2")
            return f'<td style="background:{bg}">{val:{fmt}}</td>'

        ci_a  = ab_a.get("coma_index")
        ci_b  = ab_b.get("coma_index")
        rf_a  = ab_a.get("radial_frac")
        rf_b  = ab_b.get("radial_frac")
        cs_a  = ab_a.get("collimation_circstd_deg")
        cs_b  = ab_b.get("collimation_circstd_deg")
        cc_a  = ab_a.get("corner_centre_ratio")
        cc_b  = ab_b.get("corner_centre_ratio")
        qa_a  = ab_a.get("fwhm_radial_quadratic")   # quadratic coefficient a
        qa_b  = ab_b.get("fwhm_radial_quadratic")
        li_a  = ab_a.get("fwhm_radial_linear")      # linear coefficient b
        li_b  = ab_b.get("fwhm_radial_linear")

        na_lbl = f"{ra.label} ({n_a} stars)"
        nb_lbl = f"{rb.label} ({n_b} stars)"
        html += f"""<table>
  <tr><th>Metric</th><th>{na_lbl}</th><th>{nb_lbl}</th></tr>
  <tr><td>Coma index (Pearson r, |e<sub>r</sub>| vs radius)</td>{_score_td(ci_a,0.3,0.6)}{_score_td(ci_b,0.3,0.6)}</tr>
  <tr><td>Radial elongation fraction</td>{_score_td(rf_a,0.3,0.5)}{_score_td(rf_b,0.3,0.5)}</tr>
  <tr><td>Orientation circular std (°)</td>{_score_td(cs_a,20.0,45.0,higher_is_bad=False,fmt=".1f")}{_score_td(cs_b,20.0,45.0,higher_is_bad=False,fmt=".1f")}</tr>
  <tr><td>Corner/centre FWHM ratio</td>{_score_td(cc_a,1.2,1.5,fmt=".3f")}{_score_td(cc_b,1.2,1.5,fmt=".3f")}</tr>
  <tr><td>FWHM gradient — quadratic (a)</td>{_score_td(qa_a,0.5,1.5,fmt=".3f")}{_score_td(qa_b,0.5,1.5,fmt=".3f")}</tr>
  <tr><td>FWHM gradient — linear (b)</td><td>{_val(li_a,".3f")}</td><td>{_val(li_b,".3f")}</td></tr>
</table>
<p class="footnote">Colour coding — green: no significant signal; yellow: mild; red: significant. Thresholds are heuristic. Lower orientation circular std = more uniform elongation direction. All findings should be read as &ldquo;patterns consistent with&rdquo;, not definitive diagnoses.</p>\n"""

        if warn_a:
            html += f'<p class="footnote"><em>{ra.label}: {warn_a}</em></p>\n'
        if warn_b:
            html += f'<p class="footnote"><em>{rb.label}: {warn_b}</em></p>\n'

        html += _info_box(
            '<strong>Coma index (Pearson&nbsp;r, |e<sub>r</sub>|&nbsp;vs&nbsp;radius)</strong> &mdash; '
            'Pearson correlation between each star&rsquo;s absolute radial elongation and its distance '
            'from the field centre. Range: &minus;1 to +1. Values near&nbsp;0 indicate no radial pattern; '
            'values&nbsp;&ge;&nbsp;0.5 suggest elongation growing systematically with radius, consistent '
            'with off-axis coma. Negative values (rare) indicate tangential elongation dominates at '
            'larger radii.<br><br>'
            '<strong>Radial elongation fraction</strong> &mdash; '
            'For all stars with eccentricity&nbsp;&gt;&nbsp;0.05, the mean fraction of their elongation '
            'pointing radially (away from or toward the field centre). Range:&nbsp;0 to&nbsp;1. Values '
            'near&nbsp;1 mean nearly all elongation is radial (coma or collimation); values near&nbsp;0 '
            'mean elongation is predominantly tangential or in a non-radial direction.<br><br>'
            '<strong>Orientation circular std (°)</strong> &mdash; '
            'Circular standard deviation of per-star elongation position angles across the entire field. '
            'A small value (&lt;&nbsp;20°) means all stars are elongated in nearly the same direction '
            'regardless of position &mdash; characteristic of collimation error or a tilted element. '
            'A large value (&gt;&nbsp;45°) indicates no dominant elongation direction. '
            '<em>Note: lower is not better</em> &mdash; a very low value paired with high radial '
            'elongation fraction points to a fixed collimation problem, not a healthy image.<br><br>'
            '<strong>Corner/centre FWHM ratio</strong> &mdash; '
            'Median star FWHM in the outer 30&nbsp;% of field radius divided by median FWHM in the '
            'inner 30&nbsp;%. Values near&nbsp;1.0 indicate uniform sharpness across the field. Values '
            '&gt;&nbsp;1.3 indicate measurably blurrier stars toward the corners, consistent with field '
            'curvature, defocus, or related aberration. Cannot distinguish field curvature from pure '
            'defocus without a through-focus sequence.<br><br>'
            '<strong>FWHM radial gradient</strong> &mdash; '
            'Coefficients of a quadratic fit FWHM(r)&nbsp;=&nbsp;a&thinsp;r&#x00B2;&nbsp;+&nbsp;b&thinsp;r&nbsp;+&nbsp;c '
            'on normalised radius r&nbsp;&isin;&nbsp;[0,&nbsp;1]. A large positive quadratic coefficient (a) '
            'indicates accelerating FWHM growth toward the corners (field curvature signature). A '
            'large linear coefficient (b) with small quadratic indicates more linear growth (tilt or '
            'coma). Read in conjunction with the FWHM radial profile plot below.',
            title="What each metric measures",
        ) + "\n"

        img_h_a, img_w_a = img_a.data.shape[:2]
        img_h_b, img_w_b = (img_b.data.shape[:2] if img_b is not None else (0, 0))
        stars_a = pa.get("star_data", [])
        stars_b = pb.get("star_data", [])

        rdf_fig = self._plot_fwhm_ecc_radial(
            stars_a, stars_b, ab_a, ab_b,
            ra.label, rb.label, img_h_a, img_w_a, img_h_b, img_w_b)
        if rdf_fig:
            html += _img_tag(rdf_fig, "FWHM and eccentricity radial profiles")
            html += ('<p class="caption">Radial profiles of FWHM and eccentricity vs distance '
                     'from the field centre. Scatter points show individual star measurements; '
                     'curves show quadratic fits. A rising FWHM profile indicates field '
                     'curvature or defocus; a rising eccentricity profile is consistent with '
                     'off-axis coma. The dashed line marks the eccentricity = 0.10 threshold.</p>\n')

        coma_fig = self._plot_coma_index_scatter(
            ab_a, ab_b, ra.label, rb.label, img_h_a, img_w_a, img_h_b, img_w_b)
        if coma_fig:
            html += _img_tag(coma_fig, "Coma index scatter")
            html += ('<p class="caption">Coma index scatter: absolute radial elongation '
                     'component |e<sub>r</sub>| vs radial distance from the field centre. '
                     'Points are coloured by eccentricity magnitude. A positive slope and '
                     'high Pearson&nbsp;r indicate elongation growing radially outward '
                     '(off-axis coma signature). Horizontal scatter indicates no radial '
                     'dependence. The Pearson&nbsp;r value shown is the coma index in the '
                     'score table above.</p>\n')

        vec_fig = self._plot_aberration_vector_field(
            ab_a, ab_b, ra.label, rb.label, img_h_a, img_w_a, img_h_b, img_w_b)
        er_fig = self._plot_radial_elongation_map(
            ab_a, ab_b, ra.label, rb.label, img_h_a, img_w_a, img_h_b, img_w_b)

        if vec_fig:
            html += _img_tag(vec_fig, "Elongation vector field")
            html += ('<p class="caption">Per-star elongation vector field. '
                     'Each line shows the orientation and magnitude of eccentricity. '
                     'A uniform direction across the field suggests collimation error; '
                     'elongation pointing away from the field centre is consistent with coma.</p>\n')
        if er_fig:
            html += _img_tag(er_fig, "Radial elongation component map")
            html += ('<p class="caption">Smoothed radial elongation component '
                     'e<sub>r</sub>&nbsp;=&nbsp;e&thinsp;&middot;&thinsp;cos(2(&theta;&minus;&phi;)). '
                     'Red: stars elongated radially outward from field centre (coma signature). '
                     'Blue: tangentially elongated. Near zero: circular or mixed orientation.</p>\n')

        html += _info_box(
            '<ul style="margin:0.4em 0 0 1.2em;padding:0;">'
            '<li><strong>Seeing vs optics:</strong> Atmospheric seeing produces random per-frame elongation. '
            'Optical aberrations are repeatable across nights. A single image cannot separate the two '
            '&mdash; compare results from multiple sessions to identify genuine optical signatures.</li>'
            '<li><strong>No absolute wavefront error:</strong> Converting these indices to nm of wavefront '
            'error requires focal ratio and aperture, which are not reliably encoded in image headers. '
            'All scores are relative/comparative only.</li>'
            '<li><strong>Field curvature / defocus degeneracy:</strong> Both produce isotropic FWHM growth '
            'toward the field corners and cannot be distinguished without a through-focus sequence.</li>'
            f'<li><strong>Star count sensitivity:</strong> Scores are unreliable with fewer than '
            f'{ABERRATION_MIN_STARS} well-distributed stars with valid orientation measurements.</li>'
            '<li><strong>Monochromatic:</strong> Chromatic aberration cannot be detected from a single-filter '
            'image.</li>'
            '</ul>',
            title="Limitations of single-image aberration analysis",
        ) + "\n"
        return html

    @staticmethod
    def _plot_fwhm_ecc_radial(
            stars_a: list, stars_b: list,
            ab_a: dict, ab_b: dict,
            label_a: str, label_b: str,
            img_h_a: int, img_w_a: int,
            img_h_b: int, img_w_b: int,
    ) -> "plt.Figure | None":
        """FWHM and eccentricity vs radial distance from field centre (1 row × 2 cols)."""
        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        _FIT_COLOR = {
            "steelblue": "lightskyblue" if _is_dark else "navy",
            "tomato":    "lightsalmon"  if _is_dark else "firebrick",
        }

        def _extract(stars, img_h, img_w):
            cx, cy = img_w / 2.0, img_h / 2.0
            ri, fwhm, ecc = [], [], []
            for s in stars:
                if s.get("fwhm") is not None and s.get("eccentricity") is not None:
                    ri.append(float(np.hypot(s["x"] - cx, s["y"] - cy)))
                    fwhm.append(float(s["fwhm"]))
                    ecc.append(float(s["eccentricity"]))
            return np.array(ri), np.array(fwhm), np.array(ecc)

        ri_a, fwhm_a, ecc_a = _extract(stars_a, img_h_a, img_w_a)
        ri_b, fwhm_b, ecc_b = _extract(stars_b, img_h_b, img_w_b)
        if ri_a.size == 0 and ri_b.size == 0:
            return None

        fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        fig.suptitle("Radial profiles: FWHM and eccentricity vs distance from field centre",
                     fontsize=11)

        for ax, vals_a, vals_b, ylabel, yref in [
            (ax_f, fwhm_a, fwhm_b, "FWHM (px)", None),
            (ax_e, ecc_a,  ecc_b,  "Eccentricity", 0.10),
        ]:
            for ri, vals, col, lbl in [
                (ri_a, vals_a, "steelblue", label_a),
                (ri_b, vals_b, "tomato",    label_b),
            ]:
                if ri.size == 0:
                    continue
                ax.scatter(ri, vals, color=col, alpha=0.55, s=16, label=lbl)
                if ri.size >= 3:
                    order = np.argsort(ri)
                    coeffs = np.polyfit(ri[order], vals[order], 2)
                    r_fit = np.linspace(ri.min(), ri.max(), 200)
                    ax.plot(r_fit, np.polyval(coeffs, r_fit),
                            color=_FIT_COLOR.get(col, col), lw=2.0,
                            label=f"{lbl} fit")
            if yref is not None:
                ax.axhline(yref, color="gray", lw=1.0, ls="--", alpha=0.7,
                           label=f"e = {yref}")
            ax.set_xlabel("Radial distance from centre (px)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
        return fig

    @staticmethod
    def _plot_coma_index_scatter(
            ab_a: dict, ab_b: dict, label_a: str, label_b: str,
            img_h_a: int, img_w_a: int, img_h_b: int, img_w_b: int,
    ) -> "plt.Figure | None":
        """|er| vs radial distance scatter with linear trend; annotated with Pearson r."""
        has_a = bool(ab_a.get("star_xs"))
        has_b = bool(ab_b.get("star_xs"))
        if not has_a and not has_b:
            return None

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        fig.suptitle("Coma index: |eᵣ| vs radial distance from field centre", fontsize=11)

        for ax, ab, img_h, img_w, lbl in [
            (ax_a, ab_a, img_h_a, img_w_a, label_a),
            (ax_b, ab_b, img_h_b, img_w_b, label_b),
        ]:
            xs    = ab.get("star_xs", [])
            ys    = ab.get("star_ys", [])
            er    = ab.get("star_er", [])
            ecc   = ab.get("star_ecc", [])
            ci    = ab.get("coma_index")
            if not xs:
                warn = ab.get("warning", "No data")
                ax.text(0.5, 0.5, warn, transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                ax.set_title(lbl, fontsize=10)
                continue
            cx, cy = img_w / 2.0, img_h / 2.0
            ri      = np.hypot(np.array(xs) - cx, np.array(ys) - cy)
            abs_er  = np.abs(np.array(er))
            ecc_arr = np.array(ecc)
            sc = ax.scatter(ri, abs_er, c=ecc_arr, cmap="plasma",
                            vmin=0, vmax=max(float(ecc_arr.max()), 0.01),
                            s=18, alpha=0.75, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Eccentricity", fraction=0.046, pad=0.04)
            if ri.size >= 2:
                coeffs = np.polyfit(ri, abs_er, 1)
                r_fit = np.linspace(ri.min(), ri.max(), 200)
                ax.plot(r_fit, np.polyval(coeffs, r_fit), color="crimson", lw=1.8,
                        label="Linear fit")
                ax.legend(fontsize=7)
            if ci is not None:
                ax.text(0.03, 0.96, f"r = {ci:.3f}",
                        transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#2a2a2a", ec="#555", alpha=0.8))
            ax.set_xlabel("Radial distance from centre (px)", fontsize=9)
            ax.set_ylabel("|eᵣ|  (radial elongation)", fontsize=9)
            ax.set_title(lbl, fontsize=10)
            ax.grid(True, alpha=0.25)
        return fig

    @staticmethod
    def _plot_aberration_vector_field(
            ab_a: dict, ab_b: dict, label_a: str, label_b: str,
            img_h_a: int, img_w_a: int, img_h_b: int, img_w_b: int
    ) -> "plt.Figure | None":
        from matplotlib.collections import LineCollection
        xs_a, ys_a = ab_a.get("star_xs", []), ab_a.get("star_ys", [])
        xs_b, ys_b = ab_b.get("star_xs", []), ab_b.get("star_ys", [])
        if not xs_a and not xs_b:
            return None

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle("Elongation vector field  (line length ∝ eccentricity, direction = orientation)",
                     fontsize=11)

        for ax, ab, img_h, img_w, lbl in [
            (ax_a, ab_a, img_h_a, img_w_a, label_a),
            (ax_b, ab_b, img_h_b, img_w_b, label_b),
        ]:
            xs    = ab.get("star_xs", [])
            ys    = ab.get("star_ys", [])
            ecc   = ab.get("star_ecc", [])
            theta = ab.get("star_theta", [])
            if xs:
                xs_arr    = np.array(xs)
                ys_arr    = np.array(ys)
                ecc_arr   = np.array(ecc)
                theta_arr = np.array(theta)
                scale = max(img_w, img_h) * 0.04   # px per unit eccentricity
                dx = ecc_arr * np.cos(theta_arr) * scale
                dy = ecc_arr * np.sin(theta_arr) * scale
                segs = [[(xi - dxi, yi - dyi), (xi + dxi, yi + dyi)]
                        for xi, yi, dxi, dyi in zip(xs_arr, ys_arr, dx, dy)]
                lc = LineCollection(segs, array=ecc_arr, cmap="plasma",
                                    linewidths=1.2, alpha=0.85,
                                    norm=mcolors.Normalize(vmin=0, vmax=1))
                ax.add_collection(lc)
                fig.colorbar(lc, ax=ax, label="Eccentricity", fraction=0.046, pad=0.04)
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            ax.set_aspect("equal")
        return fig

    @staticmethod
    def _plot_radial_elongation_map(
            ab_a: dict, ab_b: dict, label_a: str, label_b: str,
            img_h_a: int, img_w_a: int, img_h_b: int, img_w_b: int
    ) -> "plt.Figure | None":
        xs_a, ys_a, er_a = ab_a.get("star_xs",[]), ab_a.get("star_ys",[]), ab_a.get("star_er",[])
        xs_b, ys_b, er_b = ab_b.get("star_xs",[]), ab_b.get("star_ys",[]), ab_b.get("star_er",[])
        if not xs_a and not xs_b:
            return None

        pts_a = list(zip(xs_a, ys_a, er_a))
        pts_b = list(zip(xs_b, ys_b, er_b))
        map_a = _psf_make_map(pts_a, img_h_a, img_w_a)
        map_b = _psf_make_map(pts_b, img_h_b, img_w_b)

        all_er = er_a + er_b
        vlim = max(float(np.percentile(np.abs(all_er), 98)) if all_er else 0.0, 0.05)

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle("Radial elongation component  (er = e · cos(2·(θ − φ)))", fontsize=11)

        for ax, m, xs, ys, er, img_h, img_w, lbl in [
            (ax_a, map_a, xs_a, ys_a, er_a, img_h_a, img_w_a, label_a),
            (ax_b, map_b, xs_b, ys_b, er_b, img_h_b, img_w_b, label_b),
        ]:
            if m is not None:
                im = ax.imshow(m, origin="upper", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                               extent=[0, img_w, img_h, 0], aspect="equal",
                               interpolation="bilinear")
                fig.colorbar(im, ax=ax, label="er", fraction=0.046, pad=0.04)
            if xs:
                ax.scatter(xs, ys, c=er, cmap="RdBu_r",
                           s=18, edgecolors="white", linewidths=0.5, zorder=3,
                           norm=mcolors.Normalize(vmin=-vlim, vmax=vlim))
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        return fig

    @staticmethod
    def _plot_psf_spatial_map(
            stars_a: list, stars_b: list, field: str,
            label_a: str, label_b: str,
            img_h_a: int, img_w_a: int,
            img_h_b: int, img_w_b: int,
            title: str, cmap: str = "viridis") -> "plt.Figure | None":
        pts_a = [(s["x"], s["y"], s[field]) for s in stars_a if s.get(field) is not None]
        pts_b = [(s["x"], s["y"], s[field]) for s in stars_b if s.get(field) is not None]
        if not pts_a and not pts_b:
            return None

        all_vals = [p[2] for p in pts_a] + [p[2] for p in pts_b]
        vmin, vmax = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))

        map_a = _psf_make_map(pts_a, img_h_a, img_w_a)

        if not pts_b:
            # Single-image: one panel only
            fig, ax_a = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
            fig.suptitle(title, fontsize=11)
            if map_a is not None:
                im = ax_a.imshow(map_a, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                                 extent=[0, img_w_a, img_h_a, 0], aspect="equal",
                                 interpolation="bilinear")
                fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
            if pts_a:
                ax_a.scatter([p[0] for p in pts_a], [p[1] for p in pts_a],
                             c=[p[2] for p in pts_a], cmap=cmap, vmin=vmin, vmax=vmax,
                             s=18, edgecolors="white", linewidths=0.5, zorder=3)
            ax_a.set_title(label_a, fontsize=10)
            ax_a.set_xlabel("x (px)")
            ax_a.set_ylabel("y (px)")
            return fig

        map_b = _psf_make_map(pts_b, img_h_b, img_w_b)
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle(title, fontsize=11)

        for ax, m, pts, img_h, img_w, lbl in [
            (ax_a, map_a, pts_a, img_h_a, img_w_a, label_a),
            (ax_b, map_b, pts_b, img_h_b, img_w_b, label_b),
        ]:
            if m is not None:
                im = ax.imshow(m, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=[0, img_w, img_h, 0], aspect="equal",
                               interpolation="bilinear")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           c=[p[2] for p in pts], cmap=cmap, vmin=vmin, vmax=vmax,
                           s=18, edgecolors="white", linewidths=0.5, zorder=3)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        return fig

    @staticmethod
    def _plot_psf_histogram(
            vals_a: list, vals_b: list,
            label_a: str, label_b: str,
            xlabel: str, title: str) -> "plt.Figure | None":
        if not vals_a and not vals_b:
            return None
        all_vals = vals_a + vals_b
        rng = (float(min(all_vals)), float(max(all_vals)))
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        if vals_a:
            ax.hist(vals_a, bins=40, range=rng, alpha=XS_LINE_ALPHA,
                    color="#ff7f0e", label=label_a, edgecolor="none")
        if vals_b:
            ax.hist(vals_b, bins=40, range=rng, alpha=XS_LINE_ALPHA,
                    color="#1f77b4", label=label_b, edgecolor="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return fig

    def _plot_fwhm_scatter(self, ra: AnalysisResult, rb: AnalysisResult) -> plt.Figure | None:
        """Scatter plot of per-star FWHM_A vs FWHM_B for matched stars."""
        data_a = (ra.psf_metrics or {}).get("star_data", [])
        data_b = (rb.psf_metrics or {}).get("star_data", [])
        if not data_a or not data_b:
            return None

        # Match by nearest neighbour in image coordinates (valid post-alignment)
        matched_a, matched_b = [], []
        pos_b = np.array([[s["x"], s["y"]] for s in data_b])
        for sa in data_a:
            dists = np.sqrt((pos_b[:, 0] - sa["x"])**2 + (pos_b[:, 1] - sa["y"])**2)
            idx = int(np.argmin(dists))
            if dists[idx] < 15.0:
                matched_a.append(sa["fwhm"])
                matched_b.append(data_b[idx]["fwhm"])

        if len(matched_a) < 3:
            return None

        fa = np.array(matched_a)
        fb = np.array(matched_b)
        lo = min(fa.min(), fb.min()) * 0.9
        hi = max(fa.max(), fb.max()) * 1.1

        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if _is_dark else "black"

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(fa, fb, alpha=0.65, color="steelblue", s=25, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=orig_color, linestyle="--",
                linewidth=1.2, label="Slope = 1 (equal FWHM)")
        ax.set_xlabel(f"FWHM {ra.label} (px)")
        ax.set_ylabel(f"FWHM {rb.label} (px)")
        ax.set_title(f"Per-star FWHM correlation  (n = {len(fa)} matched stars)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_epsf(self, epsf: np.ndarray, label: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(np.log1p(epsf - epsf.min()),
                       origin="upper", cmap="viridis", interpolation="nearest")
        ax.set_title(f"ePSF — {label}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def _compute_mtf_from_kernel(self, kern: np.ndarray):
        """Return (freq_axis, mtf) for a native-pixel-scale PSF kernel."""
        kern_norm = kern / (kern.sum() or 1.0)
        fft2d = np.fft.fftshift(np.fft.fft2(kern_norm))
        otf = np.abs(fft2d)
        otf /= otf.max() or 1.0
        n = kern_norm.shape[0]
        cy, cx = n // 2, n // 2
        y, x = np.ogrid[:n, :n]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = n / 2.0
        n_bins = n // 2
        freq_edges = np.linspace(0, max_r, n_bins + 1)
        freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2.0
        mtf = np.array([
            otf[np.logical_and(r >= freq_edges[j], r < freq_edges[j + 1])].mean()
            if np.any(np.logical_and(r >= freq_edges[j], r < freq_edges[j + 1])) else 0.0
            for j in range(n_bins)
        ])
        return freq_centers / max_r * 0.5, mtf   # freq in cycles/native-pixel

    def _overlay_mtf(self,
                      freq_a: "np.ndarray | None", mtf_a: "np.ndarray | None",
                      freq_b: "np.ndarray | None", mtf_b: "np.ndarray | None",
                      label_a: str, label_b: str,
                      freq_ref=None, mtf_ref=None,
                      label_ref: str = "Reference") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        if freq_a is not None and mtf_a is not None:
            ax.plot(freq_a, mtf_a, color="steelblue", linewidth=2, label=label_a)
        if freq_b is not None and mtf_b is not None:
            ax.plot(freq_b, mtf_b, color="tomato", linewidth=2, label=label_b)
        if freq_ref is not None and mtf_ref is not None:
            ax.plot(freq_ref, mtf_ref, color="forestgreen", linewidth=1.5,
                    linestyle="--", label=label_ref)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="red", linestyle=":", linewidth=0.8, label="Nyquist")
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("MTF")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title("MTF comparison")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_psf_simulation(self, ra: AnalysisResult,
                              rb: AnalysisResult) -> dict | None:
        """Convolve the test chart with each filter's ePSF.

        Returns a dict with uint8 numpy arrays (one per panel) at native pixel
        resolution, or None if unavailable.  Keys: 'original', 'conv_a', 'conv_b',
        'diff', 'diff_max', 'label_a', 'label_b'.
        """
        epsf_a = (ra.psf_metrics or {}).get("epsf_data")
        epsf_b = (rb.psf_metrics or {}).get("epsf_data")
        if epsf_a is None:
            return None
        if not _TEST_IMAGE_PATH.exists():
            return None
        try:
            test_arr = np.array(
                _PILImage.open(_TEST_IMAGE_PATH).convert("L"), dtype=float
            ) / 255.0

            os_a = (ra.psf_metrics or {}).get("epsf_oversampling", 2)
            kern_a = _ndimage_zoom(epsf_a, 1.0 / os_a, order=1)
            kern_a = kern_a / kern_a.sum() if kern_a.sum() > 0 else kern_a

            kern_b = None
            if epsf_b is not None:
                os_b = (rb.psf_metrics or {}).get("epsf_oversampling", 2)
                kern_b = _ndimage_zoom(epsf_b, 1.0 / os_b, order=1)
                kern_b = kern_b / kern_b.sum() if kern_b.sum() > 0 else kern_b

            # Reference PSF kernel
            ref_fwhm = _ref_fwhm_px(ra.psf_metrics or {}, rb.psf_metrics or {},
                                        self._ref_seeing_arcsec)
            kern_ref  = _make_moffat_kernel(ref_fwhm) if ref_fwhm is not None else None

            # Convolution at full resolution
            conv_a = np.clip(fftconvolve(test_arr, kern_a, mode="same"), 0.0, 1.0)
            conv_b = (np.clip(fftconvolve(test_arr, kern_b, mode="same"), 0.0, 1.0)
                      if kern_b is not None else None)
            conv_ref = (np.clip(fftconvolve(test_arr, kern_ref, mode="same"), 0.0, 1.0)
                        if kern_ref is not None else None)
            diff = conv_a - conv_b if conv_b is not None else None

            # Cross-section extraction — must happen before downsampling
            _XS_Y_ROWS = {"high": 105, "medium": 425, "low": 670}
            _XS_X = slice(70, 831)
            xs_data: dict = {}
            for level, y_px in _XS_Y_ROWS.items():
                if 0 <= y_px < test_arr.shape[0] and 830 < test_arr.shape[1]:
                    half_strip = 50
                    y0_strip = max(0, y_px - half_strip)
                    y1_strip = min(test_arr.shape[0], y_px + half_strip)
                    xs_data[level] = {
                        "y_px":        y_px,
                        "original":    test_arr[y_px, _XS_X].copy(),
                        "conv_a":      conv_a[y_px, _XS_X].copy(),
                        "conv_b":      conv_b[y_px, _XS_X].copy() if conv_b is not None else None,
                        "conv_ref":    conv_ref[y_px, _XS_X].copy() if conv_ref is not None else None,
                        "image_strip": test_arr[y0_strip:y1_strip, _XS_X].copy(),
                    }

            # Downsample for display if image is very large (cap at 1200 px on long edge)
            h, w = test_arr.shape
            if max(h, w) > 1200:
                zoom_f = 1200.0 / max(h, w)
                test_arr = _ndimage_zoom(test_arr, zoom_f, order=1)
                conv_a   = _ndimage_zoom(conv_a,   zoom_f, order=1)
                if conv_b is not None:
                    conv_b = _ndimage_zoom(conv_b, zoom_f, order=1)
                if diff is not None:
                    diff   = _ndimage_zoom(diff,   zoom_f, order=1)
                if conv_ref is not None:
                    conv_ref = _ndimage_zoom(conv_ref, zoom_f, order=1)

            diff_rgb = None
            d_max = None
            if diff is not None:
                d_max = max(float(abs(diff).max()), 1e-9)
                diff_norm = (diff / d_max + 1.0) / 2.0          # [0, 1]
                diff_rgb = (plt.get_cmap("RdBu_r")(diff_norm)[:, :, :3] * 255).astype(np.uint8)

            label_ref = (f"Reference ({self._ref_seeing_arcsec:.1f}″ seeing)"
                         if conv_ref is not None else None)
            return {
                "original":  (test_arr * 255).astype(np.uint8),
                "conv_a":    (conv_a   * 255).astype(np.uint8),
                "conv_b":    (conv_b   * 255).astype(np.uint8) if conv_b is not None else None,
                "conv_ref":  (conv_ref * 255).astype(np.uint8) if conv_ref is not None else None,
                "diff":      diff_rgb,
                "diff_max":  d_max,
                "label_a":   ra.label,
                "label_b":   rb.label,
                "label_ref": label_ref,
                "xs_data":   xs_data,
            }
        except Exception:
            return None

    def _plot_psf_crosssections(self, sim: dict) -> "list[plt.Figure]":
        """One dual-subplot figure per contrast level (raw + envelope, sqrt x-axis)."""
        xs_data = sim.get("xs_data", {})
        if not xs_data:
            return []

        levels = [("high", "High contrast"), ("medium", "Medium contrast"), ("low", "Low contrast")]

        def _envelope(arr: np.ndarray, half: int = 5) -> np.ndarray:
            n = len(arr)
            env = np.empty(n)
            for i in range(n):
                window = arr[max(0, i - half): min(n, i + half + 1)]
                env[i] = window.max() - window.min()
            return env

        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if _is_dark else "black"

        figs = []
        for level, title in levels:
            if level not in xs_data:
                continue
            d = xs_data[level]
            n_pts = len(d["original"])
            # x=1 = finest bars; increases toward coarser bars (data reversed).
            x = np.arange(1, n_pts + 1)
            orig      = d["original"][::-1]
            a_arr     = d["conv_a"][::-1]
            b_arr     = d["conv_b"][::-1] if d.get("conv_b") is not None else None
            _ref_raw  = d.get("conv_ref")
            ref_arr   = _ref_raw[::-1] if _ref_raw is not None else None
            ref_label = sim.get("label_ref", "Reference seeing")
            strip = d.get("image_strip")

            # Build layout: optional image strip on top, then raw + envelope
            if strip is not None:
                fig = plt.figure(figsize=(10, 8))
                gs  = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1], hspace=0.08)
                ax_img = fig.add_subplot(gs[0])
                ax_raw = fig.add_subplot(gs[1])
                ax_env = fig.add_subplot(gs[2], sharex=ax_raw)
            else:
                fig = plt.figure(figsize=(10, 6))
                gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
                ax_raw = fig.add_subplot(gs[0])
                ax_env = fig.add_subplot(gs[1], sharex=ax_raw)

            # ── Image strip (linear x-scale, NOT shared with plots below) ─
            if strip is not None:
                strip_flipped = strip[:, ::-1]
                # vmin/vmax fixed at [0, 1] — matches ideal test chart appearance
                ax_img.imshow(strip_flipped, cmap="gray", aspect="auto",
                              interpolation="nearest", vmin=0.0, vmax=1.0)
                ax_img.set_title(f"{title}  (y = {d['y_px']} px,  original test chart strip)",
                                 fontsize=9)
                ax_img.set_ylabel("y (px)", fontsize=7)
                ax_img.tick_params(axis="x", labelbottom=False)
                ax_img.tick_params(labelsize=7)

            # ── Raw intensity traces ───────────────────────────────────
            ax_raw.plot(x, orig,  color=orig_color,  linewidth=1.0, alpha=XS_LINE_ALPHA,
                        label="Original")
            ax_raw.plot(x, a_arr, color="steelblue", linewidth=1.0, alpha=XS_LINE_ALPHA,
                        label=sim["label_a"])
            if b_arr is not None:
                ax_raw.plot(x, b_arr, color="tomato", linewidth=1.0, alpha=XS_LINE_ALPHA,
                            label=sim["label_b"])
            if ref_arr is not None:
                ax_raw.plot(x, ref_arr, color="forestgreen", linewidth=1.0,
                            linestyle="-", alpha=XS_LINE_ALPHA, label=ref_label)
            if strip is None:
                ax_raw.set_title(f"{title}  (y = {d['y_px']} px)", fontsize=9)
            ax_raw.set_ylabel("Intensity [0–1]", fontsize=8)
            ax_raw.tick_params(labelsize=7)
            ax_raw.legend(fontsize=7)
            ax_raw.grid(True, alpha=0.3, which="both")

            # ── Local contrast envelope ────────────────────────────────
            ax_env.plot(x, _envelope(orig),  color=orig_color,  linewidth=1.4,
                        alpha=XS_LINE_ALPHA, label="Original")
            ax_env.plot(x, _envelope(a_arr), color="steelblue", linewidth=1.4,
                        alpha=XS_LINE_ALPHA, label=sim["label_a"])
            if b_arr is not None:
                ax_env.plot(x, _envelope(b_arr), color="tomato", linewidth=1.4,
                            alpha=XS_LINE_ALPHA, label=sim["label_b"])
            if ref_arr is not None:
                ax_env.plot(x, _envelope(ref_arr), color="forestgreen", linewidth=1.4,
                            linestyle="-", alpha=XS_LINE_ALPHA, label=ref_label)
            ax_env.set_ylabel("Local contrast\n(peak − valley)", fontsize=8)
            ax_env.set_xlabel(
                "Distance from finest bars (px, √ scale)  —  fine ← | → coarse", fontsize=8)
            ax_env.tick_params(labelsize=7)
            ax_env.legend(fontsize=7)
            ax_env.grid(True, alpha=0.3, which="both")

            # ── Square-root x-axis on plot panels (NOT image strip) ───
            for ax in (ax_raw, ax_env):
                ax.set_xscale("function", functions=(np.sqrt, np.square))
                ax.set_xlim(1, n_pts)

            fig.tight_layout()
            figs.append(fig)

        return figs

    def _plot_psf_modulation(self, sim: dict) -> "plt.Figure | None":
        """Bar chart of Michelson contrast at each contrast level for original vs both filters."""
        xs_data = sim.get("xs_data", {})
        if not xs_data:
            return None

        def modulation(arr: np.ndarray) -> float:
            lo, hi = float(np.min(arr)), float(np.max(arr))
            denom = hi + lo
            return (hi - lo) / denom if denom > 0 else 0.0

        levels = ["high", "medium", "low"]
        x = np.arange(len(levels))

        has_b = any(xs_data[lv].get("conv_b") is not None for lv in xs_data)
        width = 0.25 if has_b else 0.3

        orig_m   = [modulation(xs_data[lv]["original"]) if lv in xs_data else 0.0 for lv in levels]
        conv_a_m = [modulation(xs_data[lv]["conv_a"])   if lv in xs_data else 0.0 for lv in levels]

        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if _is_dark else "black"

        fig, ax = plt.subplots(figsize=(7, 4))
        if has_b:
            conv_b_m = [modulation(xs_data[lv]["conv_b"]) if lv in xs_data else 0.0 for lv in levels]
            ax.bar(x - width, orig_m,   width, label="Original",        color=orig_color,  alpha=0.75)
            ax.bar(x,         conv_a_m, width, label=sim["label_a"],    color="steelblue", alpha=0.85)
            ax.bar(x + width, conv_b_m, width, label=sim["label_b"],    color="tomato",    alpha=0.85)
        else:
            ax.bar(x - width / 2, orig_m,   width, label="Original",     color=orig_color,  alpha=0.75)
            ax.bar(x + width / 2, conv_a_m, width, label=sim["label_a"], color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["High contrast", "Medium contrast", "Low contrast"])
        ax.set_ylabel("Michelson contrast  (Imax − Imin) / (Imax + Imin)")
        ax.set_ylim(0, 1.08)
        ax.set_title("Contrast modulation preserved after PSF convolution\n"
                     "(higher = better contrast retention)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    def _plot_psf_band_modulation(self, sim: dict) -> "plt.Figure | None":
        """Grouped bar chart: mean local contrast per spatial-frequency band, averaged
        across the three contrast rows.  Four bands span fine → coarse bar periods."""
        xs_data = sim.get("xs_data", {})
        if not xs_data:
            return None

        def _envelope(arr: np.ndarray, half: int = 5) -> np.ndarray:
            n = len(arr)
            env = np.empty(n)
            for i in range(n):
                window = arr[max(0, i - half): min(n, i + half + 1)]
                env[i] = window.max() - window.min()
            return env

        # Bands are index ranges into the *reversed* array (index 0 = finest bar).
        bands = [
            ("Fine\n(1–40 px)",       slice(0,   40)),
            ("Mid-fine\n(41–120 px)", slice(40,  120)),
            ("Mid\n(121–300 px)",     slice(120, 300)),
            ("Coarse\n(301+ px)",     slice(300, None)),
        ]
        level_info = [
            ("high",   "High contrast"),
            ("medium", "Medium contrast"),
            ("low",    "Low contrast"),
        ]
        available = [(k, t) for k, t in level_info if k in xs_data]
        if not available:
            return None

        has_b = any(xs_data[lv].get("conv_b") is not None for lv in xs_data)
        n_bands = len(bands)
        x       = np.arange(n_bands)
        width = 0.25 if has_b else 0.3

        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if _is_dark else "black"

        fig, axes = plt.subplots(len(available), 1,
                                  figsize=(9, 4 * len(available)), squeeze=False)

        for ax_row, (level, level_title) in zip(axes[:, 0], available):
            d        = xs_data[level]
            env_orig = _envelope(d["original"][::-1])
            env_a    = _envelope(d["conv_a"][::-1])
            orig_vals = [float(np.mean(env_orig[sl])) for _, sl in bands]
            a_vals    = [float(np.mean(env_a[sl]))    for _, sl in bands]

            if has_b and d.get("conv_b") is not None:
                env_b  = _envelope(d["conv_b"][::-1])
                b_vals = [float(np.mean(env_b[sl])) for _, sl in bands]
                ax_row.bar(x - width, orig_vals, width, label="Original",
                           color=orig_color,  alpha=0.75)
                ax_row.bar(x,         a_vals,   width, label=sim["label_a"],
                           color="steelblue", alpha=0.85)
                ax_row.bar(x + width, b_vals,   width, label=sim["label_b"],
                           color="tomato",    alpha=0.85)
            else:
                ax_row.bar(x - width / 2, orig_vals, width, label="Original",
                           color=orig_color,  alpha=0.75)
                ax_row.bar(x + width / 2, a_vals,   width, label=sim["label_a"],
                           color="steelblue", alpha=0.85)
            ax_row.set_xticks(x)
            ax_row.set_xticklabels([b[0] for b in bands], fontsize=8)
            ax_row.set_ylabel("Mean local contrast\n(peak − valley)", fontsize=8)
            ax_row.set_title(f"Contrast retention — {level_title}", fontsize=9)
            ax_row.legend(fontsize=8)
            ax_row.grid(True, alpha=0.3, axis="y")

        axes[-1, 0].set_xlabel("Spatial-frequency band (bar period in pixels)", fontsize=8)
        fig.tight_layout()
        return fig

    def _plot_psf_contrast_retention_ratios(self, sim: dict) -> "plt.Figure | None":
        """Single grouped bar chart: convolved/original contrast retention ratio per band.

        X-axis: 4 spatial-frequency bands. 6 bars per group (3 contrast levels × 2 images).
        Colors distinguish contrast levels; solid fill = Image A, hatched fill = Image B.
        Error bars show std of the pointwise per-pixel ratios within each band.
        Values <= 1.0: ratio = 1 means no blur; lower = more blur from that filter's PSF.
        """
        xs_data = sim.get("xs_data", {})
        if not xs_data:
            return None

        def _envelope(arr: np.ndarray, half: int = 5) -> np.ndarray:
            n = len(arr)
            env = np.empty(n)
            for i in range(n):
                window = arr[max(0, i - half): min(n, i + half + 1)]
                env[i] = window.max() - window.min()
            return env

        bands = [
            ("Fine\n(1–40 px)",       slice(0,   40)),
            ("Mid-fine\n(41–120 px)", slice(40,  120)),
            ("Mid\n(121–300 px)",     slice(120, 300)),
        ]
        level_info = [
            ("high",   "High",   "#2196F3"),
            ("medium", "Medium", "#FF9800"),
            ("low",    "Low",    "#4CAF50"),
        ]
        available = [(k, t, c) for k, t, c in level_info if k in xs_data]
        if not available:
            return None

        has_b     = any(xs_data[k].get("conv_b") is not None for k, _, _ in available)
        has_ref   = sim.get("label_ref") is not None and any(
            xs_data[k].get("conv_ref") is not None for k, _, _ in available
        )
        n_levels  = len(available)
        n_bands   = len(bands)
        # bars per group: A always present; B optional; Ref optional
        bars_per_level = (1 + int(has_b) + int(has_ref))
        if has_ref and has_b:
            bar_w = 0.09
        elif has_b or has_ref:
            bar_w = 0.12
        else:
            bar_w = 0.15
        group_gap = n_levels * bars_per_level * bar_w + 0.15
        x_centers = np.arange(n_bands) * group_gap

        fig, ax = plt.subplots(figsize=(10 if (has_ref or has_b) else 8, 5))

        # Compute per-level x offsets so bars are centred within each group
        offsets = np.linspace(-(n_levels - 1) * bars_per_level * bar_w / 2,
                               (n_levels - 1) * bars_per_level * bar_w / 2, n_levels)

        from matplotlib.patches import Patch
        legend_handles = []

        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if _is_dark else "black"

        for i, (level, title, color) in enumerate(available):
            d        = xs_data[level]
            env_orig = _envelope(d["original"][::-1])
            env_a    = _envelope(d["conv_a"][::-1])
            env_ref  = (_envelope(d["conv_ref"][::-1])
                        if has_ref and d.get("conv_ref") is not None else None)

            ratios_a, errs_a = [], []
            ratios_b, errs_b = [], []
            ratios_ref, errs_ref = [], []
            for _, sl in bands:
                orig_b = env_orig[sl]
                pw_a = env_a[sl] / np.clip(orig_b, 1e-6, None)
                ratios_a.append(float(pw_a.mean()))
                errs_a.append(float(pw_a.std()))
                if has_b and d.get("conv_b") is not None:
                    env_b = _envelope(d["conv_b"][::-1])
                    pw_b = env_b[sl] / np.clip(orig_b, 1e-6, None)
                    ratios_b.append(float(pw_b.mean()))
                    errs_b.append(float(pw_b.std()))
                if env_ref is not None:
                    pw_r = env_ref[sl] / np.clip(orig_b, 1e-6, None)
                    ratios_ref.append(float(pw_r.mean()))
                    errs_ref.append(float(pw_r.std()))

            # Positions: A always leftmost in the group
            half = (bars_per_level - 1) / 2.0
            pos_a = x_centers + offsets[i] - half * bar_w
            pos_b = pos_a + bar_w if has_b else None
            pos_ref = (pos_a + bar_w * int(has_b) + bar_w) if has_ref else None

            _ekw = {"linewidth": 0.8, "ecolor": orig_color}
            ax.bar(pos_a, ratios_a, bar_w, color=color, alpha=0.85,
                   yerr=errs_a, capsize=3, error_kw=_ekw)
            if has_b and ratios_b:
                ax.bar(pos_b, ratios_b, bar_w, color=color, alpha=0.85,
                       hatch="///", yerr=errs_b, capsize=3, error_kw=_ekw)
            if has_ref and ratios_ref:
                ax.bar(pos_ref, ratios_ref, bar_w, facecolor="none",
                       edgecolor="forestgreen", linewidth=1.5, linestyle=":",
                       yerr=errs_ref, capsize=3, error_kw=_ekw)

            legend_handles.append(Patch(facecolor=color, alpha=0.85, label=f"{title} — {sim['label_a']}"))
            if has_b:
                legend_handles.append(Patch(facecolor=color, alpha=0.85, hatch="///", label=f"{title} — {sim['label_b']}"))

        extra_handles = [plt.Line2D([0], [0], color=orig_color, linestyle="--", linewidth=1.0,
                                    label="Full retention (ratio = 1)")]
        if has_ref:
            extra_handles.append(Patch(facecolor="none", edgecolor="forestgreen",
                                       linewidth=1.5, label=sim["label_ref"]))

        ax.axhline(1.0, color=orig_color, linestyle="--", linewidth=1.0)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([b[0] for b in bands], fontsize=8)
        ax.set_xlabel("Spatial-frequency band (bar period in pixels)", fontsize=9)
        ax.set_ylabel("Contrast retention ratio  (convolved / original)", fontsize=9)
        title_str = (f"Spatial-Frequency Contrast Retention — {sim['label_a']} vs {sim['label_b']}"
                     if has_b else f"Spatial-Frequency Contrast Retention — {sim['label_a']}")
        ax.set_title(title_str, fontsize=10)
        ax.legend(handles=legend_handles + extra_handles, fontsize=7,
                  ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    def _psf_retention_table(self, sim: dict) -> str:
        """HTML summary table: mean ± std of convolved/original contrast retention per band and contrast level.

        Columns: Band | Contrast | Image A | Image B | paired t-test p-value.
        Paired t-test is appropriate because pw_a[i] and pw_b[i] share the same
        env_orig[i] denominator — they are paired at each pixel position.
        """
        from scipy.stats import ttest_rel
        xs_data = sim.get("xs_data", {})
        if not xs_data:
            return ""

        def _envelope(arr: np.ndarray, half: int = 5) -> np.ndarray:
            n = len(arr)
            env = np.empty(n)
            for i in range(n):
                window = arr[max(0, i - half): min(n, i + half + 1)]
                env[i] = window.max() - window.min()
            return env

        bands = [
            ("Fine (1–40 px)",        slice(0,   40)),
            ("Mid-fine (41–120 px)",  slice(40,  120)),
            ("Mid (121–300 px)",      slice(120, 300)),
        ]
        level_info = [
            ("high",   "High"),
            ("medium", "Medium"),
            ("low",    "Low"),
        ]
        available = [(k, t) for k, t in level_info if k in xs_data]
        if not available:
            return ""

        label_a   = sim.get("label_a", "A")
        label_b   = sim.get("label_b", "B")
        label_ref = sim.get("label_ref")   # None when plate scale unknown
        has_b     = any(xs_data[k].get("conv_b") is not None for k, _ in available)
        has_ref   = label_ref is not None and any(
            xs_data[k].get("conv_ref") is not None for k, _ in available
        )

        ref_header  = f"<th>{label_ref}</th>" if has_ref else ""
        b_header    = f"<th>{label_b}</th>" if has_b else ""
        stat_header = "<th>Stat. test</th>" if has_b else ""
        header = (
            f"<tr><th>Band</th><th>Contrast</th>"
            f"<th>{label_a}</th>{b_header}"
            f"{ref_header}{stat_header}</tr>"
        )

        rows = []
        for band_label, sl in bands:
            n_levels = len(available)
            for idx, (key, level_title) in enumerate(available):
                d        = xs_data[key]
                env_orig = _envelope(d["original"][::-1])
                env_a    = _envelope(d["conv_a"][::-1])
                pw_a = env_a[sl] / np.clip(env_orig[sl], 1e-6, None)

                b_cell  = ""
                p_cell  = ""
                sa_style = ""
                if has_b and d.get("conv_b") is not None:
                    env_b = _envelope(d["conv_b"][::-1])
                    pw_b = env_b[sl] / np.clip(env_orig[sl], 1e-6, None)

                    _, p = ttest_rel(pw_a, pw_b)
                    p_str = "p&lt;0.001" if p < 0.001 else f"p={p:.3f}"
                    p_style = (' style="background:#b3e5fc"' if p < 0.05
                               else ' style="background:#e0e0e0"')
                    p_cell  = f"<td{p_style}>{p_str}</td>"

                    tol = 1e-4
                    if pw_a.mean() > pw_b.mean() + tol:
                        sa_style = ' style="background:#c8e6c9"'
                        sb_style = ' style="background:#ffcdd2"'
                    elif pw_b.mean() > pw_a.mean() + tol:
                        sa_style = ' style="background:#ffcdd2"'
                        sb_style = ' style="background:#c8e6c9"'
                    else:
                        sa_style = sb_style = ""
                    b_cell = f"<td{sb_style}>{pw_b.mean():.3f}&nbsp;&plusmn;&nbsp;{pw_b.std():.3f}</td>"

                # Reference column (neutral shading, no statistical test)
                ref_cell = ""
                if has_ref and d.get("conv_ref") is not None:
                    env_ref = _envelope(d["conv_ref"][::-1])
                    pw_ref  = env_ref[sl] / np.clip(env_orig[sl], 1e-6, None)
                    ref_cell = (f'<td style="background:#f5f5f5">'
                                f'{pw_ref.mean():.3f}&nbsp;&plusmn;&nbsp;{pw_ref.std():.3f}</td>')

                # Span the Band cell across all contrast-level rows in this band
                band_cell = (
                    f'<td rowspan="{n_levels}"><b>{band_label}</b></td>'
                    if idx == 0 else ""
                )

                rows.append(
                    f"<tr>{band_cell}"
                    f"<td>{level_title}</td>"
                    f"<td{sa_style}>{pw_a.mean():.3f}&nbsp;&plusmn;&nbsp;{pw_a.std():.3f}</td>"
                    f"{b_cell}{ref_cell}{p_cell}</tr>"
                )

        ref_footnote = (
            f" The Reference column shows retention for a synthetic Moffat PSF at "
            f"{self._ref_seeing_arcsec:.1f}&Prime; seeing (&beta;&thinsp;=&thinsp;{REF_SEEING_BETA}) "
            f"— a benchmark for typical good atmospheric conditions."
        ) if has_ref else ""

        stat_footnote = (
            "<p class=\"footnote\">Paired t-test on per-pixel retention ratios (same pixel positions, "
            "different PSFs). Highlighted cells: p&lt;0.05 — the two images are statistically "
            f"distinguishable in that band/contrast combination.{ref_footnote}</p>"
        ) if has_b else ""

        return (
            "<p><strong>Contrast retention summary (mean &plusmn; std, convolved / original):"
            "</strong></p>"
            f"<table>{header}{''.join(rows)}</table>"
            f"{stat_footnote}"
        )

    def _psf_simulation_html(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        """Return HTML block with PSF simulation panels at 1:1 pixel resolution."""
        sim = self._plot_psf_simulation(ra, rb)
        if sim is None:
            return ""

        has_b = sim.get("conv_b") is not None

        def panel(arr, title, caption=""):
            tag = _arr_img_tag(arr, title)
            cap = f'<p class="caption">{caption}</p>' if caption else ""
            return f'<div style="margin-bottom:20px;"><p><strong>{title}</strong></p>{tag}{cap}</div>'

        diff_panel = ""
        if has_b and sim.get("diff") is not None:
            diff_caption = (
                f"Pixel-level difference A − B (RdBu_r colormap, range ±{sim['diff_max']:.4f}). "
                "Red = A brighter after convolution; blue = B brighter. "
                "Larger values in fine-detail regions indicate a measurable sharpness difference."
            )
            diff_panel = panel(sim['diff'], 'Difference (A − B)', diff_caption)

        xs_figs          = self._plot_psf_crosssections(sim)
        band_mod_fig     = _img_tag(self._plot_psf_band_modulation(sim),           "Frequency-band contrast retention")
        retention_fig    = _img_tag(self._plot_psf_contrast_retention_ratios(sim), "Contrast retention ratios")

        color_key = (f"Black = original; blue = {sim['label_a']}; red = {sim['label_b']}."
                     if has_b else f"Black = original; blue = {sim['label_a']}.")
        xs_caption = (
            "<strong>Top panel (image strip):</strong> ~100-row grayscale strip from the original "
            "test chart centred on the cross-section row, displayed with finest bars at the left "
            "(same left-right orientation as the plots below). Color scale is fixed at [0, 1] so "
            "bars appear as in the ideal chart. The image x-axis is linear pixel spacing; the "
            "plots below use a square-root scale, so bar widths do not align exactly. "
            "<strong>Middle subplot:</strong> raw intensity cross-section through the bar pattern "
            "at the indicated pixel row. "
            "<strong>Bottom subplot:</strong> rolling local contrast (peak&minus;valley over a "
            "±5&thinsp;px window) — the envelope of bar oscillations, showing how much contrast "
            "each PSF preserves at each spatial scale. "
            "X-axis uses a <strong>square-root scale</strong> (ticks show actual pixel distance "
            "from the finest bars): fine bars are spread across the left where PSF differences "
            f"matter most; the coarser end is gently compressed. {color_key} "
            "A filter with a tighter PSF retains a higher envelope as x approaches 1."
        )
        sanity_note = (" and should be near-identical for both filters — a useful sanity check."
                       if has_b else " — a useful sanity check.")
        band_mod_caption = (
            "Three charts — one per contrast row (high / medium / low) — each showing mean "
            "rolling local contrast (peak&minus;valley, ±5&thinsp;px window) grouped into "
            "four spatial-frequency bands. "
            "The <strong>Fine</strong> band (1&ndash;40&thinsp;px bar period) is the most "
            "sensitive to PSF width: a broader PSF smears the finest bars first, so a larger "
            "drop from <em>Original</em> to the filter columns here indicates a resolution "
            "penalty at high spatial frequencies. "
            f"The <strong>Coarse</strong> band (301+&thinsp;px) is largely PSF-insensitive"
            f"{sanity_note} "
            "Comparing per-level charts reveals whether PSF blur degrades high-contrast "
            "detail more than low-contrast nebulosity, giving a frequency-resolved "
            "contrast-retention profile directly analogous to the MTF curves above."
        )
        retention_caption_b = (f"; solid bars = {sim['label_a']}, hatched bars = {sim['label_b']}."
                                if has_b else ".")

        xs_block = ""
        if xs_figs:
            level_names = ["High contrast", "Medium contrast", "Low contrast"]
            xs_imgs = "\n".join(
                f'<h5 style="margin:1em 0 0.3em;">{level_names[i]}</h5>'
                f'{_img_tag(fig, level_names[i])}'
                for i, fig in enumerate(xs_figs)
            )
            xs_block = f"""
<h4>Horizontal cross-sections — test chart contrast bars</h4>
<p>Each figure samples a horizontal strip through Block 1 of the test chart at the indicated
pixel row. The bar pattern cycles from coarse to fine across the strip, making it a direct
probe of PSF resolution at multiple spatial frequencies in a single exposure.</p>
{xs_imgs}
<p class="caption">{xs_caption}</p>
<h4>Spatial-frequency-resolved contrast retention</h4>
{band_mod_fig}
<p class="caption">{band_mod_caption}</p>
<h4>Contrast retention ratios</h4>
{retention_fig}
<p class="caption">Fraction of original local contrast retained after convolution with each filter's ePSF
(convolved / original) per spatial-frequency band. A ratio of 1.0 indicates perfect contrast
retention; values below 1.0 indicate that the PSF reduces contrast at that spatial scale —
lower bars = more blurring. Error bars show the standard deviation of the per-pixel ratios within
each band. Colors identify contrast level (high / medium / low){retention_caption_b}</p>
{self._psf_retention_table(sim)}"""
            self._cached_retention_html = self._psf_retention_table(sim)

        diff_para = ("""<p>
  A pixel-level difference map (A &minus; B) is computed and displayed with the RdBu_r
  diverging colormap, centred at zero. Red regions indicate pixels where filter A produced
  higher intensity after convolution (A&rsquo;s PSF is locally tighter and preserved more
  contrast); blue regions indicate B is locally brighter. The horizontal cross-sections below
  isolate three contrast levels from Block&thinsp;1 of the chart and quantify the
  peak-to-valley swing preserved by each PSF.
</p>""" if has_b else "")

        return f"""
<h3>PSF Simulation — test chart convolved at native pixel resolution</h3>
<p style="border-left:4px solid #e6a817;background:#fdf6e3;padding:0.6em 0.9em;margin-bottom:1em;border-radius:3px;">
  <strong>Note on deconvolution tools:</strong> The ePSF and detail analysis in this section
  are derived from star shape. If you applied deconvolution software such as
  <em>BlurXTerminator</em>, <em>SyQon Parallax</em>, or similar tools, be aware that some of
  these modify star shape independently of how they sharpen diffuse structure like nebula detail.
  The convolved test-chart simulation below therefore reflects the measured star-based PSF and
  may not accurately represent the sharpness of extended nebula structure in your image.
  <strong>Section&nbsp;8 — Spatial Detail Comparison</strong> provides metrics that are
  independent of star shape and will give a more accurate picture of diffuse-detail contrast and
  resolution.
</p>
<p>
  <strong>How this simulation works:</strong> During PSF analysis, the empirical Point Spread
  Function (ePSF) is built by aligning and stacking the pixel profiles of the brightest, most
  isolated stars detected in each filter image (typically 5&ndash;30 stars, depending on field
  density and the minimum S/N threshold). The stacked profile is iteratively refined to
  sub-pixel accuracy using an oversampled 2&times; grid, producing a 2-D kernel that captures
  the combined blurring from the telescope optics, filter glass, atmospheric seeing, and sensor
  sampling for that specific image.
</p>
<p>
  Each ePSF is then down-sampled back to native pixel scale (dividing out the 2&times;
  oversampling) and normalised to unit sum so that total flux is conserved. A high-resolution
  ISO&thinsp;12233 resolution test chart (a standardised slanted-edge and bar-pattern target)
  is loaded as a reference scene and convolved with each kernel using FFT-based convolution
  (<code>scipy.signal.fftconvolve</code>, mode=&ldquo;same&rdquo;). The result simulates what
  the test chart <em>would look like</em> if imaged through that filter and telescope
  combination &mdash; that is, how much spatial detail the optical + atmospheric + filter
  system can resolve under the conditions that produced each image.
</p>
{diff_para}
<p>Each image is rendered at 1 image-pixel&thinsp;:&thinsp;1 screen-pixel.</p>
{panel(sim['original'], 'Original test chart')}
{panel(sim['conv_a'],   f"Convolved — {sim['label_a']}")}
{panel(sim['conv_b'],   f"Convolved — {sim['label_b']}") if has_b else ''}
{panel(sim['conv_ref'], f"Convolved — {sim['label_ref']}") if sim.get('conv_ref') is not None else ''}
{diff_panel}{xs_block}"""

    # ── Section 4: Halo ───────────────────────────────────────────────────────

    def _section_halo(self, ra: AnalysisResult, rb: AnalysisResult,
                       img_a: AstroImage, img_b: AstroImage | None) -> str:
        err = _error_box("halo", ra, rb)
        ha = ra.halo_metrics or {}
        hb = rb.halo_metrics or {}
        ca, cb = _better_worse_class(ha.get("halo_to_core_ratio"),
                                      hb.get("halo_to_core_ratio"),
                                      higher_is_better=False)
        prof_a = _img_tag((ha.get("figures") or {}).get("halo_profile"), f"Halo {ra.label}")
        prof_b = _img_tag((hb.get("figures") or {}).get("halo_profile"), f"Halo {rb.label}")
        matched = self._match_halo_stars(ra, rb)
        matched_sat = self._match_saturated_stars(ra, rb)
        sat_stars_a = [sa for sa, _sb in matched_sat]
        star_map_tag = _img_tag(self._plot_halo_star_map(matched, img_a,
                                                          saturated=sat_stars_a),
                                "Halo star field overview")
        grid_tag = _img_tag(self._plot_halo_star_grid(matched, img_a, img_b),
                            "Halo star comparison grid")
        sat_grid_tag = _img_tag(self._plot_saturated_star_grid(matched_sat, img_a, img_b),
                                "Saturated star cross-sections")

        rdf_unsat_fig = _img_tag(
            self._plot_rdf_comparison(
                ha, hb, ra.label, rb.label,
                "Aggregate RDF — unsaturated halo stars"),
            "Aggregate RDF unsaturated")

        sat_ha = {k.replace("sat_rdf_", "rdf_"): v
                  for k, v in ha.items() if k.startswith("sat_rdf_")}
        sat_hb = {k.replace("sat_rdf_", "rdf_"): v
                  for k, v in hb.items() if k.startswith("sat_rdf_")}
        rdf_sat_fig = _img_tag(
            self._plot_rdf_comparison(
                sat_ha, sat_hb, ra.label, rb.label,
                "Aggregate RDF — saturated stars"),
            "Aggregate RDF saturated")

        # Build dynamic optics note from FITS headers
        f_rat = _focal_ratio(img_a)
        pix_mm = _pixel_size_mm(img_a)
        t_mm = img_a.filter_thickness_mm
        if f_rat and pix_mm and f_rat > 0 and pix_mm > 0:
            r_expected = t_mm / (GLASS_REFRACTIVE_INDEX * f_rat * pix_mm)
            optics_note = _info_box(
                f'(f/{f_rat:.1f}, {pix_mm*1000:.2f} µm pixels, '
                f'{t_mm:.1f} mm filter thickness): '
                f'halo radius ≈ <strong>{r_expected:.0f} px</strong>. '
                f'The cutout windows in the grid below are sized to '
                f'2× this expected radius to ensure the full halo extent is visible.',
                title="Expected halo size for this telescope",
            )
        else:
            optics_note = ""

        _halo_causes_box = _info_box(
            '<strong>Focal ratio and halo size.</strong> '
            'Halos around bright stars in narrowband images arise from internal reflections '
            'within the filter substrate and its AR coatings. A fraction of the incoming '
            'light reflects off the back surface of the filter glass, travels back through '
            'the substrate, reflects off the front surface, and then exits &mdash; offset laterally '
            'from the direct beam. This offset is what appears as the circular glow surrounding '
            'bright stars.<br><br>'
            'The halo radius at the focal plane is approximately:<br>'
            '<code>R &asymp; t / (n &times; f_ratio &times; pixel_size)</code><br>'
            'where <em>t</em> is the filter substrate thickness, <em>n</em> &asymp; 1.9 (dichroic '
            'filter glass refractive index), and <em>pixel_size</em> is in mm. Because f-ratio appears in '
            'the denominator, <strong>faster telescopes (lower f-ratio) produce proportionally '
            'larger halos</strong> for the same filter. A narrowband filter that shows no '
            'visible halo on a slow f/10 refractor may produce a prominent halo on an f/4 '
            'Newtonian. This is a property of the optical system, not the filter quality alone. '
            'The halo-to-core <em>ratio</em> (amplitude of the halo relative to the star core) '
            'is a more filter-specific quality indicator than the raw halo size.',
            title="What causes halos?",
        )
        _halo_method_box = _info_box(
            f'<strong>Fitting method.</strong> '
            f'For each bright unsaturated star, the radial intensity profile (median-binned in '
            f'0.5&thinsp;px annuli out to {HALO_FIT_RADIUS_PX}&thinsp;px) is fitted in '
            f'log<sub>10</sub>&thinsp;space with a two-component Moffat model: '
            f'<code>I(r) = A<sub>core</sub>&thinsp;&middot;&thinsp;Moffat<sub>core</sub> '
            f'+ A<sub>halo</sub>&thinsp;&middot;&thinsp;Moffat<sub>halo</sub></code>. '
            f'Log-space fitting weights the wide dynamic range of the profile uniformly.<br><br>'
            f'<strong>Halo/core ratio</strong> = A<sub>halo</sub>&thinsp;/&thinsp;A<sub>core</sub>. '
            f'A value of 0 means no detectable halo; &gt;&thinsp;0.15 indicates significant internal '
            f'reflection. Because both amplitudes are normalised to the same star, the ratio is '
            f'independent of brightness and directly comparable between filters. '
            f'<strong>Halo radius</strong> is the HWHM of the halo component; if it exceeds the '
            f'{HALO_FIT_RADIUS_PX}&thinsp;px window it is shown as <em>N/A</em> and the ratio is '
            f'the more reliable indicator. Saturated stars are excluded from the fit but are shown '
            f'in the cross-sections and RDF plots.<br><br>'
            f'<strong>Radial Distribution Function (RDF) plots.</strong> '
            f'The background-subtracted image around each star is log<sub>10</sub>-transformed, '
            f'binned into 1-px concentric annuli, normalised to 1.0 at r&thinsp;=&thinsp;0, then '
            f'averaged across all fitted stars. The shaded band is ±1&sigma; within each annulus '
            f'(a wide band means the halo is asymmetric, not a perfect ring). '
            f'<strong>How to read it:</strong> a clean filter produces a smooth, monotonically '
            f'decreasing profile. A <em>shoulder</em> — where the curve levels off or decays '
            f'more slowly than the core trend at 10&ndash;60&thinsp;px — indicates halo light '
            f'from internal reflections. The filter with the lower profile at any radius produces '
            f'less scattered light at that distance from the star.<br><br>'
            f'<strong>Ideal values:</strong> halo/core ratio &lt;&thinsp;0.05 is excellent; '
            f'&gt;&thinsp;0.15 indicates significant internal reflection that will reduce contrast '
            f'on bright stars.',
            title="Measurement methodology",
        )

        return f"""
<h2>5. Halo Analysis &nbsp;<span class="metric-label-ok">&#10003; bandwidth-independent</span></h2>
{err}
{_halo_causes_box}
{optics_note}
{_halo_method_box}

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Stars fitted</td><td>{_val(ha.get("n_stars_fitted"), "d")}</td><td>{_val(hb.get("n_stars_fitted"), "d")}</td></tr>
  <tr><td>Halo / core ratio</td><td class="{ca}">{_val(ha.get("halo_to_core_ratio"), ".5f")}</td><td class="{cb}">{_val(hb.get("halo_to_core_ratio"), ".5f")}</td></tr>
  <tr><td>Halo radius (px)</td><td>{_val(ha.get("halo_radius_px"))}</td><td>{_val(hb.get("halo_radius_px"))}</td></tr>
</table>

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{prof_a}</div>
  <div style="flex:1;">{prof_b}</div>
</div>
<p class="caption">Radial profiles (semi-log). A steep drop-off indicates a clean
filter. A raised floor or shoulder beyond ~10 px indicates a halo component.</p>

{rdf_unsat_fig}
<p class="caption">Aggregate RDF — unsaturated halo stars. Mean normalised intensity vs. radius
(blue = Image A, red = Image B). Shaded band = ±1&sigma; within-annulus variability averaged
over all fitted stars. A shoulder or elevated tail relative to the other filter indicates
stronger halo contribution at that radius.</p>

{rdf_sat_fig}
<p class="caption">Aggregate RDF — saturated stars only. The flat plateau at small radii
reflects the clipped core; the slope beyond the plateau shows the halo ring structure.
Meaningful even when the core is fully overexposed.</p>

{star_map_tag}
<p class="caption">STF-stretched overview of {ra.label}.
<span style="color:red"><strong>Red circles</strong></span> mark the top {len(matched)}
brightest unsaturated halo stars (rank matches the cutout grid below).
<span style="color:magenta"><strong>Magenta dashed circles</strong></span> mark the top {len(matched_sat)}
brightest saturated stars (S1–S{len(matched_sat)}) shown in the saturated star grid below.</p>

{grid_tag}
<p class="caption">Top {len(matched)} brightest stars common to both images, side-by-side
(Image A left, Image B right per pair). STF stretch applied per image to reveal
faint halo structure. Stars ranked by peak brightness (brightest first).
<em>Turbo</em> colormap: bright = high intensity.</p>

{sat_grid_tag}
{"" if not matched_sat else '<p class="caption">Brightest saturated stars (core overexposed — halo/core ratio not computed). The cross-section shows the halo ring structure in the wings beyond the saturated core. Comparing the ring width and intensity between the two filters is still meaningful even when the core is clipped.</p>'}
"""

    def _extract_cutout(self, data: np.ndarray,
                         xc: float, yc: float, half: int) -> np.ndarray:
        h, w = data.shape
        size = 2 * half + 1
        cut = np.zeros((size, size), dtype=np.float64)
        x0_src = max(0, int(xc) - half)
        x1_src = min(w, int(xc) + half + 1)
        y0_src = max(0, int(yc) - half)
        y1_src = min(h, int(yc) + half + 1)
        x0_dst = x0_src - (int(xc) - half)
        x1_dst = x0_dst + (x1_src - x0_src)
        y0_dst = y0_src - (int(yc) - half)
        y1_dst = y0_dst + (y1_src - y0_src)
        cut[y0_dst:y1_dst, x0_dst:x1_dst] = data[y0_src:y1_src, x0_src:x1_src]
        return cut

    @staticmethod
    def _match_halo_stars(ra: AnalysisResult, rb: AnalysisResult) -> list:
        """Return up to 10 (sa, sb) pairs ranked by image-A peak brightness."""
        stars_a = (ra.halo_metrics or {}).get("star_data", [])
        stars_b = (rb.halo_metrics or {}).get("star_data", [])
        if not stars_a:
            return []
        if stars_b:
            xs_b = np.array([s["xc"] for s in stars_b])
            ys_b = np.array([s["yc"] for s in stars_b])
            all_matched = []
            for sa in stars_a:
                dists = np.sqrt((xs_b - sa["xc"]) ** 2 + (ys_b - sa["yc"]) ** 2)
                idx = int(np.argmin(dists))
                if dists[idx] <= 20.0:
                    all_matched.append((sa, stars_b[idx]))
            all_matched.sort(key=lambda pair: pair[0].get("peak", 0.0), reverse=True)
            return all_matched[:10]
        else:
            sorted_a = sorted(stars_a, key=lambda s: s.get("peak", 0.0), reverse=True)
            return [(sa, None) for sa in sorted_a[:10]]

    @staticmethod
    def _match_saturated_stars(ra: AnalysisResult, rb: AnalysisResult) -> list:
        """Return up to 10 (sa, sb) pairs of saturated stars, sorted by image-A peak."""
        stars_a = (ra.halo_metrics or {}).get("saturated_star_data", [])
        stars_b = (rb.halo_metrics or {}).get("saturated_star_data", [])
        if not stars_a:
            return []
        if stars_b:
            xs_b = np.array([s["xc"] for s in stars_b])
            ys_b = np.array([s["yc"] for s in stars_b])
            matched = []
            for sa in stars_a:
                dists = np.sqrt((xs_b - sa["xc"]) ** 2 + (ys_b - sa["yc"]) ** 2)
                idx = int(np.argmin(dists))
                if dists[idx] <= 20.0:
                    matched.append((sa, stars_b[idx]))
            matched.sort(key=lambda p: p[0].get("peak", 0.0), reverse=True)
            return matched[:10]
        else:
            return [(sa, None) for sa in stars_a[:10]]

    def _plot_halo_star_map(self, matched: list, img_a: AstroImage,
                             saturated: list = ()) -> plt.Figure | None:
        """Full-field STF-stretched overview with top-N halo stars circled and ranked."""
        if not matched:
            return None

        bgsub = img_a.background_subtracted() if img_a.background is not None else img_a.data
        from core.stretch import stf_stretch
        display = stf_stretch(bgsub).astype(np.float64)

        h, w = display.shape
        if max(h, w) > 1200:
            zoom_f = 1200.0 / max(h, w)
            display = _ndimage_zoom(display, zoom_f, order=1)
            scale = zoom_f
        else:
            scale = 1.0
        dh, dw = display.shape

        fig_w = 10.0 * (dw / max(dh, dw))
        fig_h = 10.0 * (dh / max(dh, dw))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(display, origin="upper", cmap="gray", aspect="equal",
                  interpolation="nearest", vmin=0, vmax=1)

        circle_r = max(dw, dh) * 0.012
        font_size = max(8, int(circle_r * 0.8))

        for rank, (sa, _sb) in enumerate(matched, start=1):
            xd = sa["xc"] * scale
            yd = sa["yc"] * scale
            circ = plt.Circle((xd, yd), circle_r, color="red",
                               fill=False, linewidth=1.2)
            ax.add_patch(circ)
            ax.text(xd + circle_r * 1.4, yd + circle_r * 1.4,
                    str(rank), color="red", fontsize=font_size,
                    fontweight="bold", ha="left", va="bottom",
                    clip_on=True)

        for i, sa in enumerate(saturated, start=1):
            xd = sa["xc"] * scale
            yd = sa["yc"] * scale
            circ = plt.Circle((xd, yd), circle_r, color="magenta",
                               fill=False, linewidth=1.2, linestyle="--")
            ax.add_patch(circ)
            ax.text(xd + circle_r * 1.4, yd + circle_r * 1.4,
                    f"S{i}", color="magenta", fontsize=font_size,
                    fontweight="bold", ha="left", va="bottom",
                    clip_on=True)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="none", edgecolor="red", label="Unsaturated halo stars"),
            Patch(facecolor="none", edgecolor="magenta", linestyle="--",
                  label="Saturated stars"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=font_size,
                  framealpha=0.6, facecolor="black", labelcolor="white")

        n_sat = len(saturated)
        title = f"{img_a.label} — top {len(matched)} brightest halo stars"
        if n_sat:
            title += f" + {n_sat} saturated"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        return fig

    def _plot_halo_star_grid(self, matched: list,
                              img_a: AstroImage, img_b: AstroImage | None) -> plt.Figure | None:
        if not matched:
            return None

        bgsub_a = img_a.background_subtracted() if img_a.background is not None else img_a.data
        bgsub_b = (img_b.background_subtracted() if img_b is not None and img_b.background is not None
                   else (img_b.data if img_b is not None else None))
        label_b = img_b.label if img_b is not None else "—"

        from core.stretch import stf_stretch

        # Compute optics-based cutout size once from image headers
        f_rat = _focal_ratio(img_a)
        pix_mm = _pixel_size_mm(img_a)
        t_mm = img_a.filter_thickness_mm
        if f_rat and pix_mm and f_rat > 0 and pix_mm > 0:
            optics_half = int(t_mm / (GLASS_REFRACTIVE_INDEX * f_rat * pix_mm))
        else:
            optics_half = HALO_FIT_RADIUS_PX

        n = len(matched)
        pairs_per_row = 1
        cols_per_pair = 4   # img A | img B | cross-section | RDF
        n_rows = (n + pairs_per_row - 1) // pairs_per_row
        n_cols = pairs_per_row * cols_per_pair

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 3.5, n_rows * 3.5))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        for ax in axes.flat:
            ax.axis("off")

        for idx, (sa, sb) in enumerate(matched):
            row = idx // pairs_per_row
            col_base = (idx % pairs_per_row) * cols_per_pair

            r_a = sa.get("halo_radius_px") or optics_half
            r_b = (sb.get("halo_radius_px") if sb else r_a) or optics_half
            # 1.5× shows out to well past the halo half-power point; cap at 200 px
            # to guard against unreliable extrapolated radii from sparse data
            half = min(max(int(max(r_a, r_b) * 1.5), optics_half), 200)

            cut_a = self._extract_cutout(bgsub_a, sa["xc"], sa["yc"], half)
            cut_b = (self._extract_cutout(bgsub_b, sb["xc"], sb["yc"], half)
                     if sb is not None else np.zeros_like(cut_a))

            # shared_max retained for cross-section noise floor
            peak_a = float(np.percentile(cut_a, 99.9)) if cut_a.size > 0 else 1.0
            peak_b = float(np.percentile(cut_b, 99.9)) if cut_b.size > 0 else 1.0
            shared_max = max(peak_a, peak_b, 1e-9)

            disp_a = stf_stretch(cut_a)
            disp_b = stf_stretch(cut_b)

            ax_a   = axes[row, col_base]
            ax_b   = axes[row, col_base + 1]
            ax_xs  = axes[row, col_base + 2]
            ax_rdf = axes[row, col_base + 3]

            ax_a.imshow(disp_a, origin="upper", cmap="turbo",
                        vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            h2c_a = sa.get("halo_to_core_ratio")
            ax_a.set_title(f"#{idx+1} {img_a.label}"
                           + (f"\nh/c={h2c_a:.5f}" if h2c_a is not None else ""),
                           fontsize=9)
            ax_a.axis("off")
            ax_a.plot(half, half, '+', color='magenta', markersize=12,
                      markeredgewidth=1.5, zorder=5)

            ax_b.imshow(disp_b, origin="upper", cmap="turbo",
                        vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            if sb is not None:
                h2c_b = sb.get("halo_to_core_ratio")
                ax_b.set_title(f"#{idx+1} {label_b}"
                               + (f"\nh/c={h2c_b:.5f}" if h2c_b is not None else ""),
                               fontsize=9)
            else:
                ax_b.set_title(f"#{idx+1} {label_b}\n(no match)", fontsize=9)
            ax_b.axis("off")
            if sb is not None:
                ax_b.plot(half, half, '+', color='magenta', markersize=12,
                          markeredgewidth=1.5, zorder=5)

            # Horizontal cross-section through the star centre — log y-axis
            if cut_a.shape[0] > 0 and cut_a.shape[1] > 0:
                mid_row = cut_a.shape[0] // 2
                # Stars near image edges produce differently-clipped cutouts; trim to common width
                w_min = (min(cut_a.shape[1], cut_b.shape[1]) if sb is not None
                         else cut_a.shape[1])
                px_offset = np.arange(w_min) - w_min // 2
                noise_floor = shared_max * 1e-4
                xs_a = np.maximum(cut_a[mid_row, :w_min], noise_floor)
                xs_b_vals = (np.maximum(cut_b[mid_row, :w_min], noise_floor)
                             if sb is not None else None)
                ax_xs.semilogy(px_offset, xs_a, color="steelblue",
                               linewidth=1.0, alpha=XS_LINE_ALPHA, label=img_a.label)
                if xs_b_vals is not None:
                    ax_xs.semilogy(px_offset, xs_b_vals, color="tomato",
                                   linewidth=1.0, alpha=XS_LINE_ALPHA, label=label_b)
                ax_xs.set_title(f"#{idx+1} cross-section", fontsize=8)
                ax_xs.set_xlabel("px from centre", fontsize=7)
                ax_xs.tick_params(labelsize=7)
                ax_xs.legend(fontsize=7)
                ax_xs.grid(True, alpha=0.25, which="both")
                ax_xs.axis("on")
                ax_xs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))

            # Per-star RDF (log10-space stats, inverse-transformed for display)
            rdf_r_a = sa.get("rdf_radii")
            rdf_m_a = sa.get("rdf_mean")
            rdf_s_a = sa.get("rdf_std")
            rdf_r_b = sb.get("rdf_radii") if sb else None
            rdf_m_b = sb.get("rdf_mean")  if sb else None
            rdf_s_b = sb.get("rdf_std")   if sb else None
            if rdf_m_a is not None:
                ax_rdf.semilogy(rdf_r_a, 10**rdf_m_a, color="steelblue",
                                linewidth=1.0, label=img_a.label)
                ax_rdf.fill_between(rdf_r_a,
                                    10**(rdf_m_a - rdf_s_a),
                                    10**(rdf_m_a + rdf_s_a),
                                    alpha=0.25, color="steelblue")
            if rdf_m_b is not None:
                ax_rdf.semilogy(rdf_r_b, 10**rdf_m_b, color="tomato",
                                linewidth=1.0, label=label_b)
                ax_rdf.fill_between(rdf_r_b,
                                    10**(rdf_m_b - rdf_s_b),
                                    10**(rdf_m_b + rdf_s_b),
                                    alpha=0.25, color="tomato")
            if rdf_m_a is not None or rdf_m_b is not None:
                ax_rdf.set_title(f"#{idx+1} RDF", fontsize=8)
                ax_rdf.set_xlabel("px from centre", fontsize=7)
                ax_rdf.tick_params(labelsize=7)
                ax_rdf.legend(fontsize=6)
                ax_rdf.grid(True, alpha=0.25, which="both")
                ax_rdf.axis("on")
                ax_rdf.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))

        fig.tight_layout()
        return fig

    def _plot_saturated_star_grid(self, matched: list,
                                   img_a: AstroImage, img_b: AstroImage | None) -> plt.Figure | None:
        """Cutout + cross-section grid for saturated bright stars (no Moffat fit)."""
        if not matched:
            return None

        bgsub_a = img_a.background_subtracted() if img_a.background is not None else img_a.data
        bgsub_b = (img_b.background_subtracted() if img_b is not None and img_b.background is not None
                   else (img_b.data if img_b is not None else None))
        label_b = img_b.label if img_b is not None else "—"

        from core.stretch import stf_stretch

        f_rat = _focal_ratio(img_a)
        pix_mm = _pixel_size_mm(img_a)
        t_mm = img_a.filter_thickness_mm
        if f_rat and pix_mm and f_rat > 0 and pix_mm > 0:
            half = int(t_mm / (GLASS_REFRACTIVE_INDEX * f_rat * pix_mm))
        else:
            half = HALO_FIT_RADIUS_PX
        half = max(half, HALO_FIT_RADIUS_PX)   # at least HALO_FIT_RADIUS_PX to show the ring

        n = len(matched)
        n_rows = n
        n_cols = 4   # img A | img B | cross-section | RDF

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        for ax in axes.flat:
            ax.axis("off")

        for idx, (sa, sb) in enumerate(matched):
            cut_a = self._extract_cutout(bgsub_a, sa["xc"], sa["yc"], half)
            cut_b = (self._extract_cutout(bgsub_b, sb["xc"], sb["yc"], half)
                     if sb is not None else np.zeros_like(cut_a))

            peak_a = float(np.percentile(cut_a, 99.9)) if cut_a.size > 0 else 1.0
            peak_b = float(np.percentile(cut_b, 99.9)) if cut_b.size > 0 else 1.0
            shared_max = max(peak_a, peak_b, 1e-9)

            disp_a = stf_stretch(cut_a)
            disp_b = stf_stretch(cut_b)

            ax_a   = axes[idx, 0]
            ax_b   = axes[idx, 1]
            ax_xs  = axes[idx, 2]
            ax_rdf = axes[idx, 3]

            ax_a.imshow(disp_a, origin="upper", cmap="turbo",
                        vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            ax_a.set_title(f"S{idx+1} {img_a.label}\n⚠ saturated core", fontsize=9)
            ax_a.axis("off")
            ax_a.plot(half, half, '+', color='magenta', markersize=12,
                      markeredgewidth=1.5, zorder=5)

            ax_b.imshow(disp_b, origin="upper", cmap="turbo",
                        vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            title_b = (f"S{idx+1} {label_b}\n"
                       + ("⚠ saturated core" if sb is not None else "(no match)"))
            ax_b.set_title(title_b, fontsize=9)
            ax_b.axis("off")
            if sb is not None:
                ax_b.plot(half, half, '+', color='magenta', markersize=12,
                          markeredgewidth=1.5, zorder=5)

            if cut_a.shape[0] > 0:
                mid_row = cut_a.shape[0] // 2
                w_min = (min(cut_a.shape[1], cut_b.shape[1]) if sb is not None
                         else cut_a.shape[1])
                px_offset = np.arange(w_min) - w_min // 2
                noise_floor = shared_max * 1e-4
                xs_a = np.maximum(cut_a[mid_row, :w_min], noise_floor)
                ax_xs.semilogy(px_offset, xs_a, color="steelblue",
                               linewidth=1.0, alpha=XS_LINE_ALPHA, label=img_a.label)
                if sb is not None:
                    xs_b_vals = np.maximum(cut_b[mid_row, :w_min], noise_floor)
                    ax_xs.semilogy(px_offset, xs_b_vals, color="tomato",
                                   linewidth=1.0, alpha=XS_LINE_ALPHA, label=label_b)
                ax_xs.set_title(f"S{idx+1} cross-section", fontsize=8)
                ax_xs.set_xlabel("px from centre", fontsize=7)
                ax_xs.tick_params(labelsize=7)
                ax_xs.legend(fontsize=7)
                ax_xs.grid(True, alpha=0.25, which="both")
                ax_xs.axis("on")
                ax_xs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))

            # Per-star RDF (log10-space stats, inverse-transformed for display)
            rdf_r_a = sa.get("rdf_radii")
            rdf_m_a = sa.get("rdf_mean")
            rdf_s_a = sa.get("rdf_std")
            rdf_r_b = sb.get("rdf_radii") if sb else None
            rdf_m_b = sb.get("rdf_mean")  if sb else None
            rdf_s_b = sb.get("rdf_std")   if sb else None
            if rdf_m_a is not None:
                ax_rdf.semilogy(rdf_r_a, 10**rdf_m_a, color="steelblue",
                                linewidth=1.0, label=img_a.label)
                ax_rdf.fill_between(rdf_r_a,
                                    10**(rdf_m_a - rdf_s_a),
                                    10**(rdf_m_a + rdf_s_a),
                                    alpha=0.25, color="steelblue")
            if rdf_m_b is not None:
                ax_rdf.semilogy(rdf_r_b, 10**rdf_m_b, color="tomato",
                                linewidth=1.0, label=label_b)
                ax_rdf.fill_between(rdf_r_b,
                                    10**(rdf_m_b - rdf_s_b),
                                    10**(rdf_m_b + rdf_s_b),
                                    alpha=0.25, color="tomato")
            if rdf_m_a is not None or rdf_m_b is not None:
                ax_rdf.set_title(f"S{idx+1} RDF", fontsize=8)
                ax_rdf.set_xlabel("px from centre", fontsize=7)
                ax_rdf.tick_params(labelsize=7)
                ax_rdf.legend(fontsize=6)
                ax_rdf.grid(True, alpha=0.25, which="both")
                ax_rdf.axis("on")
                ax_rdf.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))

        fig.tight_layout()
        return fig

    def _plot_rdf_comparison(self, ha: dict, hb: dict,
                              label_a: str, label_b: str,
                              title: str) -> "plt.Figure | None":
        """Overlay Image A (steelblue) and Image B (tomato) aggregate RDFs with ±1σ bands."""
        r_a = ha.get("rdf_radii")
        m_a = ha.get("rdf_mean")
        s_a = ha.get("rdf_std")
        r_b = hb.get("rdf_radii")
        m_b = hb.get("rdf_mean")
        s_b = hb.get("rdf_std")

        if m_a is None and m_b is None:
            return None

        fig, ax = plt.subplots(figsize=(7, 4))
        if m_a is not None:
            ax.semilogy(r_a, 10**m_a, color="steelblue", linewidth=1.8, label=label_a)
            ax.fill_between(r_a,
                            10**(m_a - s_a), 10**(m_a + s_a),
                            alpha=0.25, color="steelblue")
        if m_b is not None:
            ax.semilogy(r_b, 10**m_b, color="tomato", linewidth=1.8, label=label_b)
            ax.fill_between(r_b,
                            10**(m_b - s_b), 10**(m_b + s_b),
                            alpha=0.25, color="tomato")
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Normalised mean intensity")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))
        fig.tight_layout()
        return fig


    # ── Section 6: Edge ───────────────────────────────────────────────────────

    def _plot_esf_lsf_pair(self,
                            edge_a: dict, edge_b: dict,
                            label_a: str, label_b: str,
                            edge_num: int) -> plt.Figure:
        """Combined ESF (left) + LSF (right) for both images on shared axes."""
        _v = lambda d, k: np.asarray(d[k]) if d.get(k) is not None else np.array([])
        pos_a = _v(edge_a, "positions")
        esf_a = _v(edge_a, "esf")
        lsf_a = _v(edge_a, "lsf")
        pos_b = _v(edge_b, "positions")
        esf_b = _v(edge_b, "esf")
        lsf_b = _v(edge_b, "lsf")
        w_a   = edge_a.get("edge_width_10_90_px")
        w_b   = edge_b.get("edge_width_10_90_px")

        fig, (ax_esf, ax_lsf) = plt.subplots(1, 2, figsize=(10, 4))

        # ESF panel
        if esf_a.size:
            ax_esf.plot(pos_a, esf_a, color="steelblue", linewidth=1.5,
                        alpha=XS_LINE_ALPHA, label=label_a)
        if esf_b.size:
            ax_esf.plot(pos_b, esf_b, color="tomato", linewidth=1.5,
                        alpha=XS_LINE_ALPHA, label=label_b)
        ax_esf.axhline(0.10, color="gray", linestyle="--", linewidth=0.8)
        ax_esf.axhline(0.90, color="gray", linestyle="--", linewidth=0.8)
        w_label = ""
        if w_a is not None and w_b is not None:
            w_label = f"  (A: {w_a:.2f} px, B: {w_b:.2f} px)"
        elif w_a is not None:
            w_label = f"  (A: {w_a:.2f} px)"
        elif w_b is not None:
            w_label = f"  (B: {w_b:.2f} px)"
        ax_esf.set_title(f"Edge #{edge_num} ESF — 10–90% width{w_label}", fontsize=9)
        ax_esf.set_xlabel("Position (px)")
        ax_esf.set_ylabel("Normalised intensity")
        ax_esf.legend(fontsize=8)
        ax_esf.grid(alpha=0.3)

        # LSF panel
        if lsf_a.size:
            ax_lsf.plot(pos_a, lsf_a, color="steelblue", linewidth=1.5,
                        alpha=XS_LINE_ALPHA, label=label_a)
        if lsf_b.size:
            ax_lsf.plot(pos_b, lsf_b, color="tomato", linewidth=1.5,
                        alpha=XS_LINE_ALPHA, label=label_b)
        ax_lsf.set_title(f"Edge #{edge_num} LSF (derivative of ESF)", fontsize=9)
        ax_lsf.set_xlabel("Position (px)")
        ax_lsf.set_ylabel("d(ESF)/dx")
        ax_lsf.legend(fontsize=8)
        ax_lsf.grid(alpha=0.3)

        fig.tight_layout()
        return fig

    def _plot_gradient_pair(self, disp_a, shape_a, rois_a, label_a,
                             disp_b, shape_b, rois_b, label_b,
                             shared_vmax: float) -> plt.Figure:
        """Render both gradient-magnitude maps side-by-side with a shared color scale."""
        from matplotlib.patches import Rectangle
        panels = [(d, s, r, lbl) for d, s, r, lbl in
                  [(disp_a, shape_a, rois_a, label_a),
                   (disp_b, shape_b, rois_b, label_b)] if d is not None]
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]
        for ax, (disp, shape, rois, lbl) in zip(axes, panels):
            h, w = shape if shape else disp.shape
            ax.imshow(disp, origin="upper", cmap="inferno",
                      vmin=0, vmax=shared_vmax,
                      extent=[0, w, h, 0], aspect="equal", interpolation="nearest")
            half = EDGE_ROI_MAP_INDICATOR_PX // 2
            for (x0, y0, x1, y1) in (rois or []):
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                ix0, iy0 = max(0, cx - half), max(0, cy - half)
                ix1, iy1 = min(w, cx + half), min(h, cy + half)
                ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0,
                                       linewidth=2.0, edgecolor="cyan",
                                       facecolor="none", linestyle="-"))
            ax.set_title(f"Gradient magnitude — {lbl}")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
        fig.tight_layout()
        return fig

    def _section_edge(self, ra: AnalysisResult, rb: AnalysisResult,
                       bw_differ: bool) -> str:
        err = _error_box("edge", ra, rb)
        ea = ra.edge_metrics or {}
        eb = rb.edge_metrics or {}
        ca, cb = _better_worse_class(ea.get("edge_width_10_90_px"),
                                      eb.get("edge_width_10_90_px"),
                                      higher_is_better=False)
        ecr_warn = (' &nbsp;<span class="metric-label-warn">⚠ bandwidth-sensitive</span>'
                    if bw_differ else "")

        # Build per-edge figure rows (one pair per detected gradient peak)
        edges_a = ea.get("edges") or []
        edges_b = eb.get("edges") or []
        n_edges = max(len(edges_a), len(edges_b))
        edge_figures_html = ""
        for i in range(n_edges):
            ea_e = edges_a[i] if i < len(edges_a) else {}
            eb_e = edges_b[i] if i < len(edges_b) else {}
            # ROI context images (one per image — just the context window with overlays)
            img_a_i = _img_tag((ea_e.get("figures") or {}).get("edge"),
                               f"Edge #{i+1} {ra.label}")
            img_b_i = _img_tag((eb_e.get("figures") or {}).get("edge"),
                               f"Edge #{i+1} {rb.label}")
            if not img_a_i and not img_b_i:
                continue
            # Combined ESF + LSF figure — both images on shared axes
            esf_lsf_html = ""
            if ea_e or eb_e:
                esf_lsf_fig = self._plot_esf_lsf_pair(ea_e, eb_e, ra.label, rb.label, i + 1)
                esf_lsf_html = _img_tag(esf_lsf_fig, f"Edge #{i+1} ESF + LSF")
            edge_figures_html += (
                f'<h4 style="margin-top:1.2em;">Edge #{i+1}</h4>'
                f'<div style="display:flex;gap:10px;">'
                + "".join(f'<div style="flex:1;">{img}</div>'
                          for img in [img_a_i, img_b_i] if img)
                + "</div>"
                + esf_lsf_html
            )

        used_sl_a = ea.get("used_starless", False)
        used_sl_b = eb.get("used_starless", False)
        sl_note = ""
        if used_sl_a or used_sl_b:
            who = ", ".join(filter(None, [ra.label if used_sl_a else "",
                                          rb.label if used_sl_b else ""]))
            sl_note = _info_box(
                f'★ Edge analysis for <strong>{who}</strong> '
                f'used the starless image so the strongest gradient search locates '
                f'a nebula emission boundary rather than a star profile.',
                title="Starless image used",
                open=True,
            )

        # Gradient map comparison row — shared scale, rendered as a combined figure
        gm_a = ea.get("gm_display")
        gm_b = eb.get("gm_display")
        if gm_a is not None or gm_b is not None:
            shared_vmax = max(ea.get("gm_vmax") or 1.0,
                              eb.get("gm_vmax") or 1.0)
            pair_fig = self._plot_gradient_pair(
                gm_a, ea.get("gm_shape"), ea.get("rois_used"), ra.label,
                gm_b, eb.get("gm_shape"), eb.get("rois_used"), rb.label,
                shared_vmax,
            )
            gradient_row = (
                '<h4>Gradient magnitude (ROI auto-detection map)</h4>'
                + _img_tag(pair_fig, "Gradient magnitude")
                + '<p class="caption">Gaussian gradient magnitude used to locate '
                'the strongest edge regions. Cyan boxes show the three selected '
                'analysis ROI regions. Both images share the same color scale (P99 of the brighter '
                'image) for direct comparison. Sigma is pixel-scale adaptive '
                '(≈ 1.5 arcsec equivalent) so diffuse gradients in long-focal-length images '
                'are captured as reliably as sharp edges in short-focal-length data.</p>'
            )
        else:
            gradient_row = ""

        return f"""
<h2>6. Local Contrast / Edge Analysis</h2>
{err}
{sl_note}
{_info_box(
  '<strong>Edge Spread Function (ESF)</strong> — A 1-D intensity profile sampled '
  'perpendicular to the detected edge, averaged across the full height of the ROI '
  'after rotating so the edge runs vertically. An ideal ESF is a smooth sigmoid: '
  'the steeper the transition, the better the local contrast and resolution. '
  'Normalised to [0, 1], the ESF shape is <strong>bandwidth-independent ✓</strong> '
  'and directly comparable between filters.<br><br>'
  '<strong>Line Spread Function (LSF)</strong> — The derivative of the ESF, '
  'computed with a Savitzky-Golay filter (cubic polynomial, window ≈ 18% of the '
  'ESF length, typically 11 points). SG fitting smooths sample-to-sample noise '
  'while preserving the height and width of narrow peaks better than a simple '
  'finite-difference derivative or Gaussian smoothing, making it the standard '
  'method for ESF differentiation in optical MTF analysis (ISO 12233). Ideally a '
  'narrow, symmetric peak centred on the edge. A broader LSF peak indicates softer '
  'resolution; asymmetry or secondary lobes can indicate optical aberrations, '
  'atmospheric dispersion, or poor focus stability during the integration.<br><br>'
  '<strong>10–90% edge width</strong> — The pixel (or arcsec) distance between '
  'the 10% and 90% intensity points on the ESF. Smaller values indicate a '
  'sharper, better-resolved edge. Use the arcsec figure for cross-image comparison '
  'if the pixel scales differ.<br><br>'
  'The <strong>edge contrast ratio</strong> (bright-side / dark-side mean signal) '
  'is <strong>bandwidth-sensitive ⚠</strong>: a narrower filter rejects more '
  'continuum background, which can raise this ratio independently of optical quality.',
  title="Edge Spread Function (ESF) &amp; Line Spread Function (LSF)")}

{_info_box(
  'The analysis applies an STF stretch to the background-subtracted image to bring '
  'faint emission boundaries into relief, then computes a <strong>pixel-scale-adaptive '
  'Gaussian gradient magnitude</strong> (sigma ≈ 1.5 arcsec, capped 1–8 px) across '
  'the whole frame. Using a Gaussian gradient rather than a fixed 3×3 Sobel kernel '
  'means that diffuse gradients in long-focal-length images are detected as reliably '
  'as sharp edges in short-focal-length data. '
  'The <strong>three strongest, well-separated gradient peaks</strong> '
  'are located automatically (peaks are suppressed within a 90 px radius after each '
  'detection to ensure the three regions sample distinct features). A 500 × 500 px '
  'context window is shown for each, centred on the gradient peak; the 60 × 60 px '
  'analysis region (cyan box) is highlighted within it. '
  'If a starless image was provided it is used in place of the stacked image, so the '
  'search locates nebula emission boundaries rather than star profiles. '
  'Both images are measured over the <em>identical</em> pixel regions: Image A\'s '
  'detected ROI coordinates are reused for Image B after alignment. The '
  '<strong>ESF scan direction is taken from whichever image has the stronger overall '
  'gradient</strong>, then applied to both, so the two ESF curves always sample the '
  'same cross-section orientation and are directly comparable. '
  'The table below shows metrics from the strongest of the three edges; individual '
  'per-edge figures follow.',
  title="How the edge regions were selected")}
{gradient_row}

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Edge width 10–90% (px) ✓</td><td class="{ca}">{_val(ea.get("edge_width_10_90_px"))}</td><td class="{cb}">{_val(eb.get("edge_width_10_90_px"))}</td></tr>
  <tr><td>Edge width 10–90% (arcsec) ✓</td><td>{_val(ea.get("edge_width_10_90_arcsec"))}</td><td>{_val(eb.get("edge_width_10_90_arcsec"))}</td></tr>
  <tr><td>Edge contrast ratio{ecr_warn}</td><td>{_val(ea.get("edge_contrast_ratio"))}</td><td>{_val(eb.get("edge_contrast_ratio"))}</td></tr>
  <tr><td>Gradient magnitude</td><td>{_val(ea.get("gradient_magnitude"), ".2f")}</td><td>{_val(eb.get("gradient_magnitude"), ".2f")}</td></tr>
</table>

{edge_figures_html}
{_info_box(
  '<em>ROI context panels</em> (one per image) — 500 × 500 px context window centred on '
  'the detected gradient peak. '
  'The <span style="color:#90ee90;font-weight:bold;">dashed lime rectangle</span> marks '
  'the 60 × 60 px analysis region used for ESF/LSF extraction. '
  'The <span style="color:#00bcd4;font-weight:bold;">cyan line</span> shows the ESF '
  'scan direction (perpendicular to the edge), clipped to the analysis region; '
  'the <span style="color:#c8b400;font-weight:bold;">yellow dashed line</span> shows '
  'the detected edge orientation, also clipped to the analysis region. &nbsp; '
  '<em>ESF / LSF comparison</em> — both images overlaid on shared axes; '
  '<span style="color:steelblue;font-weight:bold;">Image A (steelblue)</span> vs '
  '<span style="color:tomato;font-weight:bold;">Image B (tomato)</span>. '
  'ESF: normalised intensity transition; dashed lines mark the 10% and 90% levels used '
  'for the edge width measurement. '
  'LSF: derivative of the ESF; peak width and symmetry indicate local resolution quality.',
  title="Figures per edge", style="font-size:0.9em;")}

{_info_box(
  '<ul style="margin:0.4em 0 0 1.2em;padding:0;">'
  '<li><strong>Edge width (arcsec)</strong> is the primary comparator — it is '
  'scale-independent. Prefer the arcsec figure when the two images have '
  'different pixel scales.</li>'
  '<li>A difference of less than ~10% in edge width is typically within '
  'measurement uncertainty for a single edge sample; larger differences '
  'are likely real.</li>'
  '<li>A <strong>broader LSF peak</strong> in one image suggests lower resolution '
  'at the edge spatial frequency. Common causes: worse seeing during that '
  'integration, softer focus, or greater atmospheric dispersion from a filter '
  'with a very wide bandpass.</li>'
  '<li>An <strong>asymmetric or multi-lobed LSF</strong> can indicate optical '
  'aberrations, trailing, or non-uniform atmospheric refraction.</li>'
  '<li>If edge widths are similar but <strong>gradient magnitude</strong> differs '
  'substantially, the difference is likely signal level or background contrast '
  'rather than resolution — gradient magnitude is intensity-dependent and '
  'should not be used alone to rank image quality.</li>'
  '<li>The <strong>edge contrast ratio</strong> is only directly comparable between '
  'images of identical bandwidth. A narrower filter naturally yields a higher '
  'ratio by suppressing continuum background.</li>'
  '</ul>',
  title="Interpreting the comparison")}"""

    def _plot_radial_overlay(self, ra: AnalysisResult, rb: AnalysisResult,
                              star_pa: dict | None = None,
                              star_pb: dict | None = None) -> plt.Figure | None:
        """Overlay radial power curves; adds dashed star-image curves when available."""
        pa = ra.power_metrics or {}
        pb = rb.power_metrics or {}
        freq_a = pa.get("freq_axis")
        rp_a = pa.get("radial_power")
        freq_b = pb.get("freq_axis")
        rp_b = pb.get("radial_power")
        if freq_a is None or rp_a is None or freq_b is None or rp_b is None:
            return None
        have_stars = bool(star_pa or star_pb)
        fig, ax = plt.subplots(figsize=(7, 4))
        lbl_a = f"{ra.label} (starless)" if have_stars else ra.label
        lbl_b = f"{rb.label} (starless)" if have_stars else rb.label
        ax.semilogy(freq_a, rp_a, color="steelblue", linewidth=2, label=lbl_a)
        ax.semilogy(freq_b, rp_b, color="tomato",    linewidth=2, label=lbl_b)
        if star_pa:
            sf = star_pa.get("freq_axis")
            sr = star_pa.get("radial_power")
            if sf is not None and sr is not None:
                ax.semilogy(sf, sr, color="steelblue", linewidth=1.5,
                            linestyle="--", alpha=0.6,
                            label=f"{ra.label} (with stars)")
        if star_pb:
            sf = star_pb.get("freq_axis")
            sr = star_pb.get("radial_power")
            if sf is not None and sr is not None:
                ax.semilogy(sf, sr, color="tomato", linewidth=1.5,
                            linestyle="--", alpha=0.6,
                            label=f"{rb.label} (with stars)")
        ax.axvline(0.10, color="gray", linestyle="--", linewidth=0.8,
                   label="Low / mid boundary (0.10 cyc/px)")
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("Radial power (normalised, log scale)")
        ax.set_title("Radial power spectrum — overlay")
        ax.set_xlim(0, 0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))
        fig.tight_layout()
        return fig

    # ── Section 7: Power spectrum ──────────────────────────────────────────────

    def _section_power(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("power", ra, rb)
        pa = ra.power_metrics or {}
        pb = rb.power_metrics or {}
        ca, cb = _better_worse_class(pa.get("mid_high_ratio"), pb.get("mid_high_ratio"))
        img_a = _img_tag((pa.get("figures") or {}).get("power_spectrum"), f"PS {ra.label}")
        img_b = _img_tag((pb.get("figures") or {}).get("power_spectrum"), f"PS {rb.label}")

        star_pa = pa.get("star_power") or {}
        star_pb = pb.get("star_power") or {}
        img_overlay = _img_tag(
            self._plot_radial_overlay(ra, rb,
                                      star_pa=star_pa or None,
                                      star_pb=star_pb or None),
            "Radial power overlay",
        )

        sl_note = ""
        used_a = pa.get("used_starless", False)
        used_b = pb.get("used_starless", False)
        if used_a or used_b:
            who = ", ".join(filter(None, [ra.label if used_a else "",
                                          rb.label if used_b else ""]))
            sl_note = _info_box(
                f'★ Power spectrum for <strong>{who}</strong> '
                f'was computed on the starless image to reduce star contamination '
                f'of the spatial frequency content.',
                title="Starless image used",
                open=True,
            )

        # Star-image comparison row (only when starless was the primary input)
        star_row_html = ""
        if star_pa or star_pb:
            img_star_a = _img_tag(
                (star_pa.get("figures") or {}).get("power_spectrum"),
                f"PS (with stars) {ra.label}",
            )
            img_star_b = _img_tag(
                (star_pb.get("figures") or {}).get("power_spectrum"),
                f"PS (with stars) {rb.label}",
            )
            star_row_html = f"""
<h4>With stars</h4>
<div style="display:flex;gap:10px;">
  <div style="flex:1;">{img_star_a}</div>
  <div style="flex:1;">{img_star_b}</div>
</div>
<p class="caption">Power spectrum computed on the original (star-containing) image.
Comparing with the starless curves above shows how stars elevate mid/high-frequency
power through their profiles, halos, and diffraction spikes.</p>"""

        return f"""
<h2>7. Micro-contrast / Power Spectrum &nbsp;<span class="metric-label-ok">✓ bandwidth-normalised</span></h2>
{err}
{sl_note}
{_info_box('The 2D power spectrum of a star-free nebula region reveals the '
           'spatial frequency content of the image. All data is divided by the mean signal, then '
           'mean-subtracted and multiplied by a 2D Hanning window before the FFT. Division by the '
           'mean makes the result dimensionless and comparable across filters with different '
           'bandwidths; mean subtraction and windowing suppress DC leakage from the image edges. '
           'Residual power at the lowest frequencies reflects genuine large-scale nebula structure '
           'rather than a DC artifact. The mid/high-frequency ratio (0.1–0.5 cyc/px vs 0–0.1 cyc/px) '
           'measures fine detail content relative to coarse structure.<br>'
           '<strong>Note:</strong> This comparison is only meaningful when both images cover '
           'the same target region.',
           title="About the power spectrum")}

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Mid/high ratio</td><td class="{ca}">{_val(pa.get("mid_high_ratio"), ".4f")}</td><td class="{cb}">{_val(pb.get("mid_high_ratio"), ".4f")}</td></tr>
</table>

{img_overlay}
<p class="caption">Radial power spectra overlaid (log scale). Solid curves = starless; dashed curves = with stars (when starless images were provided). Curves that diverge at
high frequencies indicate one filter preserves more fine-scale spatial detail. The
dashed vertical line marks the boundary between low (coarse structure) and mid/high frequencies.</p>

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{img_a}</div>
  <div style="flex:1;">{img_b}</div>
</div>
{star_row_html}"""

    # ── Section 8: Spatial detail ──────────────────────────────────────────────

    def _section_spatial(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("spatial", ra, rb)
        sm = ra.spatial_metrics or {}
        figs = sm.get("figures", {})

        cr_a = sm.get("contrast_ratios_a", {})
        cr_b = sm.get("contrast_ratios_b", {})

        # Contrast ratio table
        cr_rows = ""
        for ks in sorted(set(list(cr_a.keys()) + list(cr_b.keys()))):
            va = cr_a.get(ks)
            vb = cr_b.get(ks)
            ca, cb = _better_worse_class(va, vb)
            cr_rows += (f"<tr><td>{ks} px</td>"
                        f"<td class='{ca}'>{_val(va)}</td>"
                        f"<td class='{cb}'>{_val(vb)}</td></tr>")

        # Wavelet SNR table
        snr_a = sm.get("wavelet_snr_a", {})
        snr_b = sm.get("wavelet_snr_b", {})
        snr_rows = ""
        for lvl in sorted(set(list(snr_a.keys()) + list(snr_b.keys()))):
            va = snr_a.get(lvl)
            vb = snr_b.get(lvl)
            ca, cb = _better_worse_class(va, vb)
            scale_approx = 2 ** lvl
            snr_rows += (f"<tr><td>Level {lvl} (~{scale_approx}px scale)</td>"
                         f"<td class='{ca}'>{_val(va)}</td>"
                         f"<td class='{cb}'>{_val(vb)}</td></tr>")

        sigma_a = _val(sm.get("sigma_noise_a"), ".5f")
        sigma_b = _val(sm.get("sigma_noise_b"), ".5f")

        def figs_for(prefix):
            out = ""
            for key in sorted(figs):
                if key.startswith(prefix):
                    out += _hires_img_tag(figs[key], key) + "\n"
            return out

        def xs_figs_for(prefix: str) -> str:
            out = ""
            for key in sorted(figs):
                if key.startswith(prefix):
                    out += _hires_img_tag(figs[key], key) + "\n"
            return out

        def paired_figs_for(img_prefix: str, xs_prefix: str) -> str:
            """Emit each image immediately followed by its matching cross-section."""
            out = ""
            for img_key in sorted(k for k in figs if k.startswith(img_prefix)):
                suffix = img_key[len(img_prefix):]
                out += _hires_img_tag(figs[img_key], img_key) + "\n"
                xs_key = xs_prefix + suffix
                if xs_key in figs:
                    out += _hires_img_tag(figs[xs_key], xs_key) + "\n"
            return out

        has_crosshair = sm.get("crosshair") is not None
        xs_note = _info_box(
            'ℹ Cross-section profiles below are extracted along '
            'the line selected in the viewer. Left axis: both images '
            '(steelblue = A, tomato = B). Right axis (green dashed): difference A−B.',
            title="Cross-section profiles",
            open=True,
        ) if has_crosshair else ""

        sl_note = ""
        used_a = sm.get("used_starless_a", False)
        used_b = sm.get("used_starless_b", False)
        if used_a or used_b:
            who = ", ".join(filter(None, [ra.label if used_a else "",
                                          rb.label if used_b else ""]))
            sl_note = _info_box(
                f'★ Spatial detail analysis for <strong>{who}</strong> used the starless '
                f'image to reduce star contamination of the spatial frequency maps.',
                title="Starless image used",
                open=True,
            )

        roi_note = ""
        roi_used = sm.get("roi_used")
        if roi_used is not None:
            rx0, ry0, rx1, ry1 = roi_used
            roi_note = _info_box(
                f'Std / LoG / wavelet maps were computed on the user-selected region '
                f'({rx0}, {ry0}) → ({rx1}, {ry1}) px only. '
                f'Each image was first normalised by its own full-image mean signal so '
                f'the contrast ratios and wavelet SNR values are still directly '
                f'comparable between images regardless of bandwidth differences. '
                f'Cross-section profiles (if a line was drawn) sample the full image '
                f'as the line coordinates are in full-image pixel space.',
                title="ROI applied",
                open=True,
            )

        smooth_note = _info_box(
            'ℹ All spatial detail maps are smoothed with a '
            'Gaussian filter (σ = 1.0 px) <strong>for display only</strong>. '
            'Scalar metric values (contrast ratios, wavelet SNR) are computed on '
            'the raw unsmoothed data.',
            title="Display smoothing note",
        )
        _wavelet_box = _info_box(
            f'A 4-level Daubechies-4 wavelet decomposition separates the '
            f'image into spatial scale bands. Level 1 (~2 px) is noise-dominated and used only '
            f'for noise estimation. Levels 2–3 carry the most relevant signal for filter comparison. '
            f'<strong>SNR</strong> = signal energy / noise energy at each level; SNR &gt; 1 '
            f'indicates signal-dominated. '
            f'Estimated noise (σ): <strong>{ra.label}</strong> = {sigma_a}, '
            f'<strong>{rb.label}</strong> = {sigma_b} (normalised units). '
            f'Each level captures structure at roughly 2<sup>level</sup> pixel scales: '
            f'Level 1 ≈ 2 px (noise-dominated), Level 2 ≈ 4 px (fine detail — star cores, '
            f'thin filaments), Level 3 ≈ 8 px (medium structures — emission knots, shell edges), '
            f'Level 4 ≈ 16 px (broader features). A higher SNR at Level 2 indicates the filter '
            f'preserves sub-arcsecond detail better; Level 3 reflects medium-scale structure. '
            f'Cross-section profiles show how detail amplitude varies spatially along the selected line.',
            title="Wavelet decomposition",
        )

        return f"""
<h2>8. Spatial Detail Comparison &nbsp;<span class="metric-label-ok">✓ bandwidth-normalised</span></h2>
{err}
{sl_note}
{roi_note}
{smooth_note}
{_info_box('All maps below are computed on mean-signal-normalised data '
           '(each image divided by its own mean signal), making them dimensionless and comparable '
           'across different filter bandwidths. Images are shown side-by-side with a shared '
           'colour scale; the third panel shows the difference A−B.',
           title="Spatial detail maps overview")}

<h3>8b. Local Standard Deviation Maps</h3>
{_info_box('Measures how much pixel values vary within a neighbourhood. '
           'Higher values in nebula regions indicate more preserved local detail and contrast. '
           '<strong>Contrast ratio</strong> = median(nebula std) / median(background std); '
           'a higher ratio indicates better differentiation of nebula structure from background. '
           'Each map pixel contains the standard deviation of surrounding pixels within a square '
           'window. Brighter regions contain more local variation — typically nebula filaments, '
           'star halos, or noise. A filter with higher std values in targeted emission regions '
           'preserves more structure; higher std in blank sky regions indicates more photon noise. '
           'The cross-section profiles below each map pair show how local detail amplitude varies '
           'along the selected line.',
           title="Local standard deviation")}
<table>
  <tr><th>Kernel size</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {cr_rows}
</table>
{xs_note}{paired_figs_for("std_", "xs_std_")}
<p class="caption">Side-by-side local σ maps at each kernel size (shared colour scale),
each followed by its cross-section profile.
The difference map (right) highlights where one filter preserves more local variation.</p>
<h3>8c. Laplacian of Gaussian (LoG) Maps</h3>
{_info_box('The Laplacian of Gaussian highlights regions of rapid intensity '
           'change at a specific spatial scale (controlled by σ). Brighter regions in |LoG| maps '
           'indicate stronger local curvature — sharper edges and finer nebula filaments. '
           'Smaller σ highlights finer features; larger σ highlights broader structures. '
           'LoG works by Gaussian-smoothing the image (suppressing structure finer than σ) and '
           'then computing the Laplacian (second spatial derivative), which peaks at intensity '
           'boundaries. |LoG| is shown so bright-to-dark and dark-to-bright edges are treated '
           'equally. Compare maps at each σ: a sharper or higher-contrast filter will show '
           'brighter LoG response at small σ values. Cross-section profiles reveal subtle '
           'differences in edge sharpness along the selected line.',
           title="Laplacian of Gaussian (LoG)")}
{paired_figs_for("log_", "xs_log_")}
<p class="caption">|LoG| maps at σ = 1.5, 3, and 6 px (shared colour scale per row),
each followed by its cross-section profile.
A filter preserving more fine detail shows brighter, more defined boundaries at small σ.</p>
<h3>8d. Wavelet Decomposition</h3>
{_wavelet_box}

{_hires_img_tag(figs.get("wavelet_snr"), "Wavelet SNR")}
<p class="caption">Per-level SNR for both filters. Level 1 SNR &lt; 1 is expected
(noise-dominated). A filter preserving more fine detail shows higher SNR at level 2.</p>

<table>
  <tr><th>Wavelet level</th><th>{ra.label} SNR</th><th>{rb.label} SNR</th></tr>
  {snr_rows}
</table>

{paired_figs_for("wavelet_level", "xs_wavelet_level")}
<p class="caption">Reconstructed detail images at levels 2 and 3 (shared colour scale,
diverging colourmap), each followed by its cross-section profile.
The difference panel (right) shows where fine structure differs between the two filters.</p>"""

    # ── Section 9: Signal-to-Noise Ratio ─────────────────────────────────────

    def _plot_snr_pair(self, disp_a, label_a, disp_b, label_b,
                        vmin: float, vmax: float) -> plt.Figure:
        """Render both SNR maps side-by-side with a shared plasma color scale."""
        panels = [(d, lbl) for d, lbl in [(disp_a, label_a), (disp_b, label_b)]
                  if d is not None]
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]
        for ax, (disp, lbl) in zip(axes, panels):
            im = ax.imshow(disp, origin="upper", cmap="plasma",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            fig.colorbar(im, ax=ax, label="SNR (σ)", fraction=0.046, pad=0.04)
            ax.set_title(f"SNR map — {lbl}", fontsize=10)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        fig.tight_layout()
        return fig

    def _section_snr(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        err = _error_box("snr", ra, rb)
        pa = ra.snr_metrics or {}
        pb = rb.snr_metrics or {}

        ca, cb = _better_worse_class(pa.get("snr_global"), pb.get("snr_global"),
                                     higher_is_better=True)
        cs_a, cs_b = _better_worse_class(pa.get("star_snr_median"),
                                          pb.get("star_snr_median"),
                                          higher_is_better=True)
        ca_db, cb_db = _better_worse_class(pa.get("snr_global_db"), pb.get("snr_global_db"),
                                           higher_is_better=True)
        cn_a, cn_b = _better_worse_class(pa.get("noise_median"), pb.get("noise_median"),
                                         higher_is_better=False)
        cn_bg_a, cn_bg_b = _better_worse_class(pa.get("background_median"),
                                               pb.get("background_median"),
                                               higher_is_better=False)
        nf_a, nf_b = pa.get("noise_factor"), pb.get("noise_factor")
        cn_nf_a, cn_nf_b = _better_worse_class(nf_a, nf_b, higher_is_better=False)
        gain_a = pa.get("gain_e_per_adu")
        gain_b = pb.get("gain_e_per_adu")
        sky_e_a = pa.get("sky_bg_electrons")
        sky_e_b = pb.get("sky_bg_electrons")
        sky_ne_a = pa.get("sky_noise_electrons")
        sky_ne_b = pb.get("sky_noise_electrons")
        cn_sky_e_a, cn_sky_e_b = _better_worse_class(sky_e_a, sky_e_b, higher_is_better=False)
        cn_sky_ne_a, cn_sky_ne_b = _better_worse_class(sky_ne_a, sky_ne_b, higher_is_better=False)
        either_gain = (gain_a is not None or gain_b is not None)

        # Shared-scale SNR map pair
        pa_disp = pa.get("snr_display")
        pb_disp = pb.get("snr_display")
        shared_vmin = min(pa.get("snr_p2") or 0.0, pb.get("snr_p2") or 0.0)
        shared_vmax = max(pa.get("snr_p98") or 10.0, pb.get("snr_p98") or 10.0)
        snr_pair_fig = self._plot_snr_pair(
            pa_disp, ra.label, pb_disp, rb.label, shared_vmin, shared_vmax
        )
        snr_pair_html = _img_tag(snr_pair_fig, "SNR map comparison")

        # Percentile table rows
        def pct_row(label, key):
            va = pa.get(key)
            vb = pb.get(key)
            cpa, cpb = _better_worse_class(va, vb, higher_is_better=True)
            return (f"<tr><td>{label}</td>"
                    f"<td class='{cpa}'>{_val(va, '.4f')} %</td>"
                    f"<td class='{cpb}'>{_val(vb, '.4f')} %</td></tr>")

        pct_rows = "".join([
            pct_row("Pixels &gt; 3 σ <small style='color:#555'>(detected signal)</small>",  "pct_above_3"),
            pct_row("Pixels &gt; 5 σ",  "pct_above_5"),
            pct_row("Pixels &gt; 10 σ <small style='color:#555'>(reliable detail)</small>", "pct_above_10"),
            pct_row("Pixels &gt; 20 σ <small style='color:#555'>(bright structure)</small>", "pct_above_20"),
        ])

        star_snr_a = _val(pa.get("star_snr_median"), ".4f")
        star_iqr_a = _val(pa.get("star_snr_iqr"), ".4f")
        star_snr_b = _val(pb.get("star_snr_median"), ".4f")
        star_iqr_b = _val(pb.get("star_snr_iqr"), ".4f")
        star_a_cell = f"{star_snr_a} ± {star_iqr_a}" if pa.get("star_snr_median") else "—"
        star_b_cell = f"{star_snr_b} ± {star_iqr_b}" if pb.get("star_snr_median") else "—"

        # --- Starless SNR sub-section ---
        sl_a = pa.get("starless") or {}
        sl_b = pb.get("starless") or {}
        starless_html = ""
        if sl_a or sl_b:
            sl_vmin = min(sl_a.get("snr_p2") or 0.0, sl_b.get("snr_p2") or 0.0)
            sl_vmax = max(sl_a.get("snr_p98") or 10.0, sl_b.get("snr_p98") or 10.0)
            sl_pair_fig = self._plot_snr_pair(
                sl_a.get("snr_display"), ra.label + " (starless)",
                sl_b.get("snr_display"), rb.label + " (starless)",
                sl_vmin, sl_vmax,
            )
            sl_ca, sl_cb = _better_worse_class(
                sl_a.get("snr_global"), sl_b.get("snr_global"), higher_is_better=True)
            def sl_pct_row(label, key, _sla=sl_a, _slb=sl_b):
                va, vb = _sla.get(key), _slb.get(key)
                cpa, cpb = _better_worse_class(va, vb, higher_is_better=True)
                return (f"<tr><td>{label}</td>"
                        f"<td class='{cpa}'>{_val(va, '.4f')} %</td>"
                        f"<td class='{cpb}'>{_val(vb, '.4f')} %</td></tr>")
            sl_pct_rows = "".join([
                sl_pct_row("Pixels &gt; 3 σ <small style='color:#555'>(detected signal)</small>",  "pct_above_3"),
                sl_pct_row("Pixels &gt; 5 σ",  "pct_above_5"),
                sl_pct_row("Pixels &gt; 10 σ <small style='color:#555'>(reliable detail)</small>", "pct_above_10"),
                sl_pct_row("Pixels &gt; 20 σ <small style='color:#555'>(bright structure)</small>", "pct_above_20"),
            ])
            sl_pair_html = _img_tag(sl_pair_fig, "Starless SNR map comparison")
            starless_html = f"""
<h3>3b. SNR — Starless Images</h3>
{_info_box('★ SNR analysis repeated on the starless image(s). Stars inflate the '
           'global SNR and above-threshold percentages because bright star cores contribute many '
           'high-SNR pixels unrelated to the nebula emission. The starless values below reflect '
           'pure nebula depth and are recommended for comparing image quality.',
           title="Starless SNR analysis", open=True)}
<table>
  <tr><th>Metric</th><th>{ra.label} (starless)</th><th>{rb.label} (starless)</th></tr>
  <tr><td>Global SNR (σ)</td>
      <td class="{sl_ca}">{_val(sl_a.get("snr_global"), ".4f")}</td>
      <td class="{sl_cb}">{_val(sl_b.get("snr_global"), ".4f")}</td></tr>
</table>
<table>
  <tr><th>Threshold</th><th>{ra.label} (starless)</th><th>{rb.label} (starless)</th></tr>
  {sl_pct_rows}
</table>
{sl_pair_html}
<p class="caption">Per-pixel SNR map on the starless image. Star flux removed so nebula
depth drives the color scale. Both images share the same scale for direct comparison.</p>"""
        # --- Cross-section subsection (figures live in spatial_metrics) ---
        _sm = ra.spatial_metrics or {}
        _figs = _sm.get("figures", {})
        _has_xs = _sm.get("crosshair") is not None
        if _has_xs and "xs_context" in _figs:
            xs_crosshair_html = f"""
<h3>3c. Image Cross-Section</h3>
{_info_box('The cross-section extracts a 1-D brightness profile along the '
           'line drawn in the viewer. The normalised profile shows relative brightness scaled to the '
           'mean signal level — use it to compare which filter captures more emission or suppresses '
           'more continuum. The raw profile shows actual pixel counts, making it easy to assess the '
           'absolute signal difference and dynamic range. A flatter profile in a continuum-dominated '
           'field may indicate better sky suppression; a higher peak in an emission region indicates '
           'greater throughput for that line.',
           title="Image cross-section profile")}
{_hires_img_tag(_figs["xs_context"], "xs_context")}
<p class="caption">Zoomed crop centred on the cross-section line.
Orange line = {ra.label}, blue line = {rb.label}.</p>
{_hires_img_tag(_figs.get("xs_image_profile"), "xs_image_profile")}
<p class="caption">Brightness profile along the drawn line (mean-signal-normalised).</p>
{_hires_img_tag(_figs.get("xs_image_profile_raw"), "xs_image_profile_raw")}
<p class="caption">Raw pixel counts (ADU) along the cross-section line.
Use this to assess absolute signal levels and dynamic range between filters.</p>"""
        else:
            xs_crosshair_html = ""

        # --- Cross-section SNR sub-section ---
        xs_snr_html = ""
        xs_snr = _sm.get("xs_snr")
        if _has_xs and xs_snr and "xs_snr_profile" in _figs:
            snr_a_val     = xs_snr.get("snr_a")
            snr_b_val     = xs_snr.get("snr_b")
            factor_val    = xs_snr.get("exposure_factor")
            higher_label  = xs_snr.get("higher_label", ra.label)
            lower_label   = xs_snr.get("lower_label", rb.label)
            xs_width      = xs_snr.get("width", 15)

            ca_snr, cb_snr = _better_worse_class(snr_a_val, snr_b_val, higher_is_better=True)
            exp_a = 1.0 if higher_label == ra.label else (factor_val if factor_val else float("nan"))
            exp_b = 1.0 if higher_label == rb.label else (factor_val if factor_val else float("nan"))
            ca_exp, cb_exp = _better_worse_class(exp_a, exp_b, higher_is_better=False)

            if factor_val and not (isinstance(factor_val, float) and factor_val != factor_val):
                exp_sentence = (
                    f"<em>{lower_label}</em> requires <strong>{factor_val:.2f}&times;</strong> "
                    f"more exposure time than <em>{higher_label}</em> to achieve the same "
                    f"cross-section SNR."
                )
            else:
                exp_sentence = "Relative exposure factor could not be computed (one or both SNR values invalid)."

            xs_snr_html = f"""
<h3>3d. Cross-Section SNR</h3>
{_info_box(f'<p><strong>Methodology:</strong> A {xs_width}-px sample window centred on the profile '
           f'peak (bright region, gold band) and profile trough (dark region, grey band) is used '
           f'to estimate signal-to-noise ratio from the raw ADU cross-section. Both images sample '
           f'the same physical positions (determined from Image A\'s peak/trough) for a direct '
           f'comparison.</p>'
           f'<p><strong>SNR formula (std-based):</strong> '
           f'SNR&nbsp;=&nbsp;(&mu;<sub>bright</sub>&nbsp;&minus;&nbsp;&mu;<sub>dark</sub>)&nbsp;/&nbsp;'
           f'&radic;((&sigma;<sub>bright</sub>&sup2;&nbsp;+&nbsp;&sigma;<sub>dark</sub>&sup2;)&nbsp;/&nbsp;width), '
           f'where &mu; and &sigma; are the mean and standard deviation within each sample window.</p>'
           f'<p><strong>Assumptions:</strong> The cross-section line passes through a representative '
           f'bright nebula or signal feature (peak) and a dark background region (trough). Both images '
           f'are assumed to share the same sky coordinates. Region width is adjustable via the '
           f'&ldquo;XS SNR region width&rdquo; parameter.</p>'
           f'<p><strong>Relative exposure:</strong> Because SNR &prop; &radic;t, achieving equal SNR '
           f'requires (SNR<sub>higher</sub>&nbsp;/&nbsp;SNR<sub>lower</sub>)&sup2; more exposure '
           f'time on the lower-SNR image. Assumes identical sky conditions and read noise.</p>',
           title="Cross-section SNR methodology")}
{_hires_img_tag(_figs["xs_snr_profile"], "xs_snr_profile")}
<p class="caption">Raw ADU cross-section with gold (bright) and grey (dark) shaded
sample regions ({xs_width}&nbsp;px wide). Higher signal in the bright region and lower
signal in the dark region produce a higher SNR.</p>
<table>
  <thead><tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr></thead>
  <tbody>
    <tr><td>Cross-section SNR</td>
        <td class="{ca_snr}">{_val(snr_a_val, ".1f")}</td>
        <td class="{cb_snr}">{_val(snr_b_val, ".1f")}</td></tr>
    <tr><td>Relative exposure to match SNR</td>
        <td class="{ca_exp}">{_val(exp_a, ".2f")}&times;</td>
        <td class="{cb_exp}">{_val(exp_b, ".2f")}&times;</td></tr>
  </tbody>
</table>
<p>{exp_sentence}</p>"""

        _snr_metrics_box = _info_box(
            '<strong>Global SNR (sky-&sigma; units)</strong> &mdash; A single number summarising the '
            'signal strength of the entire image relative to the sky noise floor. Computed as the median '
            'pixel value of all background-subtracted pixels that lie above 3&times; the median sky RMS '
            '(i.e. pixels that contain genuine emission rather than blank sky), divided by the median sky '
            'noise (&sigma;<sub>sky</sub>). The sky noise estimate uses the 2D background RMS map produced '
            'by photutils <code>Background2D</code> with <code>SExtractorBackground</code> (background '
            'estimator) and <code>MADStdBackgroundRMS</code> (noise estimator), which partitions the image '
            'into 64 &times; 64 px grid cells, sigma-clips stars within each cell, and interpolates a smooth '
            '2D surface &mdash; the same estimate used throughout the analysis pipeline. <strong>Ideal: &gt; 10 &sigma; for nebula '
            'targets; &gt; 30 &sigma; for rich star fields.</strong> Values are dimensionless '
            '(sky-&sigma;) and directly comparable between the two images regardless of stretch or '
            'scaling. Note that this method is sky-noise-dominated: it faithfully reflects how much of the '
            'image is buried in background fluctuations but will underestimate total noise in saturated or '
            'very bright regions where photon shot noise exceeds the sky floor.<br><br>'
            '<strong>Median star SNR &plusmn; IQR</strong> &mdash; The median peak-to-noise ratio across '
            'all catalogue stars detected with photutils <code>DAOStarFinder</code> that passed quality '
            'filtering (non-saturated, isolated, minimum SNR threshold), plus the interquartile range as a '
            'measure of spread. For each star, '
            'SNR&thinsp;=&thinsp;(peak&thinsp;&minus;&thinsp;local sky)&thinsp;/&thinsp;local sky RMS, '
            'using the per-image background grid. <strong>Ideal: median &gt; 20 for a well-exposed '
            'session; IQR &lt; 15 indicates a uniform noise floor.</strong> A large IQR implies either '
            'a wide dynamic range of star brightness or strong local sky variations across the field. A '
            'lower median star SNR than the global image SNR can occur in narrowband imaging where the '
            'continuum is suppressed relative to emission-line nebulosity.<br><br>'
            '<strong>Local SNR Map</strong> &mdash; A per-pixel map of background-subtracted signal '
            'divided by the local sky RMS: SNR(x,&thinsp;y)&thinsp;=&thinsp;(data&thinsp;&minus;&thinsp;background)(x,&thinsp;y)&thinsp;/&thinsp;background_rms(x,&thinsp;y). '
            'Blank sky regions cluster near zero (&plusmn; 1&sigma; by definition); real emission appears '
            'as islands of elevated SNR. The plasma colourmap is clipped to the 2nd&ndash;98th percentile '
            'of positive pixels so that bright cores do not compress the dynamic range. '
            '<strong>What to look for:</strong> In a deeper or better-stacked image the map should be '
            'uniformly brighter across extended nebula regions. Patchwork patterns indicate residual '
            'background gradients or flat-field errors. Field edges often show elevated noise from '
            'vignetting and reduced flat-field accuracy. The spatial resolution of the noise estimate '
            'equals the background grid cell size (typically 64 &times; 64 pixels).<br><br>'
            '<strong>SNR Percentile Table</strong> &mdash; Reports the fraction of all image pixels that '
            'exceed four SNR thresholds (3&sigma;, 5&sigma;, 10&sigma;, 20&sigma;). The 3-&sigma; '
            'fraction is essentially the <em>detected area fraction</em> &mdash; the share of the field '
            'that contains statistically significant emission above the noise floor. The 10-&sigma; and '
            '20-&sigma; fractions indicate how much of the target is in the high-confidence regime where '
            'structure can be reliably measured. <strong>Ideal: 3-&sigma; fraction &gt; 20% for a rich '
            'nebula field; 10-&sigma; fraction &gt; 5% indicates strong central emission.</strong> A '
            'higher percentage across all thresholds in one image directly translates to more usable '
            'signal for further processing (deconvolution, colour mixing, detail extraction).<br><br>'
            '<strong>Sky noise &sigma;<sub>sky</sub> and sky background &mu;<sub>sky</sub></strong> &mdash; '
            'Two complementary sky characterisation metrics derived from the same photutils '
            '<code>Background2D</code> model used throughout the SNR computation. The image is divided into '
            '<strong>64 &times; 64 px grid cells</strong>; within each cell <code>SExtractorBackground</code> '
            'iteratively sigma-clips pixels above 3&sigma; (approximating the SourceExtractor background '
            'algorithm) and <code>MADStdBackgroundRMS</code> computes the cell noise from the median '
            'absolute deviation &mdash; more robust than standard deviation for fields containing stars or '
            'nebulosity. The resulting background mesh is interpolated into a smooth 2D surface covering '
            'the entire frame. <strong>No specific sky region is drawn or required</strong> &mdash; stars '
            'and bright nebula pixels are rejected automatically by sigma clipping, so &sigma;<sub>sky</sub> '
            'and &mu;<sub>sky</sub> represent the whole-image sky floor.<br><br>'
            '<em>&sigma;<sub>sky</sub></em> (Sky RMS noise, ADU) is the median of the 2D background RMS '
            'map &mdash; the pixel-to-pixel scatter of the sky and the primary noise floor used throughout '
            'this report. A lower &sigma;<sub>sky</sub> means a quieter sky; the image with the smaller '
            'value will generally record fainter signals above 3&sigma;. Differences arise from read noise, '
            'dark current, sky glow, and total integration time.<br><br>'
            '<em>&mu;<sub>sky</sub></em> (Sky background level, ADU) is the median of the smooth background '
            'model itself &mdash; how bright the blank sky is before any stretch. A higher '
            '&mu;<sub>sky</sub> does not directly harm SNR (which depends on &sigma;, not &mu;), but it '
            'reduces the dynamic range available before saturation and can indicate light pollution or short '
            'sub-exposures. <strong>What to compare:</strong> Focus on &sigma;<sub>sky</sub> as the decisive '
            'quality indicator. If &sigma;<sub>sky</sub> differs by more than &approx; 30% between the two '
            'images, the integration depth or sky conditions were meaningfully different. &mu;<sub>sky</sub> '
            'is useful context &mdash; a high background paired with low &sigma; means the sky was bright '
            'but well-sampled; a high background paired with high &sigma; suggests insufficient exposure '
            'time.<br><br>'
            '<em>Noise factor (&sigma;<sub>sky</sub>&thinsp;/&thinsp;&radic;&mu;<sub>sky</sub>)</em> '
            '&mdash; Compares the measured sky noise to the theoretical Poisson (shot-noise) floor. '
            'For a purely sky-shot-noise-limited image the pixel variance equals the mean background, '
            'so &sigma;&thinsp;=&thinsp;&radic;&mu; and the factor equals 1.0. Values above 1.0 indicate '
            'additional noise contributions &mdash; read noise, dark current, or residual fixed-pattern '
            'noise. <strong>Narrowband images in suppressed-sky conditions are commonly read-noise '
            'dominated</strong> (factor 2&ndash;10 is normal for short subs through a 3&thinsp;nm filter), '
            'because the filter reduces sky glow far more than it reduces the camera&rsquo;s read noise floor. '
            'A noise factor close to 1.0 therefore indicates the sky is bright enough &mdash; long exposures, '
            'or a broadband filter &mdash; that shot noise from sky glow dominates. <strong>When comparing two '
            'images: the lower factor is closer to the Poisson ideal, but absolute values below 3 are '
            'generally acceptable for narrowband work.</strong> A substantially higher factor in one image '
            'can indicate shorter individual sub-exposures or higher read noise from a different gain '
            'setting.<br><br>'
            '<strong>Values well below 1.0</strong> are normal for stacked images: stacking N frames '
            'reduces noise by 1/&radic;N while sky background is unchanged, so the noise factor of the '
            'stack scales accordingly. Treat very low values as an integration-depth indicator rather '
            'than a noise-regime metric.<br><br>'
            '<em>Sky background in electrons</em> &mdash; When the camera gain (e<sup>&minus;</sup>/ADU) is recorded '
            'in the FITS header (keyword <code>GAIN</code>, <code>EGAIN</code>, <code>CCDGAIN</code>, or '
            '<code>GAINDB</code>), &mu;<sub>sky</sub> and &sigma;<sub>sky</sub> are converted to electrons. '
            'This removes the camera-specific ADU offset and quantisation, placing both images on a '
            'physical scale that is directly comparable even when captured with different gain settings or '
            'cameras. A sky background of, say, 500&thinsp;e<sup>&minus;</sup> per pixel indicates that 500 sky photons '
            '(plus dark current) accumulated per pixel during the total exposure, regardless of camera model.',
            title="Understanding the SNR metrics",
        )

        return f"""
<h2>3. Signal-to-Noise Ratio (SNR)</h2>
{err}
{_snr_metrics_box}

<table>
  <tr><th>Sky metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Sky RMS noise &sigma;<sub>sky</sub> (ADU) &mdash; lower is better</td>
      <td class="{cn_a}">{_val(pa.get("noise_median"), ".6g")}</td>
      <td class="{cn_b}">{_val(pb.get("noise_median"), ".6g")}</td></tr>
  <tr><td>Sky background &mu;<sub>sky</sub> (ADU) &mdash; lower is better</td>
      <td class="{cn_bg_a}">{_val(pa.get("background_median"), ".6g")}</td>
      <td class="{cn_bg_b}">{_val(pb.get("background_median"), ".6g")}</td></tr>
  <tr><td>Noise factor &sigma;/&radic;&mu; &mdash; lower = sky-limited (ideal &asymp;1.0)</td>
      <td class="{cn_nf_a}">{_val(nf_a, ".3f")}</td>
      <td class="{cn_nf_b}">{_val(nf_b, ".3f")}</td></tr>
{"" if not either_gain else f"""  <tr><td>Gain (e<sup>&minus;</sup>/ADU, from FITS header)</td>
      <td>{"—" if gain_a is None else _val(gain_a, ".2f")}</td>
      <td>{"—" if gain_b is None else _val(gain_b, ".2f")}</td></tr>
  <tr><td>Sky background &mu;<sub>sky</sub> (e<sup>&minus;</sup>) &mdash; lower is better</td>
      <td class="{cn_sky_e_a}">{"—" if sky_e_a is None else _val(sky_e_a, ".3g")}</td>
      <td class="{cn_sky_e_b}">{"—" if sky_e_b is None else _val(sky_e_b, ".3g")}</td></tr>
  <tr><td>Sky noise &sigma;<sub>sky</sub> (e<sup>&minus;</sup>) &mdash; lower is better</td>
      <td class="{cn_sky_ne_a}">{"—" if sky_ne_a is None else _val(sky_ne_a, ".3g")}</td>
      <td class="{cn_sky_ne_b}">{"—" if sky_ne_b is None else _val(sky_ne_b, ".3g")}</td></tr>"""}
</table>

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Global SNR (&sigma;)</td>
      <td class="{ca}">{_val(pa.get("snr_global"), ".4f")} &sigma;</td>
      <td class="{cb}">{_val(pb.get("snr_global"), ".4f")} &sigma;</td></tr>
  <tr><td>Global SNR (dB)</td>
      <td class="{ca_db}">{_val(pa.get("snr_global_db"), ".2f")} dB</td>
      <td class="{cb_db}">{_val(pb.get("snr_global_db"), ".2f")} dB</td></tr>
  <tr><td>Median star SNR &plusmn; IQR</td>
      <td class="{cs_a}">{star_a_cell}</td>
      <td class="{cs_b}">{star_b_cell}</td></tr>
</table>

<table>
  <tr><th>Threshold</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {pct_rows}
</table>

{snr_pair_html}
<p class="caption">Per-pixel SNR map: signal / sky RMS at each location (plasma colourmap).
Both images share the same color scale (P2&ndash;P98 of the higher-SNR image) for direct
comparison. Bright regions have high SNR; blank sky clusters near zero.
A uniformly brighter map indicates deeper, more signal-rich data.</p>
{starless_html}
{xs_crosshair_html}
{xs_snr_html}"""

    # ── Section 10: Summary ───────────────────────────────────────────────────

    def _section_summary(self, ra: AnalysisResult, rb: AnalysisResult,
                          bw_differ: bool) -> str:
        if getattr(self, "_single_image", False):
            return ""
        sm_a = ra.spatial_metrics or {}
        sm_b = rb.spatial_metrics or {}

        def row(metric, val_a, val_b, fmt=".3f",
                higher_is_better=True, bw_flag="✓"):
            ca, cb = _better_worse_class(val_a, val_b, higher_is_better)
            label = f'{metric} <span class="metric-label-ok">{bw_flag}</span>'
            return (f"<tr><td>{label}</td>"
                    f"<td class='{ca}'>{_val(val_a, fmt)}</td>"
                    f"<td class='{cb}'>{_val(val_b, fmt)}</td></tr>")

        psf_a = ra.psf_metrics or {}
        psf_b = rb.psf_metrics or {}
        halo_a = ra.halo_metrics or {}
        halo_b = rb.halo_metrics or {}
        edge_a = ra.edge_metrics or {}
        edge_b = rb.edge_metrics or {}
        pw_a = ra.power_metrics or {}
        pw_b = rb.power_metrics or {}
        cr_a = sm_a.get("contrast_ratios_a", {})
        cr_b = sm_b.get("contrast_ratios_b", {}) if sm_b else {}
        snr_wav_a = sm_a.get("wavelet_snr_a", {})
        snr_wav_b = sm_b.get("wavelet_snr_b", {}) if sm_b else {}
        snr_ma = ra.snr_metrics or {}
        snr_mb = rb.snr_metrics or {}

        ecr_flag = "⚠" if bw_differ else "✓"

        def row_pm(metric, val_a, val_b, spread_a, spread_b, fmt=".3f",
                   higher_is_better=True, bw_flag="✓"):
            ca, cb = _better_worse_class(val_a, val_b, higher_is_better)
            label = f'{metric} <span class="metric-label-ok">{bw_flag}</span>'
            return (f"<tr><td>{label}</td>"
                    f"<td class='{ca}'>{_val_pm(val_a, spread_a, fmt)}</td>"
                    f"<td class='{cb}'>{_val_pm(val_b, spread_b, fmt)}</td></tr>")

        xs_snr_data = (ra.spatial_metrics or {}).get("xs_snr") if ra.spatial_metrics else None
        xs_snr_rows = ""
        if xs_snr_data:
            xs_snr_a_val = xs_snr_data.get("snr_a")
            xs_snr_b_val = xs_snr_data.get("snr_b")
            xs_factor    = xs_snr_data.get("exposure_factor")
            xs_higher    = xs_snr_data.get("higher_label", ra.label)
            xs_exp_a = 1.0 if xs_higher == ra.label else (xs_factor if xs_factor else float("nan"))
            xs_exp_b = 1.0 if xs_higher == rb.label else (xs_factor if xs_factor else float("nan"))
            xs_snr_rows = (
                row("XS SNR (cross-section)", xs_snr_a_val, xs_snr_b_val, fmt=".1f")
                + row("XS exposure factor (&times;)", xs_exp_a, xs_exp_b,
                      fmt=".2f", higher_is_better=False)
            )

        rows = "".join([
            row_pm("FWHM (px)", psf_a.get("fwhm_px"), psf_b.get("fwhm_px"),
                   psf_a.get("fwhm_px_mad"), psf_b.get("fwhm_px_mad"),
                   higher_is_better=False),
            row_pm("FWHM (arcsec)", psf_a.get("fwhm_arcsec"), psf_b.get("fwhm_arcsec"),
                   psf_a.get("fwhm_arcsec_mad"), psf_b.get("fwhm_arcsec_mad"),
                   higher_is_better=False),
            row_pm("Moffat β", psf_a.get("beta"), psf_b.get("beta"),
                   psf_a.get("beta_mad"), psf_b.get("beta_mad"),
                   higher_is_better=False),
            row_pm("Ellipticity", psf_a.get("ellipticity"), psf_b.get("ellipticity"),
                   psf_a.get("ellipticity_mad"), psf_b.get("ellipticity_mad"),
                   higher_is_better=False),
            row_pm("Eccentricity", psf_a.get("eccentricity"), psf_b.get("eccentricity"),
                   psf_a.get("eccentricity_mad"), psf_b.get("eccentricity_mad"),
                   higher_is_better=False),
            row("MTF50 (cyc/px)", psf_a.get("mtf50_cycles_per_px"),
                psf_b.get("mtf50_cycles_per_px"), fmt=".4f"),
            row("Halo/core ratio", halo_a.get("halo_to_core_ratio"),
                halo_b.get("halo_to_core_ratio"), fmt=".5f", higher_is_better=False),
            row("Edge width 10–90% (px)", edge_a.get("edge_width_10_90_px"),
                edge_b.get("edge_width_10_90_px"), higher_is_better=False),
            row(f"Edge contrast ratio", edge_a.get("edge_contrast_ratio"),
                edge_b.get("edge_contrast_ratio"), bw_flag=ecr_flag),
            row("Power mid/high ratio", pw_a.get("mid_high_ratio"),
                pw_b.get("mid_high_ratio"), fmt=".4f"),
            row("Std contrast ratio (15px)", cr_a.get(15), cr_b.get(15)),
            row("Wavelet SNR level 2", snr_wav_a.get(2), snr_wav_b.get(2)),
            row("Wavelet SNR level 3", snr_wav_a.get(3), snr_wav_b.get(3)),
            *([row(
                "Global SNR — starless (σ) ★",
                (snr_ma.get("starless") or {}).get("snr_global"),
                (snr_mb.get("starless") or {}).get("snr_global"),
                fmt=".4f",
            )] if (snr_ma.get("starless") or snr_mb.get("starless")) else [row(
                "Global image SNR (σ)",
                snr_ma.get("snr_global"),
                snr_mb.get("snr_global"),
                fmt=".4f",
            )]),
        ]) + xs_snr_rows

        snr_sky_rows = (
            row("Sky noise &sigma;<sub>sky</sub> (ADU)", snr_ma.get("noise_median"),
                snr_mb.get("noise_median"), fmt=".6g", higher_is_better=False)
            + row("Noise factor (&sigma;/&radic;&mu;)", snr_ma.get("noise_factor"),
                  snr_mb.get("noise_factor"), fmt=".3f", higher_is_better=False)
        )
        if (snr_ma.get("sky_bg_electrons") is not None
                or snr_mb.get("sky_bg_electrons") is not None):
            snr_sky_rows += row(
                "Sky background (e<sup>&minus;</sup>)", snr_ma.get("sky_bg_electrons"),
                snr_mb.get("sky_bg_electrons"), fmt=".3g", higher_is_better=False)
            snr_sky_rows += row(
                "Sky noise &sigma;<sub>sky</sub> (e<sup>&minus;</sup>)",
                snr_ma.get("sky_noise_electrons"), snr_mb.get("sky_noise_electrons"),
                fmt=".3g", higher_is_better=False)
        rows += snr_sky_rows

        legend = ('<p><span class="metric-label-ok">✓</span> = bandwidth-independent '
                  'comparison &nbsp;&nbsp; '
                  '<span class="metric-label-warn">⚠</span> = interpret with bandwidth '
                  'context (filters had different bandwidths)</p>')

        retention_block = getattr(self, "_cached_retention_html", "")
        retention_section = (
            "<h3>Contrast Retention Detail (convolved / original)</h3>"
            + retention_block
            if retention_block else ""
        )

        return f"""
<h2>9. Summary &amp; Recommendations</h2>
{legend}
<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {rows}
</table>
{_info_box('Green cells indicate the better value for that metric. '
           'Red cells indicate the worse value. Metrics marked ⚠ may be influenced by the '
           'difference in filter bandwidth and should not be used as the sole basis for '
           'comparison.',
           title="How to read this table")}
{retention_section}"""
