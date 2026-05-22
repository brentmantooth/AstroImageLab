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
from scipy.signal import fftconvolve
from scipy.ndimage import zoom as _ndimage_zoom, gaussian_filter as _gaussian_filter
from scipy.interpolate import griddata as _griddata
from PIL import Image as _PILImage

from core.models import AnalysisResult, HALO_FIT_RADIUS_PX, XS_LINE_ALPHA, GLASS_REFRACTIVE_INDEX, PSF_SPATIAL_MAP_SIZE, PSF_SPATIAL_MAP_SMOOTH_SIGMA, EDGE_ROI_MAP_INDICATOR_PX, LABEL_MAX_LEN
from core.astro_image import AstroImage

_TEST_IMAGE_PATH = Path(__file__).parent.parent / "resources" / "ContrastTestImage.png"


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
.info-box { background: #d1ecf1; border: 1px solid #bee5eb;
            border-radius: 4px; padding: 10px 14px; margin: 10px 0; }
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


def _arr_img_tag(arr: np.ndarray, alt: str = "") -> str:
    """Inline <img> at native (1:1) pixel resolution from a uint8 numpy array."""
    return f'<img src="data:image/png;base64,{_arr_to_b64_png(arr)}" alt="{alt}" style="max-width:100%;display:block;">'


def _val(v, fmt=".3f", fallback="—") -> str:
    if v is None:
        return fallback
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


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

    # ------------------------------------------------------------------
    # PDF writing — WeasyPrint preferred, xhtml2pdf fallback
    # ------------------------------------------------------------------

    def _write_pdf(self, html: str, output_dir: Path, stem: str,
                   html_filename: str, result_a: AnalysisResult) -> Path:
        """Try WeasyPrint, fall back to xhtml2pdf, then fall back to HTML."""
        pdf_path  = output_dir / (stem + ".pdf")
        html_path = output_dir / html_filename

        # ── Attempt 1: WeasyPrint ──────────────────────────────────────
        # importlib.import_module() is used deliberately so that PyInstaller's
        # static AST scanner does not detect weasyprint as a dependency and
        # bundle it (and its GTK3/Pango DLLs) into the compiled executable.
        try:
            import importlib as _il
            _weasy = _il.import_module("weasyprint")
            _weasy.HTML(string=html, base_url=str(output_dir)).write_pdf(str(pdf_path))
            return pdf_path
        except ImportError:
            wp_reason = "not installed"
        except OSError as e:
            wp_reason = (
                "missing native GTK/Pango libraries — "
                "install via 'conda install -c conda-forge weasyprint' "
                "or download the GTK3 runtime installer for Windows"
                if ("libpango" in str(e) or "pango" in str(e).lower() or "0x7e" in str(e))
                else str(e)
            )
        except Exception as e:
            wp_reason = str(e)

        # ── Attempt 2: xhtml2pdf ───────────────────────────────────────
        try:
            from xhtml2pdf import pisa as _pisa
            with open(pdf_path, "wb") as _f:
                result = _pisa.CreatePDF(html, dest=_f, encoding="utf-8")
            if not result.err:
                result_a.warnings.append(
                    f"WeasyPrint unavailable ({wp_reason}); PDF rendered with xhtml2pdf — "
                    "complex CSS layout may differ slightly from the HTML version."
                )
                return pdf_path
            xp_reason = f"xhtml2pdf reported errors (code {result.err})"
        except ImportError:
            xp_reason = "not installed"
        except Exception as e:
            xp_reason = str(e)

        # ── Fallback: HTML ─────────────────────────────────────────────
        result_a.warnings.append(
            f"PDF generation failed — WeasyPrint: {wp_reason}; "
            f"xhtml2pdf: {xp_reason}. "
            "Report saved as HTML instead. "
            "Install a PDF renderer: pip install xhtml2pdf  "
            "or conda install -c conda-forge weasyprint"
        )
        html_path.write_text(html, encoding="utf-8")
        return html_path

    def generate(self, image_a: AstroImage, image_b: AstroImage,
                  result_a: AnalysisResult, result_b: AnalysisResult,
                  output_dir: str | Path,
                  open_browser: bool = True,
                  report_format: str = "html") -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"report_{result_a.label}_{result_b.label}_{ts}".replace(" ", "_")
        filename = stem + ".html"

        bw_a = image_a.bandwidth_nm
        bw_b = image_b.bandwidth_nm
        bw_differ = (bw_a is not None and bw_b is not None and
                     abs(bw_a - bw_b) > 0.1)

        # Label substitution was applied in the analysis thread before any figures were
        # rendered.  Read the stored original labels from the result objects here so the
        # Section 1 info box can map "Image A/B" back to the full filenames.
        _substituted = (result_a.original_label is not None
                        or result_b.original_label is not None)
        _orig_label_a = result_a.original_label or result_a.label
        _orig_label_b = result_b.original_label or result_b.label

        sections = [
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



        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Filter Comparison: {result_a.label} vs {result_b.label}</title>
  <style>{_CSS}</style>
</head>
<body>
{"".join(sections)}
<p style="color:#999;font-size:0.85em;margin-top:40px;">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by FilterImageCompare
</p>
</body>
</html>"""

        if report_format == "pdf":
            out_path = self._write_pdf(html, output_dir, stem, filename, result_a)
        else:
            out_path = output_dir / filename
            out_path.write_text(html, encoding="utf-8")

        if open_browser:
            webbrowser.open(out_path.as_uri())
        return out_path

    # ── Section 1: Header ─────────────────────────────────────────────────────

    def _section_header(self, img_a: AstroImage, img_b: AstroImage,
                         result_a: AnalysisResult, result_b: AnalysisResult,
                         bw_differ: bool,
                         substituted: bool = False,
                         orig_label_a: str = "",
                         orig_label_b: str = "") -> str:
        bw_warn = ""
        if bw_differ:
            bw_warn = (f'<div class="bw-warn">⚠ <strong>Bandwidth warning:</strong> '
                       f'Filters have different bandwidths '
                       f'({img_a.bandwidth_nm:.1f} nm vs {img_b.bandwidth_nm:.1f} nm). '
                       f'Metrics marked <span class="metric-label-warn">⚠</span> are '
                       f'sensitive to this difference and should be interpreted with caution. '
                       f'Metrics marked <span class="metric-label-ok">✓</span> are '
                       f'bandwidth-independent.</div>')
        label_sub_box = ""
        if substituted:
            label_sub_box = (
                f'<div class="info-box"><strong>Label substitution:</strong> '
                f'One or more input filenames exceed {LABEL_MAX_LEN} characters and have been '
                f'abbreviated in all plots and legends throughout this report.<br>'
                f'&nbsp;&nbsp;<strong>Image A</strong> = {orig_label_a}<br>'
                f'&nbsp;&nbsp;<strong>Image B</strong> = {orig_label_b}</div>'
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
        sl_b = getattr(img_b, "starless_image", None)

        thumb_a = _img_tag(self._thumbnail_fig(img_a), f"Preview {img_a.label}")
        thumb_b = _img_tag(self._thumbnail_fig(img_b), f"Preview {img_b.label}")
        thumb_sl_a = _img_tag(self._thumbnail_fig(sl_a, ref_data=img_a.data),
                               f"Starless {img_a.label}") if sl_a else ""
        thumb_sl_b = _img_tag(self._thumbnail_fig(sl_b, ref_data=img_b.data),
                               f"Starless {img_b.label}") if sl_b else ""

        sl_cap_a = ('<p class="caption">Starless (STF-matched stretch)</p>'
                    if sl_a else "")
        sl_cap_b = ('<p class="caption">Starless (STF-matched stretch)</p>'
                    if sl_b else "")

        hist_tag = _img_tag(self._plot_image_histograms(img_a, img_b), "Pixel histograms")

        return f"""
<h1>Filter Image Comparison Report</h1>
<p><strong>{img_a.label}</strong> vs <strong>{img_b.label}</strong></p>
{label_sub_box}
{bw_warn}
<h2>1. Image Metadata</h2>
<div style="display:flex;gap:20px;">
  <div style="flex:1;">
    <h3>{img_a.label}</h3>
    {thumb_a}
    {thumb_sl_a}{sl_cap_a}
    <table><tbody>{meta_rows(img_a, result_a)}</tbody></table>
  </div>
  <div style="flex:1;">
    <h3>{img_b.label}</h3>
    {thumb_b}
    {thumb_sl_b}{sl_cap_b}
    <table><tbody>{meta_rows(img_b, result_b)}</tbody></table>
  </div>
</div>
<h3>Pixel Histograms</h3>
{hist_tag}
<p class="caption">Log-scale pixel value distributions. Dotted vertical lines mark the median of each image.</p>"""

    def _plot_image_histograms(self, img_a: AstroImage, img_b: AstroImage) -> plt.Figure | None:
        """Combined log-scale histogram of both images with median markers."""
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = {"a": "steelblue", "b": "tomato"}

            for img, key, label in [(img_a, "a", img_a.label), (img_b, "b", img_b.label)]:
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
                color = colors[key]
                ax.step(centers, counts, where="mid", color=color,
                        alpha=0.85, linewidth=1.4, label=label)
                median_val = float(np.median(positive))
                ax.axvline(median_val, color=color, linestyle=":", linewidth=1.5)

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
<div class="info-box">PSF/MTF comparisons are most meaningful when both images were
captured on the same night under similar atmospheric conditions. DATE-OBS values are
shown in the metadata table above.</div>"""

    # ── Section 3: PSF / MTF ──────────────────────────────────────────────────

    def _section_psf(self, ra: AnalysisResult, rb: AnalysisResult,
                      img_a: AstroImage, img_b: AstroImage) -> str:
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
        if freq_a is not None or freq_b is not None:
            fig_mtf = self._overlay_mtf(freq_a, mtf_a, freq_b, mtf_b,
                                        ra.label, rb.label)

        img_mtf = _img_tag(fig_mtf, "MTF comparison")
        img_epsf_a = _img_tag((pa.get("figures") or {}).get("epsf"), f"ePSF {ra.label}")
        img_epsf_b = _img_tag((pb.get("figures") or {}).get("epsf"), f"ePSF {rb.label}")
        img_scatter = _img_tag(self._plot_fwhm_scatter(ra, rb), "FWHM scatter")

        # Spatial maps and histograms
        img_h_a, img_w_a = img_a.data.shape[:2]
        img_h_b, img_w_b = img_b.data.shape[:2]
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

        return f"""
<h2>4. PSF / MTF &nbsp;<span class="metric-label-ok">✓ bandwidth-independent</span></h2>
{err}
<div class="info-box">The Point Spread Function (PSF) describes how a point source
(star) is rendered. FWHM measures the core width; smaller FWHM = sharper stars.
The Modulation Transfer Function (MTF) shows how well contrast is preserved at each
spatial frequency; MTF50 is the frequency at which contrast falls to 50%.
These metrics are normalised to unit amplitude and are valid regardless of filter bandwidth.</div>
<div class="info-box">Stars are detected with DAOStarFinder at a 5σ threshold, which
intentionally casts a wide net. Only isolated, high-quality stars (SNR ≥ 30,
separation ≥ 5×FWHM from neighbours, 50-pixel border margin) are passed to PSF and
ePSF fitting. A high raw detection count relative to stars used is normal and expected.</div>

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Stars detected (raw)</td><td>{_val(pa.get("n_stars_total"), "d")}</td><td>{_val(pb.get("n_stars_total"), "d")}</td></tr>
  <tr><td>Stars used for PSF</td><td>{_val(pa.get("n_stars_used"), "d")}</td><td>{_val(pb.get("n_stars_used"), "d")}</td></tr>
  <tr><td>FWHM (px)</td><td class="{ca}">{_val(pa.get("fwhm_px"))}</td><td class="{cb}">{_val(pb.get("fwhm_px"))}</td></tr>
  <tr><td>FWHM (arcsec)</td><td class="{ca}">{_val(pa.get("fwhm_arcsec"))}</td><td class="{cb}">{_val(pb.get("fwhm_arcsec"))}</td></tr>
  <tr><td>Moffat β</td><td>{_val(pa.get("beta"))}</td><td>{_val(pb.get("beta"))}</td></tr>
  <tr><td>Ellipticity</td><td>{_val(pa.get("ellipticity"))}</td><td>{_val(pb.get("ellipticity"))}</td></tr>
  <tr><td>Eccentricity</td><td>{_val(pa.get("eccentricity"))}</td><td>{_val(pb.get("eccentricity"))}</td></tr>
  <tr><td>MTF50 (cyc/px)</td><td class="{ma}">{_val(pa.get("mtf50_cycles_per_px"), ".4f")}</td><td class="{mb}">{_val(pb.get("mtf50_cycles_per_px"), ".4f")}</td></tr>
  <tr><td>MTF @ Nyquist</td><td>{_val(pa.get("mtf_nyquist"), ".4f")}</td><td>{_val(pb.get("mtf_nyquist"), ".4f")}</td></tr>
  <tr><td>Stars used in ePSF</td><td>{_epsf_stars_cell(pa)}</td><td>{_epsf_stars_cell(pb)}</td></tr>
</table>

<div class="info-box">
  <strong>Understanding the PSF metrics:</strong><br><br>

  <strong>FWHM (Full Width at Half Maximum)</strong> &mdash; The diameter of a star
  image at half its peak brightness. Smaller = sharper. Ground-based imaging is
  typically seeing-limited (1&ndash;3 arcsec); the best sites achieve sub-arcsecond
  FWHM. For filter comparison the arcsec value is the primary metric (it is
  scale-independent). A larger FWHM in one image may indicate that session had
  worse seeing, or that the filter introduces additional softening (e.g. from
  substrate wedge or coating scatter).<br><br>

  <strong>Moffat &beta; (beta)</strong> &mdash; The wing-falloff exponent of the Moffat
  profile fitted to each star: I(r) = A &times; (1 + (r/&gamma;)&sup2;)<sup>&minus;&beta;</sup>. Higher &beta;
  means the stellar wings fall off more steeply, leaving less scattered light
  outside the core. Pure Kolmogorov atmospheric turbulence predicts &beta; &asymp; 4.765;
  in practice values of 2&ndash;6 are typical. <strong>Ideal: &beta; &gt; 3.</strong> Low
  &beta; (1&ndash;2) indicates extended wings from vibration, wind shake, or poor tracking;
  very high &beta; (&gt; 6) suggests an unusually compact PSF or unusually thin
  atmosphere. A consistently lower &beta; for one filter implies it scatters more light
  into the halo/wing region &mdash; compare with the Halo Analysis section.<br><br>

  <strong>Ellipticity</strong> &mdash; How non-circular the average star shape is,
  measured from second-order image moments (0 = perfectly round, 1 = infinitely
  elongated). <strong>Ideal: &lt; 0.05.</strong> Values of 0.05&ndash;0.10 are
  borderline; &gt; 0.10 indicates a significant elongation that may reduce
  effective resolution in one axis. Common causes: tracking drift, autoguider
  lag, astigmatism, or filter substrate wedge. A large difference in ellipticity
  between the two filters is a specific indicator of filter tilt or wedge.<br><br>

  <strong>Eccentricity</strong> &mdash; A complementary measure of star elongation
  derived from the ratio of semi-minor to semi-major axis: e = &radic;(1 &minus; (b/a)&sup2;).
  <strong>Ideal: &lt; 0.10.</strong> Unlike ellipticity, eccentricity weights
  extreme elongation more strongly.<br><br>

  <strong>MTF50 (cycles/pixel)</strong> &mdash; The spatial frequency at which the
  Modulation Transfer Function falls to 50% of its peak. Higher MTF50 = the
  system preserves contrast at finer scales. The maximum physically possible
  value is 0.5 cyc/px (Nyquist limit for fully-sampled images).
  <strong>Ideal: as high as possible; typical ground-based: 0.1&ndash;0.3 cyc/px.</strong>
  MTF50 is the single most useful number for ranking overall sharpness.<br><br>

  <strong>MTF @ Nyquist</strong> &mdash; The residual MTF at exactly 0.5 cyc/px.
  For a well-sampled, diffraction-limited system this should approach 0.
  <strong>Ideal: close to 0.</strong> A notably non-zero value at Nyquist can
  indicate undersampling (FWHM &lt; ~2 px) or aliasing from a very sharp PSF.
</div>

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

<div class="info-box">
  <strong>How the Empirical PSF (ePSF) was built.</strong>
  The ePSF is constructed from all stars that passed the quality filters
  (unsaturated, adequate SNR, isolated from neighbours). Cutouts of each star
  are extracted from the background-subtracted image with a box size of
  max(25 px, 6 &times; FWHM) to capture the full wing extent.
  The <a href="https://photutils.readthedocs.io/en/stable/api/photutils.psf.EPSFBuilder.html"
  target="_blank"><em>photutils</em> <code>EPSFBuilder</code></a>
  (<a href="https://photutils.readthedocs.io/en/stable/user_guide/epsf_building.html"
  target="_blank">building guide</a>) iteratively stacks these cutouts at
  <strong>2&times; oversampling</strong>, aligning each star to its fitted
  sub-pixel centroid position to fill in the finer spatial grid. At each iteration
  the ePSF model is updated and each star is re-centred; the process repeats until
  the maximum centroid shift across all stars falls below
  <strong>0.001 px</strong> (the <code>center_accuracy</code> convergence
  criterion), or until the hard limit of <strong>15 iterations</strong> is
  reached. The number of stars actually used in the fit is shown in the table above.
  The result is displayed on a logarithmic scale to reveal structure spanning
  several orders of magnitude in brightness.
  <em>Note:</em> the current version of photutils does not expose the actual
  iteration count or a per-star residual metric through its public API; these
  fields will be populated automatically if a future photutils release returns
  them.<br><br>

  <strong>What to look for:</strong>
  <ul style="margin:0.4em 0 0 1.2em;padding:0;">
    <li><strong>Circular, compact core</strong> &mdash; ideal outcome: good focus,
        stable atmosphere, no significant aberrations.</li>
    <li><strong>Elliptical core</strong> &mdash; the elongation direction indicates
        the dominant cause: tracking drift (RA or Dec axis), astigmatism
        (diagonal elongation), or field rotation (curved smear on Alt-Az mounts).
        The eccentricity and position angle metrics in the table quantify this.</li>
    <li><strong>Asymmetric tails extending to one side</strong> &mdash; most commonly
        tracking or guiding drift in one axis, coma from the optical system
        (particularly if stars across the whole field share the same tail direction),
        or wind-induced mount vibration. If both filters show the same tail
        direction and magnitude, the cause is common to both capture sessions
        (optical or tracking), not filter-specific.</li>
    <li><strong>Extended, diffuse wings</strong> &mdash; poor seeing, thermal
        currents in the optical path, or vibration broadening the PSF without
        a directional bias.</li>
    <li><strong>Steep, clean falloff</strong> &mdash; the flux drops 2&ndash;3 orders of
        magnitude within a few FWHM. This is ideal: most of the star&rsquo;s light
        is in the core, minimising contamination of adjacent nebula structure.
        A steeper falloff (higher Moffat &beta;) is always better for contrast on
        fine detail next to bright stars.</li>
    <li><strong>Airy-ring structure</strong> &mdash; concentric rings around the
        core indicate near-diffraction-limited performance (exceptional seeing
        and optics, rarely seen in long-exposure deep-sky imaging).</li>
  </ul>
</div>

<div style="display:flex;gap:10px;">
  <div style="flex:1;">{img_epsf_a}</div>
  <div style="flex:1;">{img_epsf_b}</div>
</div>
<p class="caption">Empirical PSFs (log&#x2081;&#x208a; scale, viridis colormap). The ePSF is
built at 2&times; oversampling from all quality-filtered stars in the field. A circular,
compact core with rapid falloff is ideal. Asymmetric tails indicate tracking,
guiding, or optical aberrations &mdash; compare tail direction and magnitude between the
two images to distinguish session-specific from system-wide causes.</p>

{img_mtf}
<p class="caption">MTF curves for both filters overlaid, derived from the ePSF shown above.
Higher curve = better contrast preservation at fine scales.</p>
<div class="info-box"><strong>Reading the MTF plot — and how it is derived from the ePSF:</strong><br><br>
<strong>From ePSF to MTF:</strong>
The empirical PSF is first normalised to unit sum so that it represents a probability
distribution of where a point source&rsquo;s photons land on the detector.
A 2-D Fast Fourier Transform (FFT) is then applied, producing the complex-valued
<em>Optical Transfer Function</em> (OTF). The MTF is the magnitude of the OTF:
MTF(f<sub>x</sub>, f<sub>y</sub>) = |OTF(f<sub>x</sub>, f<sub>y</sub>)|, normalised
so that MTF(0, 0) = 1. Because the ePSF is built at 2&times; oversampling, the
frequency axes are divided by the oversampling factor so that the final MTF is
expressed in <strong>cycles per native image pixel</strong>, with the Nyquist limit
at exactly 0.5 cyc/px.<br><br>
<strong>Radial average:</strong>
The 2-D MTF is isotropically averaged into a 1-D curve by computing the mean MTF
value within concentric annular bins of width 1 sample, centred on the zero-frequency
origin. This azimuthal average assumes the PSF is roughly circular; if the ePSF
shows significant ellipticity, the radial curve represents the geometric mean of the
MTF along the major and minor axes and will understate the best-case resolution in
one direction.<br><br>
<strong>Interpreting the curve:</strong>
An ideal MTF starts at 1.0 (zero frequency) and decreases monotonically to 0 at
the Nyquist frequency (0.5 cycles/pixel). Optical aberrations, atmospheric seeing,
and focus errors lower the curve — especially at higher spatial frequencies.
<strong>MTF50</strong> is the spatial frequency where contrast falls to 50% —
analogous to a half-power point; higher MTF50 = sharper images.
If one filter&rsquo;s curve lies consistently above the other it delivers better
sharpness at all scales. If the curves cross, one filter is sharper at fine scales
while the other preserves mid-scale contrast better.<br><br>
<strong>Common causes of a lower MTF curve:</strong> poor seeing, focus offset,
filter tilt, or optical aberrations introduced by the filter glass. A significant
MTF difference between filters that should be optically identical warrants checking
filter flatness and seating.</div>

{self._psf_simulation_html(ra, rb)}

<div class="info-box"><strong>Comparing the two images:</strong>
A smaller FWHM (arcsec) and higher MTF50 indicate sharper resolution &mdash;
these are the primary quality indicators for filter comparison. A higher
Moffat &beta; indicates less scattered light in the wings. Ellipticity should be
similar between filters; a large difference suggests filter tilt, substrate
wedge, or different seeing conditions between sessions. If the ePSFs show
the same asymmetric tail in both images, the cause is common to both (optics
or tracking) and does not reflect a filter quality difference &mdash; what matters
for comparison is whether the tail is <em>more pronounced</em> in one image.</div>"""

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

        def _make_map(pts, img_h, img_w):
            if not pts:
                return None
            # Grid with same aspect ratio as image; long axis = PSF_SPATIAL_MAP_SIZE px
            if img_w >= img_h:
                gw, gh = PSF_SPATIAL_MAP_SIZE, max(1, int(PSF_SPATIAL_MAP_SIZE * img_h / img_w))
            else:
                gh, gw = PSF_SPATIAL_MAP_SIZE, max(1, int(PSF_SPATIAL_MAP_SIZE * img_w / img_h))
            gx, gy = np.meshgrid(np.linspace(0, img_w, gw),
                                  np.linspace(0, img_h, gh))
            coords = np.array([(p[0], p[1]) for p in pts])
            vals   = np.array([p[2] for p in pts])
            m = _griddata(coords, vals, (gx, gy), method="linear")
            nn = _griddata(coords, vals, (gx, gy), method="nearest")
            m = np.where(np.isnan(m), nn, m)
            return _gaussian_filter(m, sigma=PSF_SPATIAL_MAP_SMOOTH_SIGMA)

        map_a = _make_map(pts_a, img_h_a, img_w_a)
        map_b = _make_map(pts_b, img_h_b, img_w_b)

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

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(fa, fb, alpha=0.65, color="steelblue", s=25, zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Slope = 1 (equal FWHM)")
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

    def _overlay_mtf(self,
                      freq_a: "np.ndarray | None", mtf_a: "np.ndarray | None",
                      freq_b: "np.ndarray | None", mtf_b: "np.ndarray | None",
                      label_a: str, label_b: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        if freq_a is not None and mtf_a is not None:
            ax.plot(freq_a, mtf_a, color="steelblue", linewidth=2, label=label_a)
        if freq_b is not None and mtf_b is not None:
            ax.plot(freq_b, mtf_b, color="tomato", linewidth=2, label=label_b)
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
        if epsf_a is None or epsf_b is None:
            return None
        if not _TEST_IMAGE_PATH.exists():
            return None
        try:
            test_arr = np.array(
                _PILImage.open(_TEST_IMAGE_PATH).convert("L"), dtype=float
            ) / 255.0

            os_a = (ra.psf_metrics or {}).get("epsf_oversampling", 2)
            os_b = (rb.psf_metrics or {}).get("epsf_oversampling", 2)
            kern_a = _ndimage_zoom(epsf_a, 1.0 / os_a, order=1)
            kern_b = _ndimage_zoom(epsf_b, 1.0 / os_b, order=1)
            kern_a = kern_a / kern_a.sum() if kern_a.sum() > 0 else kern_a
            kern_b = kern_b / kern_b.sum() if kern_b.sum() > 0 else kern_b

            # Convolution at full resolution
            conv_a = np.clip(fftconvolve(test_arr, kern_a, mode="same"), 0.0, 1.0)
            conv_b = np.clip(fftconvolve(test_arr, kern_b, mode="same"), 0.0, 1.0)
            diff = conv_a - conv_b

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
                        "conv_b":      conv_b[y_px, _XS_X].copy(),
                        "image_strip": test_arr[y0_strip:y1_strip, _XS_X].copy(),
                    }

            # Downsample for display if image is very large (cap at 1200 px on long edge)
            h, w = test_arr.shape
            if max(h, w) > 1200:
                zoom_f = 1200.0 / max(h, w)
                test_arr = _ndimage_zoom(test_arr, zoom_f, order=1)
                conv_a   = _ndimage_zoom(conv_a,   zoom_f, order=1)
                conv_b   = _ndimage_zoom(conv_b,   zoom_f, order=1)
                diff     = _ndimage_zoom(diff,     zoom_f, order=1)

            d_max = max(float(abs(diff).max()), 1e-9)
            # Map diff to RGB using RdBu_r colormap
            diff_norm = (diff / d_max + 1.0) / 2.0          # [0, 1]
            diff_rgb = (plt.get_cmap("RdBu_r")(diff_norm)[:, :, :3] * 255).astype(np.uint8)

            return {
                "original": (test_arr * 255).astype(np.uint8),
                "conv_a":   (conv_a   * 255).astype(np.uint8),
                "conv_b":   (conv_b   * 255).astype(np.uint8),
                "diff":     diff_rgb,
                "diff_max": d_max,
                "label_a":  ra.label,
                "label_b":  rb.label,
                "xs_data":  xs_data,
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

        figs = []
        for level, title in levels:
            if level not in xs_data:
                continue
            d = xs_data[level]
            n_pts = len(d["original"])
            # x=1 = finest bars; increases toward coarser bars (data reversed).
            x = np.arange(1, n_pts + 1)
            orig  = d["original"][::-1]
            a_arr = d["conv_a"][::-1]
            b_arr = d["conv_b"][::-1]
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
            ax_raw.plot(x, orig,  color="black",     linewidth=1.0, alpha=XS_LINE_ALPHA,
                        label="Original")
            ax_raw.plot(x, a_arr, color="steelblue", linewidth=1.0, alpha=XS_LINE_ALPHA,
                        label=sim["label_a"])
            ax_raw.plot(x, b_arr, color="tomato",    linewidth=1.0, alpha=XS_LINE_ALPHA,
                        label=sim["label_b"])
            if strip is None:
                ax_raw.set_title(f"{title}  (y = {d['y_px']} px)", fontsize=9)
            ax_raw.set_ylabel("Intensity [0–1]", fontsize=8)
            ax_raw.tick_params(labelsize=7)
            ax_raw.legend(fontsize=7)
            ax_raw.grid(True, alpha=0.3, which="both")

            # ── Local contrast envelope ────────────────────────────────
            ax_env.plot(x, _envelope(orig),  color="black",     linewidth=1.4,
                        alpha=XS_LINE_ALPHA, label="Original")
            ax_env.plot(x, _envelope(a_arr), color="steelblue", linewidth=1.4,
                        alpha=XS_LINE_ALPHA, label=sim["label_a"])
            ax_env.plot(x, _envelope(b_arr), color="tomato",    linewidth=1.4,
                        alpha=XS_LINE_ALPHA, label=sim["label_b"])
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
        width = 0.25

        orig_m   = [modulation(xs_data[lv]["original"]) if lv in xs_data else 0.0 for lv in levels]
        conv_a_m = [modulation(xs_data[lv]["conv_a"])   if lv in xs_data else 0.0 for lv in levels]
        conv_b_m = [modulation(xs_data[lv]["conv_b"])   if lv in xs_data else 0.0 for lv in levels]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width, orig_m,   width, label="Original",        color="black",     alpha=0.75)
        ax.bar(x,         conv_a_m, width, label=sim["label_a"],    color="steelblue", alpha=0.85)
        ax.bar(x + width, conv_b_m, width, label=sim["label_b"],    color="tomato",    alpha=0.85)
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

        n_bands = len(bands)
        x       = np.arange(n_bands)
        width = 0.25

        fig, axes = plt.subplots(len(available), 1,
                                  figsize=(9, 4 * len(available)), squeeze=False)

        for ax_row, (level, level_title) in zip(axes[:, 0], available):
            d        = xs_data[level]
            env_orig = _envelope(d["original"][::-1])
            env_a    = _envelope(d["conv_a"][::-1])
            env_b    = _envelope(d["conv_b"][::-1])
            orig_vals = [float(np.mean(env_orig[sl])) for _, sl in bands]
            a_vals    = [float(np.mean(env_a[sl]))    for _, sl in bands]
            b_vals    = [float(np.mean(env_b[sl]))    for _, sl in bands]

            ax_row.bar(x - width, orig_vals, width, label="Original",
                       color="black",     alpha=0.75)
            ax_row.bar(x,         a_vals,   width, label=sim["label_a"],
                       color="steelblue", alpha=0.85)
            ax_row.bar(x + width, b_vals,   width, label=sim["label_b"],
                       color="tomato",    alpha=0.85)
            ax_row.set_xticks(x)
            ax_row.set_xticklabels([b[0] for b in bands], fontsize=8)
            ax_row.set_ylabel("Mean local contrast\n(peak − valley)", fontsize=8)
            ax_row.set_title(f"Contrast retention — {level_title}", fontsize=9)
            ax_row.legend(fontsize=8)
            ax_row.grid(True, alpha=0.3, axis="y")

        axes[-1, 0].set_xlabel("Spatial-frequency band (bar period in pixels)", fontsize=8)
        fig.tight_layout()
        return fig

    def _psf_simulation_html(self, ra: AnalysisResult, rb: AnalysisResult) -> str:
        """Return HTML block with four PSF simulation panels at 1:1 pixel resolution."""
        sim = self._plot_psf_simulation(ra, rb)
        if sim is None:
            return ""

        def panel(arr, title, caption=""):
            tag = _arr_img_tag(arr, title)
            cap = f'<p class="caption">{caption}</p>' if caption else ""
            return f'<div style="margin-bottom:20px;"><p><strong>{title}</strong></p>{tag}{cap}</div>'

        diff_caption = (
            f"Pixel-level difference A − B (RdBu_r colormap, range ±{sim['diff_max']:.4f}). "
            "Red = A brighter after convolution; blue = B brighter. "
            "Larger values in fine-detail regions indicate a measurable sharpness difference."
        )

        xs_figs      = self._plot_psf_crosssections(sim)
        mod_fig      = _img_tag(self._plot_psf_modulation(sim),      "Contrast modulation summary")
        band_mod_fig = _img_tag(self._plot_psf_band_modulation(sim), "Frequency-band contrast retention")

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
            "matter most; the coarser end is gently compressed. "
            f"Black = original; blue = {sim['label_a']}; red = {sim['label_b']}. "
            "A filter with a tighter PSF retains a higher envelope as x approaches 1."
        )
        mod_caption = (
            "Michelson contrast (I<sub>max</sub> &minus; I<sub>min</sub>) / "
            "(I<sub>max</sub> + I<sub>min</sub>) for each bar group. "
            "A value of 1.0 = perfect black-to-white swing; 0 = bars completely blurred. "
            "The reduction from <em>Original</em> to each filter column quantifies how much "
            "contrast that filter's PSF costs at each spatial frequency represented by the bar width."
        )
        band_mod_caption = (
            "Three charts — one per contrast row (high / medium / low) — each showing mean "
            "rolling local contrast (peak&minus;valley, ±5&thinsp;px window) grouped into "
            "four spatial-frequency bands. "
            "The <strong>Fine</strong> band (1&ndash;40&thinsp;px bar period) is the most "
            "sensitive to PSF width: a broader PSF smears the finest bars first, so a larger "
            "drop from <em>Original</em> to the filter columns here indicates a resolution "
            "penalty at high spatial frequencies. "
            "The <strong>Coarse</strong> band (301+&thinsp;px) is largely PSF-insensitive "
            "and should be near-identical for both filters — a useful sanity check. "
            "Comparing per-level charts reveals whether PSF blur degrades high-contrast "
            "detail more than low-contrast nebulosity, giving a frequency-resolved "
            "contrast-retention profile directly analogous to the MTF curves above."
        )

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
{mod_fig}
<p class="caption">{mod_caption}</p>
<h4>Spatial-frequency-resolved contrast retention</h4>
{band_mod_fig}
<p class="caption">{band_mod_caption}</p>"""

        return f"""
<h3>PSF Simulation — test chart convolved at native pixel resolution</h3>
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
<p>
  A pixel-level difference map (A &minus; B) is computed and displayed with the RdBu_r
  diverging colormap, centred at zero. Red regions indicate pixels where filter A produced
  higher intensity after convolution (A&rsquo;s PSF is locally tighter and preserved more
  contrast); blue regions indicate B is locally brighter. The horizontal cross-sections below
  isolate three contrast levels from Block&thinsp;1 of the chart and quantify the
  peak-to-valley swing preserved by each PSF.
</p>
<p>Each image is rendered at 1 image-pixel&thinsp;:&thinsp;1 screen-pixel.</p>
{panel(sim['original'], 'Original test chart')}
{panel(sim['conv_a'],   f"Convolved — {sim['label_a']}")}
{panel(sim['conv_b'],   f"Convolved — {sim['label_b']}")}
{panel(sim['diff'],     'Difference (A − B)', diff_caption)}{xs_block}"""

    # ── Section 4: Halo ───────────────────────────────────────────────────────

    def _section_halo(self, ra: AnalysisResult, rb: AnalysisResult,
                       img_a: AstroImage, img_b: AstroImage) -> str:
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
            optics_note = (
                f'<div class="info-box">'
                f'<strong>Expected halo size for this telescope</strong> '
                f'(f/{f_rat:.1f}, {pix_mm*1000:.2f} µm pixels, '
                f'{t_mm:.1f} mm filter thickness): '
                f'halo radius ≈ <strong>{r_expected:.0f} px</strong>. '
                f'The cutout windows in the grid below are sized to '
                f'2× this expected radius to ensure the full halo extent is visible.'
                f'</div>'
            )
        else:
            optics_note = ""

        return f"""
<h2>5. Halo Analysis &nbsp;<span class="metric-label-ok">&#10003; bandwidth-independent</span></h2>
{err}
<div class="info-box">
  <strong>What causes halos?</strong>
  Halos around bright stars in narrowband images arise from internal reflections
  within the filter substrate and its AR coatings. A fraction of the incoming
  light reflects off the back surface of the filter glass, travels back through
  the substrate, reflects off the front surface, and then exits &mdash; offset laterally
  from the direct beam. This offset is what appears as the circular glow surrounding
  bright stars.<br><br>
  <strong>Focal ratio and halo size.</strong>
  The halo radius at the focal plane is approximately:<br>
  <code>R &asymp; t / (n &times; f_ratio &times; pixel_size)</code><br>
  where <em>t</em> is the filter substrate thickness, <em>n</em> &asymp; 1.9 (dichroic
  filter glass refractive index), and <em>pixel_size</em> is in mm. Because f-ratio appears in
  the denominator, <strong>faster telescopes (lower f-ratio) produce proportionally
  larger halos</strong> for the same filter. A narrowband filter that shows no
  visible halo on a slow f/10 refractor may produce a prominent halo on an f/4
  Newtonian. This is a property of the optical system, not the filter quality alone.
  The halo-to-core <em>ratio</em> (amplitude of the halo relative to the star core)
  is a more filter-specific quality indicator than the raw halo size.
</div>
{optics_note}
<div class="info-box">
  <strong>How halo/core ratio and halo radius are computed.</strong>
  For each bright unsaturated star, the background-subtracted radial intensity profile
  (median-binned in 0.5&thinsp;px annuli out to {HALO_FIT_RADIUS_PX}&thinsp;px) is fitted
  with a <em>two-component Moffat model</em>:<br>
  <code>I(r) = A<sub>core</sub> &middot; Moffat(r; &gamma;<sub>core</sub>, &alpha;<sub>core</sub>)
             + A<sub>halo</sub> &middot; Moffat(r; &gamma;<sub>halo</sub>, &alpha;<sub>halo</sub>)</code><br>
  The fit is performed in log<sub>10</sub> space so the profile&rsquo;s wide dynamic range is
  weighted uniformly rather than being dominated by the bright core.<br><br>
  <strong>Halo / core ratio</strong> = A<sub>halo</sub> / A<sub>core</sub> &mdash; the
  amplitude of the wide Moffat component relative to the core peak. A value of 0 means no
  detectable halo; values above 0.15 indicate significant internal reflection. Because both
  amplitudes are normalised to the same star, the ratio is independent of absolute brightness
  and directly comparable between filters.<br><br>
  <strong>Halo radius</strong> = HWHM of the halo Moffat component:
  R&thinsp;=&thinsp;&gamma;<sub>halo</sub>&thinsp;&middot;&thinsp;&radic;(2<sup>1/&alpha;<sub>halo</sub></sup>&thinsp;&minus;&thinsp;1).
  If this value exceeds the {HALO_FIT_RADIUS_PX}&thinsp;px fit window it is marked
  <em>N/A</em> — the data do not yet show the halo half-power point, so the width cannot
  be reliably measured. In that case the halo/core ratio is the more reliable indicator.
  For strongly saturated stars the core is clipped and no Moffat fit is attempted; their
  halo structure is visible in the cross-sections and RDF plots instead.
</div>

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

<div class="info-box">
  <strong>Radial Distribution Function (RDF) — how it is computed and how to read it.</strong><br>
  For each star, the background-subtracted image is log<sub>10</sub>-transformed (compressing
  the 3&ndash;5 decade dynamic range of a stellar halo into a manageable linear scale) and then
  binned into concentric 1-pixel-wide annuli centred on the detected star centre. The
  <em>mean</em> and <em>standard deviation</em> of the log-transformed pixel values in each
  annulus are recorded, normalised so the profile starts at 1.0 (log<sub>10</sub> = 0 at
  r&thinsp;=&thinsp;0), and then inverse-transformed (10<sup>x</sup>) for display on a
  logarithmic intensity axis. Individual per-star profiles are stacked and averaged to produce
  the aggregate curves shown below. The shaded band shows the ±1&sigma; within-annulus spread —
  a wide band at a given radius means the halo is angularly asymmetric at that distance
  (not a perfect ring).<br><br>
  <strong>How to interpret the profile.</strong>
  For an ideal star with no halo the profile follows a smooth, monotonically decreasing curve
  set by the PSF shape: a Gaussian gives a downward-curving parabola on the log intensity axis;
  a Moffat (more realistic for astronomical seeing) gives a gentler, power-law-like tail.
  The key indicator of a halo is a <em>shoulder</em> — a point where the profile levels off,
  rises slightly, or decays noticeably more slowly than the extrapolated core trend at
  intermediate radii (typically 10&ndash;60 px from the star centre). This shoulder represents
  the reflected light that forms the circular glow around bright stars. A profile that tracks
  a smooth, featureless decay with no shoulder indicates little or no halo contribution.
  Comparing the two overlaid profiles shows which filter produces more halo light at each
  radius, independent of the absolute brightness of the stars used.
</div>

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

<div class="info-box"><strong>Ideal:</strong> Halo/core ratio &lt; 0.05 is excellent;
&gt; 0.15 indicates significant internal reflection that will reduce contrast on
bright stars.</div>"""

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
        font_size = max(6, int(circle_r * 0.6))

        for rank, (sa, _sb) in enumerate(matched, start=1):
            xd = sa["xc"] * scale
            yd = sa["yc"] * scale
            circ = plt.Circle((xd, yd), circle_r, color="red",
                               fill=False, linewidth=1.2)
            ax.add_patch(circ)
            ax.text(xd + circle_r * 0.8, yd + circle_r * 0.8,
                    str(rank), color="red", fontsize=font_size,
                    fontweight="bold", ha="left", va="bottom",
                    clip_on=True)

        for i, sa in enumerate(saturated, start=1):
            xd = sa["xc"] * scale
            yd = sa["yc"] * scale
            circ = plt.Circle((xd, yd), circle_r, color="magenta",
                               fill=False, linewidth=1.2, linestyle="--")
            ax.add_patch(circ)
            ax.text(xd + circle_r * 0.8, yd + circle_r * 0.8,
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
                              img_a: AstroImage, img_b: AstroImage) -> plt.Figure | None:
        if not matched:
            return None

        bgsub_a = img_a.background_subtracted() if img_a.background is not None else img_a.data
        bgsub_b = img_b.background_subtracted() if img_b.background is not None else img_b.data

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
                ax_b.set_title(f"#{idx+1} {img_b.label}"
                               + (f"\nh/c={h2c_b:.5f}" if h2c_b is not None else ""),
                               fontsize=9)
            else:
                ax_b.set_title(f"#{idx+1} {img_b.label}\n(no match)", fontsize=9)
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
                                   linewidth=1.0, alpha=XS_LINE_ALPHA, label=img_b.label)
                ax_xs.set_title(f"#{idx+1} cross-section", fontsize=8)
                ax_xs.set_xlabel("px from centre", fontsize=7)
                ax_xs.tick_params(labelsize=7)
                ax_xs.legend(fontsize=7)
                ax_xs.grid(True, alpha=0.25, which="both")
                ax_xs.axis("on")

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
                                linewidth=1.0, label=img_b.label)
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

        fig.tight_layout()
        return fig

    def _plot_saturated_star_grid(self, matched: list,
                                   img_a: AstroImage, img_b: AstroImage) -> plt.Figure | None:
        """Cutout + cross-section grid for saturated bright stars (no Moffat fit)."""
        if not matched:
            return None

        bgsub_a = img_a.background_subtracted() if img_a.background is not None else img_a.data
        bgsub_b = img_b.background_subtracted() if img_b.background is not None else img_b.data

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
            title_b = (f"S{idx+1} {img_b.label}\n"
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
                                   linewidth=1.0, alpha=XS_LINE_ALPHA, label=img_b.label)
                ax_xs.set_title(f"S{idx+1} cross-section", fontsize=8)
                ax_xs.set_xlabel("px from centre", fontsize=7)
                ax_xs.tick_params(labelsize=7)
                ax_xs.legend(fontsize=7)
                ax_xs.grid(True, alpha=0.25, which="both")
                ax_xs.axis("on")

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
                                linewidth=1.0, label=img_b.label)
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
            sl_note = (f'<div class="info-box">★ Edge analysis for <strong>{who}</strong> '
                       f'used the starless image so the strongest gradient search locates '
                       f'a nebula emission boundary rather than a star profile.</div>')

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
<div class="info-box">
  <strong>Edge Spread Function (ESF)</strong> — A 1-D intensity profile sampled
  perpendicular to the detected edge, averaged across the full height of the ROI
  after rotating so the edge runs vertically. An ideal ESF is a smooth sigmoid:
  the steeper the transition, the better the local contrast and resolution.
  Normalised to [0, 1], the ESF shape is <strong>bandwidth-independent ✓</strong>
  and directly comparable between filters.<br><br>
  <strong>Line Spread Function (LSF)</strong> — The derivative of the ESF,
  computed with a Savitzky-Golay filter (cubic polynomial, window ≈ 18% of the
  ESF length, typically 11 points). SG fitting smooths sample-to-sample noise
  while preserving the height and width of narrow peaks better than a simple
  finite-difference derivative or Gaussian smoothing, making it the standard
  method for ESF differentiation in optical MTF analysis (ISO 12233). Ideally a
  narrow, symmetric peak centred on the edge. A broader LSF peak indicates softer
  resolution; asymmetry or secondary lobes can indicate optical aberrations,
  atmospheric dispersion, or poor focus stability during the integration.<br><br>
  <strong>10–90% edge width</strong> — The pixel (or arcsec) distance between
  the 10% and 90% intensity points on the ESF. Smaller values indicate a
  sharper, better-resolved edge. Use the arcsec figure for cross-image comparison
  if the pixel scales differ.<br><br>
  The <strong>edge contrast ratio</strong> (bright-side / dark-side mean signal)
  is <strong>bandwidth-sensitive ⚠</strong>: a narrower filter rejects more
  continuum background, which can raise this ratio independently of optical quality.
</div>

<div class="info-box">
  <strong>How the edge regions were selected:</strong>
  The analysis applies an STF stretch to the background-subtracted image to bring
  faint emission boundaries into relief, then computes a <strong>pixel-scale-adaptive
  Gaussian gradient magnitude</strong> (sigma ≈ 1.5 arcsec, capped 1–8 px) across
  the whole frame. Using a Gaussian gradient rather than a fixed 3×3 Sobel kernel
  means that diffuse gradients in long-focal-length images are detected as reliably
  as sharp edges in short-focal-length data.
  The <strong>three strongest, well-separated gradient peaks</strong>
  are located automatically (peaks are suppressed within a 90 px radius after each
  detection to ensure the three regions sample distinct features). A 500 × 500 px
  context window is shown for each, centred on the gradient peak; the 60 × 60 px
  analysis region (cyan box) is highlighted within it.
  If a starless image was provided it is used in place of the stacked image, so the
  search locates nebula emission boundaries rather than star profiles.
  Both images are measured over the <em>identical</em> pixel regions: Image A's
  detected ROI coordinates are reused for Image B after alignment. The
  <strong>ESF scan direction is taken from whichever image has the stronger overall
  gradient</strong>, then applied to both, so the two ESF curves always sample the
  same cross-section orientation and are directly comparable.
  The table below shows metrics from the strongest of the three edges; individual
  per-edge figures follow.
</div>
{gradient_row}

<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Edge width 10–90% (px) ✓</td><td class="{ca}">{_val(ea.get("edge_width_10_90_px"))}</td><td class="{cb}">{_val(eb.get("edge_width_10_90_px"))}</td></tr>
  <tr><td>Edge width 10–90% (arcsec) ✓</td><td>{_val(ea.get("edge_width_10_90_arcsec"))}</td><td>{_val(eb.get("edge_width_10_90_arcsec"))}</td></tr>
  <tr><td>Edge contrast ratio{ecr_warn}</td><td>{_val(ea.get("edge_contrast_ratio"))}</td><td>{_val(eb.get("edge_contrast_ratio"))}</td></tr>
  <tr><td>Gradient magnitude</td><td>{_val(ea.get("gradient_magnitude"), ".2f")}</td><td>{_val(eb.get("gradient_magnitude"), ".2f")}</td></tr>
</table>

{edge_figures_html}
<div class="info-box" style="font-size:0.9em;">
  <strong>Figures per edge:</strong>
  <em>ROI context panels</em> (one per image) — 500 × 500 px context window centred on
  the detected gradient peak.
  The <span style="color:#90ee90;font-weight:bold;">dashed lime rectangle</span> marks
  the 60 × 60 px analysis region used for ESF/LSF extraction.
  The <span style="color:#00bcd4;font-weight:bold;">cyan line</span> shows the ESF
  scan direction (perpendicular to the edge), clipped to the analysis region;
  the <span style="color:#c8b400;font-weight:bold;">yellow dashed line</span> shows
  the detected edge orientation, also clipped to the analysis region. &nbsp;
  <em>ESF / LSF comparison</em> — both images overlaid on shared axes;
  <span style="color:steelblue;font-weight:bold;">Image A (steelblue)</span> vs
  <span style="color:tomato;font-weight:bold;">Image B (tomato)</span>.
  ESF: normalised intensity transition; dashed lines mark the 10% and 90% levels used
  for the edge width measurement.
  LSF: derivative of the ESF; peak width and symmetry indicate local resolution quality.
</div>

<div class="info-box">
  <strong>Interpreting the comparison:</strong>
  <ul style="margin:0.4em 0 0 1.2em;padding:0;">
    <li><strong>Edge width (arcsec)</strong> is the primary comparator — it is
        scale-independent. Prefer the arcsec figure when the two images have
        different pixel scales.</li>
    <li>A difference of less than ~10% in edge width is typically within
        measurement uncertainty for a single edge sample; larger differences
        are likely real.</li>
    <li>A <strong>broader LSF peak</strong> in one image suggests lower resolution
        at the edge spatial frequency. Common causes: worse seeing during that
        integration, softer focus, or greater atmospheric dispersion from a filter
        with a very wide bandpass.</li>
    <li>An <strong>asymmetric or multi-lobed LSF</strong> can indicate optical
        aberrations, trailing, or non-uniform atmospheric refraction.</li>
    <li>If edge widths are similar but <strong>gradient magnitude</strong> differs
        substantially, the difference is likely signal level or background contrast
        rather than resolution — gradient magnitude is intensity-dependent and
        should not be used alone to rank image quality.</li>
    <li>The <strong>edge contrast ratio</strong> is only directly comparable between
        images of identical bandwidth. A narrower filter naturally yields a higher
        ratio by suppressing continuum background.</li>
  </ul>
</div>"""

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
            sl_note = (f'<div class="info-box">★ Power spectrum for <strong>{who}</strong> '
                       f'was computed on the starless image to reduce star contamination '
                       f'of the spatial frequency content.</div>')

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
<div class="info-box">The 2D power spectrum of a star-free nebula region reveals the
spatial frequency content of the image. All data is divided by the mean signal, then
mean-subtracted and multiplied by a 2D Hanning window before the FFT. Division by the
mean makes the result dimensionless and comparable across filters with different
bandwidths; mean subtraction and windowing suppress DC leakage from the image edges.
Residual power at the lowest frequencies reflects genuine large-scale nebula structure
rather than a DC artifact. The mid/high-frequency ratio (0.1–0.5 cyc/px vs 0–0.1 cyc/px)
measures fine detail content relative to coarse structure.
<br><strong>Note:</strong> This comparison is only meaningful when both images cover
the same target region.</div>

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

        has_crosshair = sm.get("crosshair") is not None
        xs_note = (
            '<div class="info-box">ℹ Cross-section profiles below are extracted along '
            'the line selected in the viewer. Left axis: both images '
            '(steelblue = A, tomato = B). Right axis (green dashed): difference A−B.</div>'
        ) if has_crosshair else ""

        sl_note = ""
        used_a = sm.get("used_starless_a", False)
        used_b = sm.get("used_starless_b", False)
        if used_a or used_b:
            who = ", ".join(filter(None, [ra.label if used_a else "",
                                          rb.label if used_b else ""]))
            sl_note = (f'<div class="info-box">★ Spatial detail analysis for '
                       f'<strong>{who}</strong> used the starless image to reduce '
                       f'star contamination of the spatial frequency maps.</div>')

        roi_note = ""
        roi_used = sm.get("roi_used")
        if roi_used is not None:
            rx0, ry0, rx1, ry1 = roi_used
            roi_note = (
                f'<div class="info-box"><strong>ROI applied:</strong> '
                f'Std / LoG / wavelet maps were computed on the user-selected region '
                f'({rx0}, {ry0}) → ({rx1}, {ry1}) px only. '
                f'Each image was first normalised by its own full-image mean signal so '
                f'the contrast ratios and wavelet SNR values are still directly '
                f'comparable between images regardless of bandwidth differences. '
                f'Cross-section profiles (if a line was drawn) sample the full image '
                f'as the line coordinates are in full-image pixel space.</div>'
            )

        smooth_note = (
            '<div class="info-box">ℹ All spatial detail maps are smoothed with a '
            'Gaussian filter (σ = 1.0 px) <strong>for display only</strong>. '
            'Scalar metric values (contrast ratios, wavelet SNR) are computed on '
            'the raw unsmoothed data.</div>'
        )

        xs_context_html = ""
        if has_crosshair and "xs_context" in figs:
            xs_context_html = f"""
<div class="info-box">The cross-section extracts a 1-D brightness profile along the
line drawn in the viewer. The normalised profile shows relative brightness scaled to the
mean signal level — use it to compare which filter captures more emission or suppresses
more continuum. The raw profile shows actual pixel counts, making it easy to assess the
absolute signal difference and dynamic range. A flatter profile in a continuum-dominated
field may indicate better sky suppression; a higher peak in an emission region indicates
greater throughput for that line.</div>
{_hires_img_tag(figs["xs_context"], "xs_context")}
<p class="caption">Zoomed crop centred on the cross-section line.
Orange line = {ra.label}, blue line = {rb.label}.</p>
{_hires_img_tag(figs.get("xs_image_profile"), "xs_image_profile")}
<p class="caption">Brightness profile along the drawn line (mean-signal-normalised).</p>
{_hires_img_tag(figs.get("xs_image_profile_raw"), "xs_image_profile_raw")}
<p class="caption">Raw pixel counts (ADU) along the cross-section line.
Use this to assess absolute signal levels and dynamic range between filters.</p>"""
        else:
            xs_context_html = (
                '<div class="info-box">No cross-section line was drawn. '
                'Draw a line in the GUI before running the analysis to see '
                'cross-section profiles here.</div>'
            )

        return f"""
<h2>8. Spatial Detail Comparison &nbsp;<span class="metric-label-ok">✓ bandwidth-normalised</span></h2>
{err}
{sl_note}
{roi_note}
{smooth_note}
<div class="info-box">All maps below are computed on mean-signal-normalised data
(each image divided by its own mean signal), making them dimensionless and comparable
across different filter bandwidths. Images are shown side-by-side with a shared
colour scale; the third panel shows the difference A−B.</div>

<h3>8a. Image Cross-Section</h3>
{xs_context_html}

<h3>8b. Local Standard Deviation Maps</h3>
<div class="info-box">Measures how much pixel values vary within a neighbourhood.
Higher values in nebula regions indicate more preserved local detail and contrast.
<strong>Contrast ratio</strong> = median(nebula std) / median(background std);
a higher ratio indicates better differentiation of nebula structure from background.
Each map pixel contains the standard deviation of surrounding pixels within a square
window. Brighter regions contain more local variation — typically nebula filaments,
star halos, or noise. A filter with higher std values in targeted emission regions
preserves more structure; higher std in blank sky regions indicates more photon noise.
The cross-section profiles below each map pair show how local detail amplitude varies
along the selected line.</div>
<table>
  <tr><th>Kernel size</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {cr_rows}
</table>
{figs_for("std_")}
<p class="caption">Side-by-side local σ maps at each kernel size (shared colour scale).
The difference map (right) highlights where one filter preserves more local variation.</p>
{xs_note}{xs_figs_for("xs_std_")}
<h3>8c. Laplacian of Gaussian (LoG) Maps</h3>
<div class="info-box">The Laplacian of Gaussian highlights regions of rapid intensity
change at a specific spatial scale (controlled by σ). Brighter regions in |LoG| maps
indicate stronger local curvature — sharper edges and finer nebula filaments.
Smaller σ highlights finer features; larger σ highlights broader structures.
LoG works by Gaussian-smoothing the image (suppressing structure finer than σ) and
then computing the Laplacian (second spatial derivative), which peaks at intensity
boundaries. |LoG| is shown so bright-to-dark and dark-to-bright edges are treated
equally. Compare maps at each σ: a sharper or higher-contrast filter will show
brighter LoG response at small σ values. Cross-section profiles reveal subtle
differences in edge sharpness along the selected line.</div>
{figs_for("log_")}
<p class="caption">|LoG| maps at σ = 1.5, 3, and 6 px (shared colour scale per row).
A filter preserving more fine detail shows brighter, more defined boundaries at small σ.</p>
{xs_figs_for("xs_log_")}
<h3>8d. Wavelet Decomposition</h3>
<div class="info-box">A 4-level Daubechies-4 wavelet decomposition separates the
image into spatial scale bands. Level 1 (~2 px) is noise-dominated and used only
for noise estimation. Levels 2–3 carry the most relevant signal for filter comparison.
<strong>SNR</strong> = signal energy / noise energy at each level; SNR &gt; 1
indicates signal-dominated.
Estimated noise (σ): <strong>{ra.label}</strong> = {sigma_a},
<strong>{rb.label}</strong> = {sigma_b} (normalised units)
Each level captures structure at roughly 2<sup>level</sup> pixel scales:
Level 1 ≈ 2 px (noise-dominated), Level 2 ≈ 4 px (fine detail — star cores,
thin filaments), Level 3 ≈ 8 px (medium structures — emission knots, shell edges),
Level 4 ≈ 16 px (broader features). A higher SNR at Level 2 indicates the filter
preserves sub-arcsecond detail better; Level 3 reflects medium-scale structure.
Cross-section profiles show how detail amplitude varies spatially along the selected line.</div>

{_hires_img_tag(figs.get("wavelet_snr"), "Wavelet SNR")}
<p class="caption">Per-level SNR for both filters. Level 1 SNR &lt; 1 is expected
(noise-dominated). A filter preserving more fine detail shows higher SNR at level 2.</p>

<table>
  <tr><th>Wavelet level</th><th>{ra.label} SNR</th><th>{rb.label} SNR</th></tr>
  {snr_rows}
</table>

{figs_for("wavelet_level")}
<p class="caption">Reconstructed detail images at levels 2 and 3 (shared colour scale,
diverging colourmap). The difference panel (right) shows where fine structure differs
between the two filters.</p>
{xs_figs_for("xs_wavelet_level")}"""

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
            pct_row("Pixels &gt; 3 σ",  "pct_above_3"),
            pct_row("Pixels &gt; 5 σ",  "pct_above_5"),
            pct_row("Pixels &gt; 10 σ", "pct_above_10"),
            pct_row("Pixels &gt; 20 σ", "pct_above_20"),
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
                sl_pct_row("Pixels &gt; 3 σ",  "pct_above_3"),
                sl_pct_row("Pixels &gt; 5 σ",  "pct_above_5"),
                sl_pct_row("Pixels &gt; 10 σ", "pct_above_10"),
                sl_pct_row("Pixels &gt; 20 σ", "pct_above_20"),
            ])
            sl_pair_html = _img_tag(sl_pair_fig, "Starless SNR map comparison")
            starless_html = f"""
<h3>3b. SNR — Starless Images</h3>
<div class="info-box">★ SNR analysis repeated on the starless image(s). Stars inflate the
global SNR and above-threshold percentages because bright star cores contribute many
high-SNR pixels unrelated to the nebula emission. The starless values below reflect
pure nebula depth and are recommended for comparing image quality.</div>
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
        return f"""
<h2>3. Signal-to-Noise Ratio (SNR)</h2>
{err}
<div class="info-box">
  <strong>Understanding the SNR metrics:</strong><br><br>

  <strong>Global SNR (sky-&sigma; units)</strong> &mdash; A single number summarising the
  signal strength of the entire image relative to the sky noise floor. Computed as the median
  pixel value of all background-subtracted pixels that lie above 3&times; the median sky RMS
  (i.e. pixels that contain genuine emission rather than blank sky), divided by the median sky
  noise (&sigma;<sub>sky</sub>). The sky noise estimate uses the 2D background RMS map produced
  by photutils <code>Background2D</code> with <code>SExtractorBackground</code> (background
  estimator) and <code>MADStdBackgroundRMS</code> (noise estimator), which partitions the image
  into 64 &times; 64 px grid cells, sigma-clips stars within each cell, and interpolates a smooth
  2D surface &mdash; the same estimate used throughout the analysis pipeline. <strong>Ideal: &gt; 10 &sigma; for nebula
  targets; &gt; 30 &sigma; for rich star fields.</strong> Values are dimensionless
  (sky-&sigma;) and directly comparable between the two images regardless of stretch or
  scaling. Note that this method is sky-noise-dominated: it faithfully reflects how much of the
  image is buried in background fluctuations but will underestimate total noise in saturated or
  very bright regions where photon shot noise exceeds the sky floor.<br><br>

  <strong>Median star SNR &plusmn; IQR</strong> &mdash; The median peak-to-noise ratio across
  all catalogue stars detected with photutils <code>DAOStarFinder</code> that passed quality
  filtering (non-saturated, isolated, minimum SNR threshold), plus the interquartile range as a
  measure of spread. For each star,
  SNR&thinsp;=&thinsp;(peak&thinsp;&minus;&thinsp;local sky)&thinsp;/&thinsp;local sky RMS,
  using the per-image background grid. <strong>Ideal: median &gt; 20 for a well-exposed
  session; IQR &lt; 15 indicates a uniform noise floor.</strong> A large IQR implies either
  a wide dynamic range of star brightness or strong local sky variations across the field. A
  lower median star SNR than the global image SNR can occur in narrowband imaging where the
  continuum is suppressed relative to emission-line nebulosity.<br><br>

  <strong>Local SNR Map</strong> &mdash; A per-pixel map of background-subtracted signal
  divided by the local sky RMS: SNR(x,&thinsp;y)&thinsp;=&thinsp;(data&thinsp;&minus;&thinsp;background)(x,&thinsp;y)&thinsp;/&thinsp;background_rms(x,&thinsp;y).
  Blank sky regions cluster near zero (&plusmn; 1&sigma; by definition); real emission appears
  as islands of elevated SNR. The plasma colourmap is clipped to the 2nd&ndash;98th percentile
  of positive pixels so that bright cores do not compress the dynamic range.
  <strong>What to look for:</strong> In a deeper or better-stacked image the map should be
  uniformly brighter across extended nebula regions. Patchwork patterns indicate residual
  background gradients or flat-field errors. Field edges often show elevated noise from
  vignetting and reduced flat-field accuracy. The spatial resolution of the noise estimate
  equals the background grid cell size (typically 64 &times; 64 pixels).<br><br>

  <strong>SNR Percentile Table</strong> &mdash; Reports the fraction of all image pixels that
  exceed four SNR thresholds (3&sigma;, 5&sigma;, 10&sigma;, 20&sigma;). The 3-&sigma;
  fraction is essentially the <em>detected area fraction</em> &mdash; the share of the field
  that contains statistically significant emission above the noise floor. The 10-&sigma; and
  20-&sigma; fractions indicate how much of the target is in the high-confidence regime where
  structure can be reliably measured. <strong>Ideal: 3-&sigma; fraction &gt; 20% for a rich
  nebula field; 10-&sigma; fraction &gt; 5% indicates strong central emission.</strong> A
  higher percentage across all thresholds in one image directly translates to more usable
  signal for further processing (deconvolution, colour mixing, detail extraction).<br><br>

  <strong>Sky noise &sigma;<sub>sky</sub> and sky background &mu;<sub>sky</sub></strong> &mdash;
  Two complementary sky characterisation metrics derived from the same photutils
  <code>Background2D</code> model used throughout the SNR computation. The image is divided into
  <strong>64 &times; 64 px grid cells</strong>; within each cell <code>SExtractorBackground</code>
  iteratively sigma-clips pixels above 3&sigma; (approximating the SourceExtractor background
  algorithm) and <code>MADStdBackgroundRMS</code> computes the cell noise from the median
  absolute deviation &mdash; more robust than standard deviation for fields containing stars or
  nebulosity. The resulting background mesh is interpolated into a smooth 2D surface covering
  the entire frame. <strong>No specific sky region is drawn or required</strong> &mdash; stars
  and bright nebula pixels are rejected automatically by sigma clipping, so &sigma;<sub>sky</sub>
  and &mu;<sub>sky</sub> represent the whole-image sky floor.<br><br>
  <em>&sigma;<sub>sky</sub></em> (Sky RMS noise, ADU) is the median of the 2D background RMS
  map &mdash; the pixel-to-pixel scatter of the sky and the primary noise floor used throughout
  this report. A lower &sigma;<sub>sky</sub> means a quieter sky; the image with the smaller
  value will generally record fainter signals above 3&sigma;. Differences arise from read noise,
  dark current, sky glow, and total integration time.<br><br>
  <em>&mu;<sub>sky</sub></em> (Sky background level, ADU) is the median of the smooth background
  model itself &mdash; how bright the blank sky is before any stretch. A higher
  &mu;<sub>sky</sub> does not directly harm SNR (which depends on &sigma;, not &mu;), but it
  reduces the dynamic range available before saturation and can indicate light pollution or short
  sub-exposures. <strong>What to compare:</strong> Focus on &sigma;<sub>sky</sub> as the decisive
  quality indicator. If &sigma;<sub>sky</sub> differs by more than &approx; 30% between the two
  images, the integration depth or sky conditions were meaningfully different. &mu;<sub>sky</sub>
  is useful context &mdash; a high background paired with low &sigma; means the sky was bright
  but well-sampled; a high background paired with high &sigma; suggests insufficient exposure
  time.
</div>

<table>
  <tr><th>Sky metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  <tr><td>Sky RMS noise &sigma;<sub>sky</sub> (ADU) &mdash; lower is better</td>
      <td class="{cn_a}">{_val(pa.get("noise_median"), ".4f")}</td>
      <td class="{cn_b}">{_val(pb.get("noise_median"), ".4f")}</td></tr>
  <tr><td>Sky background &mu;<sub>sky</sub> (ADU)</td>
      <td>{_val(pa.get("background_median"), ".3f")}</td>
      <td>{_val(pb.get("background_median"), ".3f")}</td></tr>
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
{starless_html}"""

    # ── Section 10: Summary ───────────────────────────────────────────────────

    def _section_summary(self, ra: AnalysisResult, rb: AnalysisResult,
                          bw_differ: bool) -> str:
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

        rows = "".join([
            row("FWHM (px)", psf_a.get("fwhm_px"), psf_b.get("fwhm_px"),
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
        ])

        legend = ('<p><span class="metric-label-ok">✓</span> = bandwidth-independent '
                  'comparison &nbsp;&nbsp; '
                  '<span class="metric-label-warn">⚠</span> = interpret with bandwidth '
                  'context (filters had different bandwidths)</p>')

        return f"""
<h2>9. Summary &amp; Recommendations</h2>
{legend}
<table>
  <tr><th>Metric</th><th>{ra.label}</th><th>{rb.label}</th></tr>
  {rows}
</table>
<div class="info-box"><strong>How to read this table:</strong>
Green cells indicate the better value for that metric.
Red cells indicate the worse value. Metrics marked ⚠ may be influenced by the
difference in filter bandwidth and should not be used as the sole basis for
comparison.</div>"""
