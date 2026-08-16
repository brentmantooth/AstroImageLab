"""Render the Metric Atlas: what each metric's numbers actually look like.

The sensitivity sweep produces numbers. This turns them into the artifact that
answers the question the numbers alone do not -- "is a 1.12x local-sigma ratio
something I would see?" -- by putting the picture and the number side by side.

Three parts, in this order:

  1. Filmstrip. The *same* crop of a real image at each degradation level, each
     panel captioned with the blur applied, the resulting equivalent-FWHM
     change, and its practical label. A reader can see for themselves whether
     the panel labelled "subtle" looks subtle.
  2. Response curves. Each metric's log-ratio against blur sigma, one line per
     noise level. Shows both the knee (the response is flat below ~0.3 px then
     very steep) and the sharpness-noise interaction, and marks where the real
     filter comparisons sit on the curve.
  3. Scorecard. Monotone? Correctly signed? What dynamic range? Which metrics
     are therefore allowed to carry a practical label at all.

Filmstrip crops are rendered at 1:1 with image-rendering: pixelated. This is
not cosmetic -- a browser-scaled filmstrip resamples away exactly the
high-frequency difference the panel exists to show, which would make every
degradation level look identical.

Usage
-----
    python tools/metric_atlas.py --csv sweep_out/blur_grid.csv -o sweep_out/
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from core.fig_utils import fig_to_b64                # noqa: E402
from core.practical_significance import (            # noqa: E402
    MATERIAL, NONE, NOTICEABLE, SUBTLE,
    label_for_fwhm_change,
)
from core.stretch import normalize_for_display       # noqa: E402

# Where each real comparison sits, so the reader can locate their own case on
# the response curves. FWHM percent change, measured from the six reports.
REAL_CASES = {
    "OIII CXB/Opt": 0.5,
    "Rosette CXB/SV": 2.3,
    "SCT Ha6/12": -4.0,
    "HyperHa 6/12": -14.0,
    "deconv sharpen": -41.6,
    "gaussian blur": 176.0,
}

LABEL_COLOR = {
    NONE: "#d4edda",
    SUBTLE: "#fff3cd",
    NOTICEABLE: "#ffe0b2",
    MATERIAL: "#f8d7da",
}

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 24px; line-height: 1.6;
       color: #222; background: #fff; }
h1 { color: #1a3a5c; border-bottom: 3px solid #2d6da3; padding-bottom: 8px; }
h2 { color: #1a3a5c; border-bottom: 2px solid #2d6da3; padding-bottom: 4px;
     margin-top: 34px; }
h3 { color: #1a3a5c; margin-top: 22px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th { background: #2d6da3; color: white; padding: 8px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #dde; }
tr:nth-child(even) { background: #f0f4fa; }
.caption { font-style: italic; color: #555; font-size: 0.92em;
           margin-top: -4px; margin-bottom: 14px; }
.info { background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px;
        padding: 10px 14px; margin: 12px 0; }
.warn { background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
        padding: 10px 14px; margin: 12px 0; }
.strip { display: flex; gap: 10px; overflow-x: auto; padding: 6px 0 14px; }
.cell { flex: 0 0 auto; text-align: center; font-size: 0.82em; }
/* 1:1, never resampled -- a browser-scaled filmstrip would smooth away the
   very difference each panel exists to demonstrate. */
.cell img { image-rendering: pixelated; display: block;
            border: 1px solid #ccc; border-radius: 3px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px;
       margin-top: 5px; font-weight: bold; }
.disq { color: #b00; font-weight: bold; }
"""


def _png_b64(arr: np.ndarray) -> str:
    """8-bit STF-stretched PNG, at native pixel size."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(normalize_for_display(arr)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_filmstrip(base: np.ndarray, blurs, crop_px: int,
                    fwhm_by_sigma: dict[float, float]) -> str:
    """One row of same-crop panels at increasing blur."""
    from scipy.ndimage import gaussian_filter
    h, w = base.shape
    y0, x0 = (h - crop_px) // 2, (w - crop_px) // 2
    cells = []
    for sigma in [0.0] + list(blurs):
        img = gaussian_filter(base, sigma) if sigma > 0 else base
        patch = img[y0:y0 + crop_px, x0:x0 + crop_px]
        pct = fwhm_by_sigma.get(sigma, 0.0)
        label = label_for_fwhm_change(pct) or "—"
        colour = LABEL_COLOR.get(label, "#eceff4")
        head = "original" if sigma == 0 else f"blur &sigma;={sigma} px"
        cells.append(
            f'<div class="cell"><img src="data:image/png;base64,{_png_b64(patch)}" '
            f'width="{crop_px}" height="{crop_px}" alt="{head}">'
            f'<div><b>{head}</b></div><div>FWHM {pct:+.1f}%</div>'
            f'<span class="tag" style="background:{colour}">{label}</span></div>')
    return f'<div class="strip">{"".join(cells)}</div>'


def build_response_curves(rows: list[dict]) -> str:
    """Per-metric log-ratio vs blur sigma, one line per noise level."""
    by_metric: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        if not r.get("log_ratio"):
            continue
        try:
            sigma = float(r["factor_sharpness"].split("=")[1])
            noise = r["factor_snr"]
        except (IndexError, ValueError, KeyError):
            continue
        by_metric[r["metric_key"]][noise].append((sigma, float(r["log_ratio"])))

    if not by_metric:
        return "<p>No blur-grid rows in the CSV.</p>"

    keys = sorted(by_metric)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        for noise, pts in sorted(by_metric[key].items()):
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", ms=3, lw=1.2, label=noise)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("blur σ (px)", fontsize=8)
        ax.set_ylabel("log₁₀(A/B)", fontsize=8)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, title="added noise", title_fontsize=6)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.tight_layout()
    return f'<img src="data:image/png;base64,{fig_to_b64(fig, dpi=110)}" ' \
           f'alt="metric response curves">'


def build_scorecard(rows: list[dict]) -> str:
    """Monotone / sign / dynamic-range screen -> who may carry a label."""
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if not r.get("log_ratio") or r.get("factor_snr") != "noise=0.0":
            continue
        try:
            sigma = float(r["factor_sharpness"].split("=")[1])
        except (IndexError, ValueError):
            continue
        by[r["metric_key"]].append((sigma, float(r["log_ratio"])))

    out = ["<table><tr><th>Metric</th><th>Monotone in blur</th>"
           "<th>Sign correct</th><th>Dynamic range</th><th>Verdict</th></tr>"]
    for key in sorted(by):
        vals = [v for _, v in sorted(by[key])]
        diffs = np.diff(vals)
        monotone = bool(np.all(diffs >= -1e-12))
        # Blurring B lowers its detail, so log10(|A|/|B|) must be >= 0 and rising.
        sign_ok = all(v >= -1e-9 for v in vals)
        rng = max(vals) - min(vals)
        ok = monotone and sign_ok
        verdict = ("may carry a label" if ok
                   else '<span class="disq">disqualified</span>')
        out.append(f"<tr><td><b>{key}</b></td><td>{'yes' if monotone else 'NO'}</td>"
                   f"<td>{'yes' if sign_ok else 'NO'}</td><td>{rng:.3f}</td>"
                   f"<td>{verdict}</td></tr>")
    out.append("</table>")
    return "".join(out)


def build_html(rows, filmstrip_html: str) -> str:
    cases = "".join(
        f"<tr><td>{n}</td><td>{p:+.1f}%</td>"
        f"<td style='background:{LABEL_COLOR.get(label_for_fwhm_change(p), '#eee')}'>"
        f"{label_for_fwhm_change(p)}</td></tr>"
        for n, p in sorted(REAL_CASES.items(), key=lambda kv: abs(kv[1])))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Metric Atlas</title><style>{_CSS}</style></head><body>

<h1>Metric Atlas — what the numbers look like</h1>
<p class="caption">Generated by tools/metric_atlas.py from a
tools/sensitivity_sweep.py blur-grid run.</p>

<div class="info">
<b>What this is for.</b> Every metric in the report produces a number, and a
p-value that is saturated at p&lt;0.001 whatever the effect size. Neither tells
you whether you would see the difference. This page pairs each degradation level
with the picture that produced it, so the labels can be checked by eye.
</div>

<h2>1. The labels, and where the real comparisons fall</h2>
<p>Practical labels are attached to one axis — equivalent FWHM change — because
that is the only measured quantity that ranked all six real comparisons in the
order their owner reports seeing them. Other metrics earn a label by being
calibrated against it.</p>
<table><tr><th>Comparison</th><th>FWHM change</th><th>Label</th></tr>{cases}</table>

<h2>2. Filmstrip — same crop, increasing blur</h2>
<p class="caption">Shown at 1:1. If the panel labelled “material” does not look
materially different, or the one labelled “no visible difference expected” does,
the calibration is wrong — the pictures are the arbiter, not the numbers.</p>
{filmstrip_html}

<h2>3. Response curves</h2>
<p class="caption">One line per added-noise level. Two things to read off: the
response is nearly flat below ~0.3 px of blur and then very steep (so a single
conversion factor would be wrong nearly everywhere), and the curves for
different noise levels do not overlie each other (so a metric's sensitivity to
sharpness depends on the noise floor).</p>
{build_response_curves(rows)}

<h2>4. Scorecard — which metrics may carry a label</h2>
<div class="warn">
A metric that reverses sign partway through a monotone blur sweep cannot support
a practical label at any p-value. Those are marked disqualified here and are
left out of the calibration file, so the report shows their magnitude with no
practical reading rather than a confident wrong one.
</div>
{build_scorecard(rows)}

</body></html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path,
                   default=_REPO / "sweep_out" / "blur_grid.csv")
    p.add_argument("--image", type=str, default=None,
                   help="base frame for the filmstrip; defaults to the M31 stack")
    p.add_argument("--crop", type=int, default=260,
                   help="filmstrip panel size in px (rendered 1:1)")
    p.add_argument("-o", "--out", type=Path, default=_REPO / "sweep_out")
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"{args.csv} not found — run tools/sensitivity_sweep.py blur-grid first")
        return 1
    with args.csv.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["tier"] == "blur-grid"]
    print(f"{len(rows)} blur-grid rows")

    from tools.sensitivity_sweep import _default_blur_base, _load
    src = Path(args.image) if args.image else _default_blur_base()
    filmstrip = "<p>Base image unavailable; filmstrip skipped.</p>"
    if src and src.exists():
        base = _load(src, 900)
        # FWHM percent per sigma, read straight from the sweep so the filmstrip
        # captions and the CSV can never disagree.
        fwhm_by_sigma = {0.0: 0.0}
        for r in rows:
            try:
                s = float(r["factor_sharpness"].split("=")[1])
                if r["fwhm_pct"]:
                    fwhm_by_sigma[s] = float(r["fwhm_pct"])
            except (IndexError, ValueError):
                pass
        blurs = sorted(s for s in fwhm_by_sigma if s > 0)
        filmstrip = build_filmstrip(base, blurs, args.crop, fwhm_by_sigma)
        print(f"filmstrip: {src.name}, {len(blurs) + 1} panels")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "metric_atlas.html"
    path.write_text(build_html(rows, filmstrip), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
