# AstroImageLab — Claude Code Instructions

## Project Overview

AstroImageLab is a PyQt6 desktop application that characterises astrophotography
filters by running quantitative image analysis on one or two calibrated FITS/XISF
images. It produces a self-contained HTML report with embedded matplotlib figures.

**Owner:** Brent (solo project)
**Platform:** Windows 11, conda environment `astrolab`
**Entry point:** `AstroImageLab.py`

---

## Architecture

```
AstroImageLab.py       PyQt6 app + animated splash screen
analysis/              Metric engines — each returns a plain dict
  psf_analyzer.py      Moffat/ePSF fitting, MTF via FFT
  halo_analyzer.py     Radial halo profiles, two-component Moffat fit
  snr_analyzer.py      Global/per-star/local SNR, noise factor, sky electrons
  edge_analyzer.py     ESF/LSF edge analysis via Sobel
  power_spectrum.py    Signal-normalised 2D FFT power spectrum
  image_filters.py     Local σ maps, Laplacian of Gaussian, wavelet decomposition
  star_catalog.py      DAOStarFinder star detection and isolation filtering
core/
  astro_image.py       FITS/XISF loading, background estimation (photutils)
  models.py            40+ constants + AnalysisResult dataclass
  fig_utils.py         fig_to_b64() — embeds matplotlib figure as base64 PNG
gui/
  analysis_thread.py   QThread orchestrator; dark-mode rcParams save/restore lives here
  control_panel.py     Settings UI; settings() returns dict consumed by the thread
  report_inspector.py  Interactive side-by-side figure viewer
report/
  report_builder.py    HTML report generator; consumes AnalysisResult objects
```

---

## Key Utilities — Reuse These

| Utility | Location | Purpose |
| --- | --- | --- |
| `_info_box(body, title, open=False, style="")` | `report_builder.py` | Collapsible `<details>/<summary>` HTML panel |
| `_val(v, fmt, fallback="—")` | `report_builder.py:184` | Null-safe table cell formatter |
| `fig_to_b64(fig)` | `core/fig_utils.py` | Embeds matplotlib figure as base64 PNG string |

---

## Coding Conventions

### Dark mode — check rcParams before any hardcoded colour

```python
import matplotlib
_is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
orig_color = "white" if _is_dark else "black"
```

Place this **before** any loop that references `orig_color`. Dark mode is applied
globally in `analysis_thread.run()` via `plt.style.use("dark_background")` and
restored with `rcParams.update(_saved_params)` in the `finally` block. Never apply
dark mode at module import time — it bleeds into unrelated figure generation.

### Python string encoding — no CSS/HTML hex escapes

CSS hex escapes (`\25B6`, `\00A0`) inside Python string literals are parsed as
**octal escapes**, producing garbled characters in the HTML output. Always use:

- Literal Unicode characters directly: `▶`, `▼`, `—`
- Python Unicode escapes for invisible characters: ` ` (non-breaking space)

### FITS gain keywords — prefer EGAIN, guard against zero

`GAIN` in FITS headers is often a camera *mode index* (0, 100, 200) not the physical
e⁻/ADU conversion factor. Always:

1. Try keywords in this order: `EGAIN`, `GAIN`, `CCDGAIN`, `GAINDB`
2. Accept only values where `g > 0`

### Number formatting — use `g` for scientific values

Fixed decimal formats (`.1f`, `.3f`) silently round sub-ADU values to zero.
Use significant-figure formats for any value that can span orders of magnitude:

| Value type | Format |
| --- | --- |
| ADU sky values (σ_sky, μ_sky) | `.6g` |
| Electron sky values | `.3g` |
| Dimensionless ratios (noise factor) | `.3f` |
| Percentages | `.4f` |

### Long f-string HTML blocks

Pre-compute any Python variable **before** a `return f"""..."""` block. Do not nest
`{f"...{var}..."}` substitutions — they cause confusing `UnboundLocalError` and
syntax errors at runtime.

---

## Collaboration Rules

- **Never commit automatically.** Always ask the user for approval first.
- **Never delete `#` comments** that explain purpose, units, or rationale.
- **Use Edit, not Write,** for any existing file.
- **Ask before changing `core/models.py` constants** — many downstream callers depend on them.
- **Do not re-add PDF export.** WeasyPrint/xhtml2pdf have been intentionally removed.

---

## Build & Run

```bash
conda activate astrolab
python AstroImageLab.py          # run from source
pyinstaller AstroImageLab.spec   # build standalone .exe
```

PyQt6 **must** be installed via `pip`, not `conda` — the conda-forge PyQt6 package
uses a different DLL layout that breaks PyInstaller hook discovery on Windows.

---

## Known Pitfalls

| Pitfall | Rule |
| --- | --- |
| Inspector dark theme bleeds into report figures | Apply dark mode only in `analysis_thread.run()`, never at import time |
| `orig_color` UnboundLocalError | Assign dark-mode variables before any loop that references them |
| CSS characters garbled in HTML output | Use literal Unicode — never CSS hex escapes inside Python strings |
| Sky electron values display as 0.0 | Use `.3g`; `.1f`/`.2f` silently rounds sub-electron values to zero |
| GAIN = 0 accepted from FITS header | Guard with `if g > 0` after parsing the header value |

---

## Working Effectively with Claude Code

### The most useful problem statement format

A weak request describes *how* to fix something: "change the format string."
A strong request describes the *problem and the success criterion*:

> "Sky electron values display as `0.0` because `.1f` rounds values like `0.00024` to
> zero. The fix should show at least 2 significant figures for any positive value,
> including sub-electron magnitudes, without adding unnecessary digits for values like
> `150` or `2400`."

Include as many of these four elements as you know:

| Element | Example |
| --- | --- |
| Symptom | "Sky background shows 0.0 and 0.00" |
| Root cause | "GAIN=0 from FITS header was accepted as valid e⁻/ADU" |
| Constraint | "Table must allow hand-verification of σ/√μ = noise factor" |
| Acceptance criterion | "0.00024 must be readable; 150 must not gain unnecessary decimals" |

### When to use plan mode

Prefix your request with `/plan` for any change that:

- Touches more than one file
- Removes or restructures existing behaviour
- Involves a new pattern or convention not yet in this file

Claude will write a specification for your review before editing any code.

### Keeping this file current

After any session where a new pitfall or convention is discovered, ask:
> "Please add that to CLAUDE.md."

This file is the single source of truth for project context — keeping it current means
the next session starts with full context rather than re-deriving it from the code.
