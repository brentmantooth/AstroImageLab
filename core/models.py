from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

# PR → merge to main (CI runs tests + build to verify everything works)
# Tag the merge commit on main → triggers the release workflow
# git tag v0.0.12
# git push origin v0.0.12
APP_VERSION = "0.0.12"       # semver string; bump on each GitHub release tag

# === CONSTANTS ===

MIN_STAR_SNR = 30.0
HALO_MIN_STAR_SNR = 200.0
ISOLATION_RADIUS_FWHM = 5.0
PSF_BETA_MIN = 1.0    # minimum plausible Moffat β; below this the PSF wings are unphysically steep
PSF_BETA_MAX = 10.0   # maximum plausible Moffat β; above this the fit is likely unreliable
PSF_FWHM_CLIP_NSIGMA = 3.0   # MAD multiplier for FWHM outlier sigma-clipping
HALO_FIT_RADIUS_PX = 80
SATURATION_FRACTION = 0.90
DEFAULT_PIXEL_SCALE = 1.0       # arcsec/px fallback
SEEING_WARN_FWHM_ARCS = 3.0
EDGE_ROI_HALF_WIDTH = 30
EDGE_ROI_MAP_INDICATOR_PX = 500   # px; full width of the ROI indicator box drawn on the gradient magnitude map
EDGE_ESF_MIN_MONOTONICITY = 0.3   # min net/total variation ratio; below this the ESF likely crossed >1 edge (corner/filament)
EDGE_N_TOP_EDGES = 3   # number of auto-detected gradient-peak edges (line extractions) analyzed per image
FILTER_THICKNESS_MM = 1.0   # narrowband filter substrate thickness (mm); default for UI
GLASS_REFRACTIVE_INDEX = 1.9   # dichroic filter substrate refractive index
RDF_BIN_WIDTH = 1.0            # px; annular bin width for RDF mean/std computation
POWER_SPECTRUM_NPIX = 2048   # px; size of square power spectrum maps (must be 2^n)
BACKGROUND_DISPLAY_MAX_DIM = 800   # px; stride-decimation cap shared by AstroImage.background_display()/decimation_step() and the Section 3e/3f/3g background-diagnostic figures, so arrays from different call sites land on the identical pixel grid

# === Section 3g — source-masked background estimate (analysis/source_mask.py) ===
# Background2D's per-cell sigma-clip assumes each cell is sky-dominated; that fails
# when extended nebulosity covers a large fraction of the frame. These drive the
# 4-stage source-masking diagnostic. Kernel/dilation sizes are expressed relative to
# each stage's own characteristic scale (kernel sigma, FWHM) rather than as fixed
# pixel counts, matching the SECTION8_LOCALMAX_* convention.
SOURCEMASK_SCAFFOLD_REJECT_NSIGMA = 1.5   # cut for the stage-1 sky scaffold, in units of a LOWER-HALF MAD sigma. Doubly one-sided: only cells ABOVE the fit are rejected (nebula contamination is strictly positive), and the scale is measured from the below-median residuals only (a two-sided MAD is inflated by the very contamination it must detect — at >50% coverage it grew until 62 of 64 cells survived and the scaffold stayed biased +1.26σ; the lower-half estimate cut that to +0.62σ)
SOURCEMASK_SCAFFOLD_MAX_ITERS = 6         # iterations of the stage-1 reject/refit loop; converges in ~4 on real data, this is the safety bound
SOURCEMASK_POINT_NSIGMA = 3.0             # detection threshold (× sky sigma) for the full-resolution point-source tier
SOURCEMASK_POINT_NPIXELS = 5              # min connected pixels above threshold for a point-source detection
SOURCEMASK_EXT_KERNEL_SIGMAS = (4.0, 8.0, 16.0)   # px; Gaussian smoothing sigmas for the extended-structure detection tiers, matched to low-surface-brightness nebulosity rather than to the PSF
SOURCEMASK_EXT_NSIGMA = (2.5, 2.0, 2.0)   # per-tier detection threshold (× sky sigma of the SMOOTHED image); looser than the point tier since extended structure is low-contrast by definition
SOURCEMASK_EXT_NPIX_MULT = 2.0            # npixels per extended tier = max(5, round(mult × sigma_k²)) — scale-relative, so a 16 px kernel demands a proportionally larger connected region than a 4 px one
SOURCEMASK_EXTENDED_RADIUS_MULT = 3.0     # a segment is classified "extended" when equivalent_radius > mult × fwhm_px (at FWHM 4 px: r_eq > 12 px, vs. 2-4 px for a star — wide, unambiguous separation). Deliberately NOT eccentricity-based: eccentricity separates round from elongated, not compact from extended, and puts filamentary nebulosity and trailed/blended stars at the same end
SOURCEMASK_STAR_DILATION_FWHM_MULT = 2.0  # point-source mask dilation = mult × fwhm_px, to cover PSF wings that fall below the detection threshold
SOURCEMASK_EXT_DILATION_SIGMA_MULT = 1.0  # extended mask dilation = mult × the detecting tier's kernel sigma, so the grown margin tracks the structure's own scale
SOURCEMASK_MAX_HOLE_PX = 5                # px; enclosed gaps / isolated specks up to N² area are filled / stripped before dilation (mirrors SECTION8_NEBULA_MASK_MAX_HOLE_PX; skipping the speck strip lets each noise-driven false positive balloon into a (2d+1)² blob)
SOURCEMASK_EXCLUDE_PERCENTILE = 99.0      # Background2D exclude_percentile for the masked pass, kept deliberately PERMISSIVE. A cell photutils excludes does not vanish from background_mesh — it is back-filled by interpolation, indistinguishable from a real measurement, and fitting those fills wrecked the surface (mesh RMS error 55.5 ADU vs 5.1 for genuinely-measured cells). Letting photutils report every cell it can measure, then selecting cells ourselves via SOURCEMASK_MIN_CELL_UNMASKED_FRAC, keeps the fit on measurements only and makes the selection auditable rather than hidden inside photutils
SOURCEMASK_MIN_CELL_UNMASKED_FRAC = 0.5   # a mesh cell counts as a real measurement only if at least this fraction of its pixels survived the source mask. Note the bias here is NOT truncation — the surviving pixels are genuine sky, so a half-masked cell is still unbiased; the cut exists because a cell with few surviving pixels is noisy and, at the extreme, is photutils' interpolated fill rather than a measurement at all. 0.5 balances cell count against per-cell error (measured on a 63%-masked gradient frame: 19 cells at 12 ADU RMS, vs 4 cells at 5 ADU for 0.9 — too few to constrain a plane)
SOURCEMASK_MASKED_FILTER_SIZE = 1         # filter_size for the masked Background2D pass. The default 3x3 median filter mixes neighbouring cells together, which on a sparse masked mesh means mixing in interpolated fills; the parametric fit in stage 4 does its own outlier handling via BIC and inverse-variance weights, so the pre-smoothing is not needed
SOURCEMASK_MAX_COVERAGE = 0.98            # fraction; above this the mask leaves too little sky to estimate anything — abort and report the fallback rather than returning a fabricated surface
SOURCEMASK_MIN_CELLS_QUADRIC = 20         # surviving masked mesh cells required before the order-2 quadric is allowed (BIC alone picked the quadric on 11 cells and extrapolated to a 153 ADU worst-case error)
SOURCEMASK_MIN_CELLS_PLANE = 8            # surviving cells required for the order-1 plane
SOURCEMASK_MIN_CELLS_CONSTANT = 4         # surviving cells required for the order-0 constant; below this the masked estimate aborts entirely
SOURCEMASK_N_PASSES = 1                   # detect→mask→estimate repeats. Deliberately 1: iterating DIVERGES rather than converging. Each pass re-detects against the previous pass's (lower) surface, so the mask can only grow — measured coverage ran away 54%→84%→95% on a heavy-nebula frame, and the surviving cells became too few and too spatially clustered to constrain a fit across a gradient (bias +0.10σ at 1 pass vs −2.17σ at 2). Kept as a parameter because the loop is exercised by tests, but raise it only with ground-truth measurement in hand

STD_KERNEL_SIZES = (3, 5, 10)  #originally (5, 10, 15)   # px; Gaussian kernel sizes for std dev maps
LOG_SIGMAS = (1.5, 3.0, 6.0)
ENTROPY_KERNEL_SIZES = (5, 9, 17)   # px; local window for Shannon entropy — deliberately larger than STD_KERNEL_SIZES since small windows (<25 samples) give unstable histogram estimates
WAVELET_NAME = "db4"
WAVELET_LEVELS = 4

XS_LINE_ALPHA = 0.8   # alpha for all cross-section profile lines in reports
XS_SNR_REGION_WIDTH = 15   # px; width of bright/dark sample region for cross-section SNR
EPSF_MAX_STARS = 600   # maximum candidate stars passed to EPSFBuilder; limits computation time
SECTION8_BORDER_CROP_FRACTION = 0.05   # fraction of each image dimension cropped from perimeter in Section 8 display maps
SECTION8_ANALYSIS_CMAP = "viridis"     # colormap for Section 8 A/B analysis map panels (std, LoG, wavelet)
SECTION8_LOGRATIO_EPS_PERCENTILE = 1.0   # percentile of pooled positive |A|,|B| values used as the epsilon floor in log10(|A|/|B|)
SECTION8_SCATTER_MAX_SAMPLES = 100000     # per masked population, per scale — caps render cost of Section 8g correlation scatter plots
SECTION8_ENTROPY_N_BINS = 32   # gray-level bins for local entropy histograms; max possible entropy = log2(32) = 5 bits
SECTION8_ENTROPY_CLIP_PERCENTILE = 0.5   # symmetric percentile clip (0.5-99.5) applied to each image's own normalised data before binning, so a few outlier pixels don't blow out the bin range
SECTION8_NEBULA_MASK_SIGMA = 1.7   # ×RMS above background = "Nebula" pixel classification (Section 8 masks); background cut stays fixed at 0.5×RMS
SECTION8_NEBULA_MASK_DILATION_PX = 3   # px; scipy.ndimage.binary_dilation iterations to grow the nebula mask into adjacent dim/dark nebula regions
SECTION8_NEBULA_MASK_MAX_HOLE_PX = 5   # px; enclosed background gaps up to this many pixels per side (area ≤ N²) inside the nebula mask are filled before dilation
SECTION8_LOCALMAX_FOOTPRINT_MULT = 2.0           # multiplier on each metric's own characteristic scale (kernel px / sigma px / 2**level) used as the maximum_filter footprint (peak/non-max-suppression neighbourhood) for Section 8j local-maxima detection
SECTION8_LOCALMAX_PROMINENCE_PERCENTILE = 99.0   # percentile of the (smoothed) per-scale |A|,|B|-combined peak-source array used as the minimum local-maximum height (Section 8j)
SECTION8_LOCALMAX_PRESMOOTH_FRACTION = 0.5       # fraction of each metric's own scale used as the Gaussian pre-smoothing sigma applied to the peak-source array before maximum_filter detection, to suppress single-pixel noise-driven false maxima (Section 8j); detection only — masked-region statistics are still measured on the raw, unsmoothed maps
SECTION8_LOCALMAX_REGION_FRACTION = 0.5          # fraction of the maximum_filter footprint used as the binary_dilation radius (px) grown around each detected peak pixel, so the mask covers the local neighbourhood around a peak rather than a single pixel (Section 8j)
SECTION8_LOCALMAX_DIST_MAX_SAMPLES = 1000000     # per masked population, per scale — caps render cost of the Section 8j A/B distribution violin+box figure; mean/std/significance-test stats are computed from the full population, not this capped copy
SECTION8_LOCALMAX_TOP_PERCENT = 5.0              # percentile of Image A's or Image B's own pixel-value distribution unioned (OR) into the local-maxima mask, so broad bright plateaus are captured even when they never register as an isolated local-maximum peak (Section 8j). E.g. 5.0 -> top 5% of values in A OR top 5% in B are included regardless of the peak-detection result
SECTION8_LOCAL_ENERGY_WINDOW_MULT = 4.0           # multiplier on each gradient scale's own sigma (px) used as the uniform_filter window size for the Section 8l local gradient energy map (windowed mean of |gradient|^2) — scale-relative like SECTION8_LOCALMAX_FOOTPRINT_MULT, so the window grows with the gradient's own smoothing scale instead of being one fixed pixel size for every sigma

PSF_SPATIAL_MAP_SIZE = 150       # px; long-axis resolution of FWHM / eccentricity spatial maps
PSF_SPATIAL_MAP_SMOOTH_SIGMA = 5.0   # Gaussian smoothing sigma (px) applied to spatial maps before display
LABEL_MAX_LEN = 30   # characters; filenames longer than this are replaced with "Image A"/"Image B" in all plots and legends
SPLASH_DURATION_MS = 4000   # ms; minimum time the splash screen stays visible on startup
REF_SEEING_ARCSEC = 3.0     # arcsec; reference "good seeing" FWHM for the benchmark PSF in PSF/MTF reports
REF_SEEING_BETA   = 4.77    # Moffat β for Kolmogorov atmospheric turbulence (used for reference PSF)
ABERRATION_MIN_STARS = 15          # minimum stars with orientation data required for aberration analysis
ABERRATION_OUTER_RADIUS_FRAC = 0.30   # fractional radius threshold separating inner/outer field zones

REPORT_OUTPUT_SUBFOLDER = "folder"    # HTML references PNGs in a sibling <stem>/ folder (portable, LLM-friendly)
REPORT_OUTPUT_SINGLE_FILE = "single"  # all images embedded as base64 data URIs (self-contained)


# === DATA CLASSES ===

@dataclass
class AnalysisResult:
    label: str
    original_label: str | None = None   # set when label was abbreviated due to LABEL_MAX_LEN
    psf_metrics: dict | None = None
    halo_metrics: dict | None = None
    edge_metrics: dict | None = None
    power_metrics: dict | None = None
    spatial_metrics: dict | None = None
    snr_metrics: dict | None = None
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)   # metric_key -> error message
    figures: dict[str, "matplotlib.figure.Figure"] = field(default_factory=dict)