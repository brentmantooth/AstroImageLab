from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure


APP_VERSION = "0.0.9"       # semver string; bump on each GitHub release tag

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

STD_KERNEL_SIZES = (3, 5, 10)  #originally (5, 10, 15)   # px; Gaussian kernel sizes for std dev maps
LOG_SIGMAS = (1.5, 3.0, 6.0)
WEBER_KERNEL_SIZES = (3, 5, 9)   # px; local kernel for Weber fraction contrast c = ΔL/L (odd values required)
WAVELET_NAME = "db4"
WAVELET_LEVELS = 4

XS_LINE_ALPHA = 0.8   # alpha for all cross-section profile lines in reports
XS_SNR_REGION_WIDTH = 15   # px; width of bright/dark sample region for cross-section SNR
EPSF_MAX_STARS = 600   # maximum candidate stars passed to EPSFBuilder; limits computation time
SECTION8_BORDER_CROP_FRACTION = 0.05   # fraction of each image dimension cropped from perimeter in Section 8 display maps
SECTION8_ANALYSIS_CMAP = "viridis"     # colormap for Section 8 A/B analysis map panels (std, LoG, wavelet)
SECTION8_LOGRATIO_EPS_PERCENTILE = 1.0   # percentile of pooled positive |A|,|B| values used as the epsilon floor in log10(|A|/|B|)
SECTION8_SCATTER_MAX_SAMPLES = 50000     # per masked population, per scale — caps render cost of Section 8g correlation scatter plots
SECTION8_NEBULA_MASK_SIGMA = 1.7   # ×RMS above background = "Nebula" pixel classification (Section 8 masks); background cut stays fixed at 0.5×RMS
SECTION8_NEBULA_MASK_DILATION_PX = 3   # px; scipy.ndimage.binary_dilation iterations to grow the nebula mask into adjacent dim/dark nebula regions
SECTION8_NEBULA_MASK_MAX_HOLE_PX = 5   # px; enclosed background gaps up to this many pixels per side (area ≤ N²) inside the nebula mask are filled before dilation
SECTION8_LOCALMAX_FOOTPRINT_MULT = 2.0           # multiplier on each metric's own characteristic scale (kernel px / sigma px / 2**level) used as the maximum_filter footprint (peak/non-max-suppression neighbourhood) for Section 8j local-maxima detection
SECTION8_LOCALMAX_PROMINENCE_PERCENTILE = 99.0   # percentile of the (smoothed) per-scale |A|,|B|-combined peak-source array used as the minimum local-maximum height (Section 8j)
SECTION8_LOCALMAX_PRESMOOTH_FRACTION = 0.5       # fraction of each metric's own scale used as the Gaussian pre-smoothing sigma applied to the peak-source array before maximum_filter detection, to suppress single-pixel noise-driven false maxima (Section 8j); detection only — masked-region statistics are still measured on the raw, unsmoothed maps
SECTION8_LOCALMAX_REGION_FRACTION = 0.5          # fraction of the maximum_filter footprint used as the binary_dilation radius (px) grown around each detected peak pixel, so the mask covers the local neighbourhood around a peak rather than a single pixel (Section 8j)
SECTION8_LOCALMAX_DIST_MAX_SAMPLES = 1000000     # per masked population, per scale — caps render cost of the Section 8j A/B distribution violin+box figure; mean/std/significance-test stats are computed from the full population, not this capped copy
SECTION8_LOCALMAX_TOP_PERCENT = 5.0              # percentile of Image A's or Image B's own pixel-value distribution unioned (OR) into the local-maxima mask, so broad bright plateaus are captured even when they never register as an isolated local-maximum peak (Section 8j). E.g. 5.0 -> top 5% of values in A OR top 5% in B are included regardless of the peak-detection result

PSF_SPATIAL_MAP_SIZE = 150       # px; long-axis resolution of FWHM / eccentricity spatial maps
PSF_SPATIAL_MAP_SMOOTH_SIGMA = 5.0   # Gaussian smoothing sigma (px) applied to spatial maps before display
LABEL_MAX_LEN = 30   # characters; filenames longer than this are replaced with "Image A"/"Image B" in all plots and legends
SPLASH_DURATION_MS = 4000   # ms; minimum time the splash screen stays visible on startup
REF_SEEING_ARCSEC = 3.0     # arcsec; reference "good seeing" FWHM for the benchmark PSF in PSF/MTF reports
REF_SEEING_BETA   = 4.77    # Moffat β for Kolmogorov atmospheric turbulence (used for reference PSF)
ABERRATION_MIN_STARS = 15          # minimum stars with orientation data required for aberration analysis
ABERRATION_OUTER_RADIUS_FRAC = 0.30   # fractional radius threshold separating inner/outer field zones


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