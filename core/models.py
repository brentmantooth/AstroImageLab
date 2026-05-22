from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

# === CONSTANTS ===

MIN_STAR_SNR = 30.0
HALO_MIN_STAR_SNR = 200.0
ISOLATION_RADIUS_FWHM = 5.0
HALO_FIT_RADIUS_PX = 80
SATURATION_FRACTION = 0.90
DEFAULT_PIXEL_SCALE = 1.0       # arcsec/px fallback
SEEING_WARN_FWHM_ARCS = 3.0
EDGE_ROI_HALF_WIDTH = 30
EDGE_ROI_MAP_INDICATOR_PX = 500   # px; full width of the ROI indicator box drawn on the gradient magnitude map
FILTER_THICKNESS_MM = 1.0   # narrowband filter substrate thickness (mm); default for UI
GLASS_REFRACTIVE_INDEX = 1.9   # dichroic filter substrate refractive index
RDF_BIN_WIDTH = 1.0            # px; annular bin width for RDF mean/std computation
POWER_SPECTRUM_NPIX = 256

STD_KERNEL_SIZES = (5, 15, 31)
LOG_SIGMAS = (1.5, 3.0, 6.0)
WAVELET_NAME = "db4"
WAVELET_LEVELS = 4

XS_LINE_ALPHA = 0.8   # alpha for all cross-section profile lines in reports
SECTION8_BORDER_CROP_FRACTION = 0.05   # fraction of each image dimension cropped from perimeter in Section 8 display maps
SECTION8_ANALYSIS_CMAP = "viridis"     # colormap for Section 8 A/B analysis map panels (std, LoG, wavelet)

PSF_SPATIAL_MAP_SIZE = 150       # px; long-axis resolution of FWHM / eccentricity spatial maps
PSF_SPATIAL_MAP_SMOOTH_SIGMA = 2.0   # Gaussian smoothing sigma (px) applied to spatial maps before display
LABEL_MAX_LEN = 30   # characters; filenames longer than this are replaced with "Image A"/"Image B" in all plots and legends


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