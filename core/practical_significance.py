"""Practical significance: is a measured difference one an imager would see?

The problem this solves
-----------------------
core.spatial_stats gives an honest confidence interval. It does not answer the
question the user actually has. On the project's six real comparisons, every
CI excludes zero by many sigma -- including the SCT Ha6/Ha12 pair whose owner
expects it to be indistinguishable, which lands 8 sigma from zero at a 3.9%
effect. Statistical significance is saturated and carries no information;
magnitude is the only thing left that discriminates.

The common currency
-------------------
Rather than invent subtle/noticeable/material cutoffs for each of ~40 metrics
separately, every metric's change is converted to one interpretable axis and
the labels are attached there once:

    sharpness -> equivalent FWHM change, in percent
    noise     -> equivalent SNR change, in dB

This is not an arbitrary choice of axis. Measured across the six real reports,
the PSF FWHM ratio is the only quantity that ranks all of them in the order
their owner reports seeing:

    case                      FWHM B/A   expected
    OIII CXB/Opt                 1.005   -
    Rosette CXB/SV               1.023   real but modest
    SCT Ha6/12                   0.960   possibly null
    HyperHa 6/12                 0.860   -
    deconvolution sharpen        0.584   visibly obvious
    Gaussian blur                2.760   visibly obvious

The Section 8 log-ratio metrics do not: the visibly-sharpened pair moves several
of them *less* than the possibly-null SCT pair does. So FWHM is the anchor, and
other metrics earn a label by being calibrated against it.

Calibration status is part of the answer
----------------------------------------
A metric with no measured response curve gets `label=None` and
`calibrated=False`, not a guessed label. tools/sensitivity_sweep.py produces
resources/metric_calibration.json; until a metric appears there, the report
should show its magnitude and say the practical reading is not yet established.
Inventing a conversion factor would be worse than admitting the gap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── The four labels ───────────────────────────────────────────────────────────
# Ordered least to most consequential; compare with PRACTICAL_ORDER.
NONE = "no visible difference expected"
SUBTLE = "subtle"
NOTICEABLE = "noticeable under close inspection"
MATERIAL = "material"

PRACTICAL_LABELS = (NONE, SUBTLE, NOTICEABLE, MATERIAL)
PRACTICAL_ORDER = {label: i for i, label in enumerate(PRACTICAL_LABELS)}

# Short forms for table cells, where the full sentence does not fit.
PRACTICAL_SHORT = {
    NONE: "none",
    SUBTLE: "subtle",
    NOTICEABLE: "noticeable",
    MATERIAL: "material",
}

# ── Currency thresholds ───────────────────────────────────────────────────────
# Equivalent FWHM change, percent. Anchored on the six real cases above: the
# three the owner reads as no-difference sit at 0.5-4%, the two deliberately
# degraded ones at 42% and 176%. 5/15/30 separates them with margin on both
# sides and matches the imaging rule of thumb that sub-5% seeing differences do
# not survive processing.
FWHM_PCT_SUBTLE = 5.0
FWHM_PCT_NOTICEABLE = 15.0
FWHM_PCT_MATERIAL = 30.0

# Equivalent SNR change, dB (amplitude convention, 20*log10 -- see CLAUDE.md's
# dB note; SNR is amplitude-like, not power-like). ~1 dB is the usual threshold
# of noticeability for a noise-level change. Checked against the same six:
# HyperHa 0.17 dB (none), OIII 0.50 (none/subtle), SCT 0.83 (subtle),
# Rosette 2.14 (noticeable) -- which is the reading its owner gives it.
SNR_DB_SUBTLE = 0.5
SNR_DB_NOTICEABLE = 1.5
SNR_DB_MATERIAL = 3.0

_CALIBRATION_PATH = (Path(__file__).resolve().parent.parent
                     / "resources" / "metric_calibration.json")


@dataclass
class PracticalVerdict:
    """A metric change expressed in practical terms.

    label       : one of PRACTICAL_LABELS, or None when uncalibrated
    calibrated  : whether a measured response curve backed the conversion
    currency    : "fwhm_pct" | "snr_db" | None
    equivalent  : the value on that currency axis, or None
    magnitude   : the raw measured change (log10 ratio, or the metric's own unit)
    note        : why a label is missing, when it is
    """
    label: str | None
    calibrated: bool
    currency: str | None
    equivalent: float | None
    magnitude: float | None
    note: str = ""

    @property
    def short(self) -> str:
        return PRACTICAL_SHORT.get(self.label, "—") if self.label else "—"

    @property
    def rank(self) -> int:
        """Sort key; uncalibrated sorts below every real label."""
        return PRACTICAL_ORDER.get(self.label, -1) if self.label else -1


def _band(value: float, subtle: float, noticeable: float, material: float) -> str:
    v = abs(value)
    if v < subtle:
        return NONE
    if v < noticeable:
        return SUBTLE
    if v < material:
        return NOTICEABLE
    return MATERIAL


def label_for_fwhm_change(pct: float | None) -> str | None:
    """Label a percentage change in FWHM. The anchor of the whole scheme."""
    if pct is None or not np.isfinite(pct):
        return None
    return _band(pct, FWHM_PCT_SUBTLE, FWHM_PCT_NOTICEABLE, FWHM_PCT_MATERIAL)


def label_for_snr_change_db(db: float | None) -> str | None:
    """Label a dB change in SNR."""
    if db is None or not np.isfinite(db):
        return None
    return _band(db, SNR_DB_SUBTLE, SNR_DB_NOTICEABLE, SNR_DB_MATERIAL)


def fwhm_change_pct(fwhm_a: float | None, fwhm_b: float | None) -> float | None:
    """Percent change from A to B. Positive means B is larger (worse)."""
    if not fwhm_a or not fwhm_b or fwhm_a <= 0:
        return None
    return 100.0 * (fwhm_b / fwhm_a - 1.0)


def snr_change_db(snr_a: float | None, snr_b: float | None) -> float | None:
    """SNR ratio in dB. Amplitude convention (20*log10): SNR is an
    amplitude-like quantity, not a power like the Section 7 radial spectrum."""
    if not snr_a or not snr_b or snr_a <= 0 or snr_b <= 0:
        return None
    return 20.0 * float(np.log10(snr_b / snr_a))


# ── Calibrated metrics ────────────────────────────────────────────────────────

@dataclass
class Calibration:
    """Measured metric response curves, keyed by metric.

    Each entry maps a monotone sequence of |log10 ratio| values to the
    equivalent FWHM change (percent) that produced them, so a measured ratio
    can be inverted onto the currency axis by interpolation. A *curve*, never a
    scale factor: the response is strongly non-linear -- on real data Local
    sigma 3 px is flat to blur sigma 0.2 px, turns over at 0.3, and is 4.7x by
    sigma 1.0 -- so a single multiplier would be wrong nearly everywhere.
    """
    metrics: dict[str, dict]
    source: str = "unknown"

    def null_floor(self, metric_key: str) -> float | None:
        """Smallest |log ratio| this metric can distinguish from nothing.

        Measured by comparing two disjoint stacks of the *same* sub-frames --
        an image pair whose correct answer is known to be zero. A reading at or
        below this is indistinguishable from comparing an image with itself,
        whatever its confidence interval says.

        This is the screen that monotonicity and dynamic range together cannot
        replace. Local entropy is monotone in blur with the largest range of any
        family, and is still useless for filter comparison: its floor is 0.10
        log units while the four real filter pairs differ by 0.003-0.080, so
        every label it produced was noise. Without this gate the calibrated
        Section 8 verdict called a possibly-null filter pair "noticeable".
        """
        entry = self.metrics.get(metric_key) or {}
        floor = entry.get("null_floor")
        return float(floor) if floor is not None else None

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Calibration":
        p = Path(path) if path is not None else _CALIBRATION_PATH
        if not p.exists():
            return cls(metrics={}, source="absent")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(metrics={}, source="unreadable")
        return cls(metrics=raw.get("metrics", {}),
                   source=str(raw.get("source", str(p))))

    @classmethod
    def empty(cls) -> "Calibration":
        return cls(metrics={}, source="empty")

    def has(self, metric_key: str) -> bool:
        return metric_key in self.metrics

    def equivalent_fwhm_pct(self, metric_key: str,
                            log_ratio: float | None) -> float | None:
        """Invert this metric's measured response curve onto the FWHM axis."""
        if log_ratio is None or not np.isfinite(log_ratio):
            return None
        entry = self.metrics.get(metric_key)
        if not entry:
            return None
        xs = np.asarray(entry.get("abs_log_ratio", []), dtype=float)
        ys = np.asarray(entry.get("fwhm_pct", []), dtype=float)
        if xs.size < 2 or xs.size != ys.size:
            return None
        order = np.argsort(xs)
        # np.interp clamps outside the measured range rather than extrapolating,
        # which is the right failure mode here: a ratio past the largest
        # calibrated point is at least as large as that point's label.
        return float(np.interp(abs(log_ratio), xs[order], ys[order]))


def verdict_for_metric(metric_key: str,
                       log_ratio: float | None,
                       calibration: Calibration | None = None) -> PracticalVerdict:
    """Practical reading for a Section 8-style log10(|A|/|B|) metric."""
    cal = calibration if calibration is not None else Calibration.empty()
    if log_ratio is None or not np.isfinite(log_ratio):
        return PracticalVerdict(None, False, None, None, log_ratio,
                                note="no measurement")
    if not cal.has(metric_key):
        return PracticalVerdict(
            None, False, None, None, log_ratio,
            note="not yet calibrated — run tools/sensitivity_sweep.py")

    # Subtract the measured floor before converting, rather than gating on it.
    #
    # A bare `<= floor -> NONE` test is not enough, because the response curves
    # are steep near the origin: SCT's wavelet_3 reads 0.0100 against a floor of
    # 0.0068 -- barely a detection -- and the curve maps that point estimate to
    # 16% equivalent FWHM, i.e. "noticeable". Attributing the floor-sized part
    # of a reading to real signal is what produces those confident wrong
    # answers. Carrying only the excess over the floor into the conversion
    # makes a floor-sized reading collapse to zero on its own, and leaves a
    # genuinely large reading essentially untouched.
    floor = cal.null_floor(metric_key)
    magnitude = abs(log_ratio)
    note = ""
    if floor is not None:
        if magnitude <= floor:
            return PracticalVerdict(NONE, True, "fwhm_pct", 0.0, log_ratio,
                                    note=f"within measurement floor ({floor:.4f})")
        magnitude -= floor
        note = f"floor-corrected (−{floor:.4f})"

    pct = cal.equivalent_fwhm_pct(metric_key, magnitude)
    return PracticalVerdict(label_for_fwhm_change(pct), True, "fwhm_pct",
                            pct, log_ratio, note=note)


def verdict_for_fwhm(fwhm_a: float | None,
                     fwhm_b: float | None) -> PracticalVerdict:
    """FWHM needs no calibration -- it *is* the currency."""
    pct = fwhm_change_pct(fwhm_a, fwhm_b)
    return PracticalVerdict(label_for_fwhm_change(pct), True, "fwhm_pct",
                            pct, pct,
                            note="" if pct is not None else "no measurement")


def verdict_for_snr(snr_a: float | None,
                    snr_b: float | None) -> PracticalVerdict:
    db = snr_change_db(snr_a, snr_b)
    return PracticalVerdict(label_for_snr_change_db(db), True, "snr_db",
                            db, db,
                            note="" if db is not None else "no measurement")


def consensus_label(verdicts) -> str | None:
    """Robust label for several metrics measuring the *same* quantity.

    Uses the median rank, not the max. Section 8's families are eleven
    strongly-correlated estimates of one thing -- how much spatial detail
    differs -- and the maximum of a set of correlated noisy estimates is just
    the noisiest member of the set, not the best summary of them.

    Measured: on the SCT Ha6/Ha12 pair, ten of eleven calibrated metrics report
    "no visible difference" and one (localgrad_1.5, a reading only 2x its own
    null floor) reports "noticeable". Taking the max made that single weak
    detection the report's headline verdict on a filter pair its owner expects
    to be indistinguishable.
    """
    ranked = sorted(v.rank for v in verdicts if v is not None and v.label)
    if not ranked:
        return None
    return PRACTICAL_LABELS[ranked[len(ranked) // 2]]


def overall_label(verdicts) -> str | None:
    """Headline across *distinct* quantities -- sharpness vs noise vs detail.

    A max here, unlike consensus_label's median, and for the opposite reason:
    these are independent axes, so a difference that is material on any one of
    them is material to the viewer even when the others are unchanged.
    Averaging across axes would hide exactly the finding the reader opened the
    report for. Feed this the per-axis consensus labels, not raw per-metric
    verdicts.
    """
    ranked = [v.rank for v in verdicts if v is not None and v.label]
    if not ranked:
        return None
    return PRACTICAL_LABELS[max(ranked)]
