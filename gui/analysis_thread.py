from __future__ import annotations

import base64 as _base64
import concurrent.futures
from datetime import datetime as _dt
from pathlib import Path as _Path
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from core.astro_image import AstroImage
from core.models import AnalysisResult, LABEL_MAX_LEN
from analysis.psf_analyzer import PSFAnalyzer
from analysis.halo_analyzer import HaloAnalyzer
from analysis.edge_analyzer import EdgeAnalyzer
from analysis.power_spectrum import PowerSpectrumAnalyzer
from analysis.image_filters import SpatialDetailAnalyzer
from report.report_builder import ReportBuilder



class AnalysisThread(QThread):
    """Runs all selected analysis engines off the main thread."""

    progress = pyqtSignal(int, str)               # (percent, status_text)
    finished = pyqtSignal(object, object, str)    # (result_a, result_b, report_path)
    error = pyqtSignal(str)

    def __init__(self, image_a: AstroImage, image_b: AstroImage,
                 settings: dict, *,
                 starless_a: AstroImage | None = None,
                 starless_b: AstroImage | None = None,
                 parent=None):
        super().__init__(parent)
        self._image_a = image_a
        self._image_b = image_b
        self._settings = settings
        self._starless_a = starless_a
        self._starless_b = starless_b

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def _execute(self) -> None:
        img_a = self._image_a
        img_b = self._image_b
        s = self._settings
        metrics = s.get("metrics", {})
        roi = s.get("roi")
        parallel = s.get("parallel", False)

        pso = s.get("pixel_scale_override")
        if pso:
            img_a.pixel_scale = pso
            img_b.pixel_scale = pso

        # Long-filename substitution: abbreviate labels before any analyzer runs so all
        # figure titles and legends use the short label from the first render.
        # Original labels are stored in result.original_label for the report info box.
        # Starless images are synced to "{main_label} (starless)" so they always match
        # the abbreviated main label in every figure that uses a starless image.
        _orig_label_a = img_a.label
        _orig_label_b = img_b.label
        if len(_orig_label_a) > LABEL_MAX_LEN:
            img_a.label = "Image A"
        if len(_orig_label_b) > LABEL_MAX_LEN:
            img_b.label = "Image B"

        # Always sync starless labels to keep them consistent with the main image label.
        _orig_starless_label_a = self._starless_a.label if self._starless_a else None
        _orig_starless_label_b = self._starless_b.label if self._starless_b else None
        if self._starless_a is not None:
            self._starless_a.label = f"{img_a.label} (starless)"
        if self._starless_b is not None:
            self._starless_b.label = f"{img_b.label} (starless)"

        result_a = AnalysisResult(
            label=img_a.label,
            original_label=_orig_label_a if img_a.label != _orig_label_a else None,
        )
        result_b = AnalysisResult(
            label=img_b.label,
            original_label=_orig_label_b if img_b.label != _orig_label_b else None,
        )

        # Alignment (always serial — must complete before analysis begins)
        self.progress.emit(2, "Aligning images…")
        aligned = self._align(img_a, img_b, result_a)

        # Build ordered task list: (metric_key, display_label, callable)
        # Each callable is a zero-arg function that writes into result_a / result_b.
        tasks: list[tuple[str, str, Callable[[], None]]] = []

        if metrics.get("psf"):
            def _psf(img_a=img_a, img_b=img_b):
                a = PSFAnalyzer()
                result_a.psf_metrics = a.analyze(img_a)
                result_b.psf_metrics = a.analyze(img_b)
            tasks.append(("psf", "Computing PSF / MTF", _psf))

        if metrics.get("halo"):
            def _halo(img_a=img_a, img_b=img_b):
                a = HaloAnalyzer()
                result_a.halo_metrics = a.analyze(img_a)
                result_b.halo_metrics = a.analyze(img_b)
            tasks.append(("halo", "Analysing halos", _halo))


        if metrics.get("edge"):
            sl_a = self._starless_a
            sl_b = self._starless_b
            if sl_a is not None:
                sl_a.pixel_scale = img_a.pixel_scale
            if sl_b is not None:
                sl_b.pixel_scale = img_b.pixel_scale

            def _edge(src_a=sl_a or img_a, src_b=sl_b or img_b):
                ea = EdgeAnalyzer()
                _ch = s.get("crosshair")

                # Always auto-detect top-N ROIs on the full image.
                # The user ROI applies only to spatial detail and power spectrum.
                result_a.edge_metrics = ea.analyze(src_a, roi=None)
                result_b.edge_metrics = ea.analyze(src_b, roi=None)

                # Angle normalisation: use the scan direction from whichever image
                # has the stronger overall gradient, so both ESFs measure the same
                # cross-section orientation.
                edges_a = result_a.edge_metrics.get("edges", [])
                edges_b = result_b.edge_metrics.get("edges", [])
                a_max = max((e.get("gradient_magnitude") or 0 for e in edges_a), default=0)
                b_max = max((e.get("gradient_magnitude") or 0 for e in edges_b), default=0)

                if edges_a and edges_b and a_max != b_max:
                    rois_a = result_a.edge_metrics.get("rois_used")
                    rois_b = result_b.edge_metrics.get("rois_used")
                    if b_max > a_max:
                        forced = [e["angle_rad"] for e in edges_b]
                        result_a.edge_metrics = ea.analyze(src_a, roi=rois_b,
                                                           forced_angles=forced)
                    else:
                        forced = [e["angle_rad"] for e in edges_a]
                        result_b.edge_metrics = ea.analyze(src_b, roi=rois_a,
                                                           forced_angles=forced)

                # Append crosshair edge ESF/LSF if a cross-section line was drawn.
                if _ch is not None:
                    xs_a = ea.analyze_crosshair(src_a, _ch)
                    xs_b = ea.analyze_crosshair(src_b, _ch)
                    if xs_a:
                        result_a.edge_metrics["edges"].append(xs_a)
                    if xs_b:
                        result_b.edge_metrics["edges"].append(xs_b)

                result_a.edge_metrics["used_starless"] = sl_a is not None
                result_b.edge_metrics["used_starless"] = sl_b is not None
            tasks.append(("edge", "Extracting edge spread function", _edge))

        if metrics.get("power"):
            def _power(ps_a=self._starless_a or img_a, ps_b=self._starless_b or img_b):
                pa = PowerSpectrumAnalyzer()
                result_a.power_metrics = pa.analyze(ps_a, roi=roi)
                result_b.power_metrics = pa.analyze(ps_b, roi=roi)
                result_a.power_metrics["used_starless"] = self._starless_a is not None
                result_b.power_metrics["used_starless"] = self._starless_b is not None
                # When a starless image was the primary input, also run on the star
                # image so the report can show the effect of stars on the spectrum.
                if self._starless_a is not None:
                    sm = pa.analyze(img_a, roi=roi)
                    result_a.power_metrics["star_power"] = {
                        "figures":        sm.get("figures"),
                        "radial_power":   sm.get("radial_power"),
                        "freq_axis":      sm.get("freq_axis"),
                        "mid_high_ratio": sm.get("mid_high_ratio"),
                    }
                if self._starless_b is not None:
                    sm = pa.analyze(img_b, roi=roi)
                    result_b.power_metrics["star_power"] = {
                        "figures":        sm.get("figures"),
                        "radial_power":   sm.get("radial_power"),
                        "freq_axis":      sm.get("freq_axis"),
                        "mid_high_ratio": sm.get("mid_high_ratio"),
                    }
            tasks.append(("power", "Computing power spectrum", _power))

        if metrics.get("spatial"):
            wavelet_levels = s.get("wavelet_levels", 4)
            crosshair = s.get("crosshair")

            def _spatial(sd_a=self._starless_a or img_a, sd_b=self._starless_b or img_b,
                          _ch=crosshair, _roi=roi):
                sda = SpatialDetailAnalyzer()
                spatial = sda.analyze(sd_a, sd_b, levels=wavelet_levels, crosshair=_ch, roi=_roi)
                spatial["used_starless_a"] = self._starless_a is not None
                spatial["used_starless_b"] = self._starless_b is not None
                result_a.spatial_metrics = spatial
                result_b.spatial_metrics = spatial  # shared reference
            tasks.append(("spatial", "Running spatial detail analysis", _spatial))

        if metrics.get("snr"):
            sl_a = self._starless_a
            sl_b = self._starless_b
            def _snr(sl_a=sl_a, sl_b=sl_b):
                from analysis.snr_analyzer import SNRAnalyzer
                sa = SNRAnalyzer()
                result_a.snr_metrics = sa.analyze(img_a)
                result_b.snr_metrics = sa.analyze(img_b)
                if sl_a is not None:
                    result_a.snr_metrics["starless"] = sa.analyze(sl_a)
                if sl_b is not None:
                    result_b.snr_metrics["starless"] = sa.analyze(sl_b)
            tasks.append(("snr", "Computing SNR maps", _snr))

        if parallel and len(tasks) > 1:
            self._run_parallel(tasks, result_a, result_b)
        else:
            self._run_serial(tasks, result_a, result_b)

        # Report generation (always serial — needs all results)
        fmt = s.get("report_format", "html")
        self.progress.emit(96, f"Generating {fmt.upper()} report…")
        report_path = ""
        try:
            builder = ReportBuilder()
            out = builder.generate(
                img_a, img_b, result_a, result_b,
                output_dir=s.get("output_dir", "."),
                open_browser=True,
                report_format=fmt,
                ref_seeing_arcsec=s.get("ref_seeing_arcsec", 2.0),
            )
            report_path = str(out)
        except Exception as e:
            import traceback as _tb
            detail = _tb.format_exc()
            result_a.warnings.append(f"Report generation failed: {e}\n{detail}")
            self.progress.emit(97, f"⚠ Report failed — see completion dialog")

        # Optional per-metric figure export
        n_exported = self._export_figures(
            result_a, result_b,
            s.get("output_dir", "."),
            s.get("export_figures", {}),
        )
        if n_exported:
            result_a.warnings.insert(
                0, f"Exported {n_exported} figure PNG(s) to {s.get('output_dir', '.')}"
            )

        self.progress.emit(100, "Done")
        self.finished.emit(result_a, result_b, report_path)

        # Restore original labels on the image objects so they remain correct
        # for any GUI display after the analysis completes.
        img_a.label = _orig_label_a
        img_b.label = _orig_label_b
        if self._starless_a is not None and _orig_starless_label_a is not None:
            self._starless_a.label = _orig_starless_label_a
        if self._starless_b is not None and _orig_starless_label_b is not None:
            self._starless_b.label = _orig_starless_label_b

    # ------------------------------------------------------------------
    # Serial runner
    # ------------------------------------------------------------------

    def _run_serial(self, tasks: list, result_a: AnalysisResult,
                    result_b: AnalysisResult) -> None:
        total = len(tasks)
        for i, (key, label, func) in enumerate(tasks):
            pct = int((i + 1) / (total + 1) * 90) + 5
            self.progress.emit(pct, f"{label}…")
            try:
                func()
            except Exception as exc:
                msg = f"{label} failed: {exc}"
                result_a.errors[key] = msg
                result_b.errors[key] = msg

    # ------------------------------------------------------------------
    # Parallel runner
    # ------------------------------------------------------------------

    def _run_parallel(self, tasks: list, result_a: AnalysisResult,
                      result_b: AnalysisResult) -> None:
        total = len(tasks)
        labels = {key: label for key, label, _ in tasks}
        self.progress.emit(5, f"Running {total} analyses in parallel…")

        future_to_key: dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
            for key, _label, func in tasks:
                future_to_key[executor.submit(func)] = key

            completed = 0
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                completed += 1
                pct = int(completed / total * 85) + 5
                try:
                    future.result()
                    self.progress.emit(pct, f"✓ {labels[key]}")
                except Exception as exc:
                    msg = f"{labels[key]} failed: {exc}"
                    result_a.errors[key] = msg
                    result_b.errors[key] = msg
                    self.progress.emit(pct, f"✗ {labels[key]} failed")

    # ------------------------------------------------------------------
    # Figure export
    # ------------------------------------------------------------------

    def _export_figures(self, result_a: AnalysisResult, result_b: AnalysisResult,
                         output_dir: str, export_flags: dict) -> int:
        out = _Path(output_dir)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        metric_pairs = [
            ("psf",     result_a.psf_metrics,     result_b.psf_metrics),
            ("halo",    result_a.halo_metrics,     result_b.halo_metrics),
            ("edge",    result_a.edge_metrics,     result_b.edge_metrics),
            ("power",   result_a.power_metrics,    result_b.power_metrics),
            ("spatial", result_a.spatial_metrics,  result_b.spatial_metrics),
            ("snr",     result_a.snr_metrics,       result_b.snr_metrics),
        ]
        exported = 0
        for key, m_a, m_b in metric_pairs:
            if not export_flags.get(key, False):
                continue
            for lbl, metrics in [(result_a.label, m_a), (result_b.label, m_b)]:
                if not metrics:
                    continue
                for fig_key, b64 in (metrics.get("figures") or {}).items():
                    if not isinstance(b64, str):
                        continue
                    safe_lbl = lbl.replace(" ", "_")
                    fname = out / f"{key}_{fig_key}_{safe_lbl}_{ts}.png"
                    fname.write_bytes(_base64.b64decode(b64))
                    exported += 1
        return exported

    # ------------------------------------------------------------------
    # Image alignment
    # ------------------------------------------------------------------

    def _align(self, img_a: AstroImage, img_b: AstroImage,
                result: AnalysisResult) -> bool:
        try:
            import astroalign as aa
            aligned_data, _ = aa.register(img_a.data, img_b.data)
            img_a.data = aligned_data
            return True
        except Exception as e:
            result.warnings.append(f"Alignment failed ({e}); proceeding unaligned.")
            return False
