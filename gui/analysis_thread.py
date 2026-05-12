from __future__ import annotations

import base64 as _base64
import concurrent.futures
from datetime import datetime as _dt
from pathlib import Path as _Path
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from core.astro_image import AstroImage
from core.models import AnalysisResult
from analysis.psf_analyzer import PSFAnalyzer
from analysis.halo_analyzer import HaloAnalyzer
from analysis.ghost_detector import GhostDetector
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

        result_a = AnalysisResult(label=img_a.label)
        result_b = AnalysisResult(label=img_b.label)

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

        if metrics.get("ghost"):
            def _ghost(img_a=img_a, img_b=img_b):
                # In parallel mode PSF may not have completed yet; fallback to 4.0 px.
                fwhm_a = (result_a.psf_metrics or {}).get("fwhm_px") or 4.0
                fwhm_b = (result_b.psf_metrics or {}).get("fwhm_px") or 4.0
                det = GhostDetector()
                result_a.ghost_metrics = det.analyze(img_a, psf_fwhm_px=fwhm_a)
                result_b.ghost_metrics = det.analyze(img_b, psf_fwhm_px=fwhm_b)
            tasks.append(("ghost", "Searching for ghosts", _ghost))

        if metrics.get("edge"):
            sl_a = self._starless_a
            sl_b = self._starless_b
            if sl_a is not None:
                sl_a.pixel_scale = img_a.pixel_scale
            if sl_b is not None:
                sl_b.pixel_scale = img_b.pixel_scale

            def _edge(src_a=sl_a or img_a, src_b=sl_b or img_b, _aligned=aligned):
                ea = EdgeAnalyzer()

                # First pass: A auto-detects top-N ROIs (or uses user-drawn roi).
                result_a.edge_metrics = ea.analyze(src_a, roi=roi)

                # Share A's ROI list with B so both cover identical pixel regions.
                rois_a = result_a.edge_metrics.get("rois_used")
                if _aligned and roi is None:
                    shared = rois_a          # list of (x0,y0,x1,y1) tuples
                elif roi is not None:
                    shared = roi             # single user-drawn tuple
                else:
                    shared = None

                # First pass on B (auto-detects angles within A's regions).
                result_b.edge_metrics = ea.analyze(src_b, roi=shared)

                # Angle normalisation: use the scan direction from whichever image
                # has the stronger overall gradient, so both ESFs measure the same
                # cross-section orientation.
                edges_a = result_a.edge_metrics.get("edges", [])
                edges_b = result_b.edge_metrics.get("edges", [])
                a_max = max((e.get("gradient_magnitude") or 0 for e in edges_a), default=0)
                b_max = max((e.get("gradient_magnitude") or 0 for e in edges_b), default=0)

                if edges_a and edges_b and a_max != b_max:
                    if b_max > a_max:
                        forced = [e["angle_rad"] for e in edges_b]
                        result_a.edge_metrics = ea.analyze(src_a, roi=shared,
                                                           forced_angles=forced)
                    else:
                        forced = [e["angle_rad"] for e in edges_a]
                        result_b.edge_metrics = ea.analyze(src_b, roi=shared,
                                                           forced_angles=forced)

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
            tasks.append(("power", "Computing power spectrum", _power))

        if metrics.get("spatial"):
            wavelet_levels = s.get("wavelet_levels", 4)
            crosshair = s.get("crosshair")

            def _spatial(sd_a=self._starless_a or img_a, sd_b=self._starless_b or img_b,
                          _ch=crosshair):
                sda = SpatialDetailAnalyzer()
                spatial = sda.analyze(sd_a, sd_b, levels=wavelet_levels, crosshair=_ch)
                spatial["used_starless_a"] = self._starless_a is not None
                spatial["used_starless_b"] = self._starless_b is not None
                result_a.spatial_metrics = spatial
                result_b.spatial_metrics = spatial  # shared reference
            tasks.append(("spatial", "Running spatial detail analysis", _spatial))

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
            ("ghost",   result_a.ghost_metrics,    result_b.ghost_metrics),
            ("edge",    result_a.edge_metrics,     result_b.edge_metrics),
            ("power",   result_a.power_metrics,    result_b.power_metrics),
            ("spatial", result_a.spatial_metrics,  result_b.spatial_metrics),
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
