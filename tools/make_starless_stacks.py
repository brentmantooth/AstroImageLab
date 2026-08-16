"""Build outlier-rejected stacks from raw subs and their starless counterparts.

Why this exists
---------------
The report runs Section 8 on the *starless* image whenever one is attached
(gui/analysis_thread.py: `sd_a=self._starless_a or img_a`), and the deliberately
degraded reference pair was starless too. But the sensitivity sweep was running
on star-containing stacks, and measured on M31 the top-5% detail mask is 56%
stars against an 8.8% star coverage -- a 6.4x enrichment. So the calibration was
being fitted to star response where the report measures nebula detail.

This produces the matched pair per set: `<set>_stack.fits` (clean, outliers
rejected) and `<set>_starless.fits`, so the whole sweep can be re-run on the
image class the report actually analyses. `--stars` is also saved because the
stars-only layer is the direct check that the separation worked.

Usage
-----
    python tools/make_starless_stacks.py --sets "M31 Andromeda/Lum" --crop 1024
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sensitivity_sweep import (           # noqa: E402
    _list_subs, _load_cache, _reject_outlier_frames,
)

STARLESS_CLI = Path(r"D:\Astro\SyQon\starless_cli\starless_cli.exe")


def _write_fits(arr: np.ndarray, path: Path, header_note: str) -> None:
    hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32))
    hdu.header["HISTORY"] = header_note
    # The sweep reads pixel scale off the header; without it AstroImage falls
    # back to DEFAULT_PIXEL_SCALE and flags itself estimated.
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def run_starless(src: Path, dst: Path, stars: Path | None,
                 use_mtf: bool = True, cpu: bool = False) -> bool:
    """Invoke SyQon Axiom V2.1.

    --use-mtf matters here: these stacks are *linear*, and the model expects
    display-stretched data. Without it the network sees an image whose entire
    signal sits in the bottom fraction of the range. The flag applies a
    temporary midtone stretch, infers, then inverts it, so the returned starless
    frame is linear again and directly comparable to the input.
    """
    cmd = [str(STARLESS_CLI), "--input", str(src), "--output", str(dst)]
    if stars is not None:
        cmd += ["--stars", str(stars)]
    if use_mtf:
        cmd += ["--use-mtf"]
    if cpu:
        cmd += ["--cpu"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    ok = res.returncode == 0 and dst.exists()
    print(f"      starless {'OK' if ok else 'FAILED'} "
          f"[{time.perf_counter() - t0:.1f}s]")
    if not ok:
        print((res.stdout or "")[-800:])
        print((res.stderr or "")[-800:])
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--crop", type=int, default=1024)
    p.add_argument("-o", "--out", type=Path, default=_REPO / "sweep_out" / "stacks")
    p.add_argument("--keep-outliers", action="store_true")
    p.add_argument("--outlier-hi", type=float, default=1.35)
    p.add_argument("--outlier-lo", type=float, default=0.70)
    p.add_argument("--no-mtf", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    for spec in args.sets:
        paths = _list_subs(spec)
        if len(paths) < 4:
            print(f"[skip] {spec}: {len(paths)} subs")
            continue
        print(f"\n=== {spec}: {len(paths)} subs ===", flush=True)
        cache = _load_cache(paths, args.crop)
        n_before = len(cache)
        if not args.keep_outliers:
            cache, bad = _reject_outlier_frames(
                cache, paths, args.outlier_hi, args.outlier_lo)
            if bad:
                print(f"      rejected {len(bad)} of {n_before}")
        stack = cache.mean(axis=0)
        del cache

        slug = spec.replace("/", "_").replace(" ", "")
        src = args.out / f"{slug}_stack.fits"
        dst = args.out / f"{slug}_starless.fits"
        stars = args.out / f"{slug}_starsonly.fits"
        _write_fits(stack, src,
                    f"mean of {len(paths) - (0 if args.keep_outliers else len(bad))} "
                    f"outlier-rejected subs, {args.crop}px centre crop")
        print(f"      wrote {src.name}  {stack.shape}")
        run_starless(src, dst, stars, use_mtf=not args.no_mtf, cpu=args.cpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
