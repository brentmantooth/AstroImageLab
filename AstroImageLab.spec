# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

datas = []
binaries = []
hiddenimports = []

# astropy — data files only (excludes test data to keep bundle size manageable).
# PyInstaller's built-in astropy hook covers the Cython extensions.
datas += collect_data_files("astropy",
    excludes=["**/tests/**", "**/test_data/**", "**/codata/**"])

# matplotlib — ships mpl-data (fonts, styles, colormaps)
datas += collect_data_files("matplotlib")

# Bundle the resources/ directory (PNG icon assets used at runtime)
datas += [("resources", "resources")]

# xisf package metadata (needed for importlib.metadata lookups at startup)
datas += copy_metadata("xisf")

# photutils — collect_all bundles data files, Cython .pyd binaries, and all
# submodule hidden imports together.  copy_metadata is also required because
# photutils._optional_deps queries importlib.metadata for its own package at
# startup, which fails if the dist-info directory is not present in the bundle.
ph_datas, ph_bins, ph_hidden = collect_all("photutils")
datas        += ph_datas
binaries     += ph_bins
hiddenimports += ph_hidden
datas += copy_metadata("photutils")

# pyqtgraph — ships icons/, colors/maps/*.csv and .ui files as package data, and
# resolves many of its graphicsItems submodules lazily, so static analysis misses
# them.  Same collect_all treatment as photutils above.
pg_datas, pg_bins, pg_hidden = collect_all("pyqtgraph")
datas        += pg_datas
binaries     += pg_bins
hiddenimports += pg_hidden

# Hidden imports that PyInstaller's static analysis misses
hiddenimports += [
    "xisf", "zstandard", "lz4", "lz4.block",
    # astropy subpackages accessed dynamically at runtime
    "astropy.io.fits",
    "astropy.nddata",
    "astropy.stats",
    "astropy.modeling",
    "astropy.modeling.models",
    "astropy.modeling.fitting",
    "astropy.table",
    "astropy.visualization",
    # scipy subpackages
    "scipy.ndimage",
    "scipy.interpolate",
    "scipy.optimize",
    "scipy.signal",
    # other
    "pywt",
    "astroalign",
    "PIL.Image",
    "matplotlib.backends.backend_agg",
]


a = Analysis(
    ['AstroImageLab.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # pyqtgraph probes for every Qt binding at import time; without these the
        # bundle can pick up a second, unwanted binding alongside PyQt6.
        "PySide6",
        "PySide2",
        "PyQt5",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

import sys as _sys

# .ico is Windows-only; macOS uses .icns (only relevant for BUNDLE target),
# and Linux ELF binaries don't embed icons at all.
_icon = 'resources/icon.ico' if _sys.platform == 'win32' else None

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AstroImageLab',
    icon=_icon,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Post-build: create a platform-labelled zip for distribution.
import os
import shutil

spec_dir = os.path.dirname(os.path.abspath(_sys.argv[0])) if _sys.argv else os.getcwd()
dist_dir = os.path.join(spec_dir, "dist")

if _sys.platform == "win32":
    _exe_name  = "AstroImageLab.exe"
    _zip_label = "win64"
elif _sys.platform == "darwin":
    _exe_name  = "AstroImageLab"
    _zip_label = "macos"
else:
    _exe_name  = "AstroImageLab"
    _zip_label = "linux"

_exe_path = os.path.join(dist_dir, _exe_name)
_zip_base = os.path.join(dist_dir, f"AstroImageLab-{_zip_label}")

if os.path.exists(_exe_path):
    if os.path.exists(_zip_base + ".zip"):
        os.remove(_zip_base + ".zip")
    shutil.make_archive(_zip_base, "zip", dist_dir, _exe_name)