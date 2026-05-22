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
        "PySide6",
        "tkinter",
        "weasyprint",
        "cairocffi",
        "cairosvg",
        "tinycss2",
        "pydyf",
        "zopfli",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AstroImageLab',
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

# Post-build: create a zip for distribution.
import os
import shutil
import sys

spec_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else os.getcwd()
dist_dir = os.path.join(spec_dir, "dist")
exe_path = os.path.join(dist_dir, "AstroImageLab.exe")
zip_path = os.path.join(dist_dir, "AstroImageLab-win64.zip")

if os.path.exists(exe_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path[:-4], "zip", dist_dir, "AstroImageLab.exe")