# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = []
hiddenimports = []

# astropy — ships coordinate/FITS reference data files needed at runtime
datas += collect_data_files("astropy",
    excludes=["**/tests/**", "**/test_data/**", "**/codata/**"])

# matplotlib — ships mpl-data (fonts, styles, colormaps)
datas += collect_data_files("matplotlib")

# Bundle the resources/ directory (PNG assets used at runtime)
datas += [("resources", "resources")]

# xisf metadata
datas += copy_metadata("xisf")

# Hidden imports that PyInstaller's static analysis misses
hiddenimports += [
    "xisf", "zstandard", "lz4", "lz4.block",
    # astropy subpackages accessed by string at runtime
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
    binaries=[],
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
