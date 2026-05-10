# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = []

# xisf (XISF image format support) — same treatment as AstroCross template
hiddenimports = ["xisf", "zstandard", "lz4", "lz4.block"]
datas += collect_data_files("xisf")
datas += copy_metadata("xisf")
hiddenimports += collect_submodules("xisf")
hiddenimports += collect_submodules("zstandard")
hiddenimports += collect_submodules("lz4")
binaries += collect_dynamic_libs("zstandard")
binaries += collect_dynamic_libs("lz4")

# astropy — ships coordinate/FITS reference data files
datas += collect_data_files("astropy")
hiddenimports += collect_submodules("astropy")

# photutils — submodule-heavy package
hiddenimports += collect_submodules("photutils")

# scipy — C extensions and submodules often missed by static analysis
hiddenimports += collect_submodules("scipy")

# matplotlib — ships mpl-data (fonts, styles, colormaps)
datas += collect_data_files("matplotlib")
hiddenimports += collect_submodules("matplotlib")

# pywavelets
hiddenimports += collect_submodules("pywt")

# astroalign
hiddenimports += collect_submodules("astroalign")

# Pillow
hiddenimports += collect_submodules("PIL")

# Bundle the resources/ directory (PNG assets used at runtime)
datas += [("resources", "resources")]


a = Analysis(
    ['AstroImageLab.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # AstroImageLab uses PyQt6; exclude PySide6 (also installed in conda env) to avoid
    # PyInstaller's "multiple Qt bindings" error, and Tkinter to reduce bundle size.
    excludes=["PySide6", "tkinter"],
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Post-build: create a zip containing the built executable for easy distribution.
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
