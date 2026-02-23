# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for vibe-code-mic (Windows)."""

import sys
import os

block_cipher = None

a = Analysis(
    ['app_win.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'torch',
        'whisper',
        'sounddevice',
        '_sounddevice_data',
        'json5',
        'numpy',
        'keyboard',
        'pynput',
        'pynput.keyboard._win32',
        'pynput.mouse._win32',
        'pystray',
        'pystray._win32',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vibe-code-mic',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vibe-code-mic',
)
