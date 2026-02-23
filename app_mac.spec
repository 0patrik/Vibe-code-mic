# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for vibe-code-mic (macOS)."""

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect mlx and mlx_whisper with all their data files and binaries
mlx_datas, mlx_binaries, mlx_hidden = collect_all('mlx')
mlx_whisper_datas, mlx_whisper_binaries, mlx_whisper_hidden = collect_all('mlx_whisper')
mlx_metal_datas, mlx_metal_binaries, mlx_metal_hidden = collect_all('mlx_metal')

all_datas = mlx_datas + mlx_whisper_datas + mlx_metal_datas
all_binaries = mlx_binaries + mlx_whisper_binaries + mlx_metal_binaries
all_hiddenimports = mlx_hidden + mlx_whisper_hidden + mlx_metal_hidden + [
    'sounddevice',
    '_sounddevice_data',
    'json5',
    'numpy',
    'AppKit',
    'Quartz',
    'Foundation',
    'CoreFoundation',
    'ApplicationServices',
    'objc',
    'PyObjCTools',
    'huggingface_hub',
    'tiktoken',
    'scipy',
    'numba',
]

a = Analysis(
    ['app_mac.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
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
