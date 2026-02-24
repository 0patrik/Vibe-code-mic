#!/usr/bin/env python3
"""Cross-platform build script for vibe-code-mic standalone executable.

Usage:
    pip install pyinstaller
    python build.py

Detects the current platform and runs PyInstaller with the appropriate
spec file. Outputs to dist/vibe-code-mic/.
"""

import platform
import subprocess
import sys


def main():
    system = platform.system()

    if system == "Darwin":
        spec_file = "app_mac.spec"
    elif system == "Windows":
        spec_file = "app_win.spec"
    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

    print(f"Building for {system} using {spec_file}...")

    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", spec_file]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("Build successful!")
        if system == "Darwin":
            print("Run with: open dist/vibe-code-mic.app")
        else:
            print(r"Run with: dist\vibe-code-mic\vibe-code-mic.exe")
    else:
        print("Build failed.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
