# vibe-code-mic

Local speech-to-text that types into whichever window you had active when starting. Uses OpenAI Whisper. Designed with Claude Code use case in mind.

## Platform support

| Platform | Entry point | Whisper backend | GPU acceleration |
|----------|-------------|-----------------|------------------|
| macOS (Apple Silicon) | `python3 app_mac.py` | [mlx-whisper](https://github.com/ml-explore/mlx-examples) | MLX (automatic) |
| Windows | `python app_win.py` | openai-whisper + torch | CUDA (if available) |

### macOS requirements

- macOS 13+ (Ventura or later), Apple Silicon
- Python 3.9+
- **Accessibility permissions**: System Settings > Privacy & Security > Accessibility — add your terminal app (Terminal, iTerm2, etc.)

### Windows requirements

- Windows 10+
- Python 3.9+

## Install

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# macOS (Apple Silicon)
pip install -r requirements-mac.txt

# Windows
pip install -r requirements-win.txt
```

## Run

```bash
source venv/bin/activate  # Windows: venv\Scripts\activate

# macOS
python3 app_mac.py

# Windows
python app_win.py
```

## Build standalone executable

```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install pyinstaller

# macOS
python3 build.py
# Output: dist/vibe-code-mic.app (open with: open dist/vibe-code-mic.app)

# Windows
python build.py
# Output: dist\vibe-code-mic\vibe-code-mic.exe
```

The build script auto-detects the platform and uses the corresponding spec file (`app_mac.spec` or `app_win.spec`).

## Features

- Fully local — no data sent to the cloud
- Sends Enter after typing (optimal for Claude Code)
- Types into the window that was focused when recording started
- Menu bar countdown timer while recording (macOS)
- Clipboard preserved after paste
- Configurable hotkey, mode, device, and model
- Deafen system audio while recording
- Interactive TUI for settings (persisted as JSON5)

## Limitations

- No real-time streaming; text appears after recording stops
- 30-second max clip length
- macOS version requires Accessibility permissions

## CLI flags

Run with `--help` for all options:

```
python3 app_mac.py --help
python app_win.py --help
```
