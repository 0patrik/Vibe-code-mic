# vibe-code-mic

Local speech-to-text that types into whichever window you had active when starting. Uses OpenAI Whisper. macOS port. Designed with Claude Code use case in mind.

### Prerequisites

- macOS
- Python 3.9+
- **Accessibility permissions**: The app needs Accessibility access to listen for global hotkeys and type text. Go to **System Settings > Privacy & Security > Accessibility** and add your terminal app (e.g. Terminal, iTerm2).

### Install

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Run

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start the app
python app.py
```

### Features / options
- Fully local — no data sent to the cloud
- Sends "enter key" at the end (optimal for Claude Code)
- Support typing into an unfocused window (re-activates the original app)
- Configurable hotkey, mode, device, and model
- Deafen system audio while recording
- Interactive TUI for settings
- Settings persist as JSON5
- Apple Silicon (MPS) acceleration when available

### GPU acceleration (Apple Silicon / MPS)

On Apple Silicon Macs, the app auto-detects MPS and uses GPU acceleration — the status line will show `Ready on mps`. On Intel Macs, it falls back to CPU automatically.

### Limitations
- No real-time typing; text appears after recording stops
- macOS only (this fork)
- 30-second max clip length (Whisper limit)
- Requires Accessibility permissions for global hotkeys and typing

### CLI flags

Run `python app.py --help` for all options. Use `--settings` / `-s` to point to a different JSON5 settings file:

```
python app.py --settings ./settings.json5
```
