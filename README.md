# vibe-code-mic

Local speech-to-text that types into whichever window you had active when starting. Uses OpenAI Whisper. Windows only. Designed claude code use case in mind.

### Install

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Run

```bash
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Start the app
python app.py
```

### Features / options
- Fully local — no data sent to the cloud
- Sends "enter key" at the end (optimal for claude code)
- Support typing into an unfocused window
- Configurable hotkey, mode, device, and model
- Deafen system audio while recording
- Interactive TUI for settings
- Settings persist as JSON5
- CUDA acceleration

### GPU acceleration (CUDA)

The default install includes CUDA-enabled PyTorch. If you have an NVIDIA GPU, the app auto-detects it and uses GPU acceleration — the status line will show `Ready on cuda:0`. No extra setup needed. Without a GPU, it falls back to CPU automatically.

### Limitations
- No real-time typing; text appears after recording stops
- Windows only
- 30-second max clip length (Whisper limit)

### CLI flags

Run `python app.py --help` for all options. Use `--settings` / `-s` to point to a different JSON5 settings file:

```
python app.py --settings ./settings.json5
```