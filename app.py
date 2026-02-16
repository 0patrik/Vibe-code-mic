"""
vibe-code-mic Desktop App
Interactive console UI with configurable hotkey, mode, and audio device.
Hold (or toggle) a key to record speech, transcribes and types it out.
"""

import sys
import os
import argparse
import json5
import time
import threading
import ctypes
import curses
import numpy as np
import sounddevice as sd
import keyboard
import torch
import whisper
import pystray
from PIL import Image, ImageDraw

user32 = ctypes.windll.user32

WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_RETURN = 0x0D


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("hwndActive", ctypes.c_void_p),
        ("hwndFocus", ctypes.c_void_p),
        ("hwndCapture", ctypes.c_void_p),
        ("hwndMenuOwner", ctypes.c_void_p),
        ("hwndMoveSize", ctypes.c_void_p),
        ("hwndCaret", ctypes.c_void_p),
        ("rcCaret", ctypes.c_ulong * 4),
    ]


VERSION = "1.0.0"
MODELS = ["tiny", "base", "small", "medium", "large"]
SAMPLE_RATE = 16000
DEFAULT_SETTINGS_PATH = "./settings.json5"


def get_input_devices():
    """Return list of (index, name) for input devices."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append((i, d["name"]))
    return result


class SpeechToType:
    def __init__(self, settings_path=DEFAULT_SETTINGS_PATH):
        self.model = None
        self.loaded_model_idx = None
        self.icon = None
        self._key_held = False
        self._recording = False
        self._chunks = []
        self._stream = None
        self._target_hwnd = None

        # Configurable settings
        self.input_devices = get_input_devices()
        self.device_idx = 0
        self.hotkey = "f2"
        self.mode = "push"
        self.after_action = "enter"
        self.window_target = "original"
        self.model_idx = MODELS.index("small")
        self.cancel_key = "f3"
        self.no_enter_key = "f4"
        self.deafen_while_recording = "off"  # "off", "half", "on"
        self.settings_path = settings_path
        self._was_muted = False
        self._prev_volume = None
        self._skip_enter = False
        self._cancel_type_used = False

        # UI state
        self.status = "Loading model..."
        self.last_text = ""
        self.last_time = 0.0
        self._needs_redraw = True
        self._running = True

        # Set default device to the system default input
        try:
            default_idx = sd.default.device[0]
            for i, (idx, name) in enumerate(self.input_devices):
                if idx == default_idx:
                    self.device_idx = i
                    break
        except Exception:
            pass

        # Load saved settings (overrides defaults)
        self._load_settings()

    # ── Settings persistence ────────────────────────────────

    def _load_settings(self):
        path = self.settings_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json5.load(f)
            if "hotkey" in data:
                self.hotkey = data["hotkey"] or None
            if "mode" in data:
                self.mode = data["mode"]
            if "after_action" in data:
                self.after_action = data["after_action"]
            if "window_target" in data:
                self.window_target = data["window_target"]
            if "model" in data and data["model"] in MODELS:
                self.model_idx = MODELS.index(data["model"])
            if "device_name" in data:
                for i, (idx, name) in enumerate(self.input_devices):
                    if name == data["device_name"]:
                        self.device_idx = i
                        break
            if "cancel_key" in data:
                self.cancel_key = data["cancel_key"] or None
            if "no_enter_key" in data:
                self.no_enter_key = data["no_enter_key"] or None
            if "deafen_while_recording" in data:
                val = data["deafen_while_recording"]
                if val in ("off", "half", "on"):
                    self.deafen_while_recording = val
                elif val is True:
                    self.deafen_while_recording = "on"
                else:
                    self.deafen_while_recording = "off"
        except Exception:
            pass

    def _save_settings(self):
        dev_name = self.input_devices[self.device_idx][1] if self.input_devices else ""
        content = f"""\
{{
  // Global hotkey to start/stop recording
  // Examples: "f2", "f5", "scroll lock", "pause"
  "hotkey": {json5.dumps(self.hotkey)},

  // Recording mode: "push" (hold to record) or "toggle" (press to start/stop)
  "mode": {json5.dumps(self.mode)},

  // What happens after transcription is typed
  // "enter" = press Enter key (send message), "nothing" = just type the text
  "after_action": {json5.dumps(self.after_action)},

  // Where to type the transcription
  // "original" = window focused when recording started, "active" = currently focused window
  "window_target": {json5.dumps(self.window_target)},

  // Whisper model size: "tiny", "base", "small", "medium", "large"
  // Larger models are more accurate but slower and use more RAM
  "model": {json5.dumps(MODELS[self.model_idx])},

  // Audio input device name (must match a device on your system)
  "device_name": {json5.dumps(dev_name)},

  // Cancel key: cancels the current recording
  "cancel_key": {json5.dumps(self.cancel_key)},

  // Cancel but type key: stops recording and types without pressing Enter
  "no_enter_key": {json5.dumps(self.no_enter_key)},

  // Deafen (mute) system audio while recording to avoid picking up playback
  "deafen_while_recording": {json5.dumps(self.deafen_while_recording)},
}}
"""
        try:
            with open(self.settings_path, "w") as f:
                f.write(content)
        except Exception:
            pass

    # ── Audio deafen control ───────────────────────────────

    def _mute_system(self):
        try:
            import comtypes
            comtypes.CoInitialize()
            from pycaw.pycaw import AudioUtilities
            device = AudioUtilities.GetSpeakers()
            vol = device.EndpointVolume
            if self.deafen_while_recording == "on":
                self._was_muted = bool(vol.GetMute())
                if not self._was_muted:
                    vol.SetMute(True, None)
            elif self.deafen_while_recording == "half":
                self._prev_volume = vol.GetMasterVolumeLevelScalar()
                vol.SetMasterVolumeLevelScalar(self._prev_volume * 0.5, None)
        except Exception as e:
            self.status = f"Deafen error: {e}"
            self._needs_redraw = True

    def _unmute_system(self):
        try:
            import comtypes
            comtypes.CoInitialize()
            from pycaw.pycaw import AudioUtilities
            device = AudioUtilities.GetSpeakers()
            vol = device.EndpointVolume
            if self.deafen_while_recording == "on":
                if not self._was_muted:
                    vol.SetMute(False, None)
            elif self.deafen_while_recording == "half":
                if self._prev_volume is not None:
                    vol.SetMasterVolumeLevelScalar(self._prev_volume, None)
                    self._prev_volume = None
        except Exception as e:
            self.status = f"Undeafen error: {e}"
            self._needs_redraw = True

    # ── Core logic ──────────────────────────────────────────

    def load_model(self):
        model_name = MODELS[self.model_idx]
        self.status = f"Loading whisper '{model_name}' model..."
        self._needs_redraw = True
        t0 = time.perf_counter()

        # Capture stderr to show download progress in the status line
        real_stderr = sys.stderr
        app = self

        class ProgressCapture:
            def __init__(self):
                self.buf = ""
            def write(self, text):
                self.buf += text
                # tqdm writes progress like "  5%|███  | 50/1000 [00:02<00:10]"
                for line in self.buf.split("\r"):
                    line = line.strip()
                    if "%" in line and "|" in line:
                        pct = line.split("%")[0].strip()
                        try:
                            app.status = f"Downloading '{model_name}' model... {pct}%"
                            app._needs_redraw = True
                        except Exception:
                            pass
                if "\r" in self.buf:
                    self.buf = self.buf.rsplit("\r", 1)[-1]
            def flush(self):
                pass

        sys.stderr = ProgressCapture()
        try:
            # Free previous model from VRAM before loading new one
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(model_name, device=device)
        finally:
            sys.stderr = real_stderr

        elapsed = time.perf_counter() - t0
        self.loaded_model_idx = self.model_idx
        dev = self.model.device
        self.status = f"Ready on {dev} (loaded in {elapsed:.1f}s)"
        self._needs_redraw = True

    def _ready_status(self):
        if self.model:
            return f"Ready on {self.model.device}"
        return "Ready"

    def _audio_callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())

    def _hook_hotkey(self):
        if self.hotkey:
            keyboard.hook_key(self.hotkey, self._on_hotkey_event, suppress=True)
        if self.cancel_key:
            keyboard.hook_key(self.cancel_key, self._on_cancel_event, suppress=True)
        if self.no_enter_key:
            keyboard.hook_key(self.no_enter_key, self._on_no_enter_event, suppress=True)

    def _unhook_hotkey(self):
        for key in [self.hotkey, self.cancel_key, self.no_enter_key]:
            if key:
                try:
                    keyboard.unhook_key(key)
                except (ValueError, KeyError):
                    pass

    def _on_hotkey_event(self, e):
        if self.mode == "push":
            if e.event_type == "down":
                if self._key_held:
                    return
                self._key_held = True
                self._start_recording()
            else:
                self._key_held = False
                self._stop_recording()
        else:  # toggle
            if e.event_type == "down":
                if self._key_held:
                    return
                self._key_held = True
                if self._recording:
                    self._stop_recording()
                else:
                    self._start_recording()
            else:
                self._key_held = False

    def _on_cancel_event(self, e):
        if e.event_type == "down" and self._recording:
            self._recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._chunks = []
            if self.deafen_while_recording:
                self._unmute_system()
            if self.icon:
                self.icon.icon = self.create_icon_image("green")
            self._key_held = False
            self.status = "Cancelled"
            self._needs_redraw = True

    def _on_no_enter_event(self, e):
        """Stops recording and types without pressing Enter."""
        if e.event_type == "down" and self._recording:
            self._skip_enter = True
            self._cancel_type_used = True
            self._key_held = False
            self._stop_recording()

    def _start_recording(self):
        if self._recording:
            return
        self._recording = True
        self._skip_enter = False
        self._chunks = []

        hwnd = user32.GetForegroundWindow()
        thread_id = user32.GetWindowThreadProcessId(hwnd, None)
        gui_info = GUITHREADINFO()
        gui_info.cbSize = ctypes.sizeof(GUITHREADINFO)
        if user32.GetGUIThreadInfo(thread_id, ctypes.byref(gui_info)) and gui_info.hwndFocus:
            self._target_hwnd = gui_info.hwndFocus
        else:
            self._target_hwnd = hwnd

        dev_index = self.input_devices[self.device_idx][0] if self.input_devices else None
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=dev_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        if self.deafen_while_recording != "off":
            self._mute_system()
        self.status = "Recording..."
        self._needs_redraw = True
        if self.icon:
            self.icon.icon = self.create_icon_image("red")

    def _stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self.icon:
            self.icon.icon = self.create_icon_image("green")

        if self.deafen_while_recording != "off":
            self._unmute_system()

        if not self._chunks:
            self.status = self._ready_status()
            self._needs_redraw = True
            return

        audio = np.concatenate(self._chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3:
            self.status = "Too short, skipped"
            self._needs_redraw = True
            return

        self.status = f"Transcribing {duration:.1f}s of audio..."
        self._needs_redraw = True
        threading.Thread(target=self._transcribe_and_type, args=(audio,), daemon=True).start()

    def _transcribe_and_type(self, audio):
        t0 = time.perf_counter()
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_padded, n_mels=self.model.dims.n_mels).to(self.model.device)
        use_fp16 = self.model.device.type == "cuda"
        options = whisper.DecodingOptions(language="en", fp16=use_fp16)
        result = whisper.decode(self.model, mel, options)
        text = result.text.strip()
        elapsed = time.perf_counter() - t0

        if not text:
            self.status = "No speech detected"
            self._needs_redraw = True
            return

        self.last_text = text
        self.last_time = elapsed
        if self._cancel_type_used:
            self.status = "Typed without Enter (cancel+type)"
            self._cancel_type_used = False
        else:
            self.status = self._ready_status()
        self._needs_redraw = True

        if self.window_target == "active":
            hwnd = user32.GetForegroundWindow()
            thread_id = user32.GetWindowThreadProcessId(hwnd, None)
            gui_info = GUITHREADINFO()
            gui_info.cbSize = ctypes.sizeof(GUITHREADINFO)
            if user32.GetGUIThreadInfo(thread_id, ctypes.byref(gui_info)) and gui_info.hwndFocus:
                hwnd = gui_info.hwndFocus
        else:
            hwnd = self._target_hwnd

        if hwnd:
            for ch in text:
                user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)
                time.sleep(0.005)
            if self.after_action == "enter" and not self._skip_enter:
                time.sleep(0.05)
                user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0x001C0001)
                user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0xC01C0001)

    def create_icon_image(self, color="green"):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        colors = {"green": "#22c55e", "red": "#ef4444", "gray": "#6b7280"}
        draw.ellipse([8, 8, 56, 56], fill=colors.get(color, color))
        draw.rectangle([27, 16, 37, 36], fill="white")
        draw.ellipse([24, 28, 40, 44], fill="white")
        draw.rectangle([27, 38, 37, 48], fill="white")
        draw.rectangle([30, 44, 34, 52], fill="white")
        draw.rectangle([24, 50, 40, 54], fill="white")
        return img

    def quit_app(self, icon=None, item=None):
        self.status = "Quitting..."
        self._needs_redraw = True
        self._save_settings()
        self._running = False
        keyboard.unhook_all()
        if self.icon:
            self.icon.stop()

    def run_tray(self):
        menu = pystray.Menu(
            pystray.MenuItem("Quit", self.quit_app),
        )
        self.icon = pystray.Icon(
            "vibe-code-mic",
            self.create_icon_image("green"),
            "vibe-code-mic",
            menu,
        )
        self.icon.run()

    # ── Descriptions ────────────────────────────────────────

    MODEL_INFO = {
        "tiny":   "openai/whisper-tiny — 39M params, ~1GB RAM, ~75MB disk. Fastest, least accurate. Good for quick tests.",
        "base":   "openai/whisper-base — 74M params, ~1GB RAM, ~150MB disk. Fast with decent accuracy for clear speech.",
        "small":  "openai/whisper-small — 244M params, ~2GB RAM, ~500MB disk. Recommended for most computers. Best balance of speed and accuracy.",
        "medium": "openai/whisper-medium — 769M params, ~5GB RAM, ~1.5GB disk. High accuracy, slower transcription.",
        "large":  "openai/whisper-large-v3 — 1550M params, ~10GB RAM, ~3GB disk. Best accuracy, requires significant resources.",
    }

    def _get_description(self, selected):
        if selected == 0:
            return f"vibe-code-mic v{VERSION}. Local speech recognition powered by OpenAI Whisper. All processing happens on your machine — no data is sent to the cloud."
        elif selected == 1:
            if self.input_devices:
                return "Input device for recording. Use Left/Right to cycle through available microphones and audio inputs on your system."
            return "No input devices found."
        elif selected == 2:
            model_name = MODELS[self.model_idx]
            return self.MODEL_INFO.get(model_name, "")
        elif selected == 3:
            return "Global hotkey to start/stop recording. Press Enter to rebind, then press any key to set it as the new hotkey."
        elif selected == 4:
            return "A single tap during recording cancels it. Audio is discarded and nothing is typed. In toggle mode, press this before pressing the record key again."
        elif selected == 5:
            return "A single tap during recording stops it and types the text without pressing Enter. Useful when you want to edit the text before sending. In toggle mode, press this before pressing the record key again."
        elif selected == 6:
            if self.mode == "push":
                return "Push to hold: hold the key to record, release to stop and transcribe. Good for quick voice commands."
            else:
                return "Toggle: press once to start recording, press again to stop and transcribe. Good for longer dictation."
        elif selected == 7:
            if self.after_action == "enter":
                return "Presses Enter after typing the transcription. Useful for sending messages in chat apps."
            else:
                return "Just types the text with no Enter key pressed after. Useful for filling in text fields or documents."
        elif selected == 8:
            if self.window_target == "original":
                return "Types into the window that was focused when you started recording, even if you switch apps during transcription."
            else:
                return "Types into whatever window is active when transcription finishes. The target may change if you switch apps."
        elif selected == 9:
            if self.deafen_while_recording == "on":
                return "System audio will be fully muted while recording and restored when you stop. Prevents your speakers from being picked up by the mic."
            elif self.deafen_while_recording == "half":
                return "System audio volume will be reduced by 50% while recording and restored when you stop. Reduces speaker bleed without losing all audio."
            else:
                return "System audio stays on during recording. Enable this if your microphone picks up sounds from your speakers."
        return ""

    # ── Curses TUI ──────────────────────────────────────────

    def draw_ui(self, stdscr, selected, rebinding):
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        def safe_addstr(y, x, text, attr=0):
            if y < h and x < w:
                stdscr.addnstr(y, x, text, w - x - 1, attr)

        def wrap_text(text, avail):
            """Word-wrap text to fit in avail columns. Returns list of lines."""
            if avail <= 0:
                return []
            result = []
            for paragraph in text.split('\n'):
                words = paragraph.split(' ')
                cur = ''
                for word in words:
                    if not cur:
                        cur = word
                    elif len(cur) + 1 + len(word) <= avail:
                        cur += ' ' + word
                    else:
                        result.append(cur)
                        cur = word
                    # Force-break words longer than avail
                    while len(cur) > avail:
                        result.append(cur[:avail])
                        cur = cur[avail:]
                if cur or not paragraph:
                    result.append(cur)
            return result if result else ['']

        def draw_wrapped(y, x, text, attr=0):
            """Draw text wrapping across multiple lines. Returns number of lines used."""
            avail = w - x - 1
            lines = wrap_text(text, avail)
            for i, line in enumerate(lines):
                if (y + i) < h:
                    safe_addstr(y + i, x, line, attr)
            return max(len(lines), 1)

        safe_addstr(0, 0, "=== vibe-code-mic (Windows) ===", curses.A_BOLD)

        # About (not configurable, info-only)
        prefix = "> " if selected == 0 else "  "
        safe_addstr(2, 0, f"{prefix}About:         ", curses.A_BOLD if selected == 0 else 0)
        safe_addstr(2, 17, f" v{VERSION} ", curses.color_pair(4))

        # Device
        dev_name = self.input_devices[self.device_idx][1] if self.input_devices else "(none)"
        prefix = "> " if selected == 1 else "  "
        attr = curses.A_REVERSE if selected == 1 else 0
        safe_addstr(3, 0, f"{prefix}Audio Device:  ", curses.A_BOLD if selected == 1 else 0)
        safe_addstr(3, 17, f" < {dev_name} > ", attr)

        # Model
        prefix = "> " if selected == 2 else "  "
        attr = curses.A_REVERSE if selected == 2 else 0
        model_name = MODELS[self.model_idx]
        safe_addstr(4, 0, f"{prefix}Model:         ", curses.A_BOLD if selected == 2 else 0)
        model_hint = "(current)" if self.model_idx == self.loaded_model_idx else "(Enter to reload)"
        safe_addstr(4, 17, f" < {model_name} > {model_hint} ", attr)

        # Hotkey
        prefix = "> " if selected == 3 else "  "
        attr = curses.A_REVERSE if selected == 3 else 0
        safe_addstr(5, 0, f"{prefix}Record Key:    ", curses.A_BOLD if selected == 3 else 0)
        if rebinding == "hotkey":
            safe_addstr(5, 17, " [press a key / Esc to disable] ", curses.A_BLINK | curses.A_REVERSE)
        elif self.hotkey:
            safe_addstr(5, 17, f" {self.hotkey.upper()} (Enter to change) ", attr)
        else:
            safe_addstr(5, 17, " Disabled (Enter to set) ", attr)

        # Cancel Key
        prefix = "> " if selected == 4 else "  "
        attr = curses.A_REVERSE if selected == 4 else 0
        safe_addstr(6, 0, f"{prefix}Cancel Key:    ", curses.A_BOLD if selected == 4 else 0)
        if rebinding == "cancel":
            safe_addstr(6, 17, " [press a key / Esc to disable] ", curses.A_BLINK | curses.A_REVERSE)
        elif self.cancel_key:
            safe_addstr(6, 17, f" {self.cancel_key.upper()} (Enter to change) ", attr)
        else:
            safe_addstr(6, 17, " Disabled (Enter to set) ", attr)

        # Cancel But Type Key
        prefix = "> " if selected == 5 else "  "
        attr = curses.A_REVERSE if selected == 5 else 0
        safe_addstr(7, 0, f"{prefix}Cancel+Type Key:", curses.A_BOLD if selected == 5 else 0)
        if rebinding == "no_enter":
            safe_addstr(7, 18, " [press a key / Esc to disable] ", curses.A_BLINK | curses.A_REVERSE)
        elif self.no_enter_key:
            safe_addstr(7, 18, f" {self.no_enter_key.upper()} (Enter to change) ", attr)
        else:
            safe_addstr(7, 18, " Disabled (Enter to set) ", attr)

        # Mode
        prefix = "> " if selected == 6 else "  "
        attr = curses.A_REVERSE if selected == 6 else 0
        mode_label = "Push to hold" if self.mode == "push" else "Toggle"
        safe_addstr(8, 0, f"{prefix}Mode:          ", curses.A_BOLD if selected == 6 else 0)
        safe_addstr(8, 17, f" < {mode_label} > ", attr)

        # After action
        prefix = "> " if selected == 7 else "  "
        attr = curses.A_REVERSE if selected == 7 else 0
        after_label = "Press Enter" if self.after_action == "enter" else "Do nothing"
        safe_addstr(9, 0, f"{prefix}After Record:  ", curses.A_BOLD if selected == 7 else 0)
        safe_addstr(9, 17, f" < {after_label} > ", attr)

        # Window target
        prefix = "> " if selected == 8 else "  "
        attr = curses.A_REVERSE if selected == 8 else 0
        target_label = "Original window" if self.window_target == "original" else "Active window"
        safe_addstr(10, 0, f"{prefix}Type Target:   ", curses.A_BOLD if selected == 8 else 0)
        safe_addstr(10, 17, f" < {target_label} > ", attr)

        # Deafen while recording
        prefix = "> " if selected == 9 else "  "
        attr = curses.A_REVERSE if selected == 9 else 0
        deafen_labels = {"off": "Off", "half": "50%", "on": "On"}
        deafen_label = deafen_labels.get(self.deafen_while_recording, "Off")
        safe_addstr(11, 0, f"{prefix}Deafen on Rec: ", curses.A_BOLD if selected == 9 else 0)
        safe_addstr(11, 17, f" < {deafen_label} > ", attr)

        # Description for selected option (wraps across lines)
        desc = self._get_description(selected)
        desc_lines = draw_wrapped(13, 2, desc, curses.color_pair(4))

        # Bottom-pinned: help line at very bottom, status above it, last text fills middle
        help_y = h - 1
        safe_addstr(help_y, 0, "  Up/Down: navigate | Left/Right: change | Ctrl+C: quit")

        # Status with color (word-wrapped, grows upward from help line)
        if "Recording" in self.status:
            status_color = curses.color_pair(2) | curses.A_BOLD
        elif "Transcribing" in self.status:
            status_color = curses.color_pair(3) | curses.A_BOLD
        elif "Quitting" in self.status:
            status_color = curses.color_pair(3) | curses.A_BOLD
        elif "Ready" in self.status:
            status_color = curses.color_pair(1)
        else:
            status_color = curses.color_pair(4)
        model_label = MODELS[self.loaded_model_idx] if self.loaded_model_idx is not None else "none"
        status_text = f"{self.status}  [model: {model_label}]"
        status_avail = w - 10 - 1
        status_lines = wrap_text(status_text, status_avail)
        status_line_count = max(len(status_lines), 1)
        status_y = help_y - 1 - status_line_count
        safe_addstr(status_y, 0, "  Status: ")
        for i, sline in enumerate(status_lines):
            if (status_y + i) < h:
                safe_addstr(status_y + i, 10, sline, status_color)

        # Last transcription — fills space between description and status
        if self.last_text:
            last_start_y = 13 + desc_lines + 1
            last_end_y = status_y - 1
            avail_lines = last_end_y - last_start_y
            avail_width = w - 12
            if avail_lines > 0 and avail_width > 0:
                safe_addstr(last_start_y, 0, "  Last:   ", curses.color_pair(4))
                # Word-wrap the transcription text across available lines
                text_with_time = f'"{self.last_text}" ({self.last_time:.1f}s)'
                wrapped = wrap_text(text_with_time, avail_width)
                for line, chunk in enumerate(wrapped[:avail_lines]):
                    safe_addstr(last_start_y + line, 10, chunk, curses.A_BOLD)

        stdscr.refresh()

    def run_curses(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # ready
        curses.init_pair(2, curses.COLOR_RED, -1)      # recording
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # transcribing
        curses.init_pair(4, curses.COLOR_CYAN, -1)     # info

        NUM_SETTINGS = 10
        selected = 0  # 0=about, 1=device, 2=model, 3=hotkey, 4=cancel_key, 5=no_enter_key, 6=mode, 7=after, 8=target, 9=deafen
        rebinding = False

        # Load model in background
        threading.Thread(target=self._load_and_hook, daemon=True).start()

        while self._running:
            if self._needs_redraw or True:  # always redraw (cheap at 10fps)
                self.draw_ui(stdscr, selected, rebinding)
                self._needs_redraw = False

            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                self.status = "Quitting..."
                self.draw_ui(stdscr, selected, rebinding)
                stdscr.refresh()
                time.sleep(0.3)
                break
            except Exception:
                key = -1

            if key == -1:
                continue

            # Ctrl+C
            if key == 3:
                break

            if rebinding:
                rebinding = False
                self._needs_redraw = True
                continue

            if key == curses.KEY_UP:
                selected = (selected - 1) % NUM_SETTINGS
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % NUM_SETTINGS
            elif key == curses.KEY_LEFT or key == curses.KEY_RIGHT:
                if selected == 1 and self.input_devices:
                    if key == curses.KEY_LEFT:
                        self.device_idx = (self.device_idx - 1) % len(self.input_devices)
                    else:
                        self.device_idx = (self.device_idx + 1) % len(self.input_devices)
                elif selected == 2:
                    if key == curses.KEY_LEFT:
                        self.model_idx = (self.model_idx - 1) % len(MODELS)
                    else:
                        self.model_idx = (self.model_idx + 1) % len(MODELS)
                elif selected == 6:
                    self.mode = "toggle" if self.mode == "push" else "push"
                elif selected == 7:
                    self.after_action = "nothing" if self.after_action == "enter" else "enter"
                elif selected == 8:
                    self.window_target = "active" if self.window_target == "original" else "original"
                elif selected == 9:
                    deafen_options = ["off", "half", "on"]
                    cur = deafen_options.index(self.deafen_while_recording) if self.deafen_while_recording in deafen_options else 0
                    if key == curses.KEY_LEFT:
                        cur = (cur - 1) % len(deafen_options)
                    else:
                        cur = (cur + 1) % len(deafen_options)
                    self.deafen_while_recording = deafen_options[cur]
                self._save_settings()
            elif key == 10 or key == curses.KEY_ENTER:  # Enter
                if selected == 2:
                    self._unhook_hotkey()
                    self._save_settings()
                    threading.Thread(target=self._load_and_hook, daemon=True).start()
                elif selected == 3:
                    rebinding = "hotkey"
                    self._unhook_hotkey()
                    self.status = "Press the new record key..."
                    self._needs_redraw = True
                    threading.Thread(target=lambda: self._rebind_key("hotkey"), daemon=True).start()
                elif selected == 4:
                    rebinding = "cancel"
                    self._unhook_hotkey()
                    self.status = "Press the new cancel key..."
                    self._needs_redraw = True
                    threading.Thread(target=lambda: self._rebind_key("cancel_key"), daemon=True).start()
                elif selected == 5:
                    rebinding = "no_enter"
                    self._unhook_hotkey()
                    self.status = "Press the new cancel+type key..."
                    self._needs_redraw = True
                    threading.Thread(target=lambda: self._rebind_key("no_enter_key"), daemon=True).start()

    def _rebind_key(self, attr="hotkey"):
        """Wait for a single keypress and rebind the specified key. Escape to disable."""
        event = keyboard.read_event(suppress=False)
        while event.event_type != "down":
            event = keyboard.read_event(suppress=False)
        if event.name == "esc":
            setattr(self, attr, None)
            self._hook_hotkey()
            self._save_settings()
            self.status = "Key disabled"
        else:
            new_key = event.name
            setattr(self, attr, new_key)
            self._hook_hotkey()
            self._save_settings()
            self.status = f"Key set to {new_key.upper()}"
        self._needs_redraw = True

    def _load_and_hook(self):
        self.load_model()
        self._hook_hotkey()

    def run(self):
        threading.Thread(target=self.run_tray, daemon=True).start()
        curses.wrapper(self.run_curses)
        print("Quitting...")
        self.quit_app()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="vibe-code-mic",
        description="Local speech-to-text that types into your active window. Uses OpenAI Whisper for transcription.",
    )
    parser.add_argument(
        "--settings", "-s",
        default=DEFAULT_SETTINGS_PATH,
        help=f"path to JSON5 settings file (default: {DEFAULT_SETTINGS_PATH})",
    )
    parser.add_argument(
        "--model", "-m",
        choices=MODELS,
        help="whisper model to use (overrides settings file)",
    )
    parser.add_argument(
        "--hotkey", "-k",
        help="hotkey for recording (e.g. f2, f5, pause) (overrides settings file)",
    )
    parser.add_argument(
        "--mode",
        choices=["push", "toggle"],
        help="recording mode: push (hold to record) or toggle (press to start/stop) (overrides settings file)",
    )
    parser.add_argument(
        "--device", "-d",
        help="audio input device name or substring (overrides settings file)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = SpeechToType(settings_path=args.settings)
    if args.model:
        app.model_idx = MODELS.index(args.model)
    if args.hotkey:
        app.hotkey = args.hotkey
    if args.mode:
        app.mode = args.mode
    if args.device:
        for i, (idx, name) in enumerate(app.input_devices):
            if args.device.lower() in name.lower():
                app.device_idx = i
                break
    app.run()
