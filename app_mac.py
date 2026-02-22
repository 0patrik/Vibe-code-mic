#!/usr/bin/env python3
"""
vibe-code-mic Desktop App (macOS)
Interactive console UI with configurable hotkey, mode, and audio device.
Hold (or toggle) a key to record speech, transcribes and types it out.

Security model
--------------
Permissions required:
  - Accessibility (System Settings > Privacy & Security) — used for the
    CGEvent tap that listens for the global hotkey and for AXUIElement
    calls that read/raise other applications' windows during paste.
  - Microphone — used to record audio while the hotkey is held/toggled.

What the app captures and when:
  - Audio: only while the user is actively recording (hotkey held/toggled).
  - Keystrokes: intercepted *only* during the ~0.5 s paste window so they
    can be replayed afterwards.  Outside that window the event tap only
    inspects the hotkey/cancel key codes and passes everything else through.

What the app does NOT do:
  - No keystroke logging — non-hotkey keystrokes are never stored or read.
  - No network access — all transcription runs locally via mlx-whisper.
  - No disk writes except the JSON5 settings file.

Data flow:
  record audio → transcribe locally (mlx-whisper) → copy to clipboard →
  switch to target window → Cmd+V paste → (optional Enter) → switch back →
  replay buffered keystrokes → restore original clipboard.

External processes:
  - ``osascript`` is invoked for system volume get/set/mute only, with
    hardcoded AppleScript one-liners (no user-controlled arguments).

Sections that touch privileged macOS APIs are marked with
``# ~~~ SECURITY-SENSITIVE ~~~`` so reviewers can grep for them.
"""

import sys
import os
import argparse
import json5
import time
import threading
import subprocess
import curses
import numpy as np
import sounddevice as sd
import mlx_whisper

import AppKit
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
import Quartz

# Prevent the app from showing a bouncing icon in the Dock
NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    CGEventSourceCreate,
    CGEventSourceGetSourceStateID,
    CGEventGetIntegerValueField,
    CGEventCreateCopy,
    CGEventTapCreate,
    CGEventTapEnable,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventFlagsChanged,
    kCGEventTapDisabledByTimeout,
    kCGEventTapDisabledByUserInput,
    kCGHIDEventTap,
    kCGHeadInsertEventTap,
    kCGEventTapOptionDefault,
    kCGEventFlagMaskCommand,
    kCGEventSourceStatePrivate,
)
from Foundation import NSRunLoop, NSDate, CFRunLoopGetCurrent, CFRunLoopRunInMode
from AppKit import NSPasteboardTypeString, NSStatusBar, NSVariableStatusItemLength, NSFont
import CoreFoundation
import ApplicationServices

# ── macOS key code mapping ──────────────────────────────────────────

KEYCODE_MAP = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05,
    "z": 0x06, "x": 0x07, "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C,
    "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10, "t": 0x11, "1": 0x12,
    "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18,
    "9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D, "]": 0x1E,
    "o": 0x1F, "u": 0x20, "[": 0x21, "i": 0x22, "p": 0x23, "l": 0x25,
    "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29, "\\": 0x2A, ",": 0x2B,
    "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "`": 0x32, "§": 0x0A,
    "\u00a7": 0x0A,
    "return": 0x24, "enter": 0x24, "tab": 0x30, "space": 0x31,
    "delete": 0x33, "backspace": 0x33, "escape": 0x35, "esc": 0x35,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60,
    "f6": 0x61, "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D,
    "f11": 0x67, "f12": 0x6F, "f13": 0x69, "f14": 0x6B, "f15": 0x71,
    "f16": 0x6A, "f17": 0x40, "f18": 0x4F, "f19": 0x50, "f20": 0x5A,
    "up": 0x7E, "down": 0x7D, "left": 0x7B, "right": 0x7C,
    "home": 0x73, "end": 0x77, "pageup": 0x74, "pagedown": 0x79,
    "pause": 0x71,
}

# Reverse map: keycode -> name (prefer shorter canonical names)
KEYCODE_TO_NAME = {}
for _name, _code in sorted(KEYCODE_MAP.items(), key=lambda x: -len(x[0])):
    KEYCODE_TO_NAME[_code] = _name
for _name, _code in sorted(KEYCODE_MAP.items(), key=lambda x: len(x[0])):
    KEYCODE_TO_NAME[_code] = _name


VERSION = "1.0.0"
MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]

# Map model short names to mlx-community HuggingFace repos
MLX_MODEL_REPOS = {
    "tiny":   "mlx-community/whisper-tiny-mlx",
    "base":   "mlx-community/whisper-base-mlx",
    "small":  "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large":  "mlx-community/whisper-large-v3-mlx",
    "turbo":  "mlx-community/whisper-turbo",
}
SAMPLE_RATE = 16000
DEFAULT_SETTINGS_PATH = "./settings.json5"

# ── Timing & threshold constants ─────────────────────────────────
MAX_RECORDING_DURATION = 30.0        # seconds - auto-stop recording
MIN_AUDIO_DURATION = 0.3             # seconds - skip if shorter

# Paste/type workflow timing (seconds)
PASTE_KEY_DOWN_DELAY = 0.05          # after Cmd+V key down
PASTE_PRE_ENTER_DELAY = 0.10         # before pressing Enter
ENTER_KEY_DOWN_DELAY = 0.05          # after Enter key down
PASTE_SETTLE_DELAY = 0.20            # wait for paste to be consumed
CLIPBOARD_RESTORE_DELAY = 0.30       # extra wait before restoring clipboard
SWITCH_BACK_DELAY = 0.10             # after switching back to original app
PRE_REPLAY_DELAY = 0.05              # before replaying captured keystrokes
KEYSTROKE_REPLAY_INTERVAL = 0.008    # between each replayed keystroke

# Window switch polling
WINDOW_SWITCH_POLL_INTERVAL = 0.050  # CFRunLoop pump per poll iteration
WINDOW_SWITCH_MAX_POLLS = 20         # max iterations waiting for app switch
POST_SWITCH_SETTLE = 0.050           # settle after confirming switch

# CFRunLoop pump internals
PUMP_INTERVAL = 0.010                # CFRunLoop iteration length
PUMP_SLEEP = 0.005                   # sleep between pump iterations

# Event capture
MAX_CAPTURED_EVENTS = 256            # buffer limit for captured keystrokes

# Key rebinding
REBIND_POLL_INTERVAL = 0.1           # seconds between checks
REBIND_TIMEOUT_POLLS = 100           # max iterations (10s total)

# UI
CURSES_INPUT_TIMEOUT_MS = 100        # getch timeout
MAIN_LOOP_PUMP_DURATION = 0.01       # CFRunLoop pump in main loop
QUIT_DELAY = 0.3                     # pause before quitting


def get_input_devices():
    """Return list of (index, name) for input devices."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append((i, d["name"]))
    return result


# ── macOS window helpers ── # ~~~ SECURITY-SENSITIVE ~~~
# These functions use AXUIElement to read and manipulate other apps' windows.

def get_ax_focused_window(pid):
    """Get the focused window AXUIElement for a given PID."""
    app_ref = ApplicationServices.AXUIElementCreateApplication(pid)
    err, window_ref = ApplicationServices.AXUIElementCopyAttributeValue(
        app_ref, "AXFocusedWindow", None
    )
    if err == 0 and window_ref is not None:
        return app_ref, window_ref
    return app_ref, None


def ax_raise_window(window_ref):
    """Raise an AXUIElement window."""
    ApplicationServices.AXUIElementPerformAction(window_ref, "AXRaise")


def ax_get_title(window_ref):
    """Get the title of an AXUIElement window."""
    err, title = ApplicationServices.AXUIElementCopyAttributeValue(
        window_ref, "AXTitle", None
    )
    if err == 0 and title:
        return str(title)
    return "(untitled)"


def ax_set_focused_window(app_ref, window_ref):
    """Set the focused window attribute on an app element."""
    ApplicationServices.AXUIElementSetAttributeValue(
        app_ref, "AXFocusedWindow", window_ref
    )


# ── macOS volume control via osascript ── # ~~~ SECURITY-SENSITIVE ~~~
# Invokes osascript subprocess with hardcoded AppleScript commands only.

def get_system_volume():
    """Get current system output volume (0-100) and mute state."""
    try:
        vol = subprocess.check_output(
            ["osascript", "-e", "output volume of (get volume settings)"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        muted = subprocess.check_output(
            ["osascript", "-e", "output muted of (get volume settings)"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(vol), muted == "true"
    except Exception:
        return 50, False


def set_system_volume(volume):
    """Set macOS system output volume via osascript (hardcoded command)."""
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {volume}"],
        stderr=subprocess.DEVNULL,
    )


def set_system_mute(muted):
    """Set macOS system mute state via osascript (hardcoded command)."""
    val = "true" if muted else "false"
    subprocess.run(
        ["osascript", "-e", f"set volume output muted {val}"],
        stderr=subprocess.DEVNULL,
    )


# ════════════════════════════════════════════════════════════════════

class SpeechToType:
    def __init__(self, settings_path=DEFAULT_SETTINGS_PATH):
        self.model = None
        self.loaded_model_idx = None
        self._key_held = False
        self._recording = False
        self._recording_lock = threading.Lock()
        self._chunks = []
        self._stream = None

        # macOS target window state
        self._target_app = None
        self._target_app_ref = None
        self._target_window_ref = None


        # CGEvent tap for global hotkeys
        self._tap_port = None
        self._tap_source = None
        self._main_runloop = CFRunLoopGetCurrent()  # capture main thread's run loop

        # Keystroke capture during paste
        self._paste_capturing = False
        self._paste_source_state_id = 0
        self._captured_events = []

        # Menu bar status item for recording timer
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._status_item.button().setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(12, 0.0))
        self._status_item.button().setTitle_("")
        self._status_item_visible = False
        self._update_menu_bar("")  # hidden initially

        # Configurable settings
        self.input_devices = get_input_devices()
        self.device_idx = 0
        self.hotkey = "f2"
        self.mode = "push"
        self.after_action = "enter"
        self.window_target = "original"
        self.model_idx = MODELS.index("turbo")
        self.cancel_key = "f3"
        self.no_enter_key = "f4"
        self.deafen_while_recording = "off"
        self.settings_path = settings_path
        self._was_muted = False
        self._prev_volume = None
        self._skip_enter = False
        self._cancel_type_used = False
        self._record_start_time = None

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
  // Examples: "f2", "f5", "scroll_lock", "pause"
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

    # ── Audio deafen control (macOS via osascript) ──────────

    def _mute_system(self):
        try:
            if self.deafen_while_recording == "on":
                _, self._was_muted = get_system_volume()
                if not self._was_muted:
                    set_system_mute(True)
            elif self.deafen_while_recording == "half":
                self._prev_volume, _ = get_system_volume()
                set_system_volume(max(0, self._prev_volume // 2))
        except Exception as e:
            self.status = f"Deafen error: {e}"
            self._needs_redraw = True

    def _unmute_system(self):
        try:
            if self.deafen_while_recording == "on":
                if not self._was_muted:
                    set_system_mute(False)
            elif self.deafen_while_recording == "half":
                if self._prev_volume is not None:
                    set_system_volume(self._prev_volume)
                    self._prev_volume = None
        except Exception as e:
            self.status = f"Undeafen error: {e}"
            self._needs_redraw = True

    # ── macOS global hotkey via CGEvent tap ── # ~~~ SECURITY-SENSITIVE ~~~
    # Installs a global CGEvent tap that intercepts keyboard events.
    # Only hotkey/cancel key codes are inspected; all other keys pass through
    # except during the brief paste-capture window.

    def _hotkey_callback(self, proxy, event_type, event, user_info):
        """CGEvent tap callback — intercepts global keyboard events.

        Security: inspects only hotkey/cancel key codes and passes all other
        keys through unmodified.  During the ~0.5 s paste window, non-synthetic
        keystrokes are buffered (not logged) and replayed immediately after.
        """
        if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
            if self._tap_port is not None:
                CGEventTapEnable(self._tap_port, True)
            return event

        # During paste: let our synthetic events through, capture everything else
        if self._paste_capturing:
            event_source_id = CGEventGetIntegerValueField(event, 45)  # kCGEventSourceStateID
            if event_source_id == self._paste_source_state_id:
                return event
            if len(self._captured_events) < MAX_CAPTURED_EVENTS:
                self._captured_events.append(CGEventCreateCopy(event))
            return None  # suppress

        keycode = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventKeycode
        )
        key_name = KEYCODE_TO_NAME.get(keycode)
        if key_name is None:
            return event

        is_down = (event_type == kCGEventKeyDown)
        is_up = (event_type == kCGEventKeyUp)

        def matches(setting):
            return setting and key_name.lower() == setting.lower()

        # Record hotkey
        if matches(self.hotkey):
            if is_down:
                self._on_hotkey_event("down")
            elif is_up:
                self._on_hotkey_event("up")
            return None  # suppress

        # Cancel key
        if matches(self.cancel_key):
            if is_down:
                self._on_cancel_event()
            return None

        # No-enter key
        if matches(self.no_enter_key):
            if is_down:
                self._on_no_enter_event()
            return None

        return event

    def _install_hotkey_tap(self):
        """Create and install the global CGEvent tap on the main run loop.

        Security: requires Accessibility permission; taps all key-down,
        key-up, and flags-changed events system-wide.
        """
        self._uninstall_hotkey_tap()

        tap_mask = (1 << kCGEventKeyDown) | (1 << kCGEventKeyUp) | (1 << kCGEventFlagsChanged)
        self._tap_port = CGEventTapCreate(
            kCGHIDEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionDefault,
            tap_mask,
            self._hotkey_callback,
            None,
        )
        if not self._tap_port:
            self.status = "WARNING: no event tap (grant Accessibility to terminal)"
            self._needs_redraw = True
            return

        self._tap_source = CoreFoundation.CFMachPortCreateRunLoopSource(
            None, self._tap_port, 0
        )
        CoreFoundation.CFRunLoopAddSource(
            self._main_runloop, self._tap_source, CoreFoundation.kCFRunLoopDefaultMode
        )
        CGEventTapEnable(self._tap_port, True)

    def _uninstall_hotkey_tap(self):
        if self._tap_port:
            CGEventTapEnable(self._tap_port, False)
            if self._tap_source:
                CoreFoundation.CFRunLoopRemoveSource(
                    self._main_runloop,
                    self._tap_source,
                    CoreFoundation.kCFRunLoopDefaultMode,
                )
                self._tap_source = None
            self._tap_port = None

    def _pump_runloop(self, seconds=0.01):
        """Pump the main CFRunLoop so CGEvent tap callbacks fire."""
        # This must run on the main thread (where curses loop runs)
        CFRunLoopRunInMode(CoreFoundation.kCFRunLoopDefaultMode, seconds, False)

    # ── Menu bar timer ─────────────────────────────────────

    def _update_menu_bar(self, text):
        """Show or hide the menu bar status item."""
        if text:
            self._status_item.setVisible_(True)
            self._status_item.button().setTitle_(text)
            self._status_item_visible = True
        else:
            self._status_item.setVisible_(False)
            self._status_item_visible = False

    def _tick_menu_bar(self):
        """Update menu bar timer if recording, hide otherwise."""
        if self._recording and self._record_start_time is not None:
            elapsed = time.perf_counter() - self._record_start_time
            remaining = max(0, MAX_RECORDING_DURATION - elapsed)
            self._update_menu_bar(f"\U0001F534 {remaining:.0f}s")
        elif self._status_item_visible:
            self._update_menu_bar("")

    # ── Core logic ──────────────────────────────────────────

    def load_model(self):
        model_name = MODELS[self.model_idx]
        repo = MLX_MODEL_REPOS[model_name]
        self.status = f"Loading whisper '{model_name}' (MLX)..."
        self._needs_redraw = True
        t0 = time.perf_counter()

        # Warm up the model by running a tiny silent transcription.
        # mlx_whisper downloads and caches the model on first use.
        try:
            silent = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s silence
            mlx_whisper.transcribe(silent, path_or_hf_repo=repo, language="en")
        except Exception:
            pass  # model is still cached even if silence produces no output

        self.model = repo  # store repo path as the "model"
        elapsed = time.perf_counter() - t0
        self.loaded_model_idx = self.model_idx
        self.status = f"Ready on MLX (loaded in {elapsed:.1f}s)"
        self._needs_redraw = True

    def _ready_status(self):
        if self.model:
            return "Ready on MLX"
        return "Ready"

    def _audio_callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())
        if self._record_start_time and (time.perf_counter() - self._record_start_time) >= MAX_RECORDING_DURATION:
            threading.Thread(target=self._stop_recording, daemon=True).start()

    def _on_hotkey_event(self, direction):
        if self.mode == "push":
            if direction == "down":
                if self._key_held:
                    return
                self._key_held = True
                threading.Thread(target=self._start_recording, daemon=True).start()
            else:
                self._key_held = False
                threading.Thread(target=self._stop_recording, daemon=True).start()
        else:  # toggle
            if direction == "down":
                if self._key_held:
                    return
                self._key_held = True
                if self._recording:
                    threading.Thread(target=self._stop_recording, daemon=True).start()
                else:
                    threading.Thread(target=self._start_recording, daemon=True).start()
            else:
                self._key_held = False

    def _on_cancel_event(self):
        if self._recording:
            self._recording = False
            self._record_start_time = None
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._chunks = []
            if self.deafen_while_recording != "off":
                self._unmute_system()
            self._key_held = False
            self.status = "Cancelled"
            self._needs_redraw = True

    def _on_no_enter_event(self):
        if self._recording:
            self._skip_enter = True
            self._cancel_type_used = True
            self._key_held = False
            threading.Thread(target=self._stop_recording, daemon=True).start()

    def _capture_target_window(self):
        """Capture the currently focused app/window as the typing target."""
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        self._target_app = workspace.frontmostApplication()
        if self._target_app:
            pid = self._target_app.processIdentifier()
            self._target_app_ref, self._target_window_ref = get_ax_focused_window(pid)

    def _start_recording(self):
        with self._recording_lock:
            if self._recording:
                return
            self._recording = True
        self._skip_enter = False
        self._chunks = []

        self._capture_target_window()

        try:
            dev_index = self.input_devices[self.device_idx][0] if self.input_devices else None
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=dev_index,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._recording = False
            self.status = f"Recording failed: {e}"
            self._needs_redraw = True
            return
        self._record_start_time = time.perf_counter()
        if self.deafen_while_recording != "off":
            self._mute_system()
        self.status = f"Recording... 0.0s / {MAX_RECORDING_DURATION:.0f}s"
        self._needs_redraw = True

    def _stop_recording(self):
        with self._recording_lock:
            if not self._recording:
                return
            self._recording = False
        self._record_start_time = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self.deafen_while_recording != "off":
            self._unmute_system()

        if not self._chunks:
            self.status = self._ready_status()
            self._needs_redraw = True
            return

        audio = np.concatenate(self._chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < MIN_AUDIO_DURATION:
            self.status = "Too short, skipped"
            self._needs_redraw = True
            return

        self.status = f"Transcribing {duration:.1f}s of audio..."
        self._needs_redraw = True
        threading.Thread(target=self._transcribe_and_type, args=(audio,), daemon=True).start()

    def _transcribe_and_type(self, audio):
        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.model, language="en"
        )
        text = result["text"].strip()
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

        self._type_text_into_target(text)

    # ── Paste workflow ── # ~~~ SECURITY-SENSITIVE ~~~
    # Reads/writes system clipboard, posts synthetic keyboard events,
    # activates other apps' windows, and briefly captures keystrokes.

    def _resolve_paste_targets(self):
        """Determine the return-to and paste-into windows.

        Security: reads the frontmost application identity and its focused
        window via AXUIElement.

        Returns (workspace, return_to_app, return_to_app_ref,
                 return_to_window_ref, target_app, target_window_ref)
        or None if there is no valid target.
        """
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        return_to_app = workspace.frontmostApplication()
        return_to_app_ref = None
        return_to_window_ref = None
        if return_to_app:
            rt_pid = return_to_app.processIdentifier()
            return_to_app_ref, return_to_window_ref = get_ax_focused_window(rt_pid)

        if self.window_target == "active":
            target_app = return_to_app
            target_window_ref = return_to_window_ref
        else:
            target_app = self._target_app
            target_window_ref = self._target_window_ref
        if not target_app:
            return None

        return (workspace, return_to_app, return_to_app_ref,
                return_to_window_ref, target_app, target_window_ref)

    def _save_and_set_clipboard(self, text):
        """Save old clipboard contents and set clipboard to *text*.

        Security: reads and writes the system clipboard.

        Returns (pb, old_clipboard) for later restoration.
        """
        pb = AppKit.NSPasteboard.generalPasteboard()
        old_clipboard = pb.stringForType_(NSPasteboardTypeString)
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)
        return pb, old_clipboard

    def _switch_to_target_window(self, workspace, target_app, target_window_ref,
                                  return_to_window_ref):
        """Activate the target app/window, waiting for the switch to complete.

        Security: activates a cross-process window and polls until the OS
        confirms the switch.

        Returns (needs_switch, needs_window_raise).
        """
        _rl = CoreFoundation.kCFRunLoopDefaultMode
        target_pid = target_app.processIdentifier()
        front = workspace.frontmostApplication()
        same_app = front and front.processIdentifier() == target_pid
        needs_switch = not same_app
        needs_window_raise = (same_app and target_window_ref
                              and target_window_ref != return_to_window_ref)

        if needs_switch:
            target_app_ref, _ = get_ax_focused_window(target_pid)
            target_app.activateWithOptions_(0)
            if target_window_ref:
                ax_raise_window(target_window_ref)
                if target_app_ref:
                    ax_set_focused_window(target_app_ref, target_window_ref)
            for _ in range(WINDOW_SWITCH_MAX_POLLS):
                CFRunLoopRunInMode(_rl, WINDOW_SWITCH_POLL_INTERVAL, False)
                front = workspace.frontmostApplication()
                if front and front.processIdentifier() == target_pid:
                    break
        elif needs_window_raise:
            target_app_ref, _ = get_ax_focused_window(target_pid)
            ax_raise_window(target_window_ref)
            if target_app_ref:
                ax_set_focused_window(target_app_ref, target_window_ref)
        CFRunLoopRunInMode(_rl, POST_SWITCH_SETTLE, False)

        return needs_switch, needs_window_raise

    def _paste_and_enter(self, src):
        """Post synthetic Cmd+V and (optionally) Enter key events.

        Security: posts synthetic keyboard events into the active application
        via CGEventPost.  Key codes 0x09 (V) and 0x24 (Return) are macOS
        API constants.
        """
        # Paste (Cmd+V)
        down = CGEventCreateKeyboardEvent(src, 0x09, True)
        CGEventSetFlags(down, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, down)
        self._pump_runloop_for(PASTE_KEY_DOWN_DELAY)
        up = CGEventCreateKeyboardEvent(src, 0x09, False)
        CGEventSetFlags(up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, up)

        # Optionally press Enter
        if self.after_action == "enter" and not self._skip_enter:
            self._pump_runloop_for(PASTE_PRE_ENTER_DELAY)
            down = CGEventCreateKeyboardEvent(src, 0x24, True)
            CGEventPost(kCGHIDEventTap, down)
            self._pump_runloop_for(ENTER_KEY_DOWN_DELAY)
            up = CGEventCreateKeyboardEvent(src, 0x24, False)
            CGEventPost(kCGHIDEventTap, up)

        self._pump_runloop_for(PASTE_SETTLE_DELAY)

    def _switch_back_and_replay(self, workspace, return_to_app, return_to_app_ref,
                                 return_to_window_ref, needs_switch, needs_window_raise):
        """Return to the original window and replay buffered keystrokes.

        Security: activates the original window and posts all captured
        keystroke events back into the system event stream.
        """
        if (needs_switch or needs_window_raise) and return_to_app:
            if needs_switch:
                return_to_app.activateWithOptions_(0)
            if return_to_window_ref:
                ax_raise_window(return_to_window_ref)
                if return_to_app_ref:
                    ax_set_focused_window(return_to_app_ref, return_to_window_ref)
            self._pump_runloop_for(SWITCH_BACK_DELAY)

        self._paste_capturing = False
        self._pump_runloop_for(PRE_REPLAY_DELAY)
        for evt in self._captured_events:
            CGEventPost(kCGHIDEventTap, evt)
            time.sleep(KEYSTROKE_REPLAY_INTERVAL)
        self._captured_events.clear()

    def _restore_clipboard(self, pb, old_clipboard):
        """Restore the original clipboard contents (best-effort).

        Security: writes the system clipboard.
        """
        if old_clipboard is not None:
            self._pump_runloop_for(CLIPBOARD_RESTORE_DELAY)
            pb.clearContents()
            pb.setString_forType_(old_clipboard, NSPasteboardTypeString)

    def _pump_runloop_for(self, seconds):
        """Pump CFRunLoop for *seconds*, processing event tap callbacks.

        Security: processes pending CGEvent tap callbacks during the wait.
        Uses short iterations so synthetic events are delivered promptly.
        """
        _rl = CoreFoundation.kCFRunLoopDefaultMode
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            CFRunLoopRunInMode(_rl, PUMP_INTERVAL, False)
            remaining = deadline - time.monotonic()
            if remaining > PUMP_SLEEP:
                time.sleep(PUMP_SLEEP)

    def _type_text_into_target(self, text):
        """Orchestrate the full paste workflow.

        Security: touches clipboard, posts synthetic key events, activates
        cross-process windows, and temporarily captures all keystrokes.
        Wrapped in try/finally to guarantee cleanup on any failure.
        """
        targets = self._resolve_paste_targets()
        if targets is None:
            return
        (workspace, return_to_app, return_to_app_ref,
         return_to_window_ref, target_app, target_window_ref) = targets

        pb, old_clipboard = self._save_and_set_clipboard(text)

        # Create a private event source and start keystroke capture
        src = CGEventSourceCreate(kCGEventSourceStatePrivate)
        self._paste_source_state_id = CGEventSourceGetSourceStateID(src)
        self._captured_events.clear()
        self._paste_capturing = True

        try:
            needs_switch, needs_window_raise = self._switch_to_target_window(
                workspace, target_app, target_window_ref, return_to_window_ref)

            self._paste_and_enter(src)

            self._switch_back_and_replay(
                workspace, return_to_app, return_to_app_ref,
                return_to_window_ref, needs_switch, needs_window_raise)
        finally:
            # Guarantee cleanup even if an exception occurs mid-workflow
            self._paste_capturing = False
            self._captured_events.clear()
            try:
                self._restore_clipboard(pb, old_clipboard)
            except Exception:
                pass  # best-effort clipboard restore

    # ── Descriptions ────────────────────────────────────────

    MODEL_INFO = {
        "tiny":   "whisper-tiny (MLX) — 39M params, ~1GB RAM. Fastest, least accurate. Good for quick tests.",
        "base":   "whisper-base (MLX) — 74M params, ~1GB RAM. Fast with decent accuracy for clear speech.",
        "small":  "whisper-small (MLX) — 244M params, ~2GB RAM. Good balance of speed and accuracy.",
        "medium": "whisper-medium (MLX) — 769M params, ~5GB RAM. High accuracy, slower transcription.",
        "large":  "whisper-large-v3 (MLX) — 1550M params, ~10GB RAM. Best accuracy, requires significant resources.",
        "turbo":  "whisper-turbo (MLX) — Fast distilled large model. Recommended — near-large accuracy at much higher speed.",
    }

    def _get_description(self, selected):
        if selected == 0:
            return f"vibe-code-mic v{VERSION} (macOS). Local speech recognition powered by Whisper via MLX (Apple Silicon GPU). All processing happens on your machine — no data is sent to the cloud."
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
            return "A single tap during recording stops it and types the text without pressing Enter. Useful when you want to edit the text before sending."
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
                    while len(cur) > avail:
                        result.append(cur[:avail])
                        cur = cur[avail:]
                if cur or not paragraph:
                    result.append(cur)
            return result if result else ['']

        def draw_wrapped(y, x, text, attr=0):
            avail = w - x - 1
            lines = wrap_text(text, avail)
            for i, line in enumerate(lines):
                if (y + i) < h:
                    safe_addstr(y + i, x, line, attr)
            return max(len(lines), 1)

        safe_addstr(0, 0, "=== vibe-code-mic (macOS) ===", curses.A_BOLD)

        # About
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

        # Description for selected option
        desc = self._get_description(selected)
        desc_lines = draw_wrapped(13, 2, desc, curses.color_pair(4))

        # Bottom-pinned: help at very bottom, status above
        help_y = h - 1
        safe_addstr(help_y, 0, "  Up/Down: navigate | Left/Right: change | Ctrl+C: quit")

        # Status with color
        display_status = self.status
        if self._recording and self._record_start_time is not None:
            elapsed = time.perf_counter() - self._record_start_time
            display_status = f"Recording... {elapsed:.1f}s / {MAX_RECORDING_DURATION:.0f}s"

        if "Recording" in display_status:
            status_color = curses.color_pair(2) | curses.A_BOLD
        elif "Transcribing" in display_status:
            status_color = curses.color_pair(3) | curses.A_BOLD
        elif "Quitting" in display_status:
            status_color = curses.color_pair(3) | curses.A_BOLD
        elif "Ready" in display_status:
            status_color = curses.color_pair(1)
        else:
            status_color = curses.color_pair(4)
        model_label = MODELS[self.loaded_model_idx] if self.loaded_model_idx is not None else "none"
        status_text = f"{display_status}  [model: {model_label}]"
        status_avail = w - 10 - 1
        status_lines = wrap_text(status_text, status_avail)
        status_line_count = max(len(status_lines), 1)
        status_y = help_y - 1 - status_line_count
        safe_addstr(status_y, 0, "  Status: ")
        for i, sline in enumerate(status_lines):
            if (status_y + i) < h:
                safe_addstr(status_y + i, 10, sline, status_color)

        # Last transcription
        if self.last_text:
            last_start_y = 13 + desc_lines + 1
            last_end_y = status_y - 1
            avail_lines = last_end_y - last_start_y
            avail_width = w - 12
            if avail_lines > 0 and avail_width > 0:
                safe_addstr(last_start_y, 0, "  Last:   ", curses.color_pair(4))
                text_with_time = f'"{self.last_text}" ({self.last_time:.1f}s)'
                wrapped = wrap_text(text_with_time, avail_width)
                for line, chunk in enumerate(wrapped[:avail_lines]):
                    safe_addstr(last_start_y + line, 10, chunk, curses.A_BOLD)

        stdscr.refresh()

    def _rebind_key_via_tap(self, attr="hotkey"):
        """Wait for a single keypress via CGEvent tap and rebind the specified key."""
        captured = {"key": None, "done": False}

        def rebind_callback(proxy, event_type, event, user_info):
            if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
                return event
            if event_type == kCGEventKeyDown and not captured["done"]:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                captured["key"] = keycode
                captured["done"] = True
            return None  # suppress

        tap_mask = (1 << kCGEventKeyDown) | (1 << kCGEventKeyUp)
        rebind_tap = CGEventTapCreate(
            kCGHIDEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionDefault,
            tap_mask,
            rebind_callback,
            None,
        )
        if not rebind_tap:
            self.status = "Couldn't create tap for rebinding"
            self._needs_redraw = True
            self._install_hotkey_tap()
            return

        rebind_source = CoreFoundation.CFMachPortCreateRunLoopSource(
            None, rebind_tap, 0
        )
        CoreFoundation.CFRunLoopAddSource(
            self._main_runloop, rebind_source, CoreFoundation.kCFRunLoopDefaultMode
        )
        CGEventTapEnable(rebind_tap, True)

        # Wait until the main thread's pump fires the callback (timeout 10s)
        for _ in range(REBIND_TIMEOUT_POLLS):
            time.sleep(REBIND_POLL_INTERVAL)
            if captured["done"]:
                break

        CGEventTapEnable(rebind_tap, False)
        CoreFoundation.CFRunLoopRemoveSource(
            self._main_runloop, rebind_source, CoreFoundation.kCFRunLoopDefaultMode
        )

        if captured["done"] and captured["key"] is not None:
            keycode = captured["key"]
            if keycode == 0x35:  # Escape
                setattr(self, attr, None)
                self.status = "Key disabled"
            else:
                key_name = KEYCODE_TO_NAME.get(keycode, f"key_{keycode}")
                setattr(self, attr, key_name)
                self.status = f"Key set to {key_name.upper()}"
        else:
            self.status = "Rebind timed out"

        self._save_settings()
        self._needs_redraw = True
        self._install_hotkey_tap()

    def run_curses(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(CURSES_INPUT_TIMEOUT_MS)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # ready
        curses.init_pair(2, curses.COLOR_RED, -1)      # recording
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # transcribing
        curses.init_pair(4, curses.COLOR_CYAN, -1)     # info

        NUM_SETTINGS = 10
        selected = 0
        rebinding = False

        # Load model and install hotkey tap in background
        threading.Thread(target=self._load_and_hook, daemon=True).start()

        while self._running:
            # Pump the CFRunLoop so CGEvent tap callbacks fire
            self._pump_runloop(MAIN_LOOP_PUMP_DURATION)

            self._tick_menu_bar()
            self.draw_ui(stdscr, selected, rebinding)
            self._needs_redraw = False

            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                self.status = "Quitting..."
                self.draw_ui(stdscr, selected, rebinding)
                stdscr.refresh()
                time.sleep(QUIT_DELAY)
                break
            except Exception:
                key = -1

            if key == -1:
                continue

            if key == 3:  # Ctrl+C
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
            elif key == 10 or key == curses.KEY_ENTER:
                if selected == 2:
                    self._uninstall_hotkey_tap()
                    self._save_settings()
                    threading.Thread(target=self._load_and_hook, daemon=True).start()
                elif selected == 3:
                    rebinding = "hotkey"
                    self._uninstall_hotkey_tap()
                    self.status = "Press the new record key..."
                    self._needs_redraw = True
                    threading.Thread(target=lambda: self._rebind_key_via_tap("hotkey"), daemon=True).start()
                elif selected == 4:
                    rebinding = "cancel"
                    self._uninstall_hotkey_tap()
                    self.status = "Press the new cancel key..."
                    self._needs_redraw = True
                    threading.Thread(target=lambda: self._rebind_key_via_tap("cancel_key"), daemon=True).start()
                elif selected == 5:
                    rebinding = "no_enter"
                    self._uninstall_hotkey_tap()
                    self.status = "Press the new cancel+type key..."
                    self._needs_redraw = True
                    threading.Thread(target=lambda: self._rebind_key_via_tap("no_enter_key"), daemon=True).start()

    def _load_and_hook(self):
        self.load_model()
        self._install_hotkey_tap()

    def quit_app(self):
        self.status = "Quitting..."
        self._needs_redraw = True
        self._save_settings()
        self._running = False
        self._uninstall_hotkey_tap()
        NSStatusBar.systemStatusBar().removeStatusItem_(self._status_item)

    def run(self):
        curses.wrapper(self.run_curses)
        print("Quitting...")
        self.quit_app()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="vibe-code-mic",
        description="Local speech-to-text that types into your active window (macOS). Uses OpenAI Whisper.",
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
        help="hotkey for recording (e.g. f2, f5) (overrides settings file)",
    )
    parser.add_argument(
        "--mode",
        choices=["push", "toggle"],
        help="recording mode: push or toggle (overrides settings file)",
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
