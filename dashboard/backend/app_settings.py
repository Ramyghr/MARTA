"""Small persisted app-wide settings — currently just approval_mode.

Must be readable by signal_scheduler.py/live_signal.py independent of any browser
session (the scheduler ticks with no request in flight), so this is a JSON file
rather than frontend-only state. Single small file, one process — a lock is enough,
no need for SQLite here.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))
_SETTINGS_PATH = Path(
    os.environ.get("APP_SETTINGS_PATH", str(MARTA_ROOT / "data" / "app_settings.json"))
)

_DEFAULTS = {"approval_mode": False}
_lock = threading.Lock()


def _read() -> dict:
    if not _SETTINGS_PATH.exists():
        return dict(_DEFAULTS)
    try:
        return {**_DEFAULTS, **json.loads(_SETTINGS_PATH.read_text())}
    except Exception:
        return dict(_DEFAULTS)


def _write(settings: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def get_approval_mode() -> bool:
    with _lock:
        return bool(_read().get("approval_mode", False))


def set_approval_mode(enabled: bool) -> bool:
    with _lock:
        settings = _read()
        settings["approval_mode"] = bool(enabled)
        _write(settings)
        return settings["approval_mode"]
