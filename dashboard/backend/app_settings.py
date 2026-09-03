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

_DEFAULTS = {
    "approval_mode": False,
    # RL execution toggle (2026-08-25) — default False per
    # docs/RESULTS_AND_METHODOLOGY.md Section 9: a complete, controlled ablation across
    # regime conditioning, reward formulation, BC warm-start, and action-space
    # parameterization found no crypto5 RL configuration with a statistically
    # defensible edge over buy-and-hold. LLM consensus + static (rule-based) execution
    # is the default; RL stays available as an explicit, opt-in experimental toggle.
    "rl_execution_enabled": False,
    # Weekly-cadence PPO toggle (2026-09-03) — a DIFFERENT checkpoint from the one
    # rl_execution_enabled gates: native decision_interval=168 training
    # (rl/train_weekly_smoketest.py), validated with a positive Sharpe on BTC_USDT
    # (unlike the hourly checkpoint above). Default False — opt-in, and mutually
    # exclusive in effect with rl_execution_enabled for BTC_USDT specifically (see
    # live_signal.py, which checks this first and takes over entirely for that asset
    # when on). "raw_regime" is the default mode since it's the one this project's
    # regime-conditioning narrative is actually about; "raw" (no regime block) is the
    # other validated-positive option, selectable via weekly_ppo_obs_mode.
    "weekly_ppo_enabled": False,
    "weekly_ppo_obs_mode": "raw_regime",
}
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


def get_rl_execution_enabled() -> bool:
    with _lock:
        return bool(_read().get("rl_execution_enabled", False))


def set_rl_execution_enabled(enabled: bool) -> bool:
    with _lock:
        settings = _read()
        settings["rl_execution_enabled"] = bool(enabled)
        _write(settings)
        return settings["rl_execution_enabled"]


def get_weekly_ppo_enabled() -> bool:
    with _lock:
        return bool(_read().get("weekly_ppo_enabled", False))


def set_weekly_ppo_enabled(enabled: bool) -> bool:
    with _lock:
        settings = _read()
        settings["weekly_ppo_enabled"] = bool(enabled)
        _write(settings)
        return settings["weekly_ppo_enabled"]


def get_weekly_ppo_obs_mode() -> str:
    with _lock:
        return str(_read().get("weekly_ppo_obs_mode", "raw_regime"))


def set_weekly_ppo_obs_mode(obs_mode: str) -> str:
    if obs_mode not in ("raw", "raw_regime"):
        raise ValueError(f"weekly_ppo_obs_mode must be 'raw' or 'raw_regime', got {obs_mode!r}")
    with _lock:
        settings = _read()
        settings["weekly_ppo_obs_mode"] = obs_mode
        _write(settings)
        return settings["weekly_ppo_obs_mode"]
