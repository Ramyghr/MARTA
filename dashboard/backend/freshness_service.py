"""Data freshness service — reads freshness.json and per-asset raw parquets.

The nightly cron job (scripts/nightly_refresh.py) writes freshness.json after
each refresh. This service exposes it to the dashboard so staleness alerts can
surface in the UI without requiring the dashboard to re-read every parquet.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MARTA_ROOT  = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))
_FRESHNESS  = MARTA_ROOT / "data" / "freshness.json"
_RAW_DIR    = MARTA_ROOT / "data" / "raw"
_STALE_HOURS = 26

if str(MARTA_ROOT) not in sys.path:
    sys.path.insert(0, str(MARTA_ROOT))
from utils.assets import ASSET_FILE_STEM  # noqa: E402

_ASSETS = [
    {"id": "BTC_USDT",  "label": "BTC/USDT",  "type": "crypto"},
    {"id": "ETH_USDT",  "label": "ETH/USDT",  "type": "crypto"},
    {"id": "BNB_USDT",  "label": "BNB/USDT",  "type": "crypto"},
    {"id": "LTC_USDT",  "label": "LTC/USDT",  "type": "crypto"},
    {"id": "XRP_USDT",  "label": "XRP/USDT",  "type": "crypto"},
]


def _last_raw_timestamp(asset_id: str) -> str | None:
    try:
        import pandas as pd
        file_stem = ASSET_FILE_STEM.get(asset_id, asset_id)
        path = _RAW_DIR / f"{file_stem}_1h_raw.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            return None
        return df.index.max().isoformat()
    except Exception:
        return None


def _age_hours(ts_iso: str | None) -> float:
    if not ts_iso:
        return float("inf")
    try:
        import pandas as pd
        ts = pd.Timestamp(ts_iso, tz="UTC")
        now = pd.Timestamp("now", tz="UTC")
        return (now - ts).total_seconds() / 3600
    except Exception:
        return float("inf")


def get_freshness() -> dict:
    """Return freshness status for all assets.

    Reads freshness.json (written by nightly_refresh.py) if available, then
    augments with direct parquet checks for assets not in the JSON.
    """
    # Try freshness.json first (written by cron)
    cron_report: dict = {}
    if _FRESHNESS.exists():
        try:
            cron_report = json.loads(_FRESHNESS.read_text())
        except Exception:
            pass

    cron_by_id = {r["symbol"].replace("/", "_"): r for r in cron_report.get("assets", [])}

    asset_statuses = []
    for a in _ASSETS:
        aid = a["id"]
        if aid in cron_by_id:
            r = cron_by_id[aid]
            age = float(r.get("age_hours", 0))
            last_ts = r.get("last_timestamp")
        else:
            # Fall back to direct parquet check
            last_ts = _last_raw_timestamp(aid)
            age = _age_hours(last_ts)

        stale = age > _STALE_HOURS
        if not math.isfinite(age):
            age = 999_999.0  # sentinel for "no timestamp available" — JSON has no Infinity

        asset_statuses.append({
            "id":             aid,
            "label":          a["label"],
            "type":           a["type"],
            "last_timestamp": last_ts,
            "age_hours":      round(age, 1),
            "stale":          stale,
        })

    any_stale = any(s["stale"] for s in asset_statuses)
    cron_ran_at = cron_report.get("generated_at")

    return {
        "assets":       asset_statuses,
        "any_stale":    any_stale,
        "cron_last_ran": cron_ran_at,
        "checked_at":   datetime.now(timezone.utc).isoformat(),
        "stale_threshold_hours": _STALE_HOURS,
    }
