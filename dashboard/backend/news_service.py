import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

_headlines: list[dict] | None = None


def _load() -> list[dict]:
    global _headlines
    if _headlines is None:
        path = MARTA_ROOT / "data" / "news_cache.json"
        if path.exists():
            with open(path) as f:
                _headlines = json.load(f)["headlines"]
        else:
            _headlines = []
    return _headlines


def _parse_ts(ts_str: str) -> datetime:
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_news(asset: str, timestamp: str, window_hours: int = 720) -> list[dict]:
    headlines = _load()
    ts = _parse_ts(timestamp)
    window = timedelta(hours=window_hours)

    results = []
    seen: set[str] = set()
    for h in headlines:
        if h.get("asset") not in (asset, "macro"):
            continue
        hts = _parse_ts(h["timestamp"])
        diff = ts - hts
        # Accept headlines up to window_hours before OR 24h after (for near-future context)
        if -timedelta(hours=24) <= diff <= window:
            key = h["headline"]
            if key not in seen:
                seen.add(key)
                results.append({
                    **h,
                    "hours_before": round(diff.total_seconds() / 3600, 1),
                })

    return sorted(results, key=lambda x: abs(x["hours_before"]))
