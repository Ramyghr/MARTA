"""pipeline/live/economic_calendar.py — upcoming economic events for live debates.

investing.com's own economic-calendar page is Cloudflare-protected — scraping it
directly is fragile and against its ToS, and would be the single flakiest dependency
in a system that queries it every debate. Uses ForexFactory's public JSON feed instead
(nfs.faireconomy.media — no API key, no auth, carries the same event set: FOMC, NFP,
CPI, GDP, PMI, central bank rate decisions, with impact/forecast/previous). Verified
live and schema-checked 2026-08-11.

Self-contained on purpose, matching pipeline/live/news.py's style (own in-memory TTL
cache, no shared module) rather than introducing new cross-file coupling in this
package.

Usage::

    from pipeline.live.economic_calendar import get_upcoming_events, get_high_impact_soon

    events = get_upcoming_events("BTC_USDT", hours_ahead=48)
    # [{"title": "Fed Interest Rate Decision", "currency": "USD", "impact": "high",
    #   "time": <aware datetime>, "hours_until": 6.2, "forecast": "4.25%", "previous": "4.50%"}, ...]
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

_FF_BASE = "https://nfs.faireconomy.media/ff_calendar_{week}.json"

# Which currencies actually move each tracked asset. Crypto has no "currency" of its
# own in the calendar — it trades off USD macro (rate decisions, CPI, NFP) — so it maps
# to USD only, not to every currency in the feed. Mirrors the same
# "only apply signals actually about this asset" principle SentimentAgent's
# _FEAR_GREED_ASSETS / _MACRO_ASSETS already use.
CURRENCY_RELEVANCE: dict[str, set[str]] = {
    "BTC_USDT": {"USD"},
    "ETH_USDT": {"USD"},
    "EURUSD": {"USD", "EUR"},
    "GOLD": {"USD"},
}

_IMPACT_ORDER = {"low": 0, "medium": 1, "high": 2, "holiday": -1}

_TTL_SECONDS = 30 * 60  # 30 min — feed updates through the day as `actual` values land


class _MemCache:
    """Same tiny TTL cache news.py uses — duplicated rather than imported to keep
    this module self-contained (see module docstring)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        val, exp = entry
        if time.monotonic() > exp:
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: Any, ex: int = _TTL_SECONDS) -> None:
        self._store[key] = (value, time.monotonic() + ex)


_cache = _MemCache()


def _parse_ff_date(date_str: str) -> datetime:
    """ForexFactory dates look like '2026-08-12T12:30:00-04:00' — Python 3.11's
    fromisoformat handles this natively (this repo targets 3.11, per CLAUDE.md)."""
    dt = datetime.fromisoformat(date_str)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _normalize_impact(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r in ("low", "medium", "high", "holiday"):
        return r
    return "low"


def _fetch_ff_week_sync(week: Literal["thisweek", "nextweek"]) -> list[dict[str, Any]]:
    try:
        import requests
    except ImportError:
        logger.warning("[econ_calendar] requests not installed — economic calendar disabled")
        return []

    url = _FF_BASE.format(week=week)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning(f"[econ_calendar] fetch failed ({week}): {exc}")
        return []

    events: list[dict[str, Any]] = []
    for e in raw:
        try:
            events.append({
                "title": e.get("title", ""),
                "currency": e.get("country", ""),
                "time": _parse_ff_date(e["date"]),
                "impact": _normalize_impact(e.get("impact", "")),
                "forecast": e.get("forecast", ""),
                "previous": e.get("previous", ""),
            })
        except Exception:
            continue  # malformed row — skip rather than fail the whole fetch
    return events


def _get_calendar_window() -> list[dict[str, Any]]:
    """This-week's events, cached ~30 min.

    ForexFactory only publishes `ff_calendar_thisweek.json` at this host —
    `..._nextweek.json` (which would extend the lookahead near a week boundary)
    404s consistently (verified 2026-08-11, several URL variants tried); it's simply
    not served. Falling back gracefully rather than retrying every cache refresh:
    get_upcoming_events()'s hours_ahead window can silently come up short only in the
    last day or two of a Sun-Sat week — acceptable for a free, no-key feed.
    """
    cached = _cache.get("calendar")
    if cached is not None:
        return cached
    events = _fetch_ff_week_sync("thisweek")
    _cache.set("calendar", events)
    return events


def get_upcoming_events(
    asset: str, hours_ahead: int = 48, min_impact: str = "medium"
) -> list[dict[str, Any]]:
    """Events relevant to *asset* in [now, now+hours_ahead], impact >= min_impact,
    sorted soonest-first. Each event carries `hours_until` (float)."""
    currencies = CURRENCY_RELEVANCE.get(asset, set())
    if not currencies:
        return []

    min_rank = _IMPACT_ORDER.get(min_impact, 1)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    out: list[dict[str, Any]] = []
    for e in _get_calendar_window():
        if e["currency"] not in currencies:
            continue
        if _IMPACT_ORDER.get(e["impact"], -1) < min_rank:
            continue
        if not (now <= e["time"] <= cutoff):
            continue
        item = dict(e)
        item["hours_until"] = round((e["time"] - now).total_seconds() / 3600.0, 1)
        out.append(item)

    out.sort(key=lambda x: x["hours_until"])
    return out


def get_high_impact_soon(asset: str, hours_ahead: int = 6) -> list[dict[str, Any]]:
    """Tight-window, high-impact-only view — used by RiskAgent's quantitative throttle
    (deliberately separate from the broader 48h narrative list in get_upcoming_events,
    so the hard position-size cap only reacts to genuinely imminent risk)."""
    return [
        e for e in get_upcoming_events(asset, hours_ahead=hours_ahead, min_impact="high")
    ]
