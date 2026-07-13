"""Live news feed — pulls from free RSS feeds, no API key required.

Sources:
  CoinDesk  https://www.coindesk.com/arc/outboundfeeds/rss/
  CoinTelegraph  https://cointelegraph.com/rss
  Decrypt  https://decrypt.co/feed
  Reuters crypto  https://feeds.reuters.com/reuters/technologyNews
  StockTwits  https://api.stocktwits.com/api/2/streams/symbol/ — financial social
              sentiment, no auth (this replaced the former Reddit integration)

Plus a market-wide, non-article signal:
  Fear & Greed Index  https://api.alternative.me/fng/  (alternative.me, no auth)

Results are cached for 10 minutes so repeated WS connections don't hammer the feeds.
"""

import asyncio
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

# StockTwits' Cloudflare edge 403s the default httpx UA even though the endpoint
# itself needs no auth — a normal browser UA clears it.
_STOCKTWITS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

STOCKTWITS_SYMBOLS: dict[str, str] = {
    "BTC_USDT": "BTC.X",
    "ETH_USDT": "ETH.X",
    "GOLD":     "GC_F",
    "EURUSD":   "EURUSD",
}

# ── Source config ─────────────────────────────────────────────────────────────

_FEEDS: list[dict] = [
    {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "source": "CoinDesk",
        "assets": ["BTC_USDT", "ETH_USDT", "GOLD", "macro"],
    },
    {
        "url": "https://cointelegraph.com/rss",
        "source": "CoinTelegraph",
        "assets": ["BTC_USDT", "ETH_USDT", "macro"],
    },
    {
        "url": "https://decrypt.co/feed",
        "source": "Decrypt",
        "assets": ["BTC_USDT", "ETH_USDT", "macro"],
    },
    {
        "url": "https://feeds.reuters.com/reuters/technologyNews",
        "source": "Reuters",
        "assets": ["macro", "GOLD"],
    },
]

# Asset-level keyword filters to assign an RSS item to a specific asset
_ASSET_KEYWORDS: dict[str, list[str]] = {
    "BTC_USDT": ["bitcoin", "btc", "satoshi", "lightning network", "spot etf"],
    "ETH_USDT": ["ethereum", "eth", "ether", "defi", "erc-20", "polygon", "layer 2"],
    "GOLD": ["gold", "xau", "precious metal", "bullion"],
    "EURUSD": ["euro", "eur/usd", "ecb", "eurozone"],
}

_CACHE: list[dict] = []
_CACHE_EXPIRY: float = 0.0
_CACHE_TTL = 600  # 10 minutes
_FETCH_LOCK = asyncio.Lock()

_STOCKTWITS_CACHE: dict[str, list[dict]] = {}
_STOCKTWITS_EXPIRY: dict[str, float] = {}
_STOCKTWITS_TTL = 300  # 5 minutes — StockTwits streams move fast

_FEAR_GREED_CACHE: dict = {}
_FEAR_GREED_TTL = 3600  # 1 hour cache — the index itself updates roughly daily


# ── Public API ────────────────────────────────────────────────────────────────

async def get_live_news(asset: str, max_items: int = 30) -> list[dict]:
    """Return up to `max_items` recent news items relevant to `asset` — RSS headlines
    merged with StockTwits messages for the same asset, newest first."""
    await _maybe_refresh()
    filtered = [h for h in _CACHE if _matches_asset(h, asset)]

    stocktwits = await fetch_stocktwits(asset)
    filtered.extend(_stocktwits_to_news_items(stocktwits))

    filtered.sort(key=lambda h: h["timestamp"], reverse=True)
    return filtered[:max_items]


async def fetch_stocktwits(asset: str) -> list[dict]:
    """Latest StockTwits messages for `asset`. No auth required. Cached 5 minutes."""
    symbol = STOCKTWITS_SYMBOLS.get(asset)
    if not symbol:
        return []

    now = time.time()
    if _STOCKTWITS_EXPIRY.get(asset, 0) > now:
        return _STOCKTWITS_CACHE.get(asset, [])

    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        async with httpx.AsyncClient(timeout=10, headers={"User-Agent": _STOCKTWITS_USER_AGENT}) as client:
            resp = await client.get(url, params={"limit": 30})
            if resp.status_code != 200:
                print(f"[news_live] StockTwits {asset} ({symbol}): HTTP {resp.status_code}")
                return _STOCKTWITS_CACHE.get(asset, [])
            data = resp.json()

        messages = data.get("messages", [])
        result = [
            {
                "source": "stocktwits",
                "headline": (msg.get("body") or "")[:200],
                "url": f"https://stocktwits.com/message/{msg.get('id')}",
                "published_at": msg.get("created_at", ""),
                "sentiment": ((msg.get("entities") or {}).get("sentiment") or {}).get("basic", "Neutral"),
                "asset": asset,
            }
            for msg in messages
            if msg.get("body")
        ]
        _STOCKTWITS_CACHE[asset] = result
        _STOCKTWITS_EXPIRY[asset] = now + _STOCKTWITS_TTL
        return result
    except Exception as e:
        print(f"[news_live] StockTwits {asset} error: {e}")
        return _STOCKTWITS_CACHE.get(asset, [])


def _stocktwits_to_news_items(messages: list[dict]) -> list[dict]:
    """Map raw StockTwits messages (fetch_stocktwits' shape) onto the same NewsItem
    shape RSS headlines use, so the two sources merge cleanly for the News Panel."""
    items = []
    for m in messages:
        try:
            dt = datetime.strptime(m["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
        hours_before = round((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 1)
        sentiment = (m.get("sentiment") or "neutral").lower()
        if sentiment not in ("bullish", "bearish"):
            sentiment = "neutral"
        items.append({
            "timestamp": dt.isoformat(),
            "asset": m["asset"],
            "headline": m["headline"],
            "summary": "",
            "source": "StockTwits",
            "url": m["url"],
            "sentiment": sentiment,
            "hours_before": hours_before,
            "live": True,
        })
    return items


async def fetch_fear_greed() -> dict:
    """Fetch crypto Fear & Greed Index. No auth required. Updates daily."""
    now = time.time()
    if _FEAR_GREED_CACHE.get("timestamp", 0) + _FEAR_GREED_TTL > now:
        return _FEAR_GREED_CACHE.get("data", {})

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            data = resp.json()

        fng = data.get("data", [{}])[0]
        value = int(fng.get("value", 50))
        result = {
            "value": value,
            "classification": fng.get("value_classification", "Neutral"),
            "timestamp": fng.get("timestamp", ""),
            "label": _fng_label(value),
        }
        _FEAR_GREED_CACHE["data"] = result
        _FEAR_GREED_CACHE["timestamp"] = now
        return result
    except Exception as e:
        print(f"[news_live] Fear & Greed fetch error: {e}")
        return {"value": 50, "classification": "Neutral", "label": "neutral"}


def _fng_label(value: int) -> str:
    if value >= 75: return "extreme_greed"
    if value >= 55: return "greed"
    if value >= 45: return "neutral"
    if value >= 25: return "fear"
    return "extreme_fear"


# ── Internal ──────────────────────────────────────────────────────────────────

def _matches_asset(item: dict, asset: str) -> bool:
    text = (item.get("headline", "") + " " + item.get("summary", "")).lower()
    # Always include macro items
    if item.get("asset") == "macro":
        return True
    if asset == item.get("asset"):
        return True
    # Keyword check
    for kw in _ASSET_KEYWORDS.get(asset, []):
        if kw in text:
            return True
    return False


def _classify_asset(title: str, summary: str) -> str:
    text = (title + " " + summary).lower()
    for asset, kws in _ASSET_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                return asset
    return "macro"


def _parse_rss_date(date_str: Optional[str]) -> str:
    if not date_str:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _parse_feed(xml_text: str, source: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
        # Handle both RSS and Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        channel = root.find("channel") or root

        for item in channel.findall("item")[:20]:
            title_el = item.find("title")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")
            link_el = item.find("link")

            title = (title_el.text or "").strip() if title_el is not None else ""
            desc = (desc_el.text or "").strip()[:200] if desc_el is not None else ""
            pub = _parse_rss_date(pub_el.text if pub_el is not None else None)
            link = (link_el.text or "").strip() if link_el is not None else ""

            if not title:
                continue

            asset = _classify_asset(title, desc)
            sentiment = _quick_sentiment(title)
            items.append({
                "timestamp": pub,
                "asset": asset,
                "headline": title,
                "summary": desc[:150],
                "source": source,
                "url": link,
                "sentiment": sentiment,
                "hours_before": 0,  # will be filled by caller
                "live": True,
            })
    except Exception as e:
        print(f"[news_live] parse error ({source}): {e}")
    return items


def _quick_sentiment(text: str) -> str:
    t = text.lower()
    bullish = ["surge", "rally", "bull", "record high", "gain", "rises", "soars",
               "institutional", "adoption", "approve", "etf", "all-time high", "breakout",
               "recovery", "rebound", "climbs", "jumps", "wins"]
    bearish = ["crash", "bear", "drop", "fell", "crisis", "ban", "hack", "fraud",
               "collapse", "plunge", "warn", "probe", "bankrupt", "liquidation",
               "selloff", "contagion", "fine", "crackdown", "scam", "exploit"]
    b_count = sum(1 for w in bullish if w in t)
    s_count = sum(1 for w in bearish if w in t)
    if b_count > s_count:
        return "bullish"
    if s_count > b_count:
        return "bearish"
    return "neutral"


async def _fetch_feed(client: httpx.AsyncClient, feed: dict) -> list[dict]:
    try:
        r = await client.get(feed["url"], timeout=8.0, follow_redirects=True)
        if r.status_code == 200:
            return _parse_feed(r.text, feed["source"])
    except Exception as e:
        print(f"[news_live] fetch error ({feed['source']}): {e}")
    return []


async def _maybe_refresh() -> None:
    global _CACHE, _CACHE_EXPIRY
    if time.time() < _CACHE_EXPIRY and _CACHE:
        return
    async with _FETCH_LOCK:
        if time.time() < _CACHE_EXPIRY and _CACHE:
            return
        print("[news_live] refreshing RSS feeds…")
        items: list[dict] = []
        async with httpx.AsyncClient(headers={"User-Agent": "MARTA-Dashboard/1.0"}) as client:
            tasks = [_fetch_feed(client, feed) for feed in _FEEDS]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, list):
                    items.extend(r)

        # Deduplicate by headline, sort newest-first
        seen: set[str] = set()
        unique = []
        for item in sorted(items, key=lambda x: x["timestamp"], reverse=True):
            key = item["headline"][:80]
            if key not in seen:
                seen.add(key)
                # Compute how many hours ago
                try:
                    dt = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    diff_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                    item["hours_before"] = round(diff_h, 1)
                except Exception:
                    item["hours_before"] = 0
                unique.append(item)

        _CACHE = unique
        _CACHE_EXPIRY = time.time() + _CACHE_TTL
        print(f"[news_live] cached {len(_CACHE)} headlines from RSS")
