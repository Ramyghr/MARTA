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
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

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

# ── Sentiment scoring — FinBERT, not keyword-matching ──────────────────────────
#
# This panel used to label every headline via _quick_sentiment(): a fixed bag-of-words
# list ("surge"/"rally" vs "crash"/"hack"), duplicated (worse) again client-side in
# NewsPanel.tsx's sentimentFromText(). Neither reads the sentence, so "gold isn't a
# scam" or "no crash expected" score as bearish. Real trading agents (agents/
# sentiment_agent.py, pipeline/live/news.py) already use FinBERT for this — this panel
# should show the same methodology it's implicitly claiming to (the label says
# "Live Sentiment Feed", and users reasonably read the bars as analysis, not a word list).
#
# One module-level singleton, loaded lazily on first use and reused for the life of the
# process — this dashboard backend can already have up to two other independent FinBERT
# instances in memory (agents/sentiment_agent.py's SentimentAgent, and debate_stream.py's
# LiveNewsFeed) once a live debate has run once; a fourth reload per request would be
# both slow and wasteful. Falls back to the old keyword heuristic if transformers/the
# model directory isn't available — same graceful-degradation pattern as everywhere else
# in this codebase, not a hard dependency.
_finbert: Any = None
_finbert_load_attempted = False


def _get_finbert() -> Any | None:
    global _finbert, _finbert_load_attempted
    if _finbert_load_attempted:
        return _finbert
    _finbert_load_attempted = True
    try:
        from transformers import pipeline as hf_pipeline
        finbert_path = os.path.join(
            os.environ.get("MARTA_ROOT", os.path.abspath(os.path.join(__file__, "../.."))),
            "models", "finbert",
        )
        if not os.path.isdir(finbert_path):
            print(f"[news_live] FinBERT not found at {finbert_path} — falling back to keyword sentiment")
            return None
        _finbert = _load_finbert_gpu_first(hf_pipeline, finbert_path)
        return _finbert
    except Exception as e:
        print(f"[news_live] FinBERT load failed ({e}) — falling back to keyword sentiment")
        return None


def _load_finbert_gpu_first(hf_pipeline: Any, model_path: str) -> Any:
    """Try GPU first, fall back to CPU if CUDA isn't available or the model can't
    actually be used on it (e.g. the GPU's 4GB VRAM already claimed by encoder
    training or Ollama — a real, observed contention case on this project's dev box;
    this container has no GPU passthrough at all, so this always resolves to CPU here
    today — see the "GPU vs CPU" note in project memory for why that's deliberately
    left as-is for this container's other CPU-only inference).

    torch.cuda.is_available() only confirms a driver/device exists, not that there's
    free VRAM right now, so this also runs one cheap canary inference before trusting
    the GPU load — surfaces an OOM here, at first use, instead of mid-request.
    """
    try:
        import torch
        if torch.cuda.is_available():
            try:
                pipe = hf_pipeline("text-classification", model=model_path, device=0, top_k=None)
                pipe("canary")
                print("[news_live] FinBERT loaded on GPU (cuda:0)")
                return pipe
            except Exception as exc:
                print(f"[news_live] FinBERT GPU load failed ({exc}) — falling back to CPU")
    except ImportError:
        pass
    pipe = hf_pipeline("text-classification", model=model_path, device=-1, top_k=None)
    print("[news_live] FinBERT loaded on CPU — scoring headlines with it instead of keyword-matching")
    return pipe


def _finbert_sentiment_batch(texts: list[str]) -> list[str]:
    """Batch-score `texts`, one "bullish"|"bearish"|"neutral" label per text (FinBERT's
    positive/negative/neutral mapped onto the labels this panel already displays).
    Falls back to _quick_sentiment per-text when FinBERT isn't available."""
    if not texts:
        return []
    finbert = _get_finbert()
    if finbert is None:
        return [_quick_sentiment(t) for t in texts]

    try:
        raw = finbert(texts, batch_size=8, truncation=True)
    except Exception as e:
        print(f"[news_live] FinBERT batch inference failed ({e}) — falling back to keyword sentiment")
        return [_quick_sentiment(t) for t in texts]

    _LABEL_MAP = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
    out: list[str] = []
    for scores in raw:
        if not isinstance(scores, list) or not scores:
            out.append("neutral")
            continue
        top = max(scores, key=lambda s: s["score"])
        out.append(_LABEL_MAP.get(top["label"].lower(), "neutral"))
    return out


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
    # feeds.reuters.com/reuters/technologyNews (the old value here) is dead — Reuters
    # retired classic RSS years ago (DNS failure, verified 2026-08-11; confirmed live
    # in the container logs too: "fetch error (Reuters): Name or service not known").
    # Replaced with sources verified live the same day — same fix applied to
    # pipeline/live/news.py's independent RSS_FEEDS list, kept in sync here.
    {
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "source": "CNBC",
        "assets": ["macro", "GOLD", "EURUSD"],
    },
    {
        "url": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
        "source": "Dow Jones",
        "assets": ["macro", "GOLD", "EURUSD"],
    },
    {
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "source": "BBC World",
        "assets": ["macro"],
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
                # None (not a fake "Neutral" placeholder) when the poster didn't
                # self-tag — resolved just below, scored once per 5-min refresh rather
                # than re-scored on every /api/news/live hit.
                "sentiment": (((msg.get("entities") or {}).get("sentiment") or {}).get("basic") or "").lower() or None,
                "asset": asset,
            }
            for msg in messages
            if msg.get("body")
        ]
        _resolve_stocktwits_sentiment(result)
        _STOCKTWITS_CACHE[asset] = result
        _STOCKTWITS_EXPIRY[asset] = now + _STOCKTWITS_TTL
        return result
    except Exception as e:
        print(f"[news_live] StockTwits {asset} error: {e}")
        return _STOCKTWITS_CACHE.get(asset, [])


def _resolve_stocktwits_sentiment(messages: list[dict]) -> None:
    """Mutates `messages` in place, filling in `sentiment` for any message that had no
    self-tagged Bullish/Bearish label (kept as `None` by the caller above). Those used
    to silently default to "Neutral", which reads as "the content is neutral" when it
    actually means "the poster didn't tag anything" — most of this panel's "48%
    Neutral" was really "untagged". Genuine self-tags are left as-is (a trader's own
    directional call is real signal, arguably more direct than an inferred one);
    untagged messages are FinBERT-scored on their body text instead of guessed at.
    Scored once per 5-min StockTwits refresh (see fetch_stocktwits), not per request.
    """
    untagged_idx = [i for i, m in enumerate(messages) if m["sentiment"] is None]
    if not untagged_idx:
        return
    scored = _finbert_sentiment_batch([messages[i]["headline"] for i in untagged_idx])
    for i, label in zip(untagged_idx, scored):
        messages[i]["sentiment"] = label


def _stocktwits_to_news_items(messages: list[dict]) -> list[dict]:
    """Map raw StockTwits messages (fetch_stocktwits' shape, sentiment already
    resolved — see _resolve_stocktwits_sentiment) onto the same NewsItem shape RSS
    headlines use, so the two sources merge cleanly for the News Panel."""
    items = []
    for m in messages:
        try:
            dt = datetime.strptime(m["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
        hours_before = round((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 1)
        items.append({
            "timestamp": dt.isoformat(),
            "asset": m["asset"],
            "headline": m["headline"],
            "summary": "",
            "source": "StockTwits",
            "url": m["url"],
            "sentiment": m["sentiment"] or "neutral",
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
            items.append({
                "timestamp": pub,
                "asset": asset,
                "headline": title,
                "summary": desc[:150],
                "source": source,
                "url": link,
                # Filled in by _maybe_refresh()'s batched _finbert_sentiment_batch()
                # call — scoring per-item here would mean scoring one headline at a
                # time instead of a single batch across the whole refresh.
                "sentiment": None,
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

        # Score once per refresh (10-min TTL) over the whole batch, not per request —
        # a live FinBERT call per /api/news/live hit would be both slow and redundant.
        labels = _finbert_sentiment_batch([item["headline"] for item in unique])
        for item, label in zip(unique, labels):
            item["sentiment"] = label

        _CACHE = unique
        _CACHE_EXPIRY = time.time() + _CACHE_TTL
        print(f"[news_live] cached {len(_CACHE)} headlines from RSS")
