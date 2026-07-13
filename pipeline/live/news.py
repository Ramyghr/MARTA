"""pipeline/live/news.py — multi-source live news aggregation and FinBERT sentiment scoring.

Sources (in priority order):
  1. RSS feeds — no API key, completely free (feedparser)
  2. StockTwits — free public stream, no API key (financial social sentiment,
     replaces the former Reddit/PRAW integration)
  3. NewsAPI.org — free tier 100 req/day (NEWSAPI_KEY env var)

Plus a market-wide (not per-article) signal:
  4. Fear & Greed Index — alternative.me, free, no API key, updates daily

FinBERT scoring mirrors SentimentAgent._score_sentiment:
  - Batch inference on ./models/finbert (CPU)
  - Exponential recency decay, half-life 6 h
  - 6-hour momentum signal

Cache backend: Redis with asset-specific TTLs (crypto 15 min, traditional 60 min),
falls back to in-memory dict when Redis is not running.

Requirements (all optional — feed degrades gracefully when missing):
  feedparser  — pip install feedparser   (RSS, no key needed)
  requests    — pip install requests     (StockTwits + NewsAPI.org; NewsAPI needs NEWSAPI_KEY)
  transformers— pip install transformers (FinBERT; also needs ./models/finbert directory)
  redis       — pip install redis        (caching; env: REDIS_URL)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feeds — no API key
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, list[str]] = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
    ],
    "macro": [
        "https://feeds.reuters.com/reuters/businessNews",
    ],
    "commodities": [
        "https://oilprice.com/rss/main",
    ],
}

# ---------------------------------------------------------------------------
# StockTwits — free public stream, no auth. One symbol per tracked asset.
# ---------------------------------------------------------------------------

STOCKTWITS_SYMBOLS: dict[str, str] = {
    "BTC_USDT": "BTC.X",
    "ETH_USDT": "ETH.X",
    "GOLD":     "GC_F",
    "EURUSD":   "EURUSD",
}

_STOCKTWITS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Fear & Greed Index — alternative.me, free, no auth, updates daily (crypto only)
# ---------------------------------------------------------------------------

_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
_FEAR_GREED_TTL = 3600  # 1 hour — matches the index's own daily-ish update cadence

# ---------------------------------------------------------------------------
# NewsAPI queries per asset
# ---------------------------------------------------------------------------

NEWSAPI_QUERIES: dict[str, str] = {
    "BTC_USDT": "bitcoin OR BTC cryptocurrency",
    "ETH_USDT": "ethereum OR ETH cryptocurrency",
    "XAUUSD":   "gold price XAU commodities",
    "EURUSD":   "EUR USD forex Federal Reserve ECB",
}

# Logical → newsapi query key
_NEWSAPI_ASSET_ALIAS: dict[str, str] = {
    "GOLD": "XAUUSD",
}

# ---------------------------------------------------------------------------
# Asset-to-category mapping for RSS keyword filtering
# ---------------------------------------------------------------------------

_ASSET_KEYWORDS: dict[str, list[str]] = {
    "BTC_USDT": ["bitcoin", "btc", "crypto", "blockchain"],
    "ETH_USDT": ["ethereum", "eth", "defi", "crypto"],
    "EURUSD":   ["eur", "usd", "euro", "dollar", "forex", "fed", "ecb"],
    "GOLD":     ["gold", "xau", "precious metal", "bullion"],
}

_MACRO_KEYWORDS = ["inflation", "interest rate", "recession", "gdp", "federal reserve", "central bank"]

# Cache TTLs (seconds)
_TTL_CRYPTO     = 15 * 60   # 15 min — crypto markets move fast
_TTL_TRADITIONAL = 60 * 60  # 60 min — FX/commodities are slower

# Sentiment scoring parameters — mirror SentimentAgent
_DECAY_HALF_LIFE_HOURS = 6.0
_WINDOW_HOURS = 24


# ---------------------------------------------------------------------------
# Cache backend (Redis or in-memory)
# ---------------------------------------------------------------------------

def _make_cache() -> Any:
    try:
        import redis as _redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        client = _redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        logger.info(f"[news] Redis connected at {url}")
        return client
    except Exception as exc:
        logger.warning(f"[news] Redis not available ({exc}), using in-memory cache")
        return _MemCache()


class _MemCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}

    def set(self, key: str, value: str, ex: int = 900) -> None:
        self._store[key] = (value, time.monotonic() + ex)

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        val, exp = entry
        if time.monotonic() > exp:
            del self._store[key]
            return None
        return val


# ---------------------------------------------------------------------------
# Article deduplication
# ---------------------------------------------------------------------------

class _Deduplicator:
    def __init__(self, maxlen: int = 2000) -> None:
        self._seen: set[str] = set()
        self._queue: list[str] = []
        self._maxlen = maxlen

    def is_new(self, url: str) -> bool:
        if url in self._seen:
            return False
        self._seen.add(url)
        self._queue.append(url)
        if len(self._queue) > self._maxlen:
            old = self._queue.pop(0)
            self._seen.discard(old)
        return True


# ---------------------------------------------------------------------------
# RSS polling (synchronous — run in executor)
# ---------------------------------------------------------------------------

def _fetch_rss_sync(category: str, urls: list[str]) -> list[dict[str, Any]]:
    try:
        import feedparser
    except ImportError:
        logger.warning("[news] feedparser not installed — RSS feeds disabled. pip install feedparser")
        return []

    articles: list[dict[str, Any]] = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link = getattr(entry, "link", url)
                published = getattr(entry, "published_parsed", None)
                if published:
                    ts = datetime(*published[:6], tzinfo=timezone.utc)
                else:
                    ts = datetime.now(timezone.utc)
                articles.append({
                    "headline": title,
                    "body": summary[:500],
                    "url": link,
                    "category": category,
                    "timestamp": ts.isoformat(),
                    "source": "rss",
                })
        except Exception as exc:
            logger.warning(f"[news] RSS feed error ({url}): {exc}")
    return articles


# ---------------------------------------------------------------------------
# StockTwits polling (synchronous — run in executor). No auth required.
# ---------------------------------------------------------------------------

def _fetch_stocktwits_sync(asset: str) -> list[dict[str, Any]]:
    """Latest StockTwits messages for one tracked asset.

    Tagged with an explicit `"asset"` field (unlike RSS/NewsAPI articles, which are
    matched to an asset after the fact via keyword search in `_is_relevant`) since
    the StockTwits stream is already fetched per-symbol — no keyword guessing needed.
    """
    symbol = STOCKTWITS_SYMBOLS.get(asset)
    if not symbol:
        return []

    try:
        import requests
    except ImportError:
        logger.warning("[news] requests not installed — StockTwits feed disabled")
        return []

    try:
        resp = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
            params={"limit": 30},
            # StockTwits' edge (Cloudflare) 403s the default python-requests UA even
            # though the endpoint itself needs no auth — a normal browser UA clears it.
            headers={"User-Agent": _STOCKTWITS_USER_AGENT},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.debug(f"[news] StockTwits {asset} ({symbol}): HTTP {resp.status_code}")
            return []
        data = resp.json()
    except Exception as exc:
        logger.warning(f"[news] StockTwits {asset} error: {exc}")
        return []

    articles: list[dict[str, Any]] = []
    for msg in data.get("messages", []):
        body = (msg.get("body") or "")[:500]
        if not body:
            continue
        try:
            ts = datetime.strptime(msg.get("created_at", ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        sentiment = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")  # "Bullish" | "Bearish" | None
        articles.append({
            "headline": body[:200],
            "body": body,
            "url": f"https://stocktwits.com/message/{msg.get('id')}",
            "category": "social",
            "timestamp": ts.isoformat(),
            "source": "stocktwits",
            "asset": asset,
            "sentiment": sentiment,
        })
    return articles


# ---------------------------------------------------------------------------
# Fear & Greed Index (synchronous — run in executor). No auth required.
# ---------------------------------------------------------------------------

def _fng_label(value: int) -> str:
    if value >= 75: return "extreme_greed"
    if value >= 55: return "greed"
    if value >= 45: return "neutral"
    if value >= 25: return "fear"
    return "extreme_fear"


def _fetch_fear_greed_sync() -> dict[str, Any]:
    try:
        import requests
    except ImportError:
        logger.warning("[news] requests not installed — Fear & Greed feed disabled")
        return {"value": 50, "classification": "Neutral", "label": "neutral"}

    try:
        resp = requests.get(_FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        fng = resp.json().get("data", [{}])[0]
        value = int(fng.get("value", 50))
        return {
            "value": value,
            "classification": fng.get("value_classification", "Neutral"),
            "timestamp": fng.get("timestamp", ""),
            "label": _fng_label(value),
        }
    except Exception as exc:
        logger.warning(f"[news] Fear & Greed fetch error: {exc}")
        return {"value": 50, "classification": "Neutral", "label": "neutral"}


# ---------------------------------------------------------------------------
# NewsAPI.org polling (synchronous — run in executor)
# ---------------------------------------------------------------------------

def _fetch_newsapi_sync(asset: str) -> list[dict[str, Any]]:
    # Accept both NEWSAPI_KEY and the dashboard's NEWS_API_KEY convention.
    api_key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")
    if not api_key:
        logger.debug("[news] NEWSAPI_KEY not set — NewsAPI feed skipped")
        return []

    try:
        import requests
    except ImportError:
        logger.warning("[news] requests not installed — NewsAPI feed skipped")
        return []

    resolved = _NEWSAPI_ASSET_ALIAS.get(asset, asset)
    query = NEWSAPI_QUERIES.get(resolved)
    if not query:
        return []

    from_dt = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_dt,
                "sortBy": "publishedAt",
                "pageSize": 20,
                "language": "en",
                "apiKey": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"[news] NewsAPI request failed for {asset}: {exc}")
        return []

    articles: list[dict[str, Any]] = []
    for a in data.get("articles", []):
        ts_raw = a.get("publishedAt", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
        articles.append({
            "headline": a.get("title", ""),
            "body": a.get("description", "")[:500],
            "url": a.get("url", ""),
            "category": "news",
            "timestamp": ts.isoformat(),
            "source": "newsapi",
        })
    return articles


# ---------------------------------------------------------------------------
# Relevance filtering
# ---------------------------------------------------------------------------

def _is_relevant(article: dict[str, Any], asset: str) -> bool:
    """Return True when the article is relevant to *asset* (or is macro)."""
    # StockTwits messages are already fetched per-symbol and tagged with an explicit
    # asset — trust that instead of keyword-guessing against the message body.
    if "asset" in article:
        return article["asset"] == asset
    text = (article.get("headline", "") + " " + article.get("body", "")).lower()
    keywords = _ASSET_KEYWORDS.get(asset, [])
    if any(kw in text for kw in keywords):
        return True
    if any(kw in text for kw in _MACRO_KEYWORDS):
        return True
    return False


# ---------------------------------------------------------------------------
# FinBERT scoring
# ---------------------------------------------------------------------------

def _score_with_finbert(
    finbert: Any,
    headlines: list[str],
    timestamps: list[datetime],
    now: datetime,
) -> dict[str, Any]:
    """Run FinBERT on *headlines* with exponential recency decay.

    Returns the same schema as SentimentAgent._score_sentiment.
    """
    if not headlines:
        return {
            "positive": 0.33, "negative": 0.33, "neutral": 0.34,
            "momentum": 0.0, "article_count": 0, "top_headlines": [],
        }

    try:
        raw_results: list = finbert(headlines, batch_size=8)
    except Exception as exc:
        logger.warning(f"[news] FinBERT inference failed: {exc}")
        raw_results = []

    decay_k = math.log(2) / _DECAY_HALF_LIFE_HOURS
    total_w = pos_w = neg_w = neu_w = 0.0
    raw_pos_all: list[float] = []
    raw_pos_6h: list[float] = []
    scored: list[tuple[float, str]] = []  # (|pos - neg|, headline) for top-3

    for i, (hl, ts) in enumerate(zip(headlines, timestamps)):
        hours_ago = max((now - ts).total_seconds() / 3600.0, 0.0)
        weight = math.exp(-decay_k * hours_ago)

        if i < len(raw_results):
            scores = raw_results[i]
            score_map: dict[str, float] = (
                {s["label"].lower(): s["score"] for s in scores}
                if isinstance(scores, list)
                else {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            )
        else:
            score_map = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        pos = score_map.get("positive", 0.0)
        neg = score_map.get("negative", 0.0)
        neu = score_map.get("neutral", 0.0)

        pos_w += weight * pos
        neg_w += weight * neg
        neu_w += weight * neu
        total_w += weight

        raw_pos_all.append(pos)
        if hours_ago <= 6.0:
            raw_pos_6h.append(pos)
        scored.append((abs(pos - neg), hl))

    if total_w > 1e-9:
        pos_agg = pos_w / total_w
        neg_agg = neg_w / total_w
        neu_agg = neu_w / total_w
    else:
        pos_agg = neg_agg = neu_agg = 1.0 / 3.0

    avg_all = sum(raw_pos_all) / len(raw_pos_all) if raw_pos_all else 0.33
    avg_6h  = sum(raw_pos_6h)  / len(raw_pos_6h)  if raw_pos_6h  else avg_all
    momentum = round(avg_6h - avg_all, 4)

    top3 = [hl for _, hl in sorted(scored, reverse=True)[:3]]

    return {
        "positive":      round(pos_agg, 4),
        "negative":      round(neg_agg, 4),
        "neutral":       round(neu_agg, 4),
        "momentum":      momentum,
        "article_count": len(headlines),
        "top_headlines": top3,
    }


# ---------------------------------------------------------------------------
# LiveNewsFeed
# ---------------------------------------------------------------------------

class LiveNewsFeed:
    """Aggregates news from RSS, StockTwits, and NewsAPI; scores with FinBERT.
    Also exposes the crypto Fear & Greed Index (get_fear_greed) as a market-wide signal.

    Usage::

        feed = LiveNewsFeed()
        await feed.start_polling()

        sentiment = await feed.get_sentiment("BTC_USDT")
        # {"positive": 0.72, "negative": 0.15, "neutral": 0.13,
        #  "momentum": 0.05, "article_count": 14, "top_headlines": [...]}

    Polling loops run in the background.  Call ``stop()`` to cancel them.
    """

    def __init__(self) -> None:
        self._cache = _make_cache()
        self._dedup = _Deduplicator()
        self._articles: list[dict[str, Any]] = []     # global recent article pool
        self._tasks: list[asyncio.Task] = []
        self._finbert = self._load_finbert()

    # ----------------------------------------------------------------- FinBERT

    def _load_finbert(self) -> Any | None:
        finbert_path = os.path.join(
            os.environ.get("MARTA_ROOT", os.path.abspath(os.path.join(__file__, "../../.."))),
            "models", "finbert",
        )
        try:
            from transformers import pipeline as hf_pipeline
            if not os.path.isdir(finbert_path):
                logger.warning(f"[news] FinBERT not found at {finbert_path} — scoring disabled")
                return None
            fb = hf_pipeline(
                "text-classification",
                model=finbert_path,
                device=-1,     # CPU
                top_k=None,
            )
            logger.info("[news] FinBERT loaded from local checkpoint")
            return fb
        except ImportError:
            logger.warning("[news] transformers not installed — FinBERT scoring disabled")
            return None
        except Exception as exc:
            logger.warning(f"[news] FinBERT load failed: {exc}")
            return None

    # --------------------------------------------------------------- ingestion

    def _ingest(self, new_articles: list[dict[str, Any]]) -> int:
        """Deduplicate and append articles; return count of new articles added."""
        added = 0
        for a in new_articles:
            url = a.get("url", "")
            if url and not self._dedup.is_new(url):
                continue
            a.setdefault("_ingested_at", datetime.now(timezone.utc).isoformat())
            self._articles.append(a)
            added += 1
        # Trim to 2 000 most-recent articles
        if len(self._articles) > 2000:
            self._articles = self._articles[-2000:]
        return added

    # --------------------------------------------------------- polling helpers

    async def _poll_rss(self) -> None:
        loop = asyncio.get_event_loop()
        logger.info("[news] RSS polling started")
        while True:
            total = 0
            for category, urls in RSS_FEEDS.items():
                arts = await loop.run_in_executor(None, _fetch_rss_sync, category, urls)
                total += self._ingest(arts)
            logger.debug(f"[news] RSS: +{total} new articles")
            await asyncio.sleep(15 * 60)      # every 15 min

    async def _poll_stocktwits(self) -> None:
        loop = asyncio.get_event_loop()
        logger.info("[news] StockTwits polling started")
        while True:
            total = 0
            for asset in STOCKTWITS_SYMBOLS:
                arts = await loop.run_in_executor(None, _fetch_stocktwits_sync, asset)
                total += self._ingest(arts)
                await asyncio.sleep(1)        # gentle spacing between symbol requests
            logger.debug(f"[news] StockTwits: +{total} new messages")
            await asyncio.sleep(15 * 60)      # every 15 min — matches crypto TTL

    async def _poll_newsapi(self) -> None:
        loop = asyncio.get_event_loop()
        logger.info("[news] NewsAPI polling started")
        while True:
            total = 0
            for asset in NEWSAPI_QUERIES:
                arts = await loop.run_in_executor(None, _fetch_newsapi_sync, asset)
                total += self._ingest(arts)
                await asyncio.sleep(2)        # gentle rate-limit spacing
            logger.debug(f"[news] NewsAPI: +{total} new articles")
            await asyncio.sleep(60 * 60)      # every 1 h

    # ----------------------------------------------------------------- public

    async def start_polling(self) -> None:
        """Start all news polling background tasks."""
        self._tasks.append(asyncio.create_task(self._poll_rss(),        name="news_rss"))
        self._tasks.append(asyncio.create_task(self._poll_stocktwits(), name="news_stocktwits"))
        self._tasks.append(asyncio.create_task(self._poll_newsapi(),    name="news_newsapi"))
        logger.info("[news] News polling started (RSS + StockTwits + NewsAPI)")

    async def stop(self) -> None:
        """Cancel all background tasks."""
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("[news] News polling stopped")

    def score_headlines(self, headlines: list[str]) -> dict[str, Any]:
        """Run FinBERT on *headlines* synchronously.

        Returns: positive, negative, neutral, momentum, article_count, top_headlines.
        Falls back to uniform scores (0.33 each) when FinBERT is unavailable.
        """
        now = datetime.now(timezone.utc)
        timestamps = [now] * len(headlines)   # no age info when called ad-hoc
        return _score_with_finbert(self._finbert, headlines, timestamps, now)

    async def get_sentiment(self, asset: str) -> dict[str, Any]:
        """Return current sentiment for *asset* from all aggregated sources.

        Caches results with asset-specific TTL (crypto 15 min, traditional 60 min).
        Falls back to neutral scores when no articles are available.
        """
        # Check cache first.
        cache_key = f"marta:news:sentiment:{asset}"
        cached_raw = self._cache.get(cache_key)
        if cached_raw:
            return json.loads(cached_raw)

        # Trigger a synchronous fetch if pool is empty.
        if not self._articles:
            loop = asyncio.get_event_loop()
            for category, urls in RSS_FEEDS.items():
                arts = await loop.run_in_executor(None, _fetch_rss_sync, category, urls)
                self._ingest(arts)

        # Filter articles relevant to this asset within the last 24 h.
        cutoff = datetime.now(timezone.utc) - timedelta(hours=_WINDOW_HOURS)
        relevant: list[tuple[str, datetime]] = []
        for a in self._articles:
            try:
                ts = datetime.fromisoformat(a["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            if ts < cutoff:
                continue
            if _is_relevant(a, asset):
                relevant.append((a["headline"], ts))

        if not relevant:
            result: dict[str, Any] = {
                "positive": 0.33, "negative": 0.33, "neutral": 0.34,
                "momentum": 0.0, "article_count": 0, "top_headlines": [],
                "asset": asset,
            }
            ttl = _TTL_CRYPTO if asset in ("BTC_USDT", "ETH_USDT") else _TTL_TRADITIONAL
            self._cache.set(cache_key, json.dumps(result), ex=ttl)
            return result

        headlines, timestamps = zip(*relevant)
        now = datetime.now(timezone.utc)
        scores = _score_with_finbert(self._finbert, list(headlines), list(timestamps), now)
        scores["asset"] = asset

        ttl = _TTL_CRYPTO if asset in ("BTC_USDT", "ETH_USDT") else _TTL_TRADITIONAL
        self._cache.set(cache_key, json.dumps(scores), ex=ttl)
        return scores

    async def get_stocktwits_sentiment(self, asset: str) -> dict[str, Any]:
        """Bullish/bearish split of the last 30 StockTwits messages for *asset*.

        Fetched fresh per-symbol (not filtered out of the shared article pool) since
        StockTwits already scopes messages to one symbol — cached with the same
        asset-tiered TTL as get_sentiment().
        """
        cache_key = f"marta:news:stocktwits:{asset}"
        cached_raw = self._cache.get(cache_key)
        if cached_raw:
            return json.loads(cached_raw)

        loop = asyncio.get_event_loop()
        messages = await loop.run_in_executor(None, _fetch_stocktwits_sync, asset)

        bullish = sum(1 for m in messages if m.get("sentiment") == "Bullish")
        bearish = sum(1 for m in messages if m.get("sentiment") == "Bearish")
        total = len(messages) or 1
        result = {
            "asset": asset,
            "message_count": len(messages),
            "bullish_pct": round(bullish / total, 4),
            "bearish_pct": round(bearish / total, 4),
        }
        ttl = _TTL_CRYPTO if asset in ("BTC_USDT", "ETH_USDT") else _TTL_TRADITIONAL
        self._cache.set(cache_key, json.dumps(result), ex=ttl)
        return result

    async def get_fear_greed(self) -> dict[str, Any]:
        """Current crypto Fear & Greed Index — market-wide, not per-asset. 1h cache."""
        cache_key = "marta:news:fear_greed"
        cached_raw = self._cache.get(cache_key)
        if cached_raw:
            return json.loads(cached_raw)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch_fear_greed_sync)
        self._cache.set(cache_key, json.dumps(result), ex=_FEAR_GREED_TTL)
        return result
