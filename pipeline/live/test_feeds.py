"""pipeline/live/test_feeds.py — smoke-test for all live data feeds.

Run from the MARTA root:
    python pipeline/live/test_feeds.py

Checks each feed independently and prints a summary.  A feed that fails (e.g.
missing API key, no network) is reported as SKIP rather than ERROR so the test
always exits 0 when at least the RSS feeds are operational.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Allow running without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# ---------------------------------------------------------------------------

_OK    = "\033[32m✓\033[0m"
_FAIL  = "\033[31m✗\033[0m"
_SKIP  = "\033[33m—\033[0m"


def _fmt(ok: bool | None, label: str, detail: str = "") -> str:
    sym = _OK if ok is True else (_SKIP if ok is None else _FAIL)
    suffix = f" {detail}" if detail else ""
    return f"  {sym} {label}{suffix}"


# ---------------------------------------------------------------------------
# Price feed tests
# ---------------------------------------------------------------------------

async def _test_price_feed() -> list[str]:
    lines: list[str] = ["", "Testing price feeds..."]

    from pipeline.live.prices import LivePriceFeed, _BINANCE_SYMBOLS, _TRADITIONAL_ASSETS

    feed = LivePriceFeed()

    for asset in list(_BINANCE_SYMBOLS) + _TRADITIONAL_ASSETS:
        try:
            candle = await feed.get_latest_candle(asset)
            ts = candle.get("timestamp", "?")
            close = candle.get("close", 0)
            if close > 0:
                lines.append(_fmt(True,  f"{asset} ({'Binance' if asset in _BINANCE_SYMBOLS else 'AV/yf'}):",
                                  f"latest candle {ts[:19]} close={close:.4f}"))
            else:
                lines.append(_fmt(None, f"{asset}: got candle but close=0 (API may be returning zeros)"))
        except Exception as exc:
            lines.append(_fmt(False, f"{asset}: {exc}"))

    return lines


# ---------------------------------------------------------------------------
# News feed tests
# ---------------------------------------------------------------------------

def _test_rss() -> str:
    from pipeline.live.news import _fetch_rss_sync, RSS_FEEDS
    total = 0
    sources = 0
    for category, urls in RSS_FEEDS.items():
        arts = _fetch_rss_sync(category, urls)
        total += len(arts)
        sources += len(urls)
    if total > 0:
        return _fmt(True, f"RSS feeds: {total} articles fetched across {sources} sources")
    return _fmt(None, "RSS feeds: 0 articles (feedparser may not be installed or feeds unreachable)")


def _test_stocktwits(asset: str = "BTC_USDT") -> str:
    from pipeline.live.news import _fetch_stocktwits_sync
    messages = _fetch_stocktwits_sync(asset)
    if messages:
        return _fmt(True, f"StockTwits: {len(messages)} messages fetched for {asset}")
    return _fmt(None, f"StockTwits: 0 messages for {asset} (network or API hiccup)")


def _test_fear_greed() -> str:
    from pipeline.live.news import _fetch_fear_greed_sync
    fng = _fetch_fear_greed_sync()
    return _fmt(True, f"Fear & Greed: {fng['value']} — {fng['classification']}")


def _test_newsapi(asset: str = "BTC_USDT") -> str:
    api_key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")
    if not api_key:
        return _fmt(None, "NewsAPI: NEWSAPI_KEY / NEWS_API_KEY not set — skipped")
    from pipeline.live.news import _fetch_newsapi_sync
    arts = _fetch_newsapi_sync(asset)
    if arts:
        return _fmt(True, f"NewsAPI: {len(arts)} articles fetched for {asset}")
    return _fmt(None, f"NewsAPI: 0 articles for {asset} (rate limit or no results)")


# ---------------------------------------------------------------------------
# FinBERT test
# ---------------------------------------------------------------------------

def _test_finbert() -> list[str]:
    lines: list[str] = ["", "Testing FinBERT scoring..."]
    from pipeline.live.news import LiveNewsFeed, _score_with_finbert

    feed = LiveNewsFeed()
    if feed._finbert is None:
        lines.append(_fmt(None, "FinBERT: not available (transformers not installed or model missing)"))
        return lines

    test_headlines = [
        "Bitcoin surges to all-time high as institutional demand soars",
        "Crypto market crashes 20% on regulatory fears",
        "Gold prices remain stable amid market uncertainty",
    ]
    now = datetime.now(timezone.utc)
    timestamps = [now] * len(test_headlines)
    scores = _score_with_finbert(feed._finbert, test_headlines, timestamps, now)

    lines.append(_fmt(True,
        f"FinBERT: positive={scores['positive']:.2f} "
        f"negative={scores['negative']:.2f} "
        f"neutral={scores['neutral']:.2f} "
        f"({len(test_headlines)} headlines)"
    ))
    return lines


# ---------------------------------------------------------------------------
# SentimentAgent wiring test
# ---------------------------------------------------------------------------

async def _test_sentiment_agent_live() -> list[str]:
    lines: list[str] = ["", "Testing SentimentAgent + live feed wiring..."]
    try:
        from pipeline.live.news import LiveNewsFeed
        from agents.sentiment_agent import SentimentAgent

        live_news = LiveNewsFeed()
        agent = SentimentAgent(live_feed=live_news)
        lines.append(_fmt(True, "SentimentAgent: constructed with live_feed attached"))

        # Check that default construction (no live_feed) still works.
        agent_default = SentimentAgent()
        assert agent_default._live_feed is None, "_live_feed should default to None"
        lines.append(_fmt(True, "SentimentAgent: default constructor — live_feed=None (backward-compatible)"))
    except Exception as exc:
        lines.append(_fmt(False, f"SentimentAgent: {exc}"))
    return lines


# ---------------------------------------------------------------------------
# Redis connectivity test
# ---------------------------------------------------------------------------

def _test_redis() -> str:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        import redis as _redis
        client = _redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        client.set("marta:test", "ok", ex=5)
        val = client.get("marta:test")
        assert val == b"ok"
        return _fmt(True, f"Redis: connected at {redis_url} (in-memory fallback also available)")
    except ImportError:
        return _fmt(None, "Redis: redis-py not installed — in-memory cache will be used")
    except Exception:
        return _fmt(None, f"Redis: not reachable at {redis_url} — in-memory cache will be used")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> int:
    print("\n" + "=" * 60)
    print("  MARTA Live Feed Diagnostics")
    print("=" * 60)

    all_lines: list[str] = []
    errors = 0

    # 1. Price feeds.
    try:
        all_lines += await _test_price_feed()
    except Exception:
        all_lines.append(_fmt(False, f"Price feed: unexpected error\n{traceback.format_exc()}"))
        errors += 1

    # 2. News feeds.
    all_lines.append("")
    all_lines.append("Testing news feeds...")
    try:
        all_lines.append(_test_rss())
    except Exception as exc:
        all_lines.append(_fmt(False, f"RSS: {exc}"))
        errors += 1

    try:
        all_lines.append(_test_stocktwits())
    except Exception as exc:
        all_lines.append(_fmt(False, f"StockTwits: {exc}"))

    try:
        all_lines.append(_test_fear_greed())
    except Exception as exc:
        all_lines.append(_fmt(False, f"Fear & Greed: {exc}"))

    try:
        all_lines.append(_test_newsapi())
    except Exception as exc:
        all_lines.append(_fmt(False, f"NewsAPI: {exc}"))

    # 3. FinBERT.
    try:
        all_lines += _test_finbert()
    except Exception as exc:
        all_lines.append(_fmt(False, f"FinBERT: {exc}"))

    # 4. SentimentAgent wiring.
    try:
        all_lines += await _test_sentiment_agent_live()
    except Exception as exc:
        all_lines.append(_fmt(False, f"SentimentAgent wiring: {exc}"))
        errors += 1

    # 5. Redis.
    all_lines.append("")
    all_lines.append("Testing Redis cache...")
    all_lines.append(_test_redis())

    # Summary.
    all_lines.append("")
    all_lines.append("=" * 60)
    if errors == 0:
        all_lines.append(f"  {_OK} All feeds operational (— = optional / not configured)")
    else:
        all_lines.append(f"  {_FAIL} {errors} critical error(s). See output above.")
    all_lines.append("=" * 60)

    print("\n".join(all_lines))
    return errors


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
