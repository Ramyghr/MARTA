"""SentimentAgent — FinBERT news sentiment analysis and directional signal generation."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    hf_pipeline = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

if TYPE_CHECKING:
    from pipeline.live.news import LiveNewsFeed

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentThesis
from utils.logger import logger

_MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent)))
_FINBERT_PATH = _MARTA_ROOT / "models/finbert"
_NEWS_CACHE_PATH = _MARTA_ROOT / "data/news_cache.json"
_DECAY_HALF_LIFE_HOURS = 6.0
_WINDOW_HOURS = 24
# Kept at the original values — dashboard/backend/debate_stream.py constructs
# SentimentAgent() (via live_feed=...) relying on these defaults, and live GOLD/EURUSD
# debates could plausibly see an "EURUSD"-tagged headline even though the static
# data/news_cache.json never has (verified: only BTC_USDT/ETH_USDT/GOLD/macro exist
# there). Crypto-only callers (run_debate.py) pass macro_assets=/fear_greed_assets=
# explicitly instead of relying on these.
_MACRO_ASSETS = {"macro", "EURUSD"}
# Fear & Greed is a crypto-market-wide index — weighting it into e.g. a GOLD thesis
# would inject an unrelated crypto signal, so it only applies to the assets it's
# actually about.
_FEAR_GREED_ASSETS = {"BTC_USDT", "ETH_USDT"}


def _load_finbert_pipeline(model_path: str):
    """Try GPU first, fall back to CPU if CUDA isn't available or the model can't
    actually be used on it (e.g. the GPU's 4GB VRAM already claimed by encoder
    training or Ollama — a real, observed contention case on this project's dev box).

    torch.cuda.is_available() only confirms a driver/device exists, not that there's
    free VRAM right now, so this also runs one cheap canary inference before trusting
    the GPU load — surfaces an OOM here, at process startup, instead of mid-debate.
    """
    try:
        import torch
        if torch.cuda.is_available():
            try:
                pipe = hf_pipeline("text-classification", model=model_path, device=0, top_k=None)
                pipe("canary")
                logger.info("SentimentAgent: FinBERT loaded on GPU (cuda:0)")
                return pipe
            except Exception as exc:
                logger.warning(f"SentimentAgent: FinBERT GPU load failed ({exc}) — falling back to CPU")
    except ImportError:
        pass
    pipe = hf_pipeline("text-classification", model=model_path, device=-1, top_k=None)
    logger.info("SentimentAgent: FinBERT loaded on CPU")
    return pipe


class SentimentAgent(BaseAgent):
    name = "SentimentAgent"

    def __init__(
        self,
        model: str = "mistral:7b-instruct",
        temperature: float = 0.4,
        live_feed: "LiveNewsFeed | None" = None,
        macro_assets: set[str] | None = None,
        fear_greed_assets: set[str] | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self._macro_assets = macro_assets if macro_assets is not None else _MACRO_ASSETS
        self._fear_greed_assets = fear_greed_assets if fear_greed_assets is not None else _FEAR_GREED_ASSETS
        if _TRANSFORMERS_AVAILABLE and _FINBERT_PATH.exists():
            self._finbert = _load_finbert_pipeline(str(_FINBERT_PATH))
        else:
            self._finbert = None
            if not _TRANSFORMERS_AVAILABLE:
                logger.warning("SentimentAgent: transformers not installed — FinBERT disabled, returning neutral thesis")
            elif not _FINBERT_PATH.exists():
                logger.warning(f"SentimentAgent: FinBERT model not found at {_FINBERT_PATH} — returning neutral thesis")
        self._news_cache: list[dict] = self._load_news_cache()
        # Optional live news feed; None = use static cache (historical/replay mode).
        self._live_feed: "LiveNewsFeed | None" = live_feed
        if live_feed is not None:
            logger.info("SentimentAgent: live news feed attached — historical cache is backup")
        logger.info(f"SentimentAgent: loaded {len(self._news_cache)} headlines from cache")

    def _load_news_cache(self) -> list[dict]:
        if not _NEWS_CACHE_PATH.exists():
            logger.warning(f"SentimentAgent: news cache not found at {_NEWS_CACHE_PATH}")
            return []
        with open(_NEWS_CACHE_PATH) as f:
            data = json.load(f)
        return data.get("headlines", [])

    def _get_headlines(
        self, asset: str, timestamp: pd.Timestamp, window_hours: int = _WINDOW_HOURS,
        live_headlines: list | None = None,
    ) -> list[dict]:
        """Return headlines for asset (+ macro).

        When live_headlines are provided (live dashboard mode), uses those directly
        with no time-window filtering (they are already fresh). Falls back to the
        static news cache for historical/replay mode.
        """
        if live_headlines is not None:
            matched = []
            for h in live_headlines:
                h_asset = h.get("asset", "macro")
                if h_asset == asset or h_asset in self._macro_assets or h_asset == "macro":
                    matched.append({**h, "_ts": pd.Timestamp.now(tz="UTC")})
            return matched

        ts_utc = (
            pd.Timestamp(timestamp).tz_convert("UTC")
            if pd.Timestamp(timestamp).tzinfo
            else pd.Timestamp(timestamp).tz_localize("UTC")
        )
        cutoff = ts_utc - pd.Timedelta(hours=window_hours)

        matched: list[dict] = []
        for h in self._news_cache:
            raw_ts = pd.Timestamp(h["timestamp"])
            h_ts = raw_ts.tz_localize("UTC") if raw_ts.tzinfo is None else raw_ts.tz_convert("UTC")
            if h_ts < cutoff or h_ts > ts_utc:
                continue
            h_asset = h.get("asset", "")
            if h_asset == asset or h_asset in self._macro_assets:
                matched.append({**h, "_ts": h_ts})

        matched.sort(key=lambda x: x["_ts"])
        return matched

    def _score_sentiment(
        self, headlines: list[dict], timestamp: pd.Timestamp
    ) -> dict:
        """Batch FinBERT inference with exponential recency decay; compute momentum signal."""
        if not headlines:
            return {
                "asset": "unknown",
                "positive_score": 0.33,
                "negative_score": 0.33,
                "neutral_score": 0.34,
                "momentum": 0.0,
                "article_count": 0,
                "dominant_sentiment": "neutral",
            }

        ts_utc = (
            pd.Timestamp(timestamp).tz_convert("UTC")
            if pd.Timestamp(timestamp).tzinfo
            else pd.Timestamp(timestamp).tz_localize("UTC")
        )

        if self._finbert is None:
            return {
                "asset": "unknown",
                "positive_score": 0.33,
                "negative_score": 0.33,
                "neutral_score": 0.34,
                "momentum": 0.0,
                "article_count": len(headlines),
                "dominant_sentiment": "neutral",
            }

        texts = [h["headline"] for h in headlines]
        try:
            raw_results: list = self._finbert(texts, batch_size=8)
        except Exception as e:
            logger.warning(f"SentimentAgent: FinBERT batch failed — {e}")
            raw_results = []

        decay_k = math.log(2) / _DECAY_HALF_LIFE_HOURS

        total_w = 0.0
        pos_w = neg_w = neu_w = 0.0
        raw_pos_all: list[float] = []
        raw_pos_6h: list[float] = []

        for i, h in enumerate(headlines):
            hours_ago = max((ts_utc - h["_ts"]).total_seconds() / 3600.0, 0.0)
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

        if total_w > 1e-9:
            pos_agg = pos_w / total_w
            neg_agg = neg_w / total_w
            neu_agg = neu_w / total_w
        else:
            pos_agg = neg_agg = neu_agg = 1.0 / 3.0

        avg_all = float(np.mean(raw_pos_all)) if raw_pos_all else 0.33
        avg_6h = float(np.mean(raw_pos_6h)) if raw_pos_6h else avg_all
        momentum = round(avg_6h - avg_all, 4)

        dominant = max(
            [("positive", pos_agg), ("negative", neg_agg), ("neutral", neu_agg)],
            key=lambda x: x[1],
        )[0]

        return {
            "asset": headlines[0].get("asset", "unknown"),
            "positive_score": round(pos_agg, 4),
            "negative_score": round(neg_agg, 4),
            "neutral_score": round(neu_agg, 4),
            "momentum": momentum,
            "article_count": len(headlines),
            "dominant_sentiment": dominant,
        }

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        return (
            "You are the Sentiment Agent for a quantitative trading system.\n"
            "You receive FinBERT sentiment scores from financial news headlines.\n"
            "Interpret the scores in the context of the current market regime and generate a directional thesis.\n"
            "You MUST respond with a single JSON object — no preamble, no markdown.\n\n"
            + serialize_context_for_prompt(ctx)
        )

    def _get_cached_sentiment(self, ctx: RegimeContext) -> dict:
        """Score sentiment from the static news cache (historical / replay mode)."""
        headlines = self._get_headlines(ctx.asset, ctx.timestamp)
        return self._score_sentiment(headlines, ctx.timestamp)

    @staticmethod
    async def _fetch_live_signals(live_feed: "LiveNewsFeed", asset: str) -> tuple[dict, dict | None, dict | None]:
        """Concurrently fetch FinBERT sentiment + StockTwits + Fear & Greed.

        StockTwits/Fear&Greed failures are independent of the primary FinBERT
        sentiment fetch (return_exceptions=True) — a social-data hiccup should
        enrich-or-omit, never take down the core sentiment signal.
        """
        fetch_fng = asset in self._fear_greed_assets
        results = await asyncio.gather(
            live_feed.get_sentiment(asset),
            live_feed.get_stocktwits_sentiment(asset),
            live_feed.get_fear_greed() if fetch_fng else asyncio.sleep(0, result=None),
            return_exceptions=True,
        )
        sentiment, stocktwits, fear_greed = results
        if isinstance(sentiment, BaseException):
            raise sentiment
        if isinstance(stocktwits, BaseException):
            logger.warning(f"SentimentAgent: StockTwits fetch failed ({stocktwits})")
            stocktwits = None
        if isinstance(fear_greed, BaseException):
            logger.warning(f"SentimentAgent: Fear & Greed fetch failed ({fear_greed})")
            fear_greed = None
        return sentiment, stocktwits, fear_greed

    def post_thesis(self, ctx: RegimeContext) -> AgentThesis:
        # Try live feed first; fall back to static cache on any failure. StockTwits/
        # Fear & Greed are live-only signals (there's no way to ask either for "what
        # crypto Twitter thought in March 2020") — they never apply in cached/replay
        # mode, so a historical debate's inputs stay reproducible.
        stocktwits: dict | None = None
        fear_greed: dict | None = None
        if self._live_feed is not None:
            try:
                raw_sentiment, stocktwits, fear_greed = asyncio.run(
                    self._fetch_live_signals(self._live_feed, ctx.asset)
                )
                # Remap live-feed keys to match _score_sentiment output schema.
                sentiment = {
                    "asset": ctx.asset,
                    "positive_score":    raw_sentiment.get("positive",  0.33),
                    "negative_score":    raw_sentiment.get("negative",  0.33),
                    "neutral_score":     raw_sentiment.get("neutral",   0.34),
                    "momentum":          raw_sentiment.get("momentum",  0.0),
                    "article_count":     raw_sentiment.get("article_count", 0),
                    "dominant_sentiment": (
                        "positive" if raw_sentiment.get("positive", 0) >= raw_sentiment.get("negative", 0)
                        and raw_sentiment.get("positive", 0) >= raw_sentiment.get("neutral", 0)
                        else ("negative" if raw_sentiment.get("negative", 0) >= raw_sentiment.get("neutral", 0)
                              else "neutral")
                    ),
                }
                logger.debug(
                    f"SentimentAgent: live feed ({sentiment['article_count']} articles) "
                    f"for {ctx.asset}"
                )
            except Exception as exc:
                logger.warning(f"SentimentAgent: live feed failed ({exc}), falling back to cache")
                sentiment = self._get_cached_sentiment(ctx)
        else:
            sentiment = self._get_cached_sentiment(ctx)

        sentiment_context = (
            f"\n=== SENTIMENT DATA (FinBERT, {sentiment['article_count']} articles, {_WINDOW_HOURS}h window) ===\n"
            f"Dominant: {sentiment['dominant_sentiment']} | "
            f"Positive: {sentiment['positive_score']:.3f} | "
            f"Negative: {sentiment['negative_score']:.3f} | "
            f"Neutral: {sentiment['neutral_score']:.3f}\n"
            f"6h momentum vs 24h avg: {sentiment['momentum']:+.3f}\n"
        )

        # StockTwits (financial social sentiment) + Fear & Greed (crypto-wide) enrich
        # the prompt when live — the LLM interprets the blend, same as it already does
        # for the FinBERT numbers; nothing here overrides directional_bias directly.
        if stocktwits is not None or fear_greed is not None:
            rss_score = sentiment["positive_score"] - sentiment["negative_score"]
            st_score = (stocktwits["bullish_pct"] - stocktwits["bearish_pct"]) if stocktwits else 0.0
            fng_score = ((fear_greed["value"] - 50) / 50) if fear_greed else 0.0
            combined_score = rss_score * 0.5 + st_score * 0.3 + fng_score * 0.2

            social_lines = [f"\n=== SOCIAL & MARKET SENTIMENT ({', '.join(s for s in ['rss_finbert', 'stocktwits' if stocktwits else None, 'fear_greed' if fear_greed else None] if s)}) ==="]
            if stocktwits:
                social_lines.append(
                    f"StockTwits ({stocktwits['message_count']} messages): "
                    f"{stocktwits['bullish_pct']:.0%} bullish / {stocktwits['bearish_pct']:.0%} bearish"
                )
            if fear_greed:
                social_lines.append(
                    f"Fear & Greed Index: {fear_greed['value']} — {fear_greed['classification']} "
                    f"({fear_greed['label'].replace('_', ' ')})"
                )
            social_lines.append(f"Combined weighted score (FinBERT 0.5 / StockTwits 0.3 / Fear&Greed 0.2): {combined_score:+.3f}\n")
            sentiment_context += "\n".join(social_lines)

        system_prompt = self._build_system_prompt(ctx) + sentiment_context

        
        user_prompt = (
            f"Assess directional bias for {ctx.asset} from the FinBERT sentiment scores and regime.\n\n"
            "Reply ONLY with this JSON schema — replace every placeholder with your own analysis, "
            "do not copy the placeholder values:\n"
            '{"agent_name":"SentimentAgent","round":1,"directional_bias":"<bullish|bearish|neutral, your judgment>",'
            '"confidence":<your confidence as a float between 0.0 and 1.0>,'
            '"reasoning":"<your 1-2 sentence reasoning>","key_factors":["<factor1>","<factor2>"],'
            '"regime_alignment":"<aligned|contrary|neutral|mixed>"}'
        )

        raw = self._call_llm(system_prompt, user_prompt)
        result = self._parse_json_response(raw, AgentThesis)
        if isinstance(result, AgentThesis):
            result.agent_name = "SentimentAgent"
            result.round = 1
            return result
        return result
