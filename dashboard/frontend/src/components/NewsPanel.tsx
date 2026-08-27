"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchApi } from "@/lib/websocket";
import { api } from "@/lib/api";
import type { NewsItem, FearGreedIndex } from "@/lib/types";

interface Props {
  asset: string;
  timestamp: string | null;
  liveMode?: boolean;
}

const SENTIMENT_COLOR: Record<string, string> = {
  bullish: "#00ff7f",
  bearish: "#ff3737",
  neutral: "#888",
};

const SOURCE_COLORS: Record<string, string> = {
  CoinDesk: "#4a9eff",
  CoinTelegraph: "#00d4ff",
  Decrypt: "#b07fff",
  Reuters: "#ffd700",
  Bloomberg: "#ff8c00",
  StockTwits: "#4a9eff",
  Twitter: "#1da1f2",
  macro: "#999999",
  static: "#666666",
};

const FEAR_GREED_ASSETS = new Set(["BTC_USDT", "ETH_USDT"]);

function fearGreedColor(value: number): string {
  if (value >= 55) return "#00ff7f";
  if (value >= 45) return "#ffd700";
  return "#ff3737";
}

function FearGreedGauge({ data }: { data: FearGreedIndex }) {
  const color = fearGreedColor(data.value);
  return (
    <div className="mx-2 mt-2 mb-1 p-2 rounded border border-[#1a1a1a] bg-[#0d0d0d] shrink-0">
      <div className="text-[9px] text-[#444] uppercase tracking-widest mb-1">Fear &amp; Greed Index</div>
      <div className="flex items-center gap-3">
        <div className="text-2xl font-bold font-mono" style={{ color }}>{data.value}</div>
        <div className="flex-1 min-w-0">
          <div className="text-[11px] text-[#ccc]">{data.classification}</div>
          <div className="text-[9px] text-[#555] uppercase tracking-wider">{data.label?.replace(/_/g, " ")}</div>
        </div>
        <div className="flex-1 h-2 rounded-full overflow-hidden bg-[#1a1a1a]">
          <div className="h-full rounded-full transition-all" style={{ width: `${data.value}%`, background: color }} />
        </div>
      </div>
    </div>
  );
}

const ASSET_TABS: Array<{ id: string; label: string }> = [
  { id: "BTC_USDT", label: "BTC" },
  { id: "ETH_USDT", label: "ETH" },
  { id: "BNB_USDT", label: "BNB" },
  { id: "LTC_USDT", label: "LTC" },
  { id: "XRP_USDT", label: "XRP" },
];

function sentimentFromText(text: string): "bullish" | "bearish" | "neutral" {
  const t = text.toLowerCase();
  const bull = ["surge", "rally", "bull", "record high", "gain", "rises", "institutional", "adoption", "approve", "etf", "breakout", "rebound", "climbs", "jumps", "wins", "recovery"];
  const bear = ["crash", "bear", "drop", "fell", "crisis", "ban", "hack", "fraud", "collapse", "plunge", "warn", "probe", "bankrupt", "selloff", "scam", "exploit", "crackdown"];
  const b = bull.filter((w) => t.includes(w)).length;
  const s = bear.filter((w) => t.includes(w)).length;
  return b > s ? "bullish" : s > b ? "bearish" : "neutral";
}

function formatAgo(hours: number): string {
  if (hours < 0) return `+${Math.abs(hours).toFixed(0)}h`;
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))}m ago`;
  if (hours < 24) return `${Math.floor(hours)}h ago`;
  if (hours < 24 * 7) return `${Math.floor(hours / 24)}d ago`;
  return `${Math.floor(hours / 168)}w ago`;
}

function truncate(text: string, max = 80): string {
  return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
}

function HeadlineRow({ item }: { item: NewsItem }) {
  const sentiment = item.sentiment ?? sentimentFromText(item.headline);
  const dotColor = SENTIMENT_COLOR[sentiment] ?? "#888";
  const sourceColor = SOURCE_COLORS[item.source] ?? "#666";

  const content = (
    <div
      className="rounded p-2.5 text-[11px] border-l-2 transition-opacity hover:opacity-90 animate-fade-in"
      style={{ background: `${dotColor}06`, borderColor: dotColor }}
    >
      <div className="flex items-center gap-2 mb-1">
        {/* Source badge — colored per source */}
        <span
          className="text-[9px] px-1.5 py-0.5 rounded font-bold tracking-wider"
          style={{ background: `${sourceColor}18`, color: sourceColor, border: `1px solid ${sourceColor}33` }}
        >
          {item.source}
        </span>
        {/* Sentiment dot */}
        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: dotColor }} title={sentiment} />
        {item.live && (
          <span className="text-[9px] px-1 py-0.5 rounded animate-pulse" style={{ background: "rgba(255,85,85,0.15)", color: "#ff5555" }}>
            LIVE
          </span>
        )}
        <span className="text-[10px] text-[#333] ml-auto tabular-nums">{formatAgo(item.hours_before)}</span>
      </div>
      <p className="text-[#999] leading-relaxed">{truncate(item.headline)}</p>
    </div>
  );

  if (item.url) {
    return (
      <a href={item.url} target="_blank" rel="noopener noreferrer" className="block">
        {content}
      </a>
    );
  }
  return content;
}

export default function NewsPanel({ asset, timestamp, liveMode = false }: Props) {
  const [localAsset, setLocalAsset] = useState(asset);
  const [headlines, setHeadlines] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastFetch, setLastFetch] = useState<Date | null>(null);
  const [fearGreed, setFearGreed] = useState<FearGreedIndex | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fear & Greed is a live, present-moment index — showing it next to a replayed
  // historical debate would silently leak today's market mood into that debate's
  // context, so it only ever fetches in live mode (crypto only — the index doesn't
  // cover forex/commodities).
  useEffect(() => {
    if (!liveMode || !FEAR_GREED_ASSETS.has(localAsset)) {
      setFearGreed(null);
      return;
    }
    api.getFearGreed().then(setFearGreed).catch(() => setFearGreed(null));
  }, [liveMode, localAsset]);

  // Follow the parent's asset by default, but let the in-panel tabs override it
  useEffect(() => setLocalAsset(asset), [asset]);

  const fetchLive = (assetId: string) => {
    setLoading(true);
    fetchApi<{ headlines: NewsItem[] }>(`/api/news/live?asset=${encodeURIComponent(assetId)}&max_items=40`)
      .then((d) => {
        setHeadlines(d.headlines);
        setLastFetch(new Date());
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const fetchHistorical = (assetId: string, ts: string) => {
    setLoading(true);
    fetchApi<{ headlines: NewsItem[] }>(`/api/news?asset=${encodeURIComponent(assetId)}&timestamp=${encodeURIComponent(ts)}`)
      .then((d) => setHeadlines(d.headlines))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    if (liveMode) {
      fetchLive(localAsset);
      intervalRef.current = setInterval(() => fetchLive(localAsset), 10 * 60 * 1000);
    } else if (timestamp) {
      fetchHistorical(localAsset, timestamp);
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [localAsset, liveMode, timestamp]);

  const bullish = headlines.filter((h) => (h.sentiment ?? sentimentFromText(h.headline)) === "bullish").length;
  const bearish = headlines.filter((h) => (h.sentiment ?? sentimentFromText(h.headline)) === "bearish").length;
  const neutral = headlines.length - bullish - bearish;
  const total = headlines.length;

  // Momentum: bullish rate in the more-recent half of the window vs the older half
  const momentum = useMemo(() => {
    if (headlines.length < 4) return null;
    const sorted = [...headlines].sort((a, b) => a.hours_before - b.hours_before);
    const mid = Math.floor(sorted.length / 2);
    const rate = (arr: NewsItem[]) =>
      arr.length ? arr.filter((h) => (h.sentiment ?? sentimentFromText(h.headline)) === "bullish").length / arr.length : 0;
    return rate(sorted.slice(0, mid)) >= rate(sorted.slice(mid)) ? "improving" : "declining";
  }, [headlines]);

  const empty = !liveMode && !timestamp;

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Asset filter tabs */}
      <div className="flex gap-0.5 px-2 pt-2 border-b border-[#1a1a1a] shrink-0 overflow-x-auto">
        {ASSET_TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setLocalAsset(t.id)}
            className="px-2.5 py-1.5 text-[10px] uppercase tracking-wider transition-colors whitespace-nowrap"
            style={{
              color: localAsset === t.id ? "#00ff7f" : "#555",
              borderBottom: localAsset === t.id ? "2px solid #00ff7f" : "2px solid transparent",
              fontWeight: localAsset === t.id ? 700 : 400,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {fearGreed && <FearGreedGauge data={fearGreed} />}

      {/* Header */}
      <div className="border-b border-[#1a1a1a] px-3 py-2 flex items-center gap-3 shrink-0">
        <span className="text-[10px] text-[#444] uppercase tracking-widest">
          {liveMode ? "Live Sentiment Feed" : "Sentiment Feed — 30d window"}
        </span>
        {liveMode && <span className="text-[10px] text-[#ff5555] animate-pulse">● RSS Live</span>}
        {momentum && (
          <span className="text-[10px] font-semibold" style={{ color: momentum === "improving" ? "#00ff7f" : "#ff3737" }}>
            {momentum === "improving" ? "↑ Improving" : "↓ Declining"}
          </span>
        )}
        {headlines.length > 0 && (
          <div className="ml-auto flex items-center gap-2 text-[10px]">
            <span className="text-[#00ff7f]">↑ {bullish}</span>
            <span className="text-[#ff3737]">↓ {bearish}</span>
            <span className="text-[#444]">{total}</span>
            {liveMode && (
              <button onClick={() => fetchLive(localAsset)} className="text-[#444] hover:text-[#00ff7f] transition-colors ml-1" title="Refresh now">
                ↻
              </button>
            )}
          </div>
        )}
      </div>

      {/* Sentiment summary bars — with percentages */}
      {headlines.length > 0 && (
        <div className="border-b border-[#1a1a1a] px-3 py-2 flex flex-col gap-1.5 shrink-0">
          {([
            ["Positive", bullish, "#00ff7f"],
            ["Negative", bearish, "#ff3737"],
            ["Neutral", neutral, "#555555"],
          ] as [string, number, string][]).map(([label, count, color]) => {
            const pct = total ? Math.round((count / total) * 100) : 0;
            return (
              <div key={label} className="flex items-center gap-2 text-[10px]">
                <span className="w-14 text-right shrink-0" style={{ color }}>{label}</span>
                <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: "#1a1a1a" }}>
                  <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color, transition: "width 0.5s ease" }} />
                </div>
                <span className="w-9 text-right text-[#444] tabular-nums">{pct}%</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="flex-1 overflow-auto p-3 min-h-0">
        {loading && headlines.length === 0 ? (
          <div className="flex items-center justify-center h-full gap-2 text-[#333] text-[11px]">
            <span className="animate-pulse">Fetching {liveMode ? "live RSS feeds" : "headlines"}…</span>
          </div>
        ) : empty ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center">
            <div className="text-[#333] text-[11px]">No timestamp selected</div>
            <div className="text-[10px] text-[#2a2a2a]">Switch to Live mode or select a chart range</div>
          </div>
        ) : headlines.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center">
            <div className="text-[#333] text-[11px]">
              {liveMode ? "No live headlines fetched yet — check network/RSS access" : "No headlines in this window"}
            </div>
            {!liveMode && <div className="text-[10px] text-[#2a2a2a]">News cache spans 2018–2023; try a date in that range</div>}
          </div>
        ) : (
          <div className="space-y-1.5">
            {headlines.map((h, i) => <HeadlineRow key={i} item={h} />)}
          </div>
        )}
      </div>

      <div className="border-t border-[#1a1a1a] px-3 py-1.5 shrink-0 flex items-center gap-2">
        <span className="text-[10px] text-[#2a2a2a] flex-1">
          {liveMode
            ? `CoinDesk · CoinTelegraph · Decrypt · Reuters${lastFetch ? ` · ${lastFetch.toLocaleTimeString()}` : ""}`
            : "FinBERT scores — SentimentAgent 24h window"}
        </span>
      </div>
    </div>
  );
}
