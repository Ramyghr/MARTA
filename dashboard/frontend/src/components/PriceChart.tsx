"use client";
import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  type MouseEventParams,
} from "lightweight-charts";
import type { Candle, DataRange } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { useMARTAStore } from "@/lib/store";
import { Maximize2 } from "lucide-react";

interface Props {
  candles: Candle[];
  asset: string;
  dataRange: DataRange | null;
  onRangeChange: (range: { start: string; end: string } | null) => void;
  onCandleClick?: (timestamp: string) => void;
  liveMode?: boolean;
}

type RangeKey = "1W" | "1M" | "3M" | "6M" | "1Y" | "ALL";

const RANGES: RangeKey[] = ["1W", "1M", "3M", "6M", "1Y", "ALL"];

// Fallback only, for the brief window before useMARTAStore's regimeNames is populated
// from the live backend (or if that fetch fails) — see the legend render below, which
// prefers regimeNames over this. Worded as volatility/trend, not bull/bear: a
// 2026-09-01 validation (regime/validate_regime_trend_alignment.py) found sharpe-
// derived bull/bear labels didn't reliably match actual price direction per occurrence.
const REGIME_SHORT: Record<number, string> = {
  0: "Mixed",
  1: "Low Vol/Trending",
  2: "Extreme Vol/Shock",
  3: "Elevated Vol/Trending",
  4: "Low Vol/Range-bound",
  5: "FX-A",
  6: "FX-B",
};

function rangeToMs(key: RangeKey): number {
  const h = 3600_000;
  const d = 24 * h;
  switch (key) {
    case "1W": return 7 * d;
    case "1M": return 30 * d;
    case "3M": return 90 * d;
    case "6M": return 180 * d;
    case "1Y": return 365 * d;
    default: return Infinity;
  }
}

function toUTCTimestamp(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

function formatPrice(v: number, asset: string): string {
  if (asset === "EURUSD") return v.toFixed(4);
  if (asset === "GOLD") return `$${v.toFixed(0)}`;
  if (v >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(2)}`;
}

// One contiguous run of same regime_label — the unit the background bands are drawn
// from (many candles collapse into one band, not one rect per candle).
interface RegimeRun {
  startSec: number;
  endSec: number;
  regime: number;
}

// Screen-space projection of a RegimeRun, recomputed on every pan/zoom/resize so the
// band always matches whatever time window the chart is currently showing — this is
// the fix for the reported bug (bands staying put while the candles scrolled under them).
interface RegimeBand {
  left: number;
  width: number;
  color: string;
}

export default function PriceChart({ candles, asset, dataRange, onRangeChange, onCandleClick, liveMode = false }: Props) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const [activeRange, setActiveRange] = useState<RangeKey>("ALL");
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");

  // EURUSD/GOLD historical data ends 2024-12-31 — Twelve Data live ticks extend the
  // chart forward from today. Unlike BTC/ETH (continuous exchange coverage, safe to
  // just mirror the incoming `candles` prop), a live tick here must be appended to
  // the existing array rather than treated as a fresh dataset, or the gap-bridging
  // candle gets discarded every time a new tick arrives.
  const [liveCandles, setLiveCandles] = useState<Candle[]>(candles);
  const prevPropRef = useRef<Candle[]>(candles);
  const prevAssetRef = useRef(asset);
  const historicalEndRef = useRef<string | null>(null);

  useEffect(() => {
    const prevProp = prevPropRef.current;
    prevPropRef.current = candles;
    const assetChanged = prevAssetRef.current !== asset;
    prevAssetRef.current = asset;

    // Crypto-only panel (2026-08 migration): every asset now has a live-bridge
    // boundary (dashboard/backend/live_feature_bridge.py extends all 5 crypto5
    // assets the same way EURUSD/GOLD were bridged before), so this is no longer
    // asset-specific — always append-and-bridge rather than treat a live tick as a
    // fresh dataset.
    const isBridgedAsset = true;
    const datasetSwapped =
      assetChanged ||
      candles.length === 0 ||
      prevProp.length === 0 ||
      candles.length < prevProp.length ||
      candles[0]?.timestamp !== prevProp[0]?.timestamp;

    if (!isBridgedAsset || datasetSwapped) {
      historicalEndRef.current = candles.length ? candles[candles.length - 1].timestamp : null;
      setLiveCandles(candles);
      return;
    }

    // Same dataset, grown by one or more new WS ticks — append only the genuinely
    // new candles, and only those newer than the last historical candle.
    const boundary = historicalEndRef.current;
    const newTail = candles.slice(prevProp.length);
    if (!boundary || newTail.length === 0) return;

    setLiveCandles((prev) => {
      let next = prev;
      for (const c of newTail) {
        if (new Date(c.timestamp).getTime() <= new Date(boundary).getTime()) continue;
        const last = next[next.length - 1];
        if (last && last.timestamp === c.timestamp) continue;
        next = [...next, c];
      }
      return next;
    });
  }, [candles, asset]);

  const handleRangeClick = useCallback((key: RangeKey) => {
    setActiveRange(key);
    if (!dataRange) return;

    if (key === "ALL") {
      onRangeChange(null);
      return;
    }

    const endMs = new Date(dataRange.end).getTime();
    const startMs = endMs - rangeToMs(key);
    const start = new Date(Math.max(startMs, new Date(dataRange.start).getTime())).toISOString();
    const end = dataRange.end;
    onRangeChange({ start, end });
  }, [dataRange, onRangeChange]);

  const handleCustomApply = useCallback(() => {
    if (!customStart || !customEnd) return;
    onRangeChange({ start: new Date(customStart).toISOString(), end: new Date(customEnd).toISOString() });
  }, [customStart, customEnd, onRangeChange]);

  // Memoized: the data-push useEffect below depends on this array's identity, and
  // an unmemoized .slice() (a fresh array every render) previously caused an infinite
  // render loop (setBands inside the effect -> re-render -> new displayCandles
  // reference -> effect fires again). Recharts never had a useEffect keyed on this,
  // so the same unmemoized .slice() didn't loop there — introduced by this rewrite.
  const displayCandles = useMemo(() => liveCandles.slice(-2000), [liveCandles]);

  const latestClose = displayCandles.at(-1)?.close;
  const prevClose = displayCandles.at(-2)?.close;
  const pctChange = latestClose && prevClose ? ((latestClose - prevClose) / prevClose) * 100 : 0;
  const fp = useCallback((v: number) => formatPrice(v, asset), [asset]);

  // 24h high/low from visible candles
  const last24 = displayCandles.slice(-24);
  const h24High = last24.length ? Math.max(...last24.map((c) => c.high)) : null;
  const h24Low  = last24.length ? Math.min(...last24.map((c) => c.low))  : null;

  const hasVolume = displayCandles.some((c) => c.volume > 0);

  // Contiguous regime runs — many candles collapse into one band. Recomputed only
  // when the underlying data changes, NOT on every pan/zoom (that part is just a
  // cheap re-projection to pixels, done separately below).
  const regimeRuns = useMemo<RegimeRun[]>(() => {
    if (!displayCandles.length) return [];
    const runs: RegimeRun[] = [];
    const barSec = displayCandles.length > 1
      ? toUTCTimestamp(displayCandles[1].timestamp) - toUTCTimestamp(displayCandles[0].timestamp)
      : 3600;
    let cur: RegimeRun | null = null;
    for (const c of displayCandles) {
      const t = toUTCTimestamp(c.timestamp);
      if (cur && c.regime_label === cur.regime) {
        cur.endSec = t + barSec;
      } else {
        if (cur) runs.push(cur);
        cur = { startSec: t, endSec: t + barSec, regime: c.regime_label };
      }
    }
    if (cur) runs.push(cur);
    return runs;
  }, [displayCandles]);

  // Marks where historical data hands off to live-bridged ticks, so the transition
  // reads as a gap, not a bar. All 5 crypto5 assets are bridged the same way
  // EURUSD/GOLD were (see live_feature_bridge.py) — no longer asset-specific.
  const liveBridgeBoundarySec = useMemo(() => {
    const boundary = historicalEndRef.current;
    const last = displayCandles.at(-1);
    if (!boundary || !last || last.timestamp === boundary) return null;
    return toUTCTimestamp(boundary);
  }, [displayCandles]);

  // ---------------------------------------------------------------- chart lifecycle
  // wrapperRef is the sizing SOURCE OF TRUTH (flex-sized by CSS, never touched by the
  // chart library). containerRef is what lightweight-charts mounts into — resize()
  // sets an explicit inline width/height directly on that element, which fights its
  // own position:absolute;inset:0 sizing if we ever measure IT instead of its parent
  // (confirmed 2026-09-03: containerRef's rect gets permanently pinned to whatever the
  // very first, pre-layout-settle resize() call used, since every later measurement of
  // the same element just reads back that self-inflicted inline style). Always measure
  // wrapperRef, never containerRef, when deciding a new chart size.
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const candleByTimeRef = useRef<Map<number, Candle>>(new Map());
  const regimeRunsRef = useRef<RegimeRun[]>([]);
  const liveBoundarySecRef = useRef<number | null>(null);
  const onCandleClickRef = useRef(onCandleClick);
  onCandleClickRef.current = onCandleClick;

  const [bands, setBands] = useState<RegimeBand[]>([]);
  const [liveBoundaryX, setLiveBoundaryX] = useState<number | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ candle: Candle; x: number; y: number } | null>(null);

  // Re-project regime runs (and the live-bridge marker) to screen-space pixels using
  // the chart's OWN time-scale coordinate conversion. Subscribed to every visible-
  // range change (pan and zoom both fire this), so the bands always match whatever
  // window is currently on screen instead of staying pinned to their original
  // position while candles scroll underneath — that mismatch was the reported bug.
  const recomputeOverlays = useCallback(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const next: RegimeBand[] = [];
    for (const run of regimeRunsRef.current) {
      const x1 = ts.timeToCoordinate(run.startSec as UTCTimestamp);
      const x2 = ts.timeToCoordinate(run.endSec as UTCTimestamp);
      if (x1 === null && x2 === null) continue;
      const left = x1 ?? -50;
      const right = x2 ?? (containerRef.current?.clientWidth ?? 0) + 50;
      if (right <= 0 || left >= (containerRef.current?.clientWidth ?? 0)) continue;
      next.push({ left, width: Math.max(right - left, 0.5), color: REGIME_COLORS[run.regime] ?? "#444" });
    }
    setBands(next);

    if (liveBoundarySecRef.current !== null) {
      const x = ts.timeToCoordinate(liveBoundarySecRef.current as UTCTimestamp);
      setLiveBoundaryX(x);
    } else {
      setLiveBoundaryX(null);
    }
  }, []);

  // Create the chart once. Data, options and callbacks are pushed into it via refs
  // and separate effects below rather than recreating it on every prop change.
  useEffect(() => {
    if (!containerRef.current || !wrapperRef.current) return;
    // autoSize:true measured this container before the surrounding CSS grid finished
    // its layout pass, got a stale narrow width, and its internal ResizeObserver never
    // re-fired on the later real resize (confirmed 2026-09-03). Sizing the chart
    // explicitly off wrapperRef's measured rect — both here and in the ResizeObserver
    // below — sidesteps whatever autoSize's internal observer was missing.
    const initialRect = wrapperRef.current.getBoundingClientRect();
    const chart = createChart(containerRef.current, {
      width: Math.round(initialRect.width),
      height: Math.round(initialRect.height),
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#555", fontSize: 10 },
      grid: { vertLines: { visible: false }, horzLines: { color: "#141414" } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#4a9eff88", width: 1, style: LineStyle.Dashed, labelBackgroundColor: "#111" },
        horzLine: { color: "#4a9eff88", width: 1, style: LineStyle.Dashed, labelBackgroundColor: "#111" },
      },
      rightPriceScale: { borderColor: "#1a1a1a" },
      timeScale: { borderColor: "#1a1a1a", timeVisible: true, secondsVisible: false, rightOffset: 4 },
      autoSize: false,
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00ff7f", downColor: "#ff3737", borderVisible: false,
      wickUpColor: "#00ff7f", wickDownColor: "#ff3737",
      priceLineVisible: true, priceLineColor: "#2a2a2a", priceLineStyle: LineStyle.Dashed,
      lastValueVisible: true,
    }, 0);

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    }, 1);
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0 } });
    const panes = chart.panes();
    if (panes[1]) panes[1].setStretchFactor(0.18);
    if (panes[0]) panes[0].setStretchFactor(1);

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    const handleClick = (param: MouseEventParams) => {
      if (!param.time) return;
      const c = candleByTimeRef.current.get(param.time as number);
      if (c && onCandleClickRef.current) onCandleClickRef.current(c.timestamp);
    };
    const handleCrosshair = (param: MouseEventParams) => {
      if (!param.point || !param.time) { setHoverInfo(null); return; }
      const c = candleByTimeRef.current.get(param.time as number);
      setHoverInfo(c ? { candle: c, x: param.point.x, y: param.point.y } : null);
    };
    chart.subscribeClick(handleClick);
    chart.subscribeCrosshairMove(handleCrosshair);
    chart.timeScale().subscribeVisibleLogicalRangeChange(recomputeOverlays);

    // Measure wrapperRef (never mutated by the chart library), not containerRef
    // (resize() sets an explicit inline width/height directly on it — measuring that
    // same element back would just read our own last write, permanently pinning it to
    // whatever the very first, pre-layout-settle call used).
    let lastAppliedW = Math.round(initialRect.width);
    let lastAppliedH = Math.round(initialRect.height);
    const applySize = () => {
      if (!wrapperRef.current) return;
      const rect = wrapperRef.current.getBoundingClientRect();
      const w = Math.round(rect.width);
      const h = Math.round(rect.height);
      if (w > 0 && h > 0 && (w !== lastAppliedW || h !== lastAppliedH)) {
        chart.resize(w, h);
        lastAppliedW = w;
        lastAppliedH = h;
      }
      recomputeOverlays();
    };
    const resizeObserver = new ResizeObserver(applySize);
    resizeObserver.observe(wrapperRef.current);
    // Belt-and-suspenders, not just timing: found 2026-09-03 that ResizeObserver can
    // fail to fire a follow-up callback here even though wrapperRef's true rect (via a
    // fresh getBoundingClientRect() from outside React) had already grown past its
    // very-first-paint size — reproduced across reload, Fast Refresh, and a hard
    // client-side unmount/remount, so it isn't just a one-off race. Root cause not
    // pinned down (candidate: CSS Grid minmax(0, Nfr) row-track sizing settling after
    // a sibling panel's async content changes the row height, which may not always
    // notify a ResizeObserver on an absolutely-positioned descendant the same way a
    // normal-flow resize would). Polling for a few seconds after mount is a robust
    // fallback that converges regardless of the exact mechanism.
    const pollUntil = Date.now() + 4000;
    const pollInterval = setInterval(() => {
      applySize();
      if (Date.now() > pollUntil) clearInterval(pollInterval);
    }, 200);

    return () => {
      clearInterval(pollInterval);
      resizeObserver.disconnect();
      chart.unsubscribeClick(handleClick);
      chart.unsubscribeCrosshairMove(handleCrosshair);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(recomputeOverlays);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push fresh candle/volume data whenever the visible dataset changes.
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    if (!candleSeries || !volumeSeries) return;

    const byTime = new Map<number, Candle>();
    const candleData = displayCandles.map((c) => {
      const t = toUTCTimestamp(c.timestamp);
      byTime.set(t, c);
      return { time: t as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close };
    });
    candleByTimeRef.current = byTime;
    candleSeries.setData(candleData);

    volumeSeries.setData(
      displayCandles.map((c) => ({
        time: toUTCTimestamp(c.timestamp) as UTCTimestamp,
        value: c.volume,
        color: c.close >= c.open ? "rgba(0,255,127,0.45)" : "rgba(255,55,55,0.45)",
      }))
    );

    regimeRunsRef.current = regimeRuns;
    liveBoundarySecRef.current = liveBridgeBoundarySec;
    recomputeOverlays();

    // Collapse the volume pane to ~0 when this asset has no volume data (FX/commodity
    // histdata feeds have no centralised volume — see CLAUDE.md) instead of showing an
    // empty pane with axis chrome and no bars.
    const panes = chartRef.current?.panes();
    panes?.[1]?.setStretchFactor(hasVolume ? 0.18 : 0.0001);
  }, [displayCandles, regimeRuns, liveBridgeBoundarySec, recomputeOverlays, hasVolume]);

  // Y-axis autoscales to whatever's visible natively (lightweight-charts recomputes
  // it on every pan/zoom already) — no manual zoom/pan state needed here anymore.
  // "Fit" resets both axes to show the whole loaded dataset.
  const handleFit = useCallback(() => {
    chartRef.current?.timeScale().fitContent();
  }, []);

  return (
    <div className="panel h-full flex flex-col">
      {/* Header */}
      <div className="panel-header flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <span>{asset.replace("_", "/")}</span>
          {latestClose && (
            <>
              <span className="text-[#e0e0e0] font-bold text-base" style={{ letterSpacing: "-0.02em" }}>
                {fp(latestClose)}
              </span>
              <span className="text-sm font-semibold" style={{ color: pctChange >= 0 ? "#00ff7f" : "#ff3737" }}>
                {pctChange >= 0 ? "↑" : "↓"} {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
              </span>
              {h24High !== null && h24Low !== null && (
                <span className="text-[10px] text-[#444] hidden lg:inline">
                  H&nbsp;<span className="text-[#00ff7f99]">{fp(h24High)}</span>
                  &nbsp;&nbsp;L&nbsp;<span className="text-[#ff373799]">{fp(h24Low)}</span>
                </span>
              )}
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {liveMode ? (
            <span className="text-[10px] text-[#ff5555] animate-pulse">● LIVE</span>
          ) : (
            <>
              {/* Custom date range scrubber */}
              <div className="hidden xl:flex items-center gap-1">
                <input
                  type="datetime-local"
                  value={customStart}
                  min={dataRange?.start.slice(0, 16)}
                  max={dataRange?.end.slice(0, 16)}
                  onChange={(e) => setCustomStart(e.target.value)}
                  className="bg-[#111] border border-[#1a1a1a] rounded px-1.5 py-0.5 text-[9px] text-[#666] outline-none"
                />
                <span className="text-[#333] text-[10px]">→</span>
                <input
                  type="datetime-local"
                  value={customEnd}
                  min={dataRange?.start.slice(0, 16)}
                  max={dataRange?.end.slice(0, 16)}
                  onChange={(e) => setCustomEnd(e.target.value)}
                  className="bg-[#111] border border-[#1a1a1a] rounded px-1.5 py-0.5 text-[9px] text-[#666] outline-none"
                />
                <button
                  onClick={handleCustomApply}
                  disabled={!customStart || !customEnd}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-[#4a9eff44] text-[#4a9eff] disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Go
                </button>
              </div>
              <div className="flex rounded overflow-hidden border border-[#1a1a1a]">
                {RANGES.map((r) => (
                  <button
                    key={r}
                    onClick={() => handleRangeClick(r)}
                    className="px-2 py-0.5 text-[10px] transition-colors"
                    style={{
                      background: activeRange === r ? "rgba(0,255,127,0.15)" : "transparent",
                      color: activeRange === r ? "#00ff7f" : "#444",
                      borderRight: r !== "ALL" ? "1px solid #1a1a1a" : "none",
                    }}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Chart — candles + volume in one lightweight-charts instance (two panes),
          so the time scale, crosshair and pan/zoom are natively shared instead of
          synced by hand across two separate chart libraries. */}
      <div className="flex-1 min-h-0 flex flex-col px-2 pb-1 pt-0">
        <div ref={wrapperRef} style={{ flex: "1 1 0", minHeight: 0, position: "relative" }}>
          {/* Regime background bands — projected fresh from the chart's own time
              scale on every pan/zoom/resize (recomputeOverlays), so they always
              match the currently-visible window instead of staying pinned to
              wherever they were first drawn. */}
          <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
            {bands.map((b, i) => (
              <div
                key={i}
                style={{
                  position: "absolute", left: b.left, top: 0, width: b.width, height: "100%",
                  background: b.color, opacity: 0.09,
                }}
              />
            ))}
            {liveBoundaryX !== null && (
              <div
                style={{
                  position: "absolute", left: liveBoundaryX, top: 0, bottom: 0, width: 1.5,
                  background: "#4a9eff", opacity: 0.7,
                }}
              >
                <span
                  style={{
                    position: "absolute", top: 2, left: 4, fontSize: 9, fontWeight: 700,
                    color: "#4a9eff", whiteSpace: "nowrap",
                  }}
                >
                  live ›
                </span>
              </div>
            )}
          </div>

          {/* lightweight-charts mounts its canvas(es) into this element */}
          <div ref={containerRef} style={{ position: "absolute", inset: 0, zIndex: 1 }} />

          {/* Fit-to-view control — replaces the old manual Y zoom/pan buttons now
              that the price axis autoscales natively to whatever's panned/zoomed
              into view. */}
          <button
            onClick={handleFit}
            className="absolute top-1 right-16 z-10 p-1 rounded border border-[#1a1a1a] bg-[#0a0a0acc] text-[#666] hover:text-[#00ff7f] hover:bg-[#00ff7f11] transition-colors"
            title="Fit all loaded data"
          >
            <Maximize2 size={12} />
          </button>

          {/* Custom tooltip, positioned from the crosshair-move handler */}
          {hoverInfo && (
            <div
              className="absolute z-20 bg-[#111] border border-[#333] rounded p-2 text-[11px] font-mono pointer-events-none"
              style={{
                left: Math.min(hoverInfo.x + 12, (containerRef.current?.clientWidth ?? 400) - 150),
                top: 8,
              }}
            >
              <div className="text-[#666] mb-1">{new Date(hoverInfo.candle.timestamp).toLocaleString()}</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                <span className="text-[#666]">O</span><span>{fp(hoverInfo.candle.open)}</span>
                <span className="text-[#666]">H</span><span className="text-[#00ff7f]">{fp(hoverInfo.candle.high)}</span>
                <span className="text-[#666]">L</span><span className="text-[#ff3737]">{fp(hoverInfo.candle.low)}</span>
                <span className="text-[#666]">C</span>
                <span style={{ color: hoverInfo.candle.close >= hoverInfo.candle.open ? "#00ff7f" : "#ff3737" }}>
                  {fp(hoverInfo.candle.close)}
                </span>
                {hoverInfo.candle.volume > 0 && (
                  <>
                    <span className="text-[#666]">Vol</span>
                    <span className="text-[#555]">{hoverInfo.candle.volume.toLocaleString("en", { maximumFractionDigits: 0 })}</span>
                  </>
                )}
              </div>
              {hoverInfo.candle.regime_description && (
                <div className="mt-1 text-[#444]">R{hoverInfo.candle.regime_label} {hoverInfo.candle.regime_description}</div>
              )}
            </div>
          )}

        </div>
      </div>

      {/* Regime legend with full names */}
      <div className="border-t border-[#1a1a1a] px-3 py-1.5 flex gap-3 flex-wrap items-center shrink-0">
        <span className="text-[10px] text-[#333]">Regime:</span>
        {Object.entries(REGIME_COLORS).map(([k, color]) => (
          <span key={k} className="flex items-center gap-1 text-[10px]" style={{ color: "#555" }}>
            <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: color }} />
            <span>R{k} <span style={{ color }}>{regimeNames[parseInt(k)] ?? REGIME_SHORT[parseInt(k)]}</span></span>
          </span>
        ))}
        {displayCandles.length > 0 && (
          <span className="ml-auto text-[10px] text-[#333]">
            {new Date(displayCandles[0].timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
            {" — "}
            {new Date(displayCandles.at(-1)!.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
          </span>
        )}
      </div>
    </div>
  );
}
