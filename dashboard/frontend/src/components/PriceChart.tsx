"use client";
import { useMemo, useState, useCallback, useEffect, useRef, type WheelEvent } from "react";
import {
  ComposedChart,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  Cell,
} from "recharts";
import type { Candle, DataRange } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { ZoomIn, ZoomOut, ChevronUp, ChevronDown, Maximize2 } from "lucide-react";

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

const REGIME_SHORT: Record<number, string> = {
  0: "Mixed",
  1: "Calm Bull",
  2: "Crisis",
  3: "Bear",
  4: "Calm Bear",
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

function CandleShape(props: Record<string, unknown>) {
  const { x, y, width, height, open, close, regime_label } = props as {
    x: number; y: number; width: number; height: number;
    open: number; close: number; regime_label: number;
  };
  if (!x || !width) return null;
  const isGreen = close >= open;
  const bodyColor = isGreen ? "#00ff7f" : "#ff3737";
  const cx = (x as number) + (width as number) / 2;
  return (
    <g>
      <rect
        x={x as number} y={-9999} width={width as number} height={99999}
        fill={REGIME_COLORS[regime_label ?? 0] ?? "#444"} fillOpacity={0.08}
      />
      <rect
        x={(x as number) + (width as number) * 0.15}
        y={y as number}
        width={(width as number) * 0.7}
        height={Math.max(height as number, 1)}
        fill={bodyColor}
        fillOpacity={0.85}
      />
      <line x1={cx} y1={y as number} x2={cx} y2={(y as number) - 4} stroke={bodyColor} strokeWidth={1} strokeOpacity={1.0} />
      <line x1={cx} y1={(y as number) + (height as number)} x2={cx} y2={(y as number) + (height as number) + 4} stroke={bodyColor} strokeWidth={1} strokeOpacity={1.0} />
    </g>
  );
}

function formatTimestamp(ts: string, count: number): string {
  try {
    const d = new Date(ts);
    if (count > 800) return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
    if (count > 200) return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch { return ts; }
}

function formatPrice(v: number, asset: string): string {
  if (asset === "EURUSD") return v.toFixed(4);
  if (asset === "GOLD") return `$${v.toFixed(0)}`;
  if (v >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(2)}`;
}

const CustomTooltip = ({ active, payload, asset }: { active?: boolean; payload?: Array<{ payload: Candle }>; asset: string }) => {
  if (!active || !payload?.length) return null;
  const c = payload[0].payload;
  const isGreen = c.close >= c.open;
  const fp = (v: number) => formatPrice(v, asset);
  return (
    <div className="bg-[#111] border border-[#333] rounded p-2 text-[11px] font-mono z-50">
      <div className="text-[#666] mb-1">{new Date(c.timestamp).toLocaleString()}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="text-[#666]">O</span><span>{fp(c.open)}</span>
        <span className="text-[#666]">H</span><span className="text-[#00ff7f]">{fp(c.high)}</span>
        <span className="text-[#666]">L</span><span className="text-[#ff3737]">{fp(c.low)}</span>
        <span className="text-[#666]">C</span><span style={{ color: isGreen ? "#00ff7f" : "#ff3737" }}>{fp(c.close)}</span>
        {c.volume > 0 && (
          <><span className="text-[#666]">Vol</span><span className="text-[#555]">{c.volume.toLocaleString("en", { maximumFractionDigits: 0 })}</span></>
        )}
      </div>
      {c.regime_description && (
        <div className="mt-1 text-[#444]">R{c.regime_label} {c.regime_description}</div>
      )}
    </div>
  );
};

export default function PriceChart({ candles, asset, dataRange, onRangeChange, onCandleClick, liveMode = false }: Props) {
  const [activeRange, setActiveRange] = useState<RangeKey>("ALL");
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [hover, setHover] = useState<{ ts: string; price: number } | null>(null);

  // Y-axis zoom/pan — mirrors the X-axis range controls so price scale isn't stuck
  // on auto-fit. yZoom=1 is fit-to-visible-data; >1 zooms in, <1 zooms out. yPanOffset
  // shifts the visible center up/down in price units. Both reset on asset switch since
  // a stale zoom/offset from e.g. BTC's ~$40k scale is meaningless for EURUSD's ~1.1.
  const [yZoom, setYZoom] = useState(1);
  const [yPanOffset, setYPanOffset] = useState(0);
  useEffect(() => { setYZoom(1); setYPanOffset(0); }, [asset]);

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

  const handleChartHover = useCallback((state: { activePayload?: Array<{ payload: Candle }> }) => {
    const p = state?.activePayload?.[0]?.payload;
    if (p) setHover({ ts: p.timestamp, price: p.close });
  }, []);

  const handleChartLeave = useCallback(() => setHover(null), []);

  const displayCandles = liveCandles.slice(-2000);

  const { minPrice: autoMinPrice, maxPrice: autoMaxPrice } = useMemo(() => {
    if (!displayCandles.length) return { minPrice: 0, maxPrice: 1 };
    const lows = displayCandles.map((c) => c.low);
    const highs = displayCandles.map((c) => c.high);
    const min = Math.min(...lows);
    const max = Math.max(...highs);
    const pad = (max - min) * 0.04;
    return { minPrice: min - pad, maxPrice: max + pad };
  }, [displayCandles]);

  // Apply user Y zoom/pan on top of the auto-fit range computed above.
  const { minPrice, maxPrice } = useMemo(() => {
    const autoHalf = (autoMaxPrice - autoMinPrice) / 2 || 1;
    const center = (autoMinPrice + autoMaxPrice) / 2 + yPanOffset;
    const half = autoHalf / yZoom;
    return { minPrice: center - half, maxPrice: center + half };
  }, [autoMinPrice, autoMaxPrice, yZoom, yPanOffset]);

  const zoomIn = useCallback(() => setYZoom((z) => Math.min(20, z * 1.35)), []);
  const zoomOut = useCallback(() => setYZoom((z) => Math.max(0.2, z / 1.35)), []);
  const panUp = useCallback(() => {
    const half = ((autoMaxPrice - autoMinPrice) / 2 || 1) / yZoom;
    setYPanOffset((o) => o + half * 0.3);
  }, [autoMinPrice, autoMaxPrice, yZoom]);
  const panDown = useCallback(() => {
    const half = ((autoMaxPrice - autoMinPrice) / 2 || 1) / yZoom;
    setYPanOffset((o) => o - half * 0.3);
  }, [autoMinPrice, autoMaxPrice, yZoom]);
  const resetYAxis = useCallback(() => { setYZoom(1); setYPanOffset(0); }, []);
  const isYAutoFit = yZoom === 1 && yPanOffset === 0;

  const handleChartWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) zoomIn(); else zoomOut();
  }, [zoomIn, zoomOut]);

  const chartData = displayCandles.map((c) => ({
    ...c,
    body: [Math.min(c.open, c.close), Math.max(c.open, c.close)],
  }));

  const latestClose = displayCandles.at(-1)?.close;
  const prevClose = displayCandles.at(-2)?.close;
  const pctChange = latestClose && prevClose ? ((latestClose - prevClose) / prevClose) * 100 : 0;
  const fp = (v: number) => formatPrice(v, asset);

  // 24h high/low from visible candles
  const last24 = displayCandles.slice(-24);
  const h24High = last24.length ? Math.max(...last24.map((c) => c.high)) : null;
  const h24Low  = last24.length ? Math.min(...last24.map((c) => c.low))  : null;

  const hasVolume = displayCandles.some((c) => c.volume > 0);

  const tickInterval = Math.max(1, Math.floor(displayCandles.length / 6));

  // Regime change boundaries for reference lines
  const regimeBoundaries = useMemo(() => {
    const bounds: Array<{ x: string; regime: number }> = [];
    for (let i = 1; i < displayCandles.length; i++) {
      if (displayCandles[i].regime_label !== displayCandles[i - 1].regime_label) {
        bounds.push({ x: displayCandles[i].timestamp, regime: displayCandles[i].regime_label });
      }
    }
    return bounds.slice(-20);
  }, [displayCandles]);

  // Marks where historical data hands off to live-bridged ticks, so the transition
  // reads as a gap, not a bar. All 5 crypto5 assets are bridged the same way
  // EURUSD/GOLD were (see live_feature_bridge.py) — no longer asset-specific.
  const liveBridgeBoundary = useMemo(() => {
    const boundary = historicalEndRef.current;
    const last = displayCandles.at(-1);
    if (!boundary || !last || last.timestamp === boundary) return null;
    return boundary;
  }, [asset, displayCandles]);

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

      {/* Chart + Volume */}
      <div className="flex-1 min-h-0 flex flex-col px-2 pb-1 pt-0">
        {/* Price chart */}
        <div style={{ flex: "1 1 0", minHeight: 0, position: "relative" }} onWheel={handleChartWheel}>
          {/* Y-axis zoom/pan controls — mirrors the X-axis range buttons above so price
              scale has the same manual override as time range. Scroll on the chart also
              zooms the Y axis. */}
          <div
            className="absolute top-1 right-16 z-10 flex flex-col rounded border border-[#1a1a1a] bg-[#0a0a0acc] overflow-hidden"
            title="Scroll on the chart to zoom the price axis"
          >
            <button onClick={zoomIn} className="p-1 text-[#666] hover:text-[#4a9eff] hover:bg-[#4a9eff11] transition-colors" title="Zoom in (Y axis)">
              <ZoomIn size={12} />
            </button>
            <button onClick={zoomOut} className="p-1 text-[#666] hover:text-[#4a9eff] hover:bg-[#4a9eff11] transition-colors border-t border-[#1a1a1a]" title="Zoom out (Y axis)">
              <ZoomOut size={12} />
            </button>
            <button onClick={panUp} className="p-1 text-[#666] hover:text-[#4a9eff] hover:bg-[#4a9eff11] transition-colors border-t border-[#1a1a1a]" title="Pan up">
              <ChevronUp size={12} />
            </button>
            <button onClick={panDown} className="p-1 text-[#666] hover:text-[#4a9eff] hover:bg-[#4a9eff11] transition-colors border-t border-[#1a1a1a]" title="Pan down">
              <ChevronDown size={12} />
            </button>
            <button
              onClick={resetYAxis}
              disabled={isYAutoFit}
              className="p-1 text-[#666] hover:text-[#00ff7f] hover:bg-[#00ff7f11] transition-colors border-t border-[#1a1a1a] disabled:opacity-25 disabled:cursor-not-allowed"
              title="Reset to auto-fit"
            >
              <Maximize2 size={12} />
            </button>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              syncId="price-volume"
              margin={{ top: 4, right: 8, bottom: 4, left: 4 }}
              onClick={(ev) => {
                const ts = (ev?.activePayload?.[0]?.payload as Candle | undefined)?.timestamp;
                if (ts && onCandleClick) onCandleClick(ts);
              }}
              onMouseMove={handleChartHover}
              onMouseLeave={handleChartLeave}
              style={{ cursor: onCandleClick ? "crosshair" : "default" }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => formatTimestamp(ts, displayCandles.length)}
                tick={{ fill: "#444", fontSize: 10 }}
                axisLine={{ stroke: "#1a1a1a" }}
                tickLine={false}
                interval={tickInterval}
              />
              <YAxis
                domain={[minPrice, maxPrice]}
                tickFormatter={fp}
                tick={{ fill: "#444", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={64}
                orientation="right"
              />
              <Tooltip content={<CustomTooltip asset={asset} />} cursor={{ stroke: "#4a9eff", strokeWidth: 1, strokeDasharray: "3 3" }} />
              {latestClose && (
                <ReferenceLine y={latestClose} stroke="#2a2a2a" strokeDasharray="4 4" strokeWidth={1} />
              )}
              {liveBridgeBoundary && (
                <ReferenceLine
                  x={liveBridgeBoundary}
                  stroke="#4a9eff"
                  strokeWidth={1.5}
                  strokeDasharray="2 3"
                  label={{ value: "live ›", position: "insideTopRight", fill: "#4a9eff", fontSize: 9, fontWeight: 700 }}
                />
              )}
              {regimeBoundaries.map((b, i) => (
                <ReferenceLine
                  key={i}
                  x={b.x}
                  stroke={REGIME_COLORS[b.regime] ?? "#333"}
                  strokeWidth={1}
                  strokeOpacity={0.4}
                  strokeDasharray="2 4"
                />
              ))}
              {hover && (
                <>
                  <ReferenceLine
                    y={hover.price}
                    stroke="#4a9eff"
                    strokeWidth={1}
                    strokeOpacity={0.6}
                    label={{ value: fp(hover.price), position: "right", fill: "#4a9eff", fontSize: 10, fontWeight: 700 }}
                  />
                  <ReferenceLine
                    x={hover.ts}
                    stroke="#4a9eff"
                    strokeWidth={1}
                    strokeOpacity={0.35}
                    label={{ value: new Date(hover.ts).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit" }), position: "insideBottomLeft", fill: "#4a9eff", fontSize: 9 }}
                  />
                </>
              )}
              <Bar
                dataKey="body"
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                shape={(props: any) => {
                  const idx = props.index as number;
                  const c = chartData[idx];
                  return <CandleShape {...props} open={c?.open} close={c?.close} regime_label={c?.regime_label} />;
                }}
                isAnimationActive={false}
              />
              <Brush
                dataKey="timestamp"
                height={22}
                stroke="#1a1a1a"
                fill="#080808"
                travellerWidth={6}
                tickFormatter={(ts: string) => {
                  try { return new Date(ts).toLocaleDateString("en-US", { month: "short", year: "2-digit" }); }
                  catch { return ""; }
                }}
                startIndex={Math.max(0, chartData.length - Math.floor(chartData.length * 0.4))}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Volume chart */}
        {hasVolume && (
          <div style={{ height: 38, flexShrink: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} syncId="price-volume" margin={{ top: 0, right: 8, bottom: 0, left: 4 }}>
                <XAxis dataKey="timestamp" hide />
                <YAxis hide domain={[0, "auto"]} />
                <Tooltip content={() => null} cursor={{ stroke: "#4a9eff", strokeWidth: 1, strokeDasharray: "3 3" }} />
                <Bar dataKey="volume" isAnimationActive={false} maxBarSize={8}>
                  {chartData.map((c, i) => (
                    <Cell
                      key={i}
                      fill={c.close >= c.open ? "rgba(0,255,127,0.45)" : "rgba(255,55,55,0.45)"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Regime legend with full names */}
      <div className="border-t border-[#1a1a1a] px-3 py-1.5 flex gap-3 flex-wrap items-center shrink-0">
        <span className="text-[10px] text-[#333]">Regime:</span>
        {Object.entries(REGIME_COLORS).map(([k, color]) => (
          <span key={k} className="flex items-center gap-1 text-[10px]" style={{ color: "#555" }}>
            <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: color }} />
            <span>R{k} <span style={{ color }}>{REGIME_SHORT[parseInt(k)]}</span></span>
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
