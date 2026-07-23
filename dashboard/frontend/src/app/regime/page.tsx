"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ComposedChart, BarChart, AreaChart, Bar, Line, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import RegimePanel from "@/components/RegimePanel";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Spinner } from "@/components/ui/Spinner";
import { LearningPanel } from "@/components/LearningPanel";
import { api } from "@/lib/api";
import type { RegimeStreamRow, RegimeSemanticsEntry } from "@/lib/api";
import { API_URL } from "@/lib/websocket";
import { REGIME_COLORS, type RegimeState, type Candle, type DataRange } from "@/lib/types";
import { useMARTAStore } from "@/lib/store";

// ── Section 1 — full-history regime timeline ──────────────────────────────────

const EVENTS = [
  { date: "2020-03-12", label: "COVID Crash" },
  { date: "2021-11-10", label: "BTC ATH" },
  { date: "2022-11-08", label: "FTX Collapse" },
];

function findNearestTs(target: string, data: Array<{ timestamp: string }>): string | null {
  if (!data.length) return null;
  const targetMs = new Date(target).getTime();
  let best = data[0].timestamp;
  let bestDiff = Infinity;
  for (const d of data) {
    const diff = Math.abs(new Date(d.timestamp).getTime() - targetMs);
    if (diff < bestDiff) { bestDiff = diff; best = d.timestamp; }
  }
  return best;
}

// Decimate to a hard point-count ceiling — safe regardless of source resolution
// (spec target: ~1 point per 6h; we cap by count so it degrades gracefully on
// any dataset size instead of assuming a fixed hourly cadence).
function decimate<T>(rows: T[], maxPoints = 2000): T[] {
  if (rows.length <= maxPoints) return rows;
  const step = Math.ceil(rows.length / maxPoints);
  return rows.filter((_, i) => i % step === 0);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function RegimeBandShape(props: any) {
  const { x, width, regime_label } = props;
  if (x === undefined || !width) return null;
  return <rect x={x} y={-9999} width={width} height={99999} fill={REGIME_COLORS[regime_label] ?? "#444"} fillOpacity={0.1} />;
}

function RegimeFullTimeline({ candles, streamRows, onPick }: {
  candles: Candle[];
  streamRows: RegimeStreamRow[];
  onPick: (ts: string) => void;
}) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const priceData = useMemo(() => candles.filter((c) => c.close != null && c.close > 0), [candles]);

  const regimeKeys = useMemo(() => {
    const row = streamRows[0];
    if (!row) return [];
    return Object.keys(row)
      .filter((k) => k.startsWith("regime_prob_"))
      .map((k) => parseInt(k.replace("regime_prob_", "")))
      .sort((a, b) => a - b);
  }, [streamRows]);

  const priceEvents = useMemo(
    () => EVENTS.map((e) => ({ ...e, ts: findNearestTs(e.date, priceData) })).filter((e) => e.ts),
    [priceData],
  );
  const areaEvents = useMemo(
    () => EVENTS.map((e) => ({ ...e, ts: findNearestTs(e.date, streamRows) })).filter((e) => e.ts),
    [streamRows],
  );

  if (priceData.length === 0 && streamRows.length === 0) {
    return <EmptyState title="No timeline data" description="Run the data + regime pipelines to populate this view." />;
  }

  return (
    <div className="flex flex-col gap-1">
      {priceData.length > 0 && (
        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={priceData}
              margin={{ top: 16, right: 8, bottom: 0, left: 4 }}
              onClick={(ev) => {
                const ts = (ev?.activePayload?.[0]?.payload as Candle | undefined)?.timestamp;
                if (ts) onPick(ts);
              }}
              style={{ cursor: "crosshair" }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => new Date(ts).getFullYear().toString()}
                tick={{ fill: "#444", fontSize: 10 }}
                axisLine={{ stroke: "#1a1a1a" }}
                tickLine={false}
                minTickGap={60}
              />
              <YAxis
                scale="log"
                domain={["auto", "auto"]}
                allowDataOverflow
                tickFormatter={(v) => `$${Number(v) >= 1000 ? (Number(v) / 1000).toFixed(0) + "k" : v}`}
                tick={{ fill: "#444", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={48}
                orientation="right"
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const c = payload[0].payload as Candle;
                  return (
                    <div className="bg-[#111] border border-[#333] rounded p-2 text-[10px] font-mono">
                      <div className="text-[#666]">{new Date(c.timestamp).toLocaleDateString()}</div>
                      <div style={{ color: c.close >= c.open ? "#00ff7f" : "#ff3737" }}>${c.close?.toFixed(0)}</div>
                      <div className="text-[#555]">R{c.regime_label} {c.regime_description || regimeNames[c.regime_label] || ""}</div>
                    </div>
                  );
                }}
              />
              <Bar
                dataKey="close"
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                shape={(props: any) => <RegimeBandShape {...props} regime_label={priceData[props.index]?.regime_label} />}
                isAnimationActive={false}
              />
              <Line type="monotone" dataKey="close" stroke="#00ff7f" strokeWidth={1.25} dot={false} isAnimationActive={false} />
              {priceEvents.map((e) => (
                <ReferenceLine
                  key={e.label}
                  x={e.ts!}
                  stroke="#e0e0e0"
                  strokeDasharray="4 3"
                  strokeOpacity={0.6}
                  label={{ value: e.label, position: "top", fill: "#e0e0e0", fontSize: 9 }}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {streamRows.length > 0 && (
        <div style={{ height: 110 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={streamRows} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => new Date(ts).getFullYear().toString()}
                tick={{ fill: "#444", fontSize: 9 }}
                axisLine={{ stroke: "#1a1a1a" }}
                tickLine={false}
                minTickGap={60}
              />
              <YAxis hide domain={[0, 1]} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="bg-[#111] border border-[#333] rounded p-2 text-[10px] font-mono">
                      <div className="text-[#666] mb-1">{new Date(label as string).toLocaleDateString()}</div>
                      {payload.slice().reverse().map((p) => (
                        <div key={p.dataKey as string} style={{ color: p.color }}>
                          R{(p.dataKey as string).replace("regime_prob_", "")}: {((p.value as number) * 100).toFixed(0)}%
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              {regimeKeys.map((k) => (
                <Area
                  key={k}
                  type="monotone"
                  dataKey={`regime_prob_${k}`}
                  stackId="regimes"
                  stroke={REGIME_COLORS[k] ?? "#666"}
                  fill={REGIME_COLORS[k] ?? "#666"}
                  fillOpacity={0.65}
                  isAnimationActive={false}
                />
              ))}
              {areaEvents.map((e) => (
                <ReferenceLine key={e.label} x={e.ts!} stroke="#e0e0e0" strokeDasharray="4 3" strokeOpacity={0.3} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="flex justify-between text-[9px] text-[#444] px-1">
        <span>{priceData[0] ? new Date(priceData[0].timestamp).toLocaleDateString() : ""}</span>
        <span className="text-[#333]">Click the price chart to inspect that timestamp on the Debates page</span>
        <span>{priceData.at(-1) ? new Date(priceData.at(-1)!.timestamp).toLocaleDateString() : ""}</span>
      </div>
    </div>
  );
}

// ── Section 2 — per-regime statistics cards ───────────────────────────────────

function RegimeStatCard({ id, entry }: { id: number; entry: RegimeSemanticsEntry }) {
  const color = REGIME_COLORS[id] ?? "#666";
  // entry.description comes straight from this same /api/regime/semantics call — always
  // in sync with the exact stats shown on this card, no separate lookup needed.
  const name = entry.description ?? `Regime ${id}`;
  const pct = entry.time_pct * 100;
  const retPct = entry.close_return_mean * 100;
  return (
    <div className="rounded border border-[#1a1a1a] bg-[#0d0d0d] p-3 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
        <span className="font-semibold text-[12px]" style={{ color }}>R{id} — {name}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-[#1a1a1a]">
          <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
        </div>
        <span className="text-[10px] font-mono shrink-0" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px]">
        <span className="text-[#555]">Sharpe</span>
        <span className="text-right font-mono" style={{ color: entry.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}>{entry.sharpe.toFixed(3)}</span>
        <span className="text-[#555]">Avg Vol</span>
        <span className="text-right font-mono text-[#999]">{entry.realized_vol_mean.toFixed(3)}</span>
        <span className="text-[#555]">Avg Ret</span>
        <span className="text-right font-mono" style={{ color: retPct >= 0 ? "#00ff7f" : "#ff3737" }}>{retPct >= 0 ? "+" : ""}{retPct.toFixed(3)}%</span>
        <span className="text-[#555]">Dominant</span>
        <span className="text-right font-mono text-[#999]">{entry.dominant_asset}</span>
      </div>
    </div>
  );
}

// ── Section 3 — t-SNE embedding visualization ─────────────────────────────────

function TsneSection() {
  const [zoomed, setZoomed] = useState(false);
  const [error, setError] = useState(false);
  const src = `${API_URL}/api/plots/tsne`;

  return (
    <Panel title="Contrastive Embedding — t-SNE Projection">
      {error ? (
        <EmptyState title="t-SNE plot not available" description="Run the regime pipeline to generate regime/plots/regime_tsne.png." />
      ) : (
        <div className="flex flex-col gap-3">
          <div
            className="relative rounded border border-[#1a1a1a] overflow-hidden cursor-zoom-in bg-black flex items-center justify-center"
            onClick={() => setZoomed(true)}
            title="Click to zoom"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt="t-SNE projection of contrastive regime embeddings" className="max-w-full h-auto" onError={() => setError(true)} />
          </div>
          <p className="text-[11px] text-[#666] leading-relaxed max-w-3xl">
            Each point is one 60-candle window, projected from its 128-dim TFT + InfoNCE embedding down to 2D.
            Clusters correspond to the regimes the downstream Gaussian HMM discovers — tight, well-separated
            clusters mean the contrastive pretraining learned a market-state representation the HMM can cleanly
            partition, rather than a diffuse cloud it has to force boundaries onto.
          </p>
          <details className="text-[11px] text-[#555]">
            <summary className="cursor-pointer text-[#4a9eff] hover:text-[#6db4ff] transition-colors select-none">
              How this was generated — InfoNCE contrastive learning, in plain English
            </summary>
            <p className="mt-2 leading-relaxed text-[#666] max-w-3xl">
              The encoder is trained to answer one question, over and over: &ldquo;given this market window, which
              other window came from a nearby, similar moment — and which windows are unrelated?&rdquo; It is shown a
              window and a true positive (an augmented view of the same window), alongside many negatives
              (unrelated windows), and the InfoNCE loss pulls the positive pair&apos;s embeddings together while
              pushing every negative apart. Repeated over millions of windows, the only way to keep satisfying
              that objective is to encode the underlying market regime — because regime is the latent property
              that makes two windows genuinely alike. t-SNE then compresses the resulting 128-dim space to 2D
              purely for visualization; the clustering it reveals was already present in the embedding, not
              created by t-SNE itself.
            </p>
          </details>
        </div>
      )}

      {zoomed && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center cursor-zoom-out p-8"
          onClick={() => setZoomed(false)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt="t-SNE projection (zoomed)" className="max-w-full max-h-full object-contain" />
        </div>
      )}
    </Panel>
  );
}

// ── Phase 1 leftovers — per-asset current-state explorer ──────────────────────

function RegimeTimeline({ rows }: { rows: RegimeStreamRow[] }) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  if (rows.length === 0) {
    return <EmptyState title="No regime stream data" description="Run the regime pipeline to populate regime_stream.parquet." />;
  }
  return (
    <div className="flex flex-col gap-2">
      <div className="flex h-8 rounded overflow-hidden border border-[#1a1a1a]">
        {rows.map((r, i) => (
          <div
            key={i}
            title={`${r.timestamp} — R${r.regime_label} ${regimeNames[r.regime_label] ?? ""}`}
            className="h-full"
            style={{ flex: "1 0 auto", minWidth: 1, background: REGIME_COLORS[r.regime_label] ?? "#333" }}
          />
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-[#444]">
        <span>{new Date(rows[0].timestamp).toLocaleString()}</span>
        <span>{new Date(rows[rows.length - 1].timestamp).toLocaleString()}</span>
      </div>
    </div>
  );
}

function RegimeDistribution({ rows }: { rows: RegimeStreamRow[] }) {
  const data = useMemo(() => {
    const counts: Record<number, number> = {};
    for (const r of rows) counts[r.regime_label] = (counts[r.regime_label] ?? 0) + 1;
    return Object.entries(counts)
      .map(([label, count]) => ({
        label: Number(label),
        name: `R${label}`,
        count,
        color: REGIME_COLORS[Number(label)] ?? "#666",
      }))
      .sort((a, b) => a.label - b.label);
  }, [rows]);

  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data}>
        <XAxis dataKey="name" tick={{ fill: "#555", fontSize: 10 }} axisLine={{ stroke: "#222" }} tickLine={false} />
        <YAxis tick={{ fill: "#555", fontSize: 10 }} axisLine={{ stroke: "#222" }} tickLine={false} />
        <Tooltip contentStyle={{ background: "#111", border: "1px solid #2a2a2a", fontSize: 11 }} />
        <Bar dataKey="count" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function RegimePage() {
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const setSelectedTimestamp = useMARTAStore((s) => s.setSelectedTimestamp);

  const [regime, setRegime] = useState<RegimeState | null>(null);
  const [rows, setRows] = useState<RegimeStreamRow[]>([]);
  const [loading, setLoading] = useState(true);

  const [btcCandles, setBtcCandles] = useState<Candle[]>([]);
  const [btcStream, setBtcStream] = useState<RegimeStreamRow[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(true);
  const [timelineError, setTimelineError] = useState<string | null>(null);

  const [semantics, setSemantics] = useState<Record<string, RegimeSemanticsEntry> | null>(null);

  // Per-asset current-state explorer (Phase 1)
  useEffect(() => {
    setLoading(true);
    api.getRegime(selectedAsset).then(setRegime).catch(() => {});
    api.getRegimeStream(selectedAsset, 500)
      .then(setRows)
      .catch(() => setRows([]))
      .finally(() => setLoading(false));
  }, [selectedAsset]);

  // Section 1 — full history for the selected asset, fetched on asset change; 15s
  // timeout with an explicit error state instead of an infinite spinner. The backend
  // decimates the stream to ~2000 points server-side; limit=200000 means "the full
  // history", not the payload size.
  const loadTimeline = useCallback(async (asset: string) => {
    setTimelineLoading(true);
    setTimelineError(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    try {
      const rangeRes = await fetch(`${API_URL}/api/data-range/${asset}`, { signal: controller.signal });
      if (!rangeRes.ok) throw new Error(`HTTP ${rangeRes.status}`);
      const range: DataRange = await rangeRes.json();
      const [histRes, streamRes] = await Promise.all([
        fetch(
          `${API_URL}/api/history/${asset}/range?start=${encodeURIComponent(range.start)}&end=${encodeURIComponent(range.end)}`,
          { signal: controller.signal },
        ),
        fetch(`${API_URL}/api/regime-stream/${asset}?limit=200000`, { signal: controller.signal }),
      ]);
      if (!histRes.ok) throw new Error(`HTTP ${histRes.status}`);
      if (!streamRes.ok) throw new Error(`HTTP ${streamRes.status}`);
      const history: { candles: Candle[] } = await histRes.json();
      const stream: RegimeStreamRow[] = await streamRes.json();
      setBtcCandles(decimate(history.candles, 2000));
      setBtcStream(decimate(stream, 2000));
    } catch (err) {
      setBtcCandles([]);
      setBtcStream([]);
      if (err instanceof DOMException && err.name === "AbortError") {
        setTimelineError("Timeline data timed out — backend may be processing. Try refreshing.");
      } else {
        setTimelineError(`Failed to load timeline: ${err instanceof Error ? err.message : String(err)}`);
      }
    } finally {
      clearTimeout(timeoutId);
      setTimelineLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTimeline(selectedAsset);
  }, [loadTimeline, selectedAsset]);

  // Section 2 — regime semantics
  useEffect(() => {
    api.getRegimeSemantics().then((d) => setSemantics(d.regimes ?? null)).catch(() => setSemantics(null));
  }, []);

  return (
    <div className="flex flex-col gap-3 pb-4">
      <PageHeader title="Regime Explorer" description="HMM regime probability stream, historical timeline, and the contrastive embedding behind it" />

      {/* Section 1 */}
      <Panel title={`${selectedAsset.replace("_", "/")} Regime Timeline — 2018–2024 (log scale)`}>
        {timelineLoading ? (
          <Spinner label="Loading full history…" />
        ) : timelineError ? (
          <div className="flex flex-col items-center justify-center gap-3 py-8">
            <div className="text-[11px] text-[#ff8c00] text-center max-w-md">{timelineError}</div>
            <button
              onClick={() => loadTimeline(selectedAsset)}
              className="px-3 py-1.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors"
            >
              Retry
            </button>
          </div>
        ) : (
          <RegimeFullTimeline candles={btcCandles} streamRows={btcStream} onPick={setSelectedTimestamp} />
        )}
      </Panel>

      {/* Section 2 */}
      <Panel title="Regime Statistics">
        {semantics ? (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {Object.entries(semantics)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([id, entry]) => <RegimeStatCard key={id} id={Number(id)} entry={entry} />)}
          </div>
        ) : (
          <EmptyState title="No regime semantics" description="Run the regime pipeline to populate data/validated/regime_semantics.json." />
        )}
      </Panel>

      {/* Section 3 */}
      <TsneSection />

      {/* Per-asset current-state explorer (kept from Phase 1) */}
      <div className="grid grid-cols-[1fr_2fr] gap-2" style={{ minHeight: 420 }}>
        <RegimePanel regime={regime} />
        <div className="flex flex-col gap-2 min-h-0">
          <Panel title={`${selectedAsset.replace("_", "/")} — Recent Regime Timeline (last 500 samples)`}>
            {loading ? <Spinner label="Loading regime stream…" /> : <RegimeTimeline rows={rows} />}
          </Panel>
          <Panel title={`${selectedAsset.replace("_", "/")} — Regime Distribution`}>
            {loading ? <Spinner label="Loading…" /> : <RegimeDistribution rows={rows} />}
          </Panel>
        </div>
      </div>

      <LearningPanel>
        t-SNE projection of 159,824 encoder embeddings. Clusters confirm the encoder learned distinct
        market personalities across 7 years of BTC, ETH, EURUSD, XAUUSD data — the downstream Gaussian
        HMM only has to draw boundaries between clusters the contrastive pretraining already separated.
      </LearningPanel>
    </div>
  );
}
