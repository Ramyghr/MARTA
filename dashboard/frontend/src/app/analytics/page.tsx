"use client";
import { useEffect, useMemo, useState } from "react";
import {
  ComposedChart, BarChart, Bar, Line, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/ui/StatCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { ProvenanceTag } from "@/components/ui/ProvenanceTag";
import { Skeleton, SkeletonRows } from "@/components/ui/Skeleton";
import { LearningPanel } from "@/components/LearningPanel";
import { api } from "@/lib/api";
import type { AblationResults } from "@/lib/api";
import { VARIANT_LABELS, TEST_STRATEGY_ORDER } from "@/lib/ablation";
import { useMARTAStore } from "@/lib/store";
import type { Portfolio, TradeEntry, Candle, TestSetResults } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

function fmtPct(x: number | null | undefined, digits = 1): string {
  return x == null ? "—" : `${(x * 100).toFixed(digits)}%`;
}
function sharpeColor(s: number): string {
  if (s >= 1.5) return "#00ff7f";
  if (s >= 0.5) return "#ffd700";
  if (s > 0) return "#ff8c00";
  return "#ff3737";
}
function fmtPrice(v: number): string {
  return v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(2)}`;
}

// ── Shared trade-derived stats (computed once, used by Sections 1 & 2) ────────

interface TradeStats {
  meaningfulTrades: TradeEntry[];
  winRate: number | null;
  best: TradeEntry | null;
  worst: TradeEntry | null;
  avgHoldingHours: number | null;
  byRegime: Record<number, { trades: TradeEntry[]; winRate: number | null }>;
}

function computeTradeStats(trades: TradeEntry[]): TradeStats {
  const meaningful = trades.filter((t) => t.action !== "hold");
  const wins = meaningful.filter((t) => t.pnl > 0);
  const winRate = meaningful.length ? wins.length / meaningful.length : null;

  let best: TradeEntry | null = null;
  let worst: TradeEntry | null = null;
  for (const t of meaningful) {
    if (!best || t.pnl > best.pnl) best = t;
    if (!worst || t.pnl < worst.pnl) worst = t;
  }

  let avgHoldingHours: number | null = null;
  if (meaningful.length >= 2) {
    const sorted = [...meaningful].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    const gaps: number[] = [];
    for (let i = 1; i < sorted.length; i++) {
      gaps.push((new Date(sorted[i].timestamp).getTime() - new Date(sorted[i - 1].timestamp).getTime()) / 3_600_000);
    }
    avgHoldingHours = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  }

  const byRegime: Record<number, { trades: TradeEntry[]; winRate: number | null }> = {};
  for (const t of meaningful) {
    (byRegime[t.regime_label] ??= { trades: [], winRate: null }).trades.push(t);
  }
  for (const r of Object.values(byRegime)) {
    const w = r.trades.filter((t) => t.pnl > 0).length;
    r.winRate = r.trades.length ? w / r.trades.length : null;
  }

  return { meaningfulTrades: meaningful, winRate, best, worst, avgHoldingHours, byRegime };
}

// ── Section 1 — Performance Summary ───────────────────────────────────────────

function PerformanceSummarySection({ portfolio, stats }: { portfolio: Portfolio | null; stats: TradeStats | null }) {
  if (!portfolio) return <EmptyState title="No portfolio data" description="Run the RL pipeline to generate evaluation results." />;

  const deltaSharpe = portfolio.sharpe - portfolio.buy_and_hold.sharpe;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <StatCard label="Sharpe (backtest)" value={portfolio.sharpe.toFixed(3)} color={sharpeColor(portfolio.sharpe)} />
        <StatCard label="Win Rate" value={fmtPct(stats?.winRate)} color={stats?.winRate && stats.winRate >= 0.5 ? "#00ff7f" : "#ff8c00"} />
        <StatCard
          label="Avg Holding Time"
          value={stats?.avgHoldingHours != null ? `${stats.avgHoldingHours.toFixed(1)}h` : "—"}
        />
        <StatCard label="Max Drawdown" value={fmtPct(portfolio.max_drawdown)} color="#ff8c00" />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded border border-[#1a1a1a] p-3">
          <div className="stat-label mb-1">Best Trade</div>
          {stats?.best ? (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-[#888]">{new Date(stats.best.timestamp).toLocaleDateString()} · R{stats.best.regime_label}</span>
              <span className="font-mono font-semibold text-[#00ff7f]">+{(stats.best.pnl * 100).toFixed(2)}%</span>
            </div>
          ) : <span className="text-[11px] text-[#444]">—</span>}
        </div>
        <div className="rounded border border-[#1a1a1a] p-3">
          <div className="stat-label mb-1">Worst Trade</div>
          {stats?.worst ? (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-[#888]">{new Date(stats.worst.timestamp).toLocaleDateString()} · R{stats.worst.regime_label}</span>
              <span className="font-mono font-semibold text-[#ff3737]">{(stats.worst.pnl * 100).toFixed(2)}%</span>
            </div>
          ) : <span className="text-[11px] text-[#444]">—</span>}
        </div>
      </div>

      <div className="rounded border border-[#1a1a1a] p-3">
        <div className="stat-label mb-2">MARTA vs Buy &amp; Hold</div>
        <table className="data-table">
          <thead>
            <tr><th></th><th>Sharpe</th><th>Max DD</th><th>Total Return</th></tr>
          </thead>
          <tbody>
            <tr style={{ borderLeft: "3px solid #00ff7f" }}>
              <td className="font-semibold" style={{ color: "#00ff7f" }}>MARTA</td>
              <td className="font-mono" style={{ color: sharpeColor(portfolio.sharpe) }}>{portfolio.sharpe.toFixed(3)}</td>
              <td className="font-mono text-[#ff8c00]">{fmtPct(portfolio.max_drawdown)}</td>
              <td className="font-mono" style={{ color: portfolio.total_return >= 0 ? "#00ff7f" : "#ff3737" }}>
                {portfolio.total_return >= 0 ? "+" : ""}{fmtPct(portfolio.total_return)}
              </td>
            </tr>
            <tr>
              <td className="text-[#888]">Buy &amp; Hold</td>
              <td className="font-mono" style={{ color: sharpeColor(portfolio.buy_and_hold.sharpe) }}>{portfolio.buy_and_hold.sharpe.toFixed(3)}</td>
              <td className="font-mono text-[#ff8c00]">{fmtPct(portfolio.buy_and_hold.max_drawdown)}</td>
              <td className="font-mono" style={{ color: portfolio.buy_and_hold.total_return >= 0 ? "#00ff7f" : "#ff3737" }}>
                {portfolio.buy_and_hold.total_return >= 0 ? "+" : ""}{fmtPct(portfolio.buy_and_hold.total_return)}
              </td>
            </tr>
          </tbody>
        </table>
        <div className="text-[10px] text-[#555] mt-2">
          ΔSharpe {deltaSharpe >= 0 ? "+" : ""}{deltaSharpe.toFixed(3)} at {portfolio.buy_and_hold.max_drawdown > 0 ? (portfolio.buy_and_hold.max_drawdown / Math.max(portfolio.max_drawdown, 1e-6)).toFixed(0) : "—"}x lower drawdown.
        </div>
      </div>
    </div>
  );
}

// ── Section 2 — Regime Performance Heatmap ────────────────────────────────────

function RegimeHeatmapSection({ portfolio, stats }: { portfolio: Portfolio | null; stats: TradeStats | null }) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  if (!portfolio || portfolio.per_regime.length === 0) {
    return <EmptyState title="No per-regime data" description="Run the RL evaluation to populate per-regime breakdown." />;
  }

  const rows = [...portfolio.per_regime].sort((a, b) => a.regime_label - b.regime_label);
  const maxAbsReturn = Math.max(...rows.map((r) => Math.abs(r.avg_return)), 1e-9);

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {rows.map((r) => {
          const regimeStats = stats?.byRegime[r.regime_label];
          const intensity = Math.min(1, Math.abs(r.avg_return) / maxAbsReturn);
          const isPositive = r.avg_return >= 0;
          const bg = isPositive ? `rgba(0,255,127,${0.06 + intensity * 0.22})` : `rgba(255,55,55,${0.06 + intensity * 0.22})`;
          const border = isPositive ? "#00ff7f33" : "#ff373733";
          const color = REGIME_COLORS[r.regime_label] ?? "#666";

          return (
            <div key={r.regime_label} className="rounded p-3 flex flex-col gap-1.5" style={{ background: bg, border: `1px solid ${border}` }}>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                <span className="text-[11px] font-semibold" style={{ color }}>R{r.regime_label} — {regimeNames[r.regime_label] ?? ""}</span>
              </div>
              <div className="text-[10px] text-[#555]">Avg Return</div>
              <div className="font-mono text-[14px] font-bold" style={{ color: isPositive ? "#00ff7f" : "#ff3737" }}>
                {isPositive ? "+" : ""}{(r.avg_return * 100).toFixed(3)}%
              </div>
              <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px] mt-1">
                <span className="text-[#555]">Win Rate</span>
                <span className="text-right font-mono text-[#999]">{fmtPct(regimeStats?.winRate ?? null, 0)}</span>
                <span className="text-[#555]">Trades</span>
                <span className="text-right font-mono text-[#999]">{regimeStats?.trades.length ?? 0}</span>
                <span className="text-[#555]">Avg Size</span>
                <span className="text-right font-mono text-[#999]">{(r.avg_position_size * 100).toFixed(1)}%</span>
                <span className="text-[#555]">Sharpe</span>
                <span className="text-right font-mono" style={{ color: r.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}>{r.sharpe.toFixed(2)}</span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-2 text-[9px] text-[#444] px-1">
        <span>Loss</span>
        <div className="flex-1 h-1.5 rounded-full" style={{ background: "linear-gradient(90deg, rgba(255,55,55,0.5), #1a1a1a, rgba(0,255,127,0.5))" }} />
        <span>Profit</span>
        <span className="ml-2 text-[#333]">— color intensity = |avg return| within this run</span>
      </div>
    </div>
  );
}

// ── Section 3 — Agent Agreement Chart ─────────────────────────────────────────

const AGENT_COLORS: Record<string, string> = {
  MacroAgent: "#4a9eff",
  TechnicalAgent: "#ffd700",
  SentimentAgent: "#00c8d7",
  ContrarianAgent: "#b07fff",
};
const AGENT_ORDER = ["MacroAgent", "TechnicalAgent", "SentimentAgent", "ContrarianAgent"];

function AgentAgreementSection() {
  const [data, setData] = useState<Array<{ agent: string; agreementPct: number; total: number }> | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Server computes the tally over the full cached debate list in one pass (see
    // GET /api/debates/agreement) instead of the frontend fetching every debate's full
    // detail individually — with 384 real debates that used to mean 384 concurrent
    // GET /api/debates/{id} requests per page load.
    api.getDebatesAgreement().then((res) => {
      if (cancelled) return;
      const rows = res.agreement
        .map((a) => ({ agent: a.agent, agreementPct: a.agreement_pct, total: a.total }))
        .sort((a, b) => AGENT_ORDER.indexOf(a.agent) - AGENT_ORDER.indexOf(b.agent));
      setData(rows);
    }).catch(() => setData([]));
    return () => { cancelled = true; };
  }, []);

  if (data === null) return <div className="p-1"><SkeletonRows rows={4} /></div>;
  if (data.length === 0) return <EmptyState title="No debate history" description="Run debates to populate agent-agreement statistics." />;

  return (
    <div className="flex flex-col gap-2">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
          <XAxis dataKey="agent" tick={{ fill: "#666", fontSize: 10 }} axisLine={{ stroke: "#1a1a1a" }} tickLine={false} tickFormatter={(v) => v.replace("Agent", "")} />
          <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: "#444", fontSize: 10 }} axisLine={false} tickLine={false} width={36} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as { agent: string; agreementPct: number; total: number };
              return (
                <div className="bg-[#111] border border-[#333] rounded p-2 text-[10px]">
                  <div style={{ color: AGENT_COLORS[d.agent] ?? "#999" }}>{d.agent}</div>
                  <div className="text-[#888]">Agreed {d.agreementPct.toFixed(0)}% of {d.total} debates</div>
                </div>
              );
            }}
          />
          <Bar dataKey="agreementPct" radius={[3, 3, 0, 0]}>
            {data.map((d, i) => <Cell key={i} fill={AGENT_COLORS[d.agent] ?? "#666"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="text-[10px] text-[#444] px-1">
        Agreement = agent&apos;s round-1 directional bias matched the Fund Manager&apos;s final action (bullish→long, bearish→short, neutral→hold).
      </div>
    </div>
  );
}

// ── Section 3b — Ablation & Out-of-Sample summary ─────────────────────────────
//
// The full breakdown (sortable table, bootstrap CI, caveats) lives on the Backtest
// page — this is the compact version so a number on Analytics is never shown with
// no way to tell which config produced it, without duplicating that whole table.
// Every row's method is a visible caption here, not a hover-only tooltip.

function AblationOosSummarySection() {
  const [ablation, setAblation] = useState<AblationResults | null>(null);
  const [testSet, setTestSet] = useState<TestSetResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getAblation().catch(() => null),
      api.getTestSet().catch(() => null),
    ]).then(([a, t]) => {
      setAblation(a);
      setTestSet(t);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-1"><SkeletonRows rows={4} /></div>;

  const ablationRows = ablation
    ? Object.entries(ablation.variants)
        .map(([key, v]) => ({ key, label: VARIANT_LABELS[key] ?? key, ...v }))
        .sort((a, b) => b.sharpe - a.sharpe)
    : [];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <div className="stat-label mb-2">Ablation (2024 validation)</div>
        {ablationRows.length === 0 ? (
          <EmptyState title="No ablation results" description="Run the RL ablation to populate this." />
        ) : (
          <div className="flex flex-col gap-1.5">
            {ablationRows.map((r) => {
              const isBest = r.key === "marta_full";
              return (
                <div key={r.key} className="flex flex-col gap-0.5 px-2 py-1.5 rounded" style={{ background: isBest ? "rgba(0,255,127,0.05)" : undefined, borderLeft: isBest ? "2px solid #00ff7f" : "2px solid transparent" }}>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold" style={{ color: isBest ? "#00ff7f" : "#999" }}>{isBest && "🏆 "}{r.label}</span>
                    <span className="font-mono" style={{ color: r.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}>Sharpe {r.sharpe.toFixed(3)}</span>
                  </div>
                  <div className="text-[9px] text-[#555] leading-snug">{r.method}</div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <div className="stat-label mb-2">Out-of-Sample (2025 test, never used in training)</div>
        {!testSet ? (
          <EmptyState title="Test set not yet evaluated" description="Run `python run_test_set.py`." />
        ) : (
          <div className="flex flex-col gap-1.5">
            {TEST_STRATEGY_ORDER.map(({ key, label }) => {
              const m = testSet[key];
              if (!m) return null;
              const isMarta = key === "marta_ppo";
              return (
                <div key={key} className="flex flex-col gap-0.5 px-2 py-1.5 rounded" style={{ background: isMarta ? "rgba(0,255,127,0.05)" : undefined, borderLeft: isMarta ? "2px solid #00ff7f" : "2px solid transparent" }}>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold" style={{ color: isMarta ? "#00ff7f" : "#999" }}>{isMarta && "🏆 "}{label}</span>
                    <span className="font-mono" style={{ color: m.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}>Sharpe {m.sharpe.toFixed(3)}</span>
                  </div>
                  {m.method && <div className="text-[9px] text-[#555] leading-snug">{m.method}</div>}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="md:col-span-2 text-[9px] text-[#444]">
        Full sortable breakdown, bootstrap CI, and caveats on the{" "}
        <a href="/backtest" className="text-[#4a9eff] hover:text-[#6db4ff] transition-colors">Backtest</a> page.
      </div>
    </div>
  );
}

// ── Section 4 — Signal History Timeline ───────────────────────────────────────

function TradeMarker(props: unknown) {
  const p = props as { cx?: number; cy?: number; payload?: { action?: string } };
  if (p.cx == null || p.cy == null || !p.payload) return <g />;
  const isLong = p.payload.action === "long";
  const color = isLong ? "#00ff7f" : "#ff3737";
  const size = 5;
  const points = isLong
    ? `${p.cx},${p.cy - size} ${p.cx - size},${p.cy + size} ${p.cx + size},${p.cy + size}`
    : `${p.cx},${p.cy + size} ${p.cx - size},${p.cy - size} ${p.cx + size},${p.cy - size}`;
  return <polygon points={points} fill={color} stroke="#000" strokeWidth={0.5} />;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function RegimeBandShape(props: any) {
  const { x, width, regime_label } = props;
  if (x === undefined || !width) return null;
  return <rect x={x} y={-9999} width={width} height={99999} fill={REGIME_COLORS[regime_label] ?? "#444"} fillOpacity={0.12} />;
}

function SignalTimelineSection({ asset }: { asset: string }) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [trades, setTrades] = useState<TradeEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getHistory(asset, 500).then((d) => d.candles).catch(() => []),
      api.getPositions().then((p) => (asset === "BTC_USDT" ? p.trades : [])).catch(() => []),
    ]).then(([c, t]) => {
      setCandles(c);
      setTrades(t.filter((tr) => tr.action !== "hold"));
    }).finally(() => setLoading(false));
  }, [asset]);

  const chartData = useMemo(() => {
    return candles.map((c) => {
      const nearestTrade = trades.find((t) => Math.abs(new Date(t.timestamp).getTime() - new Date(c.timestamp).getTime()) < 3_600_000 * 12);
      return { ...c, tradeMarker: nearestTrade ? c.close : null, action: nearestTrade?.action };
    });
  }, [candles, trades]);

  const regimeShifts = useMemo(() => {
    const shifts: string[] = [];
    for (let i = 1; i < candles.length; i++) {
      if (candles[i].regime_label !== candles[i - 1].regime_label) shifts.push(candles[i].timestamp);
    }
    return shifts;
  }, [candles]);

  if (loading) return <Skeleton className="h-72 w-full" />;
  if (candles.length === 0) return <EmptyState title="No timeline data" description="Historical data not available for this asset." />;

  return (
    <div className="flex flex-col gap-1">
      {asset !== "BTC_USDT" && (
        <div className="text-[10px] text-[#ff8c00] bg-[#ff8c0012] border border-[#ff8c0033] rounded px-2 py-1.5">
          Trade markers are only available for BTC_USDT — the asset the deployed PPO checkpoint
          trades (GET /api/rl/positions has no per-asset breakdown). Price/regime history for{" "}
          {asset.replace("_", "/")} is shown below without them.
        </div>
      )}
      <div style={{ height: 280 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(ts) => new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
              tick={{ fill: "#444", fontSize: 10 }}
              axisLine={{ stroke: "#1a1a1a" }}
              tickLine={false}
              minTickGap={50}
            />
            <YAxis
              domain={["auto", "auto"]}
              tickFormatter={(v) => fmtPrice(Number(v))}
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
                    <div className="text-[#666]">{new Date(c.timestamp).toLocaleString()}</div>
                    <div style={{ color: "#00ff7f" }}>{fmtPrice(c.close)}</div>
                    <div className="text-[#555]">R{c.regime_label} {regimeNames[c.regime_label] ?? ""}</div>
                  </div>
                );
              }}
            />
            <Bar
              dataKey="close"
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              shape={(props: any) => <RegimeBandShape {...props} regime_label={chartData[props.index]?.regime_label} />}
              isAnimationActive={false}
            />
            <Line type="monotone" dataKey="close" stroke="#00ff7f" strokeWidth={1.25} dot={false} isAnimationActive={false} />
            <Scatter dataKey="tradeMarker" shape={TradeMarker} isAnimationActive={false} />
            {regimeShifts.map((ts, i) => (
              <ReferenceLine key={i} x={ts} stroke="#e0e0e0" strokeDasharray="3 3" strokeOpacity={0.25} />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-between text-[9px] text-[#444] px-1">
        <span className="flex items-center gap-1"><span style={{ color: "#00ff7f" }}>▲</span> Long entry <span style={{ color: "#ff3737" }} className="ml-2">▼</span> Short entry</span>
        <span className="text-[#333]">Faint vertical lines mark regime shifts · shaded bands = regime</span>
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [tradeStats, setTradeStats] = useState<TradeStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getPortfolio().catch(() => null),
      api.getPositions().catch(() => null),
    ]).then(([p, positions]) => {
      setPortfolio(p);
      setTradeStats(positions ? computeTradeStats(positions.trades) : null);
    }).finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex flex-col gap-3 pb-4">
      <PageHeader
        title="Analytics"
        description="Performance summary, regime-conditioned returns, agent agreement, and the signal timeline"
        actions={<ProvenanceTag portfolio={portfolio} />}
      />

      <Panel title="Performance Summary">
        {loading ? <Skeleton className="h-64 w-full" /> : <PerformanceSummarySection portfolio={portfolio} stats={tradeStats} />}
      </Panel>

      <Panel title="Regime Performance Heatmap">
        {loading ? <Skeleton className="h-48 w-full" /> : <RegimeHeatmapSection portfolio={portfolio} stats={tradeStats} />}
      </Panel>

      <Panel title="Ablation & Out-of-Sample — where these numbers come from">
        <AblationOosSummarySection />
      </Panel>

      <Panel title="Agent Agreement with Final Decision">
        <AgentAgreementSection />
      </Panel>

      <Panel title={`Signal History Timeline — ${selectedAsset.replace("_", "/")}`}>
        <SignalTimelineSection asset={selectedAsset} />
      </Panel>

      <LearningPanel>
        The regime heatmap is the load-bearing chart: MARTA&apos;s Sharpe is not uniform across
        regimes — the policy concentrates exposure in the regimes where the historical avg return
        is positive and abstains (near-zero position size) elsewhere, which is exactly what
        &ldquo;regime-conditioned&rdquo; is supposed to look like rather than a flat, regime-blind
        strategy. The timeline below makes the same point visually: trade markers cluster inside
        favorable regime bands rather than firing uniformly across the whole history.
      </LearningPanel>
    </div>
  );
}
