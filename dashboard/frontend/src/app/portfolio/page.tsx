"use client";
import { useEffect, useMemo, useState } from "react";
import {
  ComposedChart, LineChart, Line, Area, AreaChart, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/ui/StatCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { ProvenanceTag } from "@/components/ui/ProvenanceTag";
import { Skeleton, SkeletonRows } from "@/components/ui/Skeleton";
import { TradeReasoningPanel } from "@/components/TradeReasoningPanel";
import { LearningPanel } from "@/components/LearningPanel";
import { api } from "@/lib/api";
import type { Portfolio, PositionState, EquityCurve, TradeEntry, AccountState } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { fetchApi } from "@/lib/websocket";
import { useMARTAStore } from "@/lib/store";

function fmtPct(x: number | undefined) {
  return x === undefined ? "—" : `${(x * 100).toFixed(1)}%`;
}
function fmtNum(x: number | undefined) {
  return x === undefined ? "—" : x.toFixed(3);
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
// Dynamic regime descriptions read "Vol / Perf (Asset-dominant)" — too long for a chart
// axis tick. Prefer the performance half ("Strong Bull") over the volatility half.
function shortRegimeLabel(fullName: string | undefined): string {
  if (!fullName) return "";
  const perfPart = fullName.split(" / ")[1] ?? fullName;
  return perfPart.split("(")[0].trim();
}

// ── Equity curve — two-tone line + drawdown shading ────────────────────────────

function EquityCurveSection({ curve }: { curve: EquityCurve | null }) {
  const rows = useMemo(() => {
    if (!curve || curve.timestamps.length === 0) return [];
    const initial = curve.marta[0] ?? 10000;
    let peak = -Infinity;
    return curve.timestamps.map((ts, i) => {
      const value = curve.marta[i];
      peak = Math.max(peak, value);
      // Clamped to [-1, 0]: a portfolio value can't legitimately go negative in this
      // accounting model, so drawdown can't legitimately exceed -100% — clamp defensively
      // rather than let a corrupt/outlier data point blow the chart's axis out.
      const dd = peak > 0 ? Math.max(-1, (value - peak) / peak) : 0;
      return { ts, value, dd, aboveInitial: value >= initial };
    });
  }, [curve]);

  if (!curve || rows.length === 0) {
    return <EmptyState title="No equity curve data" description="Run generate_equity_curve.py to produce data/results/equity_curve.json." />;
  }

  const gradientStops = rows.map((r, i) => ({
    offset: `${(i / Math.max(rows.length - 1, 1)) * 100}%`,
    color: r.aboveInitial ? "#00ff7f" : "#ff3737",
  }));

  const dateTick = (ts: string) => {
    try { return new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric" }); } catch { return ts; }
  };

  const CombinedTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof rows[number] }> }) => {
    if (!active || !payload?.length) return null;
    const r = payload[0].payload;
    return (
      <div className="bg-[#111] border border-[#333] rounded p-2 text-[11px] font-mono">
        <div className="text-[#666] mb-1">{new Date(r.ts).toLocaleDateString()}</div>
        <div style={{ color: r.aboveInitial ? "#00ff7f" : "#ff3737" }}>${r.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
        <div className="text-[#ff8c00]">DD {(r.dd * 100).toFixed(1)}%</div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-1">
      {curve.synthetic && (
        <div className="text-[10px] text-[#555] italic px-1">Approximated from saved metrics — run generate_equity_curve.py for the real walk-forward curve.</div>
      )}
      <div style={{ height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} syncId="equity" margin={{ top: 8, right: 8, bottom: 4, left: 4 }}>
            <defs>
              <linearGradient id="equityColor" x1="0" y1="0" x2="1" y2="0">
                {gradientStops.map((s, i) => <stop key={i} offset={s.offset} stopColor={s.color} />)}
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
            <XAxis dataKey="ts" tickFormatter={dateTick} tick={{ fill: "#444", fontSize: 10 }} axisLine={{ stroke: "#1a1a1a" }} tickLine={false} minTickGap={40} />
            <YAxis
              tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
              tick={{ fill: "#444", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={52}
              orientation="right"
              domain={["auto", "auto"]}
            />
            <Tooltip content={<CombinedTooltip />} />
            <ReferenceLine y={rows[0].value} stroke="#2a2a2a" strokeDasharray="4 4" strokeWidth={1} />
            <Line type="monotone" dataKey="value" stroke="url(#equityColor)" strokeWidth={1.75} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div style={{ height: 70 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rows} syncId="equity" margin={{ top: 0, right: 8, bottom: 4, left: 4 }}>
            <XAxis dataKey="ts" tickFormatter={dateTick} tick={{ fill: "#333", fontSize: 9 }} axisLine={{ stroke: "#1a1a1a" }} tickLine={false} minTickGap={40} />
            <YAxis
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "#333", fontSize: 9 }}
              axisLine={false}
              tickLine={false}
              width={52}
              orientation="right"
              domain={[(dataMin: number) => Math.max(-1, dataMin * 1.15), 0]}
            />
            <Tooltip content={<CombinedTooltip />} />
            <Area type="monotone" dataKey="dd" stroke="#ff3737" strokeWidth={1} fill="#ff3737" fillOpacity={0.18} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="text-[9px] text-[#333] text-center -mt-1">Drawdown from running peak</div>
    </div>
  );
}

// ── Live account summary — one shared capital pool across assets ───────────────

function actionColor(action: string): string {
  return action === "long" ? "#00ff7f" : action === "short" ? "#ff3737" : "#666666";
}

function AccountSummary() {
  const [account, setAccount] = useState<AccountState | null>(null);

  useEffect(() => {
    api.getAccountState().then(setAccount).catch(() => setAccount(null));
  }, []);

  if (!account) return null;
  const retColor = account.total_return >= 0 ? "#00ff7f" : "#ff3737";

  return (
    <div className="rounded border border-[#1a1a1a] bg-[#0d0d0d] px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px]">
      <span className="text-[#444] uppercase tracking-wider">Live Account — shared capital</span>
      <span className="font-mono font-semibold text-[#ccc]">
        ${account.shared_capital.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </span>
      <span className="font-mono" style={{ color: retColor }}>
        {account.total_return >= 0 ? "+" : ""}{(account.total_return * 100).toFixed(2)}%
      </span>
      <span className="text-[#333]">·</span>
      <span className="text-[#666]">{(account.total_exposure * 100).toFixed(1)}% total exposure deployed</span>
      <div className="flex items-center gap-2 ml-auto">
        {account.positions.map((p) => (
          <span key={p.asset} className="flex items-center gap-1 px-1.5 py-0.5 rounded border border-[#1a1a1a]">
            <span className="text-[#555]">{p.asset.replace("_", "/")}</span>
            <span className="font-semibold uppercase" style={{ color: actionColor(p.action) }}>
              {p.exposure > 0 ? `${p.action} ${(p.exposure * 100).toFixed(1)}%` : "flat"}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Trade log table ─────────────────────────────────────────────────────────────

function TradeLogTable({ trades }: { trades: TradeEntry[] }) {
  const rows = useMemo(() => {
    let cum = 0;
    const withCum = trades.map((t) => {
      cum += t.pnl;
      return { ...t, cumPnl: cum };
    });
    return withCum.reverse();
  }, [trades]);

  if (rows.length === 0) {
    return <EmptyState title="No trades logged" description="Run generate_equity_curve.py to produce a real position/trade log." />;
  }

  return (
    <div className="overflow-auto max-h-[420px]">
      <table className="data-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Action</th>
            <th>Size</th>
            <th>Price</th>
            <th>Step P&amp;L</th>
            <th>Cumulative P&amp;L</th>
            <th>Regime</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => {
            const profitable = t.pnl >= 0;
            const actionColor = t.action === "long" ? "#00ff7f" : t.action === "short" ? "#ff3737" : "#666";
            return (
              <tr key={i} style={{ background: profitable ? "rgba(0,255,127,0.05)" : "rgba(255,55,55,0.05)" }}>
                <td className="font-mono text-[#888]">{new Date(t.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td>
                <td className="font-semibold uppercase" style={{ color: actionColor }}>{t.action}</td>
                <td className="font-mono text-[#888]">{(t.position_size * 100).toFixed(1)}%</td>
                <td className="font-mono text-[#999]">{t.price ? fmtPrice(t.price) : "—"}</td>
                <td className="font-mono" style={{ color: t.pnl >= 0 ? "#00ff7f" : "#ff3737" }}>{t.pnl >= 0 ? "+" : ""}{(t.pnl * 100).toFixed(3)}%</td>
                <td className="font-mono" style={{ color: t.cumPnl >= 0 ? "#00ff7f" : "#ff3737" }}>{t.cumPnl >= 0 ? "+" : ""}{(t.cumPnl * 100).toFixed(2)}%</td>
                <td>
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: REGIME_COLORS[t.regime_label] ?? "#555" }} />
                    <span className="text-[#777]">R{t.regime_label}</span>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Per-regime performance — horizontal bar chart ──────────────────────────────

function RegimePerformanceChart({ portfolio }: { portfolio: Portfolio }) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const data = portfolio.per_regime
    .filter((r) => r.n_steps > 0)
    .map((r) => ({
      name: `R${r.regime_label} ${shortRegimeLabel(regimeNames[r.regime_label])}`,
      sharpe: parseFloat(r.sharpe.toFixed(3)),
      color: REGIME_COLORS[r.regime_label] ?? "#666",
    }))
    .sort((a, b) => b.sharpe - a.sharpe);

  if (data.length === 0) {
    return <EmptyState title="No per-regime data" description="Run the RL evaluation to populate per-regime breakdown." />;
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#141414" horizontal={false} />
        <XAxis type="number" tick={{ fill: "#444", fontSize: 10 }} axisLine={{ stroke: "#1a1a1a" }} tickLine={false} />
        <YAxis type="category" dataKey="name" tick={{ fill: "#888", fontSize: 10 }} axisLine={false} tickLine={false} width={90} />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload as { name: string; sharpe: number; color: string };
            return (
              <div className="bg-[#111] border border-[#222] rounded p-1.5 text-[10px]">
                <div style={{ color: d.color }}>{d.name}</div>
                <div className="text-[#888]">Sharpe: {d.sharpe}</div>
              </div>
            );
          }}
        />
        <Bar dataKey="sharpe" radius={[0, 3, 3, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.sharpe >= 0 ? d.color : "#ff373766"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function PortfolioPage() {
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<PositionState | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurve | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getPortfolio().catch(() => null),
      api.getPositions().catch(() => null),
      fetchApi<EquityCurve>("/api/rl/equity-curve").catch(() => null),
    ]).then(([p, pos, curve]) => {
      setPortfolio(p);
      setPositions(pos);
      setEquityCurve(curve);
    }).finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex flex-col gap-3 pb-4">
      <PageHeader
        title="Portfolio"
        description="P&L, equity curve, trade log, and per-regime performance from the trained PPO policy"
        actions={<ProvenanceTag portfolio={portfolio} />}
      />

      {/* Top row — 6 metrics */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2 shrink-0">
        {loading || !portfolio ? (
          Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-16 w-full" />)
        ) : (
          <>
            <StatCard label="Sharpe" value={fmtNum(portfolio.sharpe)} color={sharpeColor(portfolio.sharpe)} />
            <StatCard label="Sortino" value={fmtNum(portfolio.sortino)} color={sharpeColor(portfolio.sortino)} />
            <StatCard label="Max Drawdown" value={fmtPct(portfolio.max_drawdown)} color="#ff8c00" />
            <StatCard label="Calmar" value={fmtNum(portfolio.calmar)} />
            <StatCard label="Total Return" value={fmtPct(portfolio.total_return)} color={portfolio.total_return >= 0 ? "#00ff7f" : "#ff3737"} />
            <StatCard label="# Trades" value={portfolio.n_trades} />
          </>
        )}
      </div>

      {/* Main chart — equity curve */}
      <Panel title="Equity Curve — Validation Period">
        {loading ? <Skeleton className="h-64 w-full" /> : <EquityCurveSection curve={equityCurve} />}
      </Panel>

      {/* Live paper trades — the scheduler's actual executions, kept visually distinct
          from (and above) the historical backtest log below so the two are never confused. */}
      <AccountSummary />
      <div className="rounded border-2 border-[#00ff7f33] bg-[#00ff7f08] p-2">
        <TradeReasoningPanel asset={selectedAsset} />
      </div>

      <div className="grid grid-cols-[2fr_1fr] gap-2">
        {/* Backtest trade log — historical validation only, never a live execution */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] text-[#ff8c00] bg-[#ff8c0012] border border-[#ff8c0033] rounded px-2 py-1.5">
            These trades are from the {portfolio?.val_window ? `${portfolio.val_window.start} → ${portfolio.val_window.end}` : ""} validation backtest — not live executions.
          </div>
          <Panel
            title={`Backtest Trade Log${portfolio?.val_window ? ` (${portfolio.val_window.start.slice(0, 4)} Validation)` : ""}`}
          >
            {loading ? <div className="p-1"><SkeletonRows rows={8} /></div> : <TradeLogTable trades={positions?.trades ?? []} />}
          </Panel>
        </div>

        {/* Per-regime performance */}
        <Panel title="Sharpe by Regime">
          {loading || !portfolio ? <Skeleton className="h-48 w-full" /> : <RegimePerformanceChart portfolio={portfolio} />}
        </Panel>
      </div>

      {positions?.note && (
        <div className="text-[10px] text-[#444] italic px-1">{positions.note}</div>
      )}

      <LearningPanel>
        {portfolio ? (() => {
          const ddRatio = portfolio.max_drawdown > 0 ? portfolio.buy_and_hold.max_drawdown / portfolio.max_drawdown : null;
          const totalSteps = portfolio.per_regime.reduce((s, r) => s + r.n_steps, 0);
          const flatSteps = portfolio.per_regime.filter((r) => r.avg_position_size === 0).reduce((s, r) => s + r.n_steps, 0);
          const flatPct = totalSteps > 0 ? Math.round((flatSteps / totalSteps) * 100) : null;
          const bestRegime = [...portfolio.per_regime].sort((a, b) => b.sharpe - a.sharpe)[0];
          return (
            <>
              MARTA Sharpe {portfolio.sharpe.toFixed(3)} vs Buy &amp; Hold {portfolio.buy_and_hold.sharpe.toFixed(3)} — roughly
              equal risk-adjusted return{ddRatio != null ? ` at ${ddRatio.toFixed(0)}x lower drawdown` : ""} ({fmtPct(portfolio.max_drawdown)}{" "}
              vs {fmtPct(portfolio.buy_and_hold.max_drawdown)}). {flatPct != null && `The system holds a flat position ~${flatPct}% of the time, `}
              {bestRegime && bestRegime.sharpe > 0 && (
                <>deploying capital mainly in regime R{bestRegime.regime_label} (Sharpe {bestRegime.sharpe.toFixed(2)}).</>
              )}
            </>
          );
        })() : (
          "MARTA trades selectively — most of the historical value comes from concentrating exposure in a small number of favorable regimes and abstaining elsewhere, rather than being invested at a constant size throughout."
        )}
      </LearningPanel>
    </div>
  );
}
