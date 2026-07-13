"use client";
import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { PositionState, Portfolio, FundManagerDecision, TradeEntry, LiveSignal } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { useMARTAStore } from "@/lib/store";

interface Props {
  positions: PositionState | null;
  decision: FundManagerDecision | null;
  portfolio: Portfolio | null;
  liveMode: boolean;
  asset: string;
  liveSignal?: LiveSignal | null;
}

// Dynamic regime descriptions read "Vol / Perf (Asset-dominant)" — too long for a w-20
// label. Prefer the performance half ("Strong Bull") over the volatility half.
function shortRegimeLabel(fullName: string | undefined): string {
  if (!fullName) return "";
  const perfPart = fullName.split(" / ")[1] ?? fullName;
  return perfPart.split("(")[0].trim();
}

function ActionBadge({ action, size }: { action: string; size?: number }) {
  const color = action === "long" ? "#00ff7f" : action === "short" ? "#ff3737" : "#555";
  return (
    <div className="flex items-center gap-2">
      <span
        className="px-2 py-0.5 rounded text-[11px] font-bold uppercase"
        style={{ background: `${color}20`, color, border: `1px solid ${color}44` }}
      >
        {action}
      </span>
      {size !== undefined && size > 0.001 && (
        <span className="text-[11px] text-[#888]">{(size * 100).toFixed(1)}%</span>
      )}
    </div>
  );
}

function LivePositionCard({
  decision, liveMode, liveSignal,
}: {
  decision: FundManagerDecision | null;
  liveMode: boolean;
  liveSignal?: LiveSignal | null;
}) {
  if (!liveMode) return null;

  const action = liveSignal?.action ?? decision?.action ?? "hold";
  const size = liveSignal?.position_size ?? decision?.position_size ?? 0;
  const price = liveSignal?.price ?? 0;
  const pnl = liveSignal?.paper_total_return ?? 0;
  const pnlColor = pnl >= 0 ? "#00ff7f" : "#ff3737";
  const regimeLabel = liveSignal?.regime_label ?? 0;
  const regimeDesc = liveSignal?.regime_description ?? "";
  const portfolioVal = liveSignal?.paper_portfolio_value ?? 10000;

  return (
    <div className="rounded border border-[#1a1a1a] p-3 space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#444] uppercase tracking-wider">Live Paper Position</span>
        <span className="text-[10px] text-[#ff5555] animate-pulse">● LIVE</span>
      </div>

      {/* Action + regime row */}
      <div className="flex items-center gap-3">
        <ActionBadge action={action} size={size} />
        <div className="flex items-center gap-1.5 ml-auto">
          <div className="w-2 h-2 rounded-full" style={{ background: REGIME_COLORS[regimeLabel] ?? "#555" }} />
          <span className="text-[10px] text-[#666]">R{regimeLabel} {regimeDesc}</span>
        </div>
      </div>

      {/* Price + P&L */}
      <div className="grid grid-cols-3 gap-2 text-[10px]">
        <div>
          <div className="text-[#444] mb-0.5">Price</div>
          <div className="font-mono text-[#999]">
            {price ? (price >= 1000 ? `$${(price / 1000).toFixed(1)}k` : `$${price.toFixed(2)}`) : "—"}
          </div>
        </div>
        <div>
          <div className="text-[#444] mb-0.5">Paper P&L</div>
          <div className="font-mono font-semibold" style={{ color: pnlColor }}>
            {pnl >= 0 ? "+" : ""}{(pnl * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-[#444] mb-0.5">Portfolio</div>
          <div className="font-mono text-[#999]">${Math.round(portfolioVal).toLocaleString()}</div>
        </div>
      </div>

      {/* Debate override */}
      {decision && (
        <div className="text-[10px] text-[#555] border-t border-[#111] pt-2">
          Agent consensus: <span className="text-[#888]">{decision.action}</span> {decision.conviction ? `(${Math.round(decision.conviction * 100)}% conviction)` : ""}
          {" — "}<span className="text-[#444] line-clamp-1">{decision.rationale?.slice(0, 80)}</span>
        </div>
      )}

      {liveSignal?.note && (
        <div className="text-[9px] text-[#2a2a2a] italic">{liveSignal.note}</div>
      )}
    </div>
  );
}

function TradeRow({ trade }: { trade: TradeEntry }) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const color = trade.action === "long" ? "#00ff7f" : trade.action === "short" ? "#ff3737" : "#444";
  const pnlColor = trade.pnl >= 0 ? "#00ff7f" : "#ff3737";
  const regColor = REGIME_COLORS[trade.regime_label] ?? "#555";

  const ts = new Date(trade.timestamp);
  const dateStr = ts.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const timeStr = ts.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

  return (
    <div className="grid grid-cols-[60px_52px_44px_52px_64px_36px] gap-1 py-1 border-b border-[#111] text-[10px] items-center">
      <div className="text-[#444]">
        <div>{dateStr}</div>
        <div className="text-[#2a2a2a]">{timeStr}</div>
      </div>
      <span className="font-semibold uppercase" style={{ color }}>{trade.action}</span>
      <span className="text-[#777]">{(trade.position_size * 100).toFixed(1)}%</span>
      <span className="text-[#666] font-mono">
        {trade.price >= 1000
          ? `$${(trade.price / 1000).toFixed(1)}k`
          : `$${trade.price.toFixed(2)}`}
      </span>
      <span className="font-mono" style={{ color: pnlColor }}>
        {trade.pnl >= 0 ? "+" : ""}{(trade.pnl * 100).toFixed(3)}%
      </span>
      <div
        className="w-2 h-2 rounded-full"
        style={{ background: regColor }}
        title={`R${trade.regime_label} ${regimeNames[trade.regime_label] ?? ""}`}
      />
    </div>
  );
}

function EquityMiniChart({ trades }: { trades: TradeEntry[] }) {
  if (trades.length < 2) return null;

  const step = Math.max(1, Math.floor(trades.length / 80));
  const data = trades
    .filter((_, i) => i % step === 0)
    .map((t) => ({
      ts: new Date(t.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      value: Math.round(t.portfolio_value),
    }));

  const start = data[0]?.value ?? 10000;
  const end = data.at(-1)?.value ?? 10000;
  const isUp = end >= start;

  return (
    <div style={{ height: 70 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 4, bottom: 2, left: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="#111" vertical={false} />
          <XAxis dataKey="ts" tick={{ fill: "#333", fontSize: 9 }} axisLine={false} tickLine={false}
            interval={Math.floor(data.length / 4)} />
          <YAxis
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            tick={{ fill: "#333", fontSize: 9 }} axisLine={false} tickLine={false} width={34}
          />
          <Tooltip
            contentStyle={{ background: "#111", border: "1px solid #1a1a1a", fontSize: 10 }}
            formatter={(v: number) => [`$${v.toLocaleString()}`, "Portfolio"]}
          />
          <Line type="monotone" dataKey="value" stroke={isUp ? "#00ff7f" : "#ff3737"} strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function OrdersPanel({ positions, decision, portfolio, liveMode, asset, liveSignal }: Props) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const recentTrades = useMemo(() => {
    if (liveMode && liveSignal?.paper_trades?.length) {
      return [...liveSignal.paper_trades].reverse().slice(0, 50);
    }
    if (!positions?.trades) return [];
    return [...positions.trades].reverse().slice(0, 50);
  }, [positions, liveMode, liveSignal]);

  const currentAction = liveMode
    ? (liveSignal?.action ?? decision?.action ?? "hold")
    : (positions?.current_action ?? "hold");

  const currentSize = liveMode
    ? (liveSignal?.position_size ?? decision?.position_size ?? 0)
    : (positions?.current_position ?? 0);

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Header strip */}
      <div className="border-b border-[#1a1a1a] px-3 py-2 flex items-center gap-3 shrink-0">
        <span className="text-[10px] text-[#444] uppercase tracking-widest">
          {liveMode ? "Live Execution" : "RL Backtest — Execution Log"}
        </span>
        {positions?.source && !liveMode && (
          <span className="text-[10px] text-[#2a2a2a]">({positions.source})</span>
        )}
        {!liveMode && positions && (
          <span className="ml-auto text-[10px] text-[#444]">{positions.n_trades} trades</span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-3 space-y-4 min-h-0">
        {/* Live mode: show paper position card */}
        <LivePositionCard decision={decision} liveMode={liveMode} liveSignal={liveSignal} />

        {/* Current position summary */}
        <div className="grid grid-cols-3 gap-2">
          <div className="rounded border border-[#1a1a1a] p-2">
            <div className="text-[10px] text-[#444] mb-1 uppercase">Position</div>
            <ActionBadge action={currentAction} size={currentSize} />
          </div>
          <div className="rounded border border-[#1a1a1a] p-2">
            <div className="text-[10px] text-[#444] mb-1 uppercase">Regime</div>
            <div className="flex items-center gap-1.5">
              <div
                className="w-2 h-2 rounded-full"
                style={{ background: REGIME_COLORS[positions?.current_regime ?? 0] ?? "#555" }}
              />
              <span className="text-[11px] text-[#888]">
                R{positions?.current_regime ?? "—"}
              </span>
            </div>
          </div>
          <div className="rounded border border-[#1a1a1a] p-2">
            <div className="text-[10px] text-[#444] mb-1 uppercase">P&amp;L</div>
            <div
              className="text-[11px] font-mono font-semibold"
              style={{ color: (positions?.total_return ?? 0) >= 0 ? "#00ff7f" : "#ff3737" }}
            >
              {positions ? `${(positions.total_return * 100) >= 0 ? "+" : ""}${(positions.total_return * 100).toFixed(1)}%` : "—"}
            </div>
          </div>
        </div>

        {/* Portfolio equity mini-chart (backtest mode) */}
        {!liveMode && recentTrades.length > 1 && (
          <div>
            <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">
              Portfolio Value — {asset.replace("_", "/")} val
            </div>
            <EquityMiniChart trades={[...recentTrades].reverse()} />
          </div>
        )}

        {/* Per-regime position table */}
        {!liveMode && portfolio?.per_regime && portfolio.per_regime.length > 0 && (
          <div>
            <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Position by Regime</div>
            <div className="space-y-1">
              {portfolio.per_regime.filter((r) => r.n_steps > 0).map((r) => (
                <div key={r.regime_label} className="flex items-center gap-2 text-[10px]">
                  <div
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ background: REGIME_COLORS[r.regime_label] ?? "#555" }}
                  />
                  <span className="text-[#555] w-20">
                    R{r.regime_label} {shortRegimeLabel(regimeNames[r.regime_label])}
                  </span>
                  <div className="flex-1 h-1 rounded bg-[#111] overflow-hidden">
                    <div
                      className="h-full rounded"
                      style={{
                        width: `${Math.min(100, r.avg_position_size * 1000)}%`,
                        background: REGIME_COLORS[r.regime_label] ?? "#555",
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <span className="text-[#666] w-10 text-right">{(r.avg_position_size * 100).toFixed(1)}%</span>
                  <span
                    className="w-16 text-right font-mono"
                    style={{ color: r.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}
                  >
                    Sharpe {r.sharpe.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Trade log */}
        {recentTrades.length > 0 && (
          <div>
            <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Trade Log</div>
            {/* Column headers */}
            <div className="grid grid-cols-[60px_52px_44px_52px_64px_36px] gap-1 pb-1 border-b border-[#1a1a1a] text-[9px] text-[#333] uppercase">
              <span>Time</span>
              <span>Action</span>
              <span>Size</span>
              <span>Price</span>
              <span>Step P&amp;L</span>
              <span>R</span>
            </div>
            <div className="space-y-0">
              {recentTrades.map((t, i) => <TradeRow key={i} trade={t} />)}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!liveMode && recentTrades.length === 0 && !positions && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center py-8">
            <div className="text-[#333] text-[11px]">No execution data</div>
            <div className="text-[10px] text-[#2a2a2a] leading-relaxed">
              Run <span className="font-mono text-[#3a3a3a]">python generate_equity_curve.py</span>
              {" "}to generate the real trade log from ppo_best.zip
            </div>
          </div>
        )}

        {/* Note from synthetic data */}
        {positions?.note && (
          <div className="text-[10px] text-[#2a2a2a] border-t border-[#111] pt-2 leading-relaxed italic">
            {positions.note}
          </div>
        )}
      </div>
    </div>
  );
}
