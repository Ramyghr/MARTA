"use client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { Portfolio } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { useMARTAStore } from "@/lib/store";

interface Props {
  portfolio: Portfolio | null;
}

function sharpeColor(s: number): string {
  if (s >= 1.5) return "#00ff7f";
  if (s >= 0.5) return "#ffd700";
  if (s > 0)   return "#ff8c00";
  return "#ff3737";
}

interface StatCellProps {
  label: string;
  value: string;
  color?: string;
  sub?: string;
  borderRight?: boolean;
  borderBottom?: boolean;
}

function StatCell({ label, value, color, sub, borderRight, borderBottom }: StatCellProps) {
  return (
    <div
      className="p-3 flex flex-col gap-0.5"
      style={{
        borderRight: borderRight ? "1px solid #1a1a1a" : undefined,
        borderBottom: borderBottom ? "1px solid #1a1a1a" : undefined,
      }}
    >
      <div className="stat-label">{label}</div>
      <div className="stat-value mt-0.5" style={{ color: color ?? "#e0e0e0" }}>{value}</div>
      {sub && <div className="text-[10px] text-[#444] mt-0.5">{sub}</div>}
    </div>
  );
}

export default function PortfolioPanel({ portfolio: p }: Props) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  if (!p) {
    return (
      <div className="panel h-full flex flex-col">
        <div className="panel-header">Portfolio Performance</div>
        <div className="flex-1 flex items-center justify-center text-[#333] text-[11px]">Loading...</div>
      </div>
    );
  }

  const regimeChartData = p.per_regime
    .filter((r) => r.n_steps > 0)
    .map((r) => ({
      name: `R${r.regime_label}`,
      sharpe: parseFloat(r.sharpe.toFixed(2)),
      color: REGIME_COLORS[r.regime_label] ?? "#666",
      fullName: regimeNames[r.regime_label] ?? `Regime ${r.regime_label}`,
    }));

  const totalReturnPct = (p.total_return * 100).toFixed(1);
  const bhReturnPct = (p.buy_and_hold.total_return * 100).toFixed(1);
  const ddColor = p.max_drawdown > 0.2 ? "#ff3737" : p.max_drawdown > 0.1 ? "#ff8c00" : "#00ff7f";

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header flex items-center justify-between">
        <span>Portfolio Performance</span>
        <span className="text-[10px] text-[#333]">BTC val 2021</span>
      </div>

      <div className="flex-1 flex flex-col gap-0 overflow-auto">
        {/* 2×2 stat grid */}
        <div className="grid grid-cols-2 shrink-0">
          <StatCell
            label="Sharpe"
            value={p.sharpe.toFixed(3)}
            color={sharpeColor(p.sharpe)}
            sub={`B&H: ${p.buy_and_hold.sharpe.toFixed(3)}`}
            borderRight borderBottom
          />
          <StatCell
            label="Max Drawdown"
            value={`-${(p.max_drawdown * 100).toFixed(1)}%`}
            color={ddColor}
            sub={`B&H: -${(p.buy_and_hold.max_drawdown * 100).toFixed(0)}%`}
            borderBottom
          />
          <StatCell
            label="Total Return"
            value={`+${totalReturnPct}%`}
            color="#4a9eff"
            sub={`B&H: +${bhReturnPct}%`}
            borderRight
          />
          <StatCell
            label="Calmar"
            value={p.calmar.toFixed(2)}
            color={sharpeColor(p.calmar * 0.5)}
          />
        </div>

        {/* Secondary metrics */}
        <div className="grid grid-cols-3 gap-0 text-[11px] border-t border-[#1a1a1a] shrink-0">
          <div className="px-3 py-2 border-r border-[#1a1a1a]">
            <div className="stat-label">Trades</div>
            <div className="text-[#888] font-mono mt-0.5">{p.n_trades}</div>
          </div>
          <div className="px-3 py-2 border-r border-[#1a1a1a]">
            <div className="stat-label">Val Steps</div>
            <div className="text-[#888] font-mono mt-0.5">{p.n_val_steps.toLocaleString()}</div>
          </div>
          <div className="px-3 py-2">
            <div className="stat-label">Portfolio</div>
            <div className="text-[#888] font-mono mt-0.5">${p.final_portfolio_value.toLocaleString("en", { maximumFractionDigits: 0 })}</div>
          </div>
        </div>

        {/* Per-regime Sharpe */}
        {regimeChartData.length > 0 && (
          <div className="border-t border-[#1a1a1a] px-3 pt-2 pb-1 shrink-0">
            <div className="stat-label mb-2">Sharpe by Regime</div>
            <div style={{ height: 72 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regimeChartData} margin={{ top: 0, right: 0, bottom: 0, left: -20 }} barSize={16}>
                  <XAxis dataKey="name" tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#444", fontSize: 9 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="bg-[#111] border border-[#222] rounded p-1.5 text-[10px]">
                          <div style={{ color: d.color }}>{d.fullName}</div>
                          <div className="text-[#888]">Sharpe: {d.sharpe}</div>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="sharpe" radius={[2, 2, 0, 0]}>
                    {regimeChartData.map((entry, i) => (
                      <Cell key={i} fill={entry.sharpe >= 0 ? entry.color : "#ff373744"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Ablation note */}
        <div className="border-t border-[#1a1a1a] px-3 py-2 text-[10px] text-[#333] leading-relaxed shrink-0">
          Regime ablation: Sharpe 1.809 (full) vs 0.000 (no-regime). Bootstrap 95% CI: [-0.300, 3.937].{" "}
          <span className="text-[#3a3a3a]">ppo_best.zip = Phase 2 BTC+ETH.</span>
        </div>
      </div>
    </div>
  );
}
