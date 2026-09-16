"use client";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine, Legend,
} from "recharts";
import type { RLResults, EquityCurve } from "@/lib/types";

interface Props {
  rlResults: RLResults | null;
  equityCurve: EquityCurve | null;
}

const VARIANT_COLORS: Record<string, string> = {
  "Buy & Hold": "#4a9eff",
  "PPO (no regime)": "#ffd700",
  "Agents (no regime)": "#b07fff",
  "PPO+Agents (no regime)": "#ff8c00",
  "MARTA Full": "#00ff7f",
};

function MetricBadge({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-[10px] text-[#444] uppercase tracking-wider">{label}</div>
      <div className="text-base font-bold font-mono" style={{ color: color ?? "#e0e0e0" }}>{value}</div>
      {sub && <div className="text-[10px] text-[#555]">{sub}</div>}
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] text-[#444] uppercase tracking-widest mb-2">{children}</div>;
}

// ── Equity Curve ──────────────────────────────────────────────────────────────

function EquityCurveChart({ curve }: { curve: EquityCurve }) {
  const step = Math.max(1, Math.floor(curve.timestamps.length / 200));
  const chartData = curve.timestamps
    .filter((_, i) => i % step === 0)
    .map((ts, i) => ({
      ts: new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      marta: curve.marta[i * step],
      bh: curve.buy_hold[i * step],
    }));

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <SectionTitle>Portfolio Equity Curve — BTC Val 2021</SectionTitle>
        {curve.synthetic && (
          <span className="text-[10px] text-[#555] italic">approximated — run generate_equity_curve.py for real</span>
        )}
      </div>
      <div style={{ height: 140 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
            <XAxis dataKey="ts" tick={{ fill: "#444", fontSize: 9 }} axisLine={false} tickLine={false} interval={Math.floor(chartData.length / 5)} />
            <YAxis
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
              tick={{ fill: "#444", fontSize: 9 }}
              axisLine={false} tickLine={false} width={44}
            />
            <Tooltip
              contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: 4, fontSize: 11 }}
              formatter={(v: number, name: string) => [`$${v.toLocaleString("en", { maximumFractionDigits: 0 })}`, name === "marta" ? "MARTA" : "Buy & Hold"]}
            />
            <Line type="monotone" dataKey="marta" stroke="#00ff7f" strokeWidth={1.5} dot={false} name="MARTA" />
            <Line type="monotone" dataKey="bh" stroke="#4a9eff" strokeWidth={1} dot={false} strokeDasharray="4 4" name="B&H" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-6 mt-1 text-[11px]">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-[#00ff7f]" />
          <span className="text-[#666]">MARTA: +{curve.marta_final_return}% / DD {curve.marta_max_dd}%</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-[#4a9eff] border-t border-dashed" />
          <span className="text-[#666]">Buy & Hold: +{curve.bh_final_return}% / DD {curve.bh_max_dd}%</span>
        </div>
      </div>
    </div>
  );
}

// ── Ablation ──────────────────────────────────────────────────────────────────

function AblationChart({ rlResults }: { rlResults: RLResults }) {
  const data = rlResults.ablation_chart;
  if (!data.length) return null;

  return (
    <div>
      <SectionTitle>Ablation — Sharpe Ratio by Variant</SectionTitle>
      <div style={{ height: 110 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -20 }} barSize={22}>
            <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
            <XAxis
              dataKey="variant"
              tick={{ fill: "#555", fontSize: 9 }}
              axisLine={false} tickLine={false}
              tickFormatter={(v: string) => v.replace(" (no regime)", "").replace("PPO+Agents", "PPO+A")}
            />
            <YAxis tick={{ fill: "#444", fontSize: 9 }} axisLine={false} tickLine={false} width={28} />
            <Tooltip
              contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: 4, fontSize: 11 }}
              formatter={(v: number) => [v.toFixed(3), "Sharpe"]}
            />
            <ReferenceLine y={0} stroke="#333" />
            <Bar dataKey="sharpe" radius={[2, 2, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={VARIANT_COLORS[entry.variant] ?? "#666"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {/* Bootstrap CI table */}
      {rlResults.bootstrap_ci.length > 0 && (
        <div className="mt-2 text-[10px]">
          <div className="text-[#444] mb-1">Bootstrap 95% CI (block bootstrap, 168h blocks)</div>
          <div className="grid gap-0.5">
            {rlResults.bootstrap_ci.map((ci, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-[#555] w-28 shrink-0">{ci.comparison}</span>
                <span className="text-[#e0e0e0] font-mono w-12">Δ{ci.delta_sharpe > 0 ? "+" : ""}{ci.delta_sharpe.toFixed(3)}</span>
                <span className="text-[#444] font-mono">[{ci.ci_lower_95.toFixed(2)}, {ci.ci_upper_95.toFixed(2)}]</span>
                <span className={`ml-auto text-[10px] ${ci.significant ? "text-[#00ff7f]" : "text-[#555]"}`}>
                  {ci.significant ? "significant" : "not sig."}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Curriculum ────────────────────────────────────────────────────────────────

function CurriculumTable({ rlResults }: { rlResults: RLResults }) {
  if (!rlResults.curriculum.length) return null;
  return (
    <div>
      <SectionTitle>Training Curriculum</SectionTitle>
      <div className="text-[10px] grid gap-1">
        {rlResults.curriculum.map((p, i) => (
          <div key={i} className="flex items-center gap-2 rounded px-2 py-1" style={{ background: "rgba(255,255,255,0.02)" }}>
            <span className="text-[#555] w-12">Phase {p.phase}</span>
            <span className="text-[#777] flex-1">{p.assets}</span>
            <span className="text-[#555]">{p.steps.toLocaleString()} steps</span>
            <span className="font-mono" style={{ color: p.sharpe >= p.threshold ? "#00ff7f" : "#ff8c00" }}>
              Sharpe {p.sharpe.toFixed(3)}
            </span>
            <span className={`px-1 rounded text-[9px] ${p.met ? "text-[#00ff7f] bg-[#00ff7f18]" : "text-[#ff8c00] bg-[#ff8c0018]"}`}>
              {p.met ? "✓ gate" : "collapsed"}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-1.5 text-[10px] text-[#444] leading-relaxed border-l-2 border-[#ff8c00] pl-2">
        Curriculum collapse at Phase 3+: forex assets have zero intraday variance (daily bars upsampled to hourly).
        PPO converges to always-hold on those assets. <span className="text-[#555]">ppo_best.zip = Phase 2 BTC+ETH checkpoint.</span>
      </div>
    </div>
  );
}

// ── Key findings ──────────────────────────────────────────────────────────────

function ResearchFindings({ rlResults }: { rlResults: RLResults }) {
  const marta = (rlResults.evaluation as Record<string, Record<string, number>>)?.marta_ppo ?? {};
  const bh = ((rlResults.evaluation as Record<string, Record<string, Record<string, number>>>)?.baselines ?? {})?.buy_and_hold_btc ?? {};

  return (
    <div>
      <SectionTitle>Key Research Findings</SectionTitle>
      <div className="grid grid-cols-2 gap-2 mb-3">
        <MetricBadge
          label="MARTA Sharpe"
          value={marta.sharpe?.toFixed(3) ?? "—"}
          sub={`B&H: ${bh.sharpe?.toFixed(3) ?? "—"}`}
          color={marta.sharpe >= 1 ? "#00ff7f" : "#ffd700"}
        />
        <MetricBadge
          label="Max Drawdown"
          value={`-${((marta.max_drawdown ?? 0) * 100).toFixed(1)}%`}
          sub={`B&H: -${((bh.max_drawdown ?? 0) * 100).toFixed(0)}%`}
          color={(marta.max_drawdown ?? 0) < 0.3 ? "#00ff7f" : "#ff8c00"}
        />
        <MetricBadge
          label="Total Return"
          value={`+${((marta.total_return ?? 0) * 100).toFixed(1)}%`}
          sub={`B&H: +${((bh.total_return ?? 0) * 100).toFixed(1)}%`}
          color="#4a9eff"
        />
        <MetricBadge
          label="Calmar"
          value={marta.calmar?.toFixed(2) ?? "—"}
          sub="annualised return / max DD"
          color="#b07fff"
        />
      </div>
      <div className="text-[10px] leading-relaxed space-y-1.5">
        <div className="flex gap-2">
          <span className="text-[#00ff7f] shrink-0">●</span>
          <span className="text-[#666]">Regime embedding is <span className="text-[#e0e0e0]">load-bearing</span>: zeroing the 128-dim slot collapses the policy to always-flat (Sharpe 0.000).</span>
        </div>
        <div className="flex gap-2">
          <span className="text-[#4a9eff] shrink-0">●</span>
          <span className="text-[#666]">MARTA wins on <span className="text-[#e0e0e0]">risk-adjusted</span> return and drawdown; gives up absolute return by running low exposure (avg position 2.6%).</span>
        </div>
        <div className="flex gap-2">
          <span className="text-[#ffd700] shrink-0">●</span>
          <span className="text-[#666]"><span className="text-[#e0e0e0]">Bootstrap 95% CI</span> spans negative values — qualitatively strong but not yet statistically significant on one val path.</span>
        </div>
        <div className="flex gap-2">
          <span className="text-[#b07fff] shrink-0">●</span>
          <span className="text-[#666]">State: [128-dim regime | 16-dim consensus | 8-dim portfolio] · Reward: ΔPnL/CVaR − λ×drawdown · HMM n_states=7 (BIC).</span>
        </div>
      </div>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function BacktestPanel({ rlResults, equityCurve }: Props) {
  if (!rlResults) {
    return (
      <div className="h-full flex items-center justify-center text-[#333] text-[11px]">
        Loading RL results…
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-3 space-y-5">
      {equityCurve && <EquityCurveChart curve={equityCurve} />}
      <AblationChart rlResults={rlResults} />
      <CurriculumTable rlResults={rlResults} />
      <ResearchFindings rlResults={rlResults} />
    </div>
  );
}
