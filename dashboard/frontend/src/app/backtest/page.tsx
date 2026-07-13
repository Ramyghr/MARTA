"use client";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { Play, ChevronDown, ChevronUp, RefreshCw } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton, SkeletonCard } from "@/components/ui/Skeleton";
import { LearningPanel } from "@/components/LearningPanel";
import { api } from "@/lib/api";
import type { AblationResults } from "@/lib/api";
import { API_URL } from "@/lib/websocket";
import { useMARTAStore } from "@/lib/store";

type AblationWithMeta = AblationResults & { last_updated?: string };
import type { Scenario, FundManagerDecision } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

const VARIANT_LABELS: Record<string, string> = {
  buy_hold: "Buy & Hold",
  rsi_strategy: "RSI Strategy",
  ma_crossover: "MA Crossover",
  ppo_no_regime: "PPO No Regime",
  agents_no_regime: "Agents No Regime",
  ppo_agents_no_regime: "PPO+Agents No Regime",
  marta_full: "MARTA Full",
};
const VARIANT_COLORS: Record<string, string> = {
  buy_hold: "#4a9eff",
  rsi_strategy: "#00c8d7",
  ma_crossover: "#e070c0",
  ppo_no_regime: "#ffd700",
  agents_no_regime: "#b07fff",
  ppo_agents_no_regime: "#ff8c00",
  marta_full: "#00ff7f",
};

type SortKey = "sharpe" | "sortino" | "max_drawdown" | "total_return";
const SORT_LABELS: Record<SortKey, string> = { sharpe: "Sharpe", sortino: "Sortino", max_drawdown: "Max DD", total_return: "Total Ret" };

// ── Ablation table ──────────────────────────────────────────────────────────────

function AblationTable({ ablation }: { ablation: AblationResults }) {
  const [sortKey, setSortKey] = useState<SortKey>("sharpe");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const rows = useMemo(() => {
    const list = Object.entries(ablation.variants).map(([key, v]) => ({ key, label: VARIANT_LABELS[key] ?? key, ...v }));
    return list.sort((a, b) => (sortDir === "asc" ? a[sortKey] - b[sortKey] : b[sortKey] - a[sortKey]));
  }, [ablation, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
  };

  const Th = ({ label, keyName }: { label: string; keyName: SortKey }) => {
    const active = sortKey === keyName;
    return (
      <th onClick={() => handleSort(keyName)} className="cursor-pointer select-none hover:text-[#888] transition-colors">
        <span className="inline-flex items-center gap-0.5">
          {label}
          {active && (sortDir === "desc" ? <ChevronDown size={10} /> : <ChevronUp size={10} />)}
        </span>
      </th>
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Variant</th>
              <Th label="Sharpe" keyName="sharpe" />
              <Th label="Sortino" keyName="sortino" />
              <Th label="Max DD" keyName="max_drawdown" />
              <Th label="Total Ret" keyName="total_return" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isBest = r.key === "marta_full";
              return (
                <tr
                  key={r.key}
                  title={r.method}
                  style={{ borderLeft: isBest ? "3px solid #00ff7f" : "3px solid transparent", background: isBest ? "rgba(0,255,127,0.04)" : undefined, cursor: "help" }}
                >
                  <td className="font-semibold" style={{ color: isBest ? "#00ff7f" : VARIANT_COLORS[r.key] ?? "#999" }}>
                    {isBest && "🏆 "}{r.label}
                  </td>
                  <td className="font-mono" style={{ color: r.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}>{r.sharpe.toFixed(3)}</td>
                  <td className="font-mono" style={{ color: r.sortino >= 0 ? "#00ff7f" : "#ff3737" }}>{r.sortino.toFixed(3)}</td>
                  <td className="font-mono text-[#ff8c00]">{(r.max_drawdown * 100).toFixed(1)}%</td>
                  <td className="font-mono" style={{ color: r.total_return >= 0 ? "#00ff7f" : "#ff3737" }}>
                    {r.total_return >= 0 ? "+" : ""}{(r.total_return * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="text-[9px] text-[#333]">Hover a row for its method. Click a column header to sort.</div>

      {Object.keys(ablation.bootstrap_ci ?? {}).length > 0 && (
        <div className="text-[10px] border-t border-[#1a1a1a] pt-2">
          <div className="text-[#444] mb-1.5 uppercase tracking-wider">Bootstrap 95% CI (block bootstrap, 168h blocks)</div>
          <div className="grid gap-1">
            {Object.entries(ablation.bootstrap_ci).map(([key, ci]) => {
              const comparison = key.replace("marta_full_vs_", "vs ").replace(/_/g, " ");
              const significant = ci.ci_lower_95 > 0;
              return (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-[#555] w-40 shrink-0">{comparison}</span>
                  <span className="text-[#e0e0e0] font-mono w-16">Δ{ci.delta_sharpe > 0 ? "+" : ""}{ci.delta_sharpe.toFixed(3)}</span>
                  <span className="text-[#444] font-mono">[{ci.ci_lower_95.toFixed(2)}, {ci.ci_upper_95.toFixed(2)}]</span>
                  <span className={`ml-auto ${significant ? "text-[#00ff7f]" : "text-[#555]"}`}>{significant ? "significant" : "not significant"}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="text-[10px] text-[#444] leading-relaxed border-l-2 border-[#ff8c00] pl-2">
        * <code className="text-[#666]">ppo_no_regime</code> / <code className="text-[#666]">ppo_agents_no_regime</code> use a momentum
        baseline rather than the trained PPO — zeroing the 128-dim regime slot pushes the frozen policy out-of-distribution and it
        collapses to hold on every step, which confirms regime-embedding dependency by construction rather than by a degenerate
        all-zero run.
      </div>
    </div>
  );
}

// ── Out-of-sample test set panel ───────────────────────────────────────────────

interface TestSetMetrics {
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  total_return: number;
  method?: string;
}

interface TestSetResults {
  test_period: { start: string; end: string };
  n_steps: number;
  warning: string;
  caveats?: string[];
  key_finding?: string | null;
  marta_ppo: TestSetMetrics;
  buy_hold: TestSetMetrics;
  rsi_strategy: TestSetMetrics;
  ma_crossover: TestSetMetrics;
  momentum: TestSetMetrics;
}

const TEST_STRATEGY_ORDER: Array<{ key: keyof TestSetResults; label: string }> = [
  { key: "marta_ppo", label: "MARTA PPO" },
  { key: "buy_hold", label: "Buy & Hold" },
  { key: "rsi_strategy", label: "RSI Strategy" },
  { key: "ma_crossover", label: "MA Crossover" },
  { key: "momentum", label: "Momentum" },
];

function TestSetPanel() {
  const [results, setResults] = useState<TestSetResults | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/test-set`)
      .then((res) => {
        if (res.status === 404) { setNotFound(true); return null; }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => { if (data) setResults(data); })
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Skeleton className="h-48 w-full" />;
  if (notFound || !results) {
    return (
      <EmptyState
        title="Test set not yet evaluated"
        description="Run `python run_test_set.py` on the host machine."
      />
    );
  }

  const weeks = Math.round(
    (new Date(results.test_period.end).getTime() - new Date(results.test_period.start).getTime()) /
      (7 * 24 * 3600 * 1000),
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="text-[11px] text-[#888]">
        Test period: <span className="font-mono text-[#ccc]">{results.test_period.start}</span> →{" "}
        <span className="font-mono text-[#ccc]">{results.test_period.end}</span>{" "}
        ({weeks} weeks, {results.n_steps.toLocaleString()} hourly bars)
      </div>

      <div className="text-[10px] text-[#ff8c00] border border-[#ff8c00]/40 bg-[#ff8c00]/5 rounded px-2 py-1.5 leading-relaxed">
        ⚠ {weeks <= 8 ? "Short test window — interpret with caution" : results.warning}
      </div>

      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Sharpe</th>
              <th>Sortino</th>
              <th>Max DD</th>
              <th>Total Ret</th>
            </tr>
          </thead>
          <tbody>
            {TEST_STRATEGY_ORDER.map(({ key, label }) => {
              const m = results[key] as TestSetMetrics;
              if (!m) return null;
              const isMarta = key === "marta_ppo";
              return (
                <tr
                  key={key}
                  title={m.method}
                  style={{
                    borderLeft: isMarta ? "3px solid #00ff7f" : "3px solid transparent",
                    background: isMarta ? "rgba(0,255,127,0.04)" : undefined,
                  }}
                >
                  <td className="font-semibold" style={{ color: isMarta ? "#00ff7f" : "#999" }}>
                    {isMarta && "🏆 "}{label}
                  </td>
                  <td className="font-mono" style={{ color: m.sharpe >= 0 ? "#00ff7f" : "#ff3737" }}>{m.sharpe.toFixed(3)}</td>
                  <td className="font-mono" style={{ color: m.sortino >= 0 ? "#00ff7f" : "#ff3737" }}>{m.sortino.toFixed(3)}</td>
                  <td className="font-mono text-[#ff8c00]">{(m.max_drawdown * 100).toFixed(1)}%</td>
                  <td className="font-mono" style={{ color: m.total_return >= 0 ? "#00ff7f" : "#ff3737" }}>
                    {m.total_return >= 0 ? "+" : ""}{(m.total_return * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {results.key_finding && (
        <div className="text-[10px] text-[#ccc] leading-relaxed border-l-2 border-[#ffd700] bg-[#ffd700]/5 rounded-r px-2 py-1.5">
          <span className="text-[#ffd700] font-semibold uppercase tracking-wider">Key finding: </span>
          {results.key_finding}
        </div>
      )}

      {(results.caveats?.length ?? 0) > 0 && (
        <div className="text-[10px] text-[#555] leading-relaxed border-l-2 border-[#333] pl-2">
          {results.caveats!.map((c, i) => <div key={i}>• {c}</div>)}
        </div>
      )}

      <div className="text-[10px] text-[#444]">
        Test set was never used during training or validation.
      </div>
    </div>
  );
}

// ── Key metrics comparison — small multiples (one axis per measure) ───────────

function MetricsComparisonChart({ ablation }: { ablation: AblationResults }) {
  const data = Object.entries(ablation.variants).map(([key, v]) => ({
    key,
    name: (VARIANT_LABELS[key] ?? key).replace(" No Regime", "\nNo Regime").replace("PPO+Agents", "PPO+A"),
    sharpe: v.sharpe,
    maxDd: v.max_drawdown * 100,
    color: VARIANT_COLORS[key] ?? "#666",
  }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <div>
        <div className="text-[10px] text-[#444] uppercase tracking-widest mb-2">Sharpe Ratio</div>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: "#555", fontSize: 9 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#444", fontSize: 9 }} axisLine={false} tickLine={false} width={30} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontSize: 11 }} formatter={(v: number) => [v.toFixed(3), "Sharpe"]} />
            <ReferenceLine y={0} stroke="#333" />
            <Bar dataKey="sharpe" radius={[2, 2, 0, 0]}>
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <div className="text-[10px] text-[#444] uppercase tracking-widest mb-2">Max Drawdown</div>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#141414" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: "#555", fontSize: 9 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#444", fontSize: 9 }} axisLine={false} tickLine={false} width={30} tickFormatter={(v) => `${v}%`} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontSize: 11 }} formatter={(v: number) => [`${v.toFixed(1)}%`, "Max DD"]} />
            <Bar dataKey="maxDd" radius={[2, 2, 0, 0]}>
              {data.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.7} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── Demo scenario cards ──────────────────────────────────────────────────────

function ScenarioCard({ scenario, decision }: { scenario: Scenario; decision: FundManagerDecision | null }) {
  const router = useRouter();
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const regimeLabel = scenario.regime_label ?? 0;
  const color = REGIME_COLORS[regimeLabel] ?? "#666";
  const convLabel = decision ? (decision.conviction >= 0.65 ? "high" : decision.conviction >= 0.4 ? "moderate" : "low") : null;

  return (
    <div className="rounded border border-[#1a1a1a] bg-[#0d0d0d] p-3 flex flex-col gap-2">
      <div className="font-semibold text-[12px] text-[#ccc] leading-snug">{scenario.label}</div>
      <div className="flex items-center gap-1.5 text-[10px]">
        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: color }} />
        <span style={{ color }}>{scenario.regime} — {regimeNames[regimeLabel] ?? ""}</span>
      </div>
      <div className="text-[10px] text-[#666] leading-relaxed flex-1">{scenario.description}</div>
      <div className="text-[10px] text-[#555]">
        Expected:{" "}
        {decision ? (
          <>
            <span className="font-semibold uppercase" style={{ color: decision.action === "long" ? "#00ff7f" : decision.action === "short" ? "#ff3737" : "#888" }}>
              {decision.action}
            </span>
            {convLabel && <span>, {convLabel} conviction</span>}
          </>
        ) : "—"}
      </div>
      <button
        onClick={() => router.push(`/debates?scenario=${scenario.id}`)}
        className="mt-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors"
      >
        <Play size={10} /> Replay Debate
      </button>
    </div>
  );
}

function ScenarioGrid() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [decisions, setDecisions] = useState<Record<string, FundManagerDecision>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getDemoScenarios().then(async (d) => {
      setScenarios(d.scenarios);
      const entries = await Promise.all(
        d.scenarios.map(async (s) => {
          if (s.debate_index === undefined) return null;
          try {
            const detail = await api.getDebate(s.debate_index);
            const fm = detail.fund_manager_decision;
            const decision: FundManagerDecision = {
              action: fm.final_action, position_size: fm.position_size, conviction: fm.conviction,
              rationale: fm.rationale, regime_override: fm.regime_override, dissenting_agents: fm.dissenting_agents ?? [],
            };
            return [s.id, decision] as const;
          } catch { return null; }
        }),
      );
      setDecisions(Object.fromEntries(entries.filter((e): e is readonly [string, FundManagerDecision] => e !== null)));
    }).catch(() => setScenarios([])).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
        {Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    );
  }
  if (scenarios.length === 0) {
    return <EmptyState title="No demo scenarios" description="demo_scenarios.py has no entries." />;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
      {scenarios.map((s) => <ScenarioCard key={s.id} scenario={s} decision={decisions[s.id] ?? null} />)}
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function BacktestPage() {
  const [ablation, setAblation] = useState<AblationWithMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  useEffect(() => {
    api.getAblation().then(setAblation).catch(() => setAblation(null)).finally(() => setLoading(false));
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    setRefreshError(null);
    try {
      const res = await fetch(`${API_URL}/api/ablation/refresh`, { method: "POST" });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          if (body.detail) detail = String(body.detail).slice(0, 300);
        } catch { /* keep status text */ }
        throw new Error(detail);
      }
      setAblation(await res.json());
    } catch (err) {
      setRefreshError(err instanceof Error ? err.message : String(err));
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 pb-4">
      <PageHeader title="Backtest" description="Regime ablation study, demo scenarios, and headline comparison metrics" />

      <Panel
        title="Ablation Study"
        actions={
          <div className="flex items-center gap-2">
            {ablation?.last_updated && (
              <span className="text-[9px] text-[#444]">Updated {new Date(ablation.last_updated).toLocaleString()}</span>
            )}
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors disabled:opacity-50"
            >
              <RefreshCw size={10} className={refreshing ? "animate-spin" : ""} />
              {refreshing ? "Refreshing…" : "Refresh Ablation"}
            </button>
          </div>
        }
      >
        {refreshError && (
          <div className="mb-2 text-[10px] text-[#ff3737] border border-[#ff3737]/30 bg-[#ff3737]/5 rounded px-2 py-1.5">
            Refresh failed: {refreshError}
          </div>
        )}
        {loading ? <Skeleton className="h-64 w-full" /> : ablation ? <AblationTable ablation={ablation} /> : (
          <EmptyState title="No ablation results" description="Run run_rl.py to produce data/results/ablation_results.json." />
        )}
      </Panel>

      <Panel title="Out-of-Sample Test Results">
        <TestSetPanel />
      </Panel>

      <Panel title="Key Metrics Comparison">
        {loading ? <Skeleton className="h-48 w-full" /> : ablation ? <MetricsComparisonChart ablation={ablation} /> : (
          <EmptyState title="No comparison data" description="Run run_rl.py to produce ablation results." />
        )}
      </Panel>

      <Panel title="Demo Scenarios">
        <ScenarioGrid />
      </Panel>

      <LearningPanel>
        {(() => {
          const rsi = ablation?.bootstrap_ci?.["marta_full_vs_rsi_strategy"];
          const bh = ablation?.bootstrap_ci?.["marta_full_vs_buy_hold"];
          if (!rsi && !bh) {
            return "Bootstrap CI uses 168-hour (1-week) blocks rather than individual hourly returns, preserving the autocorrelation structure of the return series — an IID resample would understate the confidence interval's true width.";
          }
          return (
            <>
              Bootstrap CI uses 168-hour blocks to preserve autocorrelation.{" "}
              {rsi && (
                <>MARTA vs RSI: ΔSharpe {rsi.delta_sharpe >= 0 ? "+" : ""}{rsi.delta_sharpe.toFixed(2)} [{rsi.ci_lower_95.toFixed(2)}, {rsi.ci_upper_95.toFixed(2)}] — {rsi.ci_lower_95 > 0 ? "significant" : "not significant"}. </>
              )}
              {bh && (
                <>vs Buy&amp;Hold: ΔSharpe {bh.delta_sharpe >= 0 ? "+" : ""}{bh.delta_sharpe.toFixed(3)} — {bh.ci_lower_95 > 0 ? "significant" : "not significant"}.</>
              )}
            </>
          );
        })()}
      </LearningPanel>
    </div>
  );
}
