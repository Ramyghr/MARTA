"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { LineChart, Line, ResponsiveContainer } from "recharts";
import PriceChart from "@/components/PriceChart";
import RegimePanel from "@/components/RegimePanel";
import DecisionPanel from "@/components/DecisionPanel";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatCard } from "@/components/ui/StatCard";
import { ActionBadge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { LearningPanel } from "@/components/LearningPanel";
import { IndicatorsPanel } from "@/components/IndicatorsPanel";
import { MartaWebSocket, fetchApi } from "@/lib/websocket";
import { api } from "@/lib/api";
import { useMARTAStore } from "@/lib/store";
import type { Candle, DataRange, Portfolio, RegimeState, EquityCurve } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { Play, Waves as WavesIcon, ArrowRight } from "lucide-react";

function fmtPct(x: number | undefined) {
  return x === undefined ? "—" : `${(x * 100).toFixed(1)}%`;
}
function fmtNum(x: number | undefined) {
  return x === undefined ? "—" : x.toFixed(3);
}

function assetColorGuess(asset: string): string {
  if (asset.endsWith("_USDT")) return "#00ff7f";
  if (asset === "GOLD") return "#ffd700";
  return "#4a9eff";
}

function actionColor(action?: string): string {
  return action === "long" ? "#00ff7f" : action === "short" ? "#ff3737" : "#666666";
}

function sharpeColor(s: number): string {
  if (s >= 1.5) return "#00ff7f";
  if (s >= 0.5) return "#ffd700";
  if (s > 0) return "#ff8c00";
  return "#ff3737";
}

function EquitySparkline({ curve }: { curve: EquityCurve | null }) {
  if (!curve || curve.timestamps.length === 0) {
    return <div className="h-10 flex items-center justify-center text-[10px] text-[#333]">No equity curve data</div>;
  }
  const step = Math.max(1, Math.floor(curve.timestamps.length / 100));
  const data = curve.marta.filter((_, i) => i % step === 0).map((v) => ({ v }));
  return (
    <ResponsiveContainer width="100%" height={40}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <Line type="monotone" dataKey="v" stroke="#00ff7f" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const mode = useMARTAStore((s) => s.mode);
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const setCurrentRegime = useMARTAStore((s) => s.setCurrentRegime);
  const setWsConnected = useMARTAStore((s) => s.setWsConnected);
  const setWsStatus = useMARTAStore((s) => s.setWsStatus);
  const setWsLastUpdate = useMARTAStore((s) => s.setWsLastUpdate);
  const currentDecision = useMARTAStore((s) => s.currentDecision);
  const setCurrentDecision = useMARTAStore((s) => s.setCurrentDecision);
  const activeDebate = useMARTAStore((s) => s.activeDebate);

  const [candles, setCandles] = useState<Candle[]>([]);
  const [regime, setRegime] = useState<RegimeState | null>(null);
  const [dataRange, setDataRange] = useState<DataRange | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurve | null>(null);
  const [chartRange, setChartRange] = useState<{ start: string; end: string } | null>(null);
  const [lastDebateTimestamp, setLastDebateTimestamp] = useState<string | null>(null);
  const [lastDebateAsset, setLastDebateAsset] = useState<string | null>(null);
  const [noDebatesRecorded, setNoDebatesRecorded] = useState(false);

  // Load the most recent recorded debate for the currently selected asset. Re-runs on
  // every asset switch so Final Decision always reflects BTC vs ETH vs whichever is active
  // — it must never keep showing a stale debate from the previously selected asset.
  useEffect(() => {
    let cancelled = false;
    async function loadLastDecision() {
      setNoDebatesRecorded(false);
      setCurrentDecision(null);
      setLastDebateTimestamp(null);
      setLastDebateAsset(null);
      try {
        const { debates } = await api.getDebates();
        if (cancelled) return;
        const assetDebates = (debates ?? []).filter((d) => d.asset === selectedAsset);
        if (assetDebates.length === 0) {
          setNoDebatesRecorded(true);
          return;
        }
        const sorted = [...assetDebates].sort(
          (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
        );
        const latest = sorted[0];
        const detail = await api.getDebate(latest.id);
        if (cancelled) return;
        const fm = detail.fund_manager_decision;
        if (fm) {
          setCurrentDecision({
            action: fm.final_action,
            position_size: fm.position_size,
            conviction: fm.conviction,
            rationale: fm.rationale,
            regime_override: fm.regime_override,
            dissenting_agents: fm.dissenting_agents ?? [],
          });
          setLastDebateTimestamp(latest.timestamp);
          setLastDebateAsset(latest.asset);
        }
      } catch (err) {
        console.error("Failed to load last decision:", err);
      }
    }
    loadLastDecision();
    return () => { cancelled = true; };
  }, [selectedAsset, setCurrentDecision]);

  useEffect(() => {
    fetchApi<DataRange>(`/api/data-range/${selectedAsset}`).then(setDataRange).catch(() => {});
    setChartRange(null);
  }, [selectedAsset]);

  useEffect(() => {
    api.getPortfolio().then(setPortfolio).catch(() => {});
    fetchApi<EquityCurve>("/api/rl/equity-curve").then(setEquityCurve).catch(() => {});
  }, []);

  const wsRef = useRef<MartaWebSocket | null>(null);
  useEffect(() => {
    // Decide live-vs-historical by the candle's actual age, not by `mode`: the default
    // "historical replay" mode ALSO extends forward to today (historical.py's live
    // extension backfills it from Binance/yfinance), so its last candle is normally just
    // as current as live mode's. Using /at for any recent timestamp was the bug — /at
    // always treats an explicit timestamp as a historical "what was true on this date"
    // lookup (regime_service.py's _lookup_regime), so it silently returned whatever
    // regime_stream.parquet's frozen last row was instead of the real current regime.
    // Only a deliberately-scrubbed-into-the-past range should ever hit /at. Threshold
    // matches regime_service.get_regime's own `stale` cutoff (48h) for consistency.
    const refreshRegimeForTimestamp = (timestamp: string | undefined) => {
      if (!timestamp) return;
      const ageHours = (Date.now() - new Date(timestamp).getTime()) / 3_600_000;
      const req = ageHours < 48
        ? api.getRegime(selectedAsset)
        : fetchApi<RegimeState>(`/api/regime/${selectedAsset}/at?timestamp=${encodeURIComponent(timestamp)}`);
      req.then((r) => { setRegime(r); setCurrentRegime(r); }).catch(() => {});
    };

    // Historical mode with an explicit scrubbed range — fetch once via REST, no WS.
    if (mode === "historical" && chartRange) {
      wsRef.current?.close();
      setWsConnected(false);
      api.getHistoryRange(selectedAsset, chartRange.start, chartRange.end)
        .then((d) => {
          setCandles(d.candles);
          refreshRegimeForTimestamp(d.candles.at(-1)?.timestamp);
          setWsConnected(true);
          setWsStatus("open");
          setWsLastUpdate(Date.now());
        })
        .catch(() => {});
      return;
    }

    wsRef.current?.close();
    setWsConnected(false);

    const path = mode === "live" ? `/ws/price/${selectedAsset}?mode=live` : `/ws/price/${selectedAsset}`;
    const ws = new MartaWebSocket(path);
    const off = ws.on((msg) => {
      setWsLastUpdate(Date.now());
      if (msg.type === "history") {
        const hist = msg.candles as Candle[];
        setCandles(hist);
        refreshRegimeForTimestamp(hist.at(-1)?.timestamp);
        setWsConnected(true);
      } else if (msg.type === "candle") {
        const candle = msg as unknown as Candle;
        setCandles((prev) => [...prev.slice(-599), candle]);
        // A new live candle just arrived — refresh the regime badge too, so it keeps
        // changing over time instead of being set once on connect and never again.
        refreshRegimeForTimestamp(candle.timestamp);
      }
    });
    const offStatus = ws.onStatus((status) => {
      setWsStatus(status);
      setWsConnected(status === "open");
    });
    ws.connect();
    wsRef.current = ws;
    return () => { off(); offStatus(); ws.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAsset, mode, chartRange]);

  const handleRunDebate = useCallback(() => {
    router.push("/debates?autorun=1");
  }, [router]);

  // activeDebate can be left over from a previously selected asset (e.g. viewed on the
  // Debates page before switching here) — only trust it when it actually matches the
  // asset currently selected, otherwise it must not leak into the Final Decision panel.
  const activeDebateForAsset = activeDebate?.asset === selectedAsset ? activeDebate : null;
  const decision = currentDecision ?? activeDebateForAsset?.decision ?? null;

  // Unify the two timestamp sources (a live/replayed debate vs. the seeded last-recorded
  // decision) into one age computation, so "6 hours ago" is never silently wrong because
  // it was reading whichever field happened to be set.
  const decisionTimestamp = activeDebateForAsset?.timestamp ?? lastDebateTimestamp;
  const decisionAsset = activeDebateForAsset?.asset ?? lastDebateAsset;
  const decisionHoursAgo = decisionTimestamp
    ? Math.round((Date.now() - new Date(decisionTimestamp).getTime()) / 3_600_000)
    : null;

  return (
    <div className="flex flex-col h-full min-h-0">
      <PageHeader
        title="Overview"
        description={`${selectedAsset.replace("_", "/")} — ${mode === "live" ? "live feed" : "historical replay"}`}
        actions={
          <>
            <button
              onClick={handleRunDebate}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors"
            >
              <Play size={12} /> Run New Debate
            </button>
            <button
              onClick={() => router.push("/regime")}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors"
            >
              <WavesIcon size={12} /> Regime History
            </button>
          </>
        }
      />

      {/* Last debate summary strip */}
      {decision && (
        <div className="mb-2 px-3 py-2 rounded border border-[#1a1a1a] bg-[#0d0d0d] flex items-center gap-3 text-[11px] shrink-0">
          <span className="text-[#444] uppercase tracking-wider">Last Decision</span>
          <ActionBadge action={decision.action} />
          <span className="text-[#666]">conviction {(decision.conviction * 100).toFixed(0)}%</span>
          {decisionHoursAgo != null && decisionHoursAgo > 24 && (
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase tracking-wide"
              style={{ background: "rgba(255,140,0,0.15)", color: "#ff8c00" }}
            >
              ⚠ Stale — run new debate
            </span>
          )}
          {decisionTimestamp && (
            <span className="text-[#333] ml-auto">
              {new Date(decisionTimestamp).toLocaleString()}
              {decisionAsset && ` · ${decisionAsset.replace("_", "/")}`}
              {decisionHoursAgo != null && (
                <span style={{ color: decisionHoursAgo > 24 ? "#ff8c00" : "#555" }}> · {decisionHoursAgo}h ago</span>
              )}
            </span>
          )}
        </div>
      )}

      {/* Top row — live metrics bar */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 mb-2 shrink-0">
        <StatCard
          label="Current Regime"
          value={regime ? `R${regime.dominant_regime} · ${regimeNames[regime.dominant_regime] ?? ""}` : "—"}
          color={regime ? REGIME_COLORS[regime.dominant_regime] : undefined}
        />
        <StatCard
          label="Last Action"
          value={decision ? decision.action.toUpperCase() : "—"}
          color={decision ? actionColor(decision.action) : undefined}
        />
        <StatCard
          label="Portfolio Sharpe"
          value={portfolio ? portfolio.sharpe.toFixed(3) : "—"}
          color={portfolio ? sharpeColor(portfolio.sharpe) : undefined}
        />
        <StatCard
          label="Active Asset"
          value={selectedAsset.replace("_", "/")}
          color={assetColorGuess(selectedAsset)}
        />
      </div>

      <div className="flex-1 min-h-0 grid grid-rows-[minmax(0,2fr)_minmax(0,1.2fr)] gap-2">
        {/* Row 1: Price Chart (2/3) + Regime Panel (1/3) */}
        <div className="grid grid-cols-[2fr_1fr] gap-2 min-h-0">
          <PriceChart
            candles={candles}
            asset={selectedAsset}
            dataRange={dataRange}
            onRangeChange={mode === "historical" ? setChartRange : () => {}}
            liveMode={mode === "live"}
          />
          <RegimePanel regime={regime} />
        </div>

        {/* Row 2: Decision Panel (1/4) + Portfolio mini-stats (3/4) */}
        <div className="grid grid-cols-[1fr_3fr] gap-2 min-h-0">
          <div className="flex flex-col min-h-0">
            {!decision && noDebatesRecorded ? (
              <div className="panel h-full flex flex-col">
                <div className="panel-header">Final Decision</div>
                <div className="flex-1 flex items-center justify-center text-[#333] text-[11px] text-center px-3">
                  No debate recorded for {selectedAsset.replace("_", "/")} — run a debate to see a decision.
                </div>
              </div>
            ) : (
              <>
                <div className="flex-1 min-h-0">
                  <DecisionPanel decision={decision} timestamp={decisionTimestamp} />
                </div>
              </>
            )}
          </div>
          <Panel
            title="Portfolio Snapshot"
            actions={
              <Link href="/portfolio" className="flex items-center gap-1 text-[10px] text-[#4a9eff] hover:text-[#6db4ff] transition-colors">
                View Full Portfolio <ArrowRight size={11} />
              </Link>
            }
          >
            {portfolio ? (
              <div className="flex flex-col gap-3 h-full">
                <div className="grid grid-cols-2 gap-2 shrink-0">
                  <StatCard label="Sharpe" value={fmtNum(portfolio.sharpe)} color={sharpeColor(portfolio.sharpe)} />
                  <StatCard label="Max Drawdown" value={fmtPct(portfolio.max_drawdown)} color="#ff8c00" />
                  <StatCard label="Total Return" value={fmtPct(portfolio.total_return)} color={portfolio.total_return >= 0 ? "#00ff7f" : "#ff3737"} />
                  <StatCard label="Calmar" value={fmtNum(portfolio.calmar)} />
                </div>
                <div className="flex-1 min-h-0 flex flex-col">
                  <div className="stat-label mb-1">Equity Curve</div>
                  <EquitySparkline curve={equityCurve} />
                </div>
              </div>
            ) : (
              <EmptyState title="No portfolio data" description="Run the RL pipeline to generate evaluation results." />
            )}
          </Panel>
        </div>
      </div>

      <Panel
        title="Indicator Values"
        className="mt-2 h-56 shrink-0"
        actions={
          <Link href="/settings" className="flex items-center gap-1 text-[10px] text-[#4a9eff] hover:text-[#6db4ff] transition-colors">
            Configure <ArrowRight size={11} />
          </Link>
        }
      >
        <IndicatorsPanel asset={selectedAsset} />
      </Panel>

      <LearningPanel>
        The regime encoder compresses 60 hours × 14 features into a 128-dim fingerprint. The HMM
        found 7 market states from that embedding space. Current state:{" "}
        {regime ? (
          <>R{regime.dominant_regime} ({regimeNames[regime.dominant_regime] ?? "Unknown"})</>
        ) : (
          "no regime loaded yet"
        )}.
      </LearningPanel>
    </div>
  );
}
