"use client";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Play, Search, X } from "lucide-react";
import DebateStream from "@/components/DebateStream";
import NewsPanel from "@/components/NewsPanel";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { ActionBadge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonRows } from "@/components/ui/Skeleton";
import { LearningPanel } from "@/components/LearningPanel";
import { api } from "@/lib/api";
import type { DebateListItem, DebateDetail, RawAgentThesis, RawAgentRebuttal } from "@/lib/api";
import { useMARTAStore } from "@/lib/store";
import type {
  AgentThesis, AgentRebuttal, FundManagerDecision, RiskLimits,
  AssetInfo, RegimeState, Scenario, DataRange, DebateMessage,
} from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

// ── Raw stored-debate → frontend shape mappers (mirrors backend debate_stream.py) ──

function mapThesis(t: RawAgentThesis): AgentThesis {
  return { agent: t.agent_name, bias: t.directional_bias, confidence: t.confidence, reasoning: t.reasoning, key_factors: t.key_factors ?? [], regime_alignment: t.regime_alignment };
}
function mapRebuttal(r: RawAgentRebuttal): AgentRebuttal {
  return { agent: r.agent_name, target: r.target_agent, rebuttal_point: r.rebuttal_point, maintains_bias: r.maintains_bias, updated_confidence: r.updated_confidence };
}
function mapDecision(fm: DebateDetail["fund_manager_decision"]): FundManagerDecision {
  return { action: fm.final_action, position_size: fm.position_size, conviction: fm.conviction, rationale: fm.rationale, regime_override: fm.regime_override, dissenting_agents: fm.dissenting_agents ?? [] };
}
function mapRisk(r: DebateDetail["risk_limits"]): RiskLimits {
  return { max_position_size: r.max_position_size, volatility_forecast: r.volatility_forecast, cvar_95: r.cvar_95, risk_regime: r.risk_regime };
}

interface DebateView {
  asset: string;
  timestamp: string;
  regimeLabel: number;
  theses: AgentThesis[];
  rebuttals: AgentRebuttal[];
  decision: FundManagerDecision | null;
  totalTokens?: number;
  durationSeconds?: number;
}

// ── Left panel — debate browser ────────────────────────────────────────────────

function DebateBrowser({ debates, selectedId, onSelect, onRunNew }: {
  debates: DebateListItem[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onRunNew: () => void;
}) {
  const [search, setSearch] = useState("");
  const [assetFilter, setAssetFilter] = useState("");
  const [regimeFilter, setRegimeFilter] = useState("");
  const [actionFilter, setActionFilter] = useState("");

  const assets = useMemo(() => Array.from(new Set(debates.map((d) => d.asset))).sort(), [debates]);
  const regimes = useMemo(() => Array.from(new Set(debates.map((d) => d.regime_label))).sort((a, b) => a - b), [debates]);

  const filtered = useMemo(() => debates.filter((d) => {
    if (assetFilter && d.asset !== assetFilter) return false;
    if (regimeFilter && String(d.regime_label) !== regimeFilter) return false;
    if (actionFilter && d.action !== actionFilter) return false;
    if (search && !`${d.asset} ${d.scenario}`.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  }), [debates, search, assetFilter, regimeFilter, actionFilter]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="p-2 border-b border-[#1a1a1a] shrink-0 flex flex-col gap-2">
        <button
          onClick={onRunNew}
          className="flex items-center justify-center gap-1.5 w-full px-3 py-2 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors"
        >
          <Play size={12} /> Run New Debate
        </button>
        <div className="relative">
          <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-[#444]" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search asset or scenario…"
            className="w-full bg-[#111] border border-[#1a1a1a] rounded pl-7 pr-2 py-1.5 text-[11px] text-[#999] outline-none"
          />
        </div>
        <div className="flex gap-1.5">
          <select value={assetFilter} onChange={(e) => setAssetFilter(e.target.value)} className="flex-1 min-w-0 bg-[#111] border border-[#1a1a1a] rounded px-1 py-1 text-[10px] text-[#888]">
            <option value="">All assets</option>
            {assets.map((a) => <option key={a} value={a}>{a.replace("_", "/")}</option>)}
          </select>
          <select value={regimeFilter} onChange={(e) => setRegimeFilter(e.target.value)} className="flex-1 min-w-0 bg-[#111] border border-[#1a1a1a] rounded px-1 py-1 text-[10px] text-[#888]">
            <option value="">All regimes</option>
            {regimes.map((r) => <option key={r} value={r}>R{r}</option>)}
          </select>
          <select value={actionFilter} onChange={(e) => setActionFilter(e.target.value)} className="flex-1 min-w-0 bg-[#111] border border-[#1a1a1a] rounded px-1 py-1 text-[10px] text-[#888]">
            <option value="">All actions</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
            <option value="hold">Hold</option>
          </select>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto min-h-0">
        {filtered.length === 0 ? (
          <EmptyState title="No debates match" description="Try clearing the search or filters." />
        ) : filtered.map((d) => (
          <button
            key={d.id}
            onClick={() => onSelect(d.id)}
            className="w-full text-left px-3 py-2.5 border-b border-[#141414] transition-colors"
            style={{
              background: selectedId === d.id ? "rgba(0,255,127,0.08)" : "transparent",
              borderLeft: selectedId === d.id ? "2px solid #00ff7f" : "2px solid transparent",
            }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-[#555]">{new Date(d.timestamp).toLocaleDateString()}</span>
              <ActionBadge action={d.action} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-[#999]">{d.asset.replace("_", "/")}</span>
              <span className="flex items-center gap-1 text-[10px]" style={{ color: REGIME_COLORS[d.regime_label] ?? "#666" }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: REGIME_COLORS[d.regime_label] ?? "#666" }} />
                R{d.regime_label}
              </span>
            </div>
            <div className="text-[10px] text-[#444] mt-0.5">conviction {(d.conviction * 100).toFixed(0)}%</div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Right panel — debate metadata ──────────────────────────────────────────────

function DebateMetadataPanel({ view, regime }: { view: DebateView | null; regime: RegimeState | null }) {
  if (!view) {
    return <EmptyState title="No debate selected" description="Pick a debate from the browser on the left, or run a new one." />;
  }
  const keyFactors = Array.from(new Set(view.theses.flatMap((t) => t.key_factors ?? [])));

  return (
    <div className="flex flex-col gap-4 text-[11px]">
      <div>
        <div className="stat-label mb-1">Timestamp</div>
        <div className="text-[#ccc] font-mono">{new Date(view.timestamp).toLocaleString()}</div>
        <div className="text-[#666] mt-0.5">{view.asset.replace("_", "/")}</div>
      </div>

      {regime && (
        <div>
          <div className="stat-label mb-1.5">Regime at Debate Time</div>
          <div className="flex items-center gap-1.5 mb-2">
            <span className="w-2 h-2 rounded-full" style={{ background: REGIME_COLORS[regime.dominant_regime] ?? "#666" }} />
            <span className="font-semibold" style={{ color: REGIME_COLORS[regime.dominant_regime] ?? "#666" }}>
              R{regime.dominant_regime} — {regime.regime_description}
            </span>
          </div>
          <div className="flex flex-col gap-1">
            {Object.entries(regime.probabilities).map(([k, v]) => {
              const label = parseInt(k.replace("regime_", ""));
              const pct = Math.round(v * 100);
              return (
                <div key={k} className="flex items-center gap-1.5">
                  <span className="w-6 text-[9px] text-[#555]">R{label}</span>
                  <div className="flex-1 h-1 rounded-full bg-[#1a1a1a] overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${pct}%`, background: REGIME_COLORS[label] ?? "#666" }} />
                  </div>
                  <span className="w-8 text-right text-[9px] text-[#666]">{pct}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {view.decision && (
        <div>
          <div className="stat-label mb-1">Final Decision</div>
          <div className="flex items-center gap-2 mb-1.5">
            <ActionBadge action={view.decision.action} />
            <span className="text-[#666]">{(view.decision.conviction * 100).toFixed(0)}% conviction</span>
          </div>
          <p className="text-[#777] leading-relaxed">{view.decision.rationale}</p>
        </div>
      )}

      {keyFactors.length > 0 && (
        <div>
          <div className="stat-label mb-1.5">Key Factors</div>
          <div className="flex flex-col gap-1">
            {keyFactors.map((f, i) => (
              <div key={i} className="text-[#888] flex gap-1.5"><span className="text-[#333]">•</span>{f}</div>
            ))}
          </div>
        </div>
      )}

      <div>
        <div className="stat-label mb-1">Dissenting Agents</div>
        {view.decision?.dissenting_agents?.length ? (
          <div className="flex flex-wrap gap-1.5">
            {view.decision.dissenting_agents.map((a) => (
              <span
                key={a}
                className="text-[10px] px-1.5 py-0.5 rounded"
                style={{ background: a === "ContrarianAgent" ? "rgba(255,55,55,0.12)" : "#1a1a1a", color: a === "ContrarianAgent" ? "#ff3737" : "#888" }}
              >
                {a}
              </span>
            ))}
          </div>
        ) : <span style={{ color: "#00ff7f88" }}>✓ Full consensus</span>}
      </div>

      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#1a1a1a]">
        <div>
          <div className="stat-label">Duration</div>
          <div className="text-[#999] font-mono mt-0.5">{view.durationSeconds != null ? `${view.durationSeconds.toFixed(0)}s` : "—"}</div>
        </div>
        <div>
          <div className="stat-label">Tokens</div>
          <div className="text-[#999] font-mono mt-0.5">{view.totalTokens != null ? view.totalTokens.toLocaleString() : "—"}</div>
        </div>
      </div>
    </div>
  );
}

// ── Run New Debate modal ───────────────────────────────────────────────────────

function RunDebateModal({ onClose, onRun }: { onClose: () => void; onRun: (asset: string, timestamp: string) => void }) {
  const mode = useMARTAStore((s) => s.mode);
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [asset, setAsset] = useState("BTC_USDT");
  const [dataRange, setDataRange] = useState<DataRange | null>(null);
  const [timestamp, setTimestamp] = useState("");

  useEffect(() => { api.getAssets().then((d) => setAssets(d.assets)).catch(() => {}); }, []);

  useEffect(() => {
    if (mode === "live") return;
    api.getDataRange(asset).then((r) => {
      setDataRange(r);
      const mid = new Date((new Date(r.start).getTime() + new Date(r.end).getTime()) / 2);
      setTimestamp(mid.toISOString().slice(0, 16));
    }).catch(() => {});
  }, [asset, mode]);

  const handleRun = () => {
    const ts = mode === "live" ? new Date().toISOString() : new Date(timestamp).toISOString();
    onRun(asset, ts);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6" onClick={onClose}>
      <div className="bg-[#0d0d0d] border border-[#2a2a2a] rounded-lg w-full max-w-sm p-5" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-[13px] font-bold text-[#e0e0e0]">Run New Debate</h2>
          <button onClick={onClose} className="text-[#555] hover:text-[#999] transition-colors"><X size={16} /></button>
        </div>
        <div className="flex flex-col gap-3">
          <div>
            <label className="stat-label block mb-1">Asset</label>
            <select value={asset} onChange={(e) => setAsset(e.target.value)} className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc]">
              {(assets.length ? assets : [{ id: "BTC_USDT", label: "BTC/USDT" } as AssetInfo]).map((a) => (
                <option key={a.id} value={a.id}>{a.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="stat-label block mb-1">Timestamp</label>
            {mode === "live" ? (
              <div className="text-[11px] text-[#4a9eff] px-2 py-1.5 border border-[#4a9eff33] rounded bg-[#4a9eff08]">● Current time (live mode)</div>
            ) : (
              <input
                type="datetime-local"
                value={timestamp}
                min={dataRange?.start.slice(0, 16)}
                max={dataRange?.end.slice(0, 16)}
                onChange={(e) => setTimestamp(e.target.value)}
                className="w-full bg-[#111] border border-[#2a2a2a] rounded px-2 py-1.5 text-[11px] text-[#ccc]"
              />
            )}
          </div>
          <button
            onClick={handleRun}
            disabled={mode !== "live" && !timestamp}
            className="mt-2 w-full py-2 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Run Debate
          </button>
          <p className="text-[10px] text-[#444] leading-relaxed">
            Calls Mistral 7B via Ollama through the live agent debate graph — typically 30–90 seconds.
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

function DebatesPageInner() {
  const searchParams = useSearchParams();
  const autoRunParam = searchParams.get("autorun") === "1";

  const globalMode = useMARTAStore((s) => s.mode);
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const setSelectedAsset = useMARTAStore((s) => s.setSelectedAsset);
  const setCurrentDecision = useMARTAStore((s) => s.setCurrentDecision);
  const setActiveDebate = useMARTAStore((s) => s.setActiveDebate);

  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [dataRangeForAsset, setDataRangeForAsset] = useState<DataRange | null>(null);
  const [debatesList, setDebatesList] = useState<DebateListItem[]>([]);
  const [debatesLoading, setDebatesLoading] = useState(true);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [storedDetail, setStoredDetail] = useState<DebateDetail | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [liveTarget, setLiveTarget] = useState<{ timestamp: string } | null>(null);
  const [viewMode, setViewMode] = useState<"stored" | "live" | null>(null);

  const [liveTheses, setLiveTheses] = useState<AgentThesis[]>([]);
  const [liveRebuttals, setLiveRebuttals] = useState<AgentRebuttal[]>([]);
  const [liveDecision, setLiveDecision] = useState<FundManagerDecision | null>(null);
  const [liveMeta, setLiveMeta] = useState<{ asset: string; timestamp: string; regime: number } | null>(null);
  const [liveDurationSec, setLiveDurationSec] = useState<number | undefined>(undefined);
  const liveStartRef = useRef<number | null>(null);

  const [regimeAtView, setRegimeAtView] = useState<RegimeState | null>(null);

  useEffect(() => {
    api.getDemoScenarios().then((d) => setScenarios(d.scenarios)).catch(() => {});
  }, []);

  useEffect(() => {
    setDebatesLoading(true);
    api.getDebates().then((d) => setDebatesList(d.debates)).catch(() => setDebatesList([])).finally(() => setDebatesLoading(false));
  }, []);

  useEffect(() => {
    api.getDataRange(selectedAsset).then(setDataRangeForAsset).catch(() => {});
  }, [selectedAsset]);

  const handleSelectDebate = useCallback((id: number) => {
    setSelectedId(id);
    setViewMode("stored");
    setModalOpen(false);
    api.getDebate(id).then(setStoredDetail).catch(() => setStoredDetail(null));
  }, []);

  // Deep link from the Backtest page's "Replay Debate" buttons: /debates?scenario=<id>
  const scenarioParam = searchParams.get("scenario");
  const replayedScenarioRef = useRef<string | null>(null);
  useEffect(() => {
    if (!scenarioParam || scenarioParam === replayedScenarioRef.current || scenarios.length === 0) return;
    const match = scenarios.find((s) => s.id === scenarioParam);
    if (match?.debate_index !== undefined) {
      replayedScenarioRef.current = scenarioParam;
      handleSelectDebate(match.debate_index);
    }
  }, [scenarioParam, scenarios, handleSelectDebate]);

  const handleRunNew = useCallback((asset: string, timestamp: string) => {
    setSelectedAsset(asset);
    setSelectedId(null);
    setStoredDetail(null);
    setViewMode("live");
    setLiveTheses([]);
    setLiveRebuttals([]);
    setLiveDecision(null);
    setLiveMeta(null);
    setLiveDurationSec(undefined);
    liveStartRef.current = Date.now();
    setModalOpen(false);
    setLiveTarget({ timestamp });
  }, [setSelectedAsset]);

  const handleEvent = useCallback((msg: DebateMessage) => {
    switch (msg.type) {
      case "debate_start":
        setLiveMeta({ asset: msg.asset as string, timestamp: msg.timestamp as string, regime: msg.regime_label as number });
        break;
      case "agent_thesis":
        setLiveTheses((prev) => [...prev, msg as unknown as AgentThesis]);
        break;
      case "agent_rebuttal":
        setLiveRebuttals((prev) => [...prev, msg as unknown as AgentRebuttal]);
        break;
      case "fund_manager":
        setLiveDecision({
          action: msg.action as "long" | "short" | "hold",
          position_size: msg.position_size as number,
          conviction: msg.conviction as number,
          rationale: msg.rationale as string,
          regime_override: msg.regime_override as boolean,
          dissenting_agents: msg.dissenting_agents as string[],
        });
        break;
      case "debate_complete":
        if (liveStartRef.current) setLiveDurationSec((Date.now() - liveStartRef.current) / 1000);
        break;
      default:
        break;
    }
  }, []);

  const handleDecision = useCallback((d: FundManagerDecision) => {
    setCurrentDecision(d);
  }, [setCurrentDecision]);

  const view: DebateView | null = useMemo(() => {
    if (viewMode === "stored" && storedDetail) {
      return {
        asset: storedDetail.asset,
        timestamp: storedDetail.timestamp,
        regimeLabel: storedDetail.regime_label,
        theses: storedDetail.round1_theses.map(mapThesis),
        rebuttals: storedDetail.round2_rebuttals.map(mapRebuttal),
        decision: mapDecision(storedDetail.fund_manager_decision),
        totalTokens: storedDetail.total_tokens_used,
        durationSeconds: storedDetail.debate_duration_seconds,
      };
    }
    if (viewMode === "live" && liveMeta) {
      return {
        asset: liveMeta.asset,
        timestamp: liveMeta.timestamp,
        regimeLabel: liveMeta.regime,
        theses: liveTheses,
        rebuttals: liveRebuttals,
        decision: liveDecision,
        durationSeconds: liveDurationSec,
      };
    }
    return null;
  }, [viewMode, storedDetail, liveMeta, liveTheses, liveRebuttals, liveDecision, liveDurationSec]);

  useEffect(() => {
    if (view?.decision) {
      setActiveDebate({
        asset: view.asset,
        timestamp: view.timestamp,
        regime_label: view.regimeLabel,
        theses: view.theses,
        rebuttals: view.rebuttals,
        decision: view.decision,
      });
    }
  }, [view, setActiveDebate]);

  useEffect(() => {
    if (view) {
      api.getRegimeAt(view.asset, view.timestamp).then(setRegimeAtView).catch(() => setRegimeAtView(null));
    }
  }, [view?.asset, view?.timestamp]);

  const contrarianInfo = useMemo(() => {
    if (!view) return null;
    const nonContrarian = view.theses.filter((t) => t.agent !== "ContrarianAgent");
    if (nonContrarian.length === 0) return null;
    const avgConfidence = nonContrarian.reduce((sum, t) => sum + t.confidence, 0) / nonContrarian.length;
    const contrarian = view.theses.find((t) => t.agent === "ContrarianAgent");
    return { avgConfidence, active: !!contrarian && contrarian.bias !== "neutral" };
  }, [view]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <PageHeader title="Debates" description="Browse stored deliberations or run a new multi-agent debate" />
      <div className="flex-1 min-h-0 grid gap-2" style={{ gridTemplateColumns: "300px minmax(0,1fr) 300px" }}>
        <Panel noPadding>
          {debatesLoading ? <div className="p-3"><SkeletonRows rows={10} /></div> : (
            <DebateBrowser debates={debatesList} selectedId={selectedId} onSelect={handleSelectDebate} onRunNew={() => setModalOpen(true)} />
          )}
        </Panel>
        <Panel title="Debate Transcript" noPadding>
          <DebateStream
            scenarios={scenarios}
            asset={selectedAsset}
            dataRange={dataRangeForAsset}
            onDecision={handleDecision}
            onEvent={handleEvent}
            liveMode={globalMode === "live"}
            autoRun={autoRunParam}
            forceReplayId={selectedId}
            forceLiveRun={liveTarget}
          />
        </Panel>
        <div className="flex flex-col gap-2 min-h-0">
          <Panel title="Debate Metadata" className="flex-[3] min-h-0 overflow-auto">
            <DebateMetadataPanel view={view} regime={regimeAtView} />
          </Panel>
          <Panel title="News Feed" noPadding className="flex-[2] min-h-0">
            <NewsPanel asset={view?.asset ?? selectedAsset} timestamp={view?.timestamp ?? null} liveMode={globalMode === "live"} />
          </Panel>
        </div>
      </div>
      {modalOpen && <RunDebateModal onClose={() => setModalOpen(false)} onRun={handleRunNew} />}

      <LearningPanel>
        {contrarianInfo ? (
          <>
            The Contrarian {contrarianInfo.active ? "activated" : "stayed quiet"} because consensus
            confidence ({(contrarianInfo.avgConfidence * 100).toFixed(0)}%) {contrarianInfo.active ? "exceeded" : "was below"}{" "}
            the 0.65 activation threshold. Without regime context, the agent-only policy&apos;s Sharpe
            drops to roughly -2.0 versus MARTA&apos;s full-pipeline 1.37 — the debate layer alone
            doesn&apos;t replace the regime embedding, it conditions on it.
          </>
        ) : (
          <>
            The Contrarian agent activates only when the other agents&apos; average round-1 confidence
            exceeds 0.65 — below that, consensus is already uncertain enough that a devil&apos;s-advocate
            rebuttal adds little. Select a debate to see its actual confidence level.
          </>
        )}
      </LearningPanel>
    </div>
  );
}

export default function DebatesPage() {
  return (
    <Suspense fallback={null}>
      <DebatesPageInner />
    </Suspense>
  );
}
