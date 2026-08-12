"use client";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { MartaWebSocket } from "@/lib/websocket";
import { useMARTAStore } from "@/lib/store";
import type {
  AgentThesis, AgentRebuttal, FundManagerDecision, RiskLimits,
  Scenario, DebateMessage, DataRange,
} from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

interface Props {
  scenarios: Scenario[];
  asset: string;
  dataRange: DataRange | null;
  onDecision?: (d: FundManagerDecision) => void;
  onEvent?: (msg: DebateMessage) => void;
  externalTimestamp?: string | null;
  liveMode?: boolean;
  currentTimestamp?: string | null;
  autoRun?: boolean;
  /** Replay a specific stored debate by index (0..n), e.g. from a debate browser list. */
  forceReplayId?: number | null;
  /** Trigger a fresh live debate run at a given ISO timestamp (asset comes from the `asset` prop). */
  forceLiveRun?: { timestamp: string } | null;
}

// "Now", floored to the top of the current hour, formatted for a datetime-local input.
function currentHourLocal(): string {
  const d = new Date();
  d.setMinutes(0, 0, 0);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:00`;
}

const AGENT_COLORS: Record<string, string> = {
  MacroAgent:       "#4a9eff",
  TechnicalAgent:   "#00d4ff",
  SentimentAgent:   "#ffd700",
  RiskAgent:        "#ff8c00",
  ContrarianAgent:  "#ff3737",
  FundManagerAgent: "#00ff7f",
};

const AGENT_ROLES: Record<string, string> = {
  MacroAgent:       "Macro Analyst",
  TechnicalAgent:   "Technical Analyst",
  SentimentAgent:   "Sentiment Analyst",
  RiskAgent:        "Risk Manager",
  ContrarianAgent:  "Devil's Advocate",
  FundManagerAgent: "Fund Manager",
};

function ExpandableText({ text, className }: { text: string; className?: string }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = text.length > 180;
  return (
    <div>
      <p className={`${className ?? ""} ${!expanded && isLong ? "line-clamp-3" : ""}`}>{text}</p>
      {isLong && (
        <button
          onClick={() => setExpanded((e) => !e)}
          className="text-[10px] text-[#4a9eff] hover:text-[#6db4ff] transition-colors mt-0.5"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

export function ThinkingIndicator({ agent }: { agent: string }) {
  const color = AGENT_COLORS[agent] ?? "#666";
  return (
    <div
      className="flex items-center gap-2 text-[11px] py-1.5 px-2.5 rounded animate-fade-in"
      style={{
        border: `1px solid ${color}33`,
        background: `${color}08`,
      }}
    >
      <span className="font-medium" style={{ color }}>{agent}</span>
      <span className="italic text-[#444]">reasoning</span>
      <span className="flex gap-1 ml-auto">
        <span className="thinking-dot" style={{ background: color }} />
        <span className="thinking-dot" style={{ background: color }} />
        <span className="thinking-dot" style={{ background: color }} />
      </span>
    </div>
  );
}

function ThesisCard({ thesis }: { thesis: AgentThesis }) {
  const isContrarian = thesis.agent === "ContrarianAgent";
  const color = AGENT_COLORS[thesis.agent] ?? "#666";
  const biasColor = thesis.bias === "bullish" ? "#00ff7f" : thesis.bias === "bearish" ? "#ff3737" : "#666";
  const confPct = Math.round(thesis.confidence * 100);
  return (
    <div
      className="rounded p-2.5 mb-2 text-[11px] border-l-2 animate-slide-in"
      style={{
        background: isContrarian ? "rgba(255,140,0,0.06)" : "rgba(255,255,255,0.02)",
        borderColor: color,
      }}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="font-semibold" style={{ color }}>{thesis.agent}</span>
          <span className="text-[9px] text-[#444] uppercase tracking-wider">{AGENT_ROLES[thesis.agent] ?? ""}</span>
          {isContrarian && (
            <span className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider"
              style={{ background: "rgba(255,55,55,0.2)", color: "#ff3737" }}>
              DEVIL&apos;S ADVOCATE
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold"
            style={{ background: `${biasColor}22`, color: biasColor }}>
            {thesis.bias}
          </span>
          <span className="font-mono text-[10px]" style={{ color: biasColor }}>{confPct}%</span>
        </div>
      </div>
      {/* Confidence bar */}
      <div className="h-0.5 rounded-full mb-1.5 overflow-hidden" style={{ background: "#1a1a1a" }}>
        <div className="h-full rounded-full" style={{ width: `${confPct}%`, background: biasColor, opacity: 0.7 }} />
      </div>
      <ExpandableText text={thesis.reasoning} className="text-[#888] leading-relaxed" />
      {thesis.key_factors?.length > 0 && (
        <div className="flex gap-1.5 mt-1.5 flex-wrap">
          {thesis.key_factors.map((f, i) => (
            <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a1a1a] text-[#555]">{f}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function RebuttalCard({ rebuttal }: { rebuttal: AgentRebuttal }) {
  const color = AGENT_COLORS[rebuttal.agent] ?? "#666";
  return (
    <div className="rounded p-2 mb-1.5 text-[11px] ml-4 border-l-2 border-[#2a2a2a] animate-slide-in">
      <div className="flex items-center gap-1.5 mb-1">
        <span style={{ color }} className="font-medium">{rebuttal.agent}</span>
        <span className="text-[#444]">→</span>
        <span className="text-[#555]">{rebuttal.target}</span>
        <span className={`ml-auto text-[10px] px-1 py-0.5 rounded ${rebuttal.maintains_bias ? "text-[#555]" : "text-[#ff3737]"}`}>
          {rebuttal.maintains_bias ? "maintains" : "⚡ flipped"}
        </span>
      </div>
      <p className="text-[#666]">{rebuttal.rebuttal_point}</p>
    </div>
  );
}

function FMCard({ decision }: { decision: FundManagerDecision }) {
  const actionColor = decision.action === "long" ? "#00ff7f" : decision.action === "short" ? "#ff3737" : "#666";
  const actionGlow = decision.action !== "hold" ? `0 0 12px ${actionColor}33` : "none";
  const convPct = Math.round(decision.conviction * 100);
  return (
    <div
      className="rounded p-3 text-[11px] animate-slide-in"
      style={{
        background: "rgba(0,255,127,0.04)",
        border: "1px solid #00ff7f33",
        boxShadow: "0 0 0 1px rgba(0,255,127,0.05)",
      }}
    >
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className="font-semibold text-[#00ff7f]">FundManagerAgent</span>
        <span
          className="px-2.5 py-1 rounded font-bold text-[12px] uppercase tracking-wider"
          style={{
            background: `${actionColor}20`,
            color: actionColor,
            border: `1px solid ${actionColor}55`,
            boxShadow: actionGlow,
          }}
        >
          {decision.action}
        </span>
        {decision.regime_override && (
          <span className="text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase"
            style={{ background: "rgba(255,140,0,0.18)", color: "#ff8c00", border: "1px solid #ff8c0033" }}>
            ⚠ REGIME OVERRIDE
          </span>
        )}
        <span className="ml-auto font-mono" style={{ color: actionColor }}>{convPct}% conviction</span>
      </div>
      {/* Conviction bar */}
      <div className="mb-2 h-1 bg-[#1a1a1a] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${convPct}%`, background: actionColor, boxShadow: `0 0 4px ${actionColor}55` }}
        />
      </div>
      <p className="text-[#777] leading-relaxed">{decision.rationale}</p>
      {decision.dissenting_agents?.length > 0 && (
        <div className="mt-1.5 flex gap-1.5 flex-wrap">
          <span className="text-[#444] text-[10px]">Dissenting:</span>
          {decision.dissenting_agents.map((a) => (
            <span key={a} className="text-[10px] px-1.5 py-0.5 rounded"
              style={{
                background: a === "ContrarianAgent" ? "rgba(255,140,0,0.12)" : "#1a1a1a",
                color: a === "ContrarianAgent" ? "#ff8c00" : "#666",
              }}>
              {a}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function RiskCard({ risk }: { risk: RiskLimits }) {
  return (
    <div className="rounded p-2 text-[11px] bg-[#0f0f0f] border border-[#1e1e1e] mb-2">
      <div className="text-[#555] font-semibold mb-1.5 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#ff8c00" }} />
        RiskAgent (GARCH)
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div>
          <div className="text-[#444] text-[10px] uppercase tracking-wider">Max Pos</div>
          <div className="text-[#ff8c00] font-mono mt-0.5">{(risk.max_position_size * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[#444] text-[10px] uppercase tracking-wider">Vol Fcst</div>
          <div className="text-[#ff8c00] font-mono mt-0.5">{(risk.volatility_forecast * 100).toFixed(2)}%</div>
        </div>
        <div>
          <div className="text-[#444] text-[10px] uppercase tracking-wider">CVaR 95%</div>
          <div className="text-[#ff3737] font-mono mt-0.5">{((risk.cvar_95 ?? 0) * 100).toFixed(2)}%</div>
        </div>
      </div>
    </div>
  );
}

// Pure message → element mapping, no closure over any DebateStream state — reused
// as-is by DebateFeed below (and by any other page that streams the same event
// shapes, e.g. the Trading page's "Run Debate & Execute" live view).
export function renderDebateMessage(msg: DebateMessage, i: number) {
  switch (msg.type) {
    case "round1_start":
      return (
        <div key={i} className="flex items-center gap-2 py-2">
          <div className="flex-1 h-px bg-[#1a1a1a]" />
          <span className="text-[10px] text-[#333] uppercase tracking-widest px-2">Round 1 · Theses</span>
          <div className="flex-1 h-px bg-[#1a1a1a]" />
        </div>
      );
    case "round2_start":
      return (
        <div key={i} className="flex items-center gap-2 py-2 mt-1">
          <div className="flex-1 h-px bg-[#1a1a1a]" />
          <span className="text-[10px] text-[#333] uppercase tracking-widest px-2">Round 2 · Rebuttals</span>
          <div className="flex-1 h-px bg-[#1a1a1a]" />
        </div>
      );
    case "contrarian_activated":
      return (
        <div
          key={i}
          className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-md my-2 text-center animate-fade-in border-2"
          style={{
            borderColor: "#ff3737",
            background: "linear-gradient(90deg, rgba(255,140,0,0.14), rgba(255,55,55,0.14))",
            boxShadow: "0 0 16px rgba(255,55,55,0.15)",
          }}
        >
          <span className="text-[13px] font-extrabold uppercase tracking-widest" style={{ color: "#ff3737" }}>
            ⚡ Contrarian Activated
          </span>
          <span className="text-[10px] font-normal normal-case tracking-normal" style={{ color: "#ff8c00" }}>
            — consensus confidence {Math.round((msg.avg_confidence as number) * 100)}% triggered devil&apos;s advocate review
          </span>
        </div>
      );
    case "agent_thesis":
      return <ThesisCard key={i} thesis={msg as unknown as AgentThesis} />;
    case "agent_rebuttal":
      return <RebuttalCard key={i} rebuttal={msg as unknown as AgentRebuttal} />;
    case "risk_limits":
      return <RiskCard key={i} risk={msg as unknown as RiskLimits} />;
    case "fund_manager":
      return <FMCard key={i} decision={msg as unknown as FundManagerDecision} />;
    case "ppo_decision":
      return (
        <div key={i} className="text-[11px] p-2 rounded border border-[#1a1a1a] bg-[#0d0d0d] mt-1 animate-fade-in">
          <span className="text-[#444]">PPO Override → </span>
          <span style={{ color: msg.action === "long" ? "#00ff7f" : msg.action === "short" ? "#ff3737" : "#666" }}>
            {(msg.action as string).toUpperCase()}
          </span>
          <span className="text-[#555]"> {((msg.position_size as number) * 100).toFixed(1)}% position</span>
          {Boolean(msg.regime_override) && <span className="ml-2 text-[#ff8c00]">⚠ regime override</span>}
        </div>
      );
    default:
      return null;
  }
}

// Scrollable transcript — messages + live "thinking" indicator, auto-scrolling to the
// newest entry. No websocket/controls of its own; any page driving its own debate
// connection (DebateStream below, or Trading page's "Run Debate & Execute") can feed
// it a running `messages` array and get the same rendering for free.
export function DebateFeed({ messages, thinkingAgent, className }: {
  messages: DebateMessage[];
  thinkingAgent: string | null;
  className?: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, thinkingAgent]);

  return (
    <div ref={scrollRef} className={className ?? "flex-1 overflow-auto p-3 min-h-0"}>
      {messages.map((m, i) => renderDebateMessage(m, i))}
      {thinkingAgent && <ThinkingIndicator agent={thinkingAgent} />}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DebateStream({ scenarios, asset, dataRange, onDecision, onEvent, externalTimestamp, liveMode = false, currentTimestamp, autoRun = false, forceReplayId = null, forceLiveRun = null }: Props) {
  const [messages, setMessages] = useState<DebateMessage[]>([]);
  const [thinkingAgent, setThinkingAgent] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [mode, setMode] = useState<"scenario" | "custom">("scenario");
  const [selectedScenario, setSelectedScenario] = useState<string>("");
  const [customDate, setCustomDate] = useState<string>("");
  const [debateInfo, setDebateInfo] = useState<{ asset?: string; timestamp?: string; regime?: number } | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const wsRef = useRef<MartaWebSocket | null>(null);
  const showContrarian = useMARTAStore((s) => s.showContrarian);
  const regimeNames = useMARTAStore((s) => s.regimeNames);

  // Default scenario to first matching asset when scenarios load
  useEffect(() => {
    const match = scenarios.find((s) => s.asset === asset);
    if (match) setSelectedScenario(match.id);
  }, [scenarios, asset]);

  // Default custom date to middle of data range
  useEffect(() => {
    if (dataRange && !customDate) {
      const mid = new Date((new Date(dataRange.start).getTime() + new Date(dataRange.end).getTime()) / 2);
      setCustomDate(mid.toISOString().slice(0, 16));
    }
  }, [dataRange, customDate]);

  // Chart click → switch to Live Debate at that timestamp
  useEffect(() => {
    if (externalTimestamp) {
      setMode("custom");
      setCustomDate(externalTimestamp.slice(0, 16));
    }
  }, [externalTimestamp]);

  // Live mode: keep the custom date in sync with the latest streaming candle
  useEffect(() => {
    if (liveMode && currentTimestamp && !isRunning) {
      setMode("custom");
      setCustomDate(currentTimestamp.slice(0, 16));
    }
  }, [liveMode, currentTimestamp, isRunning]);

  // Group scenarios by asset group
  const scenarioGroups = useMemo(() => {
    const groups: Record<string, Scenario[]> = {};
    for (const s of scenarios) {
      const g = s.group ?? s.asset;
      if (!groups[g]) groups[g] = [];
      groups[g].push(s);
    }
    return groups;
  }, [scenarios]);

  const runDebate = useCallback((override?: { replayId?: number; liveDate?: string }) => {
    if (isRunning) return;
    setMessages([]);
    setThinkingAgent(null);
    setIsRunning(true);
    setDebateInfo(null);
    setStatusMsg("");

    let wsPath: string;
    if (override?.replayId != null) {
      wsPath = `/ws/debate?mode=replay&debate_id=${override.replayId}`;
    } else if (override?.liveDate != null || mode === "custom") {
      const ts = new Date(override?.liveDate ?? customDate).toISOString();
      wsPath = `/ws/debate?mode=live&asset=${encodeURIComponent(asset)}&timestamp=${encodeURIComponent(ts)}`;
      setStatusMsg("Running live debate via Mistral 7B — may take 30-90 seconds…");
    } else {
      const scenario = scenarios.find((s) => s.id === selectedScenario);
      const scenarioParam = scenario ? `&scenario_id=${scenario.id}` : "";
      wsPath = `/ws/debate?mode=replay${scenarioParam}`;
    }

    wsRef.current?.close();
    const ws = new MartaWebSocket(wsPath);

    const off = ws.on((msg) => {
      onEvent?.(msg as DebateMessage);
      const type = msg.type as string;
      if (type === "debate_start") {
        setDebateInfo({ asset: msg.asset as string, timestamp: msg.timestamp as string, regime: msg.regime_label as number });
        setStatusMsg("");
        return;
      }
      if (type === "agent_thinking") { setThinkingAgent(msg.agent as string); return; }
      if (type === "error") {
        setStatusMsg(`Error: ${msg.message as string}`);
        setIsRunning(false);
        off();
        return;
      }
      if (type === "debate_complete") {
        setThinkingAgent(null);
        setIsRunning(false);
        setStatusMsg("");
        off();
        return;
      }
      if (type === "fund_manager") {
        onDecision?.({
          action: msg.action as "long" | "short" | "hold",
          position_size: msg.position_size as number,
          conviction: msg.conviction as number,
          rationale: msg.rationale as string,
          regime_override: msg.regime_override as boolean,
          dissenting_agents: msg.dissenting_agents as string[],
        });
      }
      setThinkingAgent(null);
      setMessages((prev) => [...prev, msg as DebateMessage]);
    });

    ws.connect();
    wsRef.current = ws;
  }, [isRunning, mode, selectedScenario, customDate, asset, scenarios, onDecision, onEvent]);

  useEffect(() => () => wsRef.current?.close(), []);

  // Auto-trigger a run once, after a scenario has been resolved (used by ?autorun=1 deep links)
  const autoRanRef = useRef(false);
  useEffect(() => {
    if (autoRun && !autoRanRef.current && selectedScenario) {
      autoRanRef.current = true;
      runDebate();
    }
  }, [autoRun, selectedScenario, runDebate]);

  // Replay a specific stored debate by id — e.g. a click in a debate browser list
  const replayedIdRef = useRef<number | null>(null);
  useEffect(() => {
    if (forceReplayId != null && forceReplayId !== replayedIdRef.current) {
      replayedIdRef.current = forceReplayId;
      setMode("scenario");
      runDebate({ replayId: forceReplayId });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forceReplayId]);

  // Trigger a fresh live debate run at an externally-chosen timestamp (e.g. "Run New Debate" modal)
  const liveRunKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (forceLiveRun && forceLiveRun.timestamp !== liveRunKeyRef.current) {
      liveRunKeyRef.current = forceLiveRun.timestamp;
      setMode("custom");
      setCustomDate(forceLiveRun.timestamp.slice(0, 16));
      runDebate({ liveDate: forceLiveRun.timestamp });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forceLiveRun]);

  const visibleMessages = useMemo(() => {
    if (showContrarian) return messages;
    return messages.filter((m) => {
      if (m.type === "contrarian_activated") return false;
      if (m.type === "agent_thesis" && m.agent === "ContrarianAgent") return false;
      if (m.type === "agent_rebuttal" && (m.agent === "ContrarianAgent" || m.target === "ContrarianAgent")) return false;
      return true;
    });
  }, [messages, showContrarian]);

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Controls bar */}
      <div className="border-b border-[#1a1a1a] px-3 py-2 flex items-center gap-2 flex-wrap shrink-0">
        {/* Mode toggle */}
        <div className="flex rounded overflow-hidden border border-[#222]">
          <button
            onClick={() => setMode("scenario")}
            className="text-[10px] px-2 py-1 transition-colors"
            style={{ background: mode === "scenario" ? "rgba(0,255,127,0.1)" : "transparent", color: mode === "scenario" ? "#00ff7f" : "#555" }}
          >
            Replay
          </button>
          <button
            onClick={() => {
              setMode("custom");
              setCustomDate(currentHourLocal());
            }}
            className="text-[10px] px-2 py-1 transition-colors border-l border-[#222]"
            style={{ background: mode === "custom" ? "rgba(74,158,255,0.1)" : "transparent", color: mode === "custom" ? "#4a9eff" : "#555" }}
          >
            Live Debate
          </button>
        </div>

        {mode === "scenario" ? (
          <select
            className="bg-[#1a1a1a] border border-[#2a2a2a] text-[#888] text-[10px] rounded px-2 py-1 flex-1 min-w-0"
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
          >
            {Object.entries(scenarioGroups).map(([group, groupScenarios]) => (
              <optgroup key={group} label={`── ${group} ──`} style={{ color: "#555" }}>
                {groupScenarios.map((s) => (
                  <option key={s.id} value={s.id}>{s.label}</option>
                ))}
              </optgroup>
            ))}
          </select>
        ) : (
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <input
              type="datetime-local"
              value={customDate}
              min={dataRange ? dataRange.start.slice(0, 16) : undefined}
              max={dataRange ? dataRange.end.slice(0, 16) : undefined}
              onChange={(e) => setCustomDate(e.target.value)}
              className="bg-[#1a1a1a] border border-[#2a2a2a] text-[#888] text-[10px] rounded px-2 py-1 flex-1 min-w-0"
            />
            <span className="text-[10px] text-[#444] whitespace-nowrap">{asset.replace("_", "/")}</span>
          </div>
        )}

        <button
          onClick={() => runDebate()}
          disabled={isRunning}
          className="text-[10px] px-3 py-1 rounded border transition-colors shrink-0"
          style={{
            background: isRunning ? "transparent" : mode === "custom" ? "rgba(74,158,255,0.1)" : "rgba(0,255,127,0.1)",
            borderColor: isRunning ? "#333" : mode === "custom" ? "#4a9eff44" : "#00ff7f44",
            color: isRunning ? "#444" : mode === "custom" ? "#4a9eff" : "#00ff7f",
            cursor: isRunning ? "not-allowed" : "pointer",
          }}
        >
          {isRunning ? "Running…" : mode === "custom" ? "Run Live Debate" : "Run Debate"}
        </button>
      </div>

      {/* Debate info */}
      {(debateInfo || statusMsg) && (
        <div className="border-b border-[#1a1a1a] px-3 py-1.5 flex items-center gap-2 shrink-0">
          {debateInfo && (
            <>
              <span className="text-[10px] text-[#555]">{debateInfo.asset?.replace("_", "/")}</span>
              {debateInfo.regime !== undefined && (
                <span className="flex items-center gap-1 text-[10px]">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: REGIME_COLORS[debateInfo.regime] ?? "#666" }} />
                  <span style={{ color: REGIME_COLORS[debateInfo.regime] ?? "#666" }}>R{debateInfo.regime} — {regimeNames[debateInfo.regime] ?? ""}</span>
                </span>
              )}
              {debateInfo.timestamp && (
                <span className="text-[10px] text-[#444] ml-auto">
                  {new Date(debateInfo.timestamp).toLocaleString()}
                </span>
              )}
            </>
          )}
          {statusMsg && <span className="text-[10px] text-[#4a9eff] italic">{statusMsg}</span>}
        </div>
      )}

      {/* Debate feed */}
      {messages.length === 0 && !thinkingAgent && !isRunning ? (
        <div className="flex-1 overflow-auto p-3 min-h-0 flex flex-col items-center justify-center h-full gap-3 text-center">
          <div className="text-[#333] text-[11px]">
            {mode === "scenario"
              ? "Select a scenario and click Run Debate"
              : "Pick any date in the historical range and run a live debate"}
          </div>
          {mode === "custom" && dataRange && (
            <div className="text-[10px] text-[#333]">
              Available: {new Date(dataRange.start).toLocaleDateString()} → {new Date(dataRange.end).toLocaleDateString()}
            </div>
          )}
          {mode === "custom" && (
            <div className="text-[10px] text-[#444] max-w-xs leading-relaxed border border-[#1a1a1a] rounded p-2">
              <span className="text-[#4a9eff]">Live Debate</span> calls Mistral 7B via Ollama and takes 30–90s.
              Make sure Ollama is running.
            </div>
          )}
        </div>
      ) : (
        <DebateFeed messages={visibleMessages} thinkingAgent={thinkingAgent} />
      )}
    </div>
  );
}
