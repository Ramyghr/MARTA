"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Zap, Loader2, MessagesSquare } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton, SkeletonRows } from "@/components/ui/Skeleton";
import { LearningPanel } from "@/components/LearningPanel";
import { ReasoningSlideOver, fmtPrice, sourceLabel } from "@/components/TradeReasoningPanel";
import { PendingApprovalsPanel } from "@/components/PendingApprovalsPanel";
import { DebateFeed } from "@/components/DebateStream";
import { api } from "@/lib/api";
import { useMARTAStore } from "@/lib/store";
import { MartaWebSocket } from "@/lib/websocket";
import type { Candle, LiveSignal, TradeWithReasoning, FundManagerDecision, DebateMessage, AlpacaOrder } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

function actionColor(action?: string): string {
  return action === "long" ? "#00ff7f" : action === "short" ? "#ff3737" : "#666666";
}

function sourceDisplayLabel(source?: string): string {
  if (!source) return "—";
  if (source.includes("+debate")) return "Debate Override";
  if (source.startsWith("ppo")) return "PPO";
  if (source.startsWith("regime_policy")) return "Regime Rules";
  return source;
}

// ── Panel A — Live Signal ───────────────────────────────────────────────────────

function LiveSignalPanel({
  asset, price, signal, loading, lastFiredAt, onTrigger, triggering,
  onRunDebate, debateRunning, debateStatus, debateResult, executing, approvalMode,
}: {
  asset: string;
  price: number | null;
  signal: LiveSignal | null;
  loading: boolean;
  lastFiredAt: string | null;
  onTrigger: () => void;
  triggering: boolean;
  onRunDebate: () => void;
  debateRunning: boolean;
  debateStatus: string;
  debateResult: FundManagerDecision | null;
  executing: boolean;
  approvalMode: boolean;
}) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const regimeLabel = signal?.regime_label ?? 0;
  const regimeColor = REGIME_COLORS[regimeLabel] ?? "#555";
  // Prefer the live per-request description from the signal itself (always in sync with
  // the current HMM); fall back to the app-wide store only when no signal has loaded yet.
  const regimeDesc = signal?.regime_description || regimeNames[regimeLabel] || "";

  return (
    <Panel title="Live Signal">
      {loading ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <div className="flex flex-col gap-4 h-full">
          <div>
            <div className="stat-label mb-1">{asset.replace("_", "/")} Price</div>
            <div className="text-3xl font-bold font-mono text-[#e0e0e0]">
              {price != null ? fmtPrice(price) : signal ? fmtPrice(signal.price) : "—"}
            </div>
          </div>

          <div>
            <div className="stat-label mb-1.5">Current Regime</div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: regimeColor }} />
              <span className="font-semibold text-[13px]" style={{ color: regimeColor }}>
                R{regimeLabel} — {regimeDesc}
              </span>
            </div>
          </div>

          <div>
            <div className="stat-label mb-1.5">Signal</div>
            {signal ? (
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span
                    className="px-2.5 py-1 rounded text-[13px] font-bold uppercase"
                    style={{ background: `${actionColor(signal.action)}20`, color: actionColor(signal.action), border: `1px solid ${actionColor(signal.action)}44` }}
                  >
                    {signal.action}
                  </span>
                  {signal.action !== "hold" && (
                    <span className="text-[12px] text-[#888] font-mono">{(signal.position_size * 100).toFixed(1)}%</span>
                  )}
                </div>
                {signal.pending_approval && (
                  <span className="text-[10px] font-semibold" style={{ color: "#ff8c00" }}>
                    ⏸ Awaiting approval — see Pending Approvals below
                  </span>
                )}
              </div>
            ) : (
              <span className="text-[11px] text-[#444]">No signal yet</span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div>
              <div className="stat-label">Source</div>
              <div className="text-[#999] mt-0.5">{sourceDisplayLabel(signal?.source)}</div>
            </div>
            <div>
              <div className="stat-label">Last Signal</div>
              <div className="text-[#999] mt-0.5 font-mono">
                {lastFiredAt ? new Date(lastFiredAt).toLocaleTimeString() : "—"}
              </div>
            </div>
          </div>

          {(debateRunning || debateResult) && (
            <div className="mt-auto rounded border border-[#1a1a1a] bg-[#0d0d0d] p-2.5 text-[11px] flex flex-col gap-1.5">
              {debateRunning ? (
                <div className="flex items-center gap-2 text-[#4a9eff]">
                  <Loader2 size={13} className="animate-spin shrink-0" />
                  <span>{debateStatus || "Agents deliberating…"}</span>
                </div>
              ) : debateResult ? (
                <>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[#666]">Debate result:</span>
                    <span className="font-bold uppercase" style={{ color: actionColor(debateResult.action) }}>
                      {debateResult.action}
                    </span>
                    <span className="text-[#888]">{(debateResult.conviction * 100).toFixed(0)}% conviction</span>
                  </div>
                  {executing && (
                    <div className="font-semibold leading-snug" style={{ color: approvalMode ? "#ff8c00" : "#00ff7f" }}>
                      {approvalMode ? "Submitting for approval" : "Executing"}: {debateResult.action.toUpperCase()} {(debateResult.position_size * 100).toFixed(0)}%
                      {" "}based on {(debateResult.conviction * 100).toFixed(0)}% conviction debate
                    </div>
                  )}
                </>
              ) : null}
            </div>
          )}

          <button
            onClick={onRunDebate}
            disabled={debateRunning || triggering}
            className={`flex items-center justify-center gap-1.5 py-2 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${debateRunning || debateResult ? "" : "mt-auto"}`}
          >
            <MessagesSquare size={12} className={debateRunning ? "animate-pulse" : ""} />
            {debateRunning ? "Agents deliberating…" : "▶ Run Debate & Execute"}
          </button>

          <button
            onClick={onTrigger}
            disabled={triggering || debateRunning}
            className="flex items-center justify-center gap-1.5 py-2 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Zap size={12} className={triggering ? "animate-pulse" : ""} />
            {triggering ? "Firing signal…" : "Trigger Manual Signal"}
          </button>
        </div>
      )}
    </Panel>
  );
}

// ── Panel B — Paper Portfolio ───────────────────────────────────────────────────

function PaperPortfolioPanel({ signal, loading }: { signal: LiveSignal | null; loading: boolean }) {
  const pnlColor = (signal?.paper_total_return ?? 0) >= 0 ? "#00ff7f" : "#ff3737";
  const curveData = (signal?.paper_trades ?? []).map((t, i) => ({ i, value: t.portfolio_value }));
  const lastTrade = signal?.paper_trades?.at(-1);
  const openPosition = lastTrade && lastTrade.action !== "hold" ? lastTrade : null;

  return (
    <Panel title="Paper Portfolio">
      {loading ? (
        <Skeleton className="h-64 w-full" />
      ) : !signal ? (
        <EmptyState title="No paper portfolio yet" description="Trigger a signal to start the paper ledger for this asset." />
      ) : (
        <div className="flex flex-col gap-3 h-full">
          <div>
            <div className="stat-label mb-1">Portfolio Value</div>
            <div className="text-3xl font-bold font-mono" style={{ color: pnlColor }}>
              ${signal.paper_portfolio_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="rounded border border-[#1a1a1a] p-2">
              <div className="stat-label">Total Return</div>
              <div className="font-mono font-semibold mt-0.5" style={{ color: pnlColor }}>
                {signal.paper_total_return >= 0 ? "+" : ""}{(signal.paper_total_return * 100).toFixed(2)}%
              </div>
            </div>
            <div className="rounded border border-[#1a1a1a] p-2">
              <div className="stat-label"># Trades</div>
              <div className="font-mono text-[#999] mt-0.5">{signal.paper_n_trades}</div>
            </div>
          </div>

          <div className="flex-1 min-h-0 flex flex-col">
            <div className="stat-label mb-1">Equity Curve</div>
            {curveData.length > 1 ? (
              <div style={{ height: 120 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curveData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke="#141414" vertical={false} />
                    <XAxis dataKey="i" hide />
                    <YAxis
                      tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
                      tick={{ fill: "#444", fontSize: 9 }}
                      axisLine={false}
                      tickLine={false}
                      width={40}
                    />
                    <Tooltip
                      contentStyle={{ background: "#111", border: "1px solid #1a1a1a", fontSize: 10 }}
                      formatter={(v: number) => [`$${v.toLocaleString()}`, "Portfolio"]}
                    />
                    <Line type="monotone" dataKey="value" stroke={pnlColor} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-[10px] text-[#333]">Not enough trades yet for a curve</div>
            )}
          </div>

          <div className="border-t border-[#141414] pt-2">
            <div className="stat-label mb-1">Open Position</div>
            {openPosition ? (
              <div className="flex items-center gap-2 text-[11px]">
                <span className="font-semibold uppercase" style={{ color: actionColor(openPosition.action) }}>{openPosition.action}</span>
                <span className="text-[#888]">{(openPosition.position_size * 100).toFixed(1)}%</span>
                <span className="text-[#555] font-mono ml-auto">{fmtPrice(openPosition.price)}</span>
              </div>
            ) : (
              <span className="text-[11px] text-[#444]">Flat</span>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

// ── Panel C1 — Alpaca Order History (real practice-account fills, no MARTA reasoning) ──

function AlpacaOrdersPanel() {
  const [orders, setOrders] = useState<AlpacaOrder[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getAlpacaOrders(20).then((res) => setOrders(res.orders)).catch(() => setOrders([])).finally(() => setLoading(false));
  }, []);

  return (
    <Panel title="Trade Log">
      {loading ? (
        <div className="p-1"><SkeletonRows rows={8} /></div>
      ) : orders.length === 0 ? (
        <EmptyState title="No Alpaca orders yet" description="Real practice-account fills placed by the hourly scheduler will appear here." />
      ) : (
        <div className="overflow-auto max-h-[520px]">
          <table className="data-table">
            <thead>
              <tr>
                <th>Filled At</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Notional</th>
                <th>Avg Price</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((o) => (
                <tr key={o.id}>
                  <td className="font-mono text-[#888]">
                    {o.filled_at && o.filled_at !== "None" ? new Date(o.filled_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}
                  </td>
                  <td className="text-[#777]">{o.symbol}</td>
                  <td className="font-semibold uppercase" style={{ color: actionColor(o.side === "OrderSide.BUY" ? "long" : "short") }}>
                    {o.side.replace("OrderSide.", "")}
                  </td>
                  <td className="font-mono text-[#999]">${o.notional}</td>
                  <td className="font-mono text-[#999]">{o.filled_avg_price && o.filled_avg_price !== "None" ? fmtPrice(parseFloat(o.filled_avg_price)) : "—"}</td>
                  <td className="text-[#777]">{o.status.replace("OrderStatus.", "")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

// ── Panel C2 — Trade Log with Reasoning (SQLite paper_db) ────────────────────────

function TradeLogPanel({ asset }: { asset: string }) {
  const [trades, setTrades] = useState<TradeWithReasoning[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<TradeWithReasoning | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    api.getTradeHistory(asset, 20)
      .then((res) => setTrades([...res.trades].reverse()))
      .catch(() => setTrades([]))
      .finally(() => setLoading(false));
  }, [asset]);

  useEffect(() => { load(); }, [load]);

  return (
    <Panel title="Trade Log">
      {loading ? (
        <div className="p-1"><SkeletonRows rows={8} /></div>
      ) : trades.length === 0 ? (
        <EmptyState title="No trades yet" description="Trades logged by the hourly scheduler (or a manual trigger) will appear here." />
      ) : (
        <div className="overflow-auto max-h-[520px]">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Asset</th>
                <th>Action</th>
                <th>Size</th>
                <th>Price</th>
                <th>P&amp;L</th>
                <th>Regime</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => {
                const rowBg = t.action === "hold"
                  ? "rgba(120,120,120,0.05)"
                  : t.pnl > 0
                    ? "rgba(0,255,127,0.05)"
                    : t.pnl < 0
                      ? "rgba(255,55,55,0.05)"
                      : "transparent";
                return (
                  <tr key={t.id} style={{ background: rowBg }}>
                    <td className="font-mono text-[#888]">
                      {new Date(t.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                    </td>
                    <td className="text-[#777]">{asset.replace("_", "/")}</td>
                    <td className="font-semibold uppercase" style={{ color: actionColor(t.action) }}>{t.action}</td>
                    <td className="font-mono text-[#888]">{(t.position_size * 100).toFixed(1)}%</td>
                    <td className="font-mono text-[#999]">{fmtPrice(t.price)}</td>
                    <td className="font-mono" style={{ color: t.pnl >= 0 ? "#00ff7f" : "#ff3737" }}>
                      {t.pnl >= 0 ? "+" : ""}{(t.pnl * 100).toFixed(3)}%
                    </td>
                    <td>
                      <span className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: REGIME_COLORS[t.regime_label] ?? "#555" }} />
                        <span className="text-[#777]">R{t.regime_label}</span>
                      </span>
                    </td>
                    <td>
                      <button
                        onClick={() => setSelected(t)}
                        className="text-[10px] text-[#4a9eff] hover:text-[#7ab8ff] font-semibold px-1.5 py-0.5 rounded border border-[#4a9eff33] hover:border-[#4a9eff66] transition-colors"
                      >
                        Why?
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {selected && <ReasoningSlideOver trade={selected} onClose={() => setSelected(null)} />}
    </Panel>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────────

export default function TradingPage() {
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const approvalMode = useMARTAStore((s) => s.approvalMode);
  const [price, setPrice] = useState<number | null>(null);
  const [signal, setSignal] = useState<LiveSignal | null>(null);
  const [loadingSignal, setLoadingSignal] = useState(true);
  const [lastFiredAt, setLastFiredAt] = useState<string | null>(null);
  const [triggering, setTriggering] = useState(false);
  const [tradeLogKey, setTradeLogKey] = useState(0);
  const [executionSource, setExecutionSource] = useState<"alpaca" | "testnet" | "simulated">("simulated");
  const [debateRunning, setDebateRunning] = useState(false);
  const [debateStatus, setDebateStatus] = useState("");
  const [debateResult, setDebateResult] = useState<FundManagerDecision | null>(null);
  const [executing, setExecuting] = useState(false);
  // Full transcript for the live "while waiting" view below — same event shapes the
  // Debates page's DebateStream renders, fed through the one websocket this page
  // already opens (a second DebateStream instance would fire a second live debate).
  const [debateMessages, setDebateMessages] = useState<DebateMessage[]>([]);
  const [thinkingAgent, setThinkingAgent] = useState<string | null>(null);
  const debateWsRef = useRef<MartaWebSocket | null>(null);

  const loadSignal = useCallback(() => {
    setLoadingSignal(true);
    Promise.all([
      api.getLiveSignal(selectedAsset).catch(() => null),
      api.getSchedulerStatus().catch(() => null),
    ]).then(([sig, sched]) => {
      setSignal(sig);
      setLastFiredAt(sched?.last_signals?.[selectedAsset]?.scheduler_fired_at ?? null);
      const isCryptoAsset = selectedAsset === "BTC_USDT" || selectedAsset === "ETH_USDT";
      setExecutionSource(
        isCryptoAsset && sched?.alpaca_configured ? "alpaca"
        : isCryptoAsset && sched?.testnet_connected ? "testnet"
        : "simulated",
      );
    }).finally(() => setLoadingSignal(false));
  }, [selectedAsset]);

  useEffect(() => { loadSignal(); }, [loadSignal]);

  useEffect(() => {
    setPrice(null);
    const ws = new MartaWebSocket(`/ws/price/${selectedAsset}?mode=live`);
    const off = ws.on((msg) => {
      if (msg.type === "history") {
        const candles = msg.candles as Candle[];
        const last = candles.at(-1);
        if (last) setPrice(last.close);
      } else if (msg.type === "candle") {
        setPrice((msg as unknown as Candle).close);
      }
    });
    ws.connect();
    return () => { off(); ws.close(); };
  }, [selectedAsset]);

  const handleTrigger = useCallback(async () => {
    setTriggering(true);
    try {
      // Backend now awaits the tick and returns the fresh signal directly (was
      // fire-and-forget with a guessed 4s setTimeout before refetching — a slow tick
      // under load could outlast that guess and the button would look like a no-op).
      await api.triggerSignal(selectedAsset);
    } catch {
      // best-effort — surface via loadSignal() below regardless
    } finally {
      await loadSignal();
      setTradeLogKey((k) => k + 1);
      setTriggering(false);
    }
  }, [selectedAsset, loadSignal]);

  const handleRunDebateAndExecute = useCallback(() => {
    if (debateRunning || triggering) return;
    debateWsRef.current?.close();
    setDebateRunning(true);
    setDebateResult(null);
    setExecuting(false);
    setDebateStatus("Starting debate…");
    setDebateMessages([]);
    setThinkingAgent(null);

    const timestamp = new Date().toISOString();
    const path = `/ws/debate?mode=live&asset=${encodeURIComponent(selectedAsset)}&timestamp=${encodeURIComponent(timestamp)}`;
    const ws = new MartaWebSocket(path);

    const off = ws.on((msg) => {
      const type = msg.type as string;
      if (type === "agent_thinking") {
        setDebateStatus(`Agents deliberating… (${msg.agent as string})`);
        setThinkingAgent(msg.agent as string);
        return;
      }
      if (type === "debate_start") {
        setDebateStatus("");
        return;
      }
      setThinkingAgent(null);
      if (type === "fund_manager") {
        setDebateResult({
          action: msg.action as "long" | "short" | "hold",
          position_size: msg.position_size as number,
          conviction: msg.conviction as number,
          rationale: msg.rationale as string,
          regime_override: msg.regime_override as boolean,
          dissenting_agents: msg.dissenting_agents as string[],
        });
        setDebateMessages((prev) => [...prev, msg as DebateMessage]);
      } else if (type === "debate_complete") {
        off();
        ws.close(); // explicit — MartaWebSocket auto-reconnects on close by default,
        // which would silently re-run this exact live debate in a loop otherwise.
        setDebateRunning(false);
        setDebateStatus("");
        // The debate is already saved to debate_results.json by the backend at this
        // point — trigger-signal's get_live_signal() call will pick it up via the
        // automatic high-conviction override in live_signal.py.
        setExecuting(true);
        // trigger-signal now awaits the tick server-side and resolves once it's really
        // done — chain off that instead of guessing with a fixed setTimeout (see
        // handleTrigger above for the same fix + rationale).
        api.triggerSignal(selectedAsset).catch(() => {}).finally(() => {
          loadSignal();
          setTradeLogKey((k) => k + 1);
          setExecuting(false);
        });
      } else if (type === "error") {
        off();
        ws.close();
        setDebateRunning(false);
        setDebateStatus(`Error: ${msg.message as string}`);
      } else {
        setDebateMessages((prev) => [...prev, msg as DebateMessage]);
      }
    });

    ws.connect();
    debateWsRef.current = ws;
  }, [selectedAsset, debateRunning, triggering, loadSignal]);

  useEffect(() => () => debateWsRef.current?.close(), []);

  return (
    <div className="flex flex-col gap-3 pb-4 h-full min-h-0">
      <PageHeader
        title="Trading"
        description={`Live paper trading — ${selectedAsset.replace("_", "/")}`}
        actions={
          <Badge variant={executionSource === "alpaca" ? "live" : executionSource === "testnet" ? "info" : "neutral"}>
            {executionSource === "alpaca" ? "Alpaca Paper" : executionSource === "testnet" ? "Binance Testnet" : "Local Simulation"}
          </Badge>
        }
      />

      <PendingApprovalsPanel onResolved={() => { loadSignal(); setTradeLogKey((k) => k + 1); }} />

      {/* Live debate transcript — visible for the whole run (30-90s via Mistral 7B) so
          there's something to watch besides a spinner, and it stays up afterward so the
          reasoning behind the executed/pending trade above is still there to scroll back through. */}
      {(debateRunning || debateMessages.length > 0) && (
        <Panel title="Live Debate Stream" className="shrink-0">
          <DebateFeed
            messages={debateMessages}
            thinkingAgent={thinkingAgent}
            className="max-h-80 overflow-auto p-2"
          />
        </Panel>
      )}

      <div className="grid grid-cols-3 gap-2 flex-1 min-h-0">
        <LiveSignalPanel
          asset={selectedAsset}
          price={price}
          signal={signal}
          loading={loadingSignal}
          lastFiredAt={lastFiredAt}
          onTrigger={handleTrigger}
          triggering={triggering}
          onRunDebate={handleRunDebateAndExecute}
          debateRunning={debateRunning}
          debateStatus={debateStatus}
          debateResult={debateResult}
          executing={executing}
          approvalMode={approvalMode}
        />
        <PaperPortfolioPanel signal={signal} loading={loadingSignal} />
        {executionSource === "alpaca" ? (
          <AlpacaOrdersPanel key={tradeLogKey} />
        ) : (
          <TradeLogPanel key={tradeLogKey} asset={selectedAsset} />
        )}
      </div>

      <LearningPanel>
        The hourly scheduler calls the same PPO checkpoint (or the regime-rule fallback when no
        checkpoint is loaded) that produced the backtest results on the Portfolio page — this page
        is that policy running forward in real time against live prices, with every trade linked
        back to the regime and debate reasoning that produced it via the &ldquo;Why?&rdquo; button.
      </LearningPanel>
    </div>
  );
}
