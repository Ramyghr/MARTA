"use client";
import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/Panel";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonRows } from "@/components/ui/Skeleton";
import { api } from "@/lib/api";
import type { TradeWithReasoning } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";
import { useMARTAStore } from "@/lib/store";
import { X } from "lucide-react";

export function fmtPrice(v: number | null | undefined): string {
  if (!v) return "—";
  return v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(2)}`;
}

export function sourceLabel(source?: string): string {
  if (!source) return "—";
  if (source.startsWith("ppo")) return "PPO";
  if (source.startsWith("regime_policy")) return "Regime rules";
  if (source.startsWith("debate_override")) return "Debate override";
  return source;
}

export function outcomeBadge(outcome?: string | null) {
  const color = outcome === "profit" ? "#00ff7f" : outcome === "loss" ? "#ff3737" : "#666";
  const label = outcome === "profit" ? "Profit" : outcome === "loss" ? "Loss" : "Open";
  return (
    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase"
      style={{ background: `${color}20`, color, border: `1px solid ${color}44` }}>
      {label}
    </span>
  );
}

// ── Slide-over: full reasoning for one trade ──────────────────────────────────

export function ReasoningSlideOver({ trade, onClose }: { trade: TradeWithReasoning; onClose: () => void }) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const regimeColor = REGIME_COLORS[trade.regime_label] ?? "#555";
  const regimeName = regimeNames[trade.regime_label] ?? "Unknown";

  return (
    <div className="fixed inset-0 z-[95] bg-black/60" onClick={onClose}>
      <div
        className="absolute right-0 top-0 h-full w-full max-w-md bg-[#0a0a0a] border-l border-[#1e1e1e] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-[#0a0a0a] border-b border-[#1a1a1a] px-4 py-3 flex items-center justify-between">
          <div>
            <div className="text-[12px] font-semibold text-[#ccc]">Trade #{trade.id} reasoning</div>
            <div className="text-[10px] text-[#555] font-mono">
              {new Date(trade.timestamp).toLocaleString()}
            </div>
          </div>
          <button onClick={onClose} className="text-[#555] hover:text-[#ccc] transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="p-4 space-y-5">
          {/* Regime at decision time */}
          <section>
            <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Regime at decision</div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: regimeColor }} />
              <span className="text-[12px] text-[#ccc]">R{trade.regime_label} — {regimeName}</span>
            </div>
          </section>

          {/* Signal source */}
          <section>
            <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Signal source</div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2 py-0.5 rounded text-[11px] font-semibold text-[#ccc] bg-[#151515] border border-[#222]">
                {sourceLabel(trade.source)}
              </span>
              {trade.ppo_override && (
                <span className="px-2 py-0.5 rounded text-[10px] text-[#ffd700] bg-[#ffd70015] border border-[#ffd70044]">
                  PPO overridden by debate
                </span>
              )}
            </div>
          </section>

          {/* Agent theses */}
          {trade.agent_theses.length > 0 && (
            <section>
              <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Agent theses</div>
              <div className="space-y-2">
                {trade.agent_theses.map((t, i) => {
                  const biasColor = t.bias === "bullish" ? "#00ff7f" : t.bias === "bearish" ? "#ff3737" : "#888";
                  return (
                    <div key={i} className="rounded border border-[#1a1a1a] p-2.5">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[11px] font-semibold text-[#bbb]">{t.agent ?? "Agent"}</span>
                        <span className="text-[10px] font-semibold uppercase" style={{ color: biasColor }}>
                          {t.bias ?? "neutral"}
                        </span>
                      </div>
                      {typeof t.confidence === "number" && (
                        <div className="h-1 rounded bg-[#151515] overflow-hidden mb-1.5">
                          <div className="h-full rounded" style={{ width: `${t.confidence * 100}%`, background: biasColor, opacity: 0.7 }} />
                        </div>
                      )}
                      {t.reasoning && <div className="text-[10px] text-[#666] leading-relaxed">{t.reasoning}</div>}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Fund manager rationale */}
          {trade.fund_manager_rationale && (
            <section>
              <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Fund Manager rationale</div>
              <div className="text-[11px] text-[#999] leading-relaxed rounded border border-[#1a1a1a] p-2.5">
                {trade.fund_manager_rationale}
              </div>
            </section>
          )}

          {trade.agent_theses.length === 0 && !trade.fund_manager_rationale && (
            <div className="text-[10px] text-[#3a3a3a] italic">
              No debate ran near this trade — signal came from {sourceLabel(trade.source).toLowerCase()} alone.
            </div>
          )}

          {/* Outcome */}
          <section className="border-t border-[#141414] pt-3">
            <div className="text-[10px] text-[#444] uppercase tracking-wider mb-1.5">Outcome</div>
            <div className="flex items-center gap-3 flex-wrap">
              {outcomeBadge(trade.outcome)}
              <span className="text-[11px] text-[#888]">
                Entry {fmtPrice(trade.entry_price)}
                {trade.exit_price ? ` → Exit ${fmtPrice(trade.exit_price)}` : ""}
              </span>
              {typeof trade.realized_pnl === "number" && (
                <span className="text-[11px] font-mono font-semibold" style={{ color: trade.realized_pnl >= 0 ? "#00ff7f" : "#ff3737" }}>
                  {trade.realized_pnl >= 0 ? "+" : ""}{(trade.realized_pnl * 100).toFixed(2)}%
                </span>
              )}
              {typeof trade.holding_hours === "number" && (
                <span className="text-[10px] text-[#555]">held {trade.holding_hours.toFixed(1)}h</span>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

// ── Trade log with Why? buttons ────────────────────────────────────────────────

export function TradeReasoningPanel({ asset }: { asset: string }) {
  const [trades, setTrades] = useState<TradeWithReasoning[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<TradeWithReasoning | null>(null);

  useEffect(() => {
    setLoading(true);
    api.getTradeHistory(asset, 50)
      .then((res) => setTrades([...res.trades].reverse()))
      .catch(() => setTrades([]))
      .finally(() => setLoading(false));
  }, [asset]);

  return (
    <Panel title={`Live Paper Trades — ${asset.replace("_", "/")}`}>
      {loading ? (
        <div className="p-1"><SkeletonRows rows={6} /></div>
      ) : trades.length === 0 ? (
        <EmptyState
          title="No live paper trades yet"
          description="The hourly signal scheduler will populate this once a trade fires for this asset."
        />
      ) : (
        <div className="overflow-auto max-h-[360px]">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Action</th>
                <th>Size</th>
                <th>Price</th>
                <th>Source</th>
                <th>Outcome</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => {
                const actionColor = t.action === "long" ? "#00ff7f" : t.action === "short" ? "#ff3737" : "#666";
                return (
                  <tr key={t.id}>
                    <td className="font-mono text-[#888]">
                      {new Date(t.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                    </td>
                    <td className="font-semibold uppercase" style={{ color: actionColor }}>{t.action}</td>
                    <td className="font-mono text-[#888]">{(t.position_size * 100).toFixed(1)}%</td>
                    <td className="font-mono text-[#999]">{fmtPrice(t.price)}</td>
                    <td className="text-[#777] text-[10px]">{sourceLabel(t.source)}</td>
                    <td>{outcomeBadge(t.outcome)}</td>
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
