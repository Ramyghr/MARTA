"use client";
import { useCallback, useEffect, useState } from "react";
import { Check, X } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import { useMARTAStore } from "@/lib/store";
import type { PendingTrade } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/types";

function actionColor(action: string): string {
  return action === "long" ? "#00ff7f" : action === "short" ? "#ff3737" : "#666666";
}

// Shown whenever approval mode is on, or a leftover pending trade exists from before
// it was turned off — those still need a human decision, so they can't just vanish.
export function PendingApprovalsPanel({ onResolved }: { onResolved?: () => void }) {
  const approvalMode = useMARTAStore((s) => s.approvalMode);
  const [pending, setPending] = useState<PendingTrade[]>([]);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(() => {
    api.getPendingTrades("pending")
      .then((r) => setPending(r.trades))
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 20_000);
    return () => clearInterval(id);
  }, [load]);

  const act = useCallback(async (id: number, kind: "approve" | "reject") => {
    setBusyId(id);
    try {
      if (kind === "approve") await api.approvePendingTrade(id);
      else await api.rejectPendingTrade(id);
      load();
      onResolved?.();
    } finally {
      setBusyId(null);
    }
  }, [load, onResolved]);

  if (!loaded || (!approvalMode && pending.length === 0)) return null;

  return (
    <Panel title={`Pending Approvals${pending.length ? ` (${pending.length})` : ""}`}>
      {pending.length === 0 ? (
        <EmptyState
          title="Nothing awaiting approval"
          description={approvalMode ? "New non-hold signals will appear here for review." : "Approval mode is off — signals execute immediately."}
        />
      ) : (
        <div className="flex flex-col gap-2">
          {pending.map((p) => (
            <div key={p.id} className="flex flex-wrap items-center gap-3 rounded border border-[#ff8c0033] bg-[#ff8c0008] px-3 py-2 text-[11px]">
              <span className="text-[#888] w-16 shrink-0">{p.asset.replace("_", "/")}</span>
              <span className="font-bold uppercase w-14 shrink-0" style={{ color: actionColor(p.action) }}>{p.action}</span>
              <span className="text-[#888] w-14 shrink-0">{(p.position_size * 100).toFixed(1)}%</span>
              <span className="text-[#666] w-20 shrink-0 font-mono">${p.price.toFixed(2)}</span>
              <span className="flex items-center gap-1 shrink-0">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: REGIME_COLORS[p.regime_label] ?? "#555" }} />
                <span className="text-[#666]">R{p.regime_label}</span>
              </span>
              <span className="text-[#3a3a3a] shrink-0">{new Date(p.created_at).toLocaleTimeString()}</span>
              <div className="flex items-center gap-1.5 ml-auto shrink-0">
                <button
                  onClick={() => act(p.id, "approve")}
                  disabled={busyId === p.id}
                  className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-semibold uppercase bg-[#00ff7f]/10 text-[#00ff7f] border border-[#00ff7f]/30 hover:bg-[#00ff7f]/20 transition-colors disabled:opacity-40"
                >
                  <Check size={11} /> Approve
                </button>
                <button
                  onClick={() => act(p.id, "reject")}
                  disabled={busyId === p.id}
                  className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-semibold uppercase bg-[#ff3737]/10 text-[#ff3737] border border-[#ff3737]/30 hover:bg-[#ff3737]/20 transition-colors disabled:opacity-40"
                >
                  <X size={11} /> Reject
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
