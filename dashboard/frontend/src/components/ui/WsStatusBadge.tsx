"use client";
import { useEffect, useState } from "react";
import { useMARTAStore } from "@/lib/store";

function timeAgo(ts: number | null): string {
  if (!ts) return "—";
  const sec = Math.floor((Date.now() - ts) / 1000);
  if (sec < 5) return "just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  return `${Math.floor(min / 60)}h ago`;
}

export function WsStatusBadge() {
  const status = useMARTAStore((s) => s.wsStatus);
  const lastUpdate = useMARTAStore((s) => s.wsLastUpdate);
  const [, forceTick] = useState(0);

  // Re-render periodically so the relative "last updated" label stays fresh
  useEffect(() => {
    const id = setInterval(() => forceTick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);

  if (status === "reconnecting") {
    return (
      <span className="flex items-center gap-1.5 text-[10px] text-[#ff8c00]">
        <span className="w-1.5 h-1.5 rounded-full bg-[#ff8c00] animate-pulse-dot" />
        Reconnecting…
      </span>
    );
  }

  const color = status === "open" ? "#00ff7f" : "#333";
  const label = status === "open" ? "Connected" : status === "connecting" ? "Connecting…" : "Idle";

  return (
    <span className="flex items-center gap-1.5 text-[10px]">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      <span className="text-[#444]">{label}</span>
      {status === "open" && lastUpdate && <span className="text-[#333]">· {timeAgo(lastUpdate)}</span>}
    </span>
  );
}
