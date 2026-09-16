"use client";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { Presentation, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { useMARTAStore } from "@/lib/store";
import { api } from "@/lib/api";
import { WsStatusBadge } from "@/components/ui/WsStatusBadge";
import { NotificationBell } from "@/components/NotificationBell";
import type { AssetInfo } from "@/lib/types";

const TYPE_COLORS: Record<string, string> = {
  crypto: "#00ff7f",
  forex: "#4a9eff",
  commodity: "#ffd700",
};

const BREADCRUMB_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  trading: "Trading",
  debates: "Debates",
  regime: "Regime",
  portfolio: "Portfolio",
  analytics: "Analytics",
  backtest: "Backtest",
  settings: "Settings",
};

function useUtcClock() {
  const [time, setTime] = useState("");
  useEffect(() => {
    const tick = () => setTime(new Date().toISOString().slice(11, 19));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

export function Topbar() {
  const pathname = usePathname();
  const selectedAsset = useMARTAStore((s) => s.selectedAsset);
  const setSelectedAsset = useMARTAStore((s) => s.setSelectedAsset);
  const setDemoMode = useMARTAStore((s) => s.setDemoMode);
  const sidebarCollapsed = useMARTAStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useMARTAStore((s) => s.toggleSidebar);

  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const utcTime = useUtcClock();

  useEffect(() => {
    api.getAssets().then((d) => setAssets(d.assets)).catch(() => {});
  }, []);

  const currentAsset = assets.find((a) => a.id === selectedAsset);
  const assetColor = currentAsset ? TYPE_COLORS[currentAsset.type] : "#00ff7f";
  const crumb = pathname?.split("/").filter(Boolean)[0] ?? "dashboard";

  return (
    <header
      className="app-shell-topbar h-12 shrink-0 px-4 flex items-center gap-4 border-b"
      style={{
        background: "linear-gradient(180deg, #0f0f0f 0%, #0a0a0a 100%)",
        borderColor: "rgba(0,255,127,0.18)",
      }}
    >
      {/* Sidebar toggle */}
      <button
        onClick={toggleSidebar}
        title={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
        className="shrink-0 p-1 rounded text-[#555] hover:text-[#00ff7f] hover:bg-[#00ff7f]/10 transition-colors"
      >
        {sidebarCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
      </button>

      {/* Breadcrumb */}
      <div className="text-[11px] text-[#555] flex items-center gap-1.5 shrink-0">
        <span className="text-[#333]">MARTA</span>
        <span className="text-[#2a2a2a]">/</span>
        <span className="text-[#999] font-medium">{BREADCRUMB_LABELS[crumb] ?? crumb}</span>
      </div>

      {/* Asset selector (center) */}
      <div className="flex-1 flex items-center justify-center gap-1.5">
        <select
          value={selectedAsset}
          onChange={(e) => setSelectedAsset(e.target.value)}
          className="bg-[#111] border border-[#2a2a2a] rounded px-2 py-1 text-[11px] outline-none cursor-pointer"
          style={{ color: assetColor }}
        >
          {assets.length === 0
            ? <option value="BTC_USDT">BTC/USDT</option>
            : assets.map((a) => (
                <option key={a.id} value={a.id} style={{ color: TYPE_COLORS[a.type] }}>
                  {a.label}
                </option>
              ))
          }
        </select>
        {currentAsset && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded uppercase"
            style={{ background: `${TYPE_COLORS[currentAsset.type]}18`, color: TYPE_COLORS[currentAsset.type] }}
          >
            {currentAsset.type}
          </span>
        )}

        <span className="ml-2 text-[10px] px-2 py-0.5 rounded uppercase font-bold tracking-wider bg-[#00ff7f]/10 text-[#00ff7f] animate-pulse-dot">
          ● Live
        </span>
      </div>

      {/* Right cluster */}
      <div className="flex items-center gap-3 shrink-0">
        <span className="text-[10px] text-[#333] hidden lg:inline font-mono tabular-nums">{utcTime} UTC</span>

        <NotificationBell />

        <button
          onClick={() => setDemoMode(true)}
          title="Enter presentation mode"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-semibold uppercase tracking-wider bg-[#b07fff]/10 text-[#b07fff] border border-[#b07fff]/30 hover:bg-[#b07fff]/20 transition-colors"
        >
          <Presentation size={11} /> Demo Mode
        </button>

        <WsStatusBadge />
      </div>
    </header>
  );
}
