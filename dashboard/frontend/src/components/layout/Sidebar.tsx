"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  LayoutDashboard, MessagesSquare, Waves, Wallet, FlaskConical, Settings, Cpu,
  Zap, BarChart3,
} from "lucide-react";
import { useMARTAStore } from "@/lib/store";
import { fetchApi } from "@/lib/websocket";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/trading", label: "Trading", icon: Zap },
  { href: "/debates", label: "Debates", icon: MessagesSquare },
  { href: "/regime", label: "Regime", icon: Waves },
  { href: "/portfolio", label: "Portfolio", icon: Wallet },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/backtest", label: "Backtest", icon: FlaskConical },
  { href: "/settings", label: "Settings", icon: Settings },
];

function useBackendReachable() {
  const [reachable, setReachable] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    const ping = () => {
      fetchApi("/api/assets")
        .then(() => { if (!cancelled) setReachable(true); })
        .catch(() => { if (!cancelled) setReachable(false); });
    };
    ping();
    const id = setInterval(ping, 30_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);
  return reachable;
}

function StatusRow({ label, ok, okLabel, badLabel }: { label: string; ok: boolean | null; okLabel: string; badLabel: string }) {
  const color = ok === null ? "#444" : ok ? "#00ff7f" : "#ff3737";
  return (
    <div className="flex items-center justify-between text-[10px] py-1">
      <span className="text-[#444] uppercase tracking-wider">{label}</span>
      <span className="flex items-center gap-1.5" style={{ color }}>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: color, boxShadow: ok ? `0 0 6px ${color}` : "none" }} />
        {ok === null ? "…" : ok ? okLabel : badLabel}
      </span>
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const wsConnected = useMARTAStore((s) => s.wsConnected);
  const sidebarCollapsed = useMARTAStore((s) => s.sidebarCollapsed);
  const backendUp = useBackendReachable();

  return (
    <aside
      className={`app-shell-sidebar shrink-0 h-full overflow-hidden border-r border-[#1a1a1a] bg-[#0f0f0f] transition-all duration-200 ${
        sidebarCollapsed ? "w-0 border-r-0" : "w-[240px]"
      }`}
    >
      {/* Fixed-width inner content — the aside above just clips it during the collapse transition */}
      <div className="w-[240px] h-full flex flex-col">
        {/* Logo */}
        <div className="flex items-center gap-2 px-4 h-12 shrink-0 border-b border-[#1a1a1a]">
          <span
            className="w-1.5 h-1.5 rounded-full inline-block shrink-0 animate-pulse-dot"
            style={{ background: "#00ff7f", boxShadow: "0 0 6px #00ff7f" }}
          />
          <span className="font-bold text-[#00ff7f] tracking-widest text-sm glow-green">MARTA</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3 px-2 flex flex-col gap-0.5 overflow-y-auto">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || pathname?.startsWith(`${href}/`);
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-2.5 px-2.5 py-2 rounded text-[12px] transition-colors border-l-2 ${
                  active
                    ? "border-green-400 bg-[#00ff7f]/10 text-green-400 font-semibold"
                    : "border-transparent text-gray-300 font-normal"
                }`}
              >
                <Icon size={14} strokeWidth={2} className={active ? "text-green-400" : "text-gray-300"} />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* System status */}
        <div className="px-3 py-3 border-t border-[#1a1a1a] shrink-0">
          <StatusRow label="WebSocket" ok={wsConnected} okLabel="Connected" badLabel="Idle" />
          <StatusRow label="Backend API" ok={backendUp} okLabel="Online" badLabel="Unreachable" />
          <div className="flex items-center gap-1.5 text-[10px] text-[#3a3a3a] pt-1.5 mt-1 border-t border-[#161616]">
            <Cpu size={11} strokeWidth={2} />
            Mistral 7B · Ollama
          </div>
        </div>
      </div>
    </aside>
  );
}
