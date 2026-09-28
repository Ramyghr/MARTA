"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  LayoutDashboard, MessagesSquare, Waves, Wallet, FlaskConical, Settings, Cpu,
  Zap, BarChart3, ChevronUp, User, LogOut,
} from "lucide-react";
import { useMARTAStore } from "@/lib/store";
import { fetchApi } from "@/lib/websocket";
import { useCurrentUser, logout } from "@/lib/auth";
import { MartaLogo } from "@/components/Logo";

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

function SidebarProfile() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { user, loading } = useCurrentUser();

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  if (loading) {
    return <div className="px-3 py-2.5 h-[52px] animate-pulse" />;
  }

  if (!user) {
    return (
      <Link
        href="/login"
        className="flex items-center gap-2 px-3 py-2.5 text-[11px] text-[#666] hover:text-[#00ff7f] hover:bg-[#00ff7f]/5 transition-colors"
      >
        <User size={14} /> Log in
      </Link>
    );
  }

  const initials = (user.display_name || user.email).slice(0, 2).toUpperCase();

  return (
    <div className="relative px-2 py-2" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-1.5 py-1.5 rounded text-left hover:bg-[#161616] transition-colors"
      >
        <span
          className="w-6 h-6 shrink-0 rounded-full flex items-center justify-center text-[10px] font-bold"
          style={{ background: "rgba(0,255,127,0.12)", color: "#00ff7f" }}
        >
          {initials}
        </span>
        <span className="flex-1 min-w-0">
          <div className="text-[11px] text-[#ccc] font-medium truncate">{user.display_name || user.email}</div>
        </span>
        <ChevronUp size={12} className={`text-[#555] transition-transform ${open ? "" : "rotate-180"}`} />
      </button>

      {open && (
        <div className="absolute bottom-full left-2 right-2 mb-1 rounded border border-[#2a2a2a] bg-[#111] shadow-lg overflow-hidden">
          <Link
            href="/profile"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-3 py-2 text-[11px] text-[#ccc] hover:bg-[#1a1a1a] transition-colors"
          >
            <User size={12} /> Profile Settings
          </Link>
          <button
            onClick={async () => {
              setOpen(false);
              await logout();
              // Full navigation, not router.push — AuthGate's useCurrentUser() is a
              // separate hook instance that won't otherwise know the session is gone.
              window.location.href = "/login";
            }}
            className="w-full flex items-center gap-2 px-3 py-2 text-[11px] text-[#ff6b6b] hover:bg-[#1a1a1a] transition-colors"
          >
            <LogOut size={12} /> Log out
          </button>
        </div>
      )}
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
          <MartaLogo className="w-6 h-6 shrink-0" />
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

        <div className="border-t border-[#1a1a1a] shrink-0">
          <SidebarProfile />
        </div>

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
