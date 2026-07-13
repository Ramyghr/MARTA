"use client";
import { useEffect, type ReactNode } from "react";
import { useMARTAStore } from "@/lib/store";
import { api } from "@/lib/api";
import { REGIME_NAMES } from "@/lib/types";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { PageTransition } from "./PageTransition";

export function AppShell({ children }: { children: ReactNode }) {
  const sidebarCollapsed = useMARTAStore((s) => s.sidebarCollapsed);
  const setRegimeNames = useMARTAStore((s) => s.setRegimeNames);
  const setApprovalMode = useMARTAStore((s) => s.setApprovalMode);

  // Regime semantics shift every time the HMM is refit — pull the live, dynamically
  // generated descriptions once at app boot instead of trusting the static REGIME_NAMES
  // fallback everywhere. Falls back to the static map if the semantics file is missing.
  useEffect(() => {
    api.getRegimeSemantics().then((sem) => {
      if (!sem?.regimes) return;
      const names: Record<number, string> = { ...REGIME_NAMES };
      for (const [label, entry] of Object.entries(sem.regimes)) {
        if (entry.description) names[Number(label)] = entry.description;
      }
      setRegimeNames(names);
    }).catch(() => {});
  }, [setRegimeNames]);

  // Approval mode is server-persisted (the scheduler must see it with no browser
  // open) — hydrate the store's mirror of it once at app boot.
  useEffect(() => {
    api.getApprovalMode().then((s) => setApprovalMode(s.approval_mode)).catch(() => {});
  }, [setApprovalMode]);

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar />
      <Topbar />
      <main className="app-shell-main">
        <PageTransition>{children}</PageTransition>
      </main>
    </div>
  );
}
