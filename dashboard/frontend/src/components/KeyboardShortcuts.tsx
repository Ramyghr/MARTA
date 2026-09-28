"use client";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { X, Keyboard } from "lucide-react";
import { useMARTAStore } from "@/lib/store";

const ROUTE_KEYS: Record<string, string> = {
  d: "/dashboard",
  l: "/trading",
  b: "/debates",
  r: "/regime",
  p: "/portfolio",
  a: "/analytics",
  t: "/backtest",
  s: "/settings",
};

const SHORTCUTS: Array<{ key: string; description: string }> = [
  { key: "D", description: "Go to Dashboard" },
  { key: "L", description: "Go to Trading (Live)" },
  { key: "B", description: "Go to Debates" },
  { key: "R", description: "Go to Regime Explorer" },
  { key: "P", description: "Go to Portfolio" },
  { key: "A", description: "Go to Analytics" },
  { key: "T", description: "Go to Backtest" },
  { key: "S", description: "Go to Settings" },
  { key: "?", description: "Show this help" },
];

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}

export function KeyboardShortcuts() {
  const router = useRouter();
  const demoMode = useMARTAStore((s) => s.demoMode);
  const [helpOpen, setHelpOpen] = useState(false);

  const handleKey = useCallback((e: KeyboardEvent) => {
    if (demoMode || isTypingTarget(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "?") { e.preventDefault(); setHelpOpen((o) => !o); return; }
    if (e.key === "Escape") { setHelpOpen(false); return; }

    const key = e.key.toLowerCase();
    const path = ROUTE_KEYS[key];
    if (path) router.push(path);
  }, [demoMode, router]);

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  if (!helpOpen) return null;

  return (
    <div className="fixed inset-0 z-[90] bg-black/70 flex items-center justify-center p-6" onClick={() => setHelpOpen(false)}>
      <div className="bg-[#0d0d0d] border border-[#2a2a2a] rounded-lg w-full max-w-sm p-5" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="flex items-center gap-2 text-[13px] font-bold text-[#e0e0e0]"><Keyboard size={14} /> Keyboard Shortcuts</h2>
          <button onClick={() => setHelpOpen(false)} className="text-[#555] hover:text-[#999] transition-colors"><X size={16} /></button>
        </div>
        <div className="flex flex-col gap-1.5">
          {SHORTCUTS.map((s) => (
            <div key={s.key} className="flex items-center justify-between text-[11px]">
              <span className="text-[#888]">{s.description}</span>
              <kbd className="px-1.5 py-0.5 rounded bg-[#1a1a1a] border border-[#2a2a2a] text-[#00ff7f] font-mono text-[10px] min-w-[20px] text-center">{s.key}</kbd>
            </div>
          ))}
        </div>
        <div className="text-[10px] text-[#444] mt-3 pt-2 border-t border-[#161616]">Disabled while typing in a field, or during Demo Mode (which has its own shortcuts).</div>
      </div>
    </div>
  );
}
