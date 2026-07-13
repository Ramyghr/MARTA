"use client";
import { useEffect, useState, type ReactNode } from "react";
import { ChevronDown, GraduationCap } from "lucide-react";
import { useMARTAStore } from "@/lib/store";

interface LearningPanelProps {
  children: ReactNode;
}

/**
 * Collapsible "How it works" explainer. Defaults collapsed; auto-expands (and
 * locks open) while Demo Mode is active, since that's when an audience benefits
 * from the plain-English framing most.
 */
export function LearningPanel({ children }: LearningPanelProps) {
  const demoMode = useMARTAStore((s) => s.demoMode);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (demoMode) setOpen(true);
  }, [demoMode]);

  return (
    <div className="rounded border border-[#1a1a1a] bg-[#0d0d0d] shrink-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        <GraduationCap size={12} className="text-[#4a9eff] shrink-0" />
        <span className="text-[10px] uppercase tracking-wider text-[#666] font-semibold">How it works</span>
        <ChevronDown
          size={13}
          className="ml-auto text-[#444] transition-transform shrink-0"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>
      {open && (
        <div className="px-3 pb-3 text-[11px] text-[#888] leading-relaxed border-t border-[#141414] pt-2">
          {children}
        </div>
      )}
    </div>
  );
}
