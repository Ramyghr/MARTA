"use client";
import type { Portfolio } from "@/lib/types";

// "Which run produced these numbers" — checkpoint filename + obs_mode + eval window,
// straight from ppo_evaluation.json's own _provenance/val_window fields (rl/evaluate.py
// already writes these). Meant to sit in a Panel's `actions` slot next to whatever
// Sharpe/return numbers it's describing, so a number is never shown with no way to
// tell which config produced it.
export function ProvenanceTag({ portfolio }: { portfolio: Portfolio | null }) {
  if (!portfolio || (!portfolio.provenance && !portfolio.val_window)) return null;
  const checkpoint = portfolio.provenance?.checkpoint.path.split("/").pop();
  const obsMode = portfolio.provenance?.env_config.obs_mode;
  const win = portfolio.val_window;

  return (
    <span
      className="inline-flex items-center gap-1 text-[9px] text-[#666] font-mono px-1.5 py-0.5 rounded border border-[#1a1a1a] bg-[#0d0d0d] whitespace-nowrap"
      title={portfolio.provenance ? `Generated ${new Date(portfolio.provenance.generated_at_utc).toLocaleString()}` : undefined}
    >
      {checkpoint && <span className="text-[#999]">{checkpoint}</span>}
      {obsMode && <span>· {obsMode} obs</span>}
      {win && <span>· val {win.start} → {win.end}</span>}
    </span>
  );
}
