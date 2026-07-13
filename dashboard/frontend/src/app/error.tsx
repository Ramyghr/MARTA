"use client";
import { useEffect } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error("[MARTA] page error:", error);
  }, [error]);

  return (
    <div className="h-full min-h-[50vh] flex flex-col items-center justify-center gap-3 text-center p-8">
      <AlertTriangle size={28} strokeWidth={1.5} className="text-[#ff3737]" />
      <div className="text-[13px] font-semibold text-[#e0e0e0]">Something went wrong rendering this page</div>
      <div className="text-[11px] text-[#666] max-w-md leading-relaxed font-mono">
        {error.message || "Unknown error"}
      </div>
      <button
        onClick={reset}
        className="flex items-center gap-1.5 mt-2 px-3 py-1.5 rounded text-[11px] font-semibold uppercase tracking-wider bg-[#4a9eff]/10 text-[#4a9eff] border border-[#4a9eff]/30 hover:bg-[#4a9eff]/20 transition-colors"
      >
        <RotateCcw size={12} /> Try again
      </button>
    </div>
  );
}
