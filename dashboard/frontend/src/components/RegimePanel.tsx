"use client";
import { useEffect, useRef, useState } from "react";
import { REGIME_COLORS, type RegimeState } from "@/lib/types";
import { useMARTAStore } from "@/lib/store";

interface Props {
  regime: RegimeState | null;
}

export default function RegimePanel({ regime }: Props) {
  const regimeNames = useMARTAStore((s) => s.regimeNames);
  const prevRegimeRef = useRef<number | null>(null);
  const [recentShift, setRecentShift] = useState(false);

  useEffect(() => {
    if (!regime) return;
    const prev = prevRegimeRef.current;
    if (prev !== null && prev !== regime.dominant_regime) {
      setRecentShift(true);
      const t = setTimeout(() => setRecentShift(false), 30_000);
      return () => clearTimeout(t);
    }
    prevRegimeRef.current = regime.dominant_regime;
  }, [regime?.dominant_regime]);

  if (!regime) {
    return (
      <div className="panel h-full flex flex-col">
        <div className="panel-header">Regime Probabilities</div>
        <div className="flex-1 flex items-center justify-center text-[#444]">Loading...</div>
      </div>
    );
  }

  const entries = Object.entries(regime.probabilities)
    .map(([k, v]) => {
      const label = parseInt(k.replace("regime_", ""));
      return { label, prob: v, name: regimeNames[label] ?? k, color: REGIME_COLORS[label] ?? "#666" };
    })
    .sort((a, b) => a.label - b.label);

  const dominant = regime.dominant_regime;
  const dominantColor = REGIME_COLORS[dominant] ?? "#666";
  const dominantName = regimeNames[dominant] ?? `Regime ${dominant}`;

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header flex items-center justify-between">
        <span>Regime</span>
        {recentShift && (
          <span className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider animate-fade-in"
            style={{ background: "rgba(255,140,0,0.15)", color: "#ff8c00", border: "1px solid #ff8c0033" }}>
            Regime Shift ↗
          </span>
        )}
      </div>

      {/* Dominant regime hero */}
      <div className="px-3 pt-4 pb-3 flex flex-col items-center gap-2 border-b border-[#1a1a1a] shrink-0">
        {/* Pulse ring + circle */}
        <div style={{ position: "relative", display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
          {/* Animated pulse ring */}
          <div style={{
            position: "absolute",
            inset: -6,
            borderRadius: "50%",
            border: `2px solid ${dominantColor}`,
            animation: "pulse-ring 2s ease-out infinite",
            pointerEvents: "none",
          }} />
          {/* Static ring */}
          <div style={{
            position: "absolute",
            inset: -6,
            borderRadius: "50%",
            border: `1px solid ${dominantColor}33`,
            pointerEvents: "none",
          }} />
          {/* Inner circle */}
          <div style={{
            width: 76, height: 76, borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
            background: `${dominantColor}12`,
            border: `2px solid ${dominantColor}`,
          }}>
            <span style={{ fontSize: 32, fontWeight: 700, color: dominantColor, letterSpacing: "-0.02em" }}>
              R{dominant}
            </span>
          </div>
        </div>

        <div className="text-center mt-1">
          <div className="font-bold text-[13px]" style={{ color: dominantColor }}>
            {dominantName}
          </div>
          {regime.regime_description && (
            <div className="text-[10px] text-[#444] mt-0.5 leading-snug px-2 text-center">
              {regime.regime_description}
            </div>
          )}
        </div>
      </div>

      {/* Probability bars */}
      <div className="flex-1 px-3 py-2 flex flex-col gap-1.5 overflow-auto">
        {entries.map(({ label, prob, name, color }) => {
          const pct = Math.round(prob * 100);
          const isDominant = label === dominant;
          return (
            <div key={label}
              className={`rounded px-2 py-1.5 transition-all ${isDominant ? "animate-fade-in" : ""}`}
              style={isDominant ? { background: `${color}0c`, border: `1px solid ${color}33` } : { border: "1px solid transparent" }}
            >
              <div className="flex items-center gap-2">
                {/* Label */}
                <span className="flex items-center gap-1.5 text-[10px] w-24 shrink-0">
                  <span className="w-1.5 h-1.5 rounded-full inline-block shrink-0" style={{ background: color }} />
                  <span className={isDominant ? "text-[#ccc]" : "text-[#555]"}>
                    R{label} {name}
                  </span>
                </span>
                {/* Bar */}
                <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "#1a1a1a" }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${pct}%`,
                      background: isDominant ? color : `${color}66`,
                      boxShadow: isDominant ? `0 0 6px ${color}55` : "none",
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
                {/* Pct */}
                <span className="text-[10px] font-mono w-8 text-right shrink-0"
                  style={{ color: isDominant ? color : "#444" }}>
                  {pct}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
