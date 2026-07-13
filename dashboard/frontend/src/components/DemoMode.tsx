"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { X, Maximize, Minimize, Pause, Play, ChevronLeft, ChevronRight } from "lucide-react";
import { useMARTAStore } from "@/lib/store";
import { api } from "@/lib/api";
import DebateStream from "@/components/DebateStream";
import RegimePanel from "@/components/RegimePanel";
import DecisionPanel from "@/components/DecisionPanel";
import type { Scenario, RegimeState, FundManagerDecision } from "@/lib/types";

const CYCLE_MS = 60_000;
const SCENARIO_COUNT = 5;

export function DemoMode() {
  const demoMode = useMARTAStore((s) => s.demoMode);
  const setDemoMode = useMARTAStore((s) => s.setDemoMode);

  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [regime, setRegime] = useState<RegimeState | null>(null);
  const [decision, setDecision] = useState<FundManagerDecision | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!demoMode) return;
    setIndex(0);
    api.getDemoScenarios()
      .then((d) => setScenarios(d.scenarios.slice(0, SCENARIO_COUNT)))
      .catch(() => setScenarios([]));
  }, [demoMode]);

  // Auto-cycle every 60s unless paused
  useEffect(() => {
    if (!demoMode || paused || scenarios.length < 2) return;
    const id = setInterval(() => setIndex((i) => (i + 1) % scenarios.length), CYCLE_MS);
    return () => clearInterval(id);
  }, [demoMode, paused, scenarios.length]);

  const current = scenarios[index] ?? null;

  useEffect(() => {
    if (!current) { setRegime(null); return; }
    setDecision(null);
    api.getRegimeAt(current.asset, current.timestamp).then(setRegime).catch(() => setRegime(null));
  }, [current]);

  const handleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onChange = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const next = useCallback(() => setIndex((i) => (i + 1) % Math.max(scenarios.length, 1)), [scenarios.length]);
  const prev = useCallback(() => setIndex((i) => (i - 1 + scenarios.length) % Math.max(scenarios.length, 1)), [scenarios.length]);

  useEffect(() => {
    if (!demoMode) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ") { e.preventDefault(); setPaused((p) => !p); }
      else if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key.toLowerCase() === "f") handleFullscreen();
      else if (e.key === "Escape") setDemoMode(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [demoMode, next, prev, handleFullscreen, setDemoMode]);

  if (!demoMode) return null;

  return (
    <div ref={containerRef} className="fixed inset-0 z-[100] bg-[#0a0a0a] flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 h-14 border-b border-[#1a1a1a] shrink-0">
        <span className="w-2 h-2 rounded-full inline-block shrink-0 animate-pulse-dot" style={{ background: "#00ff7f", boxShadow: "0 0 8px #00ff7f" }} />
        <span className="text-[#00ff7f] font-bold tracking-widest text-sm glow-green">MARTA DEMO</span>
        {current && (
          <span className="text-[12px] text-[#888]">
            Scenario {index + 1}/{scenarios.length}: {current.label}
            {current.description && <span className="text-[#555]"> — {current.description}</span>}
          </span>
        )}
        {paused && <span className="text-[10px] px-2 py-0.5 rounded bg-[#ff8c0018] text-[#ff8c00] uppercase tracking-wider">Paused</span>}
        <div className="ml-auto flex items-center gap-2">
          <button onClick={prev} title="Previous scenario (←)" className="p-1.5 rounded border border-[#2a2a2a] text-[#888] hover:text-[#e0e0e0] hover:border-[#3a3a3a] transition-colors"><ChevronLeft size={14} /></button>
          <button onClick={() => setPaused((p) => !p)} title="Pause/resume (Space)" className="p-1.5 rounded border border-[#2a2a2a] text-[#888] hover:text-[#e0e0e0] hover:border-[#3a3a3a] transition-colors">
            {paused ? <Play size={14} /> : <Pause size={14} />}
          </button>
          <button onClick={next} title="Next scenario (→)" className="p-1.5 rounded border border-[#2a2a2a] text-[#888] hover:text-[#e0e0e0] hover:border-[#3a3a3a] transition-colors"><ChevronRight size={14} /></button>
          <button onClick={handleFullscreen} title="Fullscreen (F)" className="p-1.5 rounded border border-[#2a2a2a] text-[#888] hover:text-[#e0e0e0] hover:border-[#3a3a3a] transition-colors">
            {fullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
          </button>
          <button onClick={() => setDemoMode(false)} title="Exit demo mode (Esc)" className="p-1.5 rounded border border-[#ff373744] text-[#ff3737] hover:bg-[#ff373718] transition-colors"><X size={14} /></button>
        </div>
      </div>

      {/* Body */}
      {scenarios.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-[#333] text-[12px]">Loading demo scenarios…</div>
      ) : (
        <div className="flex-1 min-h-0 grid grid-cols-[2fr_1fr] gap-3 p-4">
          <div className="rounded border border-[#1a1a1a] overflow-hidden min-h-0">
            {current && (
              <DebateStream
                key={current.id}
                scenarios={scenarios}
                asset={current.asset}
                dataRange={null}
                onDecision={setDecision}
                forceReplayId={current.debate_index ?? null}
              />
            )}
          </div>
          <div className="flex flex-col gap-3 min-h-0">
            <div className="rounded border border-[#1a1a1a] overflow-hidden" style={{ flex: "1 1 0" }}>
              <RegimePanel regime={regime} />
            </div>
            <div className="rounded border border-[#1a1a1a] overflow-hidden" style={{ flex: "1 1 0" }}>
              <DecisionPanel decision={decision} />
            </div>
          </div>
        </div>
      )}

      {/* Watermark */}
      <div className="absolute bottom-4 right-5 text-[11px] text-[#333] select-none pointer-events-none tracking-widest uppercase">
        MARTA Demo
      </div>
    </div>
  );
}
