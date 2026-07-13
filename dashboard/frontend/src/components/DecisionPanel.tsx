"use client";
import type { FundManagerDecision } from "@/lib/types";

interface Props {
  decision: FundManagerDecision | null;
  timestamp?: string | null;
}

function formatAge(hoursAgo: number): string {
  if (hoursAgo < 1) return "just now";
  if (hoursAgo < 24) return `${Math.round(hoursAgo)} hour${Math.round(hoursAgo) === 1 ? "" : "s"} ago`;
  const days = Math.round(hoursAgo / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.round(days / 30);
  return `${months} month${months === 1 ? "" : "s"} ago`;
}

// 270° arc gauge — track from 7 o'clock to 5 o'clock via 12 o'clock
function ConvictionGauge({ pct }: { pct: number }) {
  const r = 30;
  const circ = 2 * Math.PI * r;
  const arcLen = circ * 0.75;    // 270° of the full circle
  const gapLen = circ - arcLen;  // 90° gap sits at the bottom
  const filled = (pct / 100) * arcLen;
  const color = pct >= 70 ? "#00ff7f" : pct >= 40 ? "#ffd700" : "#ff3737";
  const glowColor = pct >= 70 ? "rgba(0,255,127,0.4)" : pct >= 40 ? "rgba(255,215,0,0.4)" : "rgba(255,55,55,0.4)";

  return (
    <svg width="76" height="76" viewBox="0 0 76 76">
      {/* Track */}
      <circle
        cx="38" cy="38" r={r}
        fill="none" stroke="#1e1e1e" strokeWidth="6"
        strokeDasharray={`${arcLen} ${gapLen}`}
        strokeLinecap="round"
        transform="rotate(135 38 38)"
      />
      {/* Fill */}
      <circle
        cx="38" cy="38" r={r}
        fill="none" stroke={color} strokeWidth="6"
        strokeDasharray={`${filled} ${circ - filled}`}
        strokeLinecap="round"
        transform="rotate(135 38 38)"
        style={{
          transition: "stroke-dasharray 0.5s ease",
          filter: `drop-shadow(0 0 4px ${glowColor})`,
        }}
      />
      <text x="38" y="42" textAnchor="middle" fill={color} fontSize="12" fontWeight="bold" fontFamily="monospace">
        {pct}%
      </text>
    </svg>
  );
}

export default function DecisionPanel({ decision, timestamp }: Props) {
  if (!decision) {
    return (
      <div className="panel h-full flex flex-col">
        <div className="panel-header">Final Decision</div>
        <div className="flex-1 flex items-center justify-center text-[#333] text-[11px]">
          Run a debate to see the decision
        </div>
      </div>
    );
  }

  const actionColor = decision.action === "long" ? "#00ff7f" : decision.action === "short" ? "#ff3737" : "#666";
  const actionGlow = decision.action === "long" ? "0 0 16px rgba(0,255,127,0.3)" : decision.action === "short" ? "0 0 16px rgba(255,55,55,0.3)" : "none";
  const convPct = Math.round(decision.conviction * 100);
  const hoursAgo = timestamp ? (Date.now() - new Date(timestamp).getTime()) / 3_600_000 : null;
  const isStale = hoursAgo != null && hoursAgo > 24;

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">Final Decision</div>
      <div className="flex-1 p-3 flex flex-col gap-3 overflow-auto">

        {/* Action badge — large, centered */}
        <div
          className="flex items-center justify-center font-bold uppercase tracking-widest rounded animate-fade-in"
          style={{
            minHeight: 64,
            fontSize: 32,
            background: `${actionColor}15`,
            color: actionColor,
            border: `1px solid ${actionColor}55`,
            boxShadow: actionGlow,
            letterSpacing: "0.15em",
          }}
        >
          {decision.action.toUpperCase()}
        </div>

        {/* Decision timestamp + age — always shown so a decision is never displayed
            without dating it; orange stale warning kicks in past 24h. */}
        {timestamp && (
          <div className="text-[11px] leading-relaxed">
            <div className="text-[#666]">
              Decided:{" "}
              {new Date(timestamp).toLocaleString(undefined, {
                month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
              })}
              {hoursAgo != null && <span className="text-[#444]"> ({formatAge(hoursAgo)})</span>}
            </div>
            {isStale && (
              <div className="mt-1 font-semibold" style={{ color: "#ff8c00" }}>
                ⚠ Stale decision — run new debate for current market conditions
              </div>
            )}
          </div>
        )}

        {/* Gauge + position */}
        <div className="flex items-center gap-3 justify-center">
          <ConvictionGauge pct={convPct} />
          <div className="flex flex-col gap-1.5 text-[11px]">
            <div>
              <div className="stat-label">Conviction</div>
              <div className="stat-value mt-0.5" style={{ color: convPct >= 70 ? "#00ff7f" : convPct >= 40 ? "#ffd700" : "#ff3737" }}>
                {convPct}%
              </div>
            </div>
            <div>
              <div className="stat-label">Position</div>
              <div className="stat-value mt-0.5" style={{ color: actionColor }}>
                {(decision.position_size * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        {/* Regime override */}
        {decision.regime_override && (
          <div
            className="text-[11px] px-2.5 py-2 rounded flex items-center gap-2 animate-fade-in"
            style={{ background: "rgba(255,140,0,0.1)", border: "1px solid #ff8c0044", color: "#ff8c00" }}
          >
            <span>⚠</span>
            <span className="font-semibold tracking-wide">REGIME OVERRIDE APPLIED</span>
          </div>
        )}

        {/* Rationale */}
        <div>
          <div className="stat-label mb-1">Rationale</div>
          <p className="text-[11px] text-[#777] leading-relaxed">{decision.rationale}</p>
        </div>

        {/* Dissenting agents */}
        {decision.dissenting_agents?.length > 0 ? (
          <div>
            <div className="stat-label mb-1">Dissenting</div>
            <div className="flex gap-1.5 flex-wrap">
              {decision.dissenting_agents.map((a) => (
                <span
                  key={a}
                  className="text-[10px] px-2 py-0.5 rounded"
                  style={{
                    background: a === "ContrarianAgent" ? "rgba(255,140,0,0.12)" : "#1a1a1a",
                    color: a === "ContrarianAgent" ? "#ff8c00" : "#888",
                    border: a === "ContrarianAgent" ? "1px solid #ff8c0033" : "1px solid #2a2a2a",
                  }}
                >
                  {a}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-[10px]" style={{ color: "#00ff7f44" }}>✓ All agents in consensus</div>
        )}
      </div>
    </div>
  );
}
