"use client";
import { useEffect, useRef, useState } from "react";

let nextCandleId = 0;

type Candle = { id: number; delta: number };

function seedCandles(): Candle[] {
  return [0.4, -0.3, 0.6, 0.2, -0.4, 0.5].map((delta) => ({ id: nextCandleId++, delta }));
}

export function MartaBot({
  size = 96,
  showChart = true,
  className,
}: {
  size?: number;
  showChart?: boolean;
  className?: string;
}) {
  const [blinking, setBlinking] = useState(false);
  const [reaction, setReaction] = useState<"up" | "down" | null>(null);
  const [candles, setCandles] = useState<Candle[]>(seedCandles);
  const isFirstCandleUpdate = useRef(true);

  // Idle blink loop: random 3-6s gap, ~150ms closed.
  useEffect(() => {
    let openTimer: ReturnType<typeof setTimeout>;
    let closeTimer: ReturnType<typeof setTimeout>;
    function scheduleBlink() {
      openTimer = setTimeout(() => {
        setBlinking(true);
        closeTimer = setTimeout(() => {
          setBlinking(false);
          scheduleBlink();
        }, 150);
      }, 3000 + Math.random() * 3000);
    }
    scheduleBlink();
    return () => {
      clearTimeout(openTimer);
      clearTimeout(closeTimer);
    };
  }, []);

  // Synthetic candle tick: random walk, no backend/real-data dependency.
  useEffect(() => {
    const interval = setInterval(() => {
      const magnitude = Math.random() < 0.2 ? 2.2 : 1;
      const delta = (Math.random() - 0.48) * magnitude;
      setCandles((prev) => [...prev.slice(1), { id: nextCandleId++, delta }]);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  // React to whatever the newest candle turned out to be.
  useEffect(() => {
    if (isFirstCandleUpdate.current) {
      isFirstCandleUpdate.current = false;
      return;
    }
    const last = candles[candles.length - 1];
    setReaction(last.delta >= 0 ? "up" : "down");
    const t = setTimeout(() => setReaction(null), 700);
    return () => clearTimeout(t);
  }, [candles]);

  const eyelidScale = blinking ? 1 : reaction === "down" ? 0.55 : 0;
  const haloR = reaction === "up" ? 26 : 20;
  const haloOpacity = reaction === "up" ? 0.3 : 0.15;
  const pupilFill = reaction === "up" ? "#B6FFD9" : "#00FF7F";
  const antennaOpacity = reaction === "down" ? 0.08 : reaction === "up" ? 0.35 : 0.2;
  const faceClass = reaction === "up" ? "marta-bounce" : reaction === "down" ? "marta-shake" : "";

  return (
    <div className={className} style={{ display: "flex", alignItems: "center", gap: size * 0.12 }}>
      <div className="marta-idle-bob" style={{ width: size, height: size }}>
        <svg viewBox="0 0 240 240" width={size} height={size} role="img" aria-label="MARTA">
          <g className={faceClass}>
            <rect x="34" y="106" width="16" height="28" rx="7" fill="#06140D" stroke="#00FF7F" strokeWidth="1.5" />
            <rect x="186" y="106" width="16" height="28" rx="7" fill="#06140D" stroke="#00FF7F" strokeWidth="1.5" />
            <rect x="50" y="60" width="140" height="120" rx="36" fill="#06140D" stroke="#00FF7F" strokeWidth="2" />
            <rect x="68" y="102" width="104" height="36" rx="18" fill="#00FF7F" opacity="0.15" stroke="#00FF7F" strokeWidth="1.5" />
            <circle
              cx="120" cy="120" r={haloR} fill="#00FF7F" opacity={haloOpacity}
              style={{ transition: "r 0.3s ease, opacity 0.3s ease" }}
            />
            <circle cx="120" cy="120" r="14" fill={pupilFill} style={{ transition: "fill 0.3s ease" }} />
            <circle cx="114" cy="114" r="4" fill="#B6FFD9" opacity="0.9" />
            <rect
              x="68" y="102" width="104" height="36" fill="#06140D"
              style={{
                transformBox: "fill-box",
                transformOrigin: "top",
                transform: `scaleY(${eyelidScale})`,
                transition: "transform 0.12s ease",
              }}
            />
            <line x1="102" y1="147" x2="138" y2="147" stroke="#00FF7F" strokeWidth="2" opacity="0.4" strokeLinecap="round" />
            <line x1="102" y1="153" x2="138" y2="153" stroke="#00FF7F" strokeWidth="2" opacity="0.4" strokeLinecap="round" />
            <line x1="102" y1="159" x2="138" y2="159" stroke="#00FF7F" strokeWidth="2" opacity="0.4" strokeLinecap="round" />
            <line x1="120" y1="60" x2="120" y2="36" stroke="#00FF7F" strokeWidth="3" strokeLinecap="round" />
            <circle
              cx="120" cy="36" r="10" fill="#00FF7F" opacity={antennaOpacity}
              style={{ transition: "opacity 0.3s ease" }}
            />
            <circle cx="120" cy="36" r="5" fill="#00FF7F" />
          </g>
        </svg>
      </div>
      {showChart && (
        <svg viewBox="0 0 160 90" width={size * 1.35} height={size * 0.5}>
          <line x1="8" y1="70" x2="152" y2="70" stroke="#00FF7F" strokeWidth="1" opacity="0.15" />
          {candles.map((c, i) => {
            const x = 17 + i * 22;
            const h = Math.min(Math.abs(c.delta) * 14, 34) + 4;
            const up = c.delta >= 0;
            return (
              <g key={c.id} style={{ transform: `translate(${x}px, 0)`, transition: "transform 0.4s ease" }}>
                <rect x={0} width={16} y={up ? 70 - h : 70} height={h} rx="3" fill={up ? "#00FF7F" : "#FF6B6B"} />
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
