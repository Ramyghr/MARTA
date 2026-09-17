"use client";
import { TickerTape } from "@/components/TickerTape";

// Decorative only — synthetic, static values (same rationale as MartaBot's candle
// stream: these pages are pre-auth, no backend/websocket dependency allowed here).
const TICKER_ITEMS = [
  { symbol: "BTC_USDT", change: 2.34 },
  { symbol: "ETH_USDT", change: -1.12 },
  { symbol: "BNB_USDT", change: 0.87 },
  { symbol: "LTC_USDT", change: -0.45 },
  { symbol: "XRP_USDT", change: 3.21 },
  { symbol: "EURUSD", change: 0.12 },
  { symbol: "GOLD", change: -0.28 },
];

export function AuthBackground() {
  return (
    <div className="auth-bg" aria-hidden="true">
      <div className="auth-bg-grid" />
      <div className="auth-bg-glow auth-bg-glow-1" />
      <div className="auth-bg-glow auth-bg-glow-2" />
      <div className="auth-ticker">
        <TickerTape items={TICKER_ITEMS} />
      </div>
    </div>
  );
}
