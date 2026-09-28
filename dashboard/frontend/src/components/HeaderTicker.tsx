"use client";
import { TickerTape } from "@/components/TickerTape";

// Decorative market-breadth strip under the topbar — NOT wired to selectedAsset
// or any backend data. Static/synthetic change values. Most of these (everything
// past GOLD) have no real support in this project (crypto5 pivot — see
// CLAUDE.md/docs/RESULTS_AND_METHODOLOGY.md) and are shown purely for visual
// scale, deliberately kept separate from the functional asset <select> in
// Topbar.tsx so picking one can't break any page.
const MAJOR_TICKER_ITEMS = [
  { symbol: "BTC_USDT", change: 2.34 },
  { symbol: "ETH_USDT", change: -1.12 },
  { symbol: "EURUSD", change: 0.12 },
  { symbol: "GOLD", change: -0.28 },
  { symbol: "AAPL", change: 1.05 },
  { symbol: "MSFT", change: 0.44 },
  { symbol: "GOOGL", change: -0.63 },
  { symbol: "AMZN", change: 1.87 },
  { symbol: "NVDA", change: 3.42 },
  { symbol: "TSLA", change: -2.15 },
  { symbol: "META", change: 0.91 },
  { symbol: "SPX", change: 0.38 },
  { symbol: "NDX", change: 0.52 },
  { symbol: "DJI", change: -0.14 },
  { symbol: "GBPUSD", change: -0.21 },
  { symbol: "USDJPY", change: 0.17 },
  { symbol: "USDCHF", change: -0.08 },
  { symbol: "AUDUSD", change: 0.29 },
  { symbol: "WTI", change: -1.34 },
  { symbol: "SILVER", change: 0.76 },
];

export function HeaderTicker() {
  return (
    <div className="header-ticker">
      <TickerTape items={MAJOR_TICKER_ITEMS} />
    </div>
  );
}
