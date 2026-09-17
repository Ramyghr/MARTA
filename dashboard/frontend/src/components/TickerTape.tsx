"use client";

export interface TickerItem {
  symbol: string;
  change: number;
}

// Pure scrolling marquee — no positioning/chrome of its own, callers wrap it
// (AuthBackground's fixed full-bleed strip, HeaderTicker's in-grid bar, etc).
export function TickerTape({ items }: { items: TickerItem[] }) {
  const doubled = [...items, ...items];
  return (
    <div className="ticker-track-wrap">
      <div className="ticker-track">
        {doubled.map((item, i) => (
          <span className="ticker-item" key={i}>
            <span className="ticker-symbol">{item.symbol}</span>
            <span className={item.change >= 0 ? "ticker-up" : "ticker-down"}>
              {item.change >= 0 ? "+" : ""}
              {item.change.toFixed(2)}%
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
