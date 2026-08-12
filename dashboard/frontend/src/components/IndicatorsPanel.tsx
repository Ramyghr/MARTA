"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { SlidersHorizontal } from "lucide-react";
import { EmptyState } from "@/components/ui/EmptyState";
import { Spinner } from "@/components/ui/Spinner";
import { useIndicatorSettingsStore } from "@/lib/store";
import { INDICATOR_DEFS } from "@/lib/types";
import { api } from "@/lib/api";
import type { LatestIndicators } from "@/lib/types";

function fmtIndicator(value: number | null | undefined, decimals: number): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function fmtAge(hours: number): string {
  if (hours < 1) return "<1h old";
  if (hours < 48) return `${Math.round(hours)}h old`;
  return `${Math.round(hours / 24)}d old`;
}

const REFRESH_MS = 5 * 60_000; // indicator data moves at most hourly — no need to poll faster

/**
 * Latest real indicator values for `asset`, filtered to whichever indicators the user
 * enabled on Settings → Indicators (useIndicatorSettingsStore).
 *
 * Fetched from a dedicated endpoint rather than derived from the chart's candle window:
 * for assets with a live-extension gap (BTC/ETH ~weeks, EURUSD/GOLD ~months as of this
 * writing — see historical.py), EVERY candle in the normally-displayed 200-300 candle
 * window has null indicators, so scanning that window would show blanks even though real
 * data exists further back. This asks the backend for the actual last-computed row and
 * shows its true age instead of silently pretending it's current.
 */
export function IndicatorsPanel({ asset }: { asset: string }) {
  const visible = useIndicatorSettingsStore((s) => s.visible);
  const shownDefs = INDICATOR_DEFS.filter((d) => visible[d.key]);

  const [data, setData] = useState<LatestIndicators | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    setData(null);
    const load = () => {
      api.getLatestIndicators(asset)
        .then((d) => { if (!cancelled) { setData(d); setLoading(false); } })
        .catch(() => { if (!cancelled) { setError(true); setLoading(false); } });
    };
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [asset]);

  if (shownDefs.length === 0) {
    return (
      <EmptyState
        icon={SlidersHorizontal}
        title="No indicators selected"
        description="Choose which indicators to display in Settings → Indicators."
        action={
          <Link href="/settings" className="text-[10px] text-[#4a9eff] hover:text-[#6db4ff] transition-colors">
            Go to Settings →
          </Link>
        }
      />
    );
  }

  if (loading) return <Spinner label="Loading indicator values…" />;

  if (error || !data) {
    return (
      <EmptyState
        title="No indicator data available"
        description={`No precomputed indicators found for ${asset.replace("_", "/")}.`}
      />
    );
  }

  const isStale = data.age_hours > 48;

  return (
    <div className="flex flex-col gap-2 h-full min-h-0">
      <div className="flex items-center justify-between text-[10px] text-[#444] shrink-0">
        <span>As of {new Date(data.timestamp).toLocaleString()}</span>
        <span style={{ color: isStale ? "#ff8c00" : "#555" }} title="Age of the last row with real computed indicators — not the latest live price tick.">
          {fmtAge(data.age_hours)}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-2 overflow-auto">
        {shownDefs.map((d) => (
          <div key={d.key} className="p-2.5 rounded border border-[#222] bg-[#0d0d0d]" title={d.description}>
            <div className="stat-label truncate">{d.label}</div>
            <div className="stat-value">{fmtIndicator(data.indicators[d.key], d.decimals)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
