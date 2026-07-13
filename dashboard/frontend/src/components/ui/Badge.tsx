"use client";

export type BadgeVariant = "long" | "short" | "hold" | "live" | "warning" | "neutral" | "info";

const VARIANTS: Record<BadgeVariant, string> = {
  long: "bg-[#00ff7f]/15 text-[#00ff7f] border border-[#00ff7f]/30",
  short: "bg-[#ff3737]/15 text-[#ff3737] border border-[#ff3737]/30",
  hold: "bg-[#666]/15 text-[#888] border border-[#444]/60",
  live: "bg-[#00ff7f]/15 text-[#00ff7f] border border-[#00ff7f]/30 animate-pulse-dot",
  warning: "bg-[#ff8c00]/15 text-[#ff8c00] border border-[#ff8c00]/30",
  neutral: "bg-[#666]/10 text-[#999] border border-[#333]",
  info: "bg-[#4a9eff]/15 text-[#4a9eff] border border-[#4a9eff]/30",
};

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

export function Badge({ variant = "neutral", children, className = "" }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${VARIANTS[variant]} ${className}`}
    >
      {children}
    </span>
  );
}

const ACTION_VARIANT: Record<string, BadgeVariant> = { long: "long", short: "short", hold: "hold" };

export function ActionBadge({ action }: { action: string }) {
  return <Badge variant={ACTION_VARIANT[action] ?? "neutral"}>{action}</Badge>;
}

const REGIME_VARIANT: Record<string, BadgeVariant> = {
  bullish: "long",
  bearish: "short",
  neutral: "hold",
};

export function BiasBadge({ bias }: { bias: string }) {
  return <Badge variant={REGIME_VARIANT[bias] ?? "neutral"}>{bias}</Badge>;
}
