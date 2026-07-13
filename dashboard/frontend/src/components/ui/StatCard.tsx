"use client";

interface StatCardProps {
  label: string;
  value: string | number;
  change?: string;
  color?: string;
}

export function StatCard({ label, value, change, color }: StatCardProps) {
  return (
    <div className="p-3 rounded border border-[#222] bg-[#0d0d0d]">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>{value}</div>
      {change && <div className="text-[10px] mt-1 text-[#555]">{change}</div>}
    </div>
  );
}
