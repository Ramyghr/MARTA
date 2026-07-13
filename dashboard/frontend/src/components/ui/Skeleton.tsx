"use client";

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[#1a1a1a] ${className}`} />;
}

const ROW_WIDTHS = ["w-full", "w-11/12", "w-5/6", "w-3/4", "w-2/3", "w-1/2"];

export function SkeletonRows({ rows = 4, className = "" }: { rows?: number; className?: string }) {
  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className={`h-4 ${ROW_WIDTHS[i % ROW_WIDTHS.length]}`} />
      ))}
    </div>
  );
}

export function SkeletonCard({ className = "" }: { className?: string }) {
  return (
    <div className={`rounded border border-[#1a1a1a] bg-[#0d0d0d] p-3 flex flex-col gap-2 ${className}`}>
      <Skeleton className="h-3 w-1/3" />
      <Skeleton className="h-6 w-2/3" />
    </div>
  );
}
