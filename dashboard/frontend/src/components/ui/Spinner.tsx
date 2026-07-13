"use client";

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return (
    <div className="flex items-center gap-2 text-[#555] text-[11px]">
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        className="animate-spin"
        style={{ animationDuration: "0.8s" }}
      >
        <circle cx="12" cy="12" r="9" stroke="#222" strokeWidth="3" fill="none" />
        <path d="M21 12a9 9 0 0 0-9-9" stroke="#00ff7f" strokeWidth="3" fill="none" strokeLinecap="round" />
      </svg>
      {label && <span>{label}</span>}
    </div>
  );
}
