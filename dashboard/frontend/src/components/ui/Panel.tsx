"use client";
import type { ReactNode } from "react";

interface PanelProps {
  title?: string;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
  noPadding?: boolean;
}

export function Panel({ title, children, className = "", actions, noPadding }: PanelProps) {
  return (
    <div className={`panel flex flex-col min-h-0 ${className}`}>
      {title && (
        <div className="panel-header flex items-center justify-between shrink-0">
          <span>{title}</span>
          {actions && <div className="flex items-center gap-2 normal-case tracking-normal">{actions}</div>}
        </div>
      )}
      <div className={`flex-1 min-h-0 ${noPadding ? "" : "p-3"}`}>{children}</div>
    </div>
  );
}
