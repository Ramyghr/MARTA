"use client";
import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex items-center justify-between gap-4 px-1 pb-3 shrink-0">
      <div>
        <h1 className="text-[15px] font-bold tracking-wide text-[#e0e0e0]">{title}</h1>
        {description && <p className="text-[11px] text-[#555] mt-0.5">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}
