"use client";
import type { LucideIcon } from "lucide-react";
import { Inbox } from "lucide-react";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon: Icon = Inbox, title, description, action }: EmptyStateProps) {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-2 text-center py-8 px-4">
      <Icon size={28} strokeWidth={1.25} className="text-[#333]" />
      <div className="text-[12px] text-[#555] font-medium">{title}</div>
      {description && <div className="text-[10px] text-[#3a3a3a] max-w-xs">{description}</div>}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}
