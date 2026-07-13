"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Bell, AlertTriangle, AlertOctagon } from "lucide-react";
import { API_URL, MartaWebSocket, fetchApi } from "@/lib/websocket";
import type { Notification } from "@/lib/types";

const CACHE_KEY = "marta_notifications_cache";
const MAX_KEPT = 50;
const MAX_SHOWN = 10;

function loadCache(): Notification[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    return raw ? (JSON.parse(raw) as Notification[]) : [];
  } catch {
    return [];
  }
}

function saveCache(notifications: Notification[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(notifications.slice(0, MAX_KEPT)));
  } catch {
    // localStorage unavailable (private mode, quota) — unread count just won't survive refresh
  }
}

function timeAgo(ts: string): string {
  const diffMs = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

const SEVERITY_STYLE: Record<Notification["severity"], { border: string; color: string; Icon: typeof Bell }> = {
  info:     { border: "#4a9eff", color: "#4a9eff", Icon: Bell },
  warning:  { border: "#ff8c00", color: "#ff8c00", Icon: AlertTriangle },
  critical: { border: "#ff3737", color: "#ff3737", Icon: AlertOctagon },
};

export function NotificationBell() {
  const [notifications, setNotifications] = useState<Notification[]>(() => loadCache());
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchApi<{ notifications: Notification[]; unread_count: number }>("/api/notifications")
      .then((res) => {
        setNotifications(res.notifications);
        saveCache(res.notifications);
      })
      .catch(() => {});

    const ws = new MartaWebSocket("/ws/notifications");
    const unsub = ws.on((msg) => {
      if (msg.type === "notification" && msg.data) {
        setNotifications((prev) => {
          const next = [msg.data as Notification, ...prev].slice(0, MAX_KEPT);
          saveCache(next);
          return next;
        });
      }
    });
    ws.connect();
    return () => {
      unsub();
      ws.close();
    };
  }, []);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markRead = useCallback((id: string) => {
    setNotifications((prev) => {
      const next = prev.map((n) => (n.id === id ? { ...n, read: true } : n));
      saveCache(next);
      return next;
    });
    fetch(`${API_URL}/api/notifications/${id}/read`, { method: "POST" }).catch(() => {});
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications((prev) => {
      const next = prev.map((n) => ({ ...n, read: true }));
      saveCache(next);
      return next;
    });
    fetch(`${API_URL}/api/notifications/read-all`, { method: "POST" }).catch(() => {});
  }, []);

  const recent = notifications.slice(0, MAX_SHOWN);

  return (
    <div className="relative" ref={rootRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        title="Notifications"
        className="relative p-1.5 rounded text-[#555] hover:text-[#ccc] hover:bg-white/5 transition-colors"
      >
        <Bell size={15} />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-[3px] rounded-full bg-[#ff3737] text-[9px] font-bold text-white flex items-center justify-center leading-none">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-9 w-80 max-h-[420px] overflow-y-auto bg-[#0d0d0d] border border-[#222] rounded shadow-xl z-[100] animate-fade-in">
          <div className="sticky top-0 bg-[#0d0d0d] border-b border-[#1a1a1a] px-3 py-2 flex items-center justify-between">
            <span className="text-[11px] font-semibold text-[#ccc]">Notifications</span>
            <button
              onClick={markAllRead}
              disabled={unreadCount === 0}
              className="text-[10px] text-[#4a9eff] hover:text-[#7ab8ff] disabled:text-[#333] disabled:cursor-not-allowed transition-colors"
            >
              Mark all read
            </button>
          </div>

          {recent.length === 0 ? (
            <div className="px-3 py-6 text-center text-[10px] text-[#444]">No notifications yet</div>
          ) : (
            <div className="divide-y divide-[#141414]">
              {recent.map((n) => {
                const style = SEVERITY_STYLE[n.severity] ?? SEVERITY_STYLE.info;
                const Icon = style.Icon;
                const isCriticalUnread = n.severity === "critical" && !n.read;
                return (
                  <button
                    key={n.id}
                    onClick={() => markRead(n.id)}
                    className={`w-full text-left px-3 py-2.5 flex gap-2.5 hover:bg-white/[0.03] transition-colors ${isCriticalUnread ? "notif-critical-pulse" : ""}`}
                    style={{
                      borderLeft: `2px solid ${n.read ? "transparent" : style.border}`,
                      background: n.read ? "transparent" : `${style.color}0d`,
                    }}
                  >
                    <Icon size={14} className="shrink-0 mt-0.5" style={{ color: style.color }} />
                    <div className="flex-1 min-w-0">
                      <div className="text-[11px] font-medium text-[#ccc] truncate">{n.title}</div>
                      <div className="text-[10px] text-[#666] leading-relaxed mt-0.5 line-clamp-2">{n.message}</div>
                      <div className="text-[9px] text-[#3a3a3a] mt-1">{timeAgo(n.timestamp)}</div>
                    </div>
                    {!n.read && <span className="w-1.5 h-1.5 rounded-full shrink-0 mt-1" style={{ background: style.color }} />}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
