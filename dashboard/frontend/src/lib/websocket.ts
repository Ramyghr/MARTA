const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export { WS_URL, API_URL };

export type MessageHandler = (msg: Record<string, unknown>) => void;
export type WsStatus = "idle" | "connecting" | "open" | "reconnecting" | "closed";
export type StatusHandler = (status: WsStatus) => void;

const BASE_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30_000;

export class MartaWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Set<MessageHandler> = new Set();
  private statusHandlers: Set<StatusHandler> = new Set();
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;
  private _status: WsStatus = "idle";
  private _lastMessageAt: number | null = null;

  constructor(path: string) {
    this.url = `${WS_URL}${path}`;
  }

  get status(): WsStatus {
    return this._status;
  }

  get lastMessageAt(): number | null {
    return this._lastMessageAt;
  }

  private setStatus(status: WsStatus): void {
    this._status = status;
    this.statusHandlers.forEach((h) => h(status));
  }

  connect(): void {
    if (typeof window === "undefined") return;
    this.closed = false;
    this.setStatus(this.reconnectAttempt > 0 ? "reconnecting" : "connecting");
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => {
      this.reconnectAttempt = 0;
      this.setStatus("open");
    };
    this.ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        this._lastMessageAt = Date.now();
        this.handlers.forEach((h) => h(msg));
      } catch {}
    };
    this.ws.onclose = () => {
      if (!this.closed) {
        this.reconnectAttempt += 1;
        const delay = Math.min(BASE_RECONNECT_DELAY_MS * 2 ** (this.reconnectAttempt - 1), MAX_RECONNECT_DELAY_MS);
        this.setStatus("reconnecting");
        this.reconnectTimer = setTimeout(() => this.connect(), delay);
      } else {
        this.setStatus("closed");
      }
    };
    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  send(data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  on(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    handler(this._status);
    return () => this.statusHandlers.delete(handler);
  }

  close(): void {
    this.closed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.setStatus("closed");
  }
}

export async function fetchApi<T>(path: string, init?: RequestInit): Promise<T> {
  // credentials: "include" — the session cookie set by /api/auth/login must ride
  // along on every request; fetch's default ("same-origin") drops it since the
  // frontend/backend run on different ports (cross-origin, even both on localhost).
  const res = await fetch(`${API_URL}${path}`, { credentials: "include", ...init });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}
