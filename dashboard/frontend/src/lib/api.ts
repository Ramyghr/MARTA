import { fetchApi, API_URL } from "./websocket";
import type {
  AssetInfo, RegimeState, Candle, Portfolio, Scenario, PositionState,
  TradeReasoning, TradeWithReasoning, TradeOutcomeStats, Notification,
  LiveSignal, TradeEntry, PendingTrade, AccountState,
  StockTwitsMessage, FearGreedIndex, AlpacaOrder,
} from "./types";

export interface AblationResults {
  variants: Record<string, {
    sharpe: number;
    sortino: number;
    max_drawdown: number;
    calmar: number;
    total_return: number;
    n_trades: number;
    n_meaningful_trades: number;
    mean_turnover: number;
    method: string;
  }>;
  // Raw passthrough of ablation_results.json — a dict keyed "marta_full_vs_<variant>",
  // NOT the pre-flattened array /api/rl/results produces under the same field name.
  bootstrap_ci: Record<string, { delta_sharpe: number; ci_lower_95: number; ci_upper_95: number }>;
}

export interface RegimeStreamRow {
  timestamp: string;
  asset: string;
  regime_label: number;
  [key: `regime_prob_${number}`]: number;
}

export interface RegimeSemanticsEntry {
  realized_vol_mean: number;
  realized_vol_std: number;
  close_return_mean: number;
  close_return_std: number;
  rsi_mean: number;
  adx_mean: number;
  annualized_return: number;
  annualized_vol: number;
  sharpe: number;
  time_pct: number;
  dominant_asset: string;
  n_samples: number;
  /** Generated server-side from this regime's actual vol/sharpe stats — the live
   * replacement for the static REGIME_NAMES fallback in lib/types.ts. */
  description?: string;
}

export interface RegimeSemantics {
  n_states: number;
  semantic_validation_passed: boolean;
  vol_std_across_regimes: number;
  regimes: Record<string, RegimeSemanticsEntry>;
}

export interface DebateListItem {
  id: number;
  timestamp: string;
  asset: string;
  regime_label: number;
  scenario: string;
  action: "long" | "short" | "hold";
  conviction: number;
}

export interface RawAgentThesis {
  agent_name: string;
  directional_bias: "bullish" | "bearish" | "neutral";
  confidence: number;
  reasoning: string;
  key_factors: string[];
  regime_alignment: string;
}

export interface RawAgentRebuttal {
  agent_name: string;
  target_agent: string;
  rebuttal_point: string;
  maintains_bias: boolean;
  updated_confidence: number;
}

export interface RawRiskLimits {
  max_position_size: number;
  volatility_forecast: number;
  cvar_95: number;
  risk_regime: string;
}

export interface RawFundManagerDecision {
  final_action: "long" | "short" | "hold";
  position_size: number;
  conviction: number;
  rationale: string;
  regime_override: boolean;
  dissenting_agents: string[];
}

export interface DebateDetail {
  timestamp: string;
  asset: string;
  regime_label: number;
  round1_theses: RawAgentThesis[];
  round2_rebuttals: RawAgentRebuttal[];
  risk_limits: RawRiskLimits;
  fund_manager_decision: RawFundManagerDecision;
  total_tokens_used: number;
  debate_duration_seconds: number;
  scenario_description: string;
}

export interface GpuStatus {
  available: boolean;
  reason?: string;
  name?: string;
  memory_used_mb?: number;
  memory_total_mb?: number;
  utilization_pct?: number;
}
export interface OllamaStatus {
  connected: boolean;
  host: string;
  models?: string[];
  error?: string;
}
export interface RedisStatus {
  configured: boolean;
  connected: boolean;
  reason?: string;
  error?: string;
}
export interface SystemStatus {
  gpu: GpuStatus;
  ollama: OllamaStatus;
  redis: RedisStatus;
}
export interface EncoderInfo {
  checkpoint_exists: boolean;
  checkpoint_mtime: string | null;
  epoch: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
}
export interface HmmInfo {
  n_states: number | null;
  semantic_validation_passed: boolean | null;
}
export interface RlInfo {
  checkpoint_exists: boolean;
  checkpoint_mtime: string | null;
  best_sharpe: number | null;
}
export interface SystemModels {
  encoder: EncoderInfo;
  hmm: HmmInfo;
  rl: RlInfo;
}

export const api = {
  getAssets: () => fetchApi<{ assets: AssetInfo[] }>("/api/assets"),
  getRegime: (asset: string) => fetchApi<RegimeState>(`/api/regime/${asset}`),
  getRegimeAt: (asset: string, timestamp: string) =>
    fetchApi<RegimeState>(`/api/regime/${asset}/at?timestamp=${encodeURIComponent(timestamp)}`),
  getHistory: (asset: string, n = 200) => fetchApi<{ candles: Candle[] }>(`/api/history/${asset}?n=${n}`),
  getHistoryRange: (asset: string, start: string, end: string) =>
    fetchApi<{ candles: Candle[] }>(`/api/history/${asset}/range?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`),
  getDataRange: (asset: string) => fetchApi<{ asset: string; start: string; end: string; n_rows: number }>(`/api/data-range/${asset}`),
  getPortfolio: () => fetchApi<Portfolio>("/api/portfolio"),
  getPositions: () => fetchApi<PositionState>("/api/rl/positions"),
  getDebates: () => fetchApi<{ debates: DebateListItem[] }>("/api/debates"),
  getDebate: (id: string | number) => fetchApi<DebateDetail>(`/api/debates/${id}`),
  getDemoScenarios: () => fetchApi<{ scenarios: Scenario[] }>("/api/demo-scenarios"),
  getAblation: () => fetchApi<AblationResults>("/api/ablation"),
  getRegimeStream: (asset: string, limit = 500) =>
    fetchApi<RegimeStreamRow[]>(`/api/regime-stream/${asset}?limit=${limit}`),
  getRegimeSemantics: () => fetchApi<RegimeSemantics>("/api/regime/semantics"),
  getSystemStatus: () => fetchApi<SystemStatus>("/api/system/status"),
  getSystemModels: () => fetchApi<SystemModels>("/api/system/models"),
  testSource: (source: string) => fetchApi<{ ok: boolean; message: string }>(`/api/system/test-source?source=${encodeURIComponent(source)}`),
  ping: () => fetchApi<{ ok: boolean }>("/api/system/ping"),
  getTradeReasoning: (tradeId: number) => fetchApi<TradeReasoning>(`/api/trades/reasoning/${tradeId}`),
  getTradeHistory: (asset: string, limit = 50) =>
    fetchApi<{ asset: string; trades: TradeWithReasoning[] }>(`/api/trades/history/${asset}?limit=${limit}`),
  getTradeOutcomes: (asset: string) => fetchApi<TradeOutcomeStats>(`/api/trades/outcomes/${asset}`),
  getNotifications: (opts?: { unreadOnly?: boolean; type?: string }) => {
    const params = new URLSearchParams();
    if (opts?.unreadOnly) params.set("unread_only", "true");
    if (opts?.type) params.set("type", opts.type);
    const qs = params.toString();
    return fetchApi<{ notifications: Notification[]; unread_count: number }>(`/api/notifications${qs ? `?${qs}` : ""}`);
  },
  getLiveSignal: (asset: string, debateAction?: string) =>
    fetchApi<LiveSignal>(`/api/rl/live-signal?asset=${asset}${debateAction ? `&debate_action=${debateAction}` : ""}`),
  triggerSignal: (asset: string) =>
    fetch(`${API_URL}/api/rl/trigger-signal?asset=${asset}`, { method: "POST" }).then((r) => r.json()),
  getPaperTrades: (asset: string, limit = 50) =>
    fetchApi<{ asset: string; portfolio: Record<string, unknown>; trades_local: TradeEntry[]; trades_testnet: unknown[]; source: string }>(
      `/api/paper/trades?asset=${asset}&limit=${limit}`,
    ),
  getSchedulerStatus: () =>
    fetchApi<{
      scheduler_running: boolean;
      testnet_connected: boolean;
      testnet_balance: number | null;
      alpaca_configured: boolean;
      last_signals: Record<string, LiveSignal & { scheduler_fired_at: string }>;
    }>("/api/rl/scheduler-status"),
  getAlpacaAccount: () =>
    fetchApi<{ configured: boolean; portfolio_value?: number; cash?: number; buying_power?: number; status?: string; message?: string; error?: string }>(
      "/api/alpaca/account",
    ),
  getAlpacaOrders: (limit = 20) =>
    fetchApi<{ orders: AlpacaOrder[] }>(`/api/alpaca/orders?limit=${limit}`),
  getAccountState: () => fetchApi<AccountState>("/api/account"),
  getApprovalMode: () => fetchApi<{ approval_mode: boolean }>("/api/settings/approval-mode"),
  setApprovalMode: (enabled: boolean) =>
    fetch(`${API_URL}/api/settings/approval-mode?enabled=${enabled}`, { method: "POST" }).then((r) => r.json()) as Promise<{ approval_mode: boolean }>,
  getPendingTrades: (status: "pending" | "approved" | "rejected" = "pending") =>
    fetchApi<{ trades: PendingTrade[] }>(`/api/trades/pending?status=${status}`),
  approvePendingTrade: (id: number) =>
    fetch(`${API_URL}/api/trades/pending/${id}/approve`, { method: "POST" }).then((r) => r.json()),
  rejectPendingTrade: (id: number) =>
    fetch(`${API_URL}/api/trades/pending/${id}/reject`, { method: "POST" }).then((r) => r.json()),
  getStockTwits: (asset: string) =>
    fetchApi<{ asset: string; messages: StockTwitsMessage[]; count: number }>(`/api/news/stocktwits/${asset}`),
  getFearGreed: () => fetchApi<FearGreedIndex>("/api/market/fear-greed"),
};
