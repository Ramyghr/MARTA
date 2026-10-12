// Raw (human-scale) indicator values for one candle — see features.py's 13-indicator
// layout. `null` for a given key means that candle didn't carry it (e.g. a live ccxt/
// yfinance extension bar, or a synthetic live-bridge tick — indicators aren't computed
// on the fly for those, only ever loaded from the precomputed *_features.parquet).
export type IndicatorValues = Partial<Record<IndicatorKey, number | null>>;

export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  regime_label: number;
  regime_description?: string;
  is_live?: boolean;
  indicators?: IndicatorValues | null;
}

export type IndicatorKey =
  | "atr" | "adx" | "rsi" | "bb_width" | "macd_histogram" | "obv_raw" | "vol_zscore"
  | "realized_vol" | "vol_ratio" | "return_skew" | "return_kurtosis" | "autocorr_lag1"
  | "realized_vol_raw";

export interface LatestIndicators {
  asset: string;
  /** Timestamp of the last row that actually has computed indicators — NOT "now". See
   * historical.get_latest_indicators: recent candles are frequently still inside an
   * OHLCV-only live-extension window with every indicator null. */
  timestamp: string;
  age_hours: number;
  indicators: IndicatorValues;
}

export interface IndicatorDef {
  key: IndicatorKey;
  label: string;
  description: string;
  /** Fixed decimal places for display; RSI/ADX etc. read better with fewer. */
  decimals: number;
}

// Mirrors CLAUDE.md's 13 z-scored-at-normalize indicators (features.py output, pre-
// normalize.py) — the raw, human-scale values, not the 720h rolling z-scores the model
// actually trains on.
export const INDICATOR_DEFS: IndicatorDef[] = [
  { key: "rsi",              label: "RSI",              description: "Relative Strength Index (14)", decimals: 1 },
  { key: "adx",               label: "ADX",              description: "Average Directional Index (trend strength)", decimals: 1 },
  { key: "atr",               label: "ATR",              description: "Average True Range (volatility, price units)", decimals: 2 },
  { key: "bb_width",          label: "BB Width",         description: "Bollinger Band width", decimals: 4 },
  { key: "macd_histogram",    label: "MACD Histogram",   description: "MACD − signal line", decimals: 3 },
  { key: "obv_raw",           label: "OBV",              description: "On-Balance Volume (cumulative)", decimals: 0 },
  { key: "vol_zscore",        label: "Volume Z-Score",   description: "Volume, 720h rolling z-score", decimals: 2 },
  { key: "realized_vol",      label: "Realized Vol",     description: "Annualized realized volatility", decimals: 4 },
  { key: "realized_vol_raw",  label: "Realized Vol (raw)", description: "Unnormalized realized vol — the March-2020 crash signal", decimals: 4 },
  { key: "vol_ratio",         label: "Volume Ratio",     description: "Volume vs. its rolling average", decimals: 2 },
  { key: "return_skew",       label: "Return Skew",      description: "Rolling skewness of returns", decimals: 3 },
  { key: "return_kurtosis",   label: "Return Kurtosis",  description: "Rolling kurtosis of returns", decimals: 3 },
  { key: "autocorr_lag1",     label: "Autocorrelation",  description: "Lag-1 return autocorrelation", decimals: 3 },
];

export interface AssetInfo {
  id: string;
  label: string;
  type: "crypto" | "forex" | "commodity";
  live: boolean;
}

export interface DataRange {
  asset: string;
  start: string;
  end: string;
  n_rows: number;
}

export interface RegimeState {
  regime_label: number;
  regime_description: string;
  probabilities: Record<string, number>;
  dominant_regime: number;
}

export interface AgentThesis {
  agent: string;
  bias: "bullish" | "bearish" | "neutral";
  confidence: number;
  reasoning: string;
  key_factors: string[];
  regime_alignment: string;
}

export interface AgentRebuttal {
  agent: string;
  target: string;
  rebuttal_point: string;
  maintains_bias: boolean;
  updated_confidence: number;
}

export interface FundManagerDecision {
  action: "long" | "short" | "hold";
  position_size: number;
  conviction: number;
  rationale: string;
  regime_override: boolean;
  dissenting_agents: string[];
}

export interface DebateResult {
  asset: string;
  timestamp: string;
  regime_label: number;
  theses: AgentThesis[];
  rebuttals: AgentRebuttal[];
  risk_limits?: RiskLimits;
  decision: FundManagerDecision;
}

export interface RiskLimits {
  max_position_size: number;
  volatility_forecast: number;
  cvar_95: number;
  risk_regime: string;
}

export interface DebateMessage {
  type: string;
  [key: string]: unknown;
}

// Which run produced these numbers — checkpoint/obs_mode/eval window, straight from
// ppo_evaluation.json's own `_provenance`/`val_window` fields (rl/evaluate.py already
// writes these; nothing new to compute). Surfaced on Portfolio/Analytics so a number
// is never shown without a way to answer "which config produced this".
export interface EvalProvenance {
  generated_at_utc: string;
  checkpoint: { path: string; mtime_utc: string };
  env_config: { obs_mode: string; benchmark_relative: boolean };
}

export interface Portfolio {
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  total_return: number;
  n_trades: number;
  mean_turnover: number;
  n_val_steps: number;
  final_portfolio_value: number;
  buy_and_hold: { sharpe: number; max_drawdown: number; total_return: number };
  per_regime: Array<{
    regime_label: number;
    n_steps: number;
    sharpe: number;
    avg_position_size: number;
    avg_return: number;
  }>;
  val_window?: { start: string; end: string };
  provenance?: EvalProvenance | null;
}

export interface Scenario {
  id: string;
  label: string;
  asset: string;
  timestamp: string;
  regime: string;
  regime_label?: number;
  description: string;
  debate_index?: number;
  group?: string;
}

export interface AblationVariant {
  variant: string;
  key: string;
  sharpe: number;
  max_drawdown: number;
  total_return: number;
  calmar: number;
  n_meaningful_trades: number;
}

export interface BootstrapCI {
  comparison: string;
  delta_sharpe: number;
  ci_lower_95: number;
  ci_upper_95: number;
  significant: boolean;
}

// Out-of-sample (2025 test set, never used during training/validation) — GET /api/test-set.
export interface TestSetMetrics {
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  total_return: number;
  method?: string;
}

export interface TestSetResults {
  test_period: { start: string; end: string };
  n_steps: number;
  warning: string;
  caveats?: string[];
  key_finding?: string | null;
  marta_ppo: TestSetMetrics;
  buy_hold: TestSetMetrics;
  rsi_strategy: TestSetMetrics;
  ma_crossover: TestSetMetrics;
  momentum: TestSetMetrics;
}

export interface CurriculumPhase {
  phase: number;
  assets: string;
  steps: number;
  sharpe: number;
  threshold: number;
  met: boolean;
}

export interface RLResults {
  evaluation: Record<string, unknown>;
  ablation_chart: AblationVariant[];
  bootstrap_ci: BootstrapCI[];
  curriculum: CurriculumPhase[];
  has_equity_curve: boolean;
}

export interface EquityCurve {
  timestamps: string[];
  marta: number[];
  buy_hold: number[];
  synthetic: boolean;
  note?: string;
  marta_final_return: number;
  bh_final_return: number;
  marta_max_dd: number;
  bh_max_dd: number;
}

export interface NewsItem {
  timestamp: string;
  asset: string;
  headline: string;
  summary?: string;
  source: string;
  url?: string;
  hours_before: number;
  sentiment?: "bullish" | "bearish" | "neutral";
  live?: boolean;
}

export interface StockTwitsMessage {
  source: "stocktwits";
  headline: string;
  url: string;
  published_at: string;
  sentiment: "Bullish" | "Bearish" | "Neutral";
  asset: string;
}

export interface FearGreedIndex {
  value: number;
  classification: string;
  timestamp?: string;
  label: "extreme_greed" | "greed" | "neutral" | "fear" | "extreme_fear";
}

export interface LiveSignal {
  asset: string;
  regime_label: number;
  regime_description: string;
  action: "long" | "short" | "hold";
  position_size: number;
  price: number;
  paper_portfolio_value: number;
  paper_total_return: number;
  paper_n_trades: number;
  paper_trades: TradeEntry[];
  source: string;
  note?: string;
  /** True when approval mode is on and this non-hold signal was parked in the
   * pending queue instead of executed — see pending_trade_id for the row to act on. */
  pending_approval?: boolean;
  pending_trade_id?: number | null;
}

export interface PendingTrade {
  id: number;
  asset: string;
  action: "long" | "short" | "hold";
  position_size: number;
  price: number;
  regime_label: number;
  source: string;
  status: "pending" | "approved" | "rejected";
  created_at: string;
  resolved_at: string | null;
  trade_id: number | null;
}

/** One shared capital pool across every live asset — see paper_db.py module
 * docstring. `exposure` is that asset's open position as a fraction of the ONE
 * pool, not an independent per-asset balance. */
export interface AccountState {
  shared_capital: number;
  initial_capital: number;
  total_return: number;
  total_exposure: number;
  positions: Array<{
    asset: string;
    action: "long" | "short" | "hold";
    exposure: number;
    n_trades: number;
    updated_at: string;
  }>;
}

export const REGIME_COLORS: Record<number, string> = {
  0: "#666666",
  1: "#00ff7f",
  2: "#ff3737",
  3: "#ff8c00",
  4: "#ffd700",
  5: "#4a9eff",
  6: "#00d4ff",
};

export interface TradeEntry {
  timestamp: string;
  action: "long" | "short" | "hold";
  position_size: number;
  price: number;
  pnl: number;
  portfolio_value: number;
  regime_label: number;
}

export interface AgentThesisSummary {
  agent?: string;
  bias?: string;
  confidence?: number;
  reasoning?: string;
}

export interface TradeReasoning {
  trade_id: number;
  asset: string;
  timestamp: string;
  regime_label: number;
  regime_description?: string;
  debate_id?: string | null;
  agent_theses: AgentThesisSummary[];
  fund_manager_rationale?: string | null;
  ppo_override: boolean;
  source: "ppo" | "regime_policy" | "debate_override" | string;
}

export interface TradeWithReasoning {
  id: number;
  asset: string;
  timestamp: string;
  action: "long" | "short" | "hold";
  position_size: number;
  price: number;
  slippage: number;
  pnl: number;
  portfolio_value: number;
  regime_label: number;
  debate_id?: string | null;
  agent_theses: AgentThesisSummary[];
  fund_manager_rationale?: string | null;
  ppo_override: boolean;
  source?: string;
  entry_price?: number | null;
  entry_timestamp?: string | null;
  exit_price?: number | null;
  exit_timestamp?: string | null;
  realized_pnl?: number | null;
  holding_hours?: number | null;
  regime_at_exit?: number | null;
  outcome?: "profit" | "loss" | "open" | null;
}

export interface AlpacaOrder {
  id: string;
  symbol: string;
  side: string;
  qty: string;
  notional: string;
  status: string;
  filled_at: string;
  filled_avg_price: string;
}

export interface Notification {
  id: string;
  type: "regime_shift" | "trade_signal" | "approval_required" | "system";
  title: string;
  message: string;
  severity: "info" | "warning" | "critical";
  timestamp: string;
  read: boolean;
}

export interface TradeOutcomeStats {
  asset: string;
  n_closed: number;
  n_open: number;
  win_rate: number | null;
  avg_holding_hours: number | null;
  avg_pnl: number | null;
}

export interface PositionState {
  current_action: "long" | "short" | "hold";
  current_position: number;
  current_regime: number;
  portfolio_value: number;
  total_return: number;
  n_trades: number;
  trades: TradeEntry[];
  source: "backtest" | "live" | "synthetic";
  note?: string;
}

// Bootstrap default only — AppShell overwrites this from the live
// /api/regime/semantics call on mount. Worded as volatility/trend, not bull/bear (see
// dashboard/backend/regime_service.py's _load_crypto5_regime_descriptions docstring):
// a 2026-09-01 validation found sharpe-derived bull/bear labels didn't reliably match
// actual price direction per occurrence, so this bootstrap copy shouldn't claim it does
// even for the brief window before the live fetch resolves.
export const REGIME_NAMES: Record<number, string> = {
  0: "Mixed/Transitional",
  1: "Low Volatility / Trending",
  2: "Extreme Volatility / Shock",
  3: "Elevated Volatility / Trending",
  4: "Low Volatility / Range-bound",
  5: "Trending Forex A",
};
