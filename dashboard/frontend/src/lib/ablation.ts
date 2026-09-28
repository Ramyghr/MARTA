// Shared between the Backtest page's full ablation table and Analytics' compact
// summary — one source of truth for variant labels/colors so the two pages can't
// drift (e.g. relabeling a variant on one page without the other).
export const VARIANT_LABELS: Record<string, string> = {
  buy_hold: "Buy & Hold",
  rsi_strategy: "RSI Strategy",
  ma_crossover: "MA Crossover",
  ppo_no_regime: "PPO No Regime",
  agents_no_regime: "Agents No Regime",
  ppo_agents_no_regime: "PPO+Agents No Regime",
  marta_full: "MARTA Full",
};

export const VARIANT_COLORS: Record<string, string> = {
  buy_hold: "#4a9eff",
  rsi_strategy: "#00c8d7",
  ma_crossover: "#e070c0",
  ppo_no_regime: "#ffd700",
  agents_no_regime: "#b07fff",
  ppo_agents_no_regime: "#ff8c00",
  marta_full: "#00ff7f",
};

export const TEST_STRATEGY_ORDER: Array<{ key: "marta_ppo" | "buy_hold" | "rsi_strategy" | "ma_crossover" | "momentum"; label: string }> = [
  { key: "marta_ppo", label: "MARTA PPO" },
  { key: "buy_hold", label: "Buy & Hold" },
  { key: "rsi_strategy", label: "RSI Strategy" },
  { key: "ma_crossover", label: "MA Crossover" },
  { key: "momentum", label: "Momentum" },
];
