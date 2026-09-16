# Digest: Architecture patterns for multi-consumer regime signals (round 1)

## Findings

- **Claim:** A "Market Regime Council" pattern decouples regime detection from agent weighting: a dedicated regime-shift component ("Regime Shift Perceptor") computes a continuous market-state score independently of LLM specialist agents, then modulates each agent's Shapley-value-based decision weight via regime-conditional multipliers and exponentially-weighted performance histories, recorded through a five-layer causal trace per rebalancing decision.
  **Source:** "Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems" (arXiv:2605.24490) — **Publisher:** arXiv — **Published:** 2026-05 — **Accessed:** 2026-09-14 — **Confidence:** medium (abstract-level only) — **Class:** pattern

- **Claim:** This pattern was evaluated at moderate scale (1,037 trading days, 13 crypto assets, N=3 specialist agents, 5 seeds); authors flag it "remains vulnerable to cold-start dominance under regime shifts" — during a fresh regime transition the weighting mechanism can be dominated by stale/uninformative early-period weights before stabilizing.
  **Source:** same — **Published:** 2026-05 — **Accessed:** 2026-09-14 — **Confidence:** medium — **Class:** performance-number / failure-claim (single source, below two-source bar)

- **Claim:** A common pattern threading regime/state signals to an RL execution layer — frozen LLM extracts fixed-dim feature vectors from unstructured text, appended to RL observation, feeding PPO — produced features with real standalone predictive validity (IC > 0.15 held-out) but DEGRADED actual RL policy performance specifically during macroeconomic-shock regime transitions (features "add noise," augmented agent underperforms price-only baseline; recovered once markets calmed). Headline: feature-level statistical validity does not guarantee policy-level robustness across regime boundaries.
  **Source:** "When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies" (arXiv:2604.10996) — **Publisher:** arXiv — **Published:** 2026-04 — **Accessed:** 2026-09-14 — **Confidence:** medium — **Class:** failure-claim (single source, below two-source bar)

- **Claim:** A "Regime-Gated Residual Mixture-of-Experts" (RG-ResMoE) pattern restricts the regime/state signal to influencing ONLY the gating/routing weights of a MoE layer, never feeding it directly into the underlying forecasting networks' inputs. Tested at real scale: 1,027 US equities (2018-2025, ~1.9M out-of-sample forecasts) + 1,552-stock Japanese (TSE Prime) replication, 30 non-overlapping rolling walk-forward windows, 30 seeds/configuration.
  **Source:** "Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting" (arXiv:2608.12251) — **Publisher:** arXiv — **Published:** 2026-08 — **Accessed:** 2026-09-14 — **Confidence:** high (full page fetched, specific numbers) — **Class:** pattern / performance-number

- **Claim:** In the same paper: directly concatenating nonstationary regime variables into model feature inputs (rather than restricting to a gating function) caused severe training instability — a variant with regime features appended to inputs collapsed in 30/30 runs vs 3/30 for the plain baseline, IC dropped −0.0043. A standard (non-regime-restricted) MoE gating design also collapsed in 24/30 runs vs 0/30 for the regime-gated design with load-balancing penalty + zero-initialized residual experts.
  **Source:** arXiv:2608.12251 — **Published:** 2026-08 — **Accessed:** 2026-09-14 — **Confidence:** high (strong internal replication, 30 seeds × 2 markets, but only ONE independent publication — below the cross-publication two-source bar) — **Class:** failure-claim

- **Claim:** Hard/discrete regime routing (top-1 gating, volatility-quantile bucketing, sector assignment — the "hard regime switch" pattern common in classical regime-switching models) underperformed soft/continuous gating by 0.0021-0.0034 IC in this same study, and occasionally also collapsed — discrete regime-bucket routing is less stable than continuous regime-weighted blending at this scale.
  **Source:** arXiv:2608.12251 — **Published:** 2026-08 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** performance-number

- **Claim:** TradingAgents is a named, widely-cited multi-agent LLM trading framework structuring agents into role-specialized teams (fundamental/sentiment/technical analysts → Bull/Bear researcher debate → trader → risk-management team), reporting improved cumulative return, Sharpe, and max drawdown over baselines.
  **Source:** "TradingAgents: Multi-Agents LLM Financial Trading Framework" (arXiv:2412.20138) — **Publisher:** arXiv — **Published:** 2024-12 — **Accessed:** 2026-09-14 — **Confidence:** low (only abstract retrieved — no detail on how state/regime info is structurally shared between agent roles) — **Class:** pattern

- **Claim:** The general MLOps "feature store" pattern (single precomputed, versioned store — offline for training, online for low-latency serving — shared source of truth for multiple downstream consumers) is presented in vendor/practitioner material as standard for avoiding feature duplication and training/serving skew.
  **Source:** aggregated vendor/practitioner blogs (Aerospike "Feature Store 101," JFrog/Qwak, Towards Data Science) — **Published:** 2024-2025 — **Accessed:** 2026-09-14 — **Confidence:** low (snippet-level, vendor-authored/marketing register, general-ML not trading-specific) — **Class:** pattern

- **Claim:** Recent survey work frames LLM+RL financial systems into a three-paradigm taxonomy by LLM functional role — feature-based (LLM output becomes numeric feature), auxiliary-based (LLM does macro/regime interpretation as strategic overlay), policy-based (LLM more directly drives decisions) — based on 34 studies 2023-2026.
  **Source:** "A survey on LLM-enhanced reinforcement learning in financial markets," Discover Artificial Intelligence — **Publisher:** Springer Nature — **Published:** 2026 (month unconfirmed) — **Accessed:** 2026-09-14 — **Confidence:** low (not fetched directly, search-synthesis only) — **Class:** pattern

## Leads worth chasing
- "When Valid Signals Fail" (2604.10996) and RG-ResMoE (2608.12251) point at the same mechanism from two angles — signals valid in isolation destabilize/degrade a downstream policy when injected directly rather than mediated (gating, or a separate risk/regime layer) — but are not the same paper and don't cite each other. Search for a third independent replication to clear the two-source bar for "regime signal must be isolated to gating, not concatenated to raw inputs."
- TradingAgents' full text/repo (github.com/TauricResearch/TradingAgents) not read — would reveal actual shared-state/memory design between agent roles.
- Market Regime Council's cold-start vulnerability only captured at abstract depth — fetch full paper for concrete failure-mode section.
- **No source discussed cadence mismatch between fast (hourly/intraday) and slow (weekly/macro) decision layers sharing one regime signal** — an explicit brief sub-question, essentially unaddressed. Follow-up: "asynchronous decision frequency multi-timescale trading agent," "stale signal cache trading system postmortem."
- No real production deployment (trading-firm engineering post-mortem, not academic backtest) surfaced for any pattern — significant gap given the brief's interest in "real deployments at scale."
- Three-paradigm taxonomy survey not directly verified — fetch Discover Artificial Intelligence/Springer directly.

## What was looked for and could not be found
- No real (non-academic) production deployment case study of a regime/state-signal architecture threading RL + rule-based + LLM layers simultaneously.
- No source directly addressing "stale cached regime signal" as a documented production failure mode (closest: Market Regime Council's "cold-start dominance," adjacent not identical).
- No source directly addressing mismatched update cadence between fast/hourly and slow/weekly decision layers sharing one regime signal.
- No claim met the two-independent-source bar for "a named pattern failed/doesn't scale" — every failure-mode finding traces to exactly one paper, flagged accordingly.
- TradingAgents' actual architecture/state-object design not obtained (abstract only).
- Survey's three-paradigm taxonomy not verified against primary text.
- Ran into the 10-tool-call budget before searching specifically for engineered/raw-feature vs. learned-embedding comparisons for regime detection (the brief's opening question) — largely unaddressed here (see dimensions 1a/1b/3a/3b digests for partial coverage).

## Assistant self-report
Budget used: 10 tool calls, ~duration 177s.
