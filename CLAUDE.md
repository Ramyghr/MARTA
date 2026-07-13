
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**MARTA** — Multi-Agent Regime-Aware Trading Architecture. A Final Year Engineering Project (PFE) combining contrastive pretraining, LLM multi-agent deliberation, and PPO reinforcement learning for cross-asset algorithmic trading.

**Research claim:** Conditioning a multi-agent LLM trading system on latent regime embeddings (via TFT + InfoNCE pretraining) produces statistically significant improvements in risk-adjusted returns over regime-agnostic baselines across crypto, forex, and commodity markets.

**Python 3.11 — conda environment: `marta`**

## Running the pipeline

```bash
conda run -n marta python run_pipeline.py          # full data pipeline (Month 1)
conda run -n marta python run_regime.py            # HMM regime clustering (Month 1 Week 3)
conda run -n marta python run_debate.py            # multi-agent debate demo (Month 2 Week 1)
conda run -n marta mlflow ui                       # view experiment results (http://127.0.0.1:5000)
```

All pipeline stages write parquet to `data/` and log metrics/artifacts to `mlruns/`.

## System architecture — three layers

```
Layer 1 — Regime Encoder         encoder/  regime/          ✓ COMPLETE
  TFT (pytorch-forecasting) + InfoNCE contrastive loss
  → 128-dim embedding → Gaussian HMM (n=7, BIC-selected) → regime probability vector

Layer 2 — Multi-Agent Deliberation    agents/               ✓ Week 1+2 complete
  LangGraph orchestrates agents, each injected with the regime vector:
    MacroAgent      — cross-asset correlation + contagion detection
    TechnicalAgent  — TFT attention map + momentum signals
    RiskAgent       — GARCH volatility forecast + CVaR position limits
    FundManagerAgent — synthesizes consensus → final action + rationale
    SentimentAgent  — FinBERT on news headlines              ✓ Week 2 complete
    ContrarianAgent — anti-groupthink rebuttal bias          ✓ Week 2 complete
  2-round debate → Fund Manager synthesizes weighted consensus → action proposal

Layer 3 — RL Execution Agent          (Month 3 — not yet implemented)
  PPO (Stable-Baselines3) with Beta distribution position head
  State: [128-dim regime | 16-dim consensus | 8-dim portfolio]
  Reward: ΔPnL / CVaR_t − λ × drawdown_penalty
  Curriculum: BTC → ETH → Forex → Commodities
  Execution: Alpaca paper API / CCXT sandbox
```

## Data pipeline — REFACTORED (Layer 1 v2)

Run order: `fetch → features → normalize → align → validate` (`python run_pipeline.py`)

Data sources (all genuine OHLCV, no synthetic upsampling):
- **BTC/USDT, ETH/USDT** — CCXT/Binance hourly (2018-01-01 → 2024-01-01). The old ETH
  fetch loop broke on `len(batch) < 1000` and silently capped ETH at 2021-10-12; the
  loop now breaks **only** on an empty batch or on reaching the configured end.
- **EURUSD, XAUUSD, WTIUSD** — histdata.com genuine 1-minute → resampled to hourly
  (`open=first, high=max, low=min, close=last, volume=sum`). Replaces the old yfinance
  daily→hourly forward-fill (which made 23 of every 24 windows near-duplicates and
  contaminated the contrastive embedding).

| Module | Input | Output | Key detail |
|--------|-------|--------|------------|
| `fetch.py` | config | raw parquet per asset | CCXT batch loop (loop-bug fixed) + `fetch_histdata()` (year ZIPs → 1-min parquet cache → hourly). Cache-aware: reuses raw parquet that already reaches the end date. OHLCV clamping via `_sanitize_ohlcv` |
| `features.py` | raw df | 18-col df (5 OHLCV + 13 indicators) | `ta` for ATR/ADX/RSI/BB/MACD; **`obv_raw`** = raw OBV cumsum (no z-score here); **`realized_vol_raw`** = unnormalized annualized vol (14th-slot crash signal); standard `dropna()` (no daily-upsample fillna special-case) |
| `normalize.py` | 18-col df | 14-col df (12 norm indicators + `realized_vol_raw` + `close_return`) | 720h rolling z-score, min_periods=168; drops raw OHLCV; **`realized_vol_raw` excluded from z-score** (`NO_NORMALIZE`); no more `is_gap_fill` |
| `align.py` | dict of 14-col dfs | dict of 14-col dfs + combined parquet | Inner join on BTC backbone index; saves `all_assets_aligned.parquet` with MultiIndex columns `(asset, feature)` |
| `validate.py` | aligned + raw | validation_report.json + MLflow metrics | structural (CRITICAL), range, gaps/distribution (WARNING), OHLCV sanity, cross-asset overlap, **`realized_vol_raw` range + BTC March-2020 crash-ratio** |

**14 final features per asset:** `atr, adx, rsi, bb_width, macd_histogram, obv_raw, vol_zscore, realized_vol, vol_ratio, return_skew, return_kurtosis, autocorr_lag1, realized_vol_raw, close_return`
(13 z-scored + `realized_vol_raw` passthrough; `realized_vol_raw` and `close_return` are the two non-z-scored columns. `model.input_dim` in `configs/assets.yaml` is 14.)

**Train/val/test split (real, not claimed):** Train 2018-01-01 → 2022-12-31 · Val 2023-01-01 → 2023-12-31 · Test 2024-01-01 onward (SACRED — untouched until final evaluation).

**histdata notes:** Downloads are slow (~2s/year/instrument); a full cold fetch of all 3
instruments × 6 years is ~45–60 min. Year ZIPs and per-year 1-minute parquets are cached
in `data/raw/histdata/` — **do not delete**. FX/commodity histdata has **zero volume**
(no centralised volume), so `obv_raw`/`vol_zscore` are degenerate (~constant) for EURUSD/
XAUUSD/WTIUSD — price-based features carry their signal.

**Alignment:** With ETH now reaching 2024, the bottleneck is the forex weekend structure
(no Sat/Sun bars), so the inner join is capped by the histdata instruments rather than ETH.
Actual aligned dataset: **33,387 rows × 14 features, 2018-01-12 → 2023-12-01** (the end is
capped by WTIUSD, whose 2023 histdata ends 2023-12-01). This is 63.6% of BTC's 24/7 range
— up from the old 32,768 rows, but the forex-weekend ceiling means the spec's ">40,000"
target is not reachable while inner-joining against weekday-only forex.

## Configuration

All pipeline configuration lives in `configs/assets.yaml` — asset symbols (with `source:`
+ `histdata_pair:`), date ranges, indicator windows, normalization parameters, and
validation thresholds.

Key thresholds:
- `min_timestamp_overlap_pct: 0.60` — BTC (24/7) vs. forex (weekday-only) intersection floor. Forex has no weekend bars, so this overlap is capped at ~71% (5/7 days) and lands at ~63.6% after holidays; 0.80 is unreachable for a crypto-vs-forex dataset (the refactor spec's 0.80 assumed ETH, not forex weekends, was the bottleneck — it was always forex)
- `max_gap_hours_crypto: 2`, `max_gap_hours_traditional: 72` — gap detection thresholds

## Key implementation details (pipeline)

- **histdata fetch:** `fetch_histdata(pair, start_year, end_year)` downloads GENERIC_ASCII
  M1 year ZIPs (reusing any already on disk), parses the headerless `;`-separated CSV
  (`YYYYMMDD HHMMSS`), caches per-year 1-minute parquets, and resamples to hourly.
- **`realized_vol_raw` (crash signal):** unnormalized annualized realized vol, the one
  feature `normalize.py` does NOT z-score. The rolling 720h z-score on `realized_vol`
  compresses absolute magnitude, which is why March 2020 was previously undetectable;
  `realized_vol_raw` preserves it (validate logs `btc_march2020_vol_ratio`).
- **OBV single-normalization:** `features.py` emits raw `obv_raw` cumsum; `normalize.py`
  applies the single 720h z-score (the old 168h+720h double z-score is removed).
- **OHLCV clamping:** `_sanitize_ohlcv()` ensures `high = max(open,high,close)`,
  `low = min(open,low,close)`, forward-fills zero close prices. Applied after every fetch.
- **Symbol naming:** `BTC/USDT` → `BTC_USDT` throughout; histdata instruments are
  `EURUSD`/`XAUUSD`/`WTIUSD` (Gold = XAUUSD, Oil = WTIUSD).
- All paths use `pathlib.Path`, all parquet files use `engine="pyarrow"`.

## Regime clustering (`regime/`) — complete

Run: `python run_regime.py`

Outputs:
- `regime/checkpoints/hmm_model.pkl` — fitted GaussianHMM (n_states=7, BIC-selected)
- `regime/checkpoints/pca_model.pkl` — PCA(16) fitted on training embeddings (before 2021-07-01)
- `regime/checkpoints/bic_results.json` — BIC/AIC/LL per n_states for reproducibility
- `data/processed/regime_stream.parquet` — full regime probability stream (163,540 rows)
- `data/validated/regime_semantics.json` — per-regime financial statistics + validation result
- `regime/plots/` — BIC curve, regime timeline per asset, regime t-SNE

Key output schema (`regime_stream.parquet`):
```
timestamp, asset, regime_label, regime_prob_0 … regime_prob_6
```

Pipeline details:
- PCA(16) fit on training rows only (timestamps < 2021-07-01) to avoid leakage; asserts >75% variance retained (actual: ~85.6%)
- BIC formula: `-2 * LL + (n_states² + 2 * n_states * 16) * log(n_train)`
- BIC grid search: diag covariance, 100 iters, tol=1e-3, 20K-row subsample of training data (fast)
- n_states grid: 2–8 inclusive; flat-curve guard picks simpler model if ΔBIC < 5% of best BIC
- Final HMM refit: full covariance, 200 iters, tol=1e-4, all 163,540 rows
- Semantic validation passes if `std(realized_vol μ across regimes) > 0.001`

**Regime label availability:** Not all regimes appear for all assets. BTC_USDT has regimes {0,1,4,5,6}. Regimes 2 (stress/crisis) and 3 (bear trend) appear only in GOLD/EURUSD/OIL. Always filter by asset before selecting a regime label.

## Month 2 — Multi-Agent Deliberation (`agents/`) — Week 1 complete

### Running

```bash
conda run -n marta python run_debate.py
```

Runs 3 demo debates across temporally diverse regime states and logs all results to MLflow under experiment `marta_agent_deliberation`. Outputs saved to `data/debates/debate_results.json`.

### Agent architecture

All agents inherit from `agents/base.py:BaseAgent`. LLM: **Mistral 7B Instruct via Ollama** (`mistral:7b-instruct`), GPU-accelerated (~32–70s per debate vs 300s+ on CPU). `num_predict=400`, `temperature=0.1`, `timeout=300s` with one retry.

| File | Class | Role |
|------|-------|------|
| `agents/base.py` | `BaseAgent` | LLM call, JSON parsing (5-stage recovery), timeout guard |
| `agents/macro_agent.py` | `MacroAgent` | Cross-asset correlation matrix + contagion flag over 168-candle window |
| `agents/technical_agent.py` | `TechnicalAgent` | TFT encoder attention weights + momentum/vol indicators over 60-candle window |
| `agents/risk_agent.py` | `RiskAgent` | GARCH(1,1) vol forecast → CVaR-based max position size + risk regime label |
| `agents/fund_manager.py` | `FundManagerAgent` | Weighted consensus from R1 theses + risk limits → final action/size/conviction |
| `agents/graph.py` | `DebateGraph` | LangGraph state machine: R1 theses → R2 rebuttals → fund manager |
| `agents/context.py` | `RegimeContext` | Dataclass holding regime label, prob vector, market features, risk limits |
| `agents/schemas.py` | Pydantic models | `AgentThesis`, `AgentRebuttal`, `RiskLimits`, `FundManagerDecision`, `DebateResult` |

### Schemas (`agents/schemas.py`)

```python
AgentThesis:   agent_name, round=1, directional_bias (bullish/bearish/neutral),
               confidence [0,1], reasoning (≤500 chars), key_factors (≤4),
               regime_alignment (aligned/contrary/neutral/mixed)

AgentRebuttal: agent_name, round=2, target_agent, rebuttal_point (≤400 chars),
               maintains_bias, updated_confidence [0,1]
```

Round 2 only triggers when two agents hold opposing `directional_bias` values. If all agents agree, rebuttals are skipped and "No rebuttals posted." is shown.

### JSON parse recovery (5 stages in `base.py`)

1. Direct `model_validate_json(response.strip())`
2. Extract from markdown code fence ` ```json ... ``` `
3. Regex: first JSON object anywhere in response
4. Greedy: `response[first { : last }]` + trailing-comma repair
5. Last-resort: collapse literal newlines inside extracted block + trailing-comma repair

On total failure: logs raw response (first 500 chars) and returns a safe neutral default. This means a parse failure is always visible in logs and never silently corrupts downstream results.

### Demo timestamp selection (`run_debate.py`)

Three scenarios: BTC calm bull (R1), BTC calm bear (R4, targeted to Oct 2018–Jan 2019 — genuine late-2018 bear), GOLD stress/crisis (R2). Timestamps are enforced to be ≥30 days apart. If the preferred date window yields no valid rows, falls back to the full pool and searches for any timestamp ≥30 days from existing picks.

**Timestamps are UTC-aware** (`datetime64[ns, UTC]`) — always use `pd.Timestamp(date_str, tz="UTC")` when filtering the regime stream by date.

### Key implementation details (agents)

- `RegimeContext` is built from `regime_stream.parquet` + per-asset aligned parquet at the selected timestamp. MacroAgent uses a 168-candle lookback; TechnicalAgent uses 60 candles.
- RiskAgent fits GARCH(1,1) on the lookback window's `close_return` column to produce `volatility_forecast`. Position size scales inversely with vol relative to a 2% daily vol target.
- FundManagerAgent receives the full `DebateResult` (R1 theses + R2 rebuttals + risk limits) and calls the LLM once to produce the final `FundManagerDecision`.
- MLflow experiment: `marta_agent_deliberation`. Each `run_debate.py` run creates one MLflow run logging per-debate metrics (conviction, position size, tokens, duration, vol forecast) plus aggregates.

## Month 2 Week 2 — Multi-Agent Deliberation (`agents/`) — complete

### New agents

| File | Class | Role |
|------|-------|------|
| `agents/sentiment_agent.py` | `SentimentAgent` | FinBERT sentiment on static news cache; exponential recency decay; 6h momentum signal |
| `agents/contrarian_agent.py` | `ContrarianAgent` | Devil's advocate; activates at avg_confidence ≥ 0.65; historical analogue retrieval from regime stream |

**SentimentAgent:**
- FinBERT loaded from `./models/finbert` on CPU (`device=-1`)
- News cache at `data/news_cache.json` — 60+ headlines spanning 2018–2021 for BTC, ETH, GOLD, macro
- Looks up headlines within a 24h window before the debate timestamp; includes `macro` headlines for any asset
- Batch FinBERT inference → weighted aggregation (exponential decay, half-life = 6h)
- Sentiment momentum: avg positive score in last 6h vs full 24h window
- LLM call interprets scores; FinBERT provides the numbers, Mistral interprets them

**ContrarianAgent:**
- Activation threshold: avg R1 confidence ≥ 0.65 → Contrarian fires
- Below threshold: posts `neutral` thesis with note "consensus too low to challenge"
- Historical analogue retrieval: filters `regime_stream.parquet` to same `regime_label` + `regime_prob > 0.80`; computes 24h cumulative `close_return` forward from `aligned_df` for up to 100 cases; reports loss rate and avg forward return
- LLM builds the strongest counter-argument; `directional_bias` is *always forced* to the opposite of consensus regardless of LLM output
- `regime_alignment` is always `"contrary"`

### Round 2 rebuttal logic (fixed)

Round 2 now reliably fires when:
1. Contrarian is active (directional_bias ≠ "neutral") — creates guaranteed opposition
2. R1 agents hold directly opposing views (bullish vs bearish among themselves)
3. Groupthink: avg_confidence > 0.75 with no internal R1 opposition

When active, the Contrarian also posts its own rebuttal targeting the highest-confidence opposing agent.

### Full 5-agent debate flow

```
Round 1:  MacroAgent → TechnicalAgent → SentimentAgent (sequential)  +  RiskAgent (GARCH, no LLM)
          ↓
Contrarian: sees all R1 theses; activates if avg_confidence ≥ 0.65; argues opposite direction
          ↓
Round 2:  Each R1 agent rebuts if Contrarian is active or opposing views exist
          ContrarianAgent also posts rebuttal when active
          ↓
FundManager: weighted vote (Contrarian counts at 0.5x); regime_override if Contrarian challenged
             AND any agent flipped position; dissenting_agents includes Contrarian if contrary
          ↓
PPO stub (Month 3)
```

### FundManager changes (Week 2)

- `ContrarianAgent` thesis weighted at 0.5x in the consensus vote
- `regime_override = True` when: stress/crisis regime + bullish consensus (existing), OR Contrarian was active AND any agent flipped after rebuttal
- `dissenting_agents` list includes Contrarian when it argues the opposite of consensus

### Key data files (Week 2)

- `data/news_cache.json` — static news cache, 60+ headlines 2018–2021
- `agents/sentiment_agent.py`, `agents/contrarian_agent.py` — new agents
- `agents/graph.py` — DebateState extended with `contrarian_thesis`, `consensus_confidence`, `rebuttal_triggered`
- `agents/fund_manager.py` — 5-agent synthesis with Contrarian half-weight

## Per-Asset HMM Experiment — complete (2026-05-31)

Run: `conda run -n marta python run_per_asset_regime.py`

**Motivation:** The shared GaussianHMM (n=7, trained on all 5 assets together) correctly identifies GOLD's stress/crisis regime but misclassifies BTC's March 2020 crash as R1/R4. The hypothesis was that fitting independent HMMs per-asset, with each asset's own PCA(16) fitted on its own training rows, would give BTC a crash-specific regime uncorrupted by GOLD's embedding space.

**Method:** Independent PCA(16) + BIC-selected GaussianHMM per asset (n_states grid 2–7, diag cov BIC search, full cov final refit). Outputs saved to `_per_asset` suffixed files; shared HMM untouched.

**Results:**
- n_states selected: 6 for all 5 assets (BIC flat-curve guard triggered, n=7 not sufficiently better than n=6)
- All 5 assets: semantic validation PASSED (vol_std: BTC=0.572, ETH=0.598, EURUSD=0.425, GOLD=0.469, OIL=0.405)
- Per-asset HMMs are significantly more semantically distinct than shared (BTC vol_std 0.572 vs 0.242)

**BTC March 2020 verdict: NOT_DETECTED**
- Both pre-crash (March 1–11) and crash (March 12–13) windows assigned regime R1 by the per-asset BTC HMM
- KL divergence = 0.486 (threshold 0.5 — missed by a small margin)
- Root cause: `realized_vol` in the aligned parquet is a rolling 720h z-score; the crash reads as high raw vol (~6.3) but maps to the same z-score regime bucket. The per-asset embedding space alone is insufficient to create a dedicated crash regime when the normalized features compress the signal.

**Consequence:** `agents/context.py` was NOT modified. The debate system continues to use `regime_stream.parquet` (shared HMM). Per-asset stream lives at `data/processed/regime_stream_per_asset.parquet` for reference.

**Comparison plot:** `regime/plots/regime_comparison_BTC.png` (BTC price + shared HMM bands vs per-asset HMM bands, March 12 2020 marked)

**Key files:**
- `regime/hmm_per_asset.py` — per-asset HMM pipeline (fit + generate stream)
- `regime/validate_per_asset.py` — per-asset semantic validation + `check_btc_march_2020()`
- `regime/compare.py` — `compare_regime_streams()` + `plot_regime_comparison()`
- `run_per_asset_regime.py` — master script (MLflow experiment: `marta_regime_per_asset`)
- `regime/checkpoints/hmm_{asset}.pkl`, `pca_{asset}.pkl` — per-asset model checkpoints

## Month 3 — RL Execution Agent (`rl/`) — complete

Run: `python run_rl.py`           (full curriculum, ~700k steps)
     `python run_rl.py --quick`   (fast end-to-end smoke run)

```
State:  [128-dim regime embedding | 16-dim consensus | 8-dim portfolio] = 152-dim
Action: Discrete direction {0=long, 1=short, 2=hold} × Beta(α,β) position fraction ∈ (0,1)
Reward: r_t = ΔPnL / CVaR_t − λ × drawdown_penalty   (ΔPnL is the per-step return fraction)
Curriculum: BTC → BTC+ETH → +EURUSD → +GOLD+OIL   (Sharpe gates 0.5/0.6/0.7/0.8)
```

### Architecture

- **State (152-dim Box, clipped [-10,10]):**
  - 128-dim: frozen `TFTRegimeEncoder.encode()` on the current 60-candle window (no grad).
  - 16-dim: agent consensus — per-R1-agent signed confidence (+conf bullish / −conf bearish / 0)
    plus FM signed-conviction, position size, regime_override, dissent count. Zeros when no
    debate exists for the timestamp (the common case during training).
  - 8-dim portfolio: 4 per-asset signed-position slots (canonical order BTC,ETH,EURUSD,GOLD;
    OIL shares slot 3 — only one asset trades per episode) + `[pnl_norm, drawdown,
    steps_since_trade/100, vol_forecast]`.
- **Action (hybrid):** exposed to SB3 as a flat 2-dim Box `[direction∈{0,1,2}, position_fraction∈(0.01,0.99)]`
  (SB3 PPO has no Dict action support; `MARTATradingEnv.action_space_dict` documents the semantic
  Dict, `unflatten_action()` recovers it). Direction ~ Categorical, position ~ Beta(α,β) with
  `α,β = softplus(·)+1` (>1 ⇒ unimodal, no boundary clipping bias).
- **CVaR:** GARCH(1,1) volatility forecast → 95% normal CVaR (`-pdf(ppf(.05))/.05 · vol`), the exact
  RiskAgent formula. GARCH is refit weekly and cached to `data/processed/rl_cache/vol_{asset}_*.parquet`
  — never refit per-step (intractable) and the LLM debate system is never called during RL.

### Walk-forward split (strict)

Train: 2018-01-10 → 2020-12-31. Validation: 2021-01-01 → 2021-10-12 (6,803 BTC rows). `reset()`
samples start timestamps only from the active split; evaluation runs `sequential=True` (traverses
the whole val window, ignoring the per-episode `max_steps` cap).

### Curriculum & callbacks

One `model.learn()` call (total_timesteps = Σ phase max_steps); `CurriculumCallback` advances phases
in-place by mutating the env's active asset list (PPO weights persist across phases — that's the
transfer). Each phase saves `rl/checkpoints/ppo_phase_{n}.zip` and logs to MLflow. Early stop if
val Sharpe < −0.5 for 3 consecutive evals. Best-by-val-Sharpe → `ppo_best.zip`, final → `ppo_final.zip`.

### Ablation (central research result)

5 variants on the val set: `buy_hold`, `ppo_no_regime`, `agents_no_regime`, `ppo_agents_no_regime`,
`marta_full`. **Block bootstrap CI** (n=1000) on the Sharpe delta of `marta_full` vs each baseline →
`data/results/ablation_results.json`.

**`ppo_no_regime` / `ppo_agents_no_regime` do not run the trained PPO.** Zeroing the 128-dim regime
slot pushes the frozen policy out-of-distribution and it collapses to hold on every step (Sharpe/
return/n_trades all exactly 0.0) — which reads as "identical to hold" rather than "policy collapses
without regime context." Both variants instead run a momentum rule (`_run_momentum_rule` in
`rl/ablation.py`): long/short only when the last 3 `close_return`s are unanimously positive/negative,
else hold; fixed 0.3 position size (`ppo_agents_no_regime` scales that size by nearest-debate
confidence). This represents "RL-like execution without regime awareness" and confirms regime
embedding dependency by construction, rather than proving it via a degenerate all-zero run.

**`agents_no_regime`** is rule-based on the cached debate consensus, but the debate cache only has
~33 timestamps across ~6,500 validation steps. Exact-timestamp matching (the old behaviour) missed
almost all of them and defaulted to hold ~99.5% of the time. It now uses nearest-debate interpolation
(`_make_nearest_consensus_fn`): each step snaps to the closest cached debate within 7 days (else
holds), so it trades a real block of exposure around every debate instead of a handful of exact hits.
The fund manager's `position_size` field is capped at `min(position_size, 0.05)` before being applied
— its raw fallback default (0.5) otherwise dwarfs the trained PPO's risk-capped sizing (avg position
~0.001–0.006) and inflates Sharpe/return in a way that isn't a fair regime-vs-no-regime comparison.

Each variant's entry in `ablation_results.json` carries a `"method"` string documenting what actually
ran (buy-and-hold / momentum rule / nearest-debate rule / trained PPO). `run_rl.py`'s ablation table
prints a footnote: `* ppo_no_regime/ppo_agents_no_regime use momentum baseline (PPO collapses on
out-of-distribution zero-regime input, confirming regime embedding dependency)`.

**Block bootstrap (`bootstrap_sharpe_ci`):** resamples **non-overlapping 168-hour (1-week) blocks**
instead of individual hourly returns, preserving the autocorrelation structure of the return series
(IID resampling understates the CI width for serially-correlated returns). Returns are trimmed to
complete blocks (`n_blocks = n // 168`); `delta_sharpe` reported is the **bootstrap mean** of the
per-resample Sharpe deltas (not the point estimate), with the 2.5/97.5 percentiles as the 95% CI.
Internal RNG seed is fixed at 42 for reproducibility. Falls back to a zero-width CI at the point delta
when there are <2 complete blocks.

### In-sample regime behavior analysis (`analyze_regime_behavior`, `rl/evaluate.py`)

A **diagnostic-only** probe (NOT a performance claim) of how the trained agent sizes positions across
regimes during known crisis/bear episodes. Runs the agent deterministically over `IN_SAMPLE_WINDOWS`
(March 2020 crash `2020-02-01..2020-04-30`; 2018 crypto bear `2018-11-01..2019-01-31`) — both inside
the **training split**, so every result is labelled IN-SAMPLE and is **never mixed with validation
metrics**. Per window it reports avg position per **decision-time regime** (see the regime-attribution
note under "Regime-differentiated scaling"), regime-transition count, and per-boundary position changes.
Output: `data/results/regime_behavior_analysis.json` (carries an `_disclaimer` field). The window env is
built with `split="all"` and its `_valid_starts` intersected with the window's index range so sequential
traversal covers exactly the window. `run_rl.py` runs this after the ablation and prints a per-window table.

### Key files

- `rl/env.py`        — `MARTATradingEnv` (Gymnasium env, GARCH/CVaR precompute, walk-forward split)
- `rl/policy.py`     — `MARTAActorCriticPolicy` (shared MLP 152→256→128→64; Categorical+Beta heads)
- `rl/callbacks.py`  — `CurriculumCallback` + `MLflowCallback`
- `rl/train.py`      — `train_ppo(config)`, `build_env_factory`, `CURRICULUM_PHASES`, frozen encoder load
- `rl/evaluate.py`   — walk-forward, `compute_metrics`, per-regime breakdown (decision-time regime), baselines, `analyze_regime_behavior` (in-sample)
- `rl/ablation.py`   — 5-variant ablation + `bootstrap_sharpe_ci` (**block bootstrap, 168h blocks**)
- `run_rl.py`        — master script (MLflow experiment `marta_ppo_training`); `--quick` smoke flag

Outputs: `rl/checkpoints/ppo_{phase_n,best,final}.zip`, `data/results/{ppo_evaluation,ablation_results,regime_behavior_analysis,curriculum_history}.json`.

### Results & key findings (2026-06-01 — first full run)

**Headline (BTC validation 2021-01-01 → 2021-10-12, from `ppo_best.zip`):** Sharpe **1.809**, max drawdown **5.8%**, Calmar 3.605, total return 17.0%, 31 meaningful trades, avg position 0.026. Per-regime both positive (R1=1.729, R4=1.893). vs buy-and-hold (Sharpe 0.946, 60% DD, 41% return): MARTA wins on risk-adjusted return and drawdown, gives up absolute return by running low exposure.

**Regime embedding is load-bearing.** Ablating it (zeroing the 128-dim slot) collapses the policy to always-flat → Sharpe **0.000** (ΔSharpe = 1.809 vs `ppo_no_regime`). Caveat: not statistically significant on a single val path — bootstrap 95% CI = [-0.300, 3.937]; and zeroing a trained input is partly off-distribution. Honest framing: strong qualitative dependence, significance not established on one window.

**Curriculum collapse (important).** Phases 3-4 add EURUSD/GOLD/OIL, which are daily bars upsampled to hourly → intraday returns are ~zero. On those assets "hold" (zero position) is the lowest-variance optimum, so PPO drifts the shared policy to **always-flat** (`validation_sharpe` returns exactly 0.000 when the policy holds the whole val pass). The full curriculum therefore does NOT show BTC→forex→commodities transfer — it *erodes* the BTC/ETH policy. The **Phase-2 (BTC+ETH) checkpoint `ppo_best.zip` is the correct headline result**, and `run_rl.py` now evaluates `ppo_best.zip` by default (not the final model).
  - **Future work:** cap the curriculum at Phase 2 (BTC+ETH); or add an idle/no-position penalty to the reward; or retrain forex/commodity assets at native daily frequency with a separate policy head.

**Two fixes from this run:** (1) reward was floored/clipped — `|CVaR|` floored at 0.001, `reward` clipped to [-10,10], `lambda_drawdown` 0.1→0.01 — to stop a Sharpe -33 divergence (raw rewards too large for the value net). (2) `n_trades` (any change >1e-9) is a metric artifact of the continuous Beta head; use `n_meaningful_trades` (>0.5%) and `mean_turnover` instead — actual turnover is 2.43× total / 0.2% txn cost, not "overtrading."

## CRITICAL: idle_penalty is a training signal only

idle_penalty must NEVER be set on val_env. Setting it on val_env inflates
val Sharpe by penalizing abstention during evaluation — making the policy
appear to trade more than it does in production.

Bug confirmed Jul 2026: idle_penalty=0.002 on val_env inflated
ppo_best Sharpe from 0.449 (true) to 1.413 (false).

Rule: val_env always uses idle_penalty=0.0. train_env uses whatever
the phase config specifies. This is enforced in train_ppo() and
CurriculumCallback._advance_phase().

Re-evaluating all checkpoints on clean (idle_penalty=0.0) val_envs found
`ppo_phase_4.zip` — not the previously-saved `ppo_best.zip` — is the true
best checkpoint (clean Sharpe 1.370 vs 0.449); `ppo_best.zip` was replaced
with `ppo_phase_4.zip` accordingly. `scripts/evaluate_all_checkpoints.py`
reproduces this comparison (`data/results/checkpoint_evaluation.json`).

## Stack

`pytorch-forecasting` (TFT), `hmmlearn` (Gaussian HMM), `stable-baselines3` (PPO), `langgraph` (agent orchestration), `langchain-ollama` (Mistral 7B), `fastapi + celery + redis` (live trading backend), `mlflow` (experiment tracking), `ccxt` (crypto data + execution), `yfinance` (traditional asset data), `ta` (technical indicators), `vectorbt` (backtesting), `rich` (CLI logging)
