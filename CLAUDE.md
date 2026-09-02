
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

Layer 3 — RL Execution Agent          ✓ Month 3 complete
  PPO (Stable-Baselines3) with Beta distribution position head
  State: [128-dim regime | 16-dim consensus | 8-dim portfolio]
  Reward: ΔPnL / CVaR_t − λ × drawdown_penalty
  Curriculum: BTC → BTC+ETH (capped here in practice — see Month 3 section)
  Execution: Alpaca paper API (dashboard/backend/alpaca_service.py) — live, not CCXT sandbox

Layer 4 — Dashboard & Live Trading    dashboard/  pipeline/live/   ✓ complete (undocumented below — see "Dashboard & Live Trading")
  FastAPI backend + Next.js frontend, APScheduler-driven hourly signal tick (NOT Celery/Redis),
  live price/regime/debate websockets, Telegram alerts, Alpaca paper execution, LLM chat.
```

## Data pipeline — REFACTORED (Layer 1 v2)

Run order: `fetch → features → normalize → align → validate` (`python run_pipeline.py`)

Data sources (all genuine OHLCV, no synthetic upsampling). **OIL/WTIUSD was removed from the
pipeline entirely 2026-07-04** (weekend/holiday gaps were the alignment bottleneck) — the
canonical asset set is now `BTC_USDT, ETH_USDT, EURUSD, GOLD` (4 assets, not 5):
- **BTC/USDT, ETH/USDT** — CCXT/Binance hourly. Originally fetched 2018-01-01 → 2024-01-01;
  extended repeatedly since (`pipeline/fetch_2024.py`, `pipeline/fetch_2025.py`) and currently
  reaches **2026-07-12** per `configs/assets.yaml`'s `end: "2026-01-01"`-and-later config
  comment — verify against the actual parquet before quoting an end date, this has moved
  several times. The old ETH fetch loop broke on `len(batch) < 1000` and silently capped ETH
  at 2021-10-12; the loop now breaks **only** on an empty batch or on reaching the configured end.
- **EURUSD, XAUUSD** — histdata.com genuine 1-minute → resampled to hourly
  (`open=first, high=max, low=min, close=last, volume=sum`). Capped at **2025-12-31** by
  histdata.com's whole-year-only ZIPs (no 2026 year ZIP exists yet). Replaces the old
  yfinance daily→hourly forward-fill (which made 23 of every 24 windows near-duplicates and
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

**Train/val/test split (2026-07-13 walk-forward — current):** Train < 2024-01-01 · Val 2024-01-01 → 2024-12-31 · Test 2025-01-01 → 2025-12-31 (SACRED — the Kaggle upload is trimmed to ≤ 2024-12-31 so 2025 never reaches encoder/HMM training or checkpoint selection; see `run_test_set.py`). Every split boundary (`configs/assets.yaml`, `rl/env.py`) was bumped one year in this pass; `configs/encoder.yaml` was found stale at the old 2020-12-31/2021-01-01 boundary during the 2026-07-31 audit and fixed to match.

**histdata notes:** Downloads are slow (~2s/year/instrument); a full cold fetch of both
histdata instruments across all years is ~30–40 min (was "all 3 instruments" before
OIL/WTIUSD's 2026-07-04 removal). Year ZIPs and per-year 1-minute parquets are cached
in `data/raw/histdata/` — **do not delete**. FX/commodity histdata has **zero volume**
(no centralised volume), so `obv_raw`/`vol_zscore` are degenerate (~constant) for EURUSD/
XAUUSD — price-based features carry their signal.

**Alignment (current — post 2026-07 extensions):** OIL/WTIUSD was dropped from the pipeline
entirely (2026-07-04, "4-asset pipeline") — the four canonical assets are now
`BTC_USDT, ETH_USDT, EURUSD, GOLD`, dropping the WTIUSD weekend/holiday bottleneck.
Actual aligned dataset: **45,921 rows × 14 features, 2018-01-12 → 2025-12-31** per asset
(`data/processed/{asset}_aligned.parquet` and `all_assets_aligned.parquet`), extended
through 2025 for the 2025 dashboard/live data bridge and the 2025 test-set window. The
old "33,387 rows, ends 2023-12-01, capped by WTIUSD" figure is stale — verify against the
actual parquet (`pd.read_parquet(...).shape` / `.index[-1]`) before quoting a row count,
this file has been extended multiple times.

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
  `EURUSD`/`XAUUSD` (Gold = XAUUSD). WTIUSD (Oil) existed under this same convention before
  its 2026-07-04 removal — `utils/assets.py`'s `ASSET_FILE_STEM` still only maps `GOLD`.
- All paths use `pathlib.Path`, all parquet files use `engine="pyarrow"`.

## Regime clustering (`regime/`) — complete

Run: `python run_regime.py`

Outputs:
- `regime/checkpoints/hmm_model.pkl` — fitted GaussianHMM (n_states=7, BIC-selected)
- `regime/checkpoints/pca_model.pkl` — PCA(16) fitted on training embeddings (before 2021-07-01)
- `regime/checkpoints/bic_results.json` — BIC/AIC/LL per n_states for reproducibility
- `data/processed/regime_stream.parquet` — full regime probability stream (183,444 rows as
  of the 2026-07-31 audit, 4 assets BTC/ETH/EURUSD/GOLD — row count grows with every data
  extension, verify with `pd.read_parquet(...).shape` rather than trusting a hardcoded figure)
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

**Regime label availability:** Not all regimes appear for all assets. BTC_USDT has regimes {0,1,4,5,6}. Regimes 2 (stress/crisis) and 3 (bear trend) appear only in GOLD/EURUSD. (OIL/WTIUSD was removed from the pipeline 2026-07-04 — it no longer exists as an asset; this note predates that removal.) Always filter by asset before selecting a regime label. Regime label *semantics* are not stable across HMM refits — see "Regime Label Staleness" in project memory, and `dashboard/backend/regime_service.py`'s `_load_regime_descriptions()`, which derives descriptions from `data/validated/regime_semantics.json` at runtime instead of a hardcoded table for exactly this reason.

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
| `agents/contrarian_agent.py` | `ContrarianAgent` | Devil's advocate; activates at avg_confidence ≥ 0.60; historical analogue retrieval from regime stream |

**SentimentAgent:**
- FinBERT loaded from `./models/finbert` on CPU (`device=-1`)
- News cache at `data/news_cache.json` — 60+ headlines spanning 2018–2021 for BTC, ETH, GOLD, macro
- Looks up headlines within a 24h window before the debate timestamp; includes `macro` headlines for any asset
- Batch FinBERT inference → weighted aggregation (exponential decay, half-life = 6h)
- Sentiment momentum: avg positive score in last 6h vs full 24h window
- LLM call interprets scores; FinBERT provides the numbers, Mistral interprets them

**ContrarianAgent:**
- Activation threshold: avg R1 confidence ≥ 0.60 → Contrarian fires
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
Contrarian: sees all R1 theses; activates if avg_confidence ≥ 0.60; argues opposite direction
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

Run: `python run_rl.py`           (curriculum, capped at Phase 1+2 in practice — see below)
     `python run_rl.py --quick`   (fast end-to-end smoke run)
     `python run_rl.py --eval-only`   (skip training, evaluate + ablate ppo_best.zip)
     `python run_rl.py --seeds 0 1 2`   (multi-seed variance estimate, not yet run as of 2026-07-31)

```
State:  [128-dim regime embedding | 16-dim consensus | 8-dim portfolio] = 152-dim
Action: Discrete direction {0=long, 1=short, 2=hold} × Beta(α,β) position fraction ∈ (0,1)
Reward: r_t = ΔPnL / CVaR_t − λ × drawdown_penalty   (ΔPnL is the per-step return fraction)
Curriculum config still defines 4 phases (BTC → BTC+ETH → +EURUSD → +GOLD), but `run_rl.py`'s
main() hard-caps `config["phases"]` at the first 2 whenever more than 2 are configured — see
"Curriculum collapse" below. OIL/WTIUSD no longer exists as an asset (removed 2026-07-04).
```

### Architecture

- **State (152-dim Box, clipped [-10,10]):**
  - 128-dim: frozen `TFTRegimeEncoder.encode()` on the current 60-candle window (no grad).
  - 16-dim: agent consensus — per-R1-agent signed confidence (+conf bullish / −conf bearish / 0)
    plus FM signed-conviction, position size, regime_override, dissent count. Zeros when no
    debate exists for the timestamp (the common case during training).
  - 8-dim portfolio: 4 per-asset signed-position slots (canonical order BTC,ETH,EURUSD,GOLD —
    `rl.env.CANONICAL_ASSETS`; only one asset trades per episode) + `[pnl_norm, drawdown,
    steps_since_trade/100, vol_forecast]`.
- **Action (hybrid):** exposed to SB3 as a flat 2-dim Box `[direction∈{0,1,2}, position_fraction∈(0.01,0.99)]`
  (SB3 PPO has no Dict action support; `MARTATradingEnv.action_space_dict` documents the semantic
  Dict, `unflatten_action()` recovers it). Direction ~ Categorical, position ~ Beta(α,β) with
  `α,β = softplus(·)+1` (>1 ⇒ unimodal, no boundary clipping bias).
- **CVaR:** GARCH(1,1) volatility forecast → 95% normal CVaR (`-pdf(ppf(.05))/.05 · vol`), the exact
  RiskAgent formula. GARCH is refit weekly and cached to `data/processed/rl_cache/vol_{asset}_*.parquet`
  — never refit per-step (intractable) and the LLM debate system is never called during RL.

### Walk-forward split (strict, current — `rl/env.py` constants)

Train: < 2024-01-01 (`TRAIN_END`). Validation: 2024-01-01 → 2024-12-31 (`VAL_START`/`VAL_END`,
5,928 BTC rows as of the 2026-07-31 audit). Test: 2025 (`run_test_set.py`, SACRED). `reset()`
samples start timestamps only from the active split; evaluation runs `sequential=True` (traverses
the whole val window, ignoring the per-episode `max_steps` cap). The old "train 2018-01-10 →
2020-12-31, val 2021, 6,803 rows" split is two generations stale — this file used to also carry
a per-asset OIL/WTIUSD portfolio slot and 5-asset assumptions that no longer apply.

### Curriculum & callbacks

One `model.learn()` call (total_timesteps = Σ phase max_steps); `CurriculumCallback` advances phases
in-place by mutating the env's active asset list (PPO weights persist across phases — that's the
transfer). Each phase saves `rl/checkpoints/ppo_phase_{n}.zip` and logs to MLflow. Early stop if
val Sharpe < −0.5 for 3 consecutive evals. Best-by-val-Sharpe → `ppo_best.zip`, final → `ppo_final.zip`.

### Ablation (central research result)

**7 variants** (`rl/ablation.py:ABLATION_VARIANTS`, not 5 — `rsi_strategy` and `ma_crossover`
were added after the original 5-variant design): `buy_hold`, `rsi_strategy`, `ma_crossover`,
`ppo_no_regime`, `agents_no_regime`, `ppo_agents_no_regime`, `marta_full`. **Block bootstrap CI**
(n=1000) on the Sharpe delta of `marta_full` vs each baseline → `data/results/ablation_results.json`.

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

### Results & key findings (2026-06-01 — first full run, SUPERSEDED)

**This subsection is a historical snapshot from the very first Month-3 run, on the old
2021 validation window (5 assets including OIL). It has been superseded by at least four
later retrains/splits and does not reflect the current system** (current split is val=2024,
4 assets, see "Walk-forward split" above). Kept for historical context only — do not quote
these numbers as current. See "Eval Reliability Audit (2026-07-31)" below for the current,
verified state.

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

## CRITICAL: current encoder checkpoint has a collapsed temperature

`encoder/checkpoints/best_encoder.pt` (2026-07-14, the Kaggle walk-forward retrain
currently in production) has a learned temperature of **0.0047** — 6.4x below
`encoder.yaml`'s `min_temperature: 0.03` floor. Its `val_top1_acc` was never recorded
anywhere: the checkpoint has no local MLflow run at all (trained on Kaggle, only the
`.pt` file was downloaded back), and `encoder/train.py`'s convergence check only logged
`temperature_in_range` as a WARNING (not CRITICAL), so this went to production unnoticed.

Root cause: `log_temp` in `encoder/loss.py`'s `NTXentLoss` was an unconstrained learnable
parameter — nothing stopped gradient descent from driving it arbitrarily low.

Fixed 2026-07-31 (code only — **the currently-deployed checkpoint itself is unchanged
and still has the collapsed temperature** until it's retrained):
- `NTXentLoss.clamp_temperature()` hard-bounds `log_temp` in place after every optimizer
  step (`encoder/train.py`'s `_run_epoch`), so temperature structurally cannot leave
  `[min_temperature, max_temperature]` on the next retrain.
- `train_encoder()` now embeds `val_top1_acc`, `convergence_checks`, and
  `convergence_thresholds` directly into the saved checkpoint `.pt` file (both `best` and
  `final`), not just MLflow — so a Kaggle-trained checkpoint is self-describing even when
  no local MLflow run exists for it. Check `torch.load(path, weights_only=True)["convergence_checks"]`
  before trusting a new checkpoint.

Until the next retrain happens, treat `best_encoder.pt`'s embeddings with this caveat in
mind — the collapsed temperature is a known contrastive-collapse risk factor, not
independently confirmed to have degraded the embedding space, just unverified.

## Eval Reliability Audit (2026-07-31)

Triggered by validation Sharpe swinging wildly (roughly -1.8 to +1.4) across runs on the
*same* checkpoint. Root causes found and fixed:

**1. Eval was not deterministic.** Nothing in `rl/` or `encoder/` seeded torch/numpy, and
the frozen encoder's forward pass ran on GPU (cuDNN, unseeded) once per env step. With only
8-15 meaningful trades across the whole val window, a handful of borderline direction flips
from float noise was enough to flip the sign of the annualized Sharpe. Fixed:
- `rl/train.py`'s `build_env_factory` now resolves the env's `device` to `"cpu"` for any
  non-`"train"` split (val/all/test) and DEEP-COPIES the encoder before doing so — the
  encoder module is shared across envs and `.to(device)` mutates in place, so without the
  copy a CPU val_env built after a CUDA train_env would silently drag the training encoder
  onto CPU too (verified this crashes training; the fix keeps them independent).
- `rl/evaluate.py`'s `run_walk_forward` seeds `torch`/`numpy` (seed=42) before each pass.
- Every `PPO.load(...)` used purely for evaluation (`run_rl.py`, `scripts/evaluate_all_checkpoints.py`)
  now loads on `"cpu"` instead of `config["device"]`.
- Verified: repeated eval runs on `ppo_best.zip` now return bit-identical Sharpe.

**2. A second, unrelated variable was being conflated with GPU nondeterminism.**
`evaluate_ppo()` defaults to `debate=True` (real consensus signal from
`data/debates/debate_results.json`, which has real 2024 BTC entries). An ad-hoc diagnostic
script that used `debate=False` produced a different — but *equally deterministic* — number.
**Takeaway: always match the debate-cache configuration when comparing eval runs, and note
which config a given Sharpe number came from** — it's not just a checkpoint + code + data
question, the debate cache snapshot matters too and can change over time (see item 3).

**3. Encoder/regime_stream/debate-cache staleness guard added.** A 2026-07-14→07-20 drift
(`best_encoder.pt` swapped without regenerating `regime_stream.parquet`/HMM/PCA to match)
went unnoticed for 6 days. `rl/train.py`'s `load_data()` now calls
`check_encoder_regime_stream_freshness()` first and raises `StaleRegimeStreamError` if
`best_encoder.pt` is newer than `regime_stream.parquet`. **Not yet covered:** the standalone
debate-generation scripts (`run_debate.py`, `run_weekly_debates.py`,
`run_clean_ablation_debates.py`, `run_validation_debates.py`) each define their own local
`_load_data()` and don't route through this guard — `debate_results.json` can still drift
silently relative to the encoder/regime_stream.

**4. Trade-level detail, not just an aggregate Sharpe.** `evaluate_ppo()`'s saved
`data/results/ppo_evaluation.json` now includes a `trade_log`: one record per meaningful
trade (timestamp, direction, size, decision-time regime, step/cumulative PnL) via
`rl/evaluate.py:extract_trade_log()`. With this few real trades, read the trade log, not
just the Sharpe.

**5. Not yet done: multi-seed retraining.** `run_rl.py --seeds 0 1 2` exists but hasn't been
run — the current headline number is still a single deterministic point estimate, not a
distribution. Retraining is deferred to a later session (per-seed: ~30-110 min on this GPU,
so ~1.5-5.5h for 2-3 seeds, sequential — see chat history for the per-run duration evidence).

## Stack

`pytorch-forecasting` (TFT), `hmmlearn` (Gaussian HMM), `stable-baselines3` (PPO), `langgraph` (agent orchestration), `langchain-ollama` (Mistral 7B), `fastapi + APScheduler` (live trading backend — dashboard/backend/signal_scheduler.py; NOT Celery/Redis, despite Celery/Redis appearing in requirements.txt/an optional health-check probe, neither is actually load-bearing), `mlflow` (experiment tracking), `ccxt` (crypto data + execution), `yfinance` (traditional asset data), `ta` (technical indicators), `vectorbt` (backtesting), `rich` (CLI logging)

## Dashboard & Live Trading (`dashboard/`, `pipeline/live/`) — complete, undocumented until 2026-07-31

Not previously documented here despite being fully implemented (git: "MARTA v6.0 complete:
all features, dashboard, live system, LLM chat, Telegram, Alpaca"). This section covers only
what's been directly verified — a full dashboard/live audit is still open (see "Still
missing" below).

- **Backend:** FastAPI (`dashboard/backend/main.py`) + Next.js frontend. Scheduling is
  **APScheduler** (`AsyncIOScheduler`, `signal_scheduler.py`), a `CronTrigger(minute=0)` job
  running `_run_signal_tick()` once per hour on the hour — not Celery/Redis.
- **Regime lookup has two paths, and they are NOT interchangeable:**
  - `regime_service.get_regime(asset, ts)` — nearest-row lookup in the static
    `regime_stream.parquet` snapshot. That file "stops growing the moment `run_regime.py`
    stops running" (the module's own comment) and nothing scheduled regenerates it. A
    `get_regime()` call near "now" silently snaps to the frozen last row forever.
  - `regime_service.get_live_regime(asset, ts)` — recomputes PCA→HMM fresh from a live
    168-hour embedding sequence (a single-frame `predict_proba()` was tried first and found
    to be dominated by the HMM's prior, not the actual observation — a full sequence is
    needed). This is the one that actually changes candle-to-candle.
  - `/api/regime/{asset}` (current) and the hourly `signal_scheduler`/`live_signal` tick
    correctly use `get_live_regime()`. `/api/regime/{asset}/at` (explicit historical
    timestamp) correctly uses `get_regime()`.
  - **Found + fixed 2026-07-31:** `chat_service.py`'s LLM context and `main.py`'s
    `_add_regime()` (tags every candle on the live `/ws/price` websocket) were both still
    using the frozen `get_regime()` lookup. `chat_service.py` now calls `get_live_regime()`
    directly (it's always asking about "now"). `_add_regime()` gained a `live: bool = False`
    parameter — the 300-candle history backfill still uses the cheap historical lookup
    (correct for genuinely past bars, and avoids 300 live HMM recomputes per connection),
    but the `forward()` callback that streams each new real-time tick now passes `live=True`.
    Requires a backend restart to take effect (`docker compose up -d --force-recreate backend`
    — `uvicorn --reload` is known to hang on backend edits in this project, per prior session
    feedback).
- **Execution:** Alpaca paper trading (`alpaca_service.py`), Telegram alerts
  (`telegram_service.py`), LLM chat (`chat_service.py`).
- **Found + fixed 2026-07-31: two more data-provenance bugs, same root cause as above
  (a live/current view silently reading a stale or wrong file):**
  - `debate_stream.py`'s live-mode websocket crashed with `KeyError: 'GOLD'` —
    `all_assets_aligned.parquet`'s MultiIndex columns are keyed by file stem (`XAUUSD`),
    not the logical asset id (`GOLD`) that `agents/graph.py` looks `aligned_dfs` up by.
    Added `utils.assets.logical_asset_name()` (reverse of `ASSET_FILE_STEM`) and used it
    in `debate_stream.py`. This exact mismatch class has now recurred several times across
    different files — check any new code that reads `all_assets_aligned.parquet` directly.
  - The Analytics page showed a 100% win rate next to a -1.278 Sharpe on the same card —
    not a calculation bug, two different data sources: the Sharpe came from the current
    `ppo_evaluation.json` (2024 val window), but `rl_service.get_positions()` (which feeds
    win-rate/best-trade/worst-trade) read `data/results/equity_curve.json`, a snapshot
    frozen since 2026-07-05 describing the old 2021 val window — and when that file's shape
    didn't match, it fell through to `_synthetic_positions()`, which **fabricates** a trade
    log from a sine-wave formula with a hardcoded `2021-01-01` start date. That fabricated
    data is what "Best Trade 1/11/2021 · R3" was. Fixed: `get_positions()` now prefers
    `ppo_evaluation.json`'s real `trade_log` (added earlier in this same audit) as the
    primary source — guaranteed consistent with the Sharpe shown next to it, since both
    come from the same `evaluate_ppo()` run. Verified: win rate now computes to a real
    33.3% (2/6 meaningful trades), consistent with a losing (-1.278 Sharpe) backtest,
    instead of a fabricated 100%. `equity_curve.json` and the synthetic fallback are still
    there as lower-priority fallbacks, unchanged.

**Still missing from this doc (open):** a full pass over `dashboard/backend/`'s other
services (`freshness_service.py`, `live_feed.py`, `paper_exchange.py`, `portfolio_service.py`,
etc.) and `pipeline/live/` — this section only covers what came up while answering a specific
question about regime freshness, not a complete audit.

## EURUSD/GOLD live regime — fixed (2026-08-01); BTC/ETH root cause identified, not yet fixed

Confirmed via `data/processed/regime_live_log.parquet` (the real audit trail) that
`get_live_regime()` was returning **bit-identical** regime probabilities across queries
hours to a week apart, for all 4 assets — not just EURUSD, which is what prompted the
investigation. Root cause: `ppo_inference.py`'s `_get_aligned_df`/`_locate` reads the
static training-time `{file_stem}_normalized.parquet`. `_locate()`'s `searchsorted(...) - 1`
resolves to the same last row for any query timestamp after that file's last row,
regardless of when the query happens.

Asymmetric cause, and the exact boundary differs per asset:
- **EURUSD/GOLD**: genuinely capped at 2025-12-31 by histdata.com's whole-year-only ZIPs
  (no 2026 ZIP exists yet).
- **BTC/ETH**: `{file_stem}_normalized.parquet` (pre-join) was actually already fresh
  through **2026-07-12** (matching `data/raw/BTC_USDT_1h_raw.parquet`'s own end) — the
  2025-12-31 ceiling only affects the POST-join `{file_stem}_aligned.parquet`, which
  `ppo_inference.py` never reads. Corrected from an earlier, wrong assumption in this same
  investigation that BTC/ETH shared EURUSD/GOLD's 2025-12-31 ceiling — they don't, theirs
  is 3 weeks stale (as of 2026-08-01), not 7 months, but the same frozen-`searchsorted`
  mechanism still applies and still needs the equivalent of a live bridge. **Not yet done**
  — BTC/ETH still read the plain static file, unfixed.

**Fixed for EURUSD/GOLD** — `dashboard/backend/live_feature_bridge.py`:
- Pulls real hourly OHLC bars from Twelve Data's `/time_series` (not `/price`, which the
  existing forex poller in `live_feed.py` uses for point-price ticks only — no real
  high/low range, unusable for indicators like ATR/ADX/Bollinger width).
- Verified empirically: a single `outputsize=5000` call closes the *entire* ~7-month gap
  (2026-01-02 → today) in one request — no chunked backfill needed.
- Runs the fetched bars through the exact same `pipeline/features.py` →
  `pipeline/normalize.py` (720h rolling z-score) functions training used — not a
  reimplementation — bridged onto trailing historical context so the rolling window has
  full history right at the boundary.
- Three-file design, not one — **important if touching this again**: a
  `{file_stem}_raw_live.parquet` append-only OHLCV accumulator that is *never* itself run
  through `compute_features()`. An earlier version fed its own already-warmup-trimmed
  output back in as next cycle's context, losing ~20-50 rows/cycle to indicator warmup
  against ~1 new bar/cycle at hourly cadence — that erodes to nothing within days. The
  accumulator being append-only and untouched by the trim is what prevents that.
  `{file_stem}_normalized_live.parquet` is the accumulated output —
  same 14-column schema as the training-time file, ready for `ppo_inference.py` to read.
- Wired into the existing hourly APScheduler tick (`signal_scheduler.py`'s
  `_run_signal_tick` → `_refresh_forex_bridge`, independent of the crypto trading-signal
  loop so a forex-side failure can't block it). Verified end-to-end: first run backfills
  ~5,000 rows per asset, a second run correctly adds only the handful of genuinely new
  bars, and calling it with nothing new logs and no-ops rather than erroring.

**`ppo_inference.py` rewired to match** — `_get_aligned_df` now:
- Prefers `{file_stem}_normalized_live.parquet` when present (`_resolve_aligned_path`),
  falling back to the static file otherwise — so BTC/ETH (no live bridge yet) are
  unaffected and keep working exactly as before.
- Caches by `(path, mtime)`, not "load once forever" — the live-bridge file changes on
  disk every hour and a backend process can live for days between restarts; the old
  cache-once behavior would have made this whole effort silently do nothing for a
  long-running process. Verified: a second call within the same mtime returns the
  identical cached object (no redundant reload); `_get_aligned_df('EURUSD')` returns
  53,798 rows through 2026-08-02 (was frozen at 2025-12-31 before the fix).
- End-to-end verified via `regime_service.get_live_regime()`: querying EURUSD at three
  different timestamps (2026-08-02, 2026-07-15, 2026-06-01) now returns three genuinely
  different results (regime 6/6/5, differing top-probabilities) — confirms the window
  actually slides with real data instead of being pinned to a frozen boundary.

**Still open:**
- `dashboard/docker-compose.yml`'s backend service doesn't pass `TWELVE_DATA_KEY` into the
  container at all (confirmed missing from its `environment:` block) — the key exists in
  `dashboard/backend/.env` but never reaches a dockerized backend process. Blocks this
  whole feature in the actual deployment until fixed.
- BTC/ETH: same frozen-window mechanism, a different (much smaller — weeks, not months)
  gap, and no live bridge built for them yet — not addressed by this work.

## NOTE: root CLAUDE.md staleness — read this before trusting anything above on regime/RL/agent architecture

Everything above this section predates a full regime-layer + agent-layer audit and a
crypto-only 5-asset panel pivot (`BTC_USDT, ETH_USDT, BNB_USDT, LTC_USDT, XRP_USDT`,
"crypto5_multihorizon") that happened in a burst of work this file was never updated to
reflect — including the contrastive encoder being fully abandoned (raw 14-feature input
now backs RL/regime, not TFT embeddings) and a leaky forward-backward-smoothed regime
stream being replaced by a causal one. **The real current state lives in
`docs/RESULTS_AND_METHODOLOGY.md`** (actively maintained) and `docs/REGIME_LAYER_AUDIT_REPORT.{html,pdf}`.
Treat this file as directionally useful engineering history, not a current architecture
reference, until it gets a full pass.

## Dense BTC debate coverage + RL re-eval (2026-09-02)

**Kaggle debate-coverage run completed.** `kaggle_debate_coverage_instructions.md` /
`notebooks/llm-debate-layer.ipynb` generated 366 real Mistral debates (BTC_USDT, daily
stride, all 6 regimes, full 2024 window) and merged into `data/debates/debate_results.json`
(39 → 384 total, backup at `debate_results.json.bak_before_kaggle_merge_2026-09-02`). Real
bugs hit and fixed along the way (notebook + `scripts/kaggle_debate_coverage.py` both
updated): the Ollama installer needs `apt-get install zstd` on the Kaggle base image (not
present by default); `python scripts/kaggle_debate_coverage.py` silently can't import
`agents`/`utils` because that puts `scripts/` (not the repo root) on `sys.path[0]` — fixed
with an explicit `sys.path.insert` at the top of the script itself; `agents/sentiment_agent.py`
does **not** auto-fall-back to pulling `ProsusAI/finbert` from Hugging Face despite what an
earlier version of the instructions doc claimed — it only checks a local path and silently
returns neutral for every debate if that path is empty, so the notebook now explicitly
`snapshot_download`s the weights first.

**LLM-agents-only backtest vs buy-and-hold** (`scripts/backtest_llm_agents_vs_buyhold.py`,
BTC_USDT 2024 val, 5,928 hours, old 4-asset-panel env): rule-based, no PPO — nearest
cached debate (≤7 days) drives direction/size every hour, via `rl/ablation.py`'s
`_run_agents_rule()`. This *is* the regime-conditioned variant — every debate in the cache
was generated with full `RegimeContext` injected into agent prompts, so there is no
separate "without regime" comparison available from this same cache (would need a fresh
`run_clean_ablation_debates.py` generation pass with `include_regime=False`). Result:
buy-hold Sharpe 1.12 (+38.6% return, 45.3% max DD) vs agents-only Sharpe 0.31 (+0.55%
return, **2.5% max DD**, 182 real meaningful trades). Δ Sharpe -0.93, 95% CI
[-3.67, 1.67], not significant — same conclusion as every other result in
`docs/RESULTS_AND_METHODOLOGY.md`, now backed by real dense data instead of a ~1%-coverage
guess. Saved: `data/results/llm_agents_only_backtest.json`.

**Cheap re-eval finding: dense debate coverage does not change trained-PPO behavior,
confirmed on two independent checkpoints.**
- `rl/checkpoints/ppo_best.zip` ("marta_full", hourly cadence) — Sharpe 0.322 (sparse
  cache, previously documented) → 0.326 (384-debate dense cache). Also surfaced a real
  naming/reality mismatch worth knowing: this checkpoint is `obs_mode="raw"`, **868-dim,
  no regime block in the observation at all** — despite `ABLATION_VARIANTS["marta_full"]`
  labeling it `{"regime": True}`. `rl/obs_builder.py`'s `consensus_vec()` is an exact
  timestamp match (not the ablation module's own 7-day nearest-interpolation), so even
  with 384 real debates the observation is non-zero on only ~6.5% of val steps.
  Script: `scripts/reeval_marta_full_dense_debates.py` →
  `data/results/marta_full_dense_debate_reeval.json`.
- Native weekly-cadence checkpoints (`decision_interval=168`, Section 10b's crypto5
  BTC_USDT pilot, seed 0) — condition A (raw): 2.0419 → 2.0407 Sharpe; condition C
  (raw_regime, real HMM): 1.7384 → 1.7391 Sharpe. `n_meaningful_trades` identical
  (37 → 37) in both — the specific trades taken didn't change at all, not just the
  aggregate metric. This is the stronger of the two results: weekly steps land on 168h
  boundaries from a 00:00 UTC anchor and the debate cache is daily at 00:00 UTC, so
  (168 % 24 == 0) means most weekly decisions now exact-match a real debate — a much
  bigger a priori density increase than the hourly case — and it *still* didn't move the
  policy. Script: `scripts/reeval_weekly_dense_debates.py` →
  `data/results/weekly_dense_debate_reeval.json`.

**Why this happens, for both:** the debate cache only densified the *validation* window
(2024) — the training split (pre-2024) still has ~zero debate coverage, so the policy was
never conditioned to exploit a populated consensus slot no matter how often one now
appears at eval time. This is a real, cross-validated confirmation that the RL layer's
"closed" conclusion (Section 9 of the results doc) wasn't an artifact of thin debate-cache
coverage — the trained checkpoints' behavior is essentially bit-identical with 10x more
real data behind the consensus signal.

## Strategy Tester — user-configurable backtest page (2026-09-02)

Added a "Strategy Tester" panel to the dashboard's existing `/backtest` page (not a new
route — sits above the pre-existing "Ablation Study" panel, same page/nav entry). Lets a
user pick asset + strategy + date range and get a real backtest: equity curve chart
(strategy vs. frictionless buy-and-hold) + Total Return / vs Buy&Hold / Sharpe / Max
Drawdown / Win Rate / Trades + a trade log. Default range is the trailing 365 days from
today, matching how a demo audience would expect to use it.

- **Backend:** `dashboard/backend/strategy_backtest.py` — standalone module (numpy/pandas
  only, no torch/stable-baselines3/rl.env import) that reads real OHLCV `close` directly
  from `historical._load(asset)` (the same live-extended, cached loader `/api/history`
  uses — reaches "today", not just the 2025-12-31 static-parquet ceiling) and recomputes
  every indicator itself from price, never from the parquet's precomputed indicator
  columns (those are NaN on live-extension rows — see `historical.py`'s `_extend()`
  docstring — so they can't cover a range that reaches the present). Deliberately
  bypasses `historical.get_history_range()`'s 2000-point step-decimation (fine for
  charting, would silently corrupt indicator windows and return calcs for a >2000-hour
  backtest — 1 year of hourly bars is 8,760).
  9 strategies implemented: `buy_hold`, `ma_crossover` (12/26h), `rsi_oversold`
  (Wilder RSI(14), buy <30 / exit >50), `macd_signal` (12/26/9, directional), `resistance_breakout`
  (20h prior-high entry / rolling-mean exit), `bollinger_band` (20h, 2σ), `trend_following`
  (20/50h SMA, same mechanic as ma_crossover but the literal periods the slide text specified),
  `mean_reversion` (buys in 1/3 increments on return z-score < -2, exits on rebound above SMA),
  `donchian_channel` (20h channel breakout). Every strategy's raw signal is `shift(1)`'d before
  being applied to that bar's return — no lookahead. Transaction cost 0.001 (matches
  `rl/env.py`'s default). "Trade" = a contiguous same-sign-position run; its PnL is the
  compounded product of its own realized per-bar returns (not re-derived from close
  prices) — this stays correct for `mean_reversion`'s staged position sizing, where
  position magnitude changes mid-trade but direction doesn't.
  Endpoints: `GET /api/backtest/strategies` (9 unlocked + 3 "locked"/disabled — Alpha
  Accelerator / News-verified trend / QQQ Sharpe core, shown grayed out in the UI as an
  intentional "coming soon" roadmap hint, not implemented), `GET /api/backtest/run?asset=&strategy=&start=&end=`
  (start/end optional, default trailing 365 days) → `strategy_backtest.run_backtest()`.
- **Frontend:** `dashboard/frontend/src/app/backtest/page.tsx`'s new `StrategyTesterPanel`.
  The asset selector reads/writes the **global** `useMARTAStore.selectedAsset` (not local
  state) — same store the Topbar's asset dropdown and `MARTAChat.tsx` already read, so
  picking an asset here changes it everywhere else too, including what the chat panel
  discusses (`chat_service.py`'s `asset` param). Verified live: setting the panel's asset
  select to `ETH_USDT` immediately updated the Topbar's dropdown to match.
  Verified end-to-end via the dashboard's dev servers (`.claude/launch.json`, added this
  session — conda's uvicorn directly, not `run_dev.sh`, since `conda` itself isn't on the
  preview tool's PATH): all 9 strategies produce distinct, sane metrics on real BTC/ETH
  data reaching 2026-09-02; locked strategies render disabled in the `<select>`; a
  too-short date range surfaces the backend's real 400 error message in the UI, not a
  generic failure.

**2026-09-02, same day — "Alpha Accelerator" promoted from locked placeholder to a real
strategy** (`_sig_alpha_accelerator` in `strategy_backtest.py`), moved out of
`_LOCKED_STRATEGIES` into `_STRATEGIES`. Three independently-computed filters AND-gated,
re-evaluated fresh every bar (no sticky enter/exit — a filter dropping out flattens the
position immediately): trend (20/50h SMA alignment), momentum (RSI(14) >55 bull / <55
bear, dead zone around 50 to avoid a trend-only whipsaw), volatility (20h rolling realized
vol must sit at/below its own trailing 90h percentile — mutes entries during vol spikes).
`news_verified_trend` and `qqq_sharpe_core` stay locked/unimplemented on purpose — the
former needs a historical sentiment score for arbitrary backtest dates (the news cache is
static 2018-2021 headlines only, see `agents/sentiment_agent.py`), the latter needs
US-equity price data entirely outside the crypto5-only pipeline. No frontend changes
needed — the strategy list/lock state is fully server-driven, the `<select>` just renders
whatever `/api/backtest/strategies` returns. Verified on real BTC data: -65.7% return /
Sharpe -4.2 / 587 trades over the trailing 1y window — high-frequency/whipsaw-prone by
construction (no hysteresis between filter states), an honest result rather than a
tuned-to-look-good one, consistent with this project's existing standard of reporting
real backtest numbers even when they're poor.

## Future work: Bull/Bear Researcher agent split (not implemented, 2026-09-02)

Discussed while comparing MARTA's agent roster against TradingAgents' team structure
(Analyst / Research / Trading / Risk / Portfolio Management teams) for the literature
slides. TradingAgents runs an explicit Bull Researcher vs Bear Researcher debate before
its Risk team; MARTA's closest analogue is the single `ContrarianAgent`
(`agents/contrarian_agent.py`) — one devil's-advocate voice, not a symmetric two-sided
debate.

**Proposed change:** split `ContrarianAgent` into paired Bull/Bear Researcher agents that
debate before `FundManagerAgent` synthesis, reusing the existing R1→R2 rebuttal machinery
in `agents/graph.py` rather than new infrastructure — mostly a schema (`agents/schemas.py`)
and prompt change, plus a `FundManagerAgent` synthesis rewrite to weight two new voices
instead of one.

**Why this is worth doing over other RL/encoder work:** the ablation table
(`data/results/ablation_results.json`) already shows `agents_no_regime` (Sharpe 1.456)
far outperforming `marta_full` (Sharpe -2.152, includes the trained PPO) — the LLM agent
layer is where this system's real signal lives, not the RL execution layer (consistent
with Section 9's "RL layer closed" conclusion in `docs/RESULTS_AND_METHODOLOGY.md`), so
expanding agent roster depth is higher-leverage than continuing to chase RL/encoder fixes.

**Cost, not free:** more Ollama calls per debate (latency), and any RL re-evaluation using
the debate cache would need a fresh regeneration pass (`run_clean_ablation_debates.py` or
similar) before the new Bull/Bear signal could be reflected in RL-side results.

Deliberately not implemented this session — captured as a roadmap/future-work item only.
