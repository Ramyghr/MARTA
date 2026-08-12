# MARTA Code Audit — 2026-08-08

## Executive Summary
The codebase has strong architectural intent across pipeline, encoder, HMM, agents, RL, and dashboard layers, and key logic paths are implemented (not empty stubs). However, this clone is missing critical runtime artifacts (`data/processed/*`, checkpoints, and results JSONs), so many claimed metrics and file-based outputs cannot be verified. Two major code/documentation mismatches were found: Contrarian activation threshold is `0.60` (not `0.65`) and encoder validation still uses augmented views (train-style augmentation leakage into val). Production readiness for defense depends on regenerating/verifying artifacts and resolving these correctness gaps.

## Layer-by-Layer Findings

### Layer 1 — Data Pipeline
Status: ⚠ PARTIAL

Findings:
- Assets/sources are explicitly configured as BTC/USDT + ETH/USDT via CCXT, and EURUSD + XAUUSD via histdata (`configs/assets.yaml:2-15`, `pipeline/fetch.py:363-410`).
- 14 output features are produced as 12 rolling-zscored indicators + `realized_vol_raw` passthrough + `close_return` (`pipeline/normalize.py:29-50`, `pipeline/features.py:29-43`, `pipeline/normalize.py:76-83`).
- Rolling z-score is windowed (not global): mean/std from `series.rolling(window, min_periods)` (`pipeline/normalize.py:52-57`, `pipeline/normalize.py:79-83`).
- Alignment is intersection-based inner alignment using BTC as backbone and common index intersection (`pipeline/align.py:29-53`, `pipeline/align.py:70-73`).
- `realized_vol_raw` is excluded from normalization via `NO_NORMALIZE` and retained in output (`pipeline/normalize.py:45-50`, `pipeline/normalize.py:79-83`).
- Validation includes structural integrity, NaN/Inf checks, OHLCV invariants, overlap thresholds, and crash-volatility sanity checks, so it is more than superficial (`pipeline/validate.py:85-131`, `pipeline/validate.py:139-201`, `pipeline/validate.py:202-293`).

Issues:
- `data/processed/` is absent in this clone, so configured date ranges cannot be compared to actual on-disk processed ranges (`pipeline/normalize.py:112-114`, filesystem check at `/home/runner/work/MARTA/MARTA/data/processed`). — severity: major
- Validation comments and row heuristics contain stale wording (e.g., references to daily upsampling in comments), reducing trust in thresholds/documentation clarity (`pipeline/validate.py:23-26`). — severity: minor

### Layer 2 — Encoder
Status: ⚠ PARTIAL

Findings:
- Architecture confirms VSN → LSTM → gated residual → projection head (`encoder/model.py:30-50`, `encoder/model.py:57-69`).
- Current NT-Xent implementation is one-directional CE over anchor→positive similarity; it is **not symmetric** (no reverse-direction term) (`encoder/loss.py:25-34`, `encoder/loss.py:79-90`).
- RL explicitly freezes encoder params (`requires_grad=False`) (`rl/train.py:121-124`).
- HMM stage uses precomputed embeddings (`embeddings.npy`) and does not retrain encoder (`regime/hmm.py:262-265`, `regime/hmm.py:307-324`).

Issues:
- Validation dataset uses the same contrastive dataset class that always applies augmentations, so augmentation is also applied in validation windows (`encoder/train.py:47-49`, `encoder/dataset.py:36-38`). — severity: major
- `encoder/checkpoints/best_encoder.pt` is missing in this clone, so epoch/val_loss cannot be verified (`encoder/train.py:245-246`, filesystem check at `/home/runner/work/MARTA/MARTA/encoder/checkpoints/best_encoder.pt`). — severity: critical

### Layer 3 — HMM
Status: ⚠ PARTIAL

Findings:
- BIC model selection is implemented with explicit LL/BIC/AIC logging over state grid (`regime/hmm.py:137-166`, `regime/hmm.py:179-213`).
- PCA is fit on training-mask rows only (leakage guard) (`regime/hmm.py:52`, `regime/hmm.py:296-307`).
- Final HMM is refit on training rows only (`regime/hmm.py:314-325`).
- Semantic validation computes per-regime annualized return/vol/sharpe/time-share/dominant asset and volatility distinctness (`regime/validate.py:80-111`, `regime/validate.py:145-151`).

Issues:
- `data/processed/regime_stream.parquet` is missing, so actual date range/assets/regime label coverage cannot be verified in this environment (`regime/hmm.py:343-345`, filesystem check at `/home/runner/work/MARTA/MARTA/data/processed/regime_stream.parquet`). — severity: critical
- Semantic pass criterion is weak (`std(realized_vol_mean) > 0.001` only), which can pass semantically poor regime separations (`regime/validate.py:145-149`). — severity: major
- Internal docstring/comments are inconsistent with current behavior (e.g., "full dataset/full covariance" wording remains near now-training-only/diag flow), indicating stale documentation drift (`regime/hmm.py:104-105` vs `regime/hmm.py:314-325`). — severity: minor

### Layer 4 — Agents
Status: ⚠ PARTIAL

Findings:
- Macro/Technical/Sentiment/Risk/FundManager/Contrarian prompts all include serialized regime context (`agents/macro_agent.py:99-106`, `agents/technical_agent.py:100-108`, `agents/sentiment_agent.py:217-224`, `agents/risk_agent.py:99-105`, `agents/fund_manager.py:38-44`, `agents/contrarian_agent.py:132-140`).
- RiskAgent CVaR math matches normal-CVaR formula (`-pdf(ppf(.05))/.05 * vol`) with GARCH(1,1) forecast fallback (`agents/risk_agent.py:47-57`, `agents/risk_agent.py:85-87`).
- LangGraph flow is explicitly Round1 → Contrarian → Round2 → FundManager (`agents/graph.py:63-76`, `agents/graph.py:81-214`).
- Conviction is computed from weighted thesis/rebuttal scores and regime multipliers, not hardcoded (`agents/fund_manager.py:46-93`, `agents/fund_manager.py:107-117`).

Issues:
- Contrarian activation threshold is `0.60`, not the expected `0.65` (`agents/contrarian_agent.py:18`, `agents/contrarian_agent.py:151-154`). — severity: major
- FinBERT is conditional on local model path and transformers availability; with missing `models/finbert`, runtime degrades to neutral sentiment mode (`agents/sentiment_agent.py:52-64`, filesystem check at `/home/runner/work/MARTA/MARTA/models/finbert`). — severity: major
- Debate graph still contains a `ppo_stub` node in the agent workflow (`agents/graph.py:68`, `agents/graph.py:243-245`). — severity: minor

### Layer 5 — RL
Status: ⚠ PARTIAL

Findings:
- 152-dim observation is correctly assembled as 128 regime + 16 consensus + 8 portfolio (`rl/env.py:53-57`, `rl/env.py:410-415`).
- Beta position head is correctly implemented with `softplus + 1` for α/β and combined with categorical direction (`rl/policy.py:63-65`, `rl/policy.py:79-82`, `rl/policy.py:99-103`).
- CVaR-normalized reward is implemented as `risk_adjusted - drawdown_penalty - idle_penalty` with CVaR floor (`rl/env.py:566-588`).
- `val_env` is forced to `idle_penalty=0.0` and callback keeps it zero during phase changes (`rl/train.py:243-249`, `rl/callbacks.py:164-180`).
- `REGIME_POSITION_SCALE` is disabled/commented out and not applied in step logic (`rl/env.py:77-93`, `rl/env.py:489-493`).
- Ablation variant set contains 7 variants and explicit execution paths (`rl/ablation.py:27-35`, `rl/ablation.py:371-425`).

Issues:
- `rl/checkpoints/ppo_best.zip` and `ppo_phase_4.zip` are missing here, so equality check cannot be performed (`rl/train.py:328-333`, filesystem check at `/home/runner/work/MARTA/MARTA/rl/checkpoints/`). — severity: critical
- Because `data/results/*` files are missing, expected Sharpe and checkpoint-ranking outputs are unverifiable in this clone (`rl/ablation.py:22-23`, `run_rl.py` eval output paths). — severity: major

### Layer 6 — Dashboard Backend
Status: ⚠ PARTIAL

Findings:
- API surface is broad and implemented with concrete handlers (no obvious endpoint stubs) (`dashboard/backend/main.py:133-594`, `dashboard/backend/main.py:686-794`).
- PPO inference correctly targets `ppo_best.zip`, freezes encoder when loaded, and falls back safely if unavailable (`dashboard/backend/ppo_inference.py:23-25`, `dashboard/backend/ppo_inference.py:57-75`, `dashboard/backend/ppo_inference.py:97-103`, `dashboard/backend/ppo_inference.py:107-113`).
- Hourly scheduler is configured via APScheduler CronTrigger at minute 0 (`dashboard/backend/signal_scheduler.py:51-67`).
- Telegram notifications are wired for regime shifts and trade signals (`dashboard/backend/signal_scheduler.py:226-277`, `dashboard/backend/telegram_service.py:20-21`, `dashboard/backend/telegram_service.py:47-66`).
- Chat service injects live regime + portfolio + debate + fear/greed context (`dashboard/backend/chat_service.py:84-123`, `dashboard/backend/chat_service.py:130-146`).
- Alpaca/testnet/paper flow keeps paper_db as continuous source of truth even when Alpaca is used (`dashboard/backend/signal_scheduler.py:196-203`, `dashboard/backend/live_signal.py:296-330`).
- Price fallback chain is implemented in order: Twelve Data → yfinance → stale cache → parquet (`dashboard/backend/price_service.py:44-70`).
- Docker data mount is **not** read-only (`dashboard/docker-compose.yml:7`).

Issues:
- With missing local checkpoints/results, backend will run in fallback/degraded mode for core inference/results endpoints (`dashboard/backend/ppo_inference.py:70-75`, `dashboard/backend/portfolio_service.py:9-12`). — severity: major

### Layer 7 — Dashboard Frontend
Status: ✓ COMPLETE

Findings:
- All 8 feature routes exist with real page implementations: dashboard, trading, regime, portfolio, backtest, debates, analytics, settings (`dashboard/frontend/src/app/dashboard/page.tsx:61`, `dashboard/frontend/src/app/trading/page.tsx:393`, `dashboard/frontend/src/app/regime/page.tsx:364`, `dashboard/frontend/src/app/portfolio/page.tsx:282`, `dashboard/frontend/src/app/backtest/page.tsx:406`, `dashboard/frontend/src/app/debates/page.tsx:529`, `dashboard/frontend/src/app/analytics/page.tsx:397`, `dashboard/frontend/src/app/settings/page.tsx:403`).
- Zustand store is implemented centrally for global app mode, selected asset, debate/decision, websocket state, and shared UI state (`dashboard/frontend/src/lib/store.ts:68-108`).
- Asset selector is globally wired in Topbar to store state (`dashboard/frontend/src/components/layout/Topbar.tsx:41-43`, `dashboard/frontend/src/components/layout/Topbar.tsx:86-89`), and core pages consume `selectedAsset` (`dashboard/frontend/src/app/dashboard/page.tsx:65`, `dashboard/frontend/src/app/trading/page.tsx:394`, `dashboard/frontend/src/app/regime/page.tsx:365`, `dashboard/frontend/src/app/debates/page.tsx:314`, `dashboard/frontend/src/app/analytics/page.tsx:398`).
- Final Decision panel is guarded against asset leakage and reloaded per selected asset (`dashboard/frontend/src/app/dashboard/page.tsx:84-87`, `dashboard/frontend/src/app/dashboard/page.tsx:97-107`, `dashboard/frontend/src/app/dashboard/page.tsx:212-214`).
- `MARTAChat` is mounted in root layout (`dashboard/frontend/src/app/layout.tsx:6`, `dashboard/frontend/src/app/layout.tsx:21`).
- `NotificationBell` is mounted in Topbar (`dashboard/frontend/src/components/layout/Topbar.tsx:8`, `dashboard/frontend/src/components/layout/Topbar.tsx:148`).

Issues:
- Backtest page does not consume `selectedAsset` directly (likely intentional because it is evaluation-centric, but behavior differs from other pages) (`dashboard/frontend/src/app/backtest/page.tsx:406`). — severity: minor

### Layer 8 — Results & Data Files
Status: ✗ BROKEN

Findings:
- Expected artifacts are referenced by code paths (`dashboard/backend/portfolio_service.py:9-10`, `dashboard/backend/portfolio_service.py:54-55`, `rl/ablation.py:22-23`, `dashboard/backend/ppo_inference.py:23-25`) but are missing on disk in this clone.
- `data/processed/` itself is missing, so regime stream and aligned parquet checks cannot be executed.

Issues:
- Missing required files prevent verification of all target numeric claims (Sharpe, variant counts, checkpoint ranking/equality, debate counts, regime stream stats). — severity: critical

## Critical Issues (must fix before defense)
- Missing runtime artifacts in repository clone: `data/processed/*`, `data/results/*`, `data/debates/debate_results.json`, encoder and PPO checkpoints (filesystem check; referenced by `dashboard/backend/portfolio_service.py:9-10`, `dashboard/backend/ppo_inference.py:23-25`, `rl/ablation.py:22-23`).
- Encoder best checkpoint not present, blocking checkpoint metadata verification (`encoder/train.py:245-246`).
- Regime stream parquet not present, blocking HMM output verification (`regime/hmm.py:343-345`).

## Minor Issues (polish before defense)
- Stale/misaligned comments in several modules (e.g., old split phrasing and old covariance wording) (`encoder/train.py:4-6`, `regime/hmm.py:104-105`, `pipeline/validate.py:23-26`).
- Debate graph still labels final stage as `ppo_stub` (`agents/graph.py:68`, `agents/graph.py:243-245`).
- Backtest route does not bind to global asset selection like most pages (`dashboard/frontend/src/app/backtest/page.tsx:406`).

## Confirmed Working (evidence-based)
- Rolling (windowed) normalization, not global normalization (`pipeline/normalize.py:52-57`, `pipeline/normalize.py:79-83`).
- Inner-intersection multi-asset alignment (`pipeline/align.py:49-53`, `pipeline/align.py:72-73`).
- `realized_vol_raw` passthrough unnormalized (`pipeline/normalize.py:45-50`, `pipeline/normalize.py:79-83`).
- RL val-env idle penalty remains fixed at 0.0 (`rl/train.py:243-249`, `rl/callbacks.py:164-180`).
- `REGIME_POSITION_SCALE` disabled and not used (`rl/env.py:77-93`, `rl/env.py:489-493`).
- Scheduler hourly cadence on :00 is configured (`dashboard/backend/signal_scheduler.py:59-63`).
- Docker data mount is writable (`dashboard/docker-compose.yml:7`).

## Results Verification
- `data/results/ppo_evaluation.json` → **DENY (file missing)**
- `data/results/ablation_results.json` → **DENY (file missing)**
- `data/results/test_set_results.json` → **DENY (file missing)**
- `data/results/checkpoint_evaluation.json` → **DENY (file missing)**
- `data/debates/debate_results.json` → **DENY (file missing)**
- `data/debates/debate_results_no_regime.json` → **DENY (file missing)**
- `data/processed/regime_stream.parquet` → **DENY (file missing)**
- `data/validated/regime_semantics.json` → **DENY (file missing)**
- `encoder/checkpoints/best_encoder.pt` → **DENY (file missing)**
- `rl/checkpoints/ppo_best.zip` → **DENY (file missing)**
- `rl/checkpoints/ppo_best.zip == rl/checkpoints/ppo_phase_4.zip` → **DENY (both missing)**

## Verdict
Code structure and core logic are generally solid and sophisticated, but this repository snapshot is not audit-complete for defense because critical generated artifacts/checkpoints are absent, preventing validation of the central empirical claims. In code, the largest correctness issue is validation-time augmentation leakage in encoder training and the Contrarian threshold mismatch versus stated spec. Regenerating/verifying the full artifact set and resolving those two logic mismatches is required before presenting research outcomes as confirmed.
