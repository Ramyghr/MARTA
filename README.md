# MARTA — Multi-Agent Regime-Aware Trading Architecture

Final-Year Engineering Project (PFE) combining contrastive pretraining, LLM multi-agent
deliberation, and PPO reinforcement learning for cross-asset algorithmic trading.

**Python 3.11, conda environment `marta`.**

## What this is

MARTA is a four-layer research pipeline, not a production trading system:

```
Layer 1 — Data pipeline        OHLCV fetch → features → normalize → align → validate
Layer 2 — Regime detection      Gaussian HMM on raw engineered features → regime labels
Layer 3 — Multi-agent debate    LangGraph + Mistral 7B (Ollama): Macro/Technical/Risk/
                                 Sentiment/Contrarian agents → Fund Manager consensus
Layer 4 — RL execution          PPO (Stable-Baselines3) position sizing, regime + consensus
                                 conditioned
Layer 5 — Dashboard & live      FastAPI + Next.js, Alpaca paper trading, Telegram alerts,
                                 LLM chat, live regime/price websockets
```

Everything above runs end-to-end on real data. A FastAPI/Next.js dashboard
(`dashboard/`) sits on top for live monitoring, a strategy backtester, and paper execution
via Alpaca.

## Honest status of the research claim

The original hypothesis — that conditioning agents/RL on latent regime embeddings improves
risk-adjusted returns over regime-agnostic baselines — is **not validated**. This has been
tested repeatedly, not just once: RL execution (hourly and native-weekly cadence, full
crypto5 panel, multiple seeds), and a rule-based LLM-agents-only backtest, were each checked
against buy-and-hold with block-bootstrap significance testing. No configuration produced a
confirmed, statistically defensible edge out-of-sample. Where a lead looked promising
in-sample or at a point estimate, it consistently failed to replicate out-of-sample.

What **is** defensible: a complete, working, end-to-end system; a rigorous methodology
(pre-specified hypotheses, fixed baselines, negative results kept and documented rather than
discarded); and several genuine engineering findings about how multi-stage ML pipelines fail
silently (nondeterministic GPU eval, train/val leakage, stale-artifact drift).

**Read `docs/RESULTS_AND_METHODOLOGY.md` first** — it has the full derivation behind every
result and is the actively maintained source of truth for what's been tried and what the
outcome was. `docs/ARCHITECTURE.md` covers system design. `CLAUDE.md` is a detailed,
session-by-session engineering log (useful for debugging history, not a clean overview).

## Asset panel

The actively-developed panel is **crypto5**: `BTC_USDT, ETH_USDT, BNB_USDT, LTC_USDT,
XRP_USDT`. An older 4-asset panel (`BTC, ETH, EURUSD, GOLD`) still exists and backs the
production dashboard defaults.

## Running the pipeline

```bash
conda run -n marta python run_pipeline.py          # data pipeline: fetch/features/normalize/align/validate
conda run -n marta python run_regime.py            # Gaussian HMM regime clustering
conda run -n marta python run_debate.py            # multi-agent LLM debate demo
conda run -n marta python run_rl.py                # PPO training + evaluation + ablation
conda run -n marta python run_rl.py --eval-only    # evaluate/ablate an existing checkpoint
conda run -n marta mlflow ui                       # experiment tracking UI (http://127.0.0.1:5000)
```

All stages write Parquet/JSON artifacts to `data/` and log to `mlruns/`.

### Dashboard

```bash
cd dashboard
docker compose up -d --force-recreate   # backend (FastAPI) + frontend (Next.js)
```

Requires `dashboard/backend/.env` configured (Alpaca, Telegram, Twelve Data, OAuth client
IDs as applicable). See `dashboard/backend/main.py` for required environment variables.

### Tests

```bash
conda run -n marta pytest tests/
```

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | Data fetch, feature engineering, normalization, alignment, validation |
| `encoder/` | Contrastive (TFT + InfoNCE) encoder — superseded by raw features, kept for reference |
| `regime/` | Gaussian HMM regime clustering, per-asset and shared variants |
| `agents/` | LangGraph multi-agent debate (Macro/Technical/Risk/Sentiment/Contrarian/Fund Manager) |
| `rl/` | Gymnasium env, PPO policy/training/evaluation, ablation harness |
| `dashboard/` | FastAPI backend + Next.js frontend, live trading bridge |
| `configs/` | Pipeline, encoder, and split configuration (`assets.yaml`, `splits.py`) |
| `scripts/` | One-off analysis, backtest, and audit scripts |
| `docs/` | Architecture, methodology/results, known bugs |
| `tests/` | Pytest suite |

## Stack

`pytorch-forecasting`, `hmmlearn`, `stable-baselines3`, `langgraph`, `langchain-ollama`
(Mistral 7B), `fastapi` + `APScheduler`, `mlflow`, `ccxt`, `yfinance`, `ta`, `vectorbt`.
