# Digest: 1a — SSL/contrastive-embedding landscape for regime detection (round 1)

## Findings

- **Claim:** Dolphin (Smyth & Dong, 2024), a finance-specific contrastive-learning framework, learns asset embeddings from financial time series using a statistical (hypothesis-test-based) positive/negative sampling strategy to cope with noisy, stochastic return data, explores multiple contrastive loss variants, and evaluates on industry classification and portfolio optimization, reporting the embeddings "significantly outperform existing baselines" (magnitude not in abstract).
  **Source:** "Contrastive Learning of Asset Embeddings from Financial Time Series" — **Publisher:** arXiv / ACM ICAIF '24 — **Published:** 2024-07-26 — **Accessed:** 2026-09-14 — **Confidence:** medium — **Class:** landscape (outperformance magnitude unverified, single-sourced)

- **Claim:** Kronos (Tsinghua, 2025) is a decoder-only foundation model pretrained autoregressively (next-K-line prediction) on a specialized tokenizer discretizing OHLCV "K-line" data, trained on >12 billion K-line records across 45 global exchanges and multiple asset classes (crypto, forex, equities, futures) and timeframes — domain-specific large-scale self-supervised pretraining, not a repurposed generic TSFM.
  **Source:** "Kronos: A Foundation Model for the Language of Financial Markets" — **Publisher:** arXiv (2508.02739) — **Published:** 2025-08-02 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** landscape

- **Claim:** Kronos reports large relative gains over a leading prior TSFM (+93% RankIC on price-series forecasting) and over the best non-pretrained baseline (+87% RankIC), plus 9% lower MAE on volatility forecasting and 22% better generative fidelity for synthetic K-line sequences.
  **Source:** Kronos arXiv paper, corroborated in a secondary write-up — **Publisher:** arXiv / jonathankinlay.com — **Published:** 2025-08 / 2026-02 — **Accessed:** 2026-09-14 — **Confidence:** low-medium — **Class:** performance-number — **Caveat: the "corroborating" source is derivative commentary on the same paper, not independent — treat as single-sourced/self-reported/unreplicated. Metrics are statistical-loss based (MSE/CRPS/RankIC), not backtested economic returns net of costs — a real gap for a trading-decision use case.**

- **Claim:** Time-series foundation-model benchmarks can hide regime-dependent failure: models strong on aggregate metrics degrade sharply under specific operating regimes when performance is segmented by condition rather than averaged.
  **Source:** "Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures? Evidence from Traffic Speed Forecasting" — **Publisher:** arXiv (2606.18367) — **Published:** 2026-06 — **Accessed:** 2026-09-14 — **Confidence:** medium — **Class:** failure-claim — **Caveat: traffic-speed forecasting, NOT finance. Single-sourced for the general SSL/foundation-model point, zero direct finance corroboration found — an analogy, not a verified financial-domain finding.**

- **Claim:** An emerging (Aug 2026) direction skips a separate embedding→regime-cluster→RL pipeline entirely, instead having an RL trading agent discover its own self-supervised auxiliary prediction tasks end-to-end via meta-gradients (a "question network" proposing General Value Functions, jointly trained with PPO), motivated by fixed/handcrafted auxiliary tasks "perform[ing] well under certain market conditions [but] may fail under others" — regime non-stationarity breaking static pretraining objectives.
  **Source:** "QUESTrader" / "Self-Supervised Auxiliary Task Discovery for Stable Reinforcement Learning in Stock Trading" — **Publisher:** arXiv (2608.15841) — **Published:** 2026-08 — **Accessed:** 2026-09-14 — **Confidence:** low-medium — **Class:** performance-number — **Caveat: single-sourced preprint, no independent replication; gains (DJI: 21.8% vs 15.7% annual return, Sharpe 1.46 vs 1.08 over plain PPO) self-reported, 4 equity indices only.**

- **Claim:** Masked/patch-based self-supervised pretraining (masked reconstruction — MOMENT/"Time-Series Pile", TimeBERT, TSLANet, SimMTM, TimeMAE, TimeSiam, and a newer mask-free JEPA-style variant "CF-JEPA") is an active, broad family of general (not finance-exclusive) time-series SSL objectives being proposed/refined through 2024-2026, positioned as an alternative to contrastive objectives.
  **Source:** Search synthesis over multiple arXiv papers (2504.04011 survey; 2404.08472 TSLANet; 2402.02475 TimeSiam; 2412.15315; 2606.07031 CF-JEPA) — **Publisher:** arXiv — **Published:** 2024-02 to 2026-06 — **Accessed:** 2026-09-14 — **Confidence:** low — **Class:** landscape — **Caveat: read only via search-engine synthesized snippets, not full papers; existence/titles verified, details not independently confirmed.**

- **Claim:** Generic (non-finance) TSFMs are reported to often underperform dedicated domain-specific architectures on financial data because of financial data's extreme noise, non-stationarity, heavy tails, and regime changes — motivating finance-specific pretraining (like Kronos) over reusing general-purpose SSL time-series models.
  **Source:** Kronos commentary piece — **Publisher:** jonathankinlay.com — **Published:** 2026-02 — **Accessed:** 2026-09-14 — **Confidence:** low — **Class:** landscape — **Caveat: single secondary/commentary source, not a controlled benchmark study; marketing-adjacent register in parts.**

## Leads worth chasing
- Kronos peer-review status unclear (arXiv listing vs. possible AAAI OJS proceedings link) — verify venue before treating numbers as peer-reviewed vs. self-reported.
- QUESTrader (arXiv 2608.15841) directly challenges the "pretrain fixed embedding → downstream RL/HMM" pattern, arguing static pretraining objectives break under regime shift — argued as a *replacement*, not addition. Worth a deeper read.
- "Do TSFM Benchmarks Hide Regime-Dependent Failures?" is non-financial (traffic); a finance-specific analog was not found — would meaningfully strengthen/weaken the "worth the complexity" case either way.
- DeepTrader (regime/cluster-embedding RL portfolio manager, likely ~2021-22, outside freshness bar) and an MDPI paper "Deep RL for Financial Trading: Enhanced by Cluster Embedding and Zero-Shot Prediction" (403'd) — both close structural analogs, worth a follow-up fetch.
- Graph-based contrastive learning for financial forecasting (ScienceDirect) and cross-sectional future-aligned soft contrastive learning (arXiv 2602.10711, Feb 2026) — look like multi-asset extensions of InfoNCE-style approaches, relevant to a multi-asset system; not read.
- **Key gap:** no source directly compared "contrastive-embedding-then-HMM-then-RL" as one three-stage pipeline against "raw/engineered features straight into RL" in a controlled ablation — the crux the decision needs, not found verified anywhere.

## What was looked for and could not be found
- No finance-specific, controlled study directly answering "does a learned regime/state embedding ahead of HMM+RL beat engineered features feeding HMM+RL directly" — every source addresses either the embedding step alone or the RL step alone, never the full three-stage pipeline with a no-embedding ablation control.
- No second, independent source (outside the authors' own paper and derivative commentary) corroborating Kronos's specific reported numbers — could not clear the two-source bar.
- No finance-domain-specific study of "regime-dependent failure hidden by aggregate benchmarks" — only the traffic analog found.
- Could not access the MDPI "Cluster Embedding and Zero-Shot Prediction" paper (403).
- No source explicitly declared contrastive-pretraining-for-regime-embedding a failed/abandoned approach in production trading — comparative language found frames classical baselines as outperformed, not embedding approaches as abandoned.

## Assistant self-report
Budget used: 13 tool calls, ~duration 204s.
