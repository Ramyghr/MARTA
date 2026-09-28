# Digest: 3 — Named-firm practitioner reversion stories (round 2)

## Findings

- **Claim:** Two Sigma's published regime-detection methodology ("A Machine Learning Approach to Regime Modeling") uses a Gaussian Mixture Model over 17 hand-engineered macro/equity/style factors from the in-house "Two Sigma Factor Lens" — NOT a learned/contrastive embedding. No comparison against a representation-learning alternative, no mention of ever trying or reverting from one.
  **Source:** twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/ — **Publisher:** Two Sigma (firm engineering/research blog) — **Published:** undated, several years old per external references — **Accessed:** 2026-09-14 — **Confidence:** high (fetched/read directly) — **Class:** practitioner-report (primary, non-comparative — no reversion story present)

- **Claim:** In the Kaggle "Jane Street Market Prediction" competition, the 1st-place solution ("Yirun's Solution"/team Cats Trading) used a supervised autoencoder jointly trained with an MLP (decoder MSE + autoencoder trade/no-trade BCE + downstream MLP BCE, end-to-end to avoid sequential-training leakage). Actual leaderboard/CV score delta vs. a simpler raw-feature baseline could NOT be retrieved — the writeup page is JS-rendered and the fetch tool returned only its title.
  **Source:** kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s — **Publisher:** Kaggle (competition writeup) — **Published:** ~2021 — **Accessed:** 2026-09-14 — **Confidence:** low-medium on methodology description (search-synthesis, not verified direct quote); NO quantitative delta obtained — **Class:** practitioner-report (unverified)

- **Claim:** A different competitor's own public repo for the same competition reports a concrete number: their autoencoder-based model scored 8358.763 on the public leaderboard vs. 5557.170 for their XGBoost (tree/engineered-feature) model — the autoencoder substantially outperformed the tree baseline in this individual's own experiments, and became part of their best ensemble.
  **Source:** github.com/scaomath/kaggle-jane-street/blob/main/README.md — **Publisher:** individual Kaggle competitor (GitHub, self-published) — **Published:** ~2021 — **Accessed:** 2026-09-14 — **Confidence:** medium (real numbers from a primary practitioner log, but one participant's own comparison, not an official controlled ablation; scale/methodology of "utility score" not independently verified) — **Class:** performance-number (single-sourced) — success-story direction (embeddings won), NOT a reversion story

- **Claim:** An independent quant blogger (Quantitativo) cites an academic claim (Gabaix et al.) that "a simple 4-dimensional embedding explains over 50% of the variation in relative valuations, compared to just 15% for standard firm characteristics," and presents own embedding-based strategy backtest (Sharpe 2.59, 2020-2025) — with NO side-by-side comparison against a raw/engineered-feature baseline strategy, and no reversion account.
  **Source:** quantitativo.com/p/asset-embeddings — **Publisher:** Quantitativo (independent quant blog) — **Published:** 2025 (exact date unconfirmed) — **Accessed:** 2026-09-14 — **Confidence:** medium — **Class:** practitioner-report (success-story register, no comparative/reversion evidence)

- **Claim:** No named-firm engineering blog or talk (Man Group/Man AHL, HRT, DRW, Jump Trading, Citadel, Voleon, QuantConnect) surfaced any content on representation learning/embeddings for regime detection this round — only Two Sigma's engineered-feature GMM piece surfaced from the named-firm list.
  **Source:** absence finding across multiple WebSearch queries — **Accessed:** 2026-09-14 — **Confidence:** medium (searches not exhaustive per firm; 1-2 queries per name given tool budget) — **Class:** other (absence-of-evidence, not confirmed negative result)

## Leads worth chasing
- Re-fetch Jane Street 1st-place Kaggle writeup with a JS-rendering tool (or Google cache/archive.org) for actual reported CV/utility-score numbers.
- Kaggle "Jane Street Real-Time Market Data Forecasting" (Oct 2024 rerun) winner walkthrough video (Team Patrick Yam, YouTube watch?v=lfzzPZZyzjE) — not analyzed, could contain direct 2024-era commentary.
- Optiver "Trading at the Close" and Ubiquant Market Prediction retrospectives — not reached (budget).
- r/algotrading, r/quant, QuantNet, Wilmott — only queried via generic web search, not site-restricted; a site-restricted search would likely surface more.
- scaomath GitHub repo's "8358.763 vs 5557.170" numbers deserve a direct read of the repo body (not just README) to confirm comparability (same CV fold/evaluation formula).

## What was looked for and could not be found
- No genuine reversion story — a named firm or Kaggle team that explicitly tried a learned-embedding/autoencoder approach and switched back to raw/engineered features (or vice versa) with a stated reason. Every practitioner-adjacent finding is either (a) a firm using engineered features with no comparison ever attempted (Two Sigma), or (b) success-story content where embeddings/autoencoders won and were kept (Jane Street Kaggle winners, scaomath's repo, Quantitativo).
- No r/algotrading/r/quant/QuantNet/Wilmott thread describing a negative or mixed result for embeddings vs. raw features in trading — genuine scarcity or search-coverage gap, inconclusive (coverage was thin, not site-restricted).
- Mirrors round 1's finding: survivorship/launch-bias in publicly available practitioner content appears real and persists here too.

## Assistant self-report
Budget used: 13 tool calls, ~duration 180s.
