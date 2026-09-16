# Digest: 1b — Classical/raw-feature regime-detection landscape (round 1)

## Findings

- **Claim:** Hidden Markov Models fit directly on engineered/raw features (returns, volatility) remain a live, actively-published approach for market regime detection through 2025, including newer ensemble variants (bagging/boosting + HMM) and HMM+Random Forest hybrids — none of which involve a learned/contrastive embedding stage.
  **Source:** "A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection and trading" — **Publisher:** AIMS Press — **Published:** 2025-10 — **Accessed:** 2026-09-14 — **Confidence:** medium — **Class:** landscape
  (Corroborating, same period: "Step-by-Step Python Guide for Regime-Specific Trading Using HMM and Random Forest," QuantInsti blog, 2025 — **Confidence:** low, practitioner blog not peer-reviewed — **Class:** landscape)

- **Claim:** A 2025 conference paper explicitly pairs classical HMM regime detection with RL for portfolio management — the "raw features → HMM → RL" pipeline (the exact architecture pattern in question, minus any embedding step) is a current, published research pattern, not a hypothetical.
  **Source:** "HMM-Based Market Regime Detection with RL for Portfolio Management" — **Publisher:** cloud-conf.net (IDS2025 conference proceedings) — **Published:** 2025 — **Accessed:** 2026-09-14 — **Confidence:** medium (conference proceedings, not independently verified as peer-reviewed venue quality) — **Class:** landscape

- **Claim:** CFA Institute's 2025 practitioner brief on unsupervised learning names k-means, spectral clustering, and hierarchical clustering (alongside HMM/GMM implicitly) as standard, currently-recommended tools for market-regime detection and portfolio construction, explicitly framed as tools that "augment rather than replace existing models" and can be "adopted incrementally" — i.e., simple clustering-on-features is treated as mainstream practitioner tooling, not a stopgap.
  **Source:** "Unsupervised Learning I: Overview of Techniques," Chapter 1 — **Publisher:** CFA Institute Research & Policy Center — **Published:** 2025 — **Accessed:** 2026-09-14 — **Confidence:** medium (fetched and read directly, but gave no explicit decision criterion for when clustering suffices vs. when it doesn't) — **Class:** landscape

- **Claim:** Practitioner criticism of classical HMM regime detection converges on three recurring, named failure modes: (1) detection lag — state change confirmed only after it has started; (2) overfitting via too many hidden states chasing noise rather than out-of-sample-separable regimes; (3) Gaussian-emission misspecification during crash/tail events.
  **Source:** Search-aggregated from "Filtering vs. Smoothing: Why Backtests Lie About Regime Detection" (mathandmarkets.com, Substack) and "Feature Selection for HMM Regime Detection" (ACM ICAIT 2025 proceedings) — **Publisher:** mathandmarkets.com / ACM — **Published:** 2025 — **Accessed:** 2026-09-14 — **Confidence:** low (read via search snippet only, not fetched in full) — **Class:** failure-claim

- **Claim:** A specific, named methodological failure — using smoothed (non-causal) HMM regime states in backtests rather than filtered/causal states — inflates apparent backtest performance versus what a live system (which can only use filtered/causal state estimates) would actually achieve.
  **Source:** "Filtering vs. Smoothing: Why Backtests Lie About Regime Detection" — **Publisher:** mathandmarkets.com — **Published:** 2025 (exact date unconfirmed, snippet only) — **Accessed:** 2026-09-14 — **Confidence:** low (single-sourced, snippet-only — flagged per two-source rule as unverified) — **Class:** failure-claim

- **Claim:** Financial-domain self-supervised/contrastive pretraining for time-series representations is an active but still clearly research-stage area as of 2024-2025 (ACM ICAIF 2024 contrastive-asset-embeddings paper, 2025 KTH master's thesis, 2025 ScienceDirect temporal-graph contrastive forecasting paper, Sept-2024 arXiv contrastive-pretraining-for-earnings-season paper) — none of these sources report a production deployment or a head-to-head benchmark against a raw-feature/classical-HMM baseline with concrete numbers.
  **Source:** Aggregated: ACM "Contrastive Learning of Asset Embeddings from Financial Time Series" (ICAIF 2024), KTH DiVA thesis (2025), ScienceDirect "temporal graph-based contrastive approach" (2025), arXiv 2409.17392 "Trading through Earnings Seasons..." (2024-09) — **Publisher:** ACM / KTH / Elsevier / arXiv — **Published:** 2024-2025 — **Accessed:** 2026-09-14 — **Confidence:** low (titles/abstracts only, not fetched in full) — **Class:** landscape

- **Claim:** A September 2025 arXiv paper on "Adaptive and Regime-Aware RL for Portfolio Optimization" (prob-DDPG) reports that embedding probabilistic latent-regime insight into the RL agent's observation substantially improves both profitability and robustness versus not doing so, in simulation and an equity pair-trading application.
  **Source:** "Adaptive and Regime-Aware RL for Portfolio Optimization" — **Publisher:** arXiv (preprint, not confirmed peer-reviewed) — **Published:** 2025-09 — **Accessed:** 2026-09-14 — **Confidence:** low (single-sourced performance claim, preprint, snippet-level — does not meet the two-source bar) — **Class:** performance-number

- **Claim:** A 2026 arXiv preprint argues that regime signals — whether from classical statistical methods or LLM-derived embeddings — frequently fail to translate into practical RL-policy benefit because of "regime boundary" transition periods where the signal is statistically valid but operationally unreliable, and that both classical and learned regime features show predictive-power decay over time (narrower validity windows than assumed).
  **Source:** "When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies," Zhengzhe Yang — **Publisher:** arXiv (solo-author preprint) — **Published:** 2026-04 — **Accessed:** 2026-09-14 — **Confidence:** low (single author, preprint, not peer-reviewed, single-sourced — does not meet two-source bar) — **Class:** failure-claim

- **Claim:** In general (non-finance-specific) time-series classification benchmarking, learned/evolved feature representations (e.g., EvoTSC via genetic programming, LETS-C via text embeddings) outperform or match raw-feature baselines on the majority (>70%) of tested datasets as of 2025-2026 research.
  **Source:** "EvoTSC: Evolving Feature Learning Models for Time Series Classification via Genetic Programming" (arXiv, 2026) and "Leveraging Text Embedding for Time Series Classification" (ACL 2025) — **Publisher:** arXiv / ACL Anthology — **Published:** 2025-2026 — **Accessed:** 2026-09-14 — **Confidence:** low for financial applicability (general TS-classification benchmarks, not regime detection or trading-specific) — **Class:** landscape

- **Claim:** A hybrid pattern distinct from both "pure classical" and "pretrained embedding" is emerging in 2025 research: using an LLM to produce a regime/guidance signal that augments an RL agent's observation space without retraining the RL policy, positioned as a more modular/interpretable alternative to either a from-scratch learned embedding or a purely classical HMM feature.
  **Source:** "Language Model Guided Reinforcement Learning in Quantitative Trading" — **Publisher:** arXiv preprint, under review for FLLM 2025 — **Published:** 2025-08 — **Accessed:** 2026-09-14 — **Confidence:** low (preprint, under review, snippet-level) — **Class:** landscape

## Leads worth chasing
- mathandmarkets.com "Filtering vs. Smoothing" post — most directly relevant practitioner retrospective found, not fetched in full this round; worth a direct fetch next round for the filtering/smoothing lookahead-bias claim and concrete backtest-vs-live deltas.
- alphaarchitect.com/regime-detection/ — 403 on fetch; title suggests a direct practitioner take on "does regime detection have a place in investment strategy"; retry via archive.org.
- arXiv 2605.24490 "Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems" — near-exact match for a multi-agent-LLM-plus-regime architecture; not read this round, high priority.
- arXiv 2604.10996 (Yang) argues regime signals often don't move RL-policy behavior — single preprint; search specifically for a second independent source making a similar claim to upgrade from anecdote to finding.
- **Key gap/tension:** general (non-finance) 2025-2026 TS-classification literature leans toward "learned features usually win," while finance-specific regime-detection literature retrieved here is dominated by classical HMM/clustering with no retrieved source directly benchmarking contrastive embeddings against raw-feature HMM/GMM on a trading task. This head-to-head is the single most decision-relevant thing not yet found.
- CFA Institute's 2025 brief references an underlying March-2025 Mulliner/Harvey/Xia/Fang regime-identification study that may have the "when is X good enough" framing needed.

## What was looked for and could not be found
- A primary source directly comparing a contrastive/self-supervised regime embedding against a raw-feature HMM/GMM baseline on the same downstream RL or multi-agent trading task, with quantified results — none found.
- Any source meeting the two-source bar for a specific performance number or failure/abandonment claim about contrastive pretraining for trading regimes (both candidates found are single-sourced preprints).
- Explicit practitioner guidance with a concrete threshold/criterion for "raw features are good enough" vs. "you need a learned embedding" — not found.
- alphaarchitect.com (403) and mathandmarkets.com full post — not fetched, flagged as follow-ups.

## Assistant self-report
Budget used: 11 tool calls, ~duration 160s.
