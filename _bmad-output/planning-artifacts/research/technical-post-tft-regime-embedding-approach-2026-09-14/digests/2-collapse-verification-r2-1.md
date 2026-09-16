# Digest: 2 — Verify collapse-mitigation citations + mean-pooling signal loss (round 2)

## Findings

- **Claim:** Wang & Isola (2020), "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" (ICML 2020) CONFIRMED as the origin of "alignment"/"uniformity" as formal, optimizable embedding-quality metrics. Alignment = closeness of positive-pair features (`E[||x−y||₂^α]`, lower = closer). Uniformity = how evenly normalized features spread on the hypersphere (`log E[exp(−t·||x−y||₂²)]`, lower = more uniform). The paper shows these are what contrastive loss asymptotically optimizes, and directly optimizing them can match/beat standard contrastive loss downstream — establishing them as diagnostic proxies. Low uniformity (features clustered in a small hypersphere region) is the standard signal used to flag representation/dimensional collapse in practice.
  **Source:** arxiv.org/abs/2005.10242 + tongzhouwang.info/hypersphere/ (author's page) — **Publisher:** arXiv / ICML 2020 — **Published:** 2020-05-20 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** foundational-theory (correctly foundational, not current-state)

- **Claim:** No direct, explicit documented finding that mean/average pooling SPECIFICALLY (vs. attention/last-token pooling) in a contrastive TIME-SERIES encoder discards short-horizon signal. A targeted check of "What Constitutes Good Contrastive Learning in Time-Series Forecasting?" (arXiv 2306.12086) — which surveys pooling choices including TS2Vec/HCL's hierarchical max-pooling — does NOT analyze mean-pooling vs. attention/last-token pooling or short-horizon information loss; it focuses on backbone/augmentation/loss-function comparisons instead.
  **Source:** arxiv.org/html/2306.12086v2 — **Publisher:** arXiv — **Published:** 2023 (v2 2024) — **Accessed:** 2026-09-14 — **Confidence:** high (as a negative/absence finding for this specific paper) — **Class:** other

- **Claim:** Adjacent, non-financial, non-contrastive-time-series evidence supports the general mechanism by analogy but doesn't directly confirm it. A TSC pooling review notes global/mean pooling "discards the temporal position of hidden features" (supervised TSC, not contrastive). Separately, in short-context sequence tasks, attention pooling was chosen over mean pooling because "mean pooling treats all tokens equally, diluting the signal from relevant tokens," and mean pooling "consistently underperform[s]... on short context tasks relative to attention pooling."
  **Source:** AAAI dynamic-temporal-pooling-for-TSC paper (cdn.aaai.org/ojs/17008) + "Playing DOOM with 1.3M Parameters" (arXiv 2604.07385) — **Publisher:** AAAI / arXiv — **Published:** 2021 / 2026 — **Accessed:** 2026-09-14 — **Confidence:** medium (documented and directly analogous mechanism, but off-target domain — inference-by-analogy, not a direct hit) — **Class:** other, supporting/analogous

- **Claim:** CONFIRMED — "Temporal Contrastive Transformer for Financial Crime Detection" (arXiv 2605.21490) explicitly reports its contrastive embedding is redundant with existing engineered features: standalone embedding AUC = 0.8644; combined with domain-engineered features, AUC = 0.9205 vs. engineered-features-only baseline of 0.9245 — adding the contrastive embedding produced NO measurable improvement (numerically slightly worse). Authors' own interpretation: "the learned representations largely overlap with existing feature abstractions," framed as mildly positive (competitive without manual feature engineering) but explicitly caveated as "not yet production-ready."
  **Source:** arxiv.org/abs/2605.21490 (abstract HTML, fetched successfully this round — prior PDF fetch had failed) — **Publisher:** arXiv — **Published:** 2026-05 — **Accessed:** 2026-09-14 — **Confidence:** high (quoted directly from abstract, now single-sourced-but-verified rather than unverified) — **Class:** failure-claim

- **Claim:** No documented case found of contrastive/self-supervised pretraining used specifically ahead of an HMM-style regime detector (vs. ahead of a forecasting or direct trading-signal model). Three query framings all returned only: conventional (non-contrastive) HMM regime-detection tutorials using engineered features directly; contrastive-pretraining papers in unrelated domains (financial crime/AML, kernel-regime SSL theory); and a stock-trend SSL framework (SMART/S3E) using self-supervised embeddings for direct trend prediction, not HMM regime classification. No paper combining contrastive pretraining + HMM regime detection was surfaced.
  **Source:** aggregate of 3 WebSearch queries (QuantInsti, Medium/PyQuantLab, QuestDB, arXiv 2605.21490, arXiv 2507.08835, ScienceDirect SMART/S3E) — **Published:** various, 2023-2026 — **Accessed:** 2026-09-14 — **Confidence:** medium (absence-of-evidence via web search, not exhaustive literature search) — **Class:** other (absence finding)

## Leads worth chasing
- Full Wang & Isola PDF (formal proofs) if underlying math wanted.
- SMART/S3E (ScienceDirect, self-supervised stock sequence embedding) — closest financial analogue found; check for any pooling-strategy ablation.
- arXiv 2507.08835 (contrastive transformer for money-laundering detection) — could be a second redundancy-with-engineered-features data point beyond 2605.21490.
- Citation-forward search on Wang & Isola (Semantic Scholar "cited by") for a paper ablating mean- vs. attention-pooling WITHIN a contrastive time-series pretraining objective specifically — not found this round, would most directly close the pooling gap.

## What was looked for and could not be found
- No controlled ablation of mean/average vs. attention/last-token pooling specifically within a contrastive (not supervised) time-series pretraining objective, measuring short-horizon signal retention.
- No documented instance of contrastive/self-supervised pretraining used directly ahead of an HMM regime detector for trading/markets.

## Assistant self-report
Budget used: 11 tool calls, ~duration 119s.
