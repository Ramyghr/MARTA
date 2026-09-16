# Digest: 2 — Contrastive-collapse failure modes & mitigation patterns (round 1)

## Findings

- **Claim:** Representation/dimensional collapse in contrastive SSL — embeddings collapsing to a constant vector, or (more subtly) spanning only a low-dimensional subspace rather than a genuine constant — arises from two identified mechanisms: strong/implicit-regularizing augmentation and network over-parameterization (linear projector dynamics). A general property of contrastive SSL, not modality-specific.
  **Source:** "Understanding Dimensional Collapse in Contrastive Self-supervised Learning" (Jing, Vincent, LeCun, Tian) — **Publisher:** arXiv / ICLR 2022 (Meta AI Research) — **Published:** 2021-10 (v1), ICLR 2022 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** foundational-theory

- **Claim:** The InfoNCE/NT-Xent temperature parameter τ directly controls the "hardness-aware" penalty on negative pairs; low τ sharpens the similarity distribution and pushes toward greater uniformity, but excessive uniformity destroys tolerance for semantically-similar samples — framed as a "uniformity-tolerance dilemma" requiring a properly calibrated (not extreme) temperature for good downstream features.
  **Source:** "Understanding the Behaviour of Contrastive Loss" (Wang & Liu) — **Publisher:** CVPR 2021 / arXiv 2012.09740 — **Published:** 2020-12 (v1), CVPR 2021 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** foundational-theory

- **Claim:** Low/extreme temperature values in contrastive training are associated with training instability — "overly strict separation between features," high-variance gradients, optimization difficulty, distinct from but related to collapse.
  **Source:** Aggregated secondary discussion (Medium explainer; OpenReview "Analyzing the Impact of Learnable Softmax Temperature in Contrastive Visual-Textual Alignment Systems") — **Publisher:** Medium / OpenReview — **Published:** various, 2023-2025 — **Accessed:** 2026-09-14 — **Confidence:** low — **Class:** failure-claim (single class of evidence, not two independent primary papers — unverified)

- **Claim:** Standard time-series augmentations for contrastive positive pairs (jittering, scaling, channel permutation, temporal masking, random cropping) frequently fail to preserve downstream-discriminative signal: jittering corrupts shared latent content with additive noise, channel permutation destroys multivariate dependency structure, masking strictly reduces mutual information between view and label whenever masked content is itself discriminative — producing "faulty positive pairs."
  **Source A:** "Learning by Shifting: Temporal View Construction for Time Series Contrastive Learning" (ShiFT) — **Publisher:** arXiv 2606.21957 — **Published:** 2026-06 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** failure-claim
  **Source B:** Independently cross-referenced in "Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach" (arXiv 2302.03357) and "Guidelines for Augmentation Selection in Contrastive Learning for Time Series Classification" — same "augmentation can destroy discriminative signal / create bad positive pairs" claim, independent of ShiFT — **Publisher:** arXiv — **Published:** 2023-02 / undated — **Accessed:** 2026-09-14 — **Confidence:** medium (Source B not fetched in full, via search synthesis) — **Class:** failure-claim
  *(Two independent papers converge — meets the two-source bar, though only Source A directly read in full.)*

- **Claim:** Deterministic, structurally-defined view construction (fixed temporal shift) empirically outperformed standard stochastic augmentation on time-series linear-probe/1-NN evaluation: "Jitter+Scale" 78.84% linear-probe accuracy vs. 81.49% for the deterministic method, larger gap on 1-NN (62.62% vs. 67.17%) — augmentation choice materially affects survival of discriminative information into the embedding.
  **Source:** "Learning by Shifting..." (ShiFT) — **Publisher:** arXiv 2606.21957 — **Published:** 2026-06 — **Accessed:** 2026-09-14 — **Confidence:** high — **Class:** mitigation-pattern — **Caveat: evaluated on activity-recognition/physiological/industrial-sensor datasets (PAMAP2, WISDM2, HARTH, ECG, SLEEP, SKODA) — no financial or explicitly non-stationary market data in this paper.**

- **Claim:** Established (non-collapse-specific) architectural safeguards against contrastive collapse: SimCLR-style in-batch negatives; BYOL/SimSiam (avoid negatives via momentum encoder + asymmetric branches — SimSiam shows momentum encoder isn't even required); VICReg (explicitly regularizes embedding variance/covariance to directly prevent collapse); DINO (self-distillation teacher-student, no explicit negatives).
  **Source:** Aggregated from original architecture papers via search synthesis (SimCLR, BYOL, SimSiam, VICReg, DINO — Google/DeepMind/Meta) — **Published:** 2020-2021 (foundational) — **Accessed:** 2026-09-14 — **Confidence:** medium (well-established, internally consistent, but none individually fetched this session) — **Class:** mitigation-pattern

- **Claim:** A financial-crime-detection paper applying self-supervised contrastive sequence embeddings found combining the contrastive embedding with existing domain/behavioral-engineered features produced NO improvement, attributed to the contrastive representation "primarily recover[ing] information already captured by existing behavioral aggregates" — the learned embedding was redundant with, not additive to, engineered features.
  **Source:** "Temporal Contrastive Transformer for Financial Crime Detection: Self-Supervised Sequence Embeddings via Predictive Contrastive Coding" — **Publisher:** arXiv 2605.21490 — **Published:** 2026-05 — **Accessed:** 2026-09-14 — **Confidence:** low (via search-engine synthesized snippet only; direct PDF fetch failed to render) — **Class:** failure-claim — **single-sourced, flagged per two-source rule**

- **Claim:** Pretrained TSFMs applied to financial return forecasting are "strong practical defaults" for low-data per-asset settings, but superiority over a random-walk baseline is "not universal," with improvements "sparse and small" — framed as reducing development cost rather than solving return-prediction difficulty.
  **Source:** "Pretrained Time-Series Foundation Models for Financial Return Forecasting" — **Publisher:** arXiv 2606.27100 — **Published:** 2026-06 — **Accessed:** 2026-09-14 — **Confidence:** low-medium (generic TSFMs, not contrastive/InfoNCE specifically; search synthesis only) — **Class:** failure-claim / other — single-sourced

- **Claim:** Contrastive learning's core objective (instance discrimination) is structurally in tension with domains where "positive-like" samples are common in the training pool (e.g. anomaly detection where most instances are "normal") — conceptually adjacent to financial regime detection, where many hours are drawn from the same regime.
  **Source:** Search-synthesized summary from time-series anomaly-detection contrastive papers (arXiv 2304.07898) — **Publisher:** arXiv — **Published:** 2023-04 — **Accessed:** 2026-09-14 — **Confidence:** low (not independently verified) — **Class:** failure-claim

## Leads worth chasing
- Wang & Isola (2020) "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" — never directly fetched; likely origin of the alignment/uniformity monitoring metrics the brief asked about. Fetch directly next round.
- Financial-crime-detection paper's "redundant with engineered features" claim and the temperature-instability claim both rest on search synthesis, not verified full-text reads — worth a second fetch attempt via a different route.
- No source found discusses InfoNCE/contrastive pretraining specifically for market-regime or HMM-based regime-detection pipelines — the "regime embedding → HMM" architecture has no directly-matched precedent found. Search more narrowly for "regime detection representation learning trading" / "market state contrastive embedding HMM."
- "Guidelines for Augmentation Selection in Contrastive Learning for Time Series Classification" repeatedly on-topic but couldn't be fetched (429) — chase directly via arXiv ID search.
- "Uniformity-tolerance dilemma" (Wang & Liu) and "dimensional collapse" (Jing et al.) are distinct theoretical framings — clarify how they relate to each other and to temperature-floor/clamping as a mitigation (does clamping address dimensional collapse, uniformity-tolerance collapse, both, or neither).
- Nothing found directly ties mean-pooling specifically (vs. augmentation choice generally) to short-horizon signal loss — remains an inference from the augmentation-invariance literature, not a directly confirmed finding.

## What was looked for and could not be found
- No source specific to financial/trading contrastive-pretraining failure post-mortems with real reported before/after numbers, fully verified (closest candidate — financial-crime paper — unverified/single-sourced).
- No direct evidence, positive or negative, on contrastive pretraining ahead of an HMM regime-detector specifically — this exact pipeline shape not represented in surfaced literature.
- Could not independently verify Wang & Isola (2020) — never fetched, only inferred as likely antecedent.
- No primary-source confirmation of the temperature-instability-during-training claim — only secondary/tertiary write-ups, below the two-independent-primary-source bar.
- No source directly addresses mean-pooling (as a specific pooling choice) as a mechanism for discarding short-horizon signal in financial/non-stationary time series — this sub-question remains unaddressed by retrieved evidence.
- Budget-limited: "Guidelines for Augmentation Selection" paper, Wang & Isola 2020, and a second attempt at the financial-crime PDF were identified as leads but not pursued to completion.

## Assistant self-report
Budget used: 13 tool calls, ~duration 182s.
