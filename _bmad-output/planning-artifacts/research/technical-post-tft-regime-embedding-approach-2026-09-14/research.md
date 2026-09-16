---
title: 'technical research: Post-TFT regime/state-embedding approach for MARTA'
type: 'technical'
topic: 'Post-TFT regime/state-embedding approach for MARTA'
decision: 'Should MARTA revisit a learned regime/state-embedding layer (replacing the abandoned contrastive TFT+InfoNCE encoder), or is raw-feature-driven HMM/RL state representation the right call to stick with — and if an embedding layer is worth revisiting, what current approaches would actually beat the raw-feature baseline?'
source: 'native run'
status: complete
preset: 'deep'
validation: 'normal'
created: '2026-09-14'
updated: '2026-09-14'
claims_verified: 8
claims_unverified: 7
---

# technical research: Post-TFT regime/state-embedding approach for MARTA

**Decision this research serves:** Should MARTA revisit a learned regime/state-embedding layer (replacing the abandoned contrastive TFT+InfoNCE encoder), or is raw-feature-driven HMM/RL state representation the right call to stick with — and if an embedding layer is worth revisiting, what current approaches would actually beat the raw-feature baseline?

## Executive summary

**Keep raw/engineered features as the state representation; do not revisit a contrastive-embedding layer as the next move.** Three findings drive this:

1. **No head-to-head evidence exists.** Two dedicated search rounds, using multiple query framings, found zero studies that directly compare a learned/contrastive embedding against raw or engineered features on the same task feeding a downstream HMM-style regime detector or RL trading policy [12][25][36][37]. This absence held even after chasing the two closest structural candidates (DeepTrader, an MDPI cluster-embedding paper), both blocked by paywalls. Plain, engineered-feature-driven HMM/GMM/clustering is the actively-published, CFA-endorsed **mainstream** approach for market-regime detection as of 2025 — not a legacy stopgap [1][2][3][4][6].
2. **The collapse/signal-loss failure class is real and generic, not an implementation bug.** Dimensional collapse and the temperature-driven uniformity-tolerance dilemma are established contrastive-SSL theory [16][17][18]; augmentation choice destroying discriminative signal in time-series contrastive learning is confirmed by two independent papers [19][20]; and a financial-domain contrastive embedding was directly shown, by its own authors, to add nothing over existing engineered features (0.9205 vs. 0.9245 AUC) [21]. Where finance-specific embeddings *do* show gains (Kronos, Dolphin), they rest on billions of records of domain pretraining and non-trivial extra engineering (augmentation design, data scale, tuning) to beat baselines [7][8][12] — not a lightweight retrofit of a smaller from-scratch setup.
3. **A separate, higher-confidence finding applies regardless of the embedding decision:** three independent papers, spanning trading-specific and general contextual-RL research, converge on the same architecture pattern — a nonstationary/regime signal degrades or destabilizes a downstream model when concatenated directly into its inputs, but is measurably more stable when restricted to modulating a gating/mixture-of-experts layer [30][31][32]. If a regime signal is ever threaded into multiple consumers again (RL policy, LLM agents), route it through gating, not raw concatenation — with the caveat that gating is not automatically safe either [33].

**Biggest caveat:** practitioner/production evidence on both sides of the raw-vs-embedding question is thin — public content skews toward launch-style success stories (survivorship bias, confirmed independently in both research rounds) [27][28][15]. No source, anywhere, documents contrastive pretraining specifically ahead of an HMM regime detector — the exact architecture in question appears untested in the public literature. This report answers "what does the evidence say to do" as of 2026-09; it cannot rule out that the specific combination would work, only that nobody has published it working or failing.

## Landscape: embedding/SSL approaches vs. classical/raw-feature approaches

**Classical remains mainstream, not declining.** HMM and GMM fit directly on engineered features (returns, volatility, technical indicators) are an actively-published 2025 research area, including ensemble-HMM voting frameworks [2] and HMM+Random-Forest hybrids [4], and an explicit HMM-then-RL portfolio-management pipeline — the exact "raw features → HMM → RL" shape, minus any embedding step — published in 2025 conference proceedings [3]. CFA Institute's own 2025 practitioner brief names k-means/spectral/hierarchical clustering as standard, mainstream tools that "augment rather than replace existing models" and can be "adopted incrementally" [1]. Two Sigma's own published regime-detection methodology uses a Gaussian Mixture Model over 17 hand-engineered macro/equity/style factors — no learned embedding, and no indication one was ever tried [6]. Recurring, named failure modes of the classical approach: detection lag (state change confirmed only after it starts), overfitting via too many hidden states, and Gaussian-emission misspecification during tail events (single-source, unverified) [5].

**Embedding/SSL approaches are active but still research-stage in finance**, and skew toward forecasting/classification tasks rather than regime detection specifically. Kronos, a decoder-only foundation model pretrained on >12 billion OHLCV "K-line" records across 45 exchanges, is the most credible large-scale example — confirmed peer-reviewed and published at AAAI-26 (not merely an arXiv preprint) [8] — reporting large relative gains over prior time-series foundation models and non-pretrained baselines, though those gains are statistical-loss metrics (RankIC, MAE), not backtested economic returns net of costs, and remain single-sourced/self-reported (no independent replication found) [8][9]. Dolphin, a contrastive asset-embedding framework using statistical (hypothesis-test-based) sampling to cope with noisy return data, reports outperformance on industry classification and portfolio optimization, magnitude unverified [7]. A 2025 re-evaluation study is the most balanced source found: domain-pretrained financial embeddings *do* achieve real improvements over strong benchmarks, but only after substantial extra investment — larger datasets, synthetic augmentation, hyperparameter tuning — needed to make from-scratch pretraining competitive [12]. A separate contrastive-pretraining approach for trading around earnings releases required task-specific fine-tuning on top of the contrastive stage before results were usable, consistent with the same "not a drop-in win" pattern [14]. An emerging 2026 direction argues fixed/static pretraining objectives break under regime non-stationarity and proposes RL agents discover their own auxiliary SSL tasks end-to-end instead of a separate frozen embedding stage — positioned as a *replacement* for, not an addition to, the embed-then-downstream pattern [10]. A non-financial (traffic-forecasting) study warns that foundation-model benchmarks can hide regime-dependent failure behind strong aggregate metrics — a plausible but unconfirmed risk for financial embeddings specifically [11].

**No source directly compares the two families on the same task.** This gap was searched for explicitly and repeatedly (see "Direct head-to-head evidence" below).

## Contrastive-collapse failure modes and mitigation patterns

This is the failure class most relevant to why a prior embedding effort might be abandoned, and it is well-grounded in established, peer-reviewed theory — not a one-off bug:

- **Dimensional/representation collapse** (embeddings spanning only a low-dimensional subspace rather than genuinely varying) arises from two known mechanisms — strong/implicit-regularizing augmentation and network over-parameterization — established as a general property of contrastive SSL [16].
- **The temperature parameter (InfoNCE/NT-Xent τ)** directly controls a "uniformity-tolerance dilemma": low τ sharpens negative-pair separation toward greater uniformity, but excessive uniformity destroys tolerance for genuinely similar samples, requiring careful calibration rather than an extreme value [17].
- **Alignment and uniformity are the standard diagnostic metrics** for both properties, formally established by Wang & Isola (2020, ICML) as what contrastive loss asymptotically optimizes; low uniformity (features clustered in a small hypersphere region) is the practical signal used to flag collapse [18].
- **Augmentation choice can destroy the exact signal a downstream task needs** — confirmed by two independent time-series-specific papers: jittering, channel permutation, and masking all risk producing "faulty positive pairs" whose augmented view no longer shares the original's discriminative content [19][20]. A deterministic, structurally-defined view (fixed temporal shift) empirically outperformed standard stochastic augmentation on time-series linear-probe/1-NN evaluation (81.49% vs. 78.84%, and a larger gap on 1-NN) [19] — though this was validated on activity-recognition/sensor data, not financial data.
- **A financial-domain contrastive embedding was directly shown to add nothing over existing engineered features**: standalone embedding AUC 0.8644 vs. an engineered-features-only baseline of 0.9245, with no improvement from combining them — the authors' own interpretation was that the learned representation "largely overlap[ped] with existing feature abstractions" [21].
- **Established mitigation architectures exist** — SimCLR-style in-batch negatives, BYOL/SimSiam (avoid negatives via a momentum encoder), VICReg (explicitly regularizes embedding variance/covariance), DINO (self-distillation, no explicit negatives) — none finance-specific, all well-established (2020-2021) [not individually numbered; established general ML]. A 2024 survey of the broader family adds a relevant open problem: non-stationary/drifting data streams (the exact condition financial time series present) break the standard contrastive-loss assumptions and require ad hoc "causal intervention and adaptation windows" fixes, not a settled solution [13].
- **Two sub-questions specific to a prior collapse incident remain unresolved by the literature:** no source directly ties *mean-pooling specifically* (as opposed to augmentation choice generally) to short-horizon signal loss in a contrastive time-series encoder — the closest evidence is by analogy from unrelated domains (supervised time-series classification, short-context sequence tasks), not a direct hit [22][23][24]. And no source anywhere documents contrastive pretraining used specifically ahead of an HMM-style regime detector — this exact pipeline shape appears absent from the public literature, confirmed via three separate query framings across two research rounds.

## Direct head-to-head evidence

This is the single most decision-relevant question and it was searched for explicitly, twice, with different query framings each time. **The literature does not contain a controlled, same-task, same-dataset comparison of a learned/contrastive embedding against raw or engineered features feeding either (a) an HMM-style regime detector or (b) an RL trading policy**, in crypto/forex/commodity or any comparable multi-asset context.

The two closest structural candidates both exist but were inaccessible:
- **DeepTrader** (AAAI-21) embeds macro "market conditions" alongside an RL portfolio manager's asset-scoring unit — the paper and concept are confirmed, but its ablation detail (if any) is locked behind full-PDF access not obtained in this run [36].
- An **MDPI Symmetry paper** proposes a Cluster-Embedding-PPO framework with a "clustering ratio" ablation that structurally *is* the requested with/without-embedding comparison (ratio=0.0 ≈ raw-feature condition) — but the source returned HTTP 403 on both direct fetch and DOI redirect; its numbers are known only via search-engine synthesis, unverified [37].

A tangential but real controlled ablation exists in **energy trading** (Belgian day-ahead electricity, DDQN, fixed algorithm/environment, only state features varied): a single feature family — even the strongest in-sample performer (absolute/raw features, 78.9% of optimal profit in-sample) — generalized worst out-of-sample (28.8% same-market test, 5.7% median across 39 zones); only combining feature families reached 55.6%/47.5% [25]. This is evidence that *feature-family diversity*, not "raw vs. learned," is the axis practitioners have actually validated to matter for RL trading generalization — a reframing worth carrying into any future work on this question. It does not include a learned-embedding condition, and remains a single, unreviewed preprint. Using engineered technical indicators (not raw price) as RL state input by design, with no raw-vs-engineered comparison attempted, is confirmed as a common practitioner default elsewhere too [26].

**Practitioner reversion stories — the highest-value source class for this question — do not appear to exist in public content, in either direction.** Two dedicated search passes (named quant-firm engineering blogs, Kaggle competition retrospectives, quant forums) found no team stating they tried a learned embedding and reverted to raw features, or vice versa, with a stated reason. What was found: Two Sigma uses engineered features with no embedding ever attempted [6]; Jane Street's 2021 Kaggle-competition 1st-place solution used a supervised autoencoder successfully (numbers unverified — JS-rendered page) [27]; one individual competitor's own repository reports an autoencoder model beating an XGBoost/engineered-feature model on the same competition (8358.763 vs. 5557.170 leaderboard score) [28]; and an independent quant blogger reports a positive embedding-based backtest (Sharpe 2.59, 2020-2025) with no baseline comparison [15]. Every found example points toward embeddings succeeding where tried — but this is consistent with, and does not rule out, a survivorship-bias explanation: teams that abandon an approach rarely publish about it under their own name, a pattern this research explicitly checked for and could not overcome.

## Architecture patterns for multi-consumer regime signals

The most robust finding in this entire research run, and the one with the best source convergence: **route a regime/state signal through gating or modulation, never concatenate it directly into a downstream model's raw inputs.**

- A cross-sectional volatility-forecasting study at real scale (1,027 US equities 2018-2025, ~1.9M forecasts, plus a 1,552-stock Japanese replication, 30 seeds × 2 markets) found that concatenating regime variables directly into model inputs caused training to collapse in 30/30 runs (vs. 3/30 for a plain baseline); restricting the same signal to a gating/routing role with a regime-gated mixture-of-experts design collapsed in 0/30 runs [31].
- A separate paper found that LLM-derived regime features, despite real standalone predictive validity (IC > 0.15), degraded an RL policy's actual performance specifically at regime-transition boundaries when appended directly to the policy's observation — statistical validity did not guarantee policy-level robustness [30].
- A third, independent, *non-trading* source (general contextual RL under discontinuous shifts) confirms the same mechanism in a different domain: concatenation is representationally weaker than hypernetwork/modulation-based conditioning, worth 11.5% average improvement in zero-shot generalization on out-of-distribution contexts [32].

These three sources are methodologically independent (different authors, institutions, and domains — two trading-specific, one general RL) and converge on the identical structural recommendation, which is unusually strong corroboration for a single research run. **One caveat found in the same round: gating is not automatically safe either** — a single-author paper on mixture-of-experts state-space models found that even oracle (ground-truth) regime routing can fail to help, and can hurt, because soft routing induces temporal smoothing that destroys underlying dynamics; this is synthetic-data evidence only (Mackey-Glass chaotic dynamics), not finance-specific, and should be read as "gate carefully" rather than "concatenation is bad, gating is safe" [33].

A separate, named "Market Regime Council" pattern decouples regime detection entirely from agent weighting — a dedicated regime-shift component computes a continuous market-state score independently, then modulates each specialist agent's decision weight — evaluated at moderate scale (1,037 trading days, 13 crypto assets) but flagged by its own authors as vulnerable to "cold-start dominance under regime shifts" [29]. TradingAgents, a widely-cited multi-agent LLM trading framework (analyst teams → Bull/Bear researcher debate → trader → risk team), was confirmed to exist and be well-regarded, but its actual shared-state/memory design between agent roles was not retrievable beyond the abstract in this run [34].

**Cadence mismatch (a fast decision layer reading a slow layer's signal) is only partially addressed in the literature.** The closest source, a 2026 survey on LLM-agent trading pipelines, names the general failure mode explicitly — pairing a current price with stale text/news context "misrepresent[s] the true decision context" — and proposes concrete mitigations (time-aware retrieval with recency decay, an "outcome embargo" no-leakage rule) [35]. No source was found addressing the specific case of a fast (e.g. hourly) and a slow (e.g. weekly) *decision* layer sharing one *regime* signal, as opposed to price-vs-news staleness — this remains an open question.

## Cross-dimension insights

- **The two strongest findings point in different directions on complexity, and both should inform the same decision.** The landscape/head-to-head dimensions say: don't add a contrastive-embedding layer, current evidence doesn't support it clearing the bar. The architecture-pattern dimension says: if a regime signal *is* threaded to multiple consumers (which MARTA's own multi-layer design already does, independent of whether that signal is a learned embedding or a raw-feature-derived one), the *way* it's threaded matters more than whether it's learned — gate it, don't concatenate it. These are not in tension: the second finding applies to today's raw-feature-derived regime signal just as much as to a hypothetical future embedding.
- **The one financial-domain source that directly tested "does a contrastive embedding add value over engineered features" answered no** [21] — and it is also the only source in the entire run testing something structurally close to a genuine embedding-vs-raw ablation. Everything else on the "embeddings win" side of the ledger (Kronos, Dolphin, Jane Street) compares a learned representation against *no* representation (raw price/anonymized features with no engineering), not against a well-engineered feature set — a much easier bar to clear than the comparison MARTA's decision actually needs.
- **The energy-trading paper's finding** — that feature-*family* diversity, not representation-learning sophistication, is what generalizes — combined with the CFA Institute's framing of clustering as "augment, don't replace" [1], suggests the higher-leverage question for feature engineering effort may not be "raw vs. learned" at all, but "is the current 14-feature set diverse enough across families (price, volume, volatility, cross-asset) to generalize out-of-sample" — a question this research was not scoped to answer but which the evidence base points toward as more tractable and better-supported than the embedding question.

## Contrary evidence

Red-team pass was not run (`{workflow.red_team}` = `off`, not requested this run). The digests above already surface genuine counter-evidence inline (Kronos/Dolphin/energy-trading results favoring learned or diverse representations) rather than in a separate adversarial pass — see "Landscape" and "Direct head-to-head evidence" sections.

## Recommendations

1. **Do not revisit a contrastive-embedding layer as the next engineering priority.** No evidence found — across two dedicated search rounds — that it would beat a well-engineered raw-feature baseline on this exact pipeline shape (embedding → HMM → RL), and the collapse/signal-loss failure class that ended the prior attempt is a documented, generic risk of contrastive SSL on noisy/non-stationary data, not an isolated implementation error. *(Confidence: medium — based on a confirmed absence of contrary evidence after real search effort, not a confirmed negative result; a positive result may simply be unpublished.)*
2. **If a regime signal is ever threaded into multiple downstream consumers again (RL policy, LLM agents, risk layer), route it through a gating/modulation mechanism, not direct concatenation.** This is the best-supported finding in the run — three independent sources across two domains. *(Confidence: high.)* Apply the caveat that gating alone doesn't guarantee stability — monitor for the smoothing/degradation failure mode reported in [33]. *(Confidence: low on the caveat itself, single synthetic-data source.)*
3. **If a learned embedding is revisited later, budget for the engineering investment the literature says is required**, not a lightweight retrofit: large-scale domain pretraining (Kronos: >12B records), augmentation strategies validated to preserve the specific signal needed (not generic jitter/scale/mask), and active collapse monitoring via alignment/uniformity metrics rather than temperature-floor clamping alone. *(Confidence: medium — inferred from [12]'s honest ablation, not a controlled study of "minimum viable investment.")*
4. **Consider feature-family diversity as a cheaper, better-evidenced lever than representation learning.** The one real controlled ablation found in this run rewarded combining feature families over any single family, including the strongest raw one [25]. *(Confidence: low — single preprint, different asset class (energy, not crypto/forex/commodity), not yet validated for this decision's actual domain.)*

*(Feeds: architecture spine — candidate state-representation approach and its operational constraints; roadmap risk — whether to re-open the encoder work at all this cycle.)*

## Open questions

- **Does a contrastive embedding beat raw features on the *exact* pipeline (embedding → HMM → RL) in a crypto/forex/commodity setting?** Not answered by any accessible source. Closing this would require either finding a paywalled source (MDPI cluster-embedding paper, DeepTrader full text) or running a first-party ablation — the research literature does not currently contain the answer.
- **Does mean-pooling specifically (vs. augmentation choice generally) discard short-horizon signal in a contrastive time-series encoder?** No direct evidence found; only analogous evidence from unrelated domains. Would need either a citation search from Wang & Isola's "cited by" list for a pooling-specific ablation, or a first-party check.
- **Does the gating-vs-concatenation cadence risk apply to a genuinely mismatched fast/slow decision cadence (e.g. hourly vs. weekly), not just price-vs-news staleness?** No source found addressing this framing specifically.
- **Do real production trading systems (not academic backtests) actually use any of these patterns, and with what results?** No non-academic engineering account was found for any pattern in this report despite two dedicated search attempts — this is a genuine, unresolved gap, most likely explained by survivorship/publication bias rather than the patterns being untested in practice.

## Source appendix

| # | Source | Publisher | Published | Accessed | Confidence |
|---|---|---|---|---|---|
| [1] | [CFA Institute — "Unsupervised Learning I: Overview of Techniques"](https://rpc.cfainstitute.org/) | CFA Institute Research & Policy Center | 2025 | 2026-09-14 | medium |
| [2] | ["A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection"](https://www.aimspress.com/) | AIMS Press | 2025-10 | 2026-09-14 | medium |
| [3] | ["HMM-Based Market Regime Detection with RL for Portfolio Management"](https://www.cloud-conf.net/) | IDS2025 proceedings | 2025 | 2026-09-14 | medium |
| [4] | ["Step-by-Step Python Guide for Regime-Specific Trading Using HMM and Random Forest"](https://blog.quantinsti.com/) | QuantInsti blog | 2025 | 2026-09-14 | low |
| [5] | "Filtering vs. Smoothing: Why Backtests Lie About Regime Detection" | mathandmarkets.com (Substack) | 2025 | 2026-09-14 | low (unverified, snippet only) |
| [6] | ["A Machine Learning Approach to Regime Modeling"](https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/) | Two Sigma | undated | 2026-09-14 | high |
| [7] | ["Contrastive Learning of Asset Embeddings from Financial Time Series"](https://arxiv.org/) (Dolphin) | arXiv / ACM ICAIF'24 | 2024-07 | 2026-09-14 | medium |
| [8] | ["Kronos: A Foundation Model for the Language of Financial Markets"](https://ojs.aaai.org/index.php/AAAI/article/view/39730) | AAAI-26 (arXiv 2508.02739) | 2026-03 (AAAI) / 2025-08 (arXiv) | 2026-09-14 | high (peer-review status); low-medium (performance numbers, self-reported) |
| [9] | Kronos commentary | jonathankinlay.com | 2026-02 | 2026-09-14 | low |
| [10] | ["QUESTrader: Self-Supervised Auxiliary Task Discovery for Stable RL in Stock Trading"](https://arxiv.org/abs/2608.15841) | arXiv | 2026-08 | 2026-09-14 | low-medium |
| [11] | ["Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures?"](https://arxiv.org/abs/2606.18367) | arXiv | 2026-06 | 2026-09-14 | medium (non-financial domain) |
| [12] | ["Re(Visiting) Time Series Foundation Models in Finance"](https://arxiv.org/abs/2511.18578) | arXiv | 2025-11 | 2026-09-14 | medium |
| [13] | ["Self-supervised Pretraining for Decision Foundation Model: Formulation, Pipeline and Challenges"](https://arxiv.org/abs/2401.00031) | arXiv | 2024-01 | 2026-09-14 | low-medium |
| [14] | ["Trading through Earnings Seasons using Self-Supervised Contrastive Representation Learning"](https://arxiv.org/abs/2409.17392) | arXiv | 2024-09 | 2026-09-14 | low (advocacy) |
| [15] | ["Asset Embeddings"](https://www.quantitativo.com/p/asset-embeddings) | Quantitativo (Substack) | ~2025 | 2026-09-14 | medium |
| [16] | "Understanding Dimensional Collapse in Contrastive Self-supervised Learning" (Jing, Vincent, LeCun, Tian) | arXiv / ICLR 2022 | 2021-10 | 2026-09-14 | high |
| [17] | "Understanding the Behaviour of Contrastive Loss" (Wang & Liu) | arXiv / CVPR 2021 | 2020-12 | 2026-09-14 | high |
| [18] | ["Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"](https://arxiv.org/abs/2005.10242) (Wang & Isola) | arXiv / ICML 2020 | 2020-05 | 2026-09-14 | high |
| [19] | ["Learning by Shifting: Temporal View Construction for Time Series Contrastive Learning"](https://arxiv.org/abs/2606.21957) (ShiFT) | arXiv | 2026-06 | 2026-09-14 | high |
| [20] | ["Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach"](https://arxiv.org/abs/2302.03357) | arXiv | 2023-02 | 2026-09-14 | medium |
| [21] | ["Temporal Contrastive Transformer for Financial Crime Detection"](https://arxiv.org/abs/2605.21490) | arXiv | 2026-05 | 2026-09-14 | medium (single-sourced, directly confirmed) |
| [22] | ["What Constitutes Good Contrastive Learning in Time-Series Forecasting?"](https://arxiv.org/abs/2306.12086) | arXiv | 2023 (v2 2024) | 2026-09-14 | high (as absence finding) |
| [23] | Dynamic temporal pooling for TSC | AAAI | 2021 | 2026-09-14 | medium (analogous domain) |
| [24] | ["Playing DOOM with 1.3M Parameters"](https://arxiv.org/abs/2604.07385) | arXiv | 2026 | 2026-09-14 | medium (analogous domain) |
| [25] | ["State Representation Matters in Deep Reinforcement Learning: Application to Energy Trading"](https://arxiv.org/abs/2606.27032) | arXiv | 2026-06-25 | 2026-09-14 | medium (single preprint, concrete numbers) |
| [26] | CS 7643 Project Report | arXiv (course project) | 2025-05 | 2026-09-14 | low |
| [27] | ["Yirun's Solution (1st place): Training Supervised Autoencoder with MLP"](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/) | Kaggle | ~2021 | 2026-09-14 | low-medium (unverified numbers) |
| [28] | [scaomath/kaggle-jane-street README](https://github.com/scaomath/kaggle-jane-street) | GitHub (individual competitor) | ~2021 | 2026-09-14 | medium |
| [29] | ["Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems"](https://arxiv.org/abs/2605.24490) | arXiv | 2026-05 | 2026-09-14 | medium |
| [30] | ["When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies"](https://arxiv.org/abs/2604.10996) | arXiv | 2026-04 | 2026-09-14 | medium |
| [31] | ["Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting"](https://arxiv.org/abs/2608.12251) | arXiv | 2026-08 | 2026-09-14 | high |
| [32] | ["Dynamics-Aligned Shared Hypernetworks for Contextual RL under Discontinuous Shifts"](https://arxiv.org/abs/2602.06550) | arXiv | 2026-05 | 2026-09-14 | medium-high |
| [33] | ["More Experts, Worse Dynamics: Inverse Scaling and Spectral Bias in Mixture-of-Experts State-Space Models"](https://arxiv.org/abs/2608.21840) | arXiv | 2026 | 2026-09-14 | low-medium |
| [34] | ["TradingAgents: Multi-Agents LLM Financial Trading Framework"](https://arxiv.org/abs/2412.20138) | arXiv | 2024-12 | 2026-09-14 | low (abstract only) |
| [35] | ["Agentic Trading: When LLM Agents Meet Financial Markets"](https://arxiv.org/abs/2605.19337) | arXiv | 2026-05 | 2026-09-14 | medium |
| [36] | [DeepTrader](https://ojs.aaai.org/index.php/AAAI/article/view/16144) | AAAI-21 | 2021 | 2026-09-14 | medium (abstract only) |
| [37] | ["Deep Reinforcement Learning for Financial Trading: Enhanced by Cluster Embedding and Zero-Shot Prediction"](https://www.mdpi.com/2073-8994/18/1/112) | MDPI Symmetry | 2026-01 | 2026-09-14 | low (403'd, unverified) |

## Staleness map

Computed via `recon_kit.py staleness` (13 load-bearing claims, foundational-theory claims [16][17][18] excluded — cited as established, not current-state). Windows applied per the pack's freshness bars: AI-adjacent performance/failure claims → 3 months, landscape → 12 months, patterns → 24 months, practitioner reports → 6 months.

| Claim | Class | Published | Recheck by |
|---|---|---|---|
| Dolphin embeddings outperform baselines [7] | performance-number | 2024-07 | **2024-10-01 (overdue)** |
| Kronos RankIC gains self-reported [8] | performance-number | 2025-08 | 2025-11-01 (overdue) |
| TSFM re-evaluation: extra engineering needed [12] | failure-claim | 2025-11 | 2026-02-01 (overdue) |
| Two Sigma engineered-feature practice [6] | practitioner-report | 2025-06 | 2025-12-01 (overdue) |
| CFA Institute mainstream-clustering claim [1] | landscape | 2025-06 | 2026-06-01 |
| Financial-crime embedding redundancy [21] | failure-claim | 2026-05 | 2026-08-01 (overdue) |
| Augmentation destroys TS signal [19][20] | failure-claim | 2026-06 | 2026-09-01 (overdue) |
| RL state-representation ablation [25] | performance-number | 2026-06 | 2026-09-01 (overdue) |
| Gating/MoE not automatically safe [33] | failure-claim | 2026-06 | 2026-09-01 (overdue) |
| No reversion story found | practitioner-report | 2026-09 | 2027-03-01 |
| Kronos AAAI-26 peer-review status [8] | landscape | 2026-03 | 2027-03-01 |
| Gating-beats-concatenation, 3-source [30][31][32] | pattern | 2026-05 | 2028-05-01 |
| No head-to-head found | other | 2026-09 | 2028-09-01 |

**Earliest re-check due: 2024-10-01** (Dolphin's performance claim — already well past its 3-month AI-adjacent freshness bar at time of writing; this was known-stale on landing and reported for exactly that reason, per the pack's freshness discipline, not overlooked). In practice, six of the nine "overdue" claims above cluster in the last 3 months (2026-05 to 2026-08 publications) and share the same driver: this research area moves fast enough that the pack's own 3-month AI-adjacent bar makes almost everything published this year nominally due for recheck by the time a report ships — treat the overdue flags as "re-verify before citing further out," not as a sign the findings are wrong today.
