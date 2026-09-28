# Digest: Direct embedding vs raw-feature head-to-head (round 2)

## Findings

- **Claim:** DeepTrader (AAAI-21) is a deep RL portfolio manager embedding macro "market conditions" as an auxiliary signal alongside an asset-scoring unit, explicitly motivated by prior RL portfolio models ignoring market regime. Abstract/landing page does not itself state a with-vs-without-embedding ablation result — that detail, if present, is inside the full PDF (not accessed).
  **Source:** ojs.aaai.org/index.php/AAAI/article/view/16144 — **Publisher:** AAAI (Proceedings of AAAI-21) — **Published:** 2021 — **Accessed:** 2026-09-14 — **Confidence:** medium (confirms paper/concept, not ablation numbers) — **Class:** landscape

- **Claim:** An MDPI Symmetry paper, "Deep Reinforcement Learning for Financial Trading: Enhanced by Cluster Embedding and Zero-Shot Prediction" (Symmetry 18(1):112), proposes a Cluster-Embedding-PPO framework with a "clustering ratio" ablation where ratio=0.0 (all channels in one cluster) is effectively the no-embedding/raw-feature condition vs. full clustered-embedding — the one paper found in this round structurally containing the requested with/without-embedding ablation. Full numeric results for ratio=0.0 vs full could NOT be retrieved (403 on direct fetch and DOI redirect); only existence/design confirmed via search synthesis. Separately-reported headline numbers for the full framework: short-term prediction MAE reduced ~20.09%/MSE ~30.12% vs baseline forecasting, zero-shot MAE/MSE reduced ~21.56%/~31.71%, cumulative return 137.94% on S&P 500.
  **Source:** mdpi.com/2073-8994/18/1/112 — **Publisher:** MDPI, Symmetry — **Published:** 2026-01-07 per search snippet — **Accessed:** 2026-09-14 — **Confidence:** low-medium (ablation existence confirmed by two independent search snippets; exact numeric table NOT independently verified, primary text inaccessible) — **Class:** landscape / performance-number (unverified magnitude)

- **Claim:** No paper or writeup found that runs a controlled, same-task/same-dataset comparison of a contrastive/self-supervised learned embedding against raw/engineered features specifically feeding a downstream HMM-style regime-detection/clustering stage. Targeted query "contrastive pretraining HMM regime detection ablation" returned only unrelated domains (medical electrograms, object detection, speech, heliophysics) — zero on-topic hits.
  **Source:** WebSearch result set — **Accessed:** 2026-09-14 — **Confidence:** high (as an absence finding) — **Class:** other (absence-of-evidence)

- **Claim:** A genuine, accessible controlled ablation of RL state representations exists, but in energy trading (not crypto/forex/commodity) and comparing engineered feature families only, not a learned/contrastive embedding. "State Representation Matters in Deep RL: Application to Energy Trading" (arXiv 2606.27032) holds environment/action/reward/network/training fixed for a Double DQN on Belgian day-ahead electricity arbitrage, varying only state features: absolute/raw price+calendar alone = 28.8% test / 5.7% cross-zone median; absolute+relative = 49.9%/39.8%; absolute+relative+forecast = 55.6%/47.5%. Real evidence state-representation choice materially moves RL trading performance in a controlled setting — but engineered-vs-engineered, not embedding-vs-raw, and an unreviewed single-version preprint.
  **Source:** arxiv.org/abs/2606.27032 — **Publisher:** arXiv preprint — **Published:** 2026-06-25 — **Accessed:** 2026-09-14 — **Confidence:** medium (concrete numbers directly retrieved) but single-sourced — **Class:** performance-number

- **Claim:** Kronos ("A Foundation Model for the Language of Financial Markets," arXiv 2508.02739) IS peer-reviewed and published — official AAAI proceedings record: Vol. 40 No. 30, "AAAI-26 Technical Tracks 30," DOI 10.1609/aaai.v40i30.39730, listed publication date 2026-03-14, matching the arXiv paper. arXiv abstract page shows only v1 (2025-08-02) with no explicit "accepted at AAAI" self-annotation — acceptance evidence rests on the AAAI proceedings listing, confirmed via two independent routes (direct OJS fetch + DOI search).
  **Source:** ojs.aaai.org/index.php/AAAI/article/view/39730 — **Publisher:** AAAI — **Published:** 2026-03-14 — **Accessed:** 2026-09-14 — **Confidence:** high (two independent confirmations) — **Class:** landscape

- **Claim (unverified, flagged low confidence):** A search-engine synthesis asserted "raw feature vectors resulted in 93% of data concentrated in a single cluster with low discriminative power" vs. better separation from embedding vectors, in some unnamed clustering study — could not identify or confirm the source paper. NOT attributable to a specific readable primary source.
  **Source:** WebSearch synthesis, unattributed — **Accessed:** 2026-09-14 — **Confidence:** very low — **Class:** other — **do not use as evidence, unverified belief only**

## Leads worth chasing
- Obtain full-text of MDPI Symmetry cluster-embedding paper (403'd) via alternate access — the clustering-ratio=0.0 ablation table is the single closest candidate to a real head-to-head; numbers not yet retrieved.
- Pull full DeepTrader AAAI-21 PDF to check for a market-condition-embedding-on vs -off ablation with Sharpe/return/drawdown deltas.
- arXiv 2606.27032 (energy trading) is a good template for the kind of controlled ablation needed — search for a crypto/forex-specific analogue by the same methodology, including a contrastive/embedding condition.
- Check DeepTrader's GitHub repo (CMACH508/DeepTrader) for ablation logs/readme tables not present in the paywalled paper text.

## What was looked for and could not be found
- No study/paper/practitioner writeup running a direct, same-task/same-dataset, numerically-reported comparison of a learned/contrastive/self-supervised embedding against raw/engineered features feeding either (a) a downstream HMM-style regime detector, or (b) an RL trading policy, in crypto/forex/commodity or comparable multi-asset context. "Contrastive pretraining → HMM regime detection" as a paired ablation returned zero on-topic results across multiple phrasings — appears to be a genuine literature gap, not a search-phrasing failure.
- DeepTrader and the MDPI paper both exist and are on-topic in spirit but neither yielded a directly-read, numerically-confirmed with/without-embedding contrast — DeepTrader's ablation locked behind full-PDF access not obtained; MDPI blocked by 403 on both primary URL and DOI redirect, numbers only via aggregator synthesis.
- Consistent with round 1's finding: landscape material exists on both sides but no direct head-to-head found; if one exists, it wasn't surfaced within this round's budget either.
- Kronos peer-review status resolved with high confidence: AAAI-26 accepted/published, not merely an arXiv preprint.

## Assistant self-report
Budget used: 16 tool calls, ~duration 200s.
