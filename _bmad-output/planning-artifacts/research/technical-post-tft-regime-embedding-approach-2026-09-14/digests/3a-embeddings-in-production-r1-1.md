# Digest: 3a — Practitioner evidence: embeddings in production (round 1)

## Findings

- **Claim:** The first comprehensive empirical study of time-series foundation models (TSFMs — the closest primary-source analogue to "learned/self-supervised embeddings as state representation") in global financial markets found that off-the-shelf pretrained embeddings perform poorly in both zero-shot and fine-tuned settings, while embeddings pretrained from scratch on financial data (domain-specific self-supervised pretraining, architecturally similar to the decision brief) do achieve "substantial forecasting and economic improvements" over strong benchmark models.
  **Source:** "Re(Visiting) Time Series Foundation Models in Finance" (arXiv:2511.18578), Eghbal Rahimikia et al. — **Publisher:** arXiv (q-fin.CP) — **Published:** 2025-11-23 — **Accessed:** 2026-09-14 — **Confidence:** medium (abstract-level only; full-text tables of Sharpe/accuracy numbers not retrieved) — **Class:** performance-number (partial) / practitioner-adjacent academic

- **Claim:** A large-scale, honest ablation in the same paper reports that increasing dataset size, adding synthetic data augmentation, and hyperparameter tuning were needed to make from-scratch financial pretraining competitive — implying the learned-embedding approach is not a drop-in win and requires meaningful additional engineering investment before it beats the baseline.
  **Source:** same as above — **Publisher:** arXiv — **Published:** 2025-11 — **Accessed:** 2026-09-14 — **Confidence:** low-medium (abstract-derived summary only) — **Class:** failure-claim / practitioner-report (partial)

- **Claim:** General quant-trading practitioner commentary reports simple linear/tree models and momentum/mean-reversion rules frequently outperform deep-learning approaches in live or realistic-backtest quant trading, because deep learning (including representation learning) needs much more data than typically available in finance's sample-limited regime; some top market-making/prop firms reportedly avoid deep learning entirely.
  **Source:** search-engine synthesis drawing on "The Hidden Reality of Quant Trading" (marketcalls.in) and QuantNet forum thread "Deep Learning in Quant Trading" — **Publisher:** marketcalls.in / QuantNet — **Published:** undated — **Accessed:** 2026-09-14 — **Confidence:** low (HTTP 403 blocked direct fetch of either page; search-snippet synthesis only) — **Class:** practitioner-report (unverified)

- **Claim:** An academic study on self-supervised contrastive representation learning for trading (earnings-season trading) required task-specific fine-tuning on top of contrastive pretraining before results were usable, using a staged fine-tune-dataset-fraction design (20/50/80%) — consistent with raw contrastive embeddings needing supervised adaptation, not direct downstream use.
  **Source:** "Trading through Earnings Seasons using Self-Supervised Contrastive Representation Learning" — **Publisher:** arXiv — **Published:** 2024-09 — **Accessed:** 2026-09-14 — **Confidence:** low (abstract/snippet only; >12 months old, treat as historical) — **Class:** other / historical

- **Claim:** A 2024 survey of self-supervised pretraining for "decision foundation models" (RL-adjacent) names unresolved open problems directly relevant to production burden: no settled method for negative-sample distribution choice, no reliable augmentation design that preserves label distribution without spurious correlations, and explicit acknowledgment that non-stationary/drifting data streams break standard contrastive-loss assumptions, requiring ad hoc "causal intervention and adaptation windows" fixes.
  **Source:** "Self-supervised Pretraining for Decision Foundation Model: Formulation, Pipeline and Challenges" — **Publisher:** arXiv — **Published:** 2024-01 — **Accessed:** 2026-09-14 — **Confidence:** low-medium (snippet-level, >12mo old, historical; but a genuine challenges/limitations survey) — **Class:** failure-claim / other

- **Claim (flagged likely overclaimed, included for contrast):** Two papers surfaced with strikingly strong backtest numbers for embedding-based strategies — one claiming 13.85% annualized return and a "Sharpe Ratio of 40.93%" (internally inconsistent phrasing — a Sharpe reported as a percentage suggests garbling), and a Substack post ("Asset Embeddings," Quantitativo) claiming 0.0533%/day alpha (~14% annualized, t=4.19), 78% positive months.
  **Source:** SRLP self-supervised stock-prediction study (publisher/title unconfirmed via search synthesis); "Asset Embeddings" — **Publisher:** Quantitativo (Substack) — **Published:** undated — **Accessed:** 2026-09-14 — **Confidence:** low (single-sourced per two-source rule; launch-style framing not a 6-12mo retrospective; Sharpe figure internally suspicious) — **Class:** performance-number (unverified/single-sourced)

- **Claim:** A distinct academic line (Gabaix, Koijen, Richmond — "Asset Embeddings," AFA) uses learned embeddings for asset characteristics in portfolio-demand systems, NOT regime/state representations for RL trading — a different application of "embeddings in finance," should not be conflated with the regime-embedding-for-RL question.
  **Source:** AFA Journal of Finance listing — **Publisher:** American Finance Association — **Published:** undated listing — **Accessed:** 2026-09-14 — **Confidence:** low (title/abstract only) — **Class:** other

## Leads worth chasing
- Jonathan Kinlay's blog ("Time Series Foundation Models for Financial Markets: Kronos and the Rise of Pre-Trained Market Models," Feb 2026) — individual quant practitioner writing recently; not fetched (budget). High-value follow-up.
- Sofien Kaabar's Medium post, "Time Series Foundation Models in Finance: Useful Tool or Expensive Illusion?" (June 2026) — skeptical practitioner/CFA take; blocked by 403, retry via cache/archive.org.
- QuantNet forum thread "Deep Learning in Quant Trading" — 403 blocked; exactly the honest-negative-result source class wanted, retry with archived fetch.
- Full text of arXiv:2511.18578 — only abstract accessible; almost certainly has the concrete Sharpe/accuracy tables and retraining-cadence/compute-cost discussion. Single most promising follow-up.
- Kronos and FinCast papers (arXiv:2508.02739, arXiv:2508.19609, Aug 2025) — not read; comparing their reported gains against 2511.18578's more skeptical framing could reveal launch-optimism vs. independent-evaluation-skepticism disagreement within the field's own papers.
- No source found a production RL trading system's regime-embedding layer evaluated 6-12+ months post-deployment (the brief's exact ask). Follow-up should target named-firm engineering blogs (Two Sigma, Man Group, Jane Street, Voleon) + "regime"/"representation learning," and quant conference talks (QuantMinds, Battle of the Quants) — none of these surfaced organically this round.

## What was looked for and could not be found
- Genuine negative-result production post-mortems ("we built this, ran it live a year, it didn't help") — none found; consistent with survivorship-bias risk the brief flagged.
- Two independent sources for any single performance number — none of the specific numbers found (13.85% return, Sharpe "40.93%", 14% alpha/t=4.19) meet the bar; all single-sourced/unverified.
- Operational-cost specifics (retraining cadence, GPU/compute burden, interpretability loss) from a named team — only surfaced at the academic-survey "open problems" level, not a concrete operational account.
- QuantNet forum and Kaabar's Medium post — 403'd, not read in full.
- No source specifically addressed learned embeddings as a pretraining step feeding an HMM regime-clustering stage ahead of RL execution (the brief's precise architecture) — all relevant sources address TSFMs for direct forecasting or contrastive learning for direct trading signals, adjacent but not identical.

## Assistant self-report
Budget used: 11 tool calls, ~duration 148s.
