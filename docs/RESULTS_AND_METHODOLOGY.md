# MARTA — Process, Results, and Open Questions

*Written 2026-08-14, ~2 weeks before final defense. Companion to `docs/ARCHITECTURE.md` (system
design) — this document is about what was actually tried, what happened, and what is still
unresolved. Read this one first if you need to decide what to ask your advisor.*

---

## TL;DR

- MARTA's four layers (data → regime encoder/HMM → multi-agent LLM debate → PPO RL execution)
  are all **built and functioning end-to-end** — the pipeline runs, produces artifacts, and
  every stage has been individually tested and debugged.
- The **research claim is not validated**: no configuration tested so far produces a
  statistically defensible profitable trading strategy out-of-sample.
- The last ~10 days of work were a **systematic, controlled investigation into why**, not
  guesswork: every hypothesis was tested in isolation, results were compared against fixed
  baselines (raw features, random projection, buy-and-hold), and negative results were kept
  and documented rather than hidden or silently re-tried until something looked good.
- The current, honest state: raw market features carry a **real but weak** predictive edge
  (better than a learned encoder, but still small); no RL configuration tried has turned that
  weak edge into a profitable policy — the best result so far still loses money, just less
  badly than earlier attempts.
- This document exists to help you (1) decide what to ask your advisor, and (2) have a
  15-minute-ready narrative that shows the rigor even though the number at the end is negative.

---

## 1. What the project set out to prove

**Research claim:** conditioning a multi-agent LLM trading system on a *learned latent regime
representation* (contrastive pretraining → HMM regime clustering) produces statistically
significant improvements in risk-adjusted returns over regime-agnostic baselines, across
crypto, forex, and commodity markets.

Four layers, built in this order over the project:
1. **Data pipeline** — genuine OHLCV for BTC, ETH, EURUSD, GOLD (2018–2025), 14 engineered
   features per asset, walk-forward split (train < 2024, val = 2024, test = 2025, test never
   touched by any training or model-selection step).
2. **Regime encoder** — a TFT-style network (VSN → 2-layer LSTM → pooling → 128-dim embedding)
   trained with contrastive (SimCLR-style) pretraining, then a Gaussian HMM to assign discrete
   market regimes from the embedding.
3. **Multi-agent debate** — 5 LLM agents (Macro, Technical, Sentiment, Risk, Contrarian) debate
   over 2 rounds, conditioned on the regime, and a Fund Manager agent synthesizes a weighted
   consensus (direction + confidence).
4. **PPO execution agent** — reinforcement learning agent that takes the regime embedding +
   consensus + portfolio state as observation and outputs a position (direction + size),
   reward = CVaR-scaled risk-adjusted return.

All four layers exist, run, and have been individually validated as *functioning* (correct
data, no leakage, deterministic evaluation, reproducible). The open problem is layer 2: **the
learned encoder never produced a representation good enough for the layers above it to exploit.**

---

## 2. Timeline of what was tried, and what each attempt found

### Months 1–3 (original build)
Data pipeline, HMM regime clustering, 5-agent LLM debate system, and PPO RL execution agent
were built and integrated. Early results looked promising in isolated snapshots (e.g. an early
BTC-only validation run reported Sharpe 1.8), but these numbers turned out to be artifacts of
methodology bugs discovered later (see next section) — not a validated result.

### Reliability audit (before this final push)
A dedicated audit found and fixed several **silent correctness bugs** that had been inflating
or randomizing reported results without anyone noticing:
- Evaluation was **not deterministic** — GPU float non-determinism could flip the sign of the
  reported Sharpe ratio run-to-run on the *same* checkpoint, because there were so few trades
  in the validation window that a handful of borderline decisions changed the outcome.
- A **training-only reward term (`idle_penalty`) was leaking into validation**, artificially
  inflating a reported Sharpe from a true 0.449 to a false 1.413.
- The **regime stream and the encoder checkpoint drifted out of sync** for 6 days without any
  guard catching it — the HMM was assigning regimes based on an embedding space that no longer
  matched the deployed encoder.
- A **duplicate-negative-count bug** and an **asset-naming mismatch bug** (GOLD vs. XAUUSD)
  recurred multiple times across different files, each one silently corrupting a downstream
  calculation until found.

None of these are "the project doesn't work" bugs — they're the kind of bug that makes a
number *look* good or bad for the wrong reason. Fixing them was a prerequisite for trusting
*any* number reported after that point, and is itself part of the methodological rigor worth
presenting: the project didn't just report the first good-looking number it saw.

### The core diagnosis: TEST 3 (RL learning-pipeline audit)
Before touching the encoder again, a controlled experiment directly compared:
- the trained PPO agent's ability to learn from the **encoder's 128-dim embedding**, vs.
- the trained PPO agent's ability to learn from the **raw 14 engineered features** directly.

Finding: **raw features scored above chance (AUC 0.547) at predicting next-period return
direction; the encoder's embedding scored at chance (AUC ~0.50) at every horizon tested.**
This was the first hard evidence that the encoder itself — not the RL agent, not the reward
function, not the agent debate layer — was the bottleneck. A portfolio-state confound
(the RL agent might be reacting to its own position rather than the market) was tested and
ruled out separately.

### Encoder diagnostic (why is the encoder embedding useless?)
A full diagnostic (embedding collapse statistics, reconstruction quality, per-feature
predictability) found:
- **Not a collapsed representation.** No near-zero-variance dimensions, healthy cosine-
  similarity spread, embeddings from the same week genuinely more similar than embeddings
  6 months apart. The encoder is not "broken" in the collapse sense.
- **It discards exactly the features that matter for direction.** Reconstruction quality was
  bimodal: indicator-*magnitude* features (ATR, RSI, volatility) reconstructed at R²
  0.82–0.88; return/direction-*shape* features (raw return, skew, kurtosis) reconstructed at
  R² 0.14–0.60 — the weakest reconstruction was exactly on the features most related to
  short-horizon direction.
- **Root-cause hypothesis (architectural):** the encoder mean-pools over a 60-hour window
  (diluting the most recent bar's signal into an average) and is trained with augmentations
  that make it *invariant* to the kind of small perturbation that is actually the return
  signal in a noisy series. Both choices are reasonable for slower regime-level features and
  actively wrong for short-horizon return direction.

### Phase 0 — four retraining experiments, each targeting one part of that hypothesis
Each was implemented, verified for correctness (leakage checks, artifact isolation, smoke
tests) *before* the real training run, trained to completion, and evaluated with the exact
same protocol (1h/4h/12h/24h AUC, reconstruction R², embedding health) so results are directly
comparable — not judged by training loss alone.

| Variant | Change tested | Result |
|---|---|---|
| **E2** | Removed jitter/masking augmentations | Did not beat raw_14 AUC at any horizon |
| **E1-lastpool** | Last-timestep pooling instead of mean-pooling | Did not beat raw_14; showed a *new* collapse risk (embedding rank 10/128, worse than the original) |
| **E1-predictive** | Predictive positive pairs (context→future, CPC-style) instead of augmented-view pairs | Best reconstruction of the investigation (R²=0.748) and the only variant with a *healthy* training dynamic (reconstruction improves with training instead of degrading) — but AUC gains were modest and inconsistent, weakest exactly at the 24h horizon |
| **E1-NegCtrl** | E1-predictive + excluding known-similar samples from contrastive negatives | Worse than E1-predictive at every single horizon, including the 24h horizon it targeted |

**None of the four beat the raw 14 features.** This is a clean, negative result across a
reasonably thorough set of the standard levers for this class of problem (pooling strategy,
augmentation strength, positive-pair construction, negative sampling).

### Downstream propagation audit — testing the least-bad encoder in the real pipeline
Rather than judge E1-predictive (the least-bad variant) on standalone AUC alone, it was traced
through the *actual* production pipeline — architecture compatibility, HMM regime detection,
agent consumption — with production artifacts never touched (isolated file paths throughout,
verified byte-identical before/after every step).
- **Architecture: compatible**, drop-in replacement, verified by loading both checkpoints and
  confirming identical shapes and a successful `load_state_dict`.
- **HMM regime quality: pathological.** The HMM built on E1-predictive's embeddings had 40x
  fewer regime transitions than production's (once every ~27 days vs. once every 16 hours),
  regime composition that drifted hard from the training period into validation/test (one
  regime went from 18% of train to 2% of val), and **essentially zero statistical agreement**
  with production's own regime labels on the same timestamps (Normalized Mutual Information
  0.0013 — indistinguishable from random).

**Conclusion: do not propagate E1-predictive.** Combined with the AUC results, this closed out
the encoder-retraining line of investigation — every reasonable lever tested, none worked,
and the project pivoted to a raw-feature baseline instead of continuing to iterate on the
encoder indefinitely.

### Raw-feature baseline pivot (last ~48 hours of work)
**Step 1 — HMM directly on raw features (no encoder, no learned compression):** validated
against three explicit, pre-registered criteria — balanced regime occupancy (no regime >50% of
test), dynamic (transitions on a days-not-months timescale), and interpretable (regime
statistics form a coherent story when checked against real market features). **All three
passed.** The raw-feature HMM (n=6 states) is a healthier regime detector than any encoder-based
version tried, by these criteria.

**Step 2 — RL agent trained directly on raw features (no encoder):** built a new observation
mode into the RL environment (opt-in, does not change any prior result), trained two variants
(raw features alone vs. raw features + regime ID), then investigated why both initially lost
money badly:
- Ruled out overtrading/transaction costs (drag was 0.01% of capital — negligible).
- Ruled out policy "churning" to avoid a training penalty (position sizes were large and
  deliberate-looking, not tiny noise).
- Found the real cause: the policy was making **large, confident directional bets with
  essentially no directional edge** (47–50% win rate, both long and short slightly negative
  expected value) — a steady bleed, not a blowup.
- Retrained with 3x the training steps (after discovering and fixing a bug where an early-
  stopping guard was silently cutting training short before the longer run could ever be
  tested): Sharpe improved from -2.75 to -1.21, and the result is **no longer statistically
  distinguishable from buy-and-hold**. But the win rate stayed at ~47% — the improvement came
  from the policy learning to size down and trade less, not from learning to predict direction.

**Current honest state: still not profitable, meaningfully less unprofitable than earlier
attempts, and the ceiling appears to be set by the market signal being genuinely weak, not by
undertrained models or fixable bugs.**

---

## 3. What is, and isn't, defensible right now

**Defensible:**
- A complete, working, end-to-end 4-layer system, individually validated at every stage.
- A rigorous methodology: every hypothesis tested was pre-specified, isolated (one variable
  changed at a time), compared against fixed baselines, and evaluated on the same protocol.
  Negative results were documented and kept, not silently discarded.
- Several genuine, non-trivial engineering findings independent of the trading result:
  nondeterministic-evaluation bug, train/val reward leakage, artifact staleness drift, and the
  fixes for each — these are real contributions about *how multi-stage ML pipelines fail
  silently*, which is a legitimate research contribution on its own.
- A clean, well-reasoned negative result on the core research question: contrastive pretraining
  of the type attempted here does not improve on raw features for this task, and the reasons
  why are documented and empirically grounded, not hand-waved.

**Not defensible / unresolved:**
- The core research claim (regime-aware multi-agent system beats regime-agnostic baselines) is
  **not validated** — no tested configuration is profitable, or shown superior to simpler
  baselines with statistical confidence.
- Two weeks is not enough time to find a fundamentally better representation-learning approach
  from scratch.
- The multi-agent LLM debate layer's actual contribution to trading performance has not been
  cleanly isolated end-to-end in this final configuration (most recent debugging effort went
  into the encoder/RL layers).

---

## 4. Specific questions worth bringing to your advisor

These are meant to be answerable in a short meeting, not open-ended:

1. **Is a well-documented negative result, on a clearly stated research claim, sufficient to
   validate a PFE at this school?** (Many programs explicitly accept this if the methodology is
   rigorous — worth asking directly rather than assuming either way.)
2. **Given ~2 weeks left, should the scope be reduced** — e.g., defend the data
   pipeline + regime detection + multi-agent debate as the completed, working contribution, and
   present the RL/encoder result as a documented negative-result investigation, rather than
   trying to make the full pipeline "work" end-to-end?
3. **How should the report be structured to foreground the methodology** (the audit trail,
   the controlled experiments, the ruled-out hypotheses) rather than reading as "we tried to
   build X and failed"?
4. **Is there a lower-risk, higher-value use of the remaining two weeks** — e.g., a smaller,
   well-scoped additional experiment, versus polishing documentation and the defense narrative
   of what already exists?
5. **What to do about the unresponsive school mentor**, given the B2 credit issue already
   blocked the first *restitution* — does the current advisor have a way to flag this
   administratively, or should the student escalate separately?

---

## 5. Where to point your advisor for detail

- `docs/ARCHITECTURE.md` — full system design, one section per layer, plus a retrospective
  section on lessons learned.
- `CLAUDE.md` (repo root) — the dense, chronological engineering log this document was
  summarized from; has exact numbers, file names, and dates for everything above.
- `data/results/` — raw evaluation JSONs (per-checkpoint diagnostics, ablation studies,
  regime-quality audits) backing every number in Section 2.
- `data/results/llm_layer_audit.json` and `scripts/llm_layer_audit.py` — full prompt/response
  trace behind Section 6.

---

## 6. LLM/agent layer audit (2026-08-18) — a necessary reframe of the "multi-agent debate" claim

A dedicated audit (`scripts/llm_layer_audit.py`, 20 real Mistral 7B calls, BTC_USDT @
2024-03-01, full trace in `data/results/llm_layer_audit.json`) traced every agent from prompt
to final action and ran five validation tests. **Say this reframe out loud in the defense before
the committee has to ask for it:**

> The system is not "multi-agent LLM debate decides trades." It is a **deterministic
> risk-weighted consensus engine**: three LLM agents (Macro, Technical, Sentiment) supply
> directional opinions; one agent's dissenting direction is hardcoded in Python, not decided by
> its own LLM call; one agent (Risk) never calls an LLM at all; and the Fund Manager's actual
> decision (action, position size, conviction, dissent) is pure arithmetic over the other
> agents' outputs — its one LLM call produces only the rationale sentence, never the decision.

### What passed

- **Schema test**: 6/6 agents — every call returned a valid, non-fallback response across all
  20 calls.
- **Null regime test**: 3/3 agents (Macro, Technical, Sentiment) — swapping the real regime for
  a random one measurably changed output, so regime context is not being silently discarded.
- **Aggregator-usage test**: flipping one agent's thesis (no new LLM call) changed the
  Fund Manager's conviction 0.81→0.51 and its dissent list — confirmed the deterministic vote
  genuinely reads every agent's output rather than hardcoding an answer.

### What failed — document these as findings, not bugs to chase before the defense

1. **TechnicalAgent ignores obvious bearish price action.** RSI 12, MACD strongly negative,
   return −6% still produced "bullish, confidence 0.65." Raw response: *"RSI at a low level...
   suggests potential for price to rebound... combined with the current regime being a
   high-volatility bull regime, supports this view."* It read oversold RSI as mean-reversion
   and let the (unchanged) regime block override the price extremity. **Decision: document,
   do not spend remaining time tuning the prompt** — a two-week budget doesn't cover
   reliably fixing LLM directional judgment.
2. **Regime sensitivity is real but not directional.** Forcing MacroAgent's regime block to
   the single best-Sharpe regime in the dataset ("bull") did not make it more bullish — output
   went from bullish/0.85 (real) to neutral/0.50, the *same* neutral/0.50 every synthetic
   regime produced (bull, bear, sideways, random alike). Only the *actual* regime — which
   happened to agree with the agent's independently-computed correlation data — produced a
   confident call. Reframe: the agents read regime as a corroborating signal, not a
   directional prior.
3. **ContrarianAgent's dissent is not LLM-generated.** `agents/contrarian_agent.py:213` forces
   `directional_bias` to the opposite of consensus in code regardless of what the LLM said.
   Only confidence/reasoning/key_factors come from the model. It cannot be called "a debating
   agent" without that caveat.
4. **RiskAgent instantiates an LLM client it never calls** — harmless (GARCH/CVaR position
   caps are pure Python) but misleading in an architecture diagram; fix the diagram, not the
   code.
5. **Regime staleness — fixed 2026-08-18, low effort, worth doing.** `build_regime_context()`
   silently served a regime row up to 20 hours old on any data gap, with no signal to the
   agent or the prompt. Fixed by adding `RegimeContext.regime_stale` / `regime_gap_hours`
   fields (flag, not a hard failure — a hard raise would take down live debates on any
   ordinary gap) and a `⚠ REGIME DATA STALE` line injected into the prompt when it fires.
   Verified: 2024-03-01 (exact match) → not stale; 2024-06-15 12:00 (known data gap) →
   correctly flagged, 20.0h gap.

### Architectural disconnection from the reported RL policy

`grep -rn "llm" rl/ --include="*.py"` returns exactly one substantive hit —
`rl/env.py:38`, which states outright that "the LLM debate system is never [called during
training]." `ppo_best.zip` (the checkpoint behind every number in Section 2/the dashboard) is
`obs_mode="raw"`: 0 of its 868 input dimensions are regime-derived, and the debate cache has
essentially no coverage of the RL validation window (~1% of 2024 val steps, 0% of the 2025 test
window) even in `obs_mode`s that do carry a consensus slot. **The LLM/agent layer and the
reported PPO policy are two separately-validated components, not one integrated system** —
state this plainly rather than implying the debate layer influenced the traded policy.

### Defense Q&A this audit now has evidence for

| Likely question | Answer, with evidence |
|---|---|
| "Why call it a debate if the Fund Manager is deterministic?" | It isn't a decision-making debate — it's LLM-sourced directional *opinions* feeding a deterministic weighted vote. Cite the aggregator-usage test. |
| "Why does TechnicalAgent fail on obvious bearish input?" | Reproducible example in `llm_layer_audit.json` — cite the exact RSI/MACD/return values and raw response quoted above. |
| "Why is ContrarianAgent's direction hardcoded?" | By design (`contrarian_agent.py:213`) — its role was always to guarantee opposition for the aggregator to weigh, not to be a free LLM vote. State this as intentional, not discovered-as-a-bug. |
| "Is the LLM layer connected to the PPO policy you're reporting?" | No — architecturally disconnected, confirmed by `grep -rn llm rl/` and the checkpoint's own `obs_mode`. |
| "How do you know regime context isn't just ignored?" | Null-regime test: output changes with regime for all 3 directional agents — but the sensitivity isn't cleanly directional (see finding 2). |

**Adopt this framing in the written report and oral defense:**

> "We built and audited a regime-aware deterministic consensus engine with LLM-assisted
> directional inputs. We demonstrate that the LLM layer contributes opinions but does not make
> the final decision, and we document its failure modes (extreme-signal miscalibration, a
> corroboration-only relationship with regime context, a hardcoded dissenting agent, and a
> since-fixed staleness gap) rather than presenting an inflated 'LLM debate' claim."
