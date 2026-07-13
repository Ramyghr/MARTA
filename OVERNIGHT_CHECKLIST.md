# MARTA Overnight Retrain Checklist — 2024 Extension

## Already done (this session)

- [x] `pipeline/fetch_2024.py` ran — all assets extended to 2024-12-31
      (BTC/ETH via CCXT; EURUSD/XAUUSD via the root histdata ZIPs)
- [x] **WTIUSD removed from the project** (histdata coverage ends 2023-12-01; the
      yfinance fallback could only provide daily-upsampled intraday data, which
      undermines the encoder). MARTA now runs on 4 genuine assets:
      BTC_USDT, ETH_USDT, EURUSD, XAUUSD — the aligned dataset, Kaggle upload,
      curriculum Phase 4, and dashboard all reflect this.
- [x] `run_pipeline.py` ran — aligned parquets regenerated with 2024
- [x] Splits moved: encoder/HMM train <2023 (configs/assets.yaml), RL train ≤2022 /
      val 2023 (rl/env.py), test 2024 (run_test_set.py)

## Before sleeping

- [ ] Prepare the trimmed Kaggle upload (Step 0 of `kaggle_retrain_instructions.md` —
      **mandatory**, keeps 2024 out of encoder early-stopping)
- [ ] Upload to Kaggle dataset `marta-processed-data-v2`
- [ ] Start the Kaggle encoder retrain notebook (cells in `kaggle_retrain_instructions.md`)
- [ ] Start the local pipeline in the background:
      `nohup /home/ramy/miniconda3/envs/marta/bin/python run_overnight.py > /tmp/overnight.log 2>&1 &`
- [ ] Monitor: `tail -f /tmp/overnight.log`

⚠ Ordering caveat: `run_overnight.py` Steps 3-9 use whatever
`encoder/checkpoints/best_encoder.pt` is on disk (its mtime is printed before Step 3).
If Kaggle hasn't finished by then, the overnight run completes on the OLD encoder and
Steps 3-9 must be re-run after swapping the checkpoint (see "When you wake up").

## When you wake up

- [ ] Check `/tmp/overnight.log` for `❌ FAILED` / `❌ TIMED OUT`
- [ ] Download `best_encoder.pt` from Kaggle
- [ ] Back up then replace the local checkpoint:
      `cp encoder/checkpoints/best_encoder.pt encoder/checkpoints/best_encoder_pre2024.pt`
      `cp ~/Downloads/best_encoder.pt encoder/checkpoints/best_encoder.pt`
- [ ] If the overnight run used the old encoder (compare the mtime printed in the log
      against the Kaggle download time): re-run Steps 3-9 —
      `run_regime.py` → `run_per_asset_regime.py` → `run_weekly_debates.py` →
      `run_clean_ablation_debates.py` → `run_rl.py` → `run_test_set.py` →
      `run_rl.py --eval-only` (or just re-run `run_overnight.py`; Steps 1-2 are no-ops)
- [ ] Check `data/results/test_set_results.json` — 2024 test metrics + `key_finding`
- [ ] Check `data/results/ablation_results.json` — 7-variant ablation on val 2023
- [ ] Restart dashboard: `cd dashboard && docker compose up --build -d`

## Expected results (interpret honestly)

- **Val (2023):** new territory — the old 1.836 was on val-2021 with the old encoder
  and old split, so it is NOT directly comparable. Treat 2023 val Sharpe as a fresh
  number.
- **Test (2024):** Sharpe > 0 with meaningful trades ⇒ the policy generalizes.
  **Sharpe exactly 0.000 with 0 trades ⇒ the abstention pathology again** (the
  pre-extension checkpoint held a zero position on every out-of-sample step —
  `run_test_set.py` now auto-probes the val window and writes a `key_finding` that
  distinguishes "2024 shift" from "abstains everywhere").
- BTC 2024 context: April halving, ~$73k ATH in March, ~$100k in November — a strong
  bull year, so buy-and-hold sets a high absolute-return bar; MARTA's edge, if any,
  will be risk-adjusted (Sharpe / max-drawdown), not total return.
- The aligned dataset grows vs the 5-asset version: WTI was the binding constraint
  on the inner join (its histdata ended 2023-12-01 and had the sparsest calendar),
  so the 4-asset join keeps more rows. Old 5-asset row counts are not comparable.
