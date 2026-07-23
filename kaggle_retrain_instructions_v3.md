# Kaggle Encoder Retrain — 2026-07-13 Walk-Forward (train<2024, val 2024, test 2025)

Supersedes `kaggle_retrain_instructions.md` (the 2024-extension retrain). Root cause:
the encoder trained under that split never saw data past 2023, so its embeddings for
2024/2025 bars are out-of-distribution — traced this session to the RL policy going
fully flat (Sharpe 0.000, 0 trades) on 2025 while still trading fine on 2024 (Sharpe
0.372). This retrain walks every split boundary forward one year to fix it at the
source instead of patching the RL layer.

| | |
|---|---|
| Train | 2018 → 2023-12-31 (`train_end: "2024-01-01"`, exclusive) |
| Val (early stopping) | 2024-01-01 → 2024-12-31 |
| Test | 2025 — **must never reach Kaggle** (see Step 0) |

Same 4 assets as before: BTC_USDT, ETH_USDT, EURUSD, XAUUSD.

---

## Step 0 — LOCAL: prepare the upload folder (already done this session)

`kaggle_upload/marta-processed-data-v3/` has already been trimmed to ≤ 2024-12-31
(40,016 rows/asset, ends 2024-12-31 16:00:00+00:00 — same row count as the old
untrimmed 2024-extension dataset, confirming the trim boundary is correct):

```python
# Already run — for reference / reproducibility only.
from pathlib import Path
import pandas as pd

SRC = Path("data/processed")
DST = Path("kaggle_upload/marta-processed-data-v3")
DST.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
for asset in ["BTC_USDT", "ETH_USDT", "EURUSD", "XAUUSD"]:
    df = pd.read_parquet(SRC / f"{asset}_aligned.parquet", engine="pyarrow")
    trimmed = df.loc[df.index <= CUTOFF]
    trimmed.to_parquet(DST / f"{asset}_aligned.parquet", engine="pyarrow")
    print(f"{asset}: {len(df)} -> {len(trimmed)} rows (ends {trimmed.index[-1]})")
```

**You still need to**: create/version the Kaggle dataset **`marta-processed-data-v3`**
from `kaggle_upload/marta-processed-data-v3/` (Kaggle → Datasets → New Dataset, or
`kaggle datasets version -p kaggle_upload/marta-processed-data-v3 -m "walk-forward retrain, trimmed to 2024"`).

---

## Step 1 — Notebook Cell 1: updated config

```python
%%writefile configs/assets.yaml
# (only the sections encoder training reads; dates are the walk-forward split)
model:
  input_dim: 14
  hidden_dim: 128
  proj_dim: 64
  n_lstm_layers: 2

training:
  window_size: 60
  step: 1
  batch_size: 512
  lr: 0.001
  weight_decay: 0.0001
  max_epochs: 200
  early_stopping_patience: 20
  warmup_epochs: 10
  gradient_clip_max_norm: 1.0
  train_end: "2024-01-01"   # NEW: train < 2024-01-01 (was 2023-01-01)
  val_start: "2024-01-01"   # NEW: val on 2024 (2025 is absent from the dataset)

data:
  processed_dir: "/kaggle/input/marta-processed-data-v3"   # NEW dataset
  assets:
    - BTC_USDT
    - ETH_USDT
    - EURUSD
    - XAUUSD

convergence:
  max_val_loss: 4.0
  min_top1_acc: 0.20
  min_temperature: 0.03
  max_temperature: 0.30

checkpoints:
  dir: "encoder/checkpoints"
  best: "best_encoder.pt"
  final: "final_encoder.pt"
```

## Step 2 — Notebook Cell 2: install

```python
!pip install pytorch-forecasting mlflow ta hmmlearn -q
```

## Step 3 — Notebook Cell 3: config sanity check (before the training cell)

```python
import yaml, pandas as pd
from pathlib import Path

cfg = yaml.safe_load(open("configs/assets.yaml"))
assert cfg["training"]["train_end"] == "2024-01-01"
assert cfg["training"]["val_start"] == "2024-01-01"

pdir = Path(cfg["data"]["processed_dir"])
for asset in cfg["data"]["assets"]:
    df = pd.read_parquet(pdir / f"{asset}_aligned.parquet")
    assert df.index[-1] <= pd.Timestamp("2024-12-31 23:59:59", tz="UTC"), \
        f"{asset} contains post-2024 rows — test-set leakage, STOP"
    print(f"{asset}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
print("OK — no 2025 rows on Kaggle, split dates correct")
```

## Step 4 — run the existing training cell

Unchanged from the v2/v2024 notebooks (drives `encoder/train.py` / `run_encoder.py`
off `configs/assets.yaml`). Expected: ~200 epochs with early stopping; GPU (T4/P100)
session.

Convergence gates (unchanged): `val_loss < 4.0`, `top1_acc > 0.20`,
temperature ∈ [0.03, 0.30].

---

## After training — hand back to me, or do it yourself:

1. Download `encoder/checkpoints/best_encoder.pt` from the Kaggle output.
2. Back up the old one first: `cp encoder/checkpoints/best_encoder.pt encoder/checkpoints/best_encoder_pre2025.pt`
3. Replace it: `cp ~/Downloads/best_encoder.pt encoder/checkpoints/best_encoder.pt`
4. Then, in order (do **not** run the full `run_overnight.py` — Steps 1-2 there are
   stale, pointing at `pipeline/fetch_2024.py`, and Steps 4-6 (per-asset HMM, weekly
   debates, clean no-regime debates) are deliberately deferred for this pass — see
   the approved plan):
   - `python run_regime.py` — refit HMM/PCA on the new embeddings (~30-60 min)
   - `python run_rl.py` — full curriculum retrain, same 4-phase structure (up to ~6h)
   - `python run_test_set.py` — evaluate on the new blind 2025 window

Compare against this session's baseline: old `ppo_best` scored Sharpe 0.000 (0 trades,
total abstention) on 2025, but Sharpe 0.372 (31 trades) on 2024 — the new checkpoint
should clear a similar bar on 2025, its new "one year out" case, if the encoder-staleness
fix worked.
