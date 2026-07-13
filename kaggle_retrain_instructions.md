# Kaggle Encoder Retrain — 2024 Data Extension

Retrains the TFT+InfoNCE encoder with the extended dataset under the new split:

| | |
|---|---|
| Train | 2018 → 2022-12-31 (`train_end: "2023-01-01"`, exclusive) |
| Val (early stopping) | 2023-01-01 → 2023-12-31 |
| Test | 2024 — **must never reach Kaggle** (see Step 0) |

WTIUSD was removed from the project entirely (histdata has no WTI past 2023-12-01
and yfinance cannot provide genuine hourly bars for the full test year). MARTA runs
on 4 genuine assets: BTC_USDT, ETH_USDT, EURUSD, XAUUSD.

---

## Step 0 — LOCAL: prepare the upload folder (trim 2024 out)

`encoder/train.py` splits with `val_df = df[df.index >= val_start]` and has **no
`val_end`** — if the uploaded parquets contain 2024, the encoder's early stopping
would select a checkpoint using 2024 loss and the test set is no longer untouched.
So the upload is trimmed to ≤ 2023-12-31:

```python
# Run locally from the repo root AFTER run_pipeline.py has regenerated data/processed/
from pathlib import Path
import pandas as pd

SRC = Path("data/processed")
DST = Path("kaggle_upload/marta-processed-data-v2")
DST.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
for asset in ["BTC_USDT", "ETH_USDT", "EURUSD", "XAUUSD"]:
    df = pd.read_parquet(SRC / f"{asset}_aligned.parquet", engine="pyarrow")
    trimmed = df.loc[df.index <= CUTOFF]
    trimmed.to_parquet(DST / f"{asset}_aligned.parquet", engine="pyarrow")
    print(f"{asset}: {len(df)} -> {len(trimmed)} rows (ends {trimmed.index[-1]})")
```

Then create/version the Kaggle dataset **`marta-processed-data-v2`** from
`kaggle_upload/marta-processed-data-v2/` (Kaggle → Datasets → New Dataset, or
`kaggle datasets version -p kaggle_upload/marta-processed-data-v2 -m "2024 extension, trimmed to 2023"`).

---

## Step 1 — Notebook Cell 1: updated config

```python
%%writefile configs/assets.yaml
# (only the sections encoder training reads; dates are the 2024-extension split)
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
  train_end: "2023-01-01"   # NEW: train < 2023-01-01 (was 2021-01-01)
  val_start: "2023-01-01"   # NEW: val on 2023 (2024 is absent from the dataset)

data:
  processed_dir: "/kaggle/input/marta-processed-data-v2"   # NEW dataset
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
assert cfg["training"]["train_end"] == "2023-01-01"
assert cfg["training"]["val_start"] == "2023-01-01"

pdir = Path(cfg["data"]["processed_dir"])
for asset in cfg["data"]["assets"]:
    df = pd.read_parquet(pdir / f"{asset}_aligned.parquet")
    assert df.index[-1] <= pd.Timestamp("2023-12-31 23:59:59", tz="UTC"), \
        f"{asset} contains post-2023 rows — test-set leakage, STOP"
    print(f"{asset}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
print("OK — no 2024 rows on Kaggle, split dates correct")
```

## Step 4 — run the existing training cell

The training cell itself is unchanged from the v2-retrain notebook (it drives
`encoder/train.py` / `run_encoder.py` off `configs/assets.yaml`). Expected: ~200
epochs with early stopping; GPU (T4/P100) session.

Convergence gates (unchanged): `val_loss < 4.0`, `top1_acc > 0.20`,
temperature ∈ [0.03, 0.30].

---

## After training

1. Download `encoder/checkpoints/best_encoder.pt` from the Kaggle output.
2. Replace it locally: `cp ~/Downloads/best_encoder.pt encoder/checkpoints/best_encoder.pt`
   (keep a backup of the old one first:
   `cp encoder/checkpoints/best_encoder.pt encoder/checkpoints/best_encoder_pre2024.pt`).
3. Re-run everything that consumes the encoder — from the repo root:
   `python run_overnight.py` completes Steps 3-9 (HMM → debates → RL → test set);
   Steps 1-2 are idempotent no-ops if the data is already extended.
   If the overnight run already finished on the OLD encoder, you must re-run
   Steps 3-9 anyway — the regime stream and PPO are encoder-dependent.
