# Kaggle Weekly-Cadence PPO Grid — Seeds 1-3 (crypto5 panel)

Fills in the multi-seed variance estimate for the crypto5 weekly-cadence PPO grid
(`rl/train_weekly_crypto5_grid.py`) — BTC/ETH/BNB/LTC/XRP × 4 conditions (A raw / B
raw_vol / C raw_regime / D raw_regime+placebo) × 4 seeds = 80 cells. **The seed-0 pilot
(20 cells) already completed locally** — checkpoints are in `rl/checkpoints/
ppo_best_crypto5_weekly_{asset}_{condition}_seed0.zip` and results in
`data/results/weekly_grid_crypto5/`. This run is for **seeds 1, 2, 3 only** (60 cells),
the remaining piece before `docs/RESULTS_AND_METHODOLOGY.md`'s weekly-grid finding can
be reported as a distribution across seeds instead of a single-seed point estimate.

Runs on Kaggle, not locally, so it doesn't need your PC open — same "Save & Run All
(Commit)" discipline as `kaggle_debate_coverage_instructions.md`, which runs server-side
even after you close the tab.

**CPU only, no GPU, no internet needed after the one pip-install cell — genuinely
simpler than the debate-coverage notebook.** This is PPO training on 22 raw market
features, not an LLM call: no Ollama, no Mistral pull, no FinBERT, nothing to download
mid-run. `config["device"]="cpu"` is hardcoded in `build_weekly_config()` regardless —
weekly mode never touches the encoder, so a GPU accelerator would buy nothing here and
would only cost you GPU quota better spent elsewhere (e.g. the debate-coverage notebook).

Local package prepared and verified this session: `kaggle_upload/marta-weekly-ppo-pkg/`
(110 MB — `rl/`, `run_rl.py`, `utils/`, `encoder/` (model/loss/dataset/augment only, no
checkpoints — needed because `rl/train.py` imports `encoder.model` at module level even
though weekly mode never loads it), `configs/splits.py`, and only the 7 parquet files
`rl/train_weekly_crypto5_grid.py` actually reads: the 5 assets' `*_aligned.parquet` +
`regime_stream_causal.parquet` + `regime_stream_placebo.parquet`. Deliberately excludes
`regime_stream.parquet` (leaky, unused), `embeddings.npy`/`embedding_*` (encoder-mode
only, never used since `obs_mode` is always raw/raw_vol/raw_regime here), and
`*_normalized_live.parquet` (live-inference only). Verified end-to-end from inside the
package directory this session (different cwd, the real Kaggle scenario): `rl.train`'s
path constants correctly resolve to the package root, not the real repo, and
`load_data()` successfully loads real data through it — training itself was **not**
launched, per instructions.

**Two real fixes made while preparing this, not just packaging:**
1. `rl/train.py`, `rl/callbacks.py`, and `utils/freshness.py` had bare
   `Path("data/processed")`-style constants with no `MARTA_ROOT` fallback — the same bug
   class this project has hit repeatedly in `dashboard/backend/` (most recently
   `rl/env.py`'s `_CACHE_DIR`, per CLAUDE.md). Never mattered before since these were
   always run with cwd=repo-root via `run_rl.py`, but a Kaggle notebook's cwd is
   `/kaggle/working/marta` after the `%cd` below — now anchored the same way every other
   module in this project is. `tests/test_crypto5_panel.py`'s two assertions that checked
   exact relative-path equality were updated to match (still real regression guards,
   just checking the meaningful path suffix instead of exact-equals-relative-Path).
2. `rl/train_weekly_crypto5_grid.py`'s `--pilot`/`--all` loops had no resume-skip — a
   re-run after an interrupted session retrained every cell from scratch, including ones
   already on disk. Added a `--force` flag (default off): without it, a cell whose result
   JSON already exists is skipped. `--asset --label --seed` (single-cell, already existed)
   is unaffected — an explicit single-cell request always runs.

---

## Timing (measured, not estimated) — this is why the split-by-asset plan below exists

The seed-0 pilot's 20 real cells give real per-cell wall-clock, not a guess:
**mean 738s (~12.3 min), range 147s-910s** (`train_wall_seconds` in every
`data/results/weekly_grid_crypto5/summary_*_seed0.json`). Extrapolated:

- All 60 remaining cells sequentially: **60 × 738s ≈ 12.3 hours** — too long for one
  Kaggle session/commit to safely guarantee finishing.
- One asset's 12 cells (4 conditions × 3 seeds): **12 × 738s ≈ 2.5 hours** — comfortably
  fits in a single commit.

**Plan: one Kaggle commit per asset (5 commits total, ~2.5h each)**, not one giant
80-... sorry, 60-cell commit. This sidesteps needing real cross-commit resume (Kaggle's
"Save & Run All" starts fresh from the attached input dataset every time — files written
to `/kaggle/working/` in a previous commit do **not** carry over to the next one
automatically, so resume-skip only protects retries *within* a single running session,
not across separate commits). If a single asset's 12-cell commit still times out, Step 7
below documents how to carry partial progress into a follow-up commit for that asset.

---

## Step 0 — create the Kaggle dataset (one-time)

Kaggle → Datasets → New Dataset → upload the contents of
`kaggle_upload/marta-weekly-ppo-pkg/` → name it `marta-weekly-ppo-pkg`.

(Or via CLI: `kaggle datasets create -p kaggle_upload/marta-weekly-ppo-pkg -m "weekly PPO grid pkg"`.)

If this directory doesn't exist (it's gitignored — see `.gitignore`), regenerate it from
a clean checkout:

```bash
PKG=kaggle_upload/marta-weekly-ppo-pkg
mkdir -p "$PKG/rl" "$PKG/utils" "$PKG/encoder" "$PKG/configs" \
         "$PKG/data/processed/crypto5_multihorizon" "$PKG/data/results/weekly_grid_crypto5"
cp rl/*.py "$PKG/rl/"
cp run_rl.py "$PKG/"
cp utils/*.py "$PKG/utils/"
cp encoder/__init__.py encoder/model.py encoder/loss.py encoder/dataset.py encoder/augment.py "$PKG/encoder/"
cp configs/__init__.py configs/splits.py "$PKG/configs/"
for a in BTC_USDT ETH_USDT BNB_USDT LTC_USDT XRP_USDT; do
  cp "data/processed/crypto5_multihorizon/${a}_aligned.parquet" "$PKG/data/processed/crypto5_multihorizon/"
done
cp data/processed/crypto5_multihorizon/regime_stream_causal.parquet \
   data/processed/crypto5_multihorizon/regime_stream_placebo.parquet \
   "$PKG/data/processed/crypto5_multihorizon/"
find "$PKG" -name __pycache__ -exec rm -rf {} +
```

---

## Step 1 — new Kaggle Notebook (repeat per asset — 5 notebooks, or 1 notebook re-run 5x with the asset changed)

Settings (right sidebar):
- **Accelerator: None (CPU)** — deliberately not GPU, see above. Saves your GPU quota.
- **Internet: On** — only needed for the one `pip install` cell below; nothing else
  makes a network call (no external API, no model download).
- **Add Data** → attach the `marta-weekly-ppo-pkg` dataset from Step 0.

---

## Step 2 — Cell 1: stage the package into a writable directory

```python
!cp -r /kaggle/input/marta-weekly-ppo-pkg /kaggle/working/marta
%cd /kaggle/working/marta
!mkdir -p rl/checkpoints data/results/weekly_grid_crypto5
```

## Step 3 — Cell 2: install Python deps

Pinned to match what actually produced the seed-0 checkpoints (the local conda env
`marta`) — `pandas`/`numpy`/`scipy`/`pyarrow`/`torch` are assumed already on the Kaggle
base image (standard for both the CPU and GPU images) and deliberately not reinstalled
here; if a version mismatch causes trouble, the sanity-check cell below will surface it
before the real run, not mid-sweep.

```python
!pip install -q stable-baselines3==2.8.0 gymnasium==1.2.3 mlflow-skinny==3.12.0 rich==15.0.0

import os
# mlflow 3.x deprecated the local filesystem tracking store by default (same fix
# dashboard/docker-compose.yml already needed) — this notebook has no tracking server.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
```

## Step 4 — Cell 3: sanity check before the real run

Confirms the package's data loads correctly and the path-anchoring fix actually works
from `/kaggle/working/marta` (a different cwd than this was ever run from before) —
without spending 12+ minutes finding that out via a failed real training cell.

```python
import sys
sys.path.insert(0, "/kaggle/working/marta")

import rl.train as t
print("MARTA_ROOT resolved to:", t.MARTA_ROOT)  # must be /kaggle/working/marta, not a stale local path
assert str(t.MARTA_ROOT) == "/kaggle/working/marta"

from rl.train_weekly_crypto5_grid import build_weekly_config
from rl.train import load_data
cfg = build_weekly_config("BTC_USDT", "raw", seed=1)
regime_stream, aligned_dfs = load_data(cfg)
print("Data OK —", len(aligned_dfs["BTC_USDT"]), "BTC_USDT rows")
```

If `MARTA_ROOT` prints anything other than `/kaggle/working/marta`, stop — the path
anchoring isn't working and every downstream checkpoint/result path will be wrong.

## Step 5 — Cell 4: the real run — **change `ASSET` per commit**

```python
ASSET = "BTC_USDT"  # <- change this each of the 5 times you run this notebook

!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label A --seed 1
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label A --seed 2
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label A --seed 3
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label B --seed 1
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label B --seed 2
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label B --seed 3
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label C --seed 1
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label C --seed 2
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label C --seed 3
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label D --seed 1
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label D --seed 2
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py \
    --asset {ASSET} --label D --seed 3
```

Twelve explicit single-cell invocations, not `--all --seeds 1 2 3` — each cell is its own
subprocess-free `!python` call, so a crash on cell 7 still leaves cells 1-6's results and
checkpoints on disk in this same session (`--all`'s in-process loop would too, via the
`--force`-gated skip logic, but explicit per-cell calls also give you a clean per-cell log
in the notebook's cell output, worth it at only 12 lines).

**Run this via "Save & Run All (Commit)"**, not by clicking through cells interactively —
commit runs execute server-side and keep going after you close the tab. Check progress
from the notebook's Viewer/versions page.

If this same commit needs to be re-run after an interruption (e.g. it timed out on cell
9), just re-running the identical commit is safe — Step 3's `--force`-gated skip logic
means cells 1-8's already-written result JSONs are skipped, not retrained, **as long as
you're still in the same session** (files written earlier in that same run are still on
disk in `/kaggle/working/`). A *new* "Save & Run All" starts completely fresh from the
input dataset, so this only helps within-session retries, not across separate commits —
see Step 7 if you need real cross-commit continuation.

## Step 6 — Cell 5: summarize this asset's 12 cells

```python
!cd /kaggle/working/marta && python rl/train_weekly_crypto5_grid.py --summarize-only
```

Writes `data/results/weekly_grid_crypto5/summary.json` — prints a mean/std-across-seeds
line per (asset, condition) cell that has 2+ seeds present. With only seeds 1-3 from this
asset's run, it reconciles those three; merge with the local seed-0 results afterward
(Step 7) for the full 4-seed picture.

---

## Step 7 — after each commit: bring the results back

Kaggle → your notebook → Output tab → download the whole
`data/results/weekly_grid_crypto5/` folder and `rl/checkpoints/` folder (12 new
`ppo_best_crypto5_weekly_{asset}_{condition}_seed{1,2,3}.zip` files — checkpoints, not
just eval JSONs, since nothing else on Kaggle keeps them).

Locally, drop the downloaded files straight into the matching local directories — every
filename is already asset/condition/seed-namespaced
(`{asset}_{label}_seed{seed}.json`, `ppo_best_crypto5_weekly_{asset}_{condition}_seed{seed}.zip`),
so there's no collision risk the way `debate_results.json`'s single shared filename had —
just copy, no merge/dedup script needed:

```bash
cp ~/Downloads/weekly_grid_crypto5/*.json data/results/weekly_grid_crypto5/
cp ~/Downloads/ppo_best_crypto5_weekly_*_seed{1,2,3}.zip rl/checkpoints/
```

Repeat Steps 1-7 for the other four assets (ETH/BNB/LTC/XRP), then reconcile everything
in one place:

```bash
/home/ramy/miniconda3/envs/marta/bin/python rl/train_weekly_crypto5_grid.py --summarize-only
```

**If a single asset's 12-cell commit times out partway and you need real cross-commit
continuation** (rather than just re-running the same commit, Step 5's within-session
caveat): download whatever's in `data/results/weekly_grid_crypto5/` and `rl/checkpoints/`
from that commit's Output tab, upload them as a **new version** of the
`marta-weekly-ppo-pkg` dataset (Kaggle Datasets → your dataset → "New Version", adding
those files into the same `data/results/weekly_grid_crypto5/`/`rl/checkpoints/` paths the
package already has them at), then start a fresh notebook commit with that updated
dataset version attached — the `--force`-gated resume-skip now sees those cells as
already done and picks up where the previous commit left off.

---

That gives you the seed 1-3 checkpoints and eval results this grid needs. Re-running
`docs/RESULTS_AND_METHODOLOGY.md`'s weekly-grid section with the full 4-seed picture
(mean ± std per asset × condition, not a single-seed point estimate) is separate
follow-up work, not part of this run.
