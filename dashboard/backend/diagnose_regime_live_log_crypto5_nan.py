"""Diagnostic for a regime_live_log.parquet data-quality bug found 2026-09-03.

268 crypto5-asset rows (BTC/ETH/BNB/LTC/XRP_USDT), timestamped 2026-08-08 -> today,
have NaN regime_prob_4/regime_prob_5 even though their regime_label can legitimately
be 4 or 5. Root cause: _append_live_regime_log() (regime_service.py) sized the logged
row from _n_regime_states() — the LEGACY 4-state encoder-based _hmm_model's component
count — instead of from the result dict actually passed in, whenever that legacy
model happened to also be loaded in the same backend process as the crypto5 6-state
model. FIXED going forward in regime_service.py: _append_live_regime_log now derives
n_states from len(result["probabilities"]), so every future crypto5 row always logs
all 6 real probabilities regardless of what else is loaded in-process.

EURUSD/GOLD rows with NaN regime_prob_4/5 are a SEPARATE, expected thing, not this
bug — those assets genuinely use the legacy 4-state model, so 4 populated
probabilities is correct for them.

DECISION: the existing 268 NaN rows are left unrepaired — a known, now-explained
historical gap, not backfilled. This script exists to show why, and as a diagnostic
if that decision is ever revisited.

Backfilling was attempted (re-running get_live_regime_crypto5() for each broken
timestamp) and rejected: the recompute is NOT a faithful reconstruction of what the
live system actually said at write-time. Evidence (see this script's output):
  - regime/checkpoints/crypto5_multihorizon/hmm_model.pkl has a SINGLE mtime
    (2026-08-19 15:43:33 UTC) with no version history (gitignored binary checkpoint,
    overwritten in place by whatever last retrained it) — any row before that mtime
    was logged by a DIFFERENT (now-gone) set of model weights than "today's" model
    would recompute with.
  - Empirically, 6 of 268 recomputed rows produce an argmax that DISAGREES with the
    regime_label that was actually logged live at write-time — not floating-point
    noise, a genuine posterior flip. This happens even for some rows dated AFTER the
    checkpoint's mtime, so a single retrain event doesn't fully explain the drift
    either; treat any recompute-based reconstruction here as unverified.
  - regime_label itself (unlike the 2 missing probabilities) was never touched by
    the original bug and remains trustworthy for all 268 rows — only the raw
    probability vector is the actual gap.

This is consistent with downstream consumers already handling the gap gracefully —
rl/obs_builder.py's build_regime_prob_series() fillna(0.0)s missing regime_prob_i —
so leaving it as NaN is a correctness choice, not a functional blocker.

Run: conda run -n marta python dashboard/backend/fix_regime_live_log_crypto5_nan.py
(report-only — does not write or modify any file)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
for p in (str(MARTA_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import regime_service

_CRYPTO5_ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
_LIVE_LOG_PATH = MARTA_ROOT / "data" / "processed" / "regime_live_log.parquet"


def main() -> None:
    regime_service.load_regime_stream()
    regime_service.load_regime_models()
    if regime_service._crypto5_hmm_model is None:
        print("ERROR: crypto5 HMM failed to load — cannot diagnose, aborting")
        return

    log = pd.read_parquet(_LIVE_LOG_PATH, engine="pyarrow")
    n_states = regime_service._crypto5_hmm_model.n_components
    prob_cols = [f"regime_prob_{i}" for i in range(n_states)]

    total_checked, total_label_mismatch = 0, 0
    for asset in _CRYPTO5_ASSETS:
        asset_log = log[log["asset"] == asset]
        broken = asset_log[asset_log["regime_prob_4"].isna()]
        if broken.empty:
            print(f"[{asset}] no broken rows")
            continue
        print(f"[{asset}] {len(broken)} broken rows: {broken.index.min()} -> {broken.index.max()}")

        mismatches = 0
        for ts in broken.index:
            try:
                result = regime_service.get_live_regime_crypto5(asset, ts, log=False)
                probs = result.get("probabilities", {})
                if not result.get("live", False) or len(probs) < n_states:
                    continue
                recomputed_label = max(range(n_states), key=lambda i: probs.get(f"regime_{i}", 0.0))
                stored_label = int(log.loc[ts, "regime_label"])
                total_checked += 1
                if recomputed_label != stored_label:
                    mismatches += 1
                    total_label_mismatch += 1
            except Exception as exc:
                print(f"  {ts}: recompute failed ({exc})")
        print(f"[{asset}] recompute disagrees with the originally-logged regime_label "
              f"for {mismatches}/{len(broken)} rows")

    print(f"\nTotal: {total_label_mismatch}/{total_checked} recomputed rows disagree with "
          f"their own originally-logged regime_label — recompute is not a safe stand-in "
          f"for the missing probabilities. Leaving all {len(log[log['asset'].isin(_CRYPTO5_ASSETS) & log[prob_cols[-1]].isna()])} "
          f"NaN rows as-is.")


if __name__ == "__main__":
    main()
