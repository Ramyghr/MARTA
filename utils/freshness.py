"""Shared data-freshness guards and the debate-generation data loader.

Moved out of rl/train.py (P0.4, 2026-08-11) so lightweight consumers don't have to
import rl/train.py's full SB3/torch training stack just to ask "is this artifact stale?"
The freshness check previously only protected rl/train.py's load_data() (used by
run_rl.py, scripts/evaluate_all_checkpoints.py, generate_equity_curve.py, and
regime_quality_audit.py) — the four debate-generation scripts each defined their own
~92%-identical _load_data() and none of them called it, so a stale regime_stream.parquet
could silently feed the debate cache without ever being caught.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.assets import aligned_path

# Same MARTA_ROOT anchoring as rl/train.py/rl/callbacks.py's identical constant — see
# rl/train.py's comment. This module is imported from dashboard/backend/ (a different
# cwd) as well as repo-root training/debate scripts, so a bare relative Path here is
# exactly the bug class documented throughout this project's history.
_MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent)))

# Kept at the old 4-asset, encoder+PCA-based production panel as the DEFAULT — dashboard
# code doesn't call this loader, but run_weekly_debates.py/run_clean_ablation_debates.py/
# run_validation_debates.py all call load_debate_data() with no arguments and then
# construct FundManagerAgent()/ContrarianAgent()/etc with THEIR OWN no-argument
# defaults (also still production, for the same dashboard-safety reason documented in
# agents/macro_agent.py etc.) — changing this default alone, without also updating those
# scripts' agent construction, would silently feed crypto5 data through agents still
# reading production semantics/regime streams, a worse and less consistent state than
# before. Crypto-only callers (run_debate.py) pass CRYPTO5_PROCESSED/CRYPTO5_DEBATE_ASSETS
# explicitly instead — see that file for the fully-wired example.
_PROCESSED = _MARTA_ROOT / "data/processed"
_ENCODER_PATH = _MARTA_ROOT / "encoder/checkpoints/best_encoder.pt"
_REGIME_STREAM_PATH = _PROCESSED / "regime_stream.parquet"

DEBATE_ASSETS: list[str] = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]

# Crypto-only panel (2026-08 regime-layer rebuild) — the validated, preferred config
# (raw 22-col features, no TFT encoder/PCA in the loop; see rapport_pfe results.tex and
# data/validated/regime_semantics_crypto5_multihorizon.json). Pass these explicitly to
# load_debate_data() (and the matching kwargs to every agent constructed alongside it —
# see run_debate.py) to use the crypto-only panel instead of the production default.
CRYPTO5_PROCESSED = _MARTA_ROOT / "data/processed/crypto5_multihorizon"
CRYPTO5_SEMANTICS_PATH = _MARTA_ROOT / "data/validated/regime_semantics_crypto5_multihorizon.json"
CRYPTO5_DEBATE_ASSETS: list[str] = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]


class StaleRegimeStreamError(RuntimeError):
    """regime_stream.parquet predates the deployed encoder checkpoint."""


def resolve_regime_stream_path(processed_dir: Path, prefer_causal: bool = True) -> Path:
    """Return the CAUSAL regime stream under `processed_dir` if one exists, else fall
    back to the leaky (forward-backward smoothed) one — logging a warning on fallback
    so a silent leak is at least visible.

    The canonical single interface every consumer should route through instead of
    hardcoding "regime_stream.parquet" itself (regime-layer audit, 2026-08-26+): plain
    regime_stream.parquet is built by regime/hmm.py's generate_regime_stream(), which
    calls hmmlearn's predict_proba() on an asset's ENTIRE history in one shot —
    forward-BACKWARD smoothed, so a 2019 row's probability is measurably influenced by
    2025 data (confirmed CRITICAL finding; documented root cause of obs_mode="raw_regime"
    underperforming obs_mode="raw" in the crypto5 RL ablation, docs/RESULTS_AND_METHODOLOGY.md
    Section 9). regime/causal_regime_stream.py's regime_stream_causal.parquet is the
    walk-forward, forward-filtering-only replacement — provably regime_t = f(x_{<=t}),
    never f(x_{<=T}). Not every processed_dir has a causal stream yet (built so far only
    for crypto5_multihorizon) — the fallback keeps this a safe drop-in for panels that
    don't, rather than a hard failure.
    """
    causal_path = processed_dir / "regime_stream_causal.parquet"
    if prefer_causal and causal_path.exists():
        return causal_path
    leaky_path = processed_dir / "regime_stream.parquet"
    if prefer_causal:
        # The --tag hint only makes sense for a tagged panel subdirectory (crypto5,
        # crypto5_multihorizon, btc_only, ...); the bare production data/processed/ isn't
        # structured as a --tag panel, so don't print a command that wouldn't actually work.
        # Compare by name, not the full path, so this is correct whether the caller passes
        # a relative Path("data/processed") or an absolute MARTA_ROOT/"data"/"processed".
        fix_hint = (
            "No causal generator exists yet for the production 4-asset panel."
            if processed_dir.name == "processed"
            else f"Run `regime/causal_regime_stream.py --tag {processed_dir.name}` to fix."
        )
        print(
            f"[freshness] WARNING: no causal regime stream at {causal_path} — falling back "
            f"to the leaky (forward-backward smoothed) {leaky_path}. Historical regime "
            f"labels read from this file may be influenced by future data relative to "
            f"their own timestamp. {fix_hint}"
        )
    return leaky_path


def check_encoder_regime_stream_freshness(
    encoder_path: Path = _ENCODER_PATH,
    regime_stream_path: Path = _REGIME_STREAM_PATH,
) -> None:
    """Raise if ``regime_stream.parquet`` was last built before the currently-deployed
    encoder checkpoint — i.e. the HMM/PCA/regime labels (and anything derived from them,
    including the debate consensus cache) were fit on embeddings from a DIFFERENT encoder
    than the one about to be used for training/eval/debate generation.

    This exact drift happened silently for 6 days (2026-07-14 -> 2026-07-20): a walk-forward
    retrain swapped in a new best_encoder.pt without regenerating embeddings/HMM/regime_stream
    to match. Nothing caught it until it was found by inspecting file mtimes by hand.
    """
    if not encoder_path.exists() or not regime_stream_path.exists():
        return  # the reads below raise a clearer FileNotFoundError on a missing file
    encoder_mtime = encoder_path.stat().st_mtime
    stream_mtime = regime_stream_path.stat().st_mtime
    if encoder_mtime > stream_mtime:
        raise StaleRegimeStreamError(
            f"{encoder_path} was modified {datetime.fromtimestamp(encoder_mtime)}, which is "
            f"AFTER {regime_stream_path} ({datetime.fromtimestamp(stream_mtime)}). The regime "
            f"stream was not regenerated since the encoder last changed — HMM/PCA/regime "
            f"labels and the debate consensus cache built on it reflect a stale embedding "
            f"space. Fix: `python regime/regenerate_embeddings.py` then `python run_regime.py` "
            f"before training or evaluating."
        )


def load_debate_data(
    processed_dir: Path = _PROCESSED,
    assets: list[str] = DEBATE_ASSETS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load the regime stream + per-asset aligned parquets for debate generation.

    Checks regime-stream freshness first (raises StaleRegimeStreamError if stale) when
    loading the default production panel — previously each of the four debate scripts
    loaded this data directly and skipped the check entirely. Shared by run_debate.py,
    run_weekly_debates.py, run_clean_ablation_debates.py, and run_validation_debates.py.
    Pass CRYPTO5_PROCESSED/CRYPTO5_DEBATE_ASSETS to load the crypto-only panel instead —
    that HMM is fit directly on raw 22-col features with no TFT encoder or PCA in the
    loop, so this check is skipped for it (comparing it against best_encoder.pt's mtime
    would be a category error, not a staleness signal).
    """
    if processed_dir == _PROCESSED:
        check_encoder_regime_stream_freshness()
    regime_stream = pd.read_parquet(resolve_regime_stream_path(processed_dir), engine="pyarrow")
    aligned_dfs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        path = aligned_path(asset, processed_dir)
        if path.exists():
            aligned_dfs[asset] = pd.read_parquet(path, engine="pyarrow")
    return regime_stream, aligned_dfs
