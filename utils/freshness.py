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

from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.assets import aligned_path

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
_PROCESSED = Path("data/processed")
_ENCODER_PATH = Path("encoder/checkpoints/best_encoder.pt")
_REGIME_STREAM_PATH = _PROCESSED / "regime_stream.parquet"

DEBATE_ASSETS: list[str] = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]

# Crypto-only panel (2026-08 regime-layer rebuild) — the validated, preferred config
# (raw 22-col features, no TFT encoder/PCA in the loop; see rapport_pfe results.tex and
# data/validated/regime_semantics_crypto5_multihorizon.json). Pass these explicitly to
# load_debate_data() (and the matching kwargs to every agent constructed alongside it —
# see run_debate.py) to use the crypto-only panel instead of the production default.
CRYPTO5_PROCESSED = Path("data/processed/crypto5_multihorizon")
CRYPTO5_SEMANTICS_PATH = Path("data/validated/regime_semantics_crypto5_multihorizon.json")
CRYPTO5_DEBATE_ASSETS: list[str] = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]


class StaleRegimeStreamError(RuntimeError):
    """regime_stream.parquet predates the deployed encoder checkpoint."""


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
    regime_stream = pd.read_parquet(processed_dir / "regime_stream.parquet", engine="pyarrow")
    aligned_dfs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        path = aligned_path(asset, processed_dir)
        if path.exists():
            aligned_dfs[asset] = pd.read_parquet(path, engine="pyarrow")
    return regime_stream, aligned_dfs
