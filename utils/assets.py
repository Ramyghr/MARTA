"""Asset naming helpers.

GOLD is the system's *logical* asset identifier — used by the regime
stream, ``rl.env.CANONICAL_ASSETS``, the agents, and the debate demos. The data
pipeline, however, writes the aligned parquets under the histdata *instrument*
name (XAUUSD). This module performs the single translation between the
two, applied only where ``{asset}_aligned.parquet`` is read from disk, so the
rest of the system can keep speaking the logical names everywhere.
"""

from __future__ import annotations

from pathlib import Path

# logical asset id -> aligned-parquet file stem
ASSET_FILE_STEM: dict[str, str] = {"GOLD": "XAUUSD"}

# file stem -> logical asset id (reverse of ASSET_FILE_STEM). Needed anywhere that reads
# the COMBINED all_assets_aligned.parquet (MultiIndex columns keyed by file stem, e.g.
# "XAUUSD") and then has to look results up by logical name (e.g. agents/graph.py's
# aligned_dfs.get(ctx.asset), which always uses "GOLD"). This exact mismatch has recurred
# several times as separate bugs across the backend/agents — centralising the reverse
# lookup here so future callers don't reimplement (and mis-implement) it each time.
_FILE_STEM_TO_ASSET: dict[str, str] = {stem: logical for logical, stem in ASSET_FILE_STEM.items()}


def aligned_filename(asset: str) -> str:
    """Return the aligned-parquet filename for a logical asset id."""
    return f"{ASSET_FILE_STEM.get(asset, asset)}_aligned.parquet"


def aligned_path(asset: str, processed_dir: Path) -> Path:
    """Return the full aligned-parquet path for a logical asset id."""
    return Path(processed_dir) / aligned_filename(asset)


def logical_asset_name(file_stem: str) -> str:
    """Translate a file-stem column key (as found in all_assets_aligned.parquet's
    MultiIndex, e.g. "XAUUSD") back to the logical asset id used everywhere else
    ("GOLD"). No-op for asset ids that were never translated (e.g. "BTC_USDT")."""
    return _FILE_STEM_TO_ASSET.get(file_stem, file_stem)
