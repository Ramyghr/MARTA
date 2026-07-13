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


def aligned_filename(asset: str) -> str:
    """Return the aligned-parquet filename for a logical asset id."""
    return f"{ASSET_FILE_STEM.get(asset, asset)}_aligned.parquet"


def aligned_path(asset: str, processed_dir: Path) -> Path:
    """Return the full aligned-parquet path for a logical asset id."""
    return Path(processed_dir) / aligned_filename(asset)
