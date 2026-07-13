"""PPO inference service for the MARTA dashboard live signal.

Loads ppo_best.zip + the frozen TFT encoder at startup, then builds a full
152-dim observation vector (regime embedding | consensus zeros | portfolio
zeros) to drive model.predict() for the live signal endpoint.

Falls back gracefully to the rule-based _REGIME_POLICY when the checkpoint
or encoder is absent (e.g. first-run before any training).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

_CHECKPOINT = MARTA_ROOT / "rl" / "checkpoints" / "ppo_best.zip"
_ENCODER_CKPT = MARTA_ROOT / "encoder" / "checkpoints" / "best_encoder.pt"
_REGIME_STREAM_PATH = MARTA_ROOT / "data" / "processed" / "regime_stream.parquet"
_ALIGNED_PATTERN = str(MARTA_ROOT / "data" / "processed" / "{asset}_normalized.parquet")

_WINDOW_SIZE = 60
_REGIME_DIM = 128
_CONSENSUS_DIM = 16
_PORTFOLIO_DIM = 8
_STATE_DIM = _REGIME_DIM + _CONSENSUS_DIM + _PORTFOLIO_DIM  # 152

_model = None
_encoder = None
_regime_stream = None
_aligned_cache: dict[str, "pd.DataFrame"] = {}


def load_model() -> None:
    """Load ppo_best.zip and the frozen encoder. Called once at backend startup."""
    global _model, _encoder, _regime_stream

    # Load regime stream (used for obs construction)
    if _REGIME_STREAM_PATH.exists():
        try:
            import pandas as pd
            _regime_stream = pd.read_parquet(_REGIME_STREAM_PATH, engine="pyarrow")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"ppo_inference: could not load regime stream — {exc}")

    if not _CHECKPOINT.exists():
        import logging
        logging.getLogger(__name__).info(
            f"ppo_inference: {_CHECKPOINT} not found — falling back to regime_policy rules"
        )
        return

    import logging as _logging
    _log = _logging.getLogger(__name__)

    try:
        import sys
        sys.path.insert(0, str(MARTA_ROOT))
        import torch
        from stable_baselines3 import PPO
        from encoder.model import TFTRegimeEncoder
        from rl.policy import MARTAActorCriticPolicy

        # Prefer ONNX encoder (no PyTorch overhead, sub-second CPU inference).
        # Fall back to PyTorch if the ONNX file hasn't been exported yet.
        _ONNX_CKPT = MARTA_ROOT / "encoder" / "checkpoints" / "encoder.onnx"
        try:
            from encoder.encoder_onnx import OnnxEncoder
            _encoder = OnnxEncoder(_ONNX_CKPT)
            _log.info("ppo_inference: using ONNX encoder")
        except (FileNotFoundError, ImportError):
            if _ENCODER_CKPT.exists():
                enc = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
                ckpt = torch.load(_ENCODER_CKPT, map_location="cpu", weights_only=True)
                enc.load_state_dict(ckpt["model_state"])
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False
                _encoder = enc
                _log.info("ppo_inference: using PyTorch encoder (ONNX not available)")

        # Load PPO
        _model = PPO.load(
            str(_CHECKPOINT),
            device="cpu",
            custom_objects={"policy_class": MARTAActorCriticPolicy},
        )
        _log.info("ppo_inference: loaded ppo_best.zip + encoder — live signal will use model.predict()")

    except Exception as exc:
        _log.warning(f"ppo_inference: could not load PPO model ({exc}) — falling back to regime_policy rules")
        _model = None
        _encoder = None


def _get_aligned_df(asset: str) -> Optional["pd.DataFrame"]:
    if asset in _aligned_cache:
        return _aligned_cache[asset]
    path = Path(_ALIGNED_PATTERN.format(asset=asset))
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path, engine="pyarrow")
        _aligned_cache[asset] = df
        return df
    except Exception:
        return None


def _build_regime_embedding(asset: str, timestamp: "pd.Timestamp") -> np.ndarray:
    """Build 128-dim regime embedding from the 60-candle window ending at `timestamp`."""
    if _encoder is None:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    df = _get_aligned_df(asset)
    if df is None or df.empty:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    import torch
    import pandas as pd

    ts_utc = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")
    idx_tz = df.index.tz
    ts_local = ts_utc.tz_convert(idx_tz) if idx_tz else ts_utc.tz_localize(None)

    df_sorted = df.sort_index()
    loc = df_sorted.index.searchsorted(ts_local, side="right") - 1
    if loc < _WINDOW_SIZE:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    window = df_sorted.iloc[loc - _WINDOW_SIZE + 1 : loc + 1].values.astype(np.float32)
    if window.shape[0] < _WINDOW_SIZE:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    try:
        from encoder.encoder_onnx import OnnxEncoder
        if isinstance(_encoder, OnnxEncoder):
            return _encoder.encode(window).astype(np.float32)
        # PyTorch path
        import torch
        x = torch.from_numpy(window).unsqueeze(0)  # (1, 60, 14)
        with torch.no_grad():
            emb = _encoder.encode(x)  # (1, 128)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(_REGIME_DIM, dtype=np.float32)


def _build_observation(asset: str, timestamp: "pd.Timestamp") -> np.ndarray:
    """Build the 152-dim observation for PPO inference.

    Consensus and portfolio slots are zero (no running position, no fresh debate).
    The regime embedding carries the signal.
    """
    regime = _build_regime_embedding(asset, timestamp)
    consensus = np.zeros(_CONSENSUS_DIM, dtype=np.float32)
    portfolio = np.zeros(_PORTFOLIO_DIM, dtype=np.float32)
    obs = np.concatenate([regime, consensus, portfolio])
    return np.clip(obs, -10.0, 10.0).astype(np.float32)


def predict(asset: str, timestamp: "pd.Timestamp") -> Optional[dict]:
    """Run model.predict() on the current observation.

    Returns {"action": "long"|"short"|"hold", "position_size": float, "source": "ppo"}
    or None if the model is not loaded.
    """
    if _model is None:
        return None

    obs = _build_observation(asset, timestamp)
    try:
        action_arr, _ = _model.predict(obs, deterministic=True)
        # Unflatten: action[0]=direction (0=long,1=short,2=hold), action[1]=fraction
        direction = int(round(float(np.clip(action_arr[0], 0, 2))))
        fraction = float(np.clip(action_arr[1], 0.01, 0.99))
        action_map = {0: "long", 1: "short", 2: "hold"}
        return {
            "action": action_map[direction],
            "position_size": round(fraction, 4),
            "source": "ppo",
        }
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"ppo_inference.predict failed: {exc}")
        return None


def is_loaded() -> bool:
    return _model is not None
