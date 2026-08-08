"""PPO inference service for the MARTA dashboard live signal.

Loads ppo_best.zip + the frozen TFT encoder at startup, then builds a full
156-dim observation vector (regime embedding | consensus | portfolio |
asset-ID one-hot) to drive model.predict() for the live signal endpoint.

Falls back gracefully to the rule-based _REGIME_POLICY when the checkpoint
or encoder is absent (e.g. first-run before any training).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

_CHECKPOINT = MARTA_ROOT / "rl" / "checkpoints" / "ppo_best.zip"
_ENCODER_CKPT = MARTA_ROOT / "encoder" / "checkpoints" / "best_encoder.pt"
_REGIME_STREAM_PATH = MARTA_ROOT / "data" / "processed" / "regime_stream.parquet"
_ALIGNED_PATTERN = str(MARTA_ROOT / "data" / "processed" / "{asset}_normalized.parquet")
# Live-extended file (see live_feature_bridge.py) — a superset of the training-time file,
# kept current for all 4 assets via the hourly scheduler tick (Twelve Data for EURUSD/GOLD,
# Binance for BTC/ETH). Preferred over _ALIGNED_PATTERN when present; falls back to it for
# any asset whose live bridge hasn't run yet (e.g. a fresh deploy before the first tick).
_ALIGNED_LIVE_PATTERN = str(MARTA_ROOT / "data" / "processed" / "{asset}_normalized_live.parquet")
_DEBATE_RESULTS_PATH = MARTA_ROOT / "data" / "debates" / "debate_results.json"

_WINDOW_SIZE = 60
_REGIME_DIM = 128
_CONSENSUS_DIM = 16
_PORTFOLIO_DIM = 8
_ASSET_ID_DIM = 4  # one-hot over rl.env.CANONICAL_ASSETS — added 2026-08-08, see rl/env.py
_STATE_DIM = _REGIME_DIM + _CONSENSUS_DIM + _PORTFOLIO_DIM + _ASSET_ID_DIM  # 156
_NEAREST_DEBATE_MAX_DAYS = 7  # mirrors rl/ablation.py::_NEAREST_DEBATE_MAX_DAYS

_BIAS_SIGN = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
_ACTION_SIGN = {"long": 1.0, "short": -1.0, "hold": 0.0}

_model = None
_encoder = None
_regime_stream = None
_aligned_cache: dict[str, tuple[Path, float, "pd.DataFrame"]] = {}

_vol_cache: dict[str, tuple[float, "datetime"]] = {}  # asset -> (vol_forecast, fitted_at)
_VOL_REFIT_HOURS = 168  # weekly — mirrors rl.env._GARCH_STRIDE

_ConsensusRecordCls = None  # lazily-built Pydantic model, see _get_consensus_record_cls()
_debate_cache: dict = {"mtime": None, "entries": {}}  # asset -> (DatetimeIndex, np.ndarray)
_debate_cache_lock = threading.Lock()


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


def _resolve_aligned_path(file_stem: str) -> Optional[Path]:
    """Prefer the live-bridge file when it exists — see _ALIGNED_LIVE_PATTERN above."""
    live_path = Path(_ALIGNED_LIVE_PATTERN.format(asset=file_stem))
    if live_path.exists():
        return live_path
    static_path = Path(_ALIGNED_PATTERN.format(asset=file_stem))
    return static_path if static_path.exists() else None


def _get_aligned_df(asset: str) -> Optional["pd.DataFrame"]:
    """Cached by (path, mtime), not loaded once forever — the live-bridge file changes
    on disk every hour (live_feature_bridge.py, via the scheduler tick), and a backend
    process can live for days between restarts. Without mtime-based invalidation, the
    first load would win permanently and this whole live-bridge effort would silently do
    nothing for a long-running process. Also re-resolves the path each call so a fresh
    live-bridge file appearing after startup (e.g. first-ever backfill completes while the
    backend is already running) gets picked up without a restart."""
    from utils.assets import ASSET_FILE_STEM

    file_stem = ASSET_FILE_STEM.get(asset, asset)  # GOLD -> XAUUSD on disk
    path = _resolve_aligned_path(file_stem)
    if path is None:
        return None

    mtime = path.stat().st_mtime
    cached = _aligned_cache.get(asset)
    if cached is not None and cached[0] == path and cached[1] == mtime:
        return cached[2]

    try:
        import pandas as pd
        df = pd.read_parquet(path, engine="pyarrow")
        _aligned_cache[asset] = (path, mtime, df)
        return df
    except Exception:
        return None


def _embed_window(window: np.ndarray) -> np.ndarray:
    """Encode a single (60, 14) window -> (128,) embedding, ONNX or PyTorch, whichever
    is loaded. Shared by both the single-point and sequence embedding builders below."""
    from encoder.encoder_onnx import OnnxEncoder
    if isinstance(_encoder, OnnxEncoder):
        return _encoder.encode(window).astype(np.float32)
    import torch
    x = torch.from_numpy(window).unsqueeze(0)  # (1, 60, 14)
    with torch.no_grad():
        emb = _encoder.encode(x)  # (1, 128)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def _locate(asset: str, timestamp: "pd.Timestamp") -> tuple[Optional["pd.DataFrame"], int]:
    """Sorted aligned df + integer row position of the last candle at/before
    `timestamp`. Returns (None, -1) if unavailable."""
    df = _get_aligned_df(asset)
    if df is None or df.empty:
        return None, -1
    import pandas as pd
    ts_utc = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")
    idx_tz = df.index.tz
    ts_local = ts_utc.tz_convert(idx_tz) if idx_tz else ts_utc.tz_localize(None)
    df_sorted = df.sort_index()
    loc = df_sorted.index.searchsorted(ts_local, side="right") - 1
    return df_sorted, loc


def _build_regime_embedding(asset: str, timestamp: "pd.Timestamp") -> np.ndarray:
    """Build 128-dim regime embedding from the 60-candle window ending at `timestamp`."""
    if _encoder is None:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    df_sorted, loc = _locate(asset, timestamp)
    if df_sorted is None or loc < _WINDOW_SIZE:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    window = df_sorted.iloc[loc - _WINDOW_SIZE + 1 : loc + 1].values.astype(np.float32)
    if window.shape[0] < _WINDOW_SIZE:
        return np.zeros(_REGIME_DIM, dtype=np.float32)

    try:
        return _embed_window(window)
    except Exception:
        return np.zeros(_REGIME_DIM, dtype=np.float32)


def get_live_embedding_sequence(
    asset: str, timestamp: "pd.Timestamp", lookback: int = 168
) -> Optional[np.ndarray]:
    """Sequence of `lookback` consecutive live embeddings ending at `timestamp`, oldest
    first — i.e. (lookback, 128), not a single isolated point.

    Why a sequence: the fitted HMM's predict_proba() on a single frame is dominated by
    startprob_ (the model's learned initial-state prior), not the actual observation —
    verified empirically, every single-frame query returned whatever state startprob_
    favors regardless of input. Feeding a proper time-ordered sequence lets the
    transition matrix actually inform the estimate via the forward-backward algorithm,
    the way the model was fit and validated to be used (regime_stream.parquet itself
    was built from one predict_proba() call over the full ordered historical sequence,
    never single points). Returns None if there isn't enough history before `timestamp`.
    """
    if _encoder is None:
        return None

    df_sorted, loc = _locate(asset, timestamp)
    needed = lookback + _WINDOW_SIZE - 1
    if df_sorted is None or loc < needed - 1:
        return None

    start = loc - needed + 1
    block = df_sorted.iloc[start : loc + 1].values.astype(np.float32)  # (needed, 14)

    embeddings = np.zeros((lookback, _REGIME_DIM), dtype=np.float32)
    try:
        for i in range(lookback):
            window = block[i : i + _WINDOW_SIZE]
            embeddings[i] = _embed_window(window)
    except Exception:
        return None
    return embeddings


def get_live_embedding(asset: str, timestamp: "pd.Timestamp") -> Optional[np.ndarray]:
    """Public accessor for the live 128-dim regime embedding, so other services (e.g.
    regime_service's live regime inference) can reuse the encoder already loaded here
    instead of loading a second copy of it. Returns None if unavailable (encoder not
    loaded, no aligned data for this asset, or too little history before `timestamp`)
    — `_build_regime_embedding` signals the same cases with an all-zero vector.
    """
    emb = _build_regime_embedding(asset, timestamp)
    return emb if np.any(emb) else None


def _live_vol_forecast(asset: str) -> float:
    """Weekly-cached GARCH(1,1) 1-step volatility forecast for `asset`.

    Reuses rl.env._fit_garch_vol (the fitting primitive) directly rather than
    rl.env.precompute_vol_forecast — that wrapper's disk cache invalidates by
    `len(df) == len(cached)`, an invariant built for a static training dataframe that
    breaks for a live dataframe growing one row/hour (it would either refit on every
    call or never hit its own cache). This keeps its own small time-based cache instead:
    refit only when the last fit for `asset` is older than a week, else reuse the cached
    scalar. In-memory only — losing it on a backend restart just costs one extra refit,
    which self-heals.
    """
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    cached = _vol_cache.get(asset)
    if cached is not None and (now - cached[1]) < timedelta(hours=_VOL_REFIT_HOURS):
        return cached[0]

    df = _get_aligned_df(asset)
    if df is None or "close_return" not in df.columns:
        return 0.02
    returns = df["close_return"].dropna().to_numpy(dtype=np.float64)
    if len(returns) < 20:
        return 0.02

    try:
        from rl.env import _fit_garch_vol, _GARCH_WINDOW
        vol = _fit_garch_vol(returns[-_GARCH_WINDOW:])
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"ppo_inference: GARCH fit failed for {asset} — {exc}")
        vol = float(np.std(returns)) if len(returns) > 1 else 0.02

    _vol_cache[asset] = (vol, now)
    return vol


def _build_portfolio_vec(asset: str) -> np.ndarray:
    """Build the 8-dim live portfolio vector from paper_db state.

    Sourced from paper_db.get_portfolio_state_for_obs(), called before this tick's
    paper_db.update() runs (predict() executes before update() inside
    live_signal.get_live_signal()) — so this naturally reflects pre-decision state,
    matching rl/env.py's training-time semantics.
    """
    import logging
    vec = np.zeros(_PORTFOLIO_DIM, dtype=np.float32)
    try:
        import paper_db
        state = paper_db.get_portfolio_state_for_obs(asset)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"ppo_inference: portfolio state read failed for {asset} — {exc}")
        return vec

    from rl.env import CANONICAL_ASSETS, N_POSITION_SLOTS
    if asset in CANONICAL_ASSETS:
        slot = min(CANONICAL_ASSETS.index(asset), N_POSITION_SLOTS - 1)
        vec[slot] = state["signed_position"]
    vec[4] = state["portfolio_return"]
    vec[5] = state["drawdown"]
    hours = state["hours_since_last_trade"]
    # Never-traded asset reads as maximally idle (1.0), not "just traded" (0.0) — 0.0 is
    # only a reset-boundary artifact of rl/env.py's episodic reset(); a live asset that's
    # simply never opened a position is closer to the far end of the idle scale, which is
    # also how this dim behaves during real sustained hold streaks in training (saturates
    # at 1.0 after 100 steps of no trade).
    steps_proxy = 1.0 if hours is None else min(hours / 100.0, 1.0)
    vec[6] = steps_proxy
    vec[7] = _live_vol_forecast(asset)
    return vec


def _get_consensus_record_cls():
    """Lazily build (once) a narrow Pydantic validation target for consensus extraction.

    Deliberately narrower than the full agents.schemas.DebateResult — only the fields the
    consensus vector actually reads (timestamp, asset, round1_theses, fund_manager_decision)
    — so a record with a corrupted-but-irrelevant field (risk_limits, round2_rebuttals)
    doesn't get needlessly discarded. Reuses AgentThesis/FundManagerDecision from
    agents.schemas directly (single source of truth, stays in sync with the write-side
    schema automatically). Still catches every gap in the raw `.get()`-chain readers used
    elsewhere in the codebase: a present-but-null field (e.g. "conviction": null) is
    rejected by Field(ge=0,le=1) instead of silently passing through as None; an invalid
    enum value (e.g. "Bullish") is rejected by Literal[...] instead of silently mapping to
    0.0 via `bias_sign.get(x, 0.0)`.

    Imports are lazy (module-level cache, built once) so a process that never reaches this
    code path (fast mtime-cache-hit calls) never needs agents.schemas/pydantic imported.
    """
    global _ConsensusRecordCls
    if _ConsensusRecordCls is None:
        from pydantic import BaseModel
        from agents.schemas import AgentThesis, FundManagerDecision

        class _ConsensusRecord(BaseModel):
            timestamp: str
            asset: str
            round1_theses: list[AgentThesis]
            fund_manager_decision: FundManagerDecision

        _ConsensusRecordCls = _ConsensusRecord
    return _ConsensusRecordCls


def _consensus_vec_from_result(record) -> np.ndarray:
    """Exact port of rl/env.py::_build_consensus_lookup's per-record formula, operating on
    validated Pydantic attributes instead of raw dict .get() chains. Dims 9-15 stay zero —
    never populated in training either."""
    vec = np.zeros(_CONSENSUS_DIM, dtype=np.float32)
    for i, th in enumerate(record.round1_theses[:5]):
        vec[i] = _BIAS_SIGN.get(th.directional_bias, 0.0) * th.confidence
    fmd = record.fund_manager_decision
    vec[5] = _ACTION_SIGN.get(fmd.final_action, 0.0) * fmd.conviction
    vec[6] = fmd.position_size
    vec[7] = 1.0 if fmd.regime_override else 0.0
    vec[8] = float(len(fmd.dissenting_agents))
    return vec


def _load_debate_lookup() -> dict:
    """Pure read of an already-persisted file — NEVER triggers a live debate (no import of
    agents.graph.run_debate or any LLM/Ollama call anywhere in this function). Populated
    only by whatever cadence a human has manually triggered "Run Debate" through
    debate_stream.py; if nobody ever has, this returns {} and consensus stays all-zero,
    same as before this function existed.

    mtime-gated: re-reads/re-parses/re-validates debate_results.json only when the file
    actually changed since the last call. Steady-state cost per predict() call is one
    stat() + a dict lookup — well under the latency budget, and no live LLM call is ever
    on this path.
    """
    global _debate_cache
    import logging
    log = logging.getLogger(__name__)

    if not _DEBATE_RESULTS_PATH.exists():
        return {}
    try:
        mtime = _DEBATE_RESULTS_PATH.stat().st_mtime
    except OSError:
        return _debate_cache["entries"]
    if _debate_cache["mtime"] == mtime:
        return _debate_cache["entries"]  # fast path: no re-read, no re-parse, no re-validate

    with _debate_cache_lock:
        if _debate_cache["mtime"] == mtime:  # re-check after acquiring the lock
            return _debate_cache["entries"]

        try:
            raw = json.loads(_DEBATE_RESULTS_PATH.read_text())
        except Exception as exc:
            log.warning(f"ppo_inference: could not read/parse debate_results.json — {exc}")
            return _debate_cache["entries"]  # keep last-good cache, never crash predict()

        import pandas as pd

        record_cls = _get_consensus_record_cls()
        by_asset: dict[str, list[tuple["pd.Timestamp", np.ndarray]]] = {}
        n_skipped = 0
        for i, d in enumerate(raw):
            try:
                record = record_cls.model_validate(d)
                ts = pd.Timestamp(record.timestamp)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                vec = _consensus_vec_from_result(record)
            except Exception as exc:
                # Catches pydantic.ValidationError (missing/wrong-typed/out-of-range
                # fields) and any other malformed-record failure (bad timestamp, etc.) —
                # one bad record is logged and skipped, never aborts the whole reload.
                n_skipped += 1
                log.warning(f"ppo_inference: skipping malformed debate record #{i} — {exc}")
                continue
            by_asset.setdefault(record.asset, []).append((ts, vec))

        entries = {}
        for asset, items in by_asset.items():
            items.sort(key=lambda x: x[0])
            entries[asset] = (
                pd.DatetimeIndex([x[0] for x in items]),
                np.stack([x[1] for x in items]),
            )

        if n_skipped:
            log.warning(
                f"ppo_inference: {n_skipped}/{len(raw)} debate record(s) skipped as malformed on reload"
            )
        _debate_cache = {"mtime": mtime, "entries": entries}
        return entries


def _nearest_consensus_vec(
    asset: str, timestamp: "pd.Timestamp", max_days: int = _NEAREST_DEBATE_MAX_DAYS
) -> Optional[np.ndarray]:
    """Byte-for-byte port of rl/ablation.py::_make_nearest_consensus_fn's nearest-neighbor
    search, for consistency with the already-tested agents_no_regime ablation baseline
    rather than inventing a third freshness-window convention (live_signal.py already has
    two other ad hoc ones for a different purpose)."""
    entries = _load_debate_lookup()
    if asset not in entries:
        return None
    timestamps, vecs = entries[asset]

    import pandas as pd
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    pos = int(timestamps.searchsorted(ts))
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(timestamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(timestamps[i] - ts))
    if abs(timestamps[best] - ts) > pd.Timedelta(days=max_days):
        return None
    return vecs[best]


def _build_consensus_vec(asset: str, timestamp: "pd.Timestamp") -> np.ndarray:
    import logging
    try:
        vec = _nearest_consensus_vec(asset, timestamp)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"ppo_inference: consensus lookup failed for {asset} — {exc}")
        return np.zeros(_CONSENSUS_DIM, dtype=np.float32)
    return vec if vec is not None else np.zeros(_CONSENSUS_DIM, dtype=np.float32)


def _build_asset_id_vec(asset: str) -> np.ndarray:
    from rl.env import CANONICAL_ASSETS
    vec = np.zeros(_ASSET_ID_DIM, dtype=np.float32)
    if asset in CANONICAL_ASSETS:
        idx = CANONICAL_ASSETS.index(asset)
        if idx < _ASSET_ID_DIM:
            vec[idx] = 1.0
    return vec


def _build_observation(asset: str, timestamp: "pd.Timestamp") -> np.ndarray:
    """Build the 156-dim observation for PPO inference (see rl/env.py STATE_DIM).

    Consensus is a nearest-cached-debate lookup (never triggers a live debate); portfolio
    reflects live paper_db state (position, return, drawdown, idle time, vol forecast).
    """
    regime = _build_regime_embedding(asset, timestamp)
    consensus = _build_consensus_vec(asset, timestamp)
    portfolio = _build_portfolio_vec(asset)
    asset_id = _build_asset_id_vec(asset)
    obs = np.concatenate([regime, consensus, portfolio, asset_id])
    return np.clip(obs, -10.0, 10.0).astype(np.float32)


def predict(asset: str, timestamp: "pd.Timestamp") -> Optional[dict]:
    """Run model.predict() on the current observation.

    Returns {"action": "long"|"short"|"hold", "position_size": float, "source": "ppo"}
    or None if the model is not loaded.
    """
    if _model is None:
        return None

    try:
        obs = _build_observation(asset, timestamp)
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
