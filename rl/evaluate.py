"""Walk-forward evaluation, metric computation, and per-regime breakdown for the PPO agent."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from rl.env import MARTATradingEnv, VAL_END, VAL_END_EXTENDED, VAL_START
from utils.logger import logger

_ANNUALIZE = np.sqrt(24 * 365)  # hourly → annual
_RESULTS_PATH = Path("data/results/ppo_evaluation.json")
_REGIME_BEHAVIOR_PATH = Path("data/results/regime_behavior_analysis.json")
_MULTI_ASSET_RESULTS_PATH = Path("data/results/multi_asset_evaluation.json")

# IN-SAMPLE windows used ONLY to inspect how the trained agent sizes positions across
# regimes during known crisis/bear episodes. These dates fall inside the training split
# (≤ 2020-12-31), so they are NOT performance evaluation — every result is labelled
# IN-SAMPLE and is never mixed with the validation metrics.
IN_SAMPLE_WINDOWS = [
    {
        "label": "March 2020 crash",
        "start": "2020-02-01",
        "end": "2020-04-30",
        "note": "IN-SAMPLE — for regime behavior analysis only, not performance evaluation",
    },
    {
        "label": "2018 crypto bear",
        "start": "2018-11-01",
        "end": "2019-01-31",
        "note": "IN-SAMPLE — for regime behavior analysis only",
    },
]


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    if len(portfolio_values) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_peak - portfolio_values) / np.maximum(running_peak, 1e-8)
    return float(np.max(drawdowns))


# Position change below this fraction is a micro-adjustment from the continuous Beta
# head, not a meaningful trade. Used to separate real trades from the >1e-9 noise floor.
_MEANINGFUL_TRADE_THRESHOLD = 0.005


def count_position_changes(positions: np.ndarray, threshold: float = 1e-9) -> int:
    """Number of timesteps where the signed position changed by more than ``threshold``.

    The default 1e-9 counts every micro-adjustment (kept for backward compatibility);
    pass ``threshold=_MEANINGFUL_TRADE_THRESHOLD`` for a meaningful trade count.
    """
    if len(positions) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(positions)) > threshold))


def mean_turnover_per_step(positions: np.ndarray) -> float:
    """Mean absolute change in signed position per step (fraction of portfolio traded)."""
    if len(positions) < 2:
        return 0.0
    return float(np.abs(np.diff(positions)).mean())


def compute_metrics(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    positions: np.ndarray | None = None,
) -> dict[str, float]:
    """Sharpe, Sortino, max drawdown, Calmar, total return, trade count."""
    returns = np.asarray(returns, dtype=np.float64)
    portfolio_values = np.asarray(portfolio_values, dtype=np.float64)

    if len(returns) == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "calmar": 0.0, "total_return": 0.0, "n_trades": 0,
            "n_meaningful_trades": 0, "mean_turnover": 0.0,
            "hold_ratio": 0.0, "mean_trade_duration_hours": 0.0,
            "median_trade_duration_hours": 0.0,
        }

    sharpe = float(returns.mean() / (returns.std() + 1e-8) * _ANNUALIZE)

    downside = returns[returns < 0]
    sortino_denom = (downside.std() + 1e-8) if len(downside) > 0 else 1e-8
    sortino = float(returns.mean() / sortino_denom * _ANNUALIZE)

    max_dd = compute_max_drawdown(portfolio_values)
    calmar = float(returns.mean() * 24 * 365 / (max_dd + 1e-8))
    total_return = float(portfolio_values[-1] / portfolio_values[0] - 1.0)
    if positions is not None:
        n_trades = count_position_changes(positions)
        n_meaningful = count_position_changes(positions, _MEANINGFUL_TRADE_THRESHOLD)
        turnover = mean_turnover_per_step(positions)
        hold_ratio = float(np.mean(np.abs(positions) < _MEANINGFUL_TRADE_THRESHOLD))
        trade_durations = _compute_trade_durations(positions, _MEANINGFUL_TRADE_THRESHOLD)
    else:
        n_trades = n_meaningful = 0
        turnover = 0.0
        hold_ratio = 0.0
        trade_durations = []

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "total_return": total_return,
        "n_trades": n_trades,  # >1e-9 changes (includes Beta micro-adjustments)
        "n_meaningful_trades": n_meaningful,  # > 0.5% position changes
        "mean_turnover": turnover,
        "hold_ratio": hold_ratio,  # fraction of steps with |position| < meaningful-trade threshold
        "mean_trade_duration_hours": float(np.mean(trade_durations)) if trade_durations else 0.0,
        "median_trade_duration_hours": float(np.median(trade_durations)) if trade_durations else 0.0,
    }


def _compute_trade_durations(positions: np.ndarray, threshold: float) -> list[int]:
    """Length (in steps/hours) of each contiguous run where |position| >= threshold."""
    in_trade = False
    trade_start = 0
    durations: list[int] = []
    for i, pos in enumerate(positions):
        if abs(pos) >= threshold and not in_trade:
            in_trade = True
            trade_start = i
        elif abs(pos) < threshold and in_trade:
            in_trade = False
            durations.append(i - trade_start)
    if in_trade:
        durations.append(len(positions) - trade_start)
    return durations


def run_walk_forward(
    model: BaseAlgorithm,
    env: MARTATradingEnv,
    deterministic: bool = True,
) -> dict[str, np.ndarray]:
    """Run the agent sequentially through ``env`` (one full pass) and collect a trajectory.

    ``env`` must be a raw :class:`MARTATradingEnv` with ``sequential=True``.
    """
    obs, info = env.reset()
    portfolio_values = [info["portfolio_value"]]
    positions = [info["signed_position"]]
    regimes = [info["current_regime"]]
    decision_regimes = [info.get("decision_regime", info["current_regime"])]
    returns: list[float] = []
    pnl: list[float] = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["signed_position"])
        regimes.append(info["current_regime"])
        decision_regimes.append(info.get("decision_regime", info["current_regime"]))
        returns.append(info["pnl_return"])
        pnl.append(info["pnl"])
        done = terminated or truncated

    return {
        "portfolio_values": np.asarray(portfolio_values, dtype=np.float64),
        "positions": np.asarray(positions, dtype=np.float64),
        "regimes": np.asarray(regimes, dtype=np.int64),
        "decision_regimes": np.asarray(decision_regimes, dtype=np.int64),
        "returns": np.asarray(returns, dtype=np.float64),
        "pnl": np.asarray(pnl, dtype=np.float64),
    }


def validation_sharpe(model: BaseAlgorithm, val_env: MARTATradingEnv) -> float:
    """Annualised Sharpe of the agent over a single sequential validation pass."""
    traj = run_walk_forward(model, val_env, deterministic=True)
    if len(traj["returns"]) < 2:
        return 0.0
    r = traj["returns"]
    return float(r.mean() / (r.std() + 1e-8) * _ANNUALIZE)


def per_regime_breakdown(traj: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per-regime Sharpe and average position size — the key conditioning result.

    Positions are credited to the *decision-time* regime (the one that drove the position
    scaling), not the post-step regime. These differ when regime labels interleave
    bar-to-bar (e.g. BTC R1/R4 in the 2021 val window); attributing to the post-step regime
    would invert the per-regime position sizes relative to what the policy actually did.
    """
    # decision_regimes / positions / returns index from step 1 (step-aligned).
    regimes = traj.get("decision_regimes", traj["regimes"])[1:]
    returns = traj["returns"]
    positions = np.abs(traj["positions"][1:])
    # Meaningful position-change flags, aligned to `regimes`/`returns` (step-aligned diffs).
    trade_flags = np.abs(np.diff(traj["positions"])) > _MEANINGFUL_TRADE_THRESHOLD
    out: dict[str, dict[str, float]] = {}
    for regime in range(7):
        mask = regimes == regime
        n = int(mask.sum())
        if n == 0:
            continue
        r = returns[mask]
        sharpe = float(r.mean() / (r.std() + 1e-8) * _ANNUALIZE) if n > 1 else 0.0
        out[f"regime_{regime}"] = {
            "n_steps": n,
            "n_meaningful_trades": int(trade_flags[mask].sum()),
            "sharpe": sharpe,
            "avg_position_size": float(positions[mask].mean()),
            "avg_return": float(r.mean()),
        }
    return out


def analyze_regime_behavior(
    model: BaseAlgorithm,
    env_factory: Any,
    config: dict,
    asset: str = "BTC_USDT",
) -> dict[str, Any]:
    """Inspect the trained agent's position sizing per regime on IN-SAMPLE crisis windows.

    For each window in :data:`IN_SAMPLE_WINDOWS` the agent is run deterministically over the
    window's bars and we report avg position per (decision-time) regime, the number of regime
    transitions, and the position changes that occur at regime boundaries.

    These are IN-SAMPLE diagnostics — the windows lie inside the training split, so this is a
    behavioural probe, NOT a performance claim. Results never enter the validation metrics.
    Saved to ``data/results/regime_behavior_analysis.json``.
    """
    logger.info("Analyzing IN-SAMPLE regime behavior (NOT a performance evaluation)")

    out: dict[str, Any] = {
        "_disclaimer": (
            "IN-SAMPLE windows inside the training split — regime behaviour analysis only. "
            "These are NOT performance metrics and must never be compared to validation results."
        ),
        "asset": asset,
        "attribution": "positions credited to decision-time regime",
        "windows": {},
    }

    for w in IN_SAMPLE_WINDOWS:
        # split="all" so train-period bars are valid; restrict to the window's index range.
        env = env_factory(assets=[asset], split="all", sequential=True, zero_regime=False)
        idx = env._index[asset]
        start = pd.Timestamp(w["start"], tz="UTC")
        end = pd.Timestamp(w["end"] + " 23:59:59", tz="UTC")
        in_window = np.flatnonzero((idx >= start) & (idx <= end))
        valid = np.intersect1d(env._valid_starts[asset], in_window)

        if len(valid) < 2:
            out["windows"][w["label"]] = {
                "note": w["note"], "start": w["start"], "end": w["end"],
                "error": "insufficient in-sample data in window",
            }
            logger.warning(f"[regime-behavior] '{w['label']}' has <2 valid bars — skipped")
            continue

        # Sequential traversal runs contiguously from valid[0] to valid[-1].
        env._valid_starts[asset] = valid
        traj = run_walk_forward(model, env, deterministic=True)

        regimes = traj.get("decision_regimes", traj["regimes"])[1:]
        positions = np.abs(traj["positions"][1:])
        returns = traj["returns"]

        # avg position per regime
        avg_position_by_regime: dict[str, dict[str, float]] = {}
        for regime in range(7):
            mask = regimes == regime
            n = int(mask.sum())
            if n == 0:
                continue
            avg_position_by_regime[f"regime_{regime}"] = {
                "n_steps": n,
                "avg_position_size": float(positions[mask].mean()),
                "avg_return": float(returns[mask].mean()) if n > 0 else 0.0,
            }

        # regime transitions + position changes at boundaries
        transitions = int(np.sum(np.diff(regimes) != 0)) if len(regimes) > 1 else 0
        boundary_changes: list[dict[str, Any]] = []
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1]:
                boundary_changes.append({
                    "from_regime": int(regimes[i - 1]),
                    "to_regime": int(regimes[i]),
                    "position_before": float(positions[i - 1]),
                    "position_after": float(positions[i]),
                    "position_delta": float(positions[i] - positions[i - 1]),
                })
        mean_abs_boundary_delta = (
            float(np.mean([abs(c["position_delta"]) for c in boundary_changes]))
            if boundary_changes else 0.0
        )

        out["windows"][w["label"]] = {
            "note": w["note"],
            "start": w["start"],
            "end": w["end"],
            "in_sample": True,
            "n_steps": int(len(regimes)),
            "regimes_present": sorted({int(r) for r in regimes}),
            "regime_transitions": transitions,
            "avg_position_by_regime": avg_position_by_regime,
            "mean_abs_position_change_at_boundaries": mean_abs_boundary_delta,
            "n_boundary_changes": len(boundary_changes),
            "boundary_changes": boundary_changes,
        }
        logger.success(
            f"[regime-behavior] '{w['label']}': {len(regimes)} steps, "
            f"{transitions} regime transitions, regimes {out['windows'][w['label']]['regimes_present']}"
        )

    _REGIME_BEHAVIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGIME_BEHAVIOR_PATH.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"Regime behavior analysis written to {_REGIME_BEHAVIOR_PATH}")
    return out


def _buy_and_hold(
    env: MARTATradingEnv,
    asset: str = "BTC_USDT",
    val_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Buy-and-hold baseline over the validation window for ``asset``."""
    returns = env._returns[asset]
    index = env._index[asset]
    end = val_end or VAL_END
    val_mask = (index >= VAL_START) & (index <= end)
    r = np.nan_to_num(returns[np.asarray(val_mask)])
    pv = env.initial_capital * np.cumprod(1.0 + r)
    pv = np.concatenate([[env.initial_capital], pv])
    metrics = compute_metrics(r, pv, positions=np.ones(len(pv)))
    return metrics


def _make_val_env(env_factory: Any, val_end: pd.Timestamp | None, **kwargs) -> MARTATradingEnv:
    """Build a validation env and inject a val_end override when provided."""
    env = env_factory(split="val", sequential=True, **kwargs)
    if val_end is not None:
        env._val_end_override = val_end
        # Rebuild valid_starts with the new mask
        for asset in list(env._valid_starts.keys()):
            env._valid_starts[asset] = _recompute_valid_starts(env, asset)
    return env


def _recompute_valid_starts(env: MARTATradingEnv, asset: str) -> np.ndarray:
    ts = env._index[asset]
    n = len(ts)
    in_split = env._split_mask(ts)
    valid = np.zeros(n, dtype=bool)
    for t in range(env.window_size, n - 1):
        if in_split[t] and in_split[t + 1]:
            valid[t] = True
    return np.flatnonzero(valid)


def evaluate_ppo(
    model: BaseAlgorithm,
    env_factory: Any,
    config: dict,
    val_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Full walk-forward evaluation on the validation set with per-regime + baselines.

    ``env_factory(**kwargs)`` builds a fresh sequential validation env; this lets us
    construct the regime-ablated (zeros) variant for the no-regime baseline.

    Parameters
    ----------
    val_end:
        Override the validation end date. Defaults to ``VAL_END`` (2023-12-31).
    """
    end_label = str(val_end or VAL_END)[:10]
    start_label = str(VAL_START)[:10]
    logger.info(f"Evaluating PPO on validation set ({start_label} .. {end_label})")

    val_env = _make_val_env(env_factory, val_end, zero_regime=False)
    traj = run_walk_forward(model, val_env, deterministic=True)
    metrics = compute_metrics(traj["returns"], traj["portfolio_values"], traj["positions"])
    regime_breakdown = per_regime_breakdown(traj)

    bh = _buy_and_hold(val_env, asset="BTC_USDT", val_end=val_end)

    no_regime_env = _make_val_env(env_factory, val_end, zero_regime=True)
    traj_nr = run_walk_forward(model, no_regime_env, deterministic=True)
    metrics_nr = compute_metrics(
        traj_nr["returns"], traj_nr["portfolio_values"], traj_nr["positions"]
    )

    result = {
        "marta_ppo": metrics,
        "per_regime": regime_breakdown,
        "baselines": {
            "buy_and_hold_btc": bh,
            "ppo_no_regime": metrics_nr,
        },
        "n_val_steps": int(len(traj["returns"])),
        "final_portfolio_value": float(traj["portfolio_values"][-1]),
        "val_window": {"start": start_label, "end": end_label},
    }

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float))
    logger.success(f"Evaluation written to {_RESULTS_PATH}")
    return result


def evaluate_multi_asset(
    models: list[BaseAlgorithm],
    env_factory: Any,
    config: dict,
    assets: list[str] | None = None,
    val_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Evaluate across multiple seeds × multiple assets and report mean±std Sharpe.

    Each (model, asset) pair is one independent test path.  The function returns
    per-path metrics plus an aggregate summary for the thesis.

    Parameters
    ----------
    models:
        List of PPO checkpoints (different random seeds).
    assets:
        List of assets to evaluate each model on.  Defaults to ["BTC_USDT", "ETH_USDT"].
    val_end:
        Override validation end date.  Use ``VAL_END_EXTENDED`` for the full two-year window.
    """
    if assets is None:
        assets = ["BTC_USDT", "ETH_USDT"]

    end_label = str(val_end or VAL_END)[:10]
    start_label = str(VAL_START)[:10]
    logger.info(
        f"Multi-asset/seed evaluation: {len(models)} seeds × {len(assets)} assets "
        f"(val {start_label} → {end_label})"
    )

    paths: list[dict[str, Any]] = []
    sharpes: list[float] = []

    for seed_idx, model in enumerate(models):
        for asset in assets:
            env = _make_val_env(
                env_factory, val_end,
                assets=[asset], zero_regime=False,
            )
            if asset not in env._valid_starts or len(env._valid_starts[asset]) == 0:
                logger.warning(f"Seed {seed_idx} / {asset}: no valid val-split rows — skipping")
                continue

            traj = run_walk_forward(model, env, deterministic=True)
            m = compute_metrics(traj["returns"], traj["portfolio_values"], traj["positions"])
            regime_bd = per_regime_breakdown(traj)
            bh = _buy_and_hold(env, asset=asset, val_end=val_end)

            path_result = {
                "seed": seed_idx,
                "asset": asset,
                "metrics": m,
                "per_regime": regime_bd,
                "buy_hold": bh,
                "n_steps": int(len(traj["returns"])),
            }
            paths.append(path_result)
            sharpes.append(m["sharpe"])
            logger.info(
                f"Seed {seed_idx} / {asset}: Sharpe={m['sharpe']:.3f} "
                f"MaxDD={m['max_drawdown']:.3f} Return={m['total_return']:.3%}"
            )

    aggregate: dict[str, Any] = {"n_paths": len(paths)}
    if sharpes:
        aggregate["sharpe_mean"] = statistics.mean(sharpes)
        aggregate["sharpe_std"]  = statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0
        aggregate["sharpe_min"]  = min(sharpes)
        aggregate["sharpe_max"]  = max(sharpes)
        logger.info(
            f"Aggregate Sharpe: {aggregate['sharpe_mean']:.3f} ± {aggregate['sharpe_std']:.3f} "
            f"[{aggregate['sharpe_min']:.3f}, {aggregate['sharpe_max']:.3f}]"
        )

    result = {
        "paths": paths,
        "aggregate": aggregate,
        "seeds": len(models),
        "assets": assets,
        "val_window": {"start": start_label, "end": end_label},
    }
    _MULTI_ASSET_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MULTI_ASSET_RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float))
    logger.success(f"Multi-asset evaluation written to {_MULTI_ASSET_RESULTS_PATH}")
    return result
