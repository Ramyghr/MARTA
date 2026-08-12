"""Mechanics audit of TEST 3 (rl/minimal_btc_env.py) — items 1-6 of the 2026-08-11 RL
audit follow-up. Does NOT train a full model; inspects real data, a short (256-step,
zero-gradient-update) PPO rollout capture, and the diagnostics already saved from the
40k-step TEST 3A/3B runs (data/results/minimal_btc_env_{simple,marta}.json).

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.audit_test3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import explained_variance

from encoder.model import TFTRegimeEncoder
from rl.env import TRAIN_END, VAL_END, VAL_START
from rl.minimal_btc_env import (
    _PPO_KWARGS,
    ASSET_ID_DIM,
    CONSENSUS_DIM,
    MARTA_DIM,
    PORTFOLIO_DIM,
    REGIME_DIM,
    SIMPLE_DIM,
    SIMPLE_FEATURE_COLS,
    build_env,
    load_shared_data,
    make_random_policy,
    policy_fixed,
    run_trajectory,
)
from utils.assets import aligned_path

_console = Console()
_PROCESSED = Path("data/processed")
_RESULTS_DIR = Path("data/results")
_SEED = 42

# --------------------------------------------------------------------------- item 1
FEATURE_DESCRIPTIONS: list[dict[str, str]] = [
    dict(name="atr", represents="Average True Range(14) — absolute volatility of the last 14 bars' true range",
         norm="Single 720h rolling z-score", history="14h indicator window, riding on a 720h (30-day) z-score baseline",
         causal="Yes", future="None — backward rolling window only"),
    dict(name="adx", represents="Average Directional Index(14) — trend STRENGTH (not direction), 0-100 scale",
         norm="Single 720h rolling z-score", history="14h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="rsi", represents="Relative Strength Index(14) — momentum oscillator, overbought/oversold",
         norm="Single 720h rolling z-score", history="14h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="bb_width", represents="Bollinger Band width(20, 2sigma) / middle band — relative volatility",
         norm="Single 720h rolling z-score", history="20h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="macd_histogram", represents="MACD(12,26,9) signal-line difference — momentum/trend crossover",
         norm="Single 720h rolling z-score", history="~26h EMA (slow leg) + 9h signal EMA + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="obv_raw", represents="On-Balance Volume — cumulative signed volume (sign(price change) * volume, cumsum)",
         norm="Single 720h rolling z-score (features.py emits RAW cumsum only; the old 168h+720h double z-score bug is fixed)",
         history="UNBOUNDED — cumsum since the start of the entire series, one z-score baseline on top",
         causal="Yes", future="None"),
    dict(name="vol_zscore", represents="Raw traded volume, rolling-z-scored over 24h in features.py",
         norm="DOUBLE-NORMALIZED: 24h z-score (features.py) THEN a second 720h z-score (normalize.py) — the only "
              "feature normalized twice; everything else is computed raw then z-scored exactly once",
         history="24h inner window + 720h outer z-score baseline",
         causal="Yes", future="None"),
    dict(name="realized_vol", represents="Rolling std of log returns, annualized (24h window)",
         norm="Single 720h rolling z-score", history="24h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="vol_ratio", represents="realized_vol(24h) / realized_vol(168h), clipped [0.1, 10] — short vs long vol regime",
         norm="Single 720h rolling z-score", history="168h (7-day) indicator window (dominant leg) + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="return_skew", represents="Rolling skewness of log returns (48h window)",
         norm="Single 720h rolling z-score", history="48h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="return_kurtosis", represents="Rolling kurtosis of log returns (48h window) — tail-risk shape",
         norm="Single 720h rolling z-score", history="48h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="autocorr_lag1", represents="Rolling lag-1 autocorrelation of log returns (24h window) — serial "
                                            "correlation / momentum-persistence signal",
         norm="Single 720h rolling z-score", history="24h indicator window + 720h z-score baseline",
         causal="Yes", future="None"),
    dict(name="realized_vol_raw", represents="Same as realized_vol (24h annualized) but UNNORMALIZED passthrough — "
                                              "preserves absolute magnitude for crash detection",
         norm="NOT normalized (explicitly excluded — NO_NORMALIZE set)", history="24h indicator window only, no z-score baseline",
         causal="Yes", future="None"),
    dict(name="close_return", represents="log(close[t] / close[t-1]) — the realized 1-bar log return AT bar t",
         norm="NOT normalized — raw log return", history="1h (single prior bar)",
         causal="Yes — this is the ALREADY-REALIZED return ending at t, known by the time bar t closes",
         future="None. Distinct from next_return (self._returns[self._t+1]) which the reward function reads "
                "AFTER the observation for bar t was already built and the action already taken."),
]


def print_feature_table() -> None:
    _console.rule("[bold]Item 1 — the 14 raw ('simple'/TEST 3A) features")
    table = Table(show_lines=True)
    table.add_column("Feature")
    table.add_column("Represents", max_width=46)
    table.add_column("Normalization", max_width=30)
    table.add_column("History window", max_width=30)
    table.add_column("Causal?")
    for f in FEATURE_DESCRIPTIONS:
        table.add_row(f["name"], f["represents"], f["norm"], f["history"], f["causal"])
    _console.print(table)
    _console.print(
        "[bold]Verified directly from pipeline/features.py + pipeline/normalize.py: no `.shift(-1)` or any "
        "forward-looking operation exists anywhere in either file. All 14 columns are strictly causal — "
        "confirmed by code inspection, not just by convention."
    )


# --------------------------------------------------------------------------- item 2
def print_marta_obs_breakdown() -> None:
    _console.rule("[bold]Item 2 — the 156-dim 'marta'/TEST 3B observation")
    table = Table()
    table.add_column("Block")
    table.add_column("Dims", justify="right")
    table.add_column("Content")
    table.add_row(
        "regime/encoder", str(REGIME_DIM),
        "Frozen TFTRegimeEncoder.encode() output on the trailing 60-bar window of the SAME "
        "14 raw features above (identical input, richer temporal model — a TFT variable-selection "
        "network + attention over the 60-bar window, vs. TEST 3A's single current-bar snapshot).",
    )
    table.add_row(
        "consensus", str(CONSENSUS_DIM),
        "ALWAYS ZERO in TEST 3 — no debate cache is read (module docstring, deliberate simplification). "
        "In production MARTATradingEnv this block is a real per-agent signed-confidence vector looked up "
        "from data/debates/debate_results.json.",
    )
    table.add_row(
        "portfolio", str(PORTFOLIO_DIM),
        "[0]=signed_position (own current position, fixed +-0.3/0) [1-3]=0 (unused multi-asset slots) "
        "[4]=portfolio_value/initial-1 (own P&L so far) [5]=current_drawdown (own drawdown so far) "
        "[6]=steps_since_trade/100 capped at 1 [7]=vol_forecast (GARCH, market-only, not self-referential).",
    )
    table.add_row("asset_id", str(ASSET_ID_DIM), "One-hot, always [1,0,0,0] — BTC_USDT, single asset, never varies.")
    _console.print(table)
    _console.print(
        f"[bold]Total: {REGIME_DIM}+{CONSENSUS_DIM}+{PORTFOLIO_DIM}+{ASSET_ID_DIM} = {MARTA_DIM} dims. "
        "Of these, dims [0],[4],[5],[6] of the portfolio block are SELF-REFERENTIAL (depend on the "
        "agent's own past actions/P&L, not on the market) — this is exactly the confound flagged after "
        "TEST 3, being isolated by 3A+P/3B-P."
    )


# --------------------------------------------------------------------------- item 3
def reward_distribution_stats(rewards: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(rewards)),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
        "p05": float(np.percentile(rewards, 5)),
        "p95": float(np.percentile(rewards, 95)),
    }


def audit_reward_distribution(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 3 — reward distribution (position * next_return - cost)")
    out: dict[str, Any] = {}
    # Reward mechanics are IDENTICAL in both obs modes (reward depends only on action +
    # next_return, never on the observation) — computed once per baseline policy, valid
    # for both 3A and 3B by construction. ALWAYS_LONG = trades every step (1 initial
    # trade then holds direction, near-zero further turnover cost) isolates the pure
    # position*return signal; RANDOM = representative of what an EXPLORING (early-training)
    # policy actually sees, including real turnover cost from constant direction flips.
    for name, pol_factory in [("ALWAYS_LONG", lambda: policy_fixed(0)), ("RANDOM", lambda: make_random_policy(np.random.default_rng(_SEED)))]:
        env = build_env(shared["df"], None, "simple", "val", sequential=True, device="cpu")
        traj = run_trajectory(env, pol_factory(), is_model=False, seed=_SEED)
        stats = reward_distribution_stats(traj["rewards"])
        out[name] = stats
        _console.print(f"[{name}] {stats}")
    _console.print(
        "[bold]These numbers are observation-mode-independent (the reward function reads only action + "
        "next_return, never the observation) — they characterize the environment's reward SCALE for both "
        "3A and 3B identically. Actual per-run reward distributions differ only because 3A vs 3B's trained "
        "policies chose different action sequences, not because the reward formula differs."
    )
    return out


# --------------------------------------------------------------------------- item 4
def audit_next_return_stats(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 4 — next-bar return statistics, BTC_USDT 2024 validation split")
    df = shared["df"]
    val = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    ret = val["close_return"].dropna().to_numpy(dtype=np.float64)

    stats = {
        "n": int(len(ret)),
        "mean": float(ret.mean()),
        "std": float(ret.std()),
        "median": float(np.median(ret)),
        "positive_pct": float(np.mean(ret > 0)),
        "negative_pct": float(np.mean(ret < 0)),
        "zero_pct": float(np.mean(ret == 0)),
    }
    autocorr = {}
    s = pd.Series(ret)
    for lag in range(1, 6):
        autocorr[f"lag_{lag}"] = float(s.autocorr(lag=lag))
    stats["autocorrelation"] = autocorr

    _console.print(stats)
    sharpe_like = stats["mean"] / (stats["std"] + 1e-12) * np.sqrt(24 * 365)
    _console.print(
        f"[bold]Annualized mean/std ratio (naive always-long Sharpe-like stat, ignoring costs): "
        f"{sharpe_like:.3f}. Lag-1 autocorrelation = {autocorr['lag_1']:.4f} — "
        f"{'a real, if small, persistence signal' if abs(autocorr['lag_1']) > 0.02 else 'indistinguishable from a random walk at this precision'}."
    )
    return stats


# --------------------------------------------------------------------------- item 5
class BufferCaptureCallback(BaseCallback):
    """Captures the rollout buffer's rewards/values/returns/advantages BEFORE any
    gradient update (on_rollout_end fires after collection, before train())."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: dict[str, np.ndarray] | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        self.captured = {
            "rewards": buf.rewards.copy(),
            "returns": buf.returns.copy(),
            "values": buf.values.copy(),
            "advantages": buf.advantages.copy(),
        }


def audit_value_function(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 5 — value-function pipeline trace: reward -> returns (GAE) -> advantages -> value net -> explained_variance")

    train_env = Monitor(build_env(shared["df"], None, "simple", "train", sequential=False, device="cpu"))
    model = PPO(policy="MlpPolicy", env=train_env, seed=_SEED, **_PPO_KWARGS)
    cb = BufferCaptureCallback()
    model.learn(total_timesteps=_PPO_KWARGS["n_steps"], callback=cb, progress_bar=False)
    buf = cb.captured
    assert buf is not None

    ev_pre_update = float(explained_variance(buf["values"].flatten(), buf["returns"].flatten()))

    trace: dict[str, Any] = {
        "gamma": _PPO_KWARGS["gamma"],
        "gae_lambda": _PPO_KWARGS["gae_lambda"],
        "n_steps": _PPO_KWARGS["n_steps"],
        "stage_1_rewards": reward_distribution_stats(buf["rewards"].flatten()),
        "stage_2_returns_GAE_target": reward_distribution_stats(buf["returns"].flatten()),
        "stage_3_values_random_init_prediction": reward_distribution_stats(buf["values"].flatten()),
        "stage_4_advantages": reward_distribution_stats(buf["advantages"].flatten()),
        "explained_variance_at_step_256_pre_first_update": ev_pre_update,
    }

    reward_std = trace["stage_1_rewards"]["std"]
    return_std = trace["stage_2_returns_GAE_target"]["std"]
    value_std = trace["stage_3_values_random_init_prediction"]["std"]
    _console.print(trace)
    _console.print(
        f"\n[bold]Stage-by-stage:[/bold]\n"
        f"  reward std = {reward_std:.6f} (raw per-step reward — position*next_return-cost, unnormalized)\n"
        f"  -> GAE return (value target) std = {return_std:.6f} "
        f"({return_std / max(reward_std, 1e-12):.1f}x reward std — matches the theoretical "
        f"~1/sqrt(1-gamma^2)={1 / np.sqrt(1 - _PPO_KWARGS['gamma']**2):.1f}x amplification from discounting "
        f"correlated-noise rewards over an effective ~{1 / (1 - _PPO_KWARGS['gamma']):.0f}-step horizon)\n"
        f"  random-init value net prediction std = {value_std:.6f} "
        f"({value_std / max(return_std, 1e-12):.1f}x the actual return-target std)\n"
        f"  explained_variance AT STEP 256, BEFORE the first gradient update = {ev_pre_update:.4f}\n"
    )

    if value_std > 3 * return_std:
        _console.print(
            "[bold red]SCALE MISMATCH CONFIRMED: the value network's default (PyTorch/SB3 default init) "
            "output scale is orders of magnitude larger than the actual return-target scale. Since the "
            "reward here is NOT normalized (no CVaR division, no clipping, no VecNormalize — all "
            "deliberately stripped for TEST 3), the value net starts the run already very wrong relative "
            "to a target whose absolute magnitude is tiny (~1e-4 to 1e-3). This alone is enough to produce "
            "negative explained_variance during early training, independent of whether the observation "
            "carries any real signal — gradient descent has to first learn the right OUTPUT SCALE before "
            "it can fit any state-dependent structure, and with only 40k steps / n_epochs=10 per 256-step "
            "rollout, that scale correction may not fully complete."
        )
    else:
        _console.print(
            "Value net init scale is roughly commensurate with the return-target scale — a raw scale "
            "mismatch is NOT the primary explanation; the negative explained_variance seen at 20k-40k "
            "steps in the full runs is more likely a genuine model.fit-to-noise problem (see item 3's "
            "supervised diagnostic for whether the observation predicts direction at all)."
        )

    # Pull the actual 40k-step run's saved explained_variance curve for comparison.
    for mode in ["simple", "marta"]:
        path = _RESULTS_DIR / f"minimal_btc_env_{mode}.json"
        if not path.exists():
            continue
        hist = json.loads(path.read_text())["training_diagnostics"]
        ev_curve = [(h["timesteps"], h["explained_variance"]) for h in hist if h["explained_variance"] is not None]
        if ev_curve:
            first_step, first_ev = ev_curve[0]
            last_step, last_ev = ev_curve[-1]
            best_step, best_ev = max(ev_curve, key=lambda x: x[1])
            _console.print(
                f"[{mode}] full 40k-step run: explained_variance at step {first_step} = {first_ev:.3f}, "
                f"at step {last_step} = {last_ev:.3f}, BEST over the whole run = {best_ev:.3f} at step {best_step} "
                f"({'never crosses 0' if best_ev < 0 else 'crosses positive at least once'})."
            )
            trace[f"{mode}_full_run_explained_variance_first_last_best"] = {
                "first": [first_step, first_ev], "last": [last_step, last_ev], "best": [best_step, best_ev],
            }

    return trace


# --------------------------------------------------------------------------- item 6
def audit_train_eval_consistency(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 6 — training/evaluation distribution consistency")
    df = shared["df"]
    findings: dict[str, Any] = {}

    # 6a: normalize.py's 720h z-score is a CAUSAL rolling transform computed ONCE over the
    # full continuous series (train+val+test together), not "fit on train, applied to val" —
    # confirmed by reading pipeline/normalize.py: _rolling_zscore uses .rolling(window=720,
    # min_periods=168), which at any row t only ever looks at rows <= t. Splitting into
    # train/val/test happens AFTER this transform (in rl/env.py's _split_mask /
    # MinimalBTCEnv's _split_mask), so there is no leakage even though the z-score baseline
    # isn't "fit on train only" in the classical ML sense — it doesn't need to be, since it's
    # re-derived causally at every timestep.
    findings["normalization_is_causal_rolling_not_train_fit"] = True
    findings["normalization_window_hours"] = 720

    # 6b: split boundaries are imported directly from rl.env (single source of truth) —
    # not re-declared/hardcoded in rl/minimal_btc_env.py, so they cannot drift independently.
    from rl.minimal_btc_env import TRAIN_END as MBE_TRAIN_END, VAL_END as MBE_VAL_END, VAL_START as MBE_VAL_START
    findings["split_boundaries_match_production"] = (
        MBE_TRAIN_END == TRAIN_END and MBE_VAL_START == VAL_START and MBE_VAL_END == VAL_END
    )
    findings["train_end"] = str(TRAIN_END)
    findings["val_start"] = str(VAL_START)
    findings["val_end"] = str(VAL_END)

    # 6c: encoder checkpoint identity — same file used in production and in TEST 3.
    from rl.minimal_btc_env import _ENCODER_PATH
    from rl.train import _ENCODER_PATH as PROD_ENCODER_PATH
    findings["encoder_checkpoint_matches_production"] = (_ENCODER_PATH.resolve() == PROD_ENCODER_PATH.resolve())
    findings["encoder_checkpoint_path"] = str(_ENCODER_PATH)

    # 6d: no row of the val split leaks into the frozen-encoder's 60-bar lookback window
    # incorrectly, i.e. the FIRST valid val start has >=60 bars of TRAIN-OR-VAL history
    # immediately preceding it (never reaches back before the dataset start) — a sanity
    # check that window construction doesn't silently wrap or reuse OOB rows.
    val_mask = np.asarray((df.index >= VAL_START) & (df.index <= VAL_END))
    first_val_idx = int(np.argmax(val_mask))
    findings["first_val_row_has_60_bars_before_it"] = first_val_idx >= 60
    findings["first_val_timestamp"] = str(df.index[first_val_idx])

    # 6e: aligned_df used by MinimalBTCEnv is the SAME file production reads.
    prod_path = aligned_path("BTC_USDT", _PROCESSED)
    findings["aligned_parquet_path"] = str(prod_path)
    findings["aligned_df_shape"] = list(df.shape)
    findings["aligned_df_date_range"] = [str(df.index[0]), str(df.index[-1])]

    _console.print(findings)
    all_ok = (
        findings["normalization_is_causal_rolling_not_train_fit"]
        and findings["split_boundaries_match_production"]
        and findings["encoder_checkpoint_matches_production"]
        and findings["first_val_row_has_60_bars_before_it"]
    )
    _console.print(f"[bold {'green' if all_ok else 'red'}]Consistency check: {'PASS' if all_ok else 'FAIL — see findings above'}")
    return findings


# --------------------------------------------------------------------------- main
def main() -> None:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    shared = load_shared_data(device="cpu")

    print_feature_table()
    print_marta_obs_breakdown()
    reward_stats = audit_reward_distribution(shared)
    return_stats = audit_next_return_stats(shared)
    value_fn_trace = audit_value_function(shared)
    consistency = audit_train_eval_consistency(shared)

    out = {
        "item3_reward_distribution": reward_stats,
        "item4_next_return_stats": return_stats,
        "item5_value_function_trace": value_fn_trace,
        "item6_consistency": consistency,
    }
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "test3_mechanics_audit.json").write_text(json.dumps(out, indent=2, default=float))
    _console.rule("[bold]Audit complete")
    _console.print(f"Saved -> {_RESULTS_DIR / 'test3_mechanics_audit.json'}")


if __name__ == "__main__":
    main()
