"""Controlled-policy diagnostic for MARTATradingEnv — TEST 0 + TEST 1 of the RL audit.

Runs fixed, non-learned policies (LONG / SHORT / HOLD / RANDOM / ORACLE) through the
REAL environment implementation (rl.env.MARTATradingEnv) — not a reimplementation —
to answer one question before touching PPO again:

    If a policy takes a good action, does the environment actually reward it correctly?

ORACLE is a software test only (cheats by looking at next-bar return). It must never be
used as a trading strategy or reported as a performance result — it exists purely to
prove the env's action -> position -> P&L -> reward wiring is directionally correct.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.diagnose_env
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.diagnose_env --asset BTC_USDT --split val
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.diagnose_env --n-samples-table 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from encoder.model import TFTRegimeEncoder
from rl.env import CANONICAL_ASSETS, MARTATradingEnv
from utils.assets import aligned_path
from utils.logger import logger

_console = Console()
_PROCESSED = Path("data/processed")
_ENCODER_PATH = Path("encoder/checkpoints/best_encoder.pt")
_REGIME_STREAM_PATH = _PROCESSED / "regime_stream.parquet"
_RESULTS_PATH = Path("data/results/env_diagnostic.json")
_SEED = 42

Policy = Callable[[int, float], dict[str, Any]]  # (step_idx, next_return) -> action dict


# --------------------------------------------------------------------------- policies
def policy_always_long(step_idx: int, next_return: float) -> dict[str, Any]:
    return {"direction": 0, "position_fraction": 0.5}


def policy_always_short(step_idx: int, next_return: float) -> dict[str, Any]:
    return {"direction": 1, "position_fraction": 0.5}


def policy_always_hold(step_idx: int, next_return: float) -> dict[str, Any]:
    return {"direction": 2, "position_fraction": 0.5}


def _make_random_policy(rng: np.random.Generator) -> Policy:
    def _policy(step_idx: int, next_return: float) -> dict[str, Any]:
        direction = int(rng.integers(0, 3))
        frac = float(rng.uniform(0.01, 0.99))
        return {"direction": direction, "position_fraction": frac}

    return _policy


def policy_oracle(step_idx: int, next_return: float) -> dict[str, Any]:
    """CHEATS by looking at the next-bar return. Software test only — never a strategy."""
    if next_return > 0:
        direction = 0
    elif next_return < 0:
        direction = 1
    else:
        direction = 2
    return {"direction": direction, "position_fraction": 0.5}


# --------------------------------------------------------------------------- env setup
def build_diagnostic_env(
    asset: str,
    split: str,
    idle_penalty: float = 0.0,
    min_hold_bars: int = 0,
    max_leverage: float = 1.0,
) -> MARTATradingEnv:
    """Build a real MARTATradingEnv with idle_penalty=0 and min_hold_bars=0 — the
    confounders the audit calls out (see CLAUDE.md instructions #11) are switched off
    so the one-step action->reward relationship is not obscured by reward shaping or
    the minimum-holding-period lock overriding the requested action.
    """
    aligned_dfs = {asset: pd.read_parquet(aligned_path(asset, _PROCESSED), engine="pyarrow")}
    regime_stream_df = pd.read_parquet(_REGIME_STREAM_PATH, engine="pyarrow")

    device = "cpu"
    encoder = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
    ckpt = torch.load(_ENCODER_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    env = MARTATradingEnv(
        assets=[asset],
        aligned_dfs=aligned_dfs,
        regime_stream_df=regime_stream_df,
        encoder=encoder,
        debate_results=None,
        initial_capital=10_000.0,
        window_size=60,
        transaction_cost=0.001,
        lambda_drawdown=0.01,
        max_leverage=max_leverage,
        device=device,
        split=split,
        sequential=True,
        idle_penalty=idle_penalty,
        min_hold_bars=min_hold_bars,
    )
    return env


# --------------------------------------------------------------------------- runner
def run_policy(
    env: MARTATradingEnv,
    policy: Policy,
    seed: int = _SEED,
    max_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Run one policy sequentially through env, returning full per-step trajectories.

    ``next_return`` handed to the policy is read directly from env._returns — the same
    array env.step() itself consumes — so the oracle's "cheat" and the env's own return
    realisation are guaranteed to be the same number, not a parallel reimplementation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    asset = env._asset
    returns_arr = env._returns[asset]

    signed_positions = [info["signed_position"]]
    pnl_returns: list[float] = []
    rewards: list[float] = []
    next_returns_seen: list[float] = []

    step_idx = 0
    done = False
    while not done:
        # The return that WILL be realised by whatever position is taken this step is
        # returns_arr[env._t + 1] (env.step() does self._t += 1 then reads self._returns[self._t]).
        next_ret = float(returns_arr[env._t + 1]) if env._t + 1 < len(returns_arr) else 0.0
        if np.isnan(next_ret):
            next_ret = 0.0
        action = policy(step_idx, next_ret)

        obs, reward, terminated, truncated, info = env.step(action)
        signed_positions.append(info["signed_position"])
        pnl_returns.append(info["pnl_return"])
        rewards.append(reward)
        next_returns_seen.append(next_ret)

        step_idx += 1
        done = terminated or truncated
        if max_steps is not None and step_idx >= max_steps:
            break

    return {
        "signed_positions": np.asarray(signed_positions, dtype=np.float64),
        "pnl_returns": np.asarray(pnl_returns, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "next_returns": np.asarray(next_returns_seen, dtype=np.float64),
    }


def summarize(traj: dict[str, np.ndarray]) -> dict[str, float]:
    pnl = traj["pnl_returns"]
    rew = traj["rewards"]
    pos = traj["signed_positions"][1:]  # step-aligned with pnl/rewards
    n = len(pnl)
    if n == 0:
        return {}
    wins = int(np.sum(pnl > 0))
    losses = int(np.sum(pnl < 0))
    n_trades = int(np.sum(np.abs(np.diff(traj["signed_positions"])) > 1e-9))
    total_return = float(np.prod(1.0 + pnl) - 1.0)
    return {
        "n_samples": n,
        "mean_pnl_return": float(pnl.mean()),
        "mean_reward": float(rew.mean()),
        "std_reward": float(rew.std()),
        "total_return": total_return,
        "win_rate": float(wins / max(wins + losses, 1)),
        "n_trades": n_trades,
        "mean_abs_position": float(np.abs(pos).mean()),
    }


def one_step_causal_check(traj: dict[str, np.ndarray]) -> dict[str, Any]:
    """Verify sign(pnl) matches sign(position * next_return) — the core causal claim."""
    pos = traj["signed_positions"][1:]
    pnl = traj["pnl_returns"]
    nret = traj["next_returns"]
    expected_sign = np.sign(pos * nret)
    actual_sign = np.sign(pnl)
    # Only compare where a directional prediction is even possible (position != 0 and
    # return != 0) — cost-only steps (flat or exactly-zero return) are excluded since
    # transaction cost / zero return can legitimately make sign comparison meaningless.
    mask = (np.abs(pos) > 1e-9) & (nret != 0)
    if mask.sum() == 0:
        return {"n_checked": 0, "sign_match_rate": None}
    match = expected_sign[mask] == actual_sign[mask]
    return {
        "n_checked": int(mask.sum()),
        "sign_match_rate": float(match.mean()),
    }


def sample_table(
    traj_by_policy: dict[str, dict[str, np.ndarray]], n_samples: int
) -> list[dict[str, Any]]:
    """Pick n_samples step indices (spread across the run) and show LONG/SHORT/HOLD/ORACLE
    side-by-side for the exact same next-bar return, using the ORACLE trajectory's
    next_returns as the reference timeline (all policies share the same env/asset/split
    so their step indices line up 1:1 against the same underlying price path)."""
    ref = traj_by_policy["ORACLE"]
    # Some policies (e.g. RANDOM) can trip the sequential-eval drawdown safety net and
    # terminate early — cap the sampled range to the SHORTEST trajectory so every
    # policy has a value at every sampled step.
    n = min(len(traj["next_returns"]) for traj in traj_by_policy.values())
    idxs = np.linspace(0, n - 1, num=min(n_samples, n), dtype=int)
    rows = []
    for i in idxs:
        row: dict[str, Any] = {"step": int(i), "next_return_pct": float(ref["next_returns"][i] * 100)}
        for name, traj in traj_by_policy.items():
            row[f"{name}_position"] = float(traj["signed_positions"][i + 1])
            row[f"{name}_pnl_pct"] = float(traj["pnl_returns"][i] * 100)
            row[f"{name}_reward"] = float(traj["rewards"][i])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled-policy env diagnostic")
    parser.add_argument("--asset", default="BTC_USDT", choices=CANONICAL_ASSETS)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--n-samples-table", type=int, default=12)
    parser.add_argument("--idle-penalty", type=float, default=0.0)
    parser.add_argument("--min-hold-bars", type=int, default=0)
    args = parser.parse_args()

    _console.rule(f"[bold]Env Diagnostic — {args.asset} / {args.split} split")
    logger.info(
        f"idle_penalty={args.idle_penalty} min_hold_bars={args.min_hold_bars} "
        "(both zeroed by default per audit instruction #11)"
    )

    env = build_diagnostic_env(
        args.asset, args.split, idle_penalty=args.idle_penalty, min_hold_bars=args.min_hold_bars
    )

    rng = np.random.default_rng(_SEED)
    policies: dict[str, Policy] = {
        "ALWAYS_LONG": policy_always_long,
        "ALWAYS_SHORT": policy_always_short,
        "ALWAYS_HOLD": policy_always_hold,
        "RANDOM": _make_random_policy(rng),
        "ORACLE": policy_oracle,
    }

    traj_by_policy: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, float]] = {}
    causal: dict[str, dict[str, Any]] = {}

    for name, policy in policies.items():
        # Fresh env per policy (sequential=True walks the whole split deterministically
        # from the same start each time) so every policy sees an IDENTICAL price path.
        env_i = build_diagnostic_env(
            args.asset, args.split, idle_penalty=args.idle_penalty, min_hold_bars=args.min_hold_bars
        )
        traj = run_policy(env_i, policy, seed=_SEED)
        traj_by_policy[name] = traj
        summaries[name] = summarize(traj)
        causal[name] = one_step_causal_check(traj)
        logger.success(f"{name}: {summaries[name]}")

    # ---------------------------------------------------------------- summary table
    table = Table(title=f"Controlled Policy Results ({args.asset}, {args.split})")
    table.add_column("Policy")
    table.add_column("N", justify="right")
    table.add_column("Mean PnL Return", justify="right")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Std Reward", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("N Trades", justify="right")
    table.add_column("Sign-Match Rate", justify="right")
    for name in policies:
        s = summaries[name]
        c = causal[name]
        sign_rate = "n/a" if c["sign_match_rate"] is None else f"{c['sign_match_rate']:.1%}"
        table.add_row(
            name,
            str(s["n_samples"]),
            f"{s['mean_pnl_return']:.6f}",
            f"{s['mean_reward']:.6f}",
            f"{s['std_reward']:.6f}",
            f"{s['total_return']:.4%}",
            f"{s['win_rate']:.1%}",
            str(s["n_trades"]),
            sign_rate,
        )
    _console.print(table)

    # ---------------------------------------------------------------- causal verdict
    oracle_mean_reward = summaries["ORACLE"]["mean_reward"]
    long_mean_reward = summaries["ALWAYS_LONG"]["mean_reward"]
    short_mean_reward = summaries["ALWAYS_SHORT"]["mean_reward"]
    hold_mean_reward = summaries["ALWAYS_HOLD"]["mean_reward"]
    random_mean_reward = summaries["RANDOM"]["mean_reward"]

    oracle_beats_all = (
        oracle_mean_reward > long_mean_reward
        and oracle_mean_reward > short_mean_reward
        and oracle_mean_reward > hold_mean_reward
        and oracle_mean_reward > random_mean_reward
    )
    verdict = "PASS" if oracle_beats_all else "FAIL"
    _console.rule(f"[bold]VERDICT: {verdict}")
    _console.print(
        f"ORACLE mean_reward={oracle_mean_reward:.6f} vs "
        f"LONG={long_mean_reward:.6f} SHORT={short_mean_reward:.6f} "
        f"HOLD={hold_mean_reward:.6f} RANDOM={random_mean_reward:.6f}"
    )
    if not oracle_beats_all:
        _console.print(
            "[bold red]ORACLE did not clearly outperform. This means action -> position "
            "-> P&L -> reward is broken somewhere. Do not proceed to PPO training until fixed."
        )

    # ---------------------------------------------------------------- sample table
    rows = sample_table(traj_by_policy, args.n_samples_table)
    sample_tbl = Table(title="Sampled one-step cases (same price path, all policies)")
    sample_tbl.add_column("step")
    sample_tbl.add_column("next_ret%")
    for name in policies:
        sample_tbl.add_column(f"{name}\npos")
        sample_tbl.add_column(f"{name}\npnl%")
        sample_tbl.add_column(f"{name}\nreward")
    for row in rows:
        cells = [str(row["step"]), f"{row['next_return_pct']:.3f}"]
        for name in policies:
            cells.append(f"{row[f'{name}_position']:.3f}")
            cells.append(f"{row[f'{name}_pnl_pct']:.4f}")
            cells.append(f"{row[f'{name}_reward']:.4f}")
        sample_tbl.add_row(*cells)
    _console.print(sample_tbl)

    # ---------------------------------------------------------------- save
    out = {
        "asset": args.asset,
        "split": args.split,
        "idle_penalty": args.idle_penalty,
        "min_hold_bars": args.min_hold_bars,
        "summaries": summaries,
        "causal_checks": causal,
        "verdict": verdict,
        "sample_table": rows,
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"Results written to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
