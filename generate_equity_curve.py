"""Pre-generate the equity curve by running ppo_best.zip through the validation split.

Run once before the jury demo:
    /home/ramy/miniconda3/envs/marta/bin/python generate_equity_curve.py

Saves data/results/equity_curve.json — the dashboard serves this file directly.
If the file already exists, pass --force to regenerate.

Uses the same build_env_factory(..., debate=False, idle_penalty=0.0) construction as
scripts/evaluate_all_checkpoints.py so the curve's final return/Sharpe matches the
"clean" (idle-penalty-free) numbers documented in CLAUDE.md, not an inflated val run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MARTA_ROOT = Path(__file__).parent

CONFIG = {
    "all_assets": ["BTC_USDT", "ETH_USDT"],
    "device": "cuda",
    "initial_capital": 10_000.0,
    "window_size": 60,
    "transaction_cost": 0.001,
    "lambda_drawdown": 0.01,
    "max_leverage": 1.0,
    "max_episode_steps": 2000,
}


def main(force: bool = False):
    out_path = MARTA_ROOT / "data/results/equity_curve.json"
    if out_path.exists() and not force:
        print(f"[equity_curve] {out_path} already exists — pass --force to regenerate")
        return

    checkpoint = MARTA_ROOT / "rl/checkpoints/ppo_best.zip"
    if not checkpoint.exists():
        checkpoint = MARTA_ROOT / "rl/checkpoints/ppo_final.zip"
    if not checkpoint.exists():
        print("[equity_curve] No PPO checkpoint found — run_rl.py first", file=sys.stderr)
        sys.exit(1)

    print(f"[equity_curve] Loading model: {checkpoint}")
    from stable_baselines3 import PPO
    from rl.evaluate import run_walk_forward, compute_max_drawdown
    from rl.train import build_env_factory, load_data, load_debate_results

    model = PPO.load(str(checkpoint), device=CONFIG["device"])

    aligned_dfs, regime_stream_df, encoder = load_data(CONFIG)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, CONFIG)

    # val_env must never carry idle_penalty (CLAUDE.md: "idle_penalty is a training
    # signal only") — debate=False matches the checkpoint_evaluation.json baseline.
    env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, debate=False, idle_penalty=0.0)

    print("[equity_curve] Running walk-forward evaluation on val split…")
    traj = run_walk_forward(model, env, deterministic=True)

    portfolio_values = traj["portfolio_values"].tolist()
    positions = traj["positions"].tolist()
    # Decision-time regime (falls back to current_regime when unavailable) — matches
    # the attribution convention used by rl/evaluate.py's per-regime breakdown.
    regimes = traj["decision_regimes"].tolist()

    # Buy & Hold: invest $10000 at first price, track value
    start_capital = portfolio_values[0]
    features_path = MARTA_ROOT / "data/processed/BTC_USDT_features.parquet"
    import pandas as pd
    btc = pd.read_parquet(features_path)
    btc.index = pd.to_datetime(btc.index, utc=True)
    val_start = pd.Timestamp("2021-01-01", tz="UTC")
    val_end = pd.Timestamp("2021-10-12", tz="UTC")
    btc_val = btc[(btc.index >= val_start) & (btc.index <= val_end)]

    if len(btc_val) < len(portfolio_values):
        # Extend by repeating last price if needed
        pad = len(portfolio_values) - len(btc_val)
        btc_val = pd.concat([btc_val, btc_val.iloc[[-1]].loc[btc_val.index[[-1]].repeat(pad)]])

    bh_prices = btc_val["close"].values[: len(portfolio_values)]
    bh_values = (bh_prices / bh_prices[0] * start_capital).tolist()

    timestamps = btc_val.index[: len(portfolio_values)].strftime("%Y-%m-%d").tolist()

    # Downsample to daily for the chart (keep every 24th point)
    step = 24
    ts_ds = timestamps[::step]
    marta_ds = portfolio_values[::step]
    bh_ds = bh_values[::step]
    positions_ds = positions[::step]
    regimes_ds = regimes[::step]
    prices_ds = [round(float(p), 2) for p in bh_prices[::step]]

    # Compute metrics
    marta_arr = np.array(portfolio_values)
    bh_arr = np.array(bh_values)

    marta_return = float(marta_arr[-1] / marta_arr[0] - 1)
    bh_return = float(bh_arr[-1] / bh_arr[0] - 1)
    marta_dd = float(compute_max_drawdown(marta_arr))
    bh_dd = float(compute_max_drawdown(bh_arr))

    result = {
        "timestamps": ts_ds,
        "marta": [round(v, 2) for v in marta_ds],
        "buy_hold": [round(v, 2) for v in bh_ds],
        "positions": [round(v, 4) for v in positions_ds],
        "regimes": [int(r) for r in regimes_ds],
        "prices": prices_ds,
        "synthetic": False,
        "checkpoint": str(checkpoint.name),
        "marta_final_return": round(marta_return * 100, 2),
        "bh_final_return": round(bh_return * 100, 2),
        "marta_max_dd": round(marta_dd * 100, 2),
        "bh_max_dd": round(bh_dd * 100, 2),
        "n_steps": len(portfolio_values),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[equity_curve] Saved to {out_path}")
    print(f"  MARTA:    return={result['marta_final_return']:.1f}%  DD={result['marta_max_dd']:.1f}%")
    print(f"  Buy&Hold: return={result['bh_final_return']:.1f}%  DD={result['bh_max_dd']:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate even if file exists")
    args = parser.parse_args()
    main(force=args.force)
