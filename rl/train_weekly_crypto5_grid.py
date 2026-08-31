"""rl/train_weekly_crypto5_grid.py

Generalizes rl/train_weekly_smoketest.py's native-weekly-cadence A/B/C/D grid from
BTC_USDT-only to all 5 assets of the crypto5 panel (BTC/ETH/BNB/LTC/XRP), and from
3 seeds to 4 (seeds 0-3) -- follow-up requested after the BTC-only weekly grid
(docs/RESULTS_AND_METHODOLOGY.md Section 10/10a: 12/12 BTC-only runs beat buy-hold on
2024 validation at the point estimate, none individually significant; the pre-registered
2025 test pass did not replicate it).

Same CONDITIONS (A raw / B raw_vol / C raw_regime / D raw_regime+placebo), same
decision_interval=168 (one step = one week of hourly bars), same TOTAL_TIMESTEPS /
MAX_STEPS_PER_EPISODE -- imported directly from rl.train_weekly_smoketest rather than
redefined, so results stay comparable condition-for-condition and this file can't
silently drift out of sync with the BTC-only run it's meant to extend.

Each (asset, label, seed) cell trains + evaluates independently (own checkpoint, own
val-Sharpe-selected "best" reload -- see run_one(), same convention as
train_weekly_smoketest.run_one()) and is validated on 2024 (config-driven walk-forward
split). This is a VALIDATION grid, not a test-set run -- it never touches 2025 data.
Results land in data/results/weekly_grid_crypto5/ (one JSON per cell) plus a reconciled
summary (mean/std across seeds per asset x condition) written by summarize().

Requires (none of this ships in the repo -- see CLAUDE.md's data-provenance notes):
  - data/raw/{BTC,ETH,BNB,LTC,XRP}_USDT_1h_raw.parquet          (pipeline/fetch_crypto5.py
    fetches BNB/LTC/XRP; BTC/ETH need the standard pipeline fetch first)
  - data/processed/crypto5_multihorizon/{asset}_aligned.parquet  (regime/build_crypto5_features.py)
  - regime/checkpoints/crypto5_multihorizon/hmm_model.pkl        (regime/run_hmm_crypto5_test.py)
  - data/processed/crypto5_multihorizon/regime_stream_causal.parquet (regime/causal_regime_stream.py
    --tag crypto5_multihorizon)
  - data/processed/crypto5_multihorizon/regime_stream_placebo.parquet -- rl/placebo_regime.py's
    main() only prints + sanity-checks the rotated stream, it does not save one; write it out
    with e.g.:
      python -c "
      from pathlib import Path
      import pandas as pd
      from rl.placebo_regime import rotate_regime_stream
      from rl.train import CRYPTO5_REGIME_STREAM_CAUSAL_PATH
      df = pd.read_parquet(CRYPTO5_REGIME_STREAM_CAUSAL_PATH, engine='pyarrow')
      rotate_regime_stream(df, seed=42).to_parquet(
          Path(CRYPTO5_REGIME_STREAM_CAUSAL_PATH).parent / 'regime_stream_placebo.parquet',
          engine='pyarrow')
      "
    before Condition D can run.

Run:
  # one cell
  conda run -n marta python rl/train_weekly_crypto5_grid.py --asset BTC_USDT --label A --seed 0

  # pilot: all 5 assets x 4 conditions, seed 0 only (20 runs) -- do this before --all to
  # confirm the pipeline is wired correctly and to measure real per-run wall-clock time
  # on the machine actually running it before committing to the full grid
  conda run -n marta python rl/train_weekly_crypto5_grid.py --pilot

  # full grid: all 5 assets x 4 conditions x 4 seeds (80 runs)
  conda run -n marta python rl/train_weekly_crypto5_grid.py --all --seeds 0 1 2 3

  # reconcile whatever cells have completed so far into a summary table, no training
  conda run -n marta python rl/train_weekly_crypto5_grid.py --summarize-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import mlflow

from run_rl import build_config, _policy_class_for, sanity_check_env
from rl.evaluate import evaluate_ppo
from rl.train import (
    CRYPTO5_CANONICAL_ASSETS,
    CRYPTO5_PROCESSED,
    CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    train_ppo,
)
from rl.train_weekly_smoketest import CONDITIONS, DECISION_INTERVAL, MAX_STEPS_PER_EPISODE, TOTAL_TIMESTEPS
from utils.logger import logger

CRYPTO5_ASSETS: list[str] = CRYPTO5_CANONICAL_ASSETS  # BTC_USDT, ETH_USDT, BNB_USDT, LTC_USDT, XRP_USDT
SEEDS_DEFAULT = (0, 1, 2, 3)

RESULTS_DIR = ROOT / "data" / "results" / "weekly_grid_crypto5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_weekly_config(asset: str, obs_mode: str, seed: int, regime_path: str | None = None, extra_tag: str | None = None) -> dict:
    config = build_config(quick=False)
    config["processed_dir"] = str(CRYPTO5_PROCESSED)
    config["regime_stream_path"] = regime_path or str(CRYPTO5_REGIME_STREAM_CAUSAL_PATH)
    config["canonical_assets"] = CRYPTO5_CANONICAL_ASSETS
    config["all_assets"] = [asset]
    config["obs_mode"] = obs_mode
    config["decision_interval"] = DECISION_INTERVAL
    config["device"] = "cpu"  # weekly mode never touches the encoder; CPU is free and avoids VRAM contention
    config["phases"] = [{
        "assets": [asset], "sharpe_threshold": -999, "max_steps": TOTAL_TIMESTEPS,
        "idle_penalty": 0.0,
    }]
    config["max_episode_steps"] = MAX_STEPS_PER_EPISODE
    # extra_tag disambiguates C (real causal regime) from D (placebo regime) -- both use
    # obs_mode="raw_regime", so without this their checkpoints/results would collide.
    tag_suffix = f"_{extra_tag}" if extra_tag else ""
    config["checkpoint_suffix"] = f"_crypto5_weekly_{asset}_{obs_mode}{tag_suffix}_seed{seed}"
    return config


def cell_result_path(asset: str, label: str, seed: int) -> Path:
    return RESULTS_DIR / f"{asset}_{label}_seed{seed}.json"


def run_one(asset: str, label: str, seed: int) -> dict:
    cfg = CONDITIONS[label]
    file_label = f"{asset}_{label}_seed{seed}"
    config = build_weekly_config(asset, cfg["obs_mode"], seed, cfg["regime_path"], cfg["extra_tag"])
    config["seed"] = seed

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("marta_weekly_grid_crypto5")
    with mlflow.start_run(run_name=f"weekly_{file_label}"):
        mlflow.log_params({
            "asset": asset, "label": label, "obs_mode": cfg["obs_mode"], "seed": seed,
            "extra_tag": cfg["extra_tag"] or "", "decision_interval": DECISION_INTERVAL,
            "total_timesteps": TOTAL_TIMESTEPS, "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "idle_penalty": 0.0,
        })

        t0 = time.time()
        train_out = train_ppo(config, pre_train_check=sanity_check_env)
        train_elapsed = time.time() - t0
        mlflow.log_metric("train_wall_seconds", train_elapsed)
        logger.success(f"[{file_label}] training done in {train_elapsed:.0f}s")

        # Reload best-by-validation-Sharpe before evaluating (run_rl.py's convention,
        # matches train_weekly_smoketest.run_one() -- train_out["model"] is just the
        # live end-of-.learn() state, which can materially understate true performance).
        from stable_baselines3 import PPO as _PPO
        best_path = train_out["best_path"]
        if best_path.exists():
            model = _PPO.load(str(best_path), device="cpu", custom_objects={"policy_class": _policy_class_for(config)})
        else:
            model = train_out["model"]
        env_factory = train_out["env_factory"]
        eval_path = cell_result_path(asset, label, seed)
        eval_res = evaluate_ppo(model, env_factory, config, output_path=eval_path)

        m = eval_res["marta_ppo"]
        mlflow.log_metrics({
            "val_sharpe": m["sharpe"], "val_total_return": m["total_return"],
            "val_max_drawdown": m["max_drawdown"], "val_n_meaningful_trades": m["n_meaningful_trades"],
        })

        summary = {
            "asset": asset, "label": label, "obs_mode": cfg["obs_mode"], "seed": seed,
            "sharpe": m["sharpe"], "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"], "n_meaningful_trades": m["n_meaningful_trades"],
            "train_wall_seconds": train_elapsed,
        }
        print(json.dumps(summary, indent=2))
        (RESULTS_DIR / f"summary_{file_label}.json").write_text(json.dumps(summary, indent=2))
        return summary


def summarize() -> dict:
    """Aggregate whatever summary_{asset}_{label}_seed{seed}.json cells exist on disk
    into per (asset, label) mean/std across seeds. Does not train anything -- safe to
    run at any point to see partial progress on a grid still in flight."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(RESULTS_DIR.glob("summary_*.json")):
        rec = json.loads(path.read_text())
        cells.setdefault((rec["asset"], rec["label"]), []).append(rec)

    out: dict[str, dict] = {}
    for (asset, label), recs in sorted(cells.items()):
        sharpes = [r["sharpe"] for r in recs]
        n = len(sharpes)
        mean = sum(sharpes) / n
        std = (sum((s - mean) ** 2 for s in sharpes) / n) ** 0.5 if n > 1 else 0.0
        key = f"{asset}_{label}"
        out[key] = {
            "asset": asset, "label": label, "obs_mode": recs[0]["obs_mode"],
            "n_seeds": n, "seeds": sorted(r["seed"] for r in recs),
            "sharpe_mean": mean, "sharpe_std": std,
            "sharpe_per_seed": {r["seed"]: r["sharpe"] for r in recs},
            "total_return_mean": sum(r["total_return"] for r in recs) / n,
            "max_drawdown_mean": sum(r["max_drawdown"] for r in recs) / n,
            "n_meaningful_trades_mean": sum(r["n_meaningful_trades"] for r in recs) / n,
        }
        print(f"{asset:>9} {label}  n_seeds={n}  sharpe={mean:+.3f} (std={std:.3f})  "
              f"return={out[key]['total_return_mean']:+.3f}  dd={out[key]['max_drawdown_mean']:.3f}")

    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{len(cells)} (asset, condition) cells reconciled -> {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, choices=CRYPTO5_ASSETS)
    parser.add_argument("--label", type=str, choices=list(CONDITIONS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--pilot", action="store_true",
                         help="all 5 assets x 4 conditions, seed 0 only (20 runs) -- run this "
                              "before --all to validate the pipeline and measure real timing")
    parser.add_argument("--all", action="store_true",
                         help="all 5 assets x 4 conditions x --seeds (default 0-3, 80 runs)")
    parser.add_argument("--summarize-only", action="store_true",
                         help="aggregate existing result cells into summary.json, train nothing")
    args = parser.parse_args()

    if args.summarize_only:
        summarize()
        return

    if args.pilot:
        for asset in CRYPTO5_ASSETS:
            for label in CONDITIONS:
                run_one(asset, label, seed=0)
        summarize()
    elif args.all:
        for asset in CRYPTO5_ASSETS:
            for label in CONDITIONS:
                for seed in args.seeds:
                    run_one(asset, label, seed)
        summarize()
    elif args.asset and args.label:
        run_one(args.asset, args.label, args.seed)
    else:
        parser.error("pass --asset {BTC_USDT,...} --label {A,B,C,D} --seed N, "
                      "or --pilot, or --all [--seeds 0 1 2 3], or --summarize-only")


if __name__ == "__main__":
    main()
