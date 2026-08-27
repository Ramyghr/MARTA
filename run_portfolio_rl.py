"""run_portfolio_rl.py — training + ablation grid for MultiAssetPortfolioEnv.

Step 6 of the pre-RL audit's remediation plan (Step 3: rl/portfolio_env.py, Step 4:
rl/policy_portfolio.py). Trains PPO on the crypto5 shared-portfolio env across a small
ablation grid and evaluates each variant on the 2024 val split, using the CAUSAL regime
stream (regime/causal_regime_stream.py) — never the leaky regime_stream.parquet (see
rl/train.py's CRYPTO5_REGIME_STREAM_PATH docstring).

This is a NEW, separate ablation grid from docs/RESULTS_AND_METHODOLOGY.md Section 9's —
that table is for MARTATradingEnv (single-asset-per-episode). It is informative context
only, not evidence about this portfolio agent (different action space, different
observation, different reward). Some of Section 9's axes don't translate cleanly to a
joint-allocation policy and are deliberately NOT reproduced here:
  - "discrete action space" (row 5) — meaningless for a softmax-over-simplex action.
  - "min_hold_bars" (rows 3-5's lever) — a single-asset position-lock concept; the
    portfolio env's turnover cost already penalizes rapid reallocation continuously,
    which is the portfolio-native equivalent.
This script's grid instead varies the axes that DO translate: regime conditioning
(now causal, not leaky), the weight-tied PortfolioFeatureExtractor vs. a plain flatten+MLP,
and a behavior-cloning warm-start (equal-weight-assets teacher, this env's analogue of
rl/behavior_cloning.py's buy_hold teacher for MARTATradingEnv).

Run:
  python run_portfolio_rl.py --quick                       # fast smoke test, ~2k steps/variant
  python run_portfolio_rl.py --timesteps 200000 --device cuda
  python run_portfolio_rl.py --ablations baseline,regime --timesteps 100000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor

from configs.splits import TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE, VAL_START
from rl.evaluate import compute_max_drawdown, compute_metrics
from rl.policy_portfolio import make_ppo_portfolio
from rl.portfolio_env import CRYPTO5_ASSETS, MultiAssetPortfolioEnv
from rl.train import CRYPTO5_PROCESSED, CRYPTO5_REGIME_STREAM_CAUSAL_PATH
from utils.logger import logger
from utils.reproducibility import set_seed

_DEBATE_PATH = Path("data/debates/debate_results.json")
_EVAL_SEED = 42
_ANNUALIZE = np.sqrt(24 * 365)

ABLATION_VARIANTS: dict[str, dict] = {
    "baseline": {"obs_mode": "raw", "extractor": None, "bc": False},
    "regime": {"obs_mode": "raw_regime", "extractor": None, "bc": False},
    "extractor": {"obs_mode": "raw", "extractor": "portfolio", "bc": False},
    "regime_extractor": {"obs_mode": "raw_regime", "extractor": "portfolio", "bc": False},
    "bc_warmstart": {"obs_mode": "raw", "extractor": "portfolio", "bc": True},
}


def load_portfolio_data(
    assets: list[str], processed_dir: Path, regime_stream_path: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    aligned_dfs = {
        a: pd.read_parquet(processed_dir / f"{a}_aligned.parquet", engine="pyarrow")
        for a in assets
    }
    regime_stream_df = pd.read_parquet(regime_stream_path, engine="pyarrow")
    return aligned_dfs, regime_stream_df


def load_debate_results() -> Optional[list[dict]]:
    if _DEBATE_PATH.exists():
        return json.loads(_DEBATE_PATH.read_text())
    return None


def build_envs(
    aligned_dfs: dict[str, pd.DataFrame],
    regime_stream_df: pd.DataFrame,
    obs_mode: str,
    debate_results: Optional[list[dict]],
    max_episode_steps: int,
) -> tuple[MultiAssetPortfolioEnv, MultiAssetPortfolioEnv]:
    train_env = MultiAssetPortfolioEnv(
        assets=CRYPTO5_ASSETS, aligned_dfs=aligned_dfs, regime_stream_df=regime_stream_df,
        encoder=None, debate_results=debate_results, split="train", obs_mode=obs_mode,
        sequential=False, max_steps=max_episode_steps,
    )
    # debate_results=None on val — matches rl/train.py's env_factory convention
    # (val_env always built with debate=False) so eval isn't perturbed by sparse
    # debate-cache coverage varying run to run.
    val_env = MultiAssetPortfolioEnv(
        assets=CRYPTO5_ASSETS, aligned_dfs=aligned_dfs, regime_stream_df=regime_stream_df,
        encoder=None, debate_results=None, split="val", obs_mode=obs_mode,
        sequential=True,
    )
    return train_env, val_env


def bc_warmstart_portfolio(
    model, env: MultiAssetPortfolioEnv, epochs: int = 10, batch_size: int = 256,
    lr: float = 1e-3, max_bc_steps: int = 5000,
) -> None:
    """Equal-weight-all-assets teacher, generic MSE against the Gaussian mean logits of
    SB3's standard ActorCriticPolicy — works for any Box-action policy (unlike
    rl/behavior_cloning.py's pretrain_bc, which reads MARTAActorCriticPolicy's specific
    direction_head/position_head and doesn't apply to this env's plain Gaussian-on-logits
    action space).

    Target logits: near-zero for every asset, strongly negative for cash — softmaxes to
    roughly equal weight across assets, near-zero cash. This env's analogue of
    rl/behavior_cloning.py's buy_hold teacher (there: constant long/near-max-fraction on
    the one asset; here: constant equal-weight across all assets, never cash).
    """
    policy = model.policy
    n_actions = env.action_space.shape[0]
    target_logits = np.zeros(n_actions, dtype=np.float32)
    target_logits[-1] = -5.0  # push cash weight toward ~0 after softmax

    was_sequential = env.sequential
    env.sequential = True
    obs, _ = env.reset(seed=0)
    obs_list = [obs]
    done = False
    n_steps = 0
    while not done and n_steps < max_bc_steps:
        obs, _, terminated, truncated, _ = env.step(target_logits)
        done = terminated or truncated
        n_steps += 1
        if not done:
            obs_list.append(obs)
    env.sequential = was_sequential

    obs_arr = np.stack(obs_list).astype(np.float32)
    target_arr = np.tile(target_logits, (len(obs_arr), 1)).astype(np.float32)
    logger.info(f"[bc-portfolio] {len(obs_arr):,} equal-weight teacher (obs, target) pairs")

    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(obs_arr)
    idx = np.arange(n)
    policy.train()
    for epoch in range(epochs):
        np.random.shuffle(idx)
        total, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            obs_t, _ = policy.obs_to_tensor(obs_arr[b])
            target_t = torch.as_tensor(target_arr[b], device=device)
            features = policy.extract_features(obs_t)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)
            loss = torch.nn.functional.mse_loss(mean_actions, target_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        logger.info(f"[bc-portfolio] epoch {epoch + 1}/{epochs} loss={total / max(n_batches, 1):.4f}")
    policy.eval()
    logger.success(f"[bc-portfolio] warm-start complete — {n:,} samples, {epochs} epochs")


def evaluate_portfolio(model, env: MultiAssetPortfolioEnv) -> dict:
    """Deterministic walk-forward evaluation over env's active (sequential) split."""
    assert env.sequential, "evaluate_portfolio requires a sequential env"
    set_seed(_EVAL_SEED)
    obs, _ = env.reset(seed=_EVAL_SEED)
    portfolio_values = [env.initial_capital]
    returns: list[float] = []
    turnovers: list[float] = []
    weights_log: list[dict] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        returns.append(info["step_return"])
        turnovers.append(info["turnover_cost_fraction"])
        weights_log.append(info["weights"])
        done = terminated or truncated

    metrics = compute_metrics(np.array(returns), np.array(portfolio_values))
    metrics["mean_turnover_cost_fraction"] = float(np.mean(turnovers)) if turnovers else 0.0
    metrics["mean_cash_weight"] = float(np.mean([w["cash"] for w in weights_log])) if weights_log else 1.0
    metrics["n_steps"] = len(returns)
    return metrics


def compute_equal_weight_buy_hold(aligned_dfs: dict[str, pd.DataFrame]) -> dict:
    """No-model baseline: equal-weight across all crypto5 assets, held constant over the
    val split (one rebalance cost at the start, none after) — this env's analogue of
    rl/ablation.py's buy_hold, for comparison against the trained variants above."""
    val_start = pd.Timestamp(VAL_START, tz="UTC")
    val_end = pd.Timestamp(VAL_END_EXCLUSIVE, tz="UTC") - pd.Timedelta(seconds=1)
    n = len(CRYPTO5_ASSETS)
    returns_by_asset = []
    for a in CRYPTO5_ASSETS:
        df = aligned_dfs[a]
        sub = df.loc[(df.index >= val_start) & (df.index <= val_end), "close_return"]
        returns_by_asset.append(sub.to_numpy(dtype=np.float64))
    min_len = min(len(r) for r in returns_by_asset)
    returns_by_asset = np.stack([r[:min_len] for r in returns_by_asset])  # (n_assets, T)
    port_returns = returns_by_asset.mean(axis=0)  # equal weight
    portfolio_values = np.concatenate([[10_000.0], 10_000.0 * np.cumprod(1 + port_returns)])
    metrics = compute_metrics(port_returns, portfolio_values)
    return metrics


def run_variant(
    name: str, cfg: dict, aligned_dfs: dict[str, pd.DataFrame], regime_stream_df: pd.DataFrame,
    debate_results: Optional[list[dict]], args: argparse.Namespace,
) -> dict:
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== variant {name}: {cfg} ===")

    train_env, val_env = build_envs(
        aligned_dfs, regime_stream_df, cfg["obs_mode"], debate_results, args.max_episode_steps,
    )
    monitored = Monitor(train_env)
    model = make_ppo_portfolio(
        monitored, extractor=cfg["extractor"], device=args.device, seed=args.seed,
        n_steps=args.n_steps, batch_size=args.batch_size,
        tensorboard_log=str(out_dir / "tb"),
    )

    if cfg["bc"]:
        bc_warmstart_portfolio(model, train_env, epochs=args.bc_epochs)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - t0

    ckpt_path = out_dir / "ppo_portfolio.zip"
    model.save(ckpt_path)

    metrics = evaluate_portfolio(model, val_env)
    metrics["wall_clock_seconds"] = elapsed
    metrics["timesteps"] = args.timesteps
    metrics["obs_dim"] = int(train_env.observation_space.shape[0])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    logger.success(
        f"[{name}] sharpe={metrics['sharpe']:.3f} total_return={metrics['total_return']:.3f} "
        f"mean_cash_weight={metrics['mean_cash_weight']:.3f} elapsed={elapsed:.0f}s"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000,
                         help="PPO timesteps per variant (default 20k — modest; scale up "
                              "once you have a real wall-clock-per-step number from a first run)")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--ablations", type=str, default="all",
                         help="Comma-separated variant names, or 'all' (default)")
    parser.add_argument("--output-dir", type=str, default="portfolio_runs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke test: 2k timesteps, small n_steps/batch — proves "
                              "the pipeline runs end to end, not a real result")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 2048
        args.n_steps = 256
        args.batch_size = 32
        args.bc_epochs = 2

    set_seed(args.seed)

    aligned_dfs, regime_stream_df = load_portfolio_data(
        CRYPTO5_ASSETS, CRYPTO5_PROCESSED, CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    )
    debate_results = load_debate_results()
    logger.info(
        f"Loaded {len(CRYPTO5_ASSETS)} assets, {len(regime_stream_df):,} regime-stream rows "
        f"(causal), debate_results={'yes' if debate_results else 'none'}, device={args.device}"
    )

    variant_names = (
        list(ABLATION_VARIANTS) if args.ablations == "all" else args.ablations.split(",")
    )
    for v in variant_names:
        if v not in ABLATION_VARIANTS:
            raise SystemExit(f"Unknown ablation variant {v!r} — choices: {list(ABLATION_VARIANTS)}")

    results: dict[str, dict] = {}
    for name in variant_names:
        results[name] = run_variant(
            name, ABLATION_VARIANTS[name], aligned_dfs, regime_stream_df, debate_results, args,
        )

    logger.info("=== equal-weight buy-and-hold baseline ===")
    results["buy_hold_equal_weight"] = compute_equal_weight_buy_hold(aligned_dfs)

    out_path = Path(args.output_dir) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))

    print("\n=== SUMMARY ===")
    print(f"{'variant':<22}{'sharpe':>10}{'total_ret':>12}{'max_dd':>10}{'mean_cash%':>12}")
    for name, m in results.items():
        cash = m.get("mean_cash_weight", float("nan")) * 100
        print(f"{name:<22}{m['sharpe']:>10.3f}{m['total_return']:>12.3f}{m['max_drawdown']:>10.3f}{cash:>12.1f}")
    logger.success(f"Full results -> {out_path}")


if __name__ == "__main__":
    main()
