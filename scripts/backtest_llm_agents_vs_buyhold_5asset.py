"""Debate-layer audit, full coverage: LLM-agent-consensus rule-based strategy vs
buy-and-hold, all 5 crypto5 assets, 2024 val window.

Generalizes backtest_llm_agents_vs_buyhold.py (BTC_USDT-only, ~384-debate BTC-dense
cache) now that data/debates/debate_results.json has dense (~366/asset) coverage for
all five crypto5 assets (BTC/ETH/BNB/LTC/XRP_USDT), not just BTC — see
kaggle_debate_coverage_instructions.md and the 2026-09-15 merge that brought the cache
to 1,855 records. Same method per asset: rl/ablation.py's _run_agents_rule() (nearest
cached debate within 7 days drives direction/size, no PPO) vs _run_buy_hold(), both on
the crypto5 raw-feature panel (no encoder/PCA — that pipeline was abandoned in the
crypto5 pivot). Every debate was generated with full RegimeContext injected into agent
prompts, so this is "LLM agents WITH regime" by construction, same caveat as the
BTC-only version: no "without regime" counterpart exists from this cache.

Run: /home/ramy/miniconda3/envs/marta/bin/python scripts/backtest_llm_agents_vs_buyhold_5asset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# `python scripts/foo.py` puts scripts/ (not the repo root) on sys.path[0], so `rl`/
# `agents`/`utils` fail to import — same fix kaggle_debate_coverage.py needed.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from rl.ablation import _run_agents_rule, _run_buy_hold, bootstrap_sharpe_ci
from rl.evaluate import _periods_per_year_for_env, compute_metrics
from rl.train import (
    CRYPTO5_CANONICAL_ASSETS,
    CRYPTO5_PROCESSED,
    CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    build_env_factory,
    load_data,
    load_debate_results,
)

_OUTPUT_PATH = Path("data/results/llm_agents_only_backtest_5asset.json")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "all_assets": CRYPTO5_CANONICAL_ASSETS,
        "canonical_assets": CRYPTO5_CANONICAL_ASSETS,
        "processed_dir": str(CRYPTO5_PROCESSED),
        "regime_stream_path": str(CRYPTO5_REGIME_STREAM_CAUSAL_PATH),
        "obs_mode": "raw",
        "device": device,
        "initial_capital": 10_000.0,
        "window_size": 60,
        "transaction_cost": 0.001,
        "lambda_drawdown": 0.01,
        "max_leverage": 1.0,
        "max_episode_steps": 2000,
        "min_hold_bars": 4,
        "min_hold_stop_loss_dd": 0.05,
        "max_drawdown_termination": 0.20,
        "benchmark_relative": False,
    }

    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    print(f"Loaded {len(debate_results)} cached debates total")

    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)

    per_asset: dict[str, dict] = {}
    for asset in CRYPTO5_CANONICAL_ASSETS:
        n_debates = sum(1 for r in debate_results if r.get("asset") == asset)
        print(f"\n=== {asset} ({n_debates} cached debates) ===")

        bh_env = env_factory(assets=[asset], split="val", sequential=True, debate=False)
        bh_traj = _run_buy_hold(bh_env, asset=asset)
        bh_metrics = compute_metrics(
            bh_traj["returns"], bh_traj["portfolio_values"], bh_traj["positions"],
            periods_per_year=_periods_per_year_for_env(bh_env),
        )

        agent_env = env_factory(assets=[asset], split="val", sequential=True, debate=True)
        agent_traj = _run_agents_rule(agent_env)
        agent_metrics = compute_metrics(
            agent_traj["returns"], agent_traj["portfolio_values"], agent_traj["positions"],
            periods_per_year=_periods_per_year_for_env(agent_env),
        )

        delta, lo, hi = bootstrap_sharpe_ci(agent_traj["returns"], bh_traj["returns"], n_bootstrap=1000)
        significant = lo > 0 or hi < 0

        n_steps_matched = sum(
            1 for i in range(len(agent_traj["positions"])) if abs(agent_traj["positions"][i]) > 1e-6
        )

        per_asset[asset] = {
            "buy_hold": bh_metrics,
            "llm_agents_with_regime": agent_metrics,
            "bootstrap_delta_sharpe_llm_agents_vs_buy_hold": {
                "delta": delta, "ci_lower_95": lo, "ci_upper_95": hi, "significant": significant,
            },
            "n_debates_in_cache": n_debates,
            "n_val_steps_with_nonzero_position": n_steps_matched,
            "n_val_steps_total": len(agent_traj["positions"]),
        }
        print(
            f"  buy_hold sharpe={bh_metrics['sharpe']:.3f}  "
            f"agents sharpe={agent_metrics['sharpe']:.3f}  "
            f"delta={delta:+.3f} [{lo:+.2f}, {hi:+.2f}]  "
            f"{'SIGNIFICANT' if significant else 'not significant'}"
        )

    n_significant = sum(1 for r in per_asset.values() if r["bootstrap_delta_sharpe_llm_agents_vs_buy_hold"]["significant"])
    result = {
        "per_asset": per_asset,
        "n_assets_significant": n_significant,
        "n_assets_total": len(CRYPTO5_CANONICAL_ASSETS),
        "n_debates_in_cache_total": len(debate_results),
        "note": (
            "5-asset generalization of llm_agents_only_backtest.json (BTC_USDT-only). Each "
            "asset uses rl/ablation.py's _run_agents_rule() (nearest cached debate within 7 "
            "days drives direction/size, no PPO) against data/debates/debate_results.json, "
            "now densely covered (~366/asset) for all 5 crypto5 assets, not just BTC. Every "
            "debate was generated with full RegimeContext injected into agent prompts, so "
            "this already IS the regime-conditioned variant. No 'without regime' counterpart "
            "exists from this cache — that requires a separate generation pass "
            "(run_clean_ablation_debates.py, include_regime=False)."
        ),
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n{n_significant}/{len(CRYPTO5_CANONICAL_ASSETS)} assets significant. Wrote {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
