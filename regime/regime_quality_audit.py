"""regime/regime_quality_audit.py

Standalone statistical audit of HMM regime label quality — independent of the RL/PPO
layer. Answers 5 questions about the labels in a regime_stream.parquet:

  1. Distinctness — per-regime mean/vol/autocorr/skew on RAW close_return, then ANOVA
     + pairwise KS tests on the return distributions. Also (added 2026-08-19 — the
     original audit only ever tested close_return, never the volatility features these
     regimes are actually sold on) the same ANOVA+KS battery on realized_vol_raw, the
     unnormalized crash-relevant vol feature.
  2. Stability — transition matrix + mean dwell time per regime (a "regime" that
     flips every step is noise, not a regime).
  3. Balance/production — confirms labels come from GaussianHMM.predict() (Viterbi
     decode), not quantile bucketing, and checks whether the suspicious near-equal
     state counts trace back to near-duplicate HMM states.
  4. Predictive signal — logistic regression + gradient boosting predicting
     sign(next return) from regime-only / regime+features / features-only, using both
     the original 12-feature set (for comparability with historical runs) and (added
     2026-08-19) whatever extra columns the target regime_stream's own aligned data
     actually has beyond those 12 (e.g. multi-horizon z-scores, realized_vol_raw) —
     the original audit never tested with the fuller feature set a candidate model was
     actually built from.
  5. Embedding usage — input-gradient saliency on a trained PPO policy: does it
     actually use the 128-dim regime block, or has it learned to ignore it? PPO/RL-
     specific — skipped unless explicitly requested (--sections includes 5), since it
     requires a checkpoint trained on this exact regime encoding, which most candidate
     models being audited here won't have.

Originally hardcoded to production's data/processed/regime_stream.parquet + 4-asset
panel; generalized (2026-08-19) via CLI args so the same battery can run against any
candidate model's regime stream (raw-feature baselines, crypto-only panels, etc.)
without copy-pasting the script. Running with no args reproduces the original
production audit exactly (same defaults).

--regime-stream default (no flag given): resolved via utils.freshness.resolve_regime_
stream_path(--processed-dir), which prefers regime_stream_causal.parquet
(regime/causal_regime_stream.py's walk-forward, non-leaky replacement) when one exists
in that directory, falling back to the plain (forward-backward smoothed) regime_stream.
parquet otherwise. Pass --regime-stream explicitly to force a specific file, e.g. for a
deliberate causal-vs-leaky comparison run.

Run:
  python regime/regime_quality_audit.py
  python regime/regime_quality_audit.py \
      --processed-dir data/processed/crypto5_multihorizon \
      --assets BTC_USDT,ETH_USDT,BNB_USDT,LTC_USDT,XRP_USDT \
      --tag crypto5_multihorizon --sections 1,4
Outputs: data/results/regime_quality_audit[_{tag}].json, regime/plots/regime_sequence_*.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.assets import aligned_path  # noqa: E402
from utils.freshness import resolve_regime_stream_path  # noqa: E402
from utils.logger import logger  # noqa: E402

_PLOTS_DIR = Path("regime/plots")
_DEFAULT_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_FEATURE_COLS = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram", "obv_raw",
    "vol_zscore", "realized_vol", "vol_ratio", "return_skew",
    "return_kurtosis", "autocorr_lag1",
]

# Set by main() from CLI args before any section function runs.
_PROCESSED = Path("data/processed")
# Causal-preferring: resolve_regime_stream_path() falls back to the leaky stream if
# _PROCESSED has no regime_stream_causal.parquet (true for the production 4-asset panel
# today) — see utils.freshness.resolve_regime_stream_path / regime/causal_regime_stream.py.
_REGIME_STREAM_PATH = resolve_regime_stream_path(_PROCESSED)
_RESULTS_PATH = Path("data/results/regime_quality_audit.json")
_ASSETS = list(_DEFAULT_ASSETS)


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    regime_stream = pd.read_parquet(_REGIME_STREAM_PATH, engine="pyarrow")
    aligned = {a: pd.read_parquet(aligned_path(a, _PROCESSED), engine="pyarrow") for a in _ASSETS}
    return regime_stream, aligned


def _merged(regime_stream: pd.DataFrame, aligned: dict[str, pd.DataFrame], asset: str) -> pd.DataFrame:
    """Join regime labels onto ALL of the asset's aligned columns (not a hardcoded 12-
    column subset) — so extra columns a given candidate model actually has (e.g.
    realized_vol_raw, multi-horizon z-scores) flow through to every section below
    without needing their own copy-pasted column list."""
    rs = regime_stream[regime_stream["asset"] == asset].sort_values("timestamp")
    df = aligned[asset]
    sub = rs.set_index("timestamp")[["regime_label"]].join(df, how="inner")
    return sub.dropna(subset=["close_return"])


def _distinctness_for_column(m: pd.DataFrame, column: str) -> dict:
    """ANOVA + pairwise KS across regimes for one column — shared by the close_return
    and realized_vol_raw distinctness checks below."""
    regimes = sorted(m["regime_label"].unique())
    per_regime = {}
    groups = {}
    for r in regimes:
        vals = m.loc[m["regime_label"] == r, column].to_numpy()
        groups[r] = vals
        per_regime[str(r)] = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "skew": float(stats.skew(vals)) if len(vals) > 2 else float("nan"),
        }

    anova_f, anova_p = stats.f_oneway(*groups.values())

    pairwise = {}
    indistinguishable = []
    for r1, r2 in itertools.combinations(regimes, 2):
        ks_stat, ks_p = stats.ks_2samp(groups[r1], groups[r2])
        key = f"{r1}_vs_{r2}"
        pairwise[key] = {"ks_stat": float(ks_stat), "ks_p": float(ks_p)}
        if ks_p > 0.05:
            indistinguishable.append(key)

    return {
        "per_regime": per_regime,
        "anova_f": float(anova_f),
        "anova_p": float(anova_p),
        "pairwise_ks": pairwise,
        "indistinguishable_pairs_p_gt_0.05": indistinguishable,
    }


# ------------------------------------------------------------- 1. distinctness
def section_distinctness(regime_stream: pd.DataFrame, aligned: dict[str, pd.DataFrame]) -> dict:
    out: dict[str, Any] = {}
    for asset in _ASSETS:
        m = _merged(regime_stream, aligned, asset)

        asset_out = {"close_return": _distinctness_for_column(m, "close_return")}
        logger.info(
            f"[distinctness/close_return] {asset}: ANOVA p={asset_out['close_return']['anova_p']:.2e} | "
            f"indistinguishable pairs (KS p>0.05): {asset_out['close_return']['indistinguishable_pairs_p_gt_0.05']}"
        )

        if "realized_vol_raw" in m.columns:
            asset_out["realized_vol_raw"] = _distinctness_for_column(m, "realized_vol_raw")
            logger.info(
                f"[distinctness/realized_vol_raw] {asset}: ANOVA p={asset_out['realized_vol_raw']['anova_p']:.2e} | "
                f"indistinguishable pairs (KS p>0.05): "
                f"{asset_out['realized_vol_raw']['indistinguishable_pairs_p_gt_0.05']}"
            )

        # backward-compatible flat keys (old consumers of this JSON expect these at
        # the top level, not nested under "close_return")
        out[asset] = {**asset_out["close_return"], **asset_out}
    return out


# ---------------------------------------------------------------- 2. stability
def section_stability(regime_stream: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    fig, axes = plt.subplots(len(_ASSETS), 1, figsize=(14, 10), sharex=False)
    for i, asset in enumerate(_ASSETS):
        rs = regime_stream[regime_stream["asset"] == asset].sort_values("timestamp")
        labels = rs["regime_label"].to_numpy()
        ts = rs["timestamp"].to_numpy()
        n_states = int(regime_stream["regime_label"].max()) + 1

        # transition matrix (empirical, from the observed label sequence)
        trans = np.zeros((n_states, n_states))
        for a, b in zip(labels[:-1], labels[1:]):
            trans[a, b] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        trans_prob = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums > 0)

        # dwell time: run-length encode consecutive identical labels
        run_lengths: dict[int, list[int]] = {r: [] for r in range(n_states)}
        cur_label, cur_len = labels[0], 1
        for lab in labels[1:]:
            if lab == cur_label:
                cur_len += 1
            else:
                run_lengths[int(cur_label)].append(cur_len)
                cur_label, cur_len = lab, 1
        run_lengths[int(cur_label)].append(cur_len)
        dwell = {
            str(r): {
                "mean_dwell_hours": float(np.mean(v)) if v else 0.0,
                "median_dwell_hours": float(np.median(v)) if v else 0.0,
                "n_episodes": len(v),
            }
            for r, v in run_lengths.items()
        }

        out[asset] = {
            "transition_matrix": trans_prob.round(4).tolist(),
            "dwell_time": dwell,
        }
        logger.info(
            f"[stability] {asset} mean dwell (hours) per regime: "
            f"{ {r: round(d['mean_dwell_hours'], 1) for r, d in dwell.items()} }"
        )

        ax = axes[i]
        ax.scatter(ts, labels, s=1, alpha=0.5)
        ax.set_ylabel(asset)
        ax.set_yticks(range(n_states))
    axes[-1].set_xlabel("time")
    fig.suptitle("Regime label sequence over time (per asset)")
    fig.tight_layout()
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_PLOTS_DIR / "regime_sequence_all_assets.png", dpi=120)
    plt.close(fig)
    logger.success(f"Saved plot → {_PLOTS_DIR / 'regime_sequence_all_assets.png'}")
    return out


# ------------------------------------------------------- 3. balance/production
def section_balance(regime_stream: pd.DataFrame, hmm_checkpoint: str = "regime/checkpoints/hmm_model.pkl") -> dict:
    hmm = joblib.load(hmm_checkpoint)
    n_states = hmm.n_components
    means = hmm.means_
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(means))

    # nearest-neighbor distance per state — flags near-duplicate states
    np.fill_diagonal(dist_matrix, np.inf)
    nearest = {
        str(s): {
            "nearest_state": int(np.argmin(dist_matrix[s])),
            "distance": float(np.min(dist_matrix[s])),
        }
        for s in range(n_states)
    }
    near_duplicates = [
        (s, v["nearest_state"], v["distance"])
        for s, v in nearest.items()
        if v["distance"] < 0.05  # << the ~0.15-0.28 typical inter-state distance
    ]

    dist_out = {}
    for asset in _ASSETS:
        counts = regime_stream[regime_stream["asset"] == asset]["regime_label"].value_counts().sort_index()
        dist_out[asset] = {str(k): int(v) for k, v in counts.items()}

    out = {
        "production_mechanism": (
            "Confirmed from regime/hmm.py: labels come from "
            "GaussianHMM(covariance_type='diag').predict() — genuine Viterbi decoding "
            "over a PCA(16) embedding of the frozen TFT encoder's 128-dim output, "
            "n_states BIC-selected. NOT quantile/rolling-statistic bucketing."
        ),
        "n_states": n_states,
        "regime_counts_per_asset": dist_out,
        "state_mean_nearest_neighbor_distance": nearest,
        "near_duplicate_states_dist_lt_0.05": near_duplicates,
        "transition_matrix_full_fit": hmm.transmat_.round(4).tolist(),
    }
    logger.info(f"[balance] near-duplicate HMM states (mean distance < 0.05): {near_duplicates}")
    return out


# --------------------------------------------------------- 4. predictive signal
def section_predictive_signal(regime_stream: pd.DataFrame, aligned: dict[str, pd.DataFrame]) -> dict:
    out: dict[str, Any] = {}
    for asset in _ASSETS:
        m = _merged(regime_stream, aligned, asset).sort_index()
        m["target"] = (m["close_return"].shift(-1) > 0).astype(int)
        m = m.iloc[:-1]  # drop last row (no next return)

        train_mask = m.index < pd.Timestamp("2024-01-01", tz="UTC")
        test_mask = (m.index >= pd.Timestamp("2024-01-01", tz="UTC")) & (
            m.index <= pd.Timestamp("2024-12-31", tz="UTC")
        )
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            out[asset] = {"error": "insufficient train/test rows"}
            continue

        regime_dummies = pd.get_dummies(m["regime_label"], prefix="regime").astype(float)
        # base: the original 12-feature set, for comparability with historical runs.
        feat_cols_base = [c for c in _FEATURE_COLS if c in m.columns]
        # extended: every non-target, non-label column this candidate model's aligned
        # data actually has (e.g. realized_vol_raw, multi-horizon z-scores) — the
        # original audit never tested with the fuller set a candidate was built from.
        feat_cols_extended = [c for c in m.columns if c not in ("regime_label", "close_return", "target")]

        X_regime = regime_dummies
        X_feat = m[feat_cols_base]
        X_feat_ext = m[feat_cols_extended]
        X_both = pd.concat([regime_dummies, m[feat_cols_base]], axis=1)
        X_both_ext = pd.concat([regime_dummies, m[feat_cols_extended]], axis=1)
        y = m["target"]

        variants = [
            ("regime_only", X_regime),
            ("features_only", X_feat),
            ("regime_plus_features", X_both),
        ]
        if feat_cols_extended != feat_cols_base:
            variants += [
                ("features_only_extended", X_feat_ext),
                ("regime_plus_features_extended", X_both_ext),
            ]

        results = {}
        for name, X in variants:
            Xtr, Xte = X[train_mask], X[test_mask]
            ytr, yte = y[train_mask], y[test_mask]
            asset_res = {}
            for model_name, model in [
                ("logistic", LogisticRegression(max_iter=1000)),
                ("gbt", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
            ]:
                model.fit(Xtr, ytr)
                proba = model.predict_proba(Xte)[:, 1]
                pred = model.predict(Xte)
                try:
                    auc = float(roc_auc_score(yte, proba))
                except ValueError:
                    auc = float("nan")
                asset_res[model_name] = {
                    "accuracy": float(accuracy_score(yte, pred)),
                    "auc": auc,
                }
            results[name] = asset_res
        out[asset] = results
        summary = "  ".join(f"{name}={results[name]['logistic']['auc']:.3f}" for name, _ in variants)
        logger.info(f"[predictive] {asset} AUC (logistic): {summary}")
    return out


# ----------------------------------------------------------- 5. embedding usage
def section_embedding_usage(checkpoint: str = "rl/checkpoints/ppo_best.zip", n_obs: int = 500) -> dict:
    import torch
    from stable_baselines3 import PPO

    from rl.policy import MARTAActorCriticPolicy
    from rl.train import build_env_factory, load_data as rl_load_data, load_debate_results

    config = {
        "device": "cpu", "initial_capital": 10_000.0, "window_size": 60,
        "transaction_cost": 0.001, "lambda_drawdown": 0.01, "max_leverage": 1.0,
        "max_episode_steps": 2000, "all_assets": _ASSETS,
    }
    aligned_dfs, regime_stream_df, encoder = rl_load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)
    env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, device="cpu")

    model = PPO.load(checkpoint, device="cpu", custom_objects={"policy_class": MARTAActorCriticPolicy})
    policy = model.policy
    policy.eval()

    obs, _ = env.reset(seed=42)
    obs_batch = [obs]
    for _ in range(n_obs - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        obs_batch.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()
    obs_arr = np.stack(obs_batch).astype(np.float32)

    obs_tensor = torch.tensor(obs_arr, requires_grad=True)
    latent = policy.shared_net(policy.extract_features(obs_tensor))
    value = policy.value_head(latent)
    dir_logits = policy.direction_head(latent)
    pos_params = policy.position_head(latent)
    # sum multiple heads' outputs so gradient reflects usage across the whole policy,
    # not just the value function
    scalar = value.sum() + dir_logits.sum() + pos_params.sum()
    scalar.backward()

    grad = obs_tensor.grad.abs().mean(dim=0).detach().numpy()  # (152,)
    regime_grad = grad[:128]
    consensus_grad = grad[128:144]
    portfolio_grad = grad[144:152]

    # static, architecture-level complement: first-layer weight column norms per block
    w0 = policy.shared_net[0].weight.detach().numpy()  # (256, 152)
    col_norms = np.linalg.norm(w0, axis=0)
    regime_w = col_norms[:128]
    consensus_w = col_norms[128:144]
    portfolio_w = col_norms[144:152]

    out = {
        "checkpoint": checkpoint,
        "n_obs_sampled": int(len(obs_arr)),
        "input_gradient_saliency": {
            "regime_block_mean_abs_grad": float(regime_grad.mean()),
            "consensus_block_mean_abs_grad": float(consensus_grad.mean()),
            "portfolio_block_mean_abs_grad": float(portfolio_grad.mean()),
            "regime_block_total_abs_grad": float(regime_grad.sum()),
            "consensus_block_total_abs_grad": float(consensus_grad.sum()),
            "portfolio_block_total_abs_grad": float(portfolio_grad.sum()),
        },
        "first_layer_weight_norm": {
            "regime_block_mean_col_norm": float(regime_w.mean()),
            "consensus_block_mean_col_norm": float(consensus_w.mean()),
            "portfolio_block_mean_col_norm": float(portfolio_w.mean()),
        },
    }
    logger.info(
        f"[embedding usage] mean |grad| per input dim — regime={regime_grad.mean():.5f} "
        f"consensus={consensus_grad.mean():.5f} portfolio={portfolio_grad.mean():.5f}"
    )
    return out


def main() -> None:
    global _PROCESSED, _REGIME_STREAM_PATH, _RESULTS_PATH, _ASSETS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime-stream", type=str, default=None,
        help="Path to a regime stream parquet. Default: auto-resolved from --processed-dir "
             "via resolve_regime_stream_path() (prefers regime_stream_causal.parquet, falls "
             "back to the leaky regime_stream.parquet if no causal version exists there).",
    )
    parser.add_argument(
        "--processed-dir", type=str, default=None,
        help="Dir containing {asset}_aligned.parquet (default: data/processed, production)",
    )
    parser.add_argument(
        "--assets", type=str, default=None,
        help="Comma-separated logical asset ids (default: BTC_USDT,ETH_USDT,EURUSD,GOLD, production)",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Suffix for the output filename (data/results/regime_quality_audit_{tag}.json). "
             "Default: none (writes regime_quality_audit.json, the production filename).",
    )
    parser.add_argument(
        "--sections", type=str, default="1,2,3,4,5",
        help="Comma-separated section numbers to run (5 is PPO/RL-specific, requires a "
             "checkpoint trained on this exact regime encoding — most candidate models "
             "being audited won't have one).",
    )
    parser.add_argument(
        "--hmm-checkpoint", type=str, default="regime/checkpoints/hmm_model.pkl",
        help="HMM checkpoint for section 3 (balance/near-duplicate-state check).",
    )
    args = parser.parse_args()

    if args.processed_dir:
        _PROCESSED = Path(args.processed_dir)
    if args.regime_stream:
        # Explicit override — respected exactly as given, including a deliberate
        # request for the leaky stream (e.g. a causal-vs-leaky comparison run).
        _REGIME_STREAM_PATH = Path(args.regime_stream)
    elif args.processed_dir:
        _REGIME_STREAM_PATH = resolve_regime_stream_path(_PROCESSED)
    if args.assets:
        _ASSETS = args.assets.split(",")
    if args.tag:
        _RESULTS_PATH = Path(f"data/results/regime_quality_audit_{args.tag}.json")

    sections = {int(s) for s in args.sections.split(",")}
    logger.info(f"regime_stream={_REGIME_STREAM_PATH}  processed_dir={_PROCESSED}  "
                f"assets={_ASSETS}  sections={sorted(sections)}")

    regime_stream, aligned = load_data()

    results: dict[str, Any] = {}
    if 1 in sections:
        results["1_distinctness"] = section_distinctness(regime_stream, aligned)
    if 2 in sections:
        results["2_stability"] = section_stability(regime_stream)
    if 3 in sections:
        results["3_balance_production"] = section_balance(regime_stream, args.hmm_checkpoint)
    if 4 in sections:
        results["4_predictive_signal"] = section_predictive_signal(regime_stream, aligned)
    if 5 in sections:
        try:
            results["5_embedding_usage"] = section_embedding_usage()
        except Exception as exc:
            logger.warning(f"Embedding usage section failed: {exc}")
            results["5_embedding_usage"] = {"error": str(exc)}

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Full audit written → {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
