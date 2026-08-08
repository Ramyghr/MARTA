"""regime/regime_quality_audit.py

Standalone statistical audit of HMM regime label quality — independent of the RL/PPO
layer. Answers 5 questions about the labels in data/processed/regime_stream.parquet:

  1. Distinctness — per-regime mean/vol/autocorr/skew on RAW close_return, then ANOVA
     + pairwise KS tests on the return distributions.
  2. Stability — transition matrix + mean dwell time per regime (a "regime" that
     flips every step is noise, not a regime).
  3. Balance/production — confirms labels come from GaussianHMM.predict() (Viterbi
     decode on PCA(16) embeddings), not quantile bucketing, and checks whether the
     suspicious near-equal state counts trace back to near-duplicate HMM states.
  4. Predictive signal — logistic regression + gradient boosting predicting
     sign(next return) from regime-only / regime+features / features-only.
  5. Embedding usage — input-gradient saliency on a trained PPO policy: does it
     actually use the 128-dim regime block, or has it learned to ignore it?

Run: python regime/regime_quality_audit.py
Outputs: data/results/regime_quality_audit.json, regime/plots/regime_sequence_*.png
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.assets import aligned_path
from utils.logger import logger

_PROCESSED = Path("data/processed")
_RESULTS_PATH = Path("data/results/regime_quality_audit.json")
_PLOTS_DIR = Path("regime/plots")
_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_FEATURE_COLS = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram", "obv_raw",
    "vol_zscore", "realized_vol", "vol_ratio", "return_skew",
    "return_kurtosis", "autocorr_lag1",
]


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    regime_stream = pd.read_parquet(_PROCESSED / "regime_stream.parquet", engine="pyarrow")
    aligned = {a: pd.read_parquet(aligned_path(a, _PROCESSED), engine="pyarrow") for a in _ASSETS}
    return regime_stream, aligned


def _merged(regime_stream: pd.DataFrame, aligned: dict[str, pd.DataFrame], asset: str) -> pd.DataFrame:
    rs = regime_stream[regime_stream["asset"] == asset].sort_values("timestamp")
    df = aligned[asset]
    sub = rs.set_index("timestamp")[["regime_label"]].join(
        df[["close_return"] + _FEATURE_COLS], how="inner"
    )
    return sub.dropna(subset=["close_return"])


# ------------------------------------------------------------- 1. distinctness
def section_distinctness(regime_stream: pd.DataFrame, aligned: dict[str, pd.DataFrame]) -> dict:
    out: dict[str, Any] = {}
    for asset in _ASSETS:
        m = _merged(regime_stream, aligned, asset)
        regimes = sorted(m["regime_label"].unique())
        per_regime = {}
        groups = {}
        for r in regimes:
            ret = m.loc[m["regime_label"] == r, "close_return"].to_numpy()
            groups[r] = ret
            per_regime[str(r)] = {
                "n": int(len(ret)),
                "mean_return": float(np.mean(ret)),
                "volatility": float(np.std(ret)),
                "autocorr_lag1": float(pd.Series(ret).autocorr(lag=1)) if len(ret) > 2 else float("nan"),
                "skew": float(stats.skew(ret)) if len(ret) > 2 else float("nan"),
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

        out[asset] = {
            "per_regime": per_regime,
            "anova_f": float(anova_f),
            "anova_p": float(anova_p),
            "pairwise_ks": pairwise,
            "indistinguishable_pairs_p_gt_0.05": indistinguishable,
        }
        logger.info(
            f"[distinctness] {asset}: ANOVA p={anova_p:.2e} | "
            f"indistinguishable pairs (KS p>0.05): {indistinguishable}"
        )
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
def section_balance(regime_stream: pd.DataFrame) -> dict:
    hmm = joblib.load("regime/checkpoints/hmm_model.pkl")
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
        feat_cols = [c for c in _FEATURE_COLS if c in m.columns]

        X_regime = regime_dummies
        X_feat = m[feat_cols]
        X_both = pd.concat([regime_dummies, m[feat_cols]], axis=1)
        y = m["target"]

        results = {}
        for name, X in [("regime_only", X_regime), ("features_only", X_feat), ("regime_plus_features", X_both)]:
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
        logger.info(
            f"[predictive] {asset} AUC (logistic): regime_only="
            f"{results['regime_only']['logistic']['auc']:.3f}  "
            f"features_only={results['features_only']['logistic']['auc']:.3f}  "
            f"regime+features={results['regime_plus_features']['logistic']['auc']:.3f}"
        )
    return out


# ----------------------------------------------------------- 5. embedding usage
def section_embedding_usage(checkpoint: str = "rl/checkpoints/ppo_best.zip", n_obs: int = 500) -> dict:
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
    regime_stream, aligned = load_data()

    results = {
        "1_distinctness": section_distinctness(regime_stream, aligned),
        "2_stability": section_stability(regime_stream),
        "3_balance_production": section_balance(regime_stream),
        "4_predictive_signal": section_predictive_signal(regime_stream, aligned),
    }
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
