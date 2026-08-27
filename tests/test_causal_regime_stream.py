"""Tests for regime/causal_regime_stream.py — the walk-forward causal replacement for
regime/hmm.py's generate_regime_stream() (which is provably non-causal, see that
module's docstring for the audit finding).

The property under test that actually matters: perturbing a FUTURE row must never change
an EARLIER row's computed regime probability. That's the entire bug this module exists to
fix (predict_proba's forward-backward smoothing lets future rows leak backward), so it's
the one test that would fail against the old regime/hmm.py:generate_regime_stream()
approach and must pass here.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from regime.causal_regime_stream import generate_causal_regime_stream
from regime.hmm import generate_regime_stream as leaky_generate_regime_stream

N_STATES = 3
N_FEATURES = 4
N_ROWS = 60
LOOKBACK = 10


def _make_synthetic_hmm(seed: int = 0) -> GaussianHMM:
    rng = np.random.default_rng(seed)
    model = GaussianHMM(n_components=N_STATES, covariance_type="diag", random_state=seed)
    # Set params directly rather than fitting — a fitted-looking, valid, fixed model is
    # all these tests need, and avoids EM non-determinism/convergence noise.
    model.startprob_ = np.full(N_STATES, 1.0 / N_STATES)
    trans = rng.dirichlet(np.ones(N_STATES) * 5, size=N_STATES)  # diagonal-heavy-ish
    model.transmat_ = trans
    model.means_ = rng.standard_normal((N_STATES, N_FEATURES)) * 2
    model.covars_ = np.abs(rng.standard_normal((N_STATES, N_FEATURES))) + 0.5
    model.n_features = N_FEATURES
    return model


@pytest.fixture()
def hmm_checkpoint(tmp_path: Path) -> Path:
    model = _make_synthetic_hmm()
    path = tmp_path / "hmm_model.pkl"
    joblib.dump(model, path)
    return path


@pytest.fixture()
def processed_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    d = tmp_path / "processed"
    d.mkdir()
    idx = pd.date_range("2020-01-01", periods=N_ROWS, freq="1h", tz="UTC")
    cols = [f"f{i}" for i in range(N_FEATURES)]
    for asset in ("ASSET_A", "ASSET_B"):
        data = rng.standard_normal((N_ROWS, N_FEATURES))
        df = pd.DataFrame(data, index=idx, columns=cols)
        df.to_parquet(d / f"{asset}_aligned.parquet", engine="pyarrow")
    return d


def test_output_schema(processed_dir: Path, hmm_checkpoint: Path):
    out = generate_causal_regime_stream(
        processed_dir, hmm_checkpoint, ["ASSET_A", "ASSET_B"],
        lookback_hours=LOOKBACK, n_jobs=1,
    )
    assert set(out["asset"].unique()) == {"ASSET_A", "ASSET_B"}
    assert len(out) == 2 * N_ROWS
    for i in range(N_STATES):
        assert f"regime_prob_{i}" in out.columns
    probs = out[[f"regime_prob_{i}" for i in range(N_STATES)]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (out["regime_label"] == probs.argmax(axis=1)).all()


def test_no_nans_even_before_full_lookback(processed_dir: Path, hmm_checkpoint: Path):
    """The first LOOKBACK-1 rows use a growing window, not a skipped/NaN one."""
    out = generate_causal_regime_stream(
        processed_dir, hmm_checkpoint, ["ASSET_A"], lookback_hours=LOOKBACK, n_jobs=1,
    )
    assert out.isna().sum().sum() == 0
    assert len(out) == N_ROWS


def test_causal_invariance_perturbing_future_does_not_change_past(
    processed_dir: Path, hmm_checkpoint: Path,
):
    """The core property. Compute the causal stream once, then overwrite every row
    AFTER a cutoff with random noise and recompute — every row AT OR BEFORE the cutoff
    must produce bit-identical probabilities. This is the property regime/hmm.py's
    generate_regime_stream() (predict_proba over the whole series at once) violates,
    since its backward pass lets rows after the cutoff influence rows before it.
    """
    baseline = generate_causal_regime_stream(
        processed_dir, hmm_checkpoint, ["ASSET_A"], lookback_hours=LOOKBACK, n_jobs=1,
    )

    cutoff = N_ROWS // 2
    df = pd.read_parquet(processed_dir / "ASSET_A_aligned.parquet", engine="pyarrow")
    rng = np.random.default_rng(999)
    perturbed = df.copy()
    perturbed.iloc[cutoff:] = rng.standard_normal(perturbed.iloc[cutoff:].shape) * 100
    perturbed.to_parquet(processed_dir / "ASSET_A_aligned.parquet", engine="pyarrow")

    perturbed_result = generate_causal_regime_stream(
        processed_dir, hmm_checkpoint, ["ASSET_A"], lookback_hours=LOOKBACK, n_jobs=1,
    )

    prob_cols = [f"regime_prob_{i}" for i in range(N_STATES)]
    base_past = baseline.iloc[:cutoff][prob_cols].to_numpy()
    pert_past = perturbed_result.iloc[:cutoff][prob_cols].to_numpy()
    np.testing.assert_allclose(
        base_past, pert_past, atol=1e-10,
        err_msg="Perturbing FUTURE rows changed PAST rows' regime probabilities — the "
                "causal-invariance property this module exists to guarantee is broken.",
    )

    # Sanity: the perturbed future rows themselves should generally differ (proves the
    # perturbation actually had an effect somewhere, i.e. this isn't a vacuously-passing
    # test where nothing ever changes).
    base_future = baseline.iloc[cutoff:][prob_cols].to_numpy()
    pert_future = perturbed_result.iloc[cutoff:][prob_cols].to_numpy()
    assert not np.allclose(base_future, pert_future, atol=1e-6)


def test_leaky_baseline_actually_violates_causality(processed_dir: Path, hmm_checkpoint: Path):
    """Negative control: confirm the OLD (regime/hmm.py) approach really is non-causal
    on this same synthetic setup, so the previous test isn't passing because the
    synthetic HMM happens to be insensitive to perturbation. If this test ever starts
    failing, it means the HMM/data fixture stopped being able to detect the leak at all,
    and the positive test above would need a stronger fixture to mean anything.
    """
    df_a = pd.read_parquet(processed_dir / "ASSET_A_aligned.parquet", engine="pyarrow")
    model = joblib.load(hmm_checkpoint)
    cutoff = N_ROWS // 2

    timestamps = pd.DataFrame({"timestamp": df_a.index, "asset": "ASSET_A"})

    mlflow.set_tracking_uri(f"file:{processed_dir.parent / 'mlruns_test'}")
    mlflow.set_experiment("test_causal_regime_stream")
    with mlflow.start_run():
        baseline = leaky_generate_regime_stream(df_a.to_numpy(), timestamps, None, model, None)

        rng = np.random.default_rng(999)
        perturbed_vals = df_a.to_numpy().copy()
        perturbed_vals[cutoff:] = rng.standard_normal(perturbed_vals[cutoff:].shape) * 100
        perturbed = leaky_generate_regime_stream(perturbed_vals, timestamps, None, model, None)

    prob_cols = [f"regime_prob_{i}" for i in range(N_STATES)]
    base_past = baseline.iloc[:cutoff][prob_cols].to_numpy()
    pert_past = perturbed.iloc[:cutoff][prob_cols].to_numpy()
    # Expect a real difference somewhere in the "past" half — proves predict_proba's
    # smoothing really does let the future perturbation leak backward.
    assert not np.allclose(base_past, pert_past, atol=1e-6), (
        "Expected the leaky generator to be affected by a future perturbation "
        "(that's the bug being fixed) — got no measurable difference."
    )
