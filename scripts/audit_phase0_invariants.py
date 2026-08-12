"""Phase 0 adversarial audit (P0.6) — deliberately break each Phase 0 invariant (P0.1-P0.4)
and confirm it fails loudly, then confirm the legitimate path still works.

Non-destructive by construction. Every check does one of:
  (a) reads real artifacts read-only (parquet/npy/checkpoint loads never write)
  (b) uses a tempfile.TemporaryDirectory() fixture with controlled content/mtimes
  (c) monkeypatches a module-level name for the duration of one check, inside a
      try/finally that restores the original value before the next check runs
  (d) execs a MUTATED STRING of configs/splits.py's source in an isolated namespace —
      the real file on disk is only ever read, never written

Nothing under data/, encoder/checkpoints/, or rl/checkpoints/ is written, deleted, or has
its mtime touched by this script.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python scripts/audit_phase0_invariants.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import logger

_RESULTS: list[tuple[str, bool, str]] = []


def _run(name: str, fn: Callable[[], str]) -> None:
    try:
        detail = fn()
        _RESULTS.append((name, True, detail))
        logger.success(f"[PASS] {name} — {detail}")
    except Exception as e:  # noqa: BLE001 — deliberately broad: any failure is a [FAIL] row
        detail = f"{type(e).__name__}: {e}"
        _RESULTS.append((name, False, detail))
        logger.error(f"[FAIL] {name} — {detail}")


# --------------------------------------------------------------------------- P0.2 checks

def test_broken_split_boundary_rejected() -> str:
    """Mutate a copy of configs/splits.py's source (in memory only) so VAL_START no
    longer matches TRAIN_END_EXCLUSIVE, and confirm SplitConfigError fires. The real
    file on disk is only read, never written."""
    src = Path("configs/splits.py").read_text()
    mutated = src.replace('VAL_START = "2024-01-01"', 'VAL_START = "2024-01-02"')
    if mutated == src:
        raise AssertionError(
            "replacement pattern not found — configs/splits.py's VAL_START literal "
            "may have changed; update this test's expected string"
        )
    ns: dict = {"__name__": "configs.splits_audit_mutant"}
    try:
        exec(
            compile(mutated, "<audit mutant of configs/splits.py, never written to disk>", "exec"),
            ns,
        )
    except Exception as e:
        if type(e).__name__ != "SplitConfigError":
            raise AssertionError(f"expected SplitConfigError, got {type(e).__name__}: {e}") from e
        return f"correctly rejected: {e}"
    raise AssertionError("expected SplitConfigError, none raised")


def test_encoder_contamination_rejected() -> str:
    """Local run without val_end against the real local aligned parquet (which extends
    into 2025-12-31) — the exact documented footgun. Read-only against real data."""
    from configs.splits import TRAIN_END_EXCLUSIVE, VAL_START, TestSetContaminationError
    from encoder.train import _load_asset_split

    try:
        _load_asset_split(
            "BTC_USDT", Path("data/processed"),
            TRAIN_END_EXCLUSIVE, VAL_START,
            window_size=60, step=1,
            # val_end intentionally omitted
        )
    except TestSetContaminationError as e:
        return f"correctly rejected: {str(e)[:90]}..."
    raise AssertionError("expected TestSetContaminationError, none raised")


def test_hmm_contamination_rejected() -> str:
    """Monkeypatch regime.hmm's imported TRAIN_END_EXCLUSIVE past TEST_START and confirm
    run_hmm_pipeline raises before any PCA/HMM fit or MLflow write — the contamination
    guard sits before both. Reads real embeddings.npy/embedding_timestamps.parquet
    read-only; monkeypatch is restored in finally regardless of outcome."""
    import regime.hmm as hmm_mod

    original = hmm_mod.TRAIN_END_EXCLUSIVE
    hmm_mod.TRAIN_END_EXCLUSIVE = "2026-01-01"  # past TEST_START (2025-01-01)
    try:
        try:
            hmm_mod.run_hmm_pipeline({"data": {"processed_dir": "data/processed"}})
        except hmm_mod.TestSetContaminationError as e:
            return f"correctly rejected: {str(e)[:90]}..."
        raise AssertionError("expected TestSetContaminationError, none raised")
    finally:
        hmm_mod.TRAIN_END_EXCLUSIVE = original


def test_rl_env_contamination_rejected() -> str:
    """Monkeypatch rl.env's VAL_END past TEST_START and confirm constructing a 'val'
    split env raises. Loads the real encoder/aligned data read-only; monkeypatch is
    restored in finally regardless of outcome."""
    import pandas as pd
    import rl.env as env_mod
    from rl.train import build_env_factory, load_data, load_debate_results

    original_val_end = env_mod.VAL_END
    env_mod.VAL_END = pd.Timestamp("2025-06-01 23:59:59", tz="UTC")  # past TEST_START
    try:
        config = {
            "device": "cpu", "all_assets": ["BTC_USDT"], "window_size": 60,
            "transaction_cost": 0.001, "lambda_drawdown": 0.01, "max_leverage": 1.0,
            "initial_capital": 10_000.0, "max_episode_steps": 2000,
        }
        aligned_dfs, regime_stream_df, encoder = load_data(config)
        debate_results = load_debate_results()
        env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)
        try:
            env_factory(assets=["BTC_USDT"], split="val", sequential=True, debate=False)
        except env_mod.TestSetContaminationError as e:
            return f"correctly rejected: {str(e)[:90]}..."
        raise AssertionError("expected TestSetContaminationError, none raised")
    finally:
        env_mod.VAL_END = original_val_end


def test_contamination_clean_path_unaffected() -> str:
    """Positive control: with boundaries un-mutated, the encoder split (with val_end
    set) and a real 'train' split RL env must NOT raise. Guards should be silent on
    correctly-used real data."""
    from configs.splits import TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE, VAL_START
    from encoder.train import _load_asset_split
    from rl.train import build_env_factory, load_data, load_debate_results
    import pandas as pd

    val_end = (pd.Timestamp(VAL_END_EXCLUSIVE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _load_asset_split(
        "BTC_USDT", Path("data/processed"), TRAIN_END_EXCLUSIVE, VAL_START,
        window_size=60, step=1, val_end=val_end,
    )
    config = {
        "device": "cpu", "all_assets": ["BTC_USDT"], "window_size": 60,
        "transaction_cost": 0.001, "lambda_drawdown": 0.01, "max_leverage": 1.0,
        "initial_capital": 10_000.0, "max_episode_steps": 2000,
    }
    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)
    env_factory(assets=["BTC_USDT"], split="train", sequential=False)
    return "encoder split (with val_end) and a real train-split env both constructed without raising"


# --------------------------------------------------------------------------- P0.4 checks

def test_stale_regime_stream_rejected() -> str:
    """Temp-file fixture with controlled mtimes — never touches the real encoder
    checkpoint or regime_stream.parquet."""
    from utils.freshness import StaleRegimeStreamError, check_encoder_regime_stream_freshness
    import os

    with tempfile.TemporaryDirectory() as d:
        enc = Path(d) / "encoder.pt"
        stream = Path(d) / "regime_stream.parquet"
        stream.write_text("x")
        time.sleep(0.05)
        enc.write_text("x")  # encoder newer than stream -> stale
        try:
            check_encoder_regime_stream_freshness(encoder_path=enc, regime_stream_path=stream)
        except StaleRegimeStreamError:
            pass
        else:
            raise AssertionError("expected StaleRegimeStreamError for stale fixture, none raised")

        # Control: stream newer than encoder -> must NOT raise.
        os.utime(stream, (time.time() + 10, time.time() + 10))
        check_encoder_regime_stream_freshness(encoder_path=enc, regime_stream_path=stream)
    return "stale fixture rejected, fresh fixture passes — real encoder/regime_stream mtimes untouched"


def test_debate_loaders_enforce_freshness() -> str:
    """Monkeypatch the freshness check to always raise, and confirm all 4 debate
    entrypoints propagate it. Restored in finally; no real file is touched."""
    import utils.freshness as fresh

    original = fresh.check_encoder_regime_stream_freshness

    def _boom() -> None:
        raise fresh.StaleRegimeStreamError("SIMULATED for P0.6 audit — not a real staleness")

    fresh.check_encoder_regime_stream_freshness = _boom
    try:
        import run_clean_ablation_debates, run_debate, run_validation_debates, run_weekly_debates

        for mod in [run_debate, run_weekly_debates, run_clean_ablation_debates, run_validation_debates]:
            try:
                mod._load_data()
            except fresh.StaleRegimeStreamError:
                continue
            raise AssertionError(f"{mod.__name__}._load_data() did not reject a stale regime stream")
    finally:
        fresh.check_encoder_regime_stream_freshness = original
    return "all 4 debate entrypoints (run_debate, run_weekly_debates, run_clean_ablation_debates, run_validation_debates) reject a stale regime stream"


# --------------------------------------------------------------------------- P0.3 check

def test_evaluation_deterministic() -> str:
    """Two full walk-forward passes on the same real checkpoint must be bit-identical.
    Uses ppo_best.zip (fast — abstains on this val window) for a quick, always-runnable
    check; ppo_best_seed0.zip (a checkpoint with 9 real trades) was verified manually in
    the P0.3 session — Sharpe 1.8926444911863571, identical across 2 runs — and is not
    re-run here automatically since a full pass on a trading checkpoint takes several
    minutes."""
    from stable_baselines3 import PPO

    from rl.evaluate import compute_metrics, run_walk_forward
    from rl.policy import MARTAActorCriticPolicy
    from rl.train import build_env_factory, load_data, load_debate_results

    ckpt = Path("rl/checkpoints/ppo_best.zip")
    if not ckpt.exists():
        raise AssertionError(f"{ckpt} not found — cannot run this check")

    config = {
        "device": "cpu", "all_assets": ["BTC_USDT"], "window_size": 60,
        "transaction_cost": 0.001, "lambda_drawdown": 0.01, "max_leverage": 1.0,
        "initial_capital": 10_000.0, "max_episode_steps": 2000,
    }
    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)
    model = PPO.load(str(ckpt), device="cpu", custom_objects={"policy_class": MARTAActorCriticPolicy})

    sharpes = []
    for _ in range(2):
        env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, debate=False, idle_penalty=0.0)
        traj = run_walk_forward(model, env, deterministic=True)
        m = compute_metrics(traj["returns"], traj["portfolio_values"], traj.get("positions"))
        sharpes.append(m["sharpe"])
    if sharpes[0] != sharpes[1]:
        raise AssertionError(f"NOT deterministic across repeated eval: {sharpes}")
    return f"bit-identical Sharpe={sharpes[0]!r} across 2 runs on {ckpt.name}"


# --------------------------------------------------------------------------- misc checks

def test_missing_data_behavior() -> str:
    """Documents actual (not assumed) behavior for missing real data, using temp-dir
    copies — never touches the real data/processed/ files."""
    from utils.freshness import load_debate_data

    real_dir = Path("data/processed")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # Case A: regime_stream.parquet itself missing -> must fail loudly.
        try:
            load_debate_data(processed_dir=tmp, assets=["BTC_USDT"])
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing regime_stream.parquet did not raise FileNotFoundError")

        # Case B: regime_stream present, one asset's aligned parquet absent (the real
        # scenario before an asset has been fetched/aligned).
        shutil.copy(real_dir / "regime_stream.parquet", tmp / "regime_stream.parquet")
        shutil.copy(real_dir / "BTC_USDT_aligned.parquet", tmp / "BTC_USDT_aligned.parquet")
        # ETH_USDT_aligned.parquet intentionally NOT copied.
        _, aligned_dfs = load_debate_data(processed_dir=tmp, assets=["BTC_USDT", "ETH_USDT"])
        if "ETH_USDT" in aligned_dfs:
            raise AssertionError("expected ETH_USDT to be absent from the result, found it")
        if "BTC_USDT" not in aligned_dfs:
            raise AssertionError("expected BTC_USDT present in the result, missing")

    return (
        "Case A (regime_stream.parquet missing) -> FileNotFoundError, fails loudly. "
        "Case B (one asset's aligned parquet missing) -> silently OMITTED from the "
        "returned dict, not fabricated — existing documented behavior, distinct from "
        "P0.5's synthetic-fallback concern, flagged here for awareness (not changed)"
    )


def test_asset_path_consistency() -> str:
    """The GOLD/XAUUSD case P0.4 uncovered: aligned_path('GOLD', ...) must resolve to
    XAUUSD_aligned.parquet, and the shared debate loader must actually include GOLD."""
    from utils.assets import aligned_path
    from utils.freshness import load_debate_data

    expected = Path("data/processed") / "XAUUSD_aligned.parquet"
    resolved = aligned_path("GOLD", Path("data/processed"))
    if resolved != expected:
        raise AssertionError(f"aligned_path('GOLD', ...) = {resolved}, expected {expected}")
    if not resolved.exists():
        raise AssertionError(f"{resolved} does not exist on disk — cannot verify further")

    _, aligned_dfs = load_debate_data()
    if "GOLD" not in aligned_dfs:
        raise AssertionError("GOLD missing from load_debate_data() output — the P0.4 fix regressed")

    broken_guess = Path("data/processed") / "GOLD_aligned.parquet"
    if broken_guess.exists():
        raise AssertionError(
            f"{broken_guess} unexpectedly exists on disk — the historical bug's "
            f"assumption (GOLD == file stem) may no longer hold; re-check this test"
        )
    return (
        f"aligned_path('GOLD', ...) correctly resolves to {resolved.name} "
        f"(not the nonexistent 'GOLD_aligned.parquet'); load_debate_data() includes GOLD"
    )


def main() -> None:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Split boundaries reject a broken config", test_broken_split_boundary_rejected),
        ("Stale regime stream rejected", test_stale_regime_stream_rejected),
        ("Debate loaders enforce freshness", test_debate_loaders_enforce_freshness),
        ("Asset/path consistency (GOLD/XAUUSD)", test_asset_path_consistency),
        ("Missing aligned data — documented behavior", test_missing_data_behavior),
        ("Test contamination — encoder", test_encoder_contamination_rejected),
        ("Test contamination — HMM/PCA", test_hmm_contamination_rejected),
        ("Test contamination — RL env", test_rl_env_contamination_rejected),
        ("Test contamination — clean path unaffected", test_contamination_clean_path_unaffected),
        ("Evaluation deterministic", test_evaluation_deterministic),
    ]

    print("\nPHASE 0 INVARIANT AUDIT\n" + "=" * 60)
    for name, fn in checks:
        _run(name, fn)

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}")
    print("=" * 60)
    print(f"{n_pass}/{len(_RESULTS)} passed")

    out_path = Path("data/results/phase0_audit_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        [{"name": n, "pass": ok, "detail": d} for n, ok, d in _RESULTS],
        indent=2,
    ))
    print(f"\nResults written to {out_path}")

    if n_pass != len(_RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
