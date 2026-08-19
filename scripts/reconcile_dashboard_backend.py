"""Reconcile backend evaluation output against what the dashboard actually displays.

Backend "ground truth" = data/results/ppo_evaluation.json (+ ablation_results.json),
the files rl/evaluate.py writes and dashboard/backend/{portfolio_service,rl_service,
system_service}.py read verbatim. "Dashboard value" = a live call to the running
FastAPI backend's /api endpoints (falls back to importing the service modules
directly if the API isn't reachable, so this still works with no backend running).

Usage:
    python scripts/reconcile_dashboard_backend.py [--base-url http://localhost:8001]

Exits non-zero if any metric is flagged WRONG (a genuine value mismatch) or if
provenance shows the canonical eval file doesn't describe the checkpoint currently
on disk. STALE (a fresher eval file/checkpoint exists but hasn't been promoted to
the canonical path) is reported but does not fail the run — it's a signal for the
next canonical re-run, not this run's data being wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

MARTA_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = MARTA_ROOT / "data" / "results"
CHECKPOINTS_DIR = MARTA_ROOT / "rl" / "checkpoints"

CANONICAL_EVAL = RESULTS_DIR / "ppo_evaluation.json"
CANONICAL_ABLATION = RESULTS_DIR / "ablation_results.json"
CANONICAL_CHECKPOINT = CHECKPOINTS_DIR / "ppo_best.zip"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    import datetime
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()


def _fetch_dashboard_live(base_url: str) -> dict[str, Any] | None:
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=10.0) as client:
            portfolio = client.get(f"{base_url}/api/portfolio").json()
            equity_curve = client.get(f"{base_url}/api/rl/equity-curve").json()
            positions = client.get(f"{base_url}/api/rl/positions").json()
            system_models = client.get(f"{base_url}/api/system/models").json()
        return {
            "mode": "live_api",
            "base_url": base_url,
            "portfolio": portfolio,
            "equity_curve": equity_curve,
            "positions": positions,
            "system_models": system_models,
        }
    except Exception:
        return None


def _fetch_dashboard_direct() -> dict[str, Any]:
    """Import the service modules directly — used when no backend is running."""
    sys.path.insert(0, str(MARTA_ROOT / "dashboard" / "backend"))
    import portfolio_service
    import rl_service
    import system_service

    return {
        "mode": "direct_import",
        "base_url": None,
        "portfolio": portfolio_service.get_portfolio(),
        "equity_curve": rl_service.get_equity_curve(),
        "positions": rl_service.get_positions(),
        "system_models": {"rl": system_service.get_rl_info()},
    }


def _win_rate_from_trades(trades: list[dict]) -> float | None:
    """Mirror dashboard/frontend/src/app/analytics/page.tsx's client-side win-rate calc."""
    meaningful = [t for t in trades if t.get("action") != "hold"]
    if not meaningful:
        return None
    wins = [t for t in meaningful if t.get("pnl", 0) > 0]
    return len(wins) / len(meaningful)


def _isclose(a: Any, b: Any, rel_tol: float = 1e-6) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-9)
    except (TypeError, ValueError):
        return a == b


def _scan_newer_eval_variants() -> list[str]:
    """Warn if any ppo_evaluation*.json variant on disk is newer than the canonical file."""
    if not CANONICAL_EVAL.exists():
        return []
    canonical_mtime = CANONICAL_EVAL.stat().st_mtime
    newer = []
    for p in sorted(RESULTS_DIR.glob("ppo_evaluation_*.json")):
        if p.name == CANONICAL_EVAL.name:
            continue
        if p.stat().st_mtime > canonical_mtime:
            newer.append(p.name)
    return newer


def reconcile(base_url: str) -> dict[str, Any]:
    backend = _load(CANONICAL_EVAL) or {}
    marta = backend.get("marta_ppo", {})
    provenance = backend.get("_provenance", {})

    dash = _fetch_dashboard_live(base_url)
    if dash is None:
        dash = _fetch_dashboard_direct()

    dp = dash["portfolio"]
    dec = dash["equity_curve"]
    dpos = dash["positions"]
    dsys = dash["system_models"].get("rl", {})

    dashboard_win_rate = _win_rate_from_trades(dpos.get("trades", []))

    rows = []

    def add_row(metric, backend_val, dashboard_val, note="", status=None):
        if status is None:
            if backend_val is None and dashboard_val is None:
                status = "N/A"
            elif backend_val is None or dashboard_val is None:
                status = "WRONG"
            elif _isclose(backend_val, dashboard_val):
                status = "OK"
            else:
                status = "WRONG"
        rows.append({
            "metric": metric,
            "backend_value": backend_val,
            "dashboard_value": dashboard_val,
            "status": status,
            "note": note,
        })

    add_row("sharpe", marta.get("sharpe"), dp.get("sharpe"))
    add_row("sortino", marta.get("sortino"), dp.get("sortino"))
    add_row("max_drawdown", marta.get("max_drawdown"), dp.get("max_drawdown"))
    add_row("calmar", marta.get("calmar"), dp.get("calmar"))
    add_row("total_return", marta.get("total_return"), dp.get("total_return"))
    add_row("n_trades (meaningful)", marta.get("n_meaningful_trades"), dp.get("n_trades"))
    add_row("mean_turnover", marta.get("mean_turnover"), dp.get("mean_turnover"))
    add_row(
        "cagr", None, None,
        "NOT IMPLEMENTED anywhere in backend or frontend — no field exists. "
        "For this specific val window (exactly 2024-01-01..2024-12-31, 1 year) "
        "total_return is numerically equivalent to CAGR, but it is never labeled or "
        "surfaced as such.",
    )
    add_row(
        "win_rate", None, dashboard_win_rate,
        "Backend does not compute win_rate at all (rl/evaluate.py:compute_metrics has "
        "no such field). The dashboard computes it client-side "
        "(analytics/page.tsx) from /api/rl/positions' trade list. Reconciliation here "
        "verifies that trade list traces back to the same trade_log as every other "
        "metric on this table (see equity_curve/positions 'source' below), not that "
        "backend and dashboard agree on a shared number — there is no backend number.",
        status="DERIVED" if dashboard_win_rate is not None else "N/A",
    )
    add_row(
        "exposure", None, None,
        "No backend-eval exposure metric exists. The only 'exposure' in the system is "
        "paper_db.py's LIVE paper-trading exposure (/api/alpaca/positions, total_exposure) "
        "— a different, real-time paper-trading concept, not a backtest statistic. "
        "avg_position_size is reported per-regime in the eval JSON as the closest analogue.",
        status="N/A",
    )
    add_row(
        "equity_curve_points", len(backend.get("trade_log", [])), len(dec.get("timestamps", [])),
    )
    ec_source_ok = bool(dec.get("source")) and "trade_log" in str(dec.get("source")) and dec.get("stale") is False
    add_row(
        "equity_curve_source", "ppo_evaluation.json trade_log (canonical)", dec.get("source"),
        f"stale={dec.get('stale')}",
        status="OK" if ec_source_ok else "STALE",
    )
    pos_source_ok = bool(dpos.get("source")) and "trade_log" in str(dpos.get("source")) and dpos.get("stale") is False
    add_row(
        "positions_source", "ppo_evaluation.json trade_log (canonical)", dpos.get("source"),
        f"stale={dpos.get('stale')}",
        status="OK" if pos_source_ok else "STALE",
    )

    # Provenance / identity checks — these are what actually caught the real bugs.
    identity = {
        "run_id": provenance.get("mlflow_run_id"),
        "checkpoint_path": (provenance.get("checkpoint") or {}).get("path"),
        "checkpoint_mtime_in_eval_json": (provenance.get("checkpoint") or {}).get("mtime_utc"),
        "checkpoint_mtime_on_disk_now": _mtime_iso(CANONICAL_CHECKPOINT),
        "checkpoint_matches_disk": (
            (provenance.get("checkpoint") or {}).get("mtime_utc") == _mtime_iso(CANONICAL_CHECKPOINT)
        ),
        "split": backend.get("val_window"),
        "env_config_hash": provenance.get("env_config_hash"),
        "env_config": provenance.get("env_config"),
        "git_commit": (provenance.get("git") or {}).get("commit"),
        "git_dirty": (provenance.get("git") or {}).get("dirty"),
        "data_sources": provenance.get("data_sources"),
        "dashboard_fetch_mode": dash["mode"],
        "dashboard_system_eval_stale_vs_checkpoint": dsys.get("eval_stale_vs_checkpoint"),
    }

    newer_variants = _scan_newer_eval_variants()

    overall_status = "OK"
    if any(r["status"] == "WRONG" for r in rows):
        overall_status = "WRONG"
    elif (
        identity["checkpoint_matches_disk"] is False
        or newer_variants
        or any(r["status"] == "STALE" for r in rows)
    ):
        overall_status = "STALE"

    return {
        "overall_status": overall_status,
        "rows": rows,
        "identity": identity,
        "newer_unpromoted_eval_variants": newer_variants,
    }


def print_report(report: dict) -> None:
    rows = report["rows"]
    w_metric = max(len("metric"), max(len(r["metric"]) for r in rows))
    w_backend = max(len("backend"), max(len(str(r["backend_value"])[:24]) for r in rows))
    w_dash = max(len("dashboard"), max(len(str(r["dashboard_value"])[:24]) for r in rows))

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    header = f"{'metric':<{w_metric}}  {'backend':<{w_backend}}  {'dashboard':<{w_dash}}  status"
    print(header)
    print("-" * len(header))
    for r in rows:
        b = fmt(r["backend_value"])[:24]
        d = fmt(r["dashboard_value"])[:24]
        print(f"{r['metric']:<{w_metric}}  {b:<{w_backend}}  {d:<{w_dash}}  {r['status']}")
        if r["note"]:
            print(f"{'':<{w_metric}}    note: {r['note']}")

    print()
    print("Identity / provenance:")
    for k, v in report["identity"].items():
        print(f"  {k}: {v}")

    if report["newer_unpromoted_eval_variants"]:
        print()
        print("WARNING — newer eval variant files exist but are NOT the canonical file:")
        for name in report["newer_unpromoted_eval_variants"]:
            print(f"  data/results/{name}")

    print()
    print(f"OVERALL: {report['overall_status']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a table")
    args = parser.parse_args()

    report = reconcile(args.base_url)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    sys.exit(0 if report["overall_status"] == "OK" else 1)


if __name__ == "__main__":
    main()
