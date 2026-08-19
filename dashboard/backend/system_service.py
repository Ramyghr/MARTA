"""System status for the Settings page — GPU, Ollama, Redis, and model checkpoint info.

Every check degrades gracefully: a missing binary, unset env var, or unreachable
service is reported as a status, never raised as an exception to the caller.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from pathlib import Path

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))


def _mtime_iso(path: Path) -> str:
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()


def get_gpu_status() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {"available": False, "reason": "nvidia-smi returned no data"}
        name, used, total, util = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "available": True,
            "name": name,
            "memory_used_mb": int(float(used)),
            "memory_total_mb": int(float(total)),
            "utilization_pct": int(float(util)),
        }
    except FileNotFoundError:
        return {"available": False, "reason": "nvidia-smi not found — no GPU access from this container"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


async def get_ollama_status() -> dict:
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{host}/api/tags")
            r.raise_for_status()
            models = [m.get("name") for m in r.json().get("models", [])]
            return {"connected": True, "host": host, "models": models}
    except Exception as e:
        return {"connected": False, "host": host, "error": str(e)}


def get_redis_status() -> dict:
    try:
        import redis
    except ImportError:
        return {"configured": False, "connected": False, "reason": "redis package not installed"}
    url = os.environ.get("REDIS_URL")
    if not url:
        return {"configured": False, "connected": False, "reason": "REDIS_URL not set"}
    try:
        client = redis.from_url(url, socket_connect_timeout=2)
        client.ping()
        return {"configured": True, "connected": True}
    except Exception as e:
        return {"configured": True, "connected": False, "error": str(e)}


# ── Model checkpoint info (mix of MLflow file-store params/metrics + checkpoint mtimes) ──

def _latest_run_dir(experiment_name: str) -> Path | None:
    mlruns = MARTA_ROOT / "mlruns"
    if not mlruns.exists():
        return None
    for meta_path in mlruns.glob("*/meta.yaml"):
        try:
            import yaml
            meta = yaml.safe_load(meta_path.read_text())
        except Exception:
            continue
        if meta and meta.get("name") == experiment_name:
            exp_root = meta_path.parent
            run_dirs = [d for d in exp_root.iterdir() if d.is_dir() and (d / "meta.yaml").exists()]
            if not run_dirs:
                return None
            run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return run_dirs[0]
    return None


def _read_param(run_dir: Path, name: str) -> str | None:
    p = run_dir / "params" / name
    return p.read_text().strip() if p.exists() else None


def _read_last_metric(run_dir: Path, name: str) -> float | None:
    p = run_dir / "metrics" / name
    if not p.exists():
        return None
    lines = p.read_text().strip().splitlines()
    if not lines:
        return None
    parts = lines[-1].split()
    return float(parts[1]) if len(parts) >= 2 else None


def get_encoder_info() -> dict:
    ckpt = MARTA_ROOT / "encoder" / "checkpoints" / "best_encoder.pt"
    info: dict = {
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_mtime": _mtime_iso(ckpt) if ckpt.exists() else None,
        "epoch": None,
        "val_loss": None,
        "val_accuracy": None,
    }
    run_dir = _latest_run_dir("marta_encoder_pretraining")
    if run_dir:
        epoch = _read_param(run_dir, "best_epoch")
        val_loss = _read_param(run_dir, "best_val_loss")
        info["epoch"] = int(epoch) if epoch is not None else None
        info["val_loss"] = float(val_loss) if val_loss is not None else None
        val_accuracy = _read_last_metric(run_dir, "val_top1_acc")
        # A top-1 accuracy below 50% on a 7-class regime task means MLflow handed back
        # the wrong run/metric (the real encoder is ~75.3%) — surface that as "unknown"
        # rather than display a number that reads as a broken model.
        info["val_accuracy"] = val_accuracy if val_accuracy is not None and val_accuracy >= 0.5 else None
    return info


def get_hmm_info() -> dict:
    path = MARTA_ROOT / "data" / "validated" / "regime_semantics.json"
    if not path.exists():
        return {"n_states": None, "semantic_validation_passed": None}
    d = json.loads(path.read_text())
    return {"n_states": d.get("n_states"), "semantic_validation_passed": d.get("semantic_validation_passed")}


def get_rl_info() -> dict:
    ckpt = MARTA_ROOT / "rl" / "checkpoints" / "ppo_best.zip"
    eval_path = MARTA_ROOT / "data" / "results" / "ppo_evaluation.json"
    sharpe = None
    provenance = None
    eval_stale_vs_checkpoint = None
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text())
        sharpe = eval_data.get("marta_ppo", {}).get("sharpe")
        provenance = eval_data.get("_provenance")
        # Flag when the on-disk checkpoint is newer than the checkpoint this eval JSON
        # actually ran against — a fresher ppo_best.zip landed after the last canonical
        # eval, so the Sharpe shown here no longer describes the checkpoint on disk.
        if provenance and provenance.get("checkpoint") and ckpt.exists():
            eval_ckpt_mtime = provenance["checkpoint"].get("mtime_utc")
            current_ckpt_mtime = _mtime_iso(ckpt)
            eval_stale_vs_checkpoint = eval_ckpt_mtime != current_ckpt_mtime
    return {
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_mtime": _mtime_iso(ckpt) if ckpt.exists() else None,
        "best_sharpe": sharpe,
        "eval_provenance": provenance,
        "eval_stale_vs_checkpoint": eval_stale_vs_checkpoint,
    }
