import json
import os
from pathlib import Path

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))


def get_portfolio() -> dict:
    path = MARTA_ROOT / "data" / "results" / "ppo_evaluation.json"
    if not path.exists():
        return _empty_portfolio()
    with open(path) as f:
        data = json.load(f)

    marta = data.get("marta_ppo", {})
    baselines = data.get("baselines", {})
    per_regime = data.get("per_regime", {})
    bh = baselines.get("buy_and_hold_btc", {})

    regime_breakdown = []
    for k, v in per_regime.items():
        try:
            label = int(k.split("_")[-1])
        except ValueError:
            label = 0
        regime_breakdown.append({
            "regime_label": label,
            "n_steps": int(v.get("n_steps", 0)),
            "sharpe": float(v.get("sharpe", 0)),
            "avg_position_size": float(v.get("avg_position_size", 0)),
            "avg_return": float(v.get("avg_return", 0)),
        })

    return {
        "sharpe": float(marta.get("sharpe", 0)),
        "sortino": float(marta.get("sortino", 0)),
        "max_drawdown": float(marta.get("max_drawdown", 0)),
        "calmar": float(marta.get("calmar", 0)),
        "total_return": float(marta.get("total_return", 0)),
        "n_trades": int(marta.get("n_meaningful_trades", 0)),
        "mean_turnover": float(marta.get("mean_turnover", 0)),
        "n_val_steps": int(data.get("n_val_steps", 0)),
        "final_portfolio_value": float(data.get("final_portfolio_value", 10000)),
        "buy_and_hold": {
            "sharpe": float(bh.get("sharpe", 0)),
            "max_drawdown": float(bh.get("max_drawdown", 0)),
            "total_return": float(bh.get("total_return", 0)),
        },
        "per_regime": regime_breakdown,
    }


def get_ablation() -> dict:
    path = MARTA_ROOT / "data" / "results" / "ablation_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _empty_portfolio() -> dict:
    return {
        "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
        "total_return": 0.0, "n_trades": 0, "mean_turnover": 0.0,
        "n_val_steps": 0, "final_portfolio_value": 10000.0,
        "buy_and_hold": {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0},
        "per_regime": [],
    }
