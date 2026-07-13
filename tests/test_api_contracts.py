"""API endpoint contract tests using FastAPI TestClient with mocked service layer.

We mock every service module (no data files or Ollama needed) and only verify
the HTTP contract: status codes, response shape, and key field names.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root and dashboard/backend to path
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "dashboard" / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Mock the heavy service modules before importing app
# ---------------------------------------------------------------------------

_MOCK_REGIME_SERVICE = MagicMock()
_MOCK_HISTORICAL = MagicMock()
_MOCK_PORTFOLIO_SERVICE = MagicMock()
_MOCK_DEBATE_STREAM = MagicMock()
_MOCK_LIVE_FEED = MagicMock()
_MOCK_DEMO_SCENARIOS = MagicMock()
_MOCK_RL_SERVICE = MagicMock()
_MOCK_NEWS_SERVICE = MagicMock()
_MOCK_NEWS_LIVE = MagicMock()

# Configure return values
_MOCK_REGIME_SERVICE.get_regime.return_value = {
    "asset": "BTC_USDT",
    "regime_label": 1,
    "regime_description": "Calm bull",
    "regime_probs": {"regime_1": 0.8},
}
_MOCK_HISTORICAL.get_data_range.return_value = {
    "asset": "BTC_USDT",
    "start": "2021-01-01T00:00:00Z",
    "end": "2021-10-12T00:00:00Z",
    "n_rows": 6803,
}
_MOCK_HISTORICAL.get_history.return_value = [
    {"timestamp": "2021-01-01T00:00:00Z", "open": 29000, "high": 30000, "low": 28000, "close": 29500, "volume": 500}
]
_MOCK_HISTORICAL.get_history_range.return_value = []
_MOCK_PORTFOLIO_SERVICE.get_portfolio.return_value = {"portfolio_value": 10000.0, "positions": []}
_MOCK_PORTFOLIO_SERVICE.get_ablation.return_value = {"variants": {}}
_MOCK_DEBATE_STREAM.get_debates_list.return_value = [
    {"id": 0, "timestamp": "2021-01-15T12:00:00Z", "asset": "BTC_USDT",
     "regime_label": 1, "scenario": "test", "action": "long", "conviction": 0.7}
]
_MOCK_DEMO_SCENARIOS.DEMO_SCENARIOS = []
_MOCK_DEMO_SCENARIOS.SCENARIO_BY_ID = {}
_MOCK_RL_SERVICE.get_full_results.return_value = {
    "evaluation": {"marta_ppo": {"sharpe": 1.809, "max_drawdown": 0.058}},
    "ablation_chart": [
        {"variant": "MARTA Full", "key": "marta_full", "sharpe": 1.809,
         "max_drawdown": 5.8, "total_return": 17.0, "calmar": 3.605, "n_meaningful_trades": 31}
    ],
    "bootstrap_ci": [
        {"comparison": "vs buy hold", "delta_sharpe": 0.863,
         "ci_lower_95": -0.300, "ci_upper_95": 3.937, "significant": False}
    ],
    "curriculum": [],
    "has_equity_curve": False,
}
_MOCK_RL_SERVICE.get_equity_curve.return_value = {
    "timestamps": ["2021-01-01", "2021-01-02"],
    "marta": [10000.0, 10050.0],
    "buy_hold": [10000.0, 10120.0],
    "synthetic": True,
}
_MOCK_RL_SERVICE.get_positions.return_value = {
    "current_action": "hold",
    "current_position": 0.026,
    "current_regime": 1,
    "portfolio_value": 11700.0,
    "total_return": 0.17,
    "n_trades": 31,
    "trades": [],
    "source": "synthetic",
}
_MOCK_NEWS_SERVICE.get_news.return_value = []

# Patch all modules before importing app
with (
    patch.dict(sys.modules, {
        "regime_service": _MOCK_REGIME_SERVICE,
        "historical": _MOCK_HISTORICAL,
        "portfolio_service": _MOCK_PORTFOLIO_SERVICE,
        "debate_stream": _MOCK_DEBATE_STREAM,
        "live_feed": _MOCK_LIVE_FEED,
        "demo_scenarios": _MOCK_DEMO_SCENARIOS,
        "rl_service": _MOCK_RL_SERVICE,
        "news_service": _MOCK_NEWS_SERVICE,
        "news_live": _MOCK_NEWS_LIVE,
        "live_signal": MagicMock(),
    }),
):
    from fastapi.testclient import TestClient
    import main as app_module
    client = TestClient(app_module.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /api/assets
# ---------------------------------------------------------------------------

def test_get_assets_status():
    r = client.get("/api/assets")
    assert r.status_code == 200


def test_get_assets_schema():
    r = client.get("/api/assets")
    body = r.json()
    assert "assets" in body
    assets = body["assets"]
    assert len(assets) == 5
    ids = [a["id"] for a in assets]
    assert "BTC_USDT" in ids
    assert "EURUSD" in ids


def test_get_assets_fields():
    assets = client.get("/api/assets").json()["assets"]
    for a in assets:
        assert "id" in a
        assert "label" in a
        assert "type" in a


# ---------------------------------------------------------------------------
# /api/data-range/{asset}
# ---------------------------------------------------------------------------

def test_get_data_range_200():
    r = client.get("/api/data-range/BTC_USDT")
    assert r.status_code == 200


def test_get_data_range_has_asset():
    body = client.get("/api/data-range/BTC_USDT").json()
    assert "asset" in body


# ---------------------------------------------------------------------------
# /api/history/{asset}
# ---------------------------------------------------------------------------

def test_get_history_200():
    r = client.get("/api/history/BTC_USDT")
    assert r.status_code == 200


def test_get_history_has_candles():
    body = client.get("/api/history/BTC_USDT").json()
    assert "candles" in body
    assert isinstance(body["candles"], list)


# ---------------------------------------------------------------------------
# /api/debates
# ---------------------------------------------------------------------------

def test_list_debates_200():
    r = client.get("/api/debates")
    assert r.status_code == 200


def test_list_debates_schema():
    body = client.get("/api/debates").json()
    assert "debates" in body
    debates = body["debates"]
    assert len(debates) >= 1
    d = debates[0]
    assert "id" in d
    assert "asset" in d
    assert "action" in d


# ---------------------------------------------------------------------------
# /api/rl/results  (F-19 contract: ci_lower_95 / ci_upper_95 key names)
# ---------------------------------------------------------------------------

def test_rl_results_200():
    r = client.get("/api/rl/results")
    assert r.status_code == 200


def test_rl_results_top_level_keys():
    body = client.get("/api/rl/results").json()
    assert "evaluation" in body
    assert "ablation_chart" in body
    assert "bootstrap_ci" in body
    assert "curriculum" in body


def test_rl_results_ci_key_names():
    """F-19 regression: CI keys must be ci_lower_95 / ci_upper_95, not ci_low / ci_high."""
    body = client.get("/api/rl/results").json()
    for ci in body["bootstrap_ci"]:
        assert "ci_lower_95" in ci, f"ci_lower_95 missing from: {ci}"
        assert "ci_upper_95" in ci, f"ci_upper_95 missing from: {ci}"
        assert "ci_low" not in ci, "stale key ci_low still present"
        assert "ci_high" not in ci, "stale key ci_high still present"


def test_rl_results_ablation_chart_schema():
    body = client.get("/api/rl/results").json()
    for row in body["ablation_chart"]:
        assert "variant" in row
        assert "sharpe" in row
        assert "max_drawdown" in row


# ---------------------------------------------------------------------------
# /api/rl/equity-curve
# ---------------------------------------------------------------------------

def test_equity_curve_200():
    r = client.get("/api/rl/equity-curve")
    assert r.status_code == 200


def test_equity_curve_schema():
    body = client.get("/api/rl/equity-curve").json()
    assert "timestamps" in body
    assert "marta" in body
    assert isinstance(body["timestamps"], list)
    assert isinstance(body["marta"], list)


# ---------------------------------------------------------------------------
# /api/rl/positions
# ---------------------------------------------------------------------------

def test_positions_200():
    r = client.get("/api/rl/positions")
    assert r.status_code == 200


def test_positions_schema():
    body = client.get("/api/rl/positions").json()
    assert "current_action" in body
    assert "portfolio_value" in body
    assert "trades" in body
    assert isinstance(body["trades"], list)


# ---------------------------------------------------------------------------
# /api/regime/{asset}
# ---------------------------------------------------------------------------

def test_get_regime_200():
    r = client.get("/api/regime/BTC_USDT")
    assert r.status_code == 200


def test_get_regime_has_label():
    body = client.get("/api/regime/BTC_USDT").json()
    assert "regime_label" in body
