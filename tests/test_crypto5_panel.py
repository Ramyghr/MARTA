"""Tests for the crypto-only (crypto5) panel wiring added 2026-08-23.

Context: agents/macro_agent.py, agents/risk_agent.py, agents/contrarian_agent.py, and
agents/sentiment_agent.py all hardcoded paths/asset-lists pointing at the old 4-asset
production panel (BTC_USDT, ETH_USDT, EURUSD, GOLD) — most seriously,
ContrarianAgent always read data/processed/regime_stream.parquet directly regardless
of what stream the rest of a debate used, silently returning 0 historical analogues
for any asset not in that stream (BNB_USDT/LTC_USDT/XRP_USDT). Fixed by adding
optional constructor overrides (assets=, data_dir=, regime_stream_df=, macro_assets=,
fear_greed_assets=, semantics_path=) so crypto-only callers (run_debate.py) can inject
the crypto5_multihorizon panel explicitly.

Every module-level DEFAULT was deliberately kept pointed at the old production panel,
NOT switched to crypto5 — dashboard/backend/debate_stream.py constructs these same
agent classes with zero arguments and hardcodes the old production regime_stream.parquet
elsewhere, so it depends on these defaults staying exactly as they were. The first test
class below is a regression guard for that: if any of these defaults ever drift, the
live dashboard's debate feature silently breaks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

CRYPTO5_ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
PRODUCTION_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]


@pytest.fixture()
def no_ollama(monkeypatch):
    """Prevent BaseAgent.__init__ from trying to connect to a real Ollama instance —
    same pattern as tests/test_agent_parse.py's `agent` fixture."""
    import agents.base as base_mod
    monkeypatch.setattr(base_mod, "ChatOllama", MagicMock)


# ---------------------------------------------------------------------------
# Regression guard: every module-level default must stay at the production panel
# ---------------------------------------------------------------------------

class TestProductionDefaultsUnchanged:
    """dashboard/backend/debate_stream.py constructs MacroAgent()/RiskAgent()/
    ContrarianAgent()/SentimentAgent()/FundManagerAgent() with no arguments and
    hardcodes the old production regime_stream.parquet + all_assets_aligned.parquet
    itself — it was NOT migrated to the crypto5 panel. If any of these defaults
    change, live dashboard debates silently start reading mismatched or missing data.
    """

    def test_macro_agent_default_assets_and_dir(self):
        from agents.macro_agent import _ASSETS, _DATA_DIR
        assert _ASSETS == PRODUCTION_ASSETS
        assert _DATA_DIR == Path("data/processed")

    def test_risk_agent_default_dir(self):
        from agents.risk_agent import _DATA_DIR
        assert _DATA_DIR == Path("data/processed")

    def test_contrarian_agent_default_stream_path(self):
        from agents.contrarian_agent import _REGIME_STREAM_PATH
        assert _REGIME_STREAM_PATH.name == "regime_stream.parquet"
        assert "crypto5" not in str(_REGIME_STREAM_PATH)

    def test_sentiment_agent_default_asset_sets(self):
        from agents.sentiment_agent import _MACRO_ASSETS, _FEAR_GREED_ASSETS
        assert _MACRO_ASSETS == {"macro", "EURUSD"}
        assert _FEAR_GREED_ASSETS == {"BTC_USDT", "ETH_USDT"}

    def test_context_default_semantics_path(self):
        from agents.context import _DEFAULT_SEMANTICS_PATH
        assert _DEFAULT_SEMANTICS_PATH.name == "regime_semantics.json"
        assert "crypto5" not in str(_DEFAULT_SEMANTICS_PATH)

    def test_fund_manager_default_semantics_path_is_none(self):
        import inspect
        from agents.fund_manager import FundManagerAgent
        default = inspect.signature(FundManagerAgent.__init__).parameters["semantics_path"].default
        assert default is None  # resolves to agents.context._DEFAULT_SEMANTICS_PATH

    def test_utils_freshness_default_panel(self):
        # MARTA_ROOT-anchored (2026-09, same fix as rl/train.py/rl/callbacks.py) —
        # absolute now, not a bare relative path, so compare the meaningful suffix
        # rather than exact equality against a relative Path.
        from utils.freshness import _PROCESSED, DEBATE_ASSETS
        assert _PROCESSED.parts[-2:] == ("data", "processed")
        assert DEBATE_ASSETS == PRODUCTION_ASSETS

    def test_utils_freshness_crypto5_constants_exist_and_differ(self):
        from utils.freshness import CRYPTO5_PROCESSED, CRYPTO5_DEBATE_ASSETS, CRYPTO5_SEMANTICS_PATH
        assert CRYPTO5_PROCESSED.parts[-3:] == ("data", "processed", "crypto5_multihorizon")
        assert CRYPTO5_DEBATE_ASSETS == CRYPTO5_ASSETS
        assert CRYPTO5_SEMANTICS_PATH.name == "regime_semantics_crypto5_multihorizon.json"


# ---------------------------------------------------------------------------
# Synthetic crypto5-shaped data fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def crypto5_aligned_dir(tmp_path: Path) -> Path:
    """Tiny synthetic aligned parquets for all 5 crypto5 assets, keyed by their own
    logical name (no GOLD->XAUUSD-style aliasing needed for crypto symbols)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=300, freq="1h", tz="UTC")
    for asset in CRYPTO5_ASSETS:
        df = pd.DataFrame(
            {
                "close_return": rng.normal(0, 0.01, len(idx)),
                "vol_ratio": rng.uniform(0.5, 2.0, len(idx)),
            },
            index=idx,
        )
        df.to_parquet(tmp_path / f"{asset}_aligned.parquet", engine="pyarrow")
    return tmp_path


@pytest.fixture()
def crypto5_regime_stream() -> pd.DataFrame:
    """Synthetic regime stream covering all 5 crypto5 assets, including the 3
    (BNB/LTC/XRP) that never appeared in the old production stream at all."""
    idx = pd.date_range("2023-01-01", periods=300, freq="1h", tz="UTC")
    rows = []
    for asset in CRYPTO5_ASSETS:
        for i, ts in enumerate(idx):
            rows.append({
                "timestamp": ts,
                "asset": asset,
                "regime_label": i % 6,
                "regime_prob_0": 0.9 if i % 6 == 0 else 0.02,
                "regime_prob_1": 0.9 if i % 6 == 1 else 0.02,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Override behavior: crypto5 injection actually works
# ---------------------------------------------------------------------------

def test_macro_agent_override_covers_all_5_crypto5_assets(no_ollama, crypto5_aligned_dir):
    from agents.macro_agent import MacroAgent

    agent = MacroAgent(assets=CRYPTO5_ASSETS, data_dir=crypto5_aligned_dir)
    corr = agent._compute_correlation_data(pd.Timestamp("2023-01-05", tz="UTC"))

    assert set(corr["vol_ratios"].keys()) == set(CRYPTO5_ASSETS)
    assert len(corr["correlation_matrix"]) == 10  # C(5, 2) pairs


def test_macro_agent_default_construction_still_uses_production_assets(no_ollama):
    from agents.macro_agent import MacroAgent, _ASSETS, _DATA_DIR

    agent = MacroAgent()
    assert agent._assets == _ASSETS
    assert agent._data_dir == _DATA_DIR


def test_risk_agent_override_computes_real_limits_for_new_crypto5_asset(no_ollama, crypto5_aligned_dir):
    """BNB_USDT never existed in the production panel at all — this is the asset
    class that would previously have silently hit _default_risk_limits()."""
    from agents.risk_agent import RiskAgent, _default_risk_limits

    agent = RiskAgent(data_dir=crypto5_aligned_dir)
    limits = agent.compute_risk_limits(pd.Timestamp("2023-01-10", tz="UTC"), "BNB_USDT")

    defaults = _default_risk_limits()
    # A real GARCH/rolling-std fit on synthetic returns will not exactly equal the
    # hardcoded fallback constants — confirms real data was read, not the empty-df path.
    assert limits.volatility_forecast != defaults.volatility_forecast


def test_contrarian_agent_injected_stream_finds_analogues_for_new_asset(no_ollama, crypto5_regime_stream):
    """The core bug: ContrarianAgent used to always read the OLD production
    regime_stream.parquet, which has no BNB_USDT rows at all -> always 0 analogues
    for this asset, silently. Injecting the real stream must fix that."""
    from agents.contrarian_agent import ContrarianAgent
    from agents.context import RegimeContext

    agent = ContrarianAgent(regime_stream_df=crypto5_regime_stream)

    idx = pd.date_range("2023-01-01", periods=300, freq="1h", tz="UTC")
    aligned_df = pd.DataFrame(
        {"close_return": np.random.default_rng(1).normal(0, 0.01, len(idx))}, index=idx
    )
    ctx = RegimeContext(
        timestamp=pd.Timestamp("2023-01-20", tz="UTC"),
        asset="BNB_USDT",
        regime_label=0,
        regime_probs={"regime_0": 0.9},
        dominant_regime=0,
        regime_description="test regime",
        price_features={},
        window_features={},
    )
    analogues = agent._find_historical_analogues(ctx, "bullish", aligned_df)
    assert analogues["total_cases"] > 0


def test_contrarian_agent_default_construction_reads_production_path(no_ollama, monkeypatch, tmp_path):
    """No regime_stream_df/path given -> falls back to the module-level default
    (production path), not crypto5 — this is exactly what debate_stream.py relies on."""
    import agents.contrarian_agent as ca_mod

    fake_stream = pd.DataFrame({
        "timestamp": [pd.Timestamp("2023-01-01", tz="UTC")],
        "asset": ["BTC_USDT"],
        "regime_label": [0],
        "regime_prob_0": [0.9],
    })
    fake_path = tmp_path / "regime_stream.parquet"
    fake_stream.to_parquet(fake_path, engine="pyarrow")
    monkeypatch.setattr(ca_mod, "_REGIME_STREAM_PATH", fake_path)

    agent = ca_mod.ContrarianAgent()
    assert agent._regime_stream.equals(pd.read_parquet(fake_path, engine="pyarrow"))


def test_sentiment_agent_override_expands_fear_greed_to_crypto5(no_ollama, monkeypatch):
    from agents import sentiment_agent as sa_mod

    monkeypatch.setattr(sa_mod, "_TRANSFORMERS_AVAILABLE", False)  # skip real FinBERT load
    monkeypatch.setattr(sa_mod, "_NEWS_CACHE_PATH", Path("/nonexistent/news_cache.json"))

    agent = sa_mod.SentimentAgent(
        macro_assets={"macro"}, fear_greed_assets=set(CRYPTO5_ASSETS)
    )
    assert agent._fear_greed_assets == set(CRYPTO5_ASSETS)
    assert "BNB_USDT" in agent._fear_greed_assets  # was never covered before this fix


def test_sentiment_agent_default_construction_uses_original_sets(no_ollama, monkeypatch):
    from agents import sentiment_agent as sa_mod

    monkeypatch.setattr(sa_mod, "_TRANSFORMERS_AVAILABLE", False)
    monkeypatch.setattr(sa_mod, "_NEWS_CACHE_PATH", Path("/nonexistent/news_cache.json"))

    agent = sa_mod.SentimentAgent()
    assert agent._macro_assets == {"macro", "EURUSD"}
    assert agent._fear_greed_assets == {"BTC_USDT", "ETH_USDT"}
