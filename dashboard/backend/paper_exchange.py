"""Binance testnet paper trading execution via CCXT.

Connects to https://testnet.binance.vision — real Binance testnet with simulated
funds ($10,000 USDT by default).  Falls back gracefully when env vars are absent
(keeps the local paper_db simulation running so the dashboard is never broken).

Setup:
    1. Go to https://testnet.binance.vision and log in with GitHub.
    2. Generate an API key pair.
    3. Export in your shell (or add to dashboard/backend/.env):
           BINANCE_TESTNET_API_KEY=<key>
           BINANCE_TESTNET_SECRET=<secret>
    4. Restart the backend — ppo_inference.load_model() and the scheduler both
       call paper_exchange.init() automatically.

Position sizing:
    The PPO Beta head outputs a fraction of portfolio value (e.g. 0.04 = 4%).
    We translate that to a USDT notional and place a market order.
    Direction: long → BUY, short → SELL-to-open (requires margin; testnet
    supports futures — set BINANCE_TESTNET_FUTURES=true to use
    testnet.binancefuture.com instead of spot testnet).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_API_KEY    = os.environ.get("BINANCE_TESTNET_API_KEY", "")
_SECRET     = os.environ.get("BINANCE_TESTNET_SECRET", "")
_USE_FUTURES = os.environ.get("BINANCE_TESTNET_FUTURES", "false").lower() == "true"

SYMBOL_MAP = {
    "BTC_USDT": "BTC/USDT",
    "ETH_USDT": "ETH/USDT",
}

MIN_NOTIONAL   = 10.0    # Binance minimum order: $10 USDT
MAX_POSITION_PCT = 0.20  # never risk more than 20% on one trade
SLIPPAGE_BPS   = 2.5     # estimated slippage for logging

_exchange = None   # CCXT exchange instance, created in init()


# ── Init ──────────────────────────────────────────────────────────────────────

def init() -> bool:
    """Initialise the CCXT Binance testnet connection.

    Returns True if the connection is live, False if credentials are missing
    (falls back silently to local paper_db simulation).
    """
    global _exchange
    if not _API_KEY or not _SECRET:
        logger.info(
            "paper_exchange: BINANCE_TESTNET_API_KEY / SECRET not set — "
            "using local paper_db simulation only"
        )
        return False

    try:
        import ccxt  # type: ignore
        if _USE_FUTURES:
            ex = ccxt.binanceusdm({
                "apiKey": _API_KEY,
                "secret": _SECRET,
                "enableRateLimit": True,
            })
            ex.set_sandbox_mode(True)
        else:
            ex = ccxt.binance({
                "apiKey": _API_KEY,
                "secret": _SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            # Use ccxt's built-in sandbox mode rather than a manual string-replace
            # across every URL namespace. Binance spot testnet only implements the
            # public/private (api/v3) endpoints, not sapi (wallet/account management)
            # — a blanket replace() previously pointed sapi at testnet.binance.vision
            # too, and ccxt's fetch_currencies() (auto-triggered by load_markets()
            # inside fetch_balance() since binance defaults options.fetchCurrencies
            # to True whenever API keys are set) called sapiGetCapitalConfigGetall,
            # 404ing and silently dropping the connection to local paper_db fallback.
            # set_sandbox_mode(True) only remaps the endpoints binance's own 'test'
            # URL block defines (public/private), and sets urls['apiBackup'], which
            # fetch_currencies() checks to skip the unsupported sapi call entirely.
            ex.set_sandbox_mode(True)

        # Verify credentials by fetching balance
        bal = ex.fetch_balance()
        usdt = bal.get("USDT", {}).get("free", 0.0)
        logger.info(f"paper_exchange: Binance testnet connected. USDT balance: {usdt:.2f}")
        _exchange = ex
        return True

    except Exception as exc:
        logger.warning(
            f"paper_exchange: could not connect to Binance testnet ({exc}) — "
            "falling back to local paper_db simulation"
        )
        _exchange = None
        return False


def is_live() -> bool:
    """True when a real testnet exchange connection is active."""
    return _exchange is not None


# ── Balance / portfolio ───────────────────────────────────────────────────────

def get_balance() -> dict:
    """Return free USDT balance from testnet, or {} on failure."""
    if _exchange is None:
        return {}
    try:
        bal = _exchange.fetch_balance()
        return {
            "USDT": bal.get("USDT", {}).get("free", 0.0),
            "BTC":  bal.get("BTC",  {}).get("free", 0.0),
            "ETH":  bal.get("ETH",  {}).get("free", 0.0),
        }
    except Exception as exc:
        logger.warning(f"paper_exchange.get_balance: {exc}")
        return {}


def get_open_position(asset: str) -> dict:
    """Return the current open position for `asset`, or an empty dict."""
    if _exchange is None:
        return {}
    symbol = SYMBOL_MAP.get(asset)
    if not symbol:
        return {}
    try:
        if _USE_FUTURES:
            positions = _exchange.fetch_positions([symbol])
            for p in positions:
                if p.get("symbol") == symbol:
                    return {
                        "side":   p.get("side", "flat"),
                        "size":   float(p.get("contracts", 0)),
                        "entry":  float(p.get("entryPrice") or 0),
                        "pnl":    float(p.get("unrealizedPnl") or 0),
                    }
        else:
            bal = _exchange.fetch_balance()
            base = symbol.split("/")[0]
            qty = float(bal.get(base, {}).get("free", 0))
            return {"side": "long" if qty > 0 else "flat", "size": qty}
    except Exception as exc:
        logger.warning(f"paper_exchange.get_open_position({asset}): {exc}")
    return {}


# ── Order execution ───────────────────────────────────────────────────────────

def place_order(
    asset: str,
    action: str,
    position_fraction: float,
    portfolio_value: float,
    price: float,
) -> Optional[dict]:
    """Place a market order on Binance testnet.

    Args:
        asset:             e.g. "BTC_USDT"
        action:            "long" | "short" | "hold"
        position_fraction: PPO Beta output (e.g. 0.04 = 4% of portfolio)
        portfolio_value:   current paper portfolio value in USDT
        price:             latest market price (for logging and qty calc)

    Returns:
        order dict from CCXT, or None if the action is hold / order failed.
    """
    if _exchange is None or action == "hold" or price <= 0:
        return None

    symbol = SYMBOL_MAP.get(asset)
    if not symbol:
        return None

    # Compute USDT notional and coin quantity
    notional = min(portfolio_value * position_fraction, portfolio_value * MAX_POSITION_PCT)
    if notional < MIN_NOTIONAL:
        logger.debug(f"paper_exchange: notional {notional:.2f} < MIN_NOTIONAL — skipping")
        return None

    qty = notional / price
    side = "buy" if action == "long" else "sell"

    try:
        order = _exchange.create_market_order(symbol, side, qty)
        filled_price = float(order.get("average") or order.get("price") or price)
        logger.info(
            f"paper_exchange: {side.upper()} {qty:.6f} {symbol} "
            f"@ ~{filled_price:.2f} USDT (notional={notional:.2f})"
        )
        return order
    except Exception as exc:
        logger.warning(f"paper_exchange.place_order({asset}, {action}): {exc}")
        return None


def cancel_all_orders(asset: str) -> None:
    """Cancel any open orders for `asset` — called before placing a new one."""
    if _exchange is None:
        return
    symbol = SYMBOL_MAP.get(asset)
    if not symbol:
        return
    try:
        open_orders = _exchange.fetch_open_orders(symbol)
        for o in open_orders:
            _exchange.cancel_order(o["id"], symbol)
    except Exception as exc:
        logger.debug(f"paper_exchange.cancel_all_orders({asset}): {exc}")


def get_trade_history(asset: str, limit: int = 50) -> list[dict]:
    """Fetch recent fills from testnet for `asset`."""
    if _exchange is None:
        return []
    symbol = SYMBOL_MAP.get(asset)
    if not symbol:
        return []
    try:
        trades = _exchange.fetch_my_trades(symbol, limit=limit)
        return [
            {
                "timestamp": t.get("datetime", ""),
                "side":      t.get("side", ""),
                "price":     float(t.get("price") or 0),
                "amount":    float(t.get("amount") or 0),
                "cost":      float(t.get("cost") or 0),
                "fee_usdt":  float((t.get("fee") or {}).get("cost") or 0),
            }
            for t in trades
        ]
    except Exception as exc:
        logger.warning(f"paper_exchange.get_trade_history({asset}): {exc}")
        return []
