"""MARTA Alpaca paper trading integration.

Uses the Alpaca Practice API (paper-api.alpaca.markets) for order execution when
ALPACA_API_KEY/ALPACA_SECRET_KEY are configured (Settings page or .env). Empty keys
are the expected default — every function degrades to a harmless no-op/empty result
so the rest of the dashboard (paper_exchange.py Binance testnet + paper_db SQLite
ledger, both untouched) keeps working exactly as before.
"""
import logging
import os

logger = logging.getLogger(__name__)

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Asset mapping: MARTA internal id -> Alpaca crypto symbol. Same two assets
# paper_exchange.py (Binance testnet) already covers; Alpaca takes priority over
# testnet in signal_scheduler.py when both are configured.
ALPACA_SYMBOLS = {
    "BTC_USDT": "BTC/USD",
    "ETH_USDT": "ETH/USD",
}

# Skip rebalancing when the already-held position is within this fraction of the
# target — mirrors paper_db.MIN_TRADE_SIZE so "stay long 2%" doesn't look like a new
# trade on either ledger. Without this, place_order() used to submit a fresh BUY for
# the full target notional on every tick the signal stayed "long", so the real Alpaca
# account's exposure silently grew far past what paper_db (the dashboard's logical
# source of truth) believed was held.
MIN_REBALANCE_FRACTION = 0.001


def is_configured() -> bool:
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)


def get_client():
    if not is_configured():
        return None
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=True)
    except Exception as e:
        logger.warning(f"Alpaca client init failed: {e}")
        return None


def get_account() -> dict:
    if not is_configured():
        return {"configured": False, "message": "Alpaca credentials not set"}
    client = get_client()
    if not client:
        return {"configured": True, "message": "Alpaca client unavailable"}
    try:
        account = client.get_account()
        return {
            "configured": True,
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "status": str(account.status),
        }
    except Exception as e:
        return {"configured": True, "error": str(e)}


def place_order(asset: str, action: str, position_size: float, current_price: float) -> dict:
    """Place a practice-account order on Alpaca. Caller falls back to the existing
    Binance-testnet / paper_db chain whenever `source` != "alpaca"."""
    alpaca_symbol = ALPACA_SYMBOLS.get(asset)
    if not is_configured() or not alpaca_symbol:
        return {"source": "unavailable"}

    client = get_client()
    if not client:
        return {"source": "unavailable"}

    if action == "hold" or current_price <= 0:
        return {"source": "alpaca", "action": "hold", "order_id": None}

    if action == "short":
        # Alpaca's crypto trading is spot-only — no margin/borrow — so an actual short
        # position is impossible; a SELL for more than what's currently held always
        # 404s as "insufficient balance" (verified in practice, not hypothetical).
        # paper_db's local ledger is the dashboard's real source of truth and does
        # model signed short positions correctly (see signal_scheduler.py's comment on
        # this call site) — this is only the best-effort mirror to the real practice
        # account, which structurally cannot represent a short at all. Still worth
        # flattening any existing Alpaca-side long so that account doesn't sit stale
        # relative to paper_db's short.
        try:
            client.close_position(alpaca_symbol)
        except Exception:
            pass
        logger.info(f"Alpaca order skipped ({asset}, short): crypto shorting isn't supported on Alpaca — flattened any existing long instead")
        return {"source": "unsupported", "action": "short"}

    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        account = client.get_account()
        portfolio_value = float(account.portfolio_value)

        # Reconcile against what this symbol is already holding on the real account —
        # only trade the delta to reach `position_size`, not the full target every time.
        # Alpaca's positions endpoint wants the symbol WITHOUT the slash (e.g. "BTCUSD"),
        # unlike submit_order's "BTC/USD" — confirmed empirically, a slash here 404s.
        try:
            current_position = client.get_open_position(alpaca_symbol.replace("/", ""))
            current_fraction = float(current_position.market_value) / portfolio_value if portfolio_value else 0.0
        except Exception:
            current_fraction = 0.0  # no open position for this symbol

        delta_fraction = position_size - current_fraction
        if abs(delta_fraction) < MIN_REBALANCE_FRACTION:
            return {"source": "alpaca", "action": "hold", "order_id": None, "reason": "already at target position"}

        side = OrderSide.BUY if delta_fraction > 0 else OrderSide.SELL
        notional = round(abs(portfolio_value * delta_fraction), 2)

        order = client.submit_order(MarketOrderRequest(
            symbol=alpaca_symbol,
            notional=notional,
            side=side,
            time_in_force=TimeInForce.GTC,
        ))

        return {
            "source": "alpaca",
            "order_id": str(order.id),
            "status": str(order.status),
            "symbol": alpaca_symbol,
            "notional": notional,
            "side": str(side),
            "action": action,
        }
    except Exception as e:
        logger.warning(f"Alpaca order failed ({asset}, {action}): {e}")
        return {"source": "unavailable", "error": str(e)}


def get_positions() -> list[dict]:
    client = get_client()
    if not client:
        return []
    try:
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": str(p.side),
            }
            for p in positions
        ]
    except Exception as e:
        logger.warning(f"Alpaca positions failed: {e}")
        return []


def get_orders_history(limit: int = 20) -> list[dict]:
    client = get_client()
    if not client:
        return []
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side),
                "qty": str(o.qty),
                "notional": str(o.notional),
                "status": str(o.status),
                "filled_at": str(o.filled_at),
                "filled_avg_price": str(o.filled_avg_price),
            }
            for o in orders
        ]
    except Exception as e:
        logger.warning(f"Alpaca order history failed: {e}")
        return []
