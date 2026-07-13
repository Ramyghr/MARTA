"""SQLite-backed paper trading ledger — survives server restarts.

Replaces the in-memory ``_paper`` dict in ``live_signal.py``.
The DB file is written to ``MARTA_ROOT/data/paper_trading.db`` (created on first use).

Slippage model:
    slippage = SLIPPAGE_BPS / 10_000 × |Δposition| × portfolio_value
    This represents a one-way half-spread of 2.5 bps — conservative for BTC/ETH,
    realistic for illiquid hours.

Position-sizing constraints:
    - Signed position always clipped to [−MAX_POSITION, +MAX_POSITION] (default 0.20).
    - Minimum trade size: MIN_TRADE_SIZE (default 0.001) — positions smaller than this
      are snapped to 0 to avoid rounding noise from the Beta head.

Capital structure — ONE shared pool across assets:
    Every live asset (BTC_USDT, ETH_USDT, ...) draws from a single ``shared_capital``
    row rather than each getting its own independent $10,000 balance. Two assets each
    silently getting their own $10k pool would mean $20k deployed simultaneously,
    which doesn't match the trained RL model — it manages one portfolio value and one
    signed position at a time (see rl/env.py). Per-asset rows in the `portfolio` table
    now hold only *position state* (prev_action/prev_price/prev_size/n_trades) for that
    asset; the account's actual value/return always comes from `shared_capital`.
    `update()` additionally clips a new position so this asset's exposure plus every
    other asset's already-open exposure never exceeds 100% of the shared pool.

Approval mode (see app_settings.get_approval_mode):
    When ON, live_signal.get_live_signal() records a non-hold signal as a row in
    `pending_trades` instead of calling `update()` — no capital moves and no position
    opens until a human calls approve_pending_trade()/reject_pending_trade() (wired to
    POST /api/trades/pending/{id}/approve|reject in main.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))
_DB_PATH = Path(
    os.environ.get("PAPER_DB_PATH", str(MARTA_ROOT / "data" / "paper_trading.db"))
)

INITIAL_CAPITAL   = 10_000.0
MAX_POSITION      = 0.20      # max |signed_position| — protects against runaway Beta
MIN_TRADE_SIZE    = 0.001     # positions smaller than this snap to zero
SLIPPAGE_BPS      = 2.5       # one-way half-spread in basis points
TRANSACTION_COST  = 0.001     # exchange fee fraction (same as rl/env.py)

_DDL = """
CREATE TABLE IF NOT EXISTS portfolio (
    asset            TEXT PRIMARY KEY,
    portfolio_value  REAL NOT NULL DEFAULT 10000.0,
    total_return     REAL NOT NULL DEFAULT 0.0,
    n_trades         INTEGER NOT NULL DEFAULT 0,
    prev_action      TEXT NOT NULL DEFAULT 'hold',
    prev_price       REAL NOT NULL DEFAULT 0.0,
    prev_size        REAL NOT NULL DEFAULT 0.0,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asset            TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    action           TEXT NOT NULL,
    position_size    REAL NOT NULL,
    price            REAL NOT NULL,
    slippage         REAL NOT NULL DEFAULT 0.0,
    pnl              REAL NOT NULL DEFAULT 0.0,
    portfolio_value  REAL NOT NULL,
    regime_label     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trade_reasoning (
    trade_id             INTEGER PRIMARY KEY,
    asset                TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    regime_label         INTEGER NOT NULL,
    regime_description   TEXT,
    debate_id            TEXT,
    agent_theses         TEXT,        -- JSON: list of {agent, bias, confidence, reasoning}
    fund_manager_rationale TEXT,
    ppo_override         INTEGER NOT NULL DEFAULT 0,
    source               TEXT NOT NULL,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS trade_outcomes (
    trade_id         INTEGER PRIMARY KEY,
    asset            TEXT NOT NULL,
    entry_price      REAL NOT NULL,
    entry_timestamp  TEXT NOT NULL,
    exit_price       REAL,
    exit_timestamp   TEXT,
    realized_pnl     REAL,
    holding_hours    REAL,
    regime_at_exit   INTEGER,
    outcome          TEXT NOT NULL DEFAULT 'open',   -- 'profit' | 'loss' | 'open'
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS shared_capital (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    capital          REAL NOT NULL DEFAULT 10000.0,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asset            TEXT NOT NULL,
    action           TEXT NOT NULL,
    position_size    REAL NOT NULL,
    price            REAL NOT NULL,
    regime_label     INTEGER NOT NULL DEFAULT 0,
    source           TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'pending',   -- 'pending' | 'approved' | 'rejected'
    created_at       TEXT NOT NULL,
    resolved_at      TEXT,
    trade_id         INTEGER,
    reasoning        TEXT   -- JSON: same shape saved to trade_reasoning once approved
);
"""


@contextmanager
def _conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_DDL)
        yield con
        con.commit()
    finally:
        con.close()


def _apply_constraints(size: float) -> float:
    """Clip position to [−MAX, +MAX]; snap micro-sizes to 0."""
    if abs(size) < MIN_TRADE_SIZE:
        return 0.0
    return float(max(-MAX_POSITION, min(MAX_POSITION, size)))


def _get_shared_capital(con: sqlite3.Connection) -> float:
    row = con.execute("SELECT capital FROM shared_capital WHERE id=1").fetchone()
    if row:
        return float(row["capital"])
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        "INSERT INTO shared_capital (id, capital, updated_at) VALUES (1, ?, ?)",
        (INITIAL_CAPITAL, now),
    )
    return INITIAL_CAPITAL


def _set_shared_capital(con: sqlite3.Connection, value: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        """INSERT INTO shared_capital (id, capital, updated_at) VALUES (1, ?, ?)
           ON CONFLICT(id) DO UPDATE SET capital=excluded.capital, updated_at=excluded.updated_at""",
        (round(value, 4), now),
    )


def _other_assets_exposure(con: sqlite3.Connection, asset: str) -> float:
    """Sum of |signed position| currently open on every OTHER asset, as a fraction of
    the shared pool — used to cap this asset's new position so combined exposure
    across all assets never exceeds 100% of the one shared capital pool."""
    rows = con.execute(
        "SELECT prev_action, prev_size FROM portfolio WHERE asset != ?", (asset,)
    ).fetchall()
    return sum(abs(float(r["prev_size"])) for r in rows if r["prev_action"] in ("long", "short"))


def get_portfolio(asset: str) -> dict:
    """Position state for `asset` + the account-wide shared capital value/return.

    portfolio_value/total_return always come from the single shared pool (never from
    a per-asset column) so they can't go stale relative to what other assets did.
    """
    with _conn() as con:
        capital = _get_shared_capital(con)
        row = con.execute("SELECT * FROM portfolio WHERE asset=?", (asset,)).fetchone()
        p = dict(row) if row else {
            "n_trades": 0, "prev_action": "hold", "prev_price": 0.0, "prev_size": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "asset":           asset,
        "portfolio_value": round(capital, 2),
        "total_return":    round(capital / INITIAL_CAPITAL - 1.0, 6),
        "n_trades":        int(p["n_trades"]),
        "prev_action":     p["prev_action"],
        "prev_price":      float(p["prev_price"]),
        "prev_size":       float(p["prev_size"]),
        "updated_at":      p["updated_at"],
    }


def get_account_state() -> dict:
    """Account-wide view of the one shared capital pool — the live counterpart to the
    trained model's single-portfolio assumption. Each entry's `exposure` is that
    asset's currently open position as a fraction of the ONE pool, not an independent
    per-asset balance."""
    with _conn() as con:
        capital = _get_shared_capital(con)
        rows = con.execute(
            "SELECT asset, prev_action, prev_size, n_trades, updated_at FROM portfolio"
        ).fetchall()
    positions = [
        {
            "asset":      r["asset"],
            "action":     r["prev_action"],
            "exposure":   round(float(r["prev_size"]), 4) if r["prev_action"] in ("long", "short") else 0.0,
            "n_trades":   int(r["n_trades"]),
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
    return {
        "shared_capital":  round(capital, 2),
        "initial_capital": INITIAL_CAPITAL,
        "total_return":    round(capital / INITIAL_CAPITAL - 1.0, 6),
        "total_exposure":  round(sum(p["exposure"] for p in positions), 4),
        "positions":       positions,
    }


def get_trades(asset: str, limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM trades WHERE asset=? ORDER BY id DESC LIMIT ?",
            (asset, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def update(
    asset: str,
    action: str,
    size: float,
    price: float,
    regime_label: int = 0,
) -> dict:
    """Update the paper portfolio for one signal tick.

    Returns the updated portfolio state dict.
    """
    size = _apply_constraints(size)

    with _conn() as con:
        row = con.execute("SELECT * FROM portfolio WHERE asset=?", (asset,)).fetchone()
        if row:
            p = dict(row)
        else:
            p = {
                "asset":       asset,
                "n_trades":    0,
                "prev_action": "hold",
                "prev_price":  0.0,
                "prev_size":   0.0,
            }

        prev_size  = float(p["prev_size"])
        prev_price = float(p["prev_price"])
        capital    = _get_shared_capital(con)

        # Shared pool across assets: this asset's new exposure is capped so that,
        # combined with every other asset's already-open exposure, total deployed
        # capital never exceeds 100% of the one shared pool (no silent 2x deployment
        # from BTC + ETH each drawing an independent $10k — see module docstring).
        if action in ("long", "short"):
            room = max(0.0, 1.0 - _other_assets_exposure(con, asset))
            size = min(size, room)

        # P&L from the previous position (held between ticks)
        step_pnl = 0.0
        if p["prev_action"] == "long" and prev_price > 0 and price > 0:
            step_pnl = (price - prev_price) / prev_price * prev_size
        elif p["prev_action"] == "short" and prev_price > 0 and price > 0:
            step_pnl = (prev_price - price) / prev_price * prev_size

        # Transaction costs are driven by the change in SIGNED exposure, not raw
        # position-size magnitude. A "hold" always means zero signed exposure even
        # though upstream signal sources (e.g. live_signal._REGIME_POLICY) attach a
        # nonzero `size` to some hold rows — comparing raw magnitudes charged a phantom
        # fee/slippage on hold ticks that never actually opened a position. Comparing
        # signed exposure also correctly prices a same-magnitude long→short flip as a
        # full round-trip trade, which a naive |size - prev_size| would under-charge.
        prev_signed = prev_size if p["prev_action"] == "long" else -prev_size if p["prev_action"] == "short" else 0.0
        new_signed = size if action == "long" else -size if action == "short" else 0.0
        delta = abs(new_signed - prev_signed)
        fee      = TRANSACTION_COST * delta * capital
        slippage = (SLIPPAGE_BPS / 10_000) * delta * capital

        capital = capital * (1 + step_pnl) - fee - slippage
        total_return = capital / INITIAL_CAPITAL - 1.0

        now = datetime.now(timezone.utc).isoformat()
        n_trades = int(p["n_trades"])

        changed = delta > MIN_TRADE_SIZE
        trade_id: Optional[int] = None
        if changed:
            n_trades += 1
            # Store the realized position size, not the raw upstream `size` argument —
            # a "hold" always realizes zero exposure regardless of what magnitude the
            # signal source attached to it.
            logged_size = abs(new_signed)
            cur = con.execute(
                """INSERT INTO trades
                   (asset, timestamp, action, position_size, price, slippage, pnl, portfolio_value, regime_label)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (asset, now, action, logged_size, price, round(slippage, 8),
                 round(step_pnl, 8), round(capital, 2), regime_label),
            )
            trade_id = cur.lastrowid
            if action in ("long", "short"):
                con.execute(
                    """INSERT INTO trade_outcomes
                       (trade_id, asset, entry_price, entry_timestamp, outcome)
                       VALUES (?, ?, ?, ?, 'open')""",
                    (trade_id, asset, price, now),
                )

        _set_shared_capital(con, capital)

        con.execute(
            """INSERT INTO portfolio (asset, portfolio_value, total_return, n_trades,
               prev_action, prev_price, prev_size, updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(asset) DO UPDATE SET
                   portfolio_value=excluded.portfolio_value,
                   total_return=excluded.total_return,
                   n_trades=excluded.n_trades,
                   prev_action=excluded.prev_action,
                   prev_price=excluded.prev_price,
                   prev_size=excluded.prev_size,
                   updated_at=excluded.updated_at""",
            # portfolio_value/total_return are mirrored here for forensic/debug reads
            # only — get_portfolio()/get_account_state() always source the authoritative
            # value from shared_capital, never from this per-asset column.
            (asset, round(capital, 4), round(total_return, 6), n_trades,
             action, price, abs(new_signed), now),
        )

    return {
        "portfolio_value": round(capital, 2),
        "total_return":    round(total_return, 6),
        "n_trades":        n_trades,
        "trade_id":        trade_id,
    }


def create_pending_trade(
    asset: str, action: str, size: float, price: float,
    regime_label: int = 0, source: str = "", reasoning: Optional[dict] = None,
) -> dict:
    """Record a signal awaiting human approval — no capital moves, no position opens.

    Used by live_signal.get_live_signal() when app_settings.get_approval_mode() is on.
    `reasoning` (the same shape passed to save_trade_reasoning) is carried along so the
    "Why?" button still works once/if this trade is approved.
    """
    now = datetime.now(timezone.utc).isoformat()
    reasoning_json = json.dumps(reasoning) if reasoning else None
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO pending_trades
               (asset, action, position_size, price, regime_label, source, status, created_at, reasoning)
               VALUES (?,?,?,?,?,?, 'pending', ?, ?)""",
            (asset, action, size, price, regime_label, source, now, reasoning_json),
        )
        pending_id = cur.lastrowid
    return {
        "id": pending_id, "asset": asset, "action": action, "position_size": size,
        "price": price, "regime_label": regime_label, "source": source,
        "status": "pending", "created_at": now, "resolved_at": None, "trade_id": None,
    }


def get_pending_trades(status: str = "pending") -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM pending_trades WHERE status=? ORDER BY id DESC", (status,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d.pop("reasoning", None)  # internal only — not part of the public pending-trade shape
        out.append(d)
    return out


def get_pending_trade(pending_id: int) -> Optional[dict]:
    with _conn() as con:
        row = con.execute("SELECT * FROM pending_trades WHERE id=?", (pending_id,)).fetchone()
    return dict(row) if row else None


def approve_pending_trade(pending_id: int) -> dict:
    """Execute a pending trade for real (through the normal shared-pool `update()`
    path), re-attach its original reasoning to the resulting trade, and mark it
    approved. Raises ValueError if not found or already resolved."""
    pending = get_pending_trade(pending_id)
    if pending is None:
        raise ValueError(f"No pending trade with id={pending_id}")
    if pending["status"] != "pending":
        raise ValueError(f"Pending trade {pending_id} already {pending['status']}")

    result = update(
        pending["asset"], pending["action"], pending["position_size"],
        pending["price"], regime_label=pending["regime_label"],
    )
    trade_id = result.get("trade_id")
    if trade_id is not None and pending.get("reasoning"):
        save_trade_reasoning(trade_id, json.loads(pending["reasoning"]))

    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            "UPDATE pending_trades SET status='approved', resolved_at=?, trade_id=? WHERE id=?",
            (now, trade_id, pending_id),
        )
    return result


def reject_pending_trade(pending_id: int) -> None:
    pending = get_pending_trade(pending_id)
    if pending is None:
        raise ValueError(f"No pending trade with id={pending_id}")
    if pending["status"] != "pending":
        raise ValueError(f"Pending trade {pending_id} already {pending['status']}")
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            "UPDATE pending_trades SET status='rejected', resolved_at=? WHERE id=?",
            (now, pending_id),
        )


def save_trade_reasoning(trade_id: int, reasoning: dict) -> None:
    """Attach the agent/regime reasoning chain that produced `trade_id`."""
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO trade_reasoning
               (trade_id, asset, timestamp, regime_label, regime_description,
                debate_id, agent_theses, fund_manager_rationale, ppo_override, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                reasoning.get("asset"),
                reasoning.get("timestamp"),
                reasoning.get("regime_label"),
                reasoning.get("regime_description"),
                reasoning.get("debate_id"),
                json.dumps(reasoning.get("agent_theses", [])),
                reasoning.get("fund_manager_rationale"),
                int(reasoning.get("ppo_override", False)),
                reasoning.get("source"),
            ),
        )


def get_trade_reasoning(trade_id: int) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM trade_reasoning WHERE trade_id=?", (trade_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["agent_theses"] = json.loads(d["agent_theses"]) if d.get("agent_theses") else []
    d["ppo_override"] = bool(d["ppo_override"])
    return d


def get_trades_with_reasoning(asset: str, limit: int = 50) -> list[dict]:
    """Last `limit` trades for `asset`, each joined with its reasoning + outcome (if any)."""
    with _conn() as con:
        rows = con.execute(
            """SELECT t.*,
                      r.regime_description AS reasoning_regime_description,
                      r.debate_id, r.agent_theses, r.fund_manager_rationale,
                      r.ppo_override, r.source,
                      o.entry_price, o.entry_timestamp, o.exit_price, o.exit_timestamp,
                      o.realized_pnl, o.holding_hours, o.regime_at_exit, o.outcome
               FROM trades t
               LEFT JOIN trade_reasoning r ON r.trade_id = t.id
               LEFT JOIN trade_outcomes o ON o.trade_id = t.id
               WHERE t.asset = ?
               ORDER BY t.id DESC LIMIT ?""",
            (asset, limit),
        ).fetchall()

    out = []
    for row in reversed(rows):
        d = dict(row)
        d["agent_theses"] = json.loads(d["agent_theses"]) if d.get("agent_theses") else []
        d["ppo_override"] = bool(d["ppo_override"]) if d.get("ppo_override") is not None else False
        out.append(d)
    return out


def get_open_trades(asset: str) -> list[dict]:
    """Trades still marked 'open' in trade_outcomes, with entry price/action attached."""
    with _conn() as con:
        rows = con.execute(
            """SELECT t.id AS id, t.asset AS asset, t.action AS action,
                      o.entry_price AS price, o.entry_timestamp AS entry_timestamp
               FROM trade_outcomes o
               JOIN trades t ON t.id = o.trade_id
               WHERE o.asset = ? AND o.outcome = 'open'""",
            (asset,),
        ).fetchall()
    return [dict(r) for r in rows]


def close_trade_outcome(
    trade_id: int,
    exit_price: float,
    realized_pnl: float,
    outcome: str,
    regime_at_exit: Optional[int] = None,
) -> None:
    with _conn() as con:
        row = con.execute(
            "SELECT entry_timestamp FROM trade_outcomes WHERE trade_id=?", (trade_id,)
        ).fetchone()
        holding_hours = None
        exit_ts = datetime.now(timezone.utc)
        if row and row["entry_timestamp"]:
            entry_ts = datetime.fromisoformat(row["entry_timestamp"])
            holding_hours = round((exit_ts - entry_ts).total_seconds() / 3600, 4)

        con.execute(
            """UPDATE trade_outcomes SET
                   exit_price=?, exit_timestamp=?, realized_pnl=?,
                   holding_hours=?, regime_at_exit=?, outcome=?
               WHERE trade_id=?""",
            (
                exit_price, exit_ts.isoformat(), round(realized_pnl, 6),
                holding_hours, regime_at_exit, outcome, trade_id,
            ),
        )


def get_trade_outcome_stats(asset: str) -> dict:
    """Win rate / avg holding time / avg PnL from closed trade_outcomes rows."""
    with _conn() as con:
        rows = con.execute(
            """SELECT realized_pnl, holding_hours, outcome FROM trade_outcomes
               WHERE asset = ? AND outcome IN ('profit', 'loss')""",
            (asset,),
        ).fetchall()

    n_closed = len(rows)
    if n_closed == 0:
        return {
            "asset": asset, "n_closed": 0, "n_open": len(get_open_trades(asset)),
            "win_rate": None, "avg_holding_hours": None, "avg_pnl": None,
        }

    n_wins = sum(1 for r in rows if r["outcome"] == "profit")
    avg_pnl = sum(r["realized_pnl"] or 0.0 for r in rows) / n_closed
    holding_vals = [r["holding_hours"] for r in rows if r["holding_hours"] is not None]
    avg_holding = sum(holding_vals) / len(holding_vals) if holding_vals else None

    return {
        "asset": asset,
        "n_closed": n_closed,
        "n_open": len(get_open_trades(asset)),
        "win_rate": round(n_wins / n_closed, 4),
        "avg_holding_hours": round(avg_holding, 2) if avg_holding is not None else None,
        "avg_pnl": round(avg_pnl, 6),
    }


def reset(asset: str) -> None:
    """Wipe paper portfolio for an asset (for testing / manual resets)."""
    with _conn() as con:
        con.execute("DELETE FROM portfolio WHERE asset=?", (asset,))
        con.execute("DELETE FROM trades WHERE asset=?", (asset,))
        con.execute("DELETE FROM trade_reasoning WHERE asset=?", (asset,))
        con.execute("DELETE FROM trade_outcomes WHERE asset=?", (asset,))
