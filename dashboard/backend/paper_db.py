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

Capital structure — ONE shared pool across assets, PER USER:
    Every live asset (BTC_USDT, ETH_USDT, ...) draws from a single capital row rather
    than each getting its own independent $10,000 balance — two assets each silently
    getting their own $10k pool would mean $20k deployed simultaneously, which doesn't
    match the trained RL model (it manages one portfolio value and one signed position
    at a time, see rl/env.py). Per-asset rows in the `portfolio` table hold only
    *position state* (prev_action/prev_price/prev_size/n_trades) for that asset; the
    account's actual value/return always comes from `shared_capital`.
    `update()` additionally clips a new position so this asset's exposure plus every
    other asset's already-open exposure never exceeds 100% of the pool.

    As of the per-user migration (see _migrate_schema), this pooling is scoped per
    *user*, not per app — every function below takes a leading `user_id` and every
    query/write is filtered by it. `portfolio` is keyed by `(user_id, asset)` and
    `shared_capital` has one row per user. The underlying pooling/exposure-cap logic
    is unchanged; it just no longer spans across different people's accounts.

    The trading *decision* itself (what live_signal.compute_signal() resolves — PPO
    or LLM-consensus action/size for an asset) is still computed once per asset per
    tick, not per user — only the bookkeeping here is per-user. See
    signal_scheduler.py's per-user fan-out loop.

Approval mode (see app_settings.get_approval_mode):
    When ON, live_signal.book_signal() records a non-hold signal as a row in
    `pending_trades` instead of calling `update()` — no capital moves and no position
    opens until a human calls approve_pending_trade()/reject_pending_trade() (wired to
    POST /api/trades/pending/{id}/approve|reject in main.py). Pending trades are
    per-user too — one user approving theirs never touches another user's queue.
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
    user_id          INTEGER NOT NULL,
    asset            TEXT NOT NULL,
    portfolio_value  REAL NOT NULL DEFAULT 10000.0,
    total_return     REAL NOT NULL DEFAULT 0.0,
    n_trades         INTEGER NOT NULL DEFAULT 0,
    prev_action      TEXT NOT NULL DEFAULT 'hold',
    prev_price       REAL NOT NULL DEFAULT 0.0,
    prev_size        REAL NOT NULL DEFAULT 0.0,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (user_id, asset)
);

CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER,
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
    user_id              INTEGER,
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
    user_id          INTEGER,
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
    user_id          INTEGER PRIMARY KEY,
    capital          REAL NOT NULL DEFAULT 10000.0,
    peak_capital     REAL NOT NULL DEFAULT 10000.0,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER,
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


_schema_migrated = False  # module-level flag — run the ALTER TABLE probe once per process


def _migrate_schema(con: sqlite3.Connection) -> None:
    """Two independent, additive migrations. Each is guarded so it's a no-op on a DB
    that already has it (including brand-new DBs, which get everything straight from
    `_DDL`):

    1. `peak_capital` on `shared_capital` (pre-multi-user fix, unchanged from before).
    2. `user_id` everywhere — the per-user paper trading migration. `portfolio`'s PK
       changes from `asset` alone to `(user_id, asset)`, and `shared_capital`'s PK
       changes from a hardcoded singleton (`id=1`) to one row per user — SQLite can't
       ALTER either change in place, so both get the create-new/copy/drop/rename
       rebuild (same idiom as auth_db.py's own migration). `trades`/`trade_reasoning`/
       `trade_outcomes`/`pending_trades` just gain a plain `user_id` column.

       Existing rows attach to the PRIMARY (lowest-id) user in auth_db's `users` table
       — auth_db.py is a separate SQLite file, hence the cross-file lookup via
       `auth_db.get_primary_user_id()` rather than a join. If no user exists yet at
       migration time, the schema rebuild still happens (a schema is a schema
       regardless of data) but rows are left with `user_id IS NULL` until a human
       backfills them — this whole function only runs once per process (see
       `_conn()`'s `_schema_migrated` flag), so a signup moments later in the same
       process won't retroactively retrigger the backfill. Vanishingly unlikely in
       practice: a paper_trading.db with real trade history implies the app has
       already been running, and by then at least one account almost certainly exists.
    """
    try:
        con.execute(
            "ALTER TABLE shared_capital ADD COLUMN peak_capital REAL NOT NULL DEFAULT 10000.0"
        )
    except sqlite3.OperationalError:
        pass
    else:
        con.execute(
            """UPDATE shared_capital SET peak_capital = MAX(
                   peak_capital, capital, COALESCE((SELECT MAX(portfolio_value) FROM trades), 0)
               ) WHERE id = 1"""
        )

    portfolio_cols = {row["name"] for row in con.execute("PRAGMA table_info(portfolio)")}
    if "user_id" in portfolio_cols:
        return  # already on the per-user schema

    import auth_db
    primary_id = auth_db.get_primary_user_id()

    con.execute(
        """CREATE TABLE portfolio_new (
               user_id          INTEGER NOT NULL,
               asset            TEXT NOT NULL,
               portfolio_value  REAL NOT NULL DEFAULT 10000.0,
               total_return     REAL NOT NULL DEFAULT 0.0,
               n_trades         INTEGER NOT NULL DEFAULT 0,
               prev_action      TEXT NOT NULL DEFAULT 'hold',
               prev_price       REAL NOT NULL DEFAULT 0.0,
               prev_size        REAL NOT NULL DEFAULT 0.0,
               updated_at       TEXT NOT NULL,
               PRIMARY KEY (user_id, asset)
           )"""
    )
    if primary_id is not None:
        con.execute(
            """INSERT INTO portfolio_new (user_id, asset, portfolio_value, total_return,
                   n_trades, prev_action, prev_price, prev_size, updated_at)
               SELECT ?, asset, portfolio_value, total_return, n_trades,
                      prev_action, prev_price, prev_size, updated_at FROM portfolio""",
            (primary_id,),
        )
    con.execute("DROP TABLE portfolio")
    con.execute("ALTER TABLE portfolio_new RENAME TO portfolio")

    con.execute(
        """CREATE TABLE shared_capital_new (
               user_id      INTEGER PRIMARY KEY,
               capital      REAL NOT NULL DEFAULT 10000.0,
               peak_capital REAL NOT NULL DEFAULT 10000.0,
               updated_at   TEXT NOT NULL
           )"""
    )
    if primary_id is not None:
        con.execute(
            """INSERT INTO shared_capital_new (user_id, capital, peak_capital, updated_at)
               SELECT ?, capital, peak_capital, updated_at FROM shared_capital WHERE id = 1""",
            (primary_id,),
        )
    con.execute("DROP TABLE shared_capital")
    con.execute("ALTER TABLE shared_capital_new RENAME TO shared_capital")

    for table in ("trades", "trade_reasoning", "trade_outcomes", "pending_trades"):
        cols = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
        if "user_id" not in cols:
            con.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")
            if primary_id is not None:
                con.execute(f"UPDATE {table} SET user_id = ?", (primary_id,))


@contextmanager
def _conn():
    global _schema_migrated
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_DDL)
        if not _schema_migrated:
            _migrate_schema(con)
            _schema_migrated = True
        yield con
        con.commit()
    finally:
        con.close()


def _apply_constraints(size: float) -> float:
    """Clip position to [−MAX, +MAX]; snap micro-sizes to 0."""
    if abs(size) < MIN_TRADE_SIZE:
        return 0.0
    return float(max(-MAX_POSITION, min(MAX_POSITION, size)))


def _get_shared_capital(con: sqlite3.Connection, user_id: int) -> float:
    row = con.execute("SELECT capital FROM shared_capital WHERE user_id=?", (user_id,)).fetchone()
    if row:
        return float(row["capital"])
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        "INSERT INTO shared_capital (user_id, capital, updated_at) VALUES (?, ?, ?)",
        (user_id, INITIAL_CAPITAL, now),
    )
    return INITIAL_CAPITAL


def _set_shared_capital(con: sqlite3.Connection, user_id: int, value: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    v = round(value, 4)
    con.execute(
        """INSERT INTO shared_capital (user_id, capital, peak_capital, updated_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
               capital=excluded.capital,
               peak_capital=MAX(shared_capital.peak_capital, excluded.capital),
               updated_at=excluded.updated_at""",
        (user_id, v, v, now),
    )


def _other_assets_exposure(con: sqlite3.Connection, user_id: int, asset: str) -> float:
    """Sum of |signed position| this USER currently has open on every OTHER asset, as
    a fraction of their own pool — used to cap a new position so this user's combined
    exposure across all assets never exceeds 100% of their own capital."""
    rows = con.execute(
        "SELECT prev_action, prev_size FROM portfolio WHERE user_id=? AND asset != ?",
        (user_id, asset),
    ).fetchall()
    return sum(abs(float(r["prev_size"])) for r in rows if r["prev_action"] in ("long", "short"))


def get_portfolio(user_id: int, asset: str) -> dict:
    """Position state for `asset` + this user's shared capital value/return.

    portfolio_value/total_return always come from the user's shared_capital row (never
    from a per-asset column) so they can't go stale relative to what other assets did.
    """
    with _conn() as con:
        capital = _get_shared_capital(con, user_id)
        row = con.execute(
            "SELECT * FROM portfolio WHERE user_id=? AND asset=?", (user_id, asset)
        ).fetchone()
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


def get_account_state(user_id: int) -> dict:
    """Account-wide view of this user's shared capital pool — the live counterpart to
    the trained model's single-portfolio assumption. Each entry's `exposure` is that
    asset's currently open position as a fraction of THIS USER's pool, not an
    independent per-asset balance and not shared with any other user."""
    with _conn() as con:
        capital = _get_shared_capital(con, user_id)
        rows = con.execute(
            "SELECT asset, prev_action, prev_size, n_trades, updated_at FROM portfolio WHERE user_id=?",
            (user_id,),
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


def get_portfolio_state_for_obs(user_id: int, asset: str) -> dict:
    """Read-only snapshot of everything ppo_inference._build_portfolio_vec needs for
    this user + `asset`. Called from inside ppo_inference._build_observation() ->
    predict(), itself called (per-user, when PPO is the active signal source — see
    live_signal.book_signal) BEFORE this tick's paper_db.update() runs — so this
    naturally returns PRE-decision state, matching rl/env.py's training-time semantics
    where _signed_position/_steps_since_trade reflect state before the current action.
    """
    with _conn() as con:
        capital = _get_shared_capital(con, user_id)
        peak_row = con.execute(
            "SELECT peak_capital FROM shared_capital WHERE user_id=?", (user_id,)
        ).fetchone()
        peak = float(peak_row["peak_capital"]) if peak_row and peak_row["peak_capital"] is not None else capital
        pos_row = con.execute(
            "SELECT prev_action, prev_size FROM portfolio WHERE user_id=? AND asset=?",
            (user_id, asset),
        ).fetchone()
        last_trade = con.execute(
            "SELECT timestamp FROM trades WHERE user_id=? AND asset=? ORDER BY id DESC LIMIT 1",
            (user_id, asset),
        ).fetchone()

    signed_position = 0.0
    if pos_row:
        if pos_row["prev_action"] == "long":
            signed_position = float(pos_row["prev_size"])
        elif pos_row["prev_action"] == "short":
            signed_position = -float(pos_row["prev_size"])

    drawdown = max(0.0, (peak - capital) / peak) if peak > 0 else 0.0

    hours_since_trade: Optional[float] = None
    if last_trade and last_trade["timestamp"]:
        delta = datetime.now(timezone.utc) - datetime.fromisoformat(last_trade["timestamp"])
        hours_since_trade = max(0.0, delta.total_seconds() / 3600.0)

    return {
        "portfolio_value":        capital,
        "portfolio_return":       capital / INITIAL_CAPITAL - 1.0,
        "peak_capital":           peak,
        "drawdown":               drawdown,
        "signed_position":        signed_position,
        "hours_since_last_trade": hours_since_trade,  # None => this user has never traded this asset
    }


def get_trades(user_id: int, asset: str, limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM trades WHERE user_id=? AND asset=? ORDER BY id DESC LIMIT ?",
            (user_id, asset, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def update(
    user_id: int,
    asset: str,
    action: str,
    size: float,
    price: float,
    regime_label: int = 0,
) -> dict:
    """Update the paper portfolio for one signal tick, for one user.

    Returns the updated portfolio state dict.
    """
    size = _apply_constraints(size)

    with _conn() as con:
        row = con.execute(
            "SELECT * FROM portfolio WHERE user_id=? AND asset=?", (user_id, asset)
        ).fetchone()
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
        capital    = _get_shared_capital(con, user_id)

        # Shared pool across assets (per user): this asset's new exposure is capped so
        # that, combined with every other asset this user already has open, total
        # deployed capital never exceeds 100% of their own pool (no silent 2x
        # deployment from BTC + ETH each drawing an independent $10k — see module
        # docstring).
        if action in ("long", "short"):
            room = max(0.0, 1.0 - _other_assets_exposure(con, user_id, asset))
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
                   (user_id, asset, timestamp, action, position_size, price, slippage, pnl, portfolio_value, regime_label)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (user_id, asset, now, action, logged_size, price, round(slippage, 8),
                 round(step_pnl, 8), round(capital, 2), regime_label),
            )
            trade_id = cur.lastrowid
            if action in ("long", "short"):
                con.execute(
                    """INSERT INTO trade_outcomes
                       (trade_id, user_id, asset, entry_price, entry_timestamp, outcome)
                       VALUES (?, ?, ?, ?, ?, 'open')""",
                    (trade_id, user_id, asset, price, now),
                )

        _set_shared_capital(con, user_id, capital)

        con.execute(
            """INSERT INTO portfolio (user_id, asset, portfolio_value, total_return, n_trades,
               prev_action, prev_price, prev_size, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id, asset) DO UPDATE SET
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
            (user_id, asset, round(capital, 4), round(total_return, 6), n_trades,
             action, price, abs(new_signed), now),
        )

    return {
        "portfolio_value": round(capital, 2),
        "total_return":    round(total_return, 6),
        "n_trades":        n_trades,
        "trade_id":        trade_id,
    }


def create_pending_trade(
    user_id: int, asset: str, action: str, size: float, price: float,
    regime_label: int = 0, source: str = "", reasoning: Optional[dict] = None,
) -> dict:
    """Record a signal awaiting human approval — no capital moves, no position opens.

    Used by live_signal.book_signal() when app_settings.get_approval_mode() is on.
    `reasoning` (the same shape passed to save_trade_reasoning) is carried along so the
    "Why?" button still works once/if this trade is approved.
    """
    now = datetime.now(timezone.utc).isoformat()
    reasoning_json = json.dumps(reasoning) if reasoning else None
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO pending_trades
               (user_id, asset, action, position_size, price, regime_label, source, status, created_at, reasoning)
               VALUES (?,?,?,?,?,?,?, 'pending', ?, ?)""",
            (user_id, asset, action, size, price, regime_label, source, now, reasoning_json),
        )
        pending_id = cur.lastrowid
    return {
        "id": pending_id, "asset": asset, "action": action, "position_size": size,
        "price": price, "regime_label": regime_label, "source": source,
        "status": "pending", "created_at": now, "resolved_at": None, "trade_id": None,
    }


def get_pending_trades(user_id: int, status: str = "pending") -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM pending_trades WHERE user_id=? AND status=? ORDER BY id DESC",
            (user_id, status),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d.pop("reasoning", None)  # internal only — not part of the public pending-trade shape
        out.append(d)
    return out


def get_pending_trade(pending_id: int, expected_user_id: Optional[int] = None) -> Optional[dict]:
    """Look up a pending trade by its globally-unique id. If `expected_user_id` is
    given, raises ValueError on a mismatch — a defense-in-depth guard so a caller can't
    approve/reject another user's pending trade even by guessing an id."""
    with _conn() as con:
        row = con.execute("SELECT * FROM pending_trades WHERE id=?", (pending_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if expected_user_id is not None and d.get("user_id") != expected_user_id:
        raise ValueError(f"Pending trade {pending_id} does not belong to user {expected_user_id}")
    return d


def approve_pending_trade(pending_id: int, user_id: int) -> dict:
    """Execute a pending trade for real (through the normal shared-pool `update()`
    path) for `user_id`, re-attach its original reasoning to the resulting trade, and
    mark it approved. Raises ValueError if not found, already resolved, or if it
    belongs to a different user."""
    pending = get_pending_trade(pending_id, expected_user_id=user_id)
    if pending is None:
        raise ValueError(f"No pending trade with id={pending_id}")
    if pending["status"] != "pending":
        raise ValueError(f"Pending trade {pending_id} already {pending['status']}")

    result = update(
        user_id, pending["asset"], pending["action"], pending["position_size"],
        pending["price"], regime_label=pending["regime_label"],
    )
    trade_id = result.get("trade_id")
    if trade_id is not None and pending.get("reasoning"):
        save_trade_reasoning(trade_id, user_id, json.loads(pending["reasoning"]))

    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            "UPDATE pending_trades SET status='approved', resolved_at=?, trade_id=? WHERE id=?",
            (now, trade_id, pending_id),
        )
    return result


def reject_pending_trade(pending_id: int, user_id: int) -> None:
    pending = get_pending_trade(pending_id, expected_user_id=user_id)
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


def save_trade_reasoning(trade_id: int, user_id: int, reasoning: dict) -> None:
    """Attach the agent/regime reasoning chain that produced `trade_id`."""
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO trade_reasoning
               (trade_id, user_id, asset, timestamp, regime_label, regime_description,
                debate_id, agent_theses, fund_manager_rationale, ppo_override, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                user_id,
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


def get_trade_reasoning(trade_id: int, user_id: int) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM trade_reasoning WHERE trade_id=? AND user_id=?", (trade_id, user_id)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["agent_theses"] = json.loads(d["agent_theses"]) if d.get("agent_theses") else []
    d["ppo_override"] = bool(d["ppo_override"])
    return d


def get_trades_with_reasoning(user_id: int, asset: str, limit: int = 50) -> list[dict]:
    """Last `limit` trades for this user + `asset`, each joined with its reasoning + outcome (if any)."""
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
               WHERE t.user_id = ? AND t.asset = ?
               ORDER BY t.id DESC LIMIT ?""",
            (user_id, asset, limit),
        ).fetchall()

    out = []
    for row in reversed(rows):
        d = dict(row)
        d["agent_theses"] = json.loads(d["agent_theses"]) if d.get("agent_theses") else []
        d["ppo_override"] = bool(d["ppo_override"]) if d.get("ppo_override") is not None else False
        out.append(d)
    return out


def get_open_trades(user_id: int, asset: str) -> list[dict]:
    """Trades still marked 'open' in trade_outcomes, with entry price/action attached."""
    with _conn() as con:
        rows = con.execute(
            """SELECT t.id AS id, t.asset AS asset, t.action AS action,
                      o.entry_price AS price, o.entry_timestamp AS entry_timestamp
               FROM trade_outcomes o
               JOIN trades t ON t.id = o.trade_id
               WHERE o.user_id = ? AND o.asset = ? AND o.outcome = 'open'""",
            (user_id, asset),
        ).fetchall()
    return [dict(r) for r in rows]


def close_trade_outcome(
    trade_id: int,
    exit_price: float,
    realized_pnl: float,
    outcome: str,
    regime_at_exit: Optional[int] = None,
) -> None:
    """No user_id param needed — `trade_id` is already a globally unique key into
    trade_outcomes, and every caller (signal_scheduler._update_trade_outcomes) already
    obtained it from a user-scoped get_open_trades() call."""
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


def get_trade_outcome_stats(user_id: int, asset: str) -> dict:
    """Win rate / avg holding time / avg PnL from closed trade_outcomes rows."""
    with _conn() as con:
        rows = con.execute(
            """SELECT realized_pnl, holding_hours, outcome FROM trade_outcomes
               WHERE user_id = ? AND asset = ? AND outcome IN ('profit', 'loss')""",
            (user_id, asset),
        ).fetchall()

    n_closed = len(rows)
    if n_closed == 0:
        return {
            "asset": asset, "n_closed": 0, "n_open": len(get_open_trades(user_id, asset)),
            "win_rate": None, "avg_holding_hours": None, "avg_pnl": None,
        }

    n_wins = sum(1 for r in rows if r["outcome"] == "profit")
    avg_pnl = sum(r["realized_pnl"] or 0.0 for r in rows) / n_closed
    holding_vals = [r["holding_hours"] for r in rows if r["holding_hours"] is not None]
    avg_holding = sum(holding_vals) / len(holding_vals) if holding_vals else None

    return {
        "asset": asset,
        "n_closed": n_closed,
        "n_open": len(get_open_trades(user_id, asset)),
        "win_rate": round(n_wins / n_closed, 4),
        "avg_holding_hours": round(avg_holding, 2) if avg_holding is not None else None,
        "avg_pnl": round(avg_pnl, 6),
    }


def reset(user_id: int, asset: str) -> None:
    """Wipe one user's paper portfolio for an asset (for testing / manual resets)."""
    with _conn() as con:
        con.execute("DELETE FROM portfolio WHERE user_id=? AND asset=?", (user_id, asset))
        con.execute("DELETE FROM trades WHERE user_id=? AND asset=?", (user_id, asset))
        con.execute("DELETE FROM trade_reasoning WHERE user_id=? AND asset=?", (user_id, asset))
        con.execute("DELETE FROM trade_outcomes WHERE user_id=? AND asset=?", (user_id, asset))
