"""SQLite-backed multi-user auth — users, password hashes, and server-side sessions.

Same raw-sqlite3 convention as paper_db.py/chat_db.py: own DB file (not shared),
a `_DDL` string run via executescript on every connection, a `_conn()` context
manager, plain dict-returning functions. Session tokens are opaque random strings
(secrets.token_urlsafe), not JWTs — simpler to revoke (just delete the row) and
nothing to decode/verify client-side.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))
_DB_PATH = Path(os.environ.get("AUTH_DB_PATH", str(MARTA_ROOT / "data" / "auth.db")))

SESSION_TTL_DAYS = 30
SESSION_COOKIE_NAME = "marta_session"

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT,
    display_name   TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    oauth_provider TEXT,
    oauth_id       TEXT,
    avatar_url     TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token          TEXT PRIMARY KEY,
    user_id        INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


def _migrate(con: sqlite3.Connection) -> None:
    """Upgrade DBs created before OAuth support (password_hash was NOT NULL, no
    oauth_* columns). Guarded so it's a cheap no-op on every call after the
    first — _conn() re-runs this on every connection, same as _DDL above.
    """
    cols = {row["name"]: row for row in con.execute("PRAGMA table_info(users)")}
    if "oauth_provider" not in cols:
        con.execute("ALTER TABLE users ADD COLUMN oauth_provider TEXT")
        con.execute("ALTER TABLE users ADD COLUMN oauth_id TEXT")
        con.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
        cols = {row["name"]: row for row in con.execute("PRAGMA table_info(users)")}
    if cols["password_hash"]["notnull"]:
        # SQLite can't relax a column's NOT NULL in place — rebuild the table.
        con.executescript("""
            CREATE TABLE users_new (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                email          TEXT NOT NULL UNIQUE,
                password_hash  TEXT,
                display_name   TEXT NOT NULL DEFAULT '',
                created_at     TEXT NOT NULL,
                oauth_provider TEXT,
                oauth_id       TEXT,
                avatar_url     TEXT
            );
            INSERT INTO users_new (id, email, password_hash, display_name, created_at,
                                    oauth_provider, oauth_id, avatar_url)
                SELECT id, email, password_hash, display_name, created_at,
                       oauth_provider, oauth_id, avatar_url FROM users;
            DROP TABLE users;
            ALTER TABLE users_new RENAME TO users;
        """)
    # Unconditional + IF NOT EXISTS: covers the branches above, and the fresh-DB
    # case where _DDL already created the final schema and neither branch ran.
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth "
        "ON users(oauth_provider, oauth_id) WHERE oauth_provider IS NOT NULL"
    )


@contextmanager
def _conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        con.executescript(_DDL)
        _migrate(con)
        yield con
        con.commit()
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_public(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "avatar_url": row["avatar_url"],
    }


# ── Passwords ────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    if not password_hash:
        # OAuth-only account (no password set) — never crash, just refuse.
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed/legacy hash — never crash a login attempt over it.
        return False


# ── Users ────────────────────────────────────────────────────────────────────

def create_user(email: str, password: str, display_name: str = "") -> dict:
    """Raises sqlite3.IntegrityError if the email is already registered."""
    email = email.strip().lower()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
            (email, hash_password(password), display_name or email.split("@")[0], _now_iso()),
        )
        row = con.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _user_public(row)


def authenticate(email: str, password: str) -> Optional[dict]:
    email = email.strip().lower()
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return _user_public(row)


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_public(row) if row else None


def list_user_ids() -> list[int]:
    """All user ids, ascending — the lowest is the primary/first account. Used by
    paper_db.py's per-user migration backfill and by signal_scheduler.py's per-user
    fan-out loop (paper_db.py and auth_db.py are separate SQLite files, so this is
    the cross-file lookup point rather than a join)."""
    with _conn() as con:
        rows = con.execute("SELECT id FROM users ORDER BY id").fetchall()
    return [int(r["id"]) for r in rows]


def get_primary_user_id() -> Optional[int]:
    """Lowest user id, or None if no users exist yet (fresh deployment)."""
    ids = list_user_ids()
    return ids[0] if ids else None


def get_or_create_oauth_user(
    provider: str, oauth_id: str, email: str, display_name: str = "", avatar_url: str = ""
) -> dict:
    """Find-or-create for OAuth logins (Google, etc). Matches an existing
    password-based account by email first (so the same person can log in
    either way), else creates a password-less account.
    """
    email = email.strip().lower()
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE oauth_provider = ? AND oauth_id = ?", (provider, oauth_id)
        ).fetchone()
        if row is None:
            row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            cur = con.execute(
                """INSERT INTO users (email, password_hash, display_name, created_at,
                                       oauth_provider, oauth_id, avatar_url)
                   VALUES (?, NULL, ?, ?, ?, ?, ?)""",
                (email, display_name or email.split("@")[0], _now_iso(), provider, oauth_id, avatar_url),
            )
            row = con.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        elif row["oauth_provider"] is None:
            con.execute(
                "UPDATE users SET oauth_provider = ?, oauth_id = ?, avatar_url = ? WHERE id = ?",
                (provider, oauth_id, avatar_url, row["id"]),
            )
            row = con.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
        return _user_public(row)


def update_profile(user_id: int, display_name: Optional[str] = None, new_password: Optional[str] = None) -> dict:
    with _conn() as con:
        if display_name is not None:
            con.execute("UPDATE users SET display_name = ? WHERE id = ?", (display_name, user_id))
        if new_password is not None:
            con.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(new_password), user_id),
            )
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_public(row)


# ── Sessions ─────────────────────────────────────────────────────────────────

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=SESSION_TTL_DAYS)
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires.isoformat()),
        )
    return token


def get_session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    with _conn() as con:
        row = con.execute(
            """SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token = ? AND s.expires_at > ?""",
            (token, _now_iso()),
        ).fetchone()
        return _user_public(row) if row else None


def delete_session(token: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM sessions WHERE token = ?", (token,))


def set_session_cookie(response, token: str) -> None:
    """Shared by main.py's password login/signup and oauth_google.py's callback."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_DAYS * 24 * 3600,
        httponly=True,
        samesite="lax",
        # Not marked secure= — see auth.py's MARTA_API_KEY docstring for the same
        # plain-http-on-localhost/LAN assumption. Revisit if this is ever behind TLS.
        path="/",
    )
