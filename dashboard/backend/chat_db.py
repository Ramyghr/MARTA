"""SQLite-backed chat conversation history — survives server restarts.

Replaces chat_service.py's in-memory ``_chat_histories`` dict, which lost every
conversation on backend restart and gave the frontend nothing to list or reload.
Same raw-sqlite3 pattern as paper_db.py (own DB file, not a shared one — chat
history and the paper trading ledger have no reason to be coupled, and keeping
them separate means either can be wiped/inspected independently).

The DB file is written to ``MARTA_ROOT/data/chat_history.db`` (created on first use).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))
_DB_PATH = Path(
    os.environ.get("CHAT_DB_PATH", str(MARTA_ROOT / "data" / "chat_history.db"))
)

MAX_TITLE_LEN = 60

_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    asset       TEXT NOT NULL DEFAULT 'BTC_USDT',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_title(first_user_message: str) -> str:
    title = " ".join(first_user_message.split())  # collapse newlines/whitespace
    if len(title) > MAX_TITLE_LEN:
        title = title[:MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"
    return title or "New conversation"


def append_message(session_id: str, role: str, content: str, asset: Optional[str] = None) -> None:
    """Store one message and bump the conversation's updated_at. Titles the
    conversation from the first user message the first time one arrives."""
    with _conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO conversations (id, asset, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, asset or "BTC_USDT", _now(), _now()),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        update_fields = ["updated_at = ?"]
        params: list = [_now()]
        if role == "user":
            row = con.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ? AND role = 'user'",
                (session_id,),
            ).fetchone()
            if row["n"] == 1:  # this insert was the first user message
                update_fields.append("title = ?")
                params.append(_derive_title(content))
        if asset:
            update_fields.append("asset = ?")
            params.append(asset)
        params.append(session_id)
        con.execute(f"UPDATE conversations SET {', '.join(update_fields)} WHERE id = ?", params)


def list_conversations(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT c.id, c.title, c.asset, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
               FROM conversations c
               WHERE message_count > 0
               ORDER BY c.updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(session_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(session_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM messages WHERE conversation_id = ?", (session_id,))
        con.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
