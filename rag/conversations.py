from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from rag.config import CONVERSATIONS_DB_PATH


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class Conversation:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class Message:
    message_id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    sources: list[dict]


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);"
            )
            # Migrate old schemas lacking the sources column.
            info = conn.execute("PRAGMA table_info(messages);").fetchall()
            cols = {row[1] for row in info}
            if "sources" not in cols:
                conn.execute(
                    "ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT '[]';"
                )
            conn.commit()

    def _hydrate_conversation(self, row: tuple) -> Conversation:
        return Conversation(
            conversation_id=row[0], title=row[1], created_at=row[2], updated_at=row[3]
        )

    def _hydrate_message(self, row: tuple) -> Message:
        return Message(
            message_id=row[0],
            conversation_id=row[1],
            role=row[2],
            content=row[3],
            created_at=row[4],
            sources=json.loads(row[5]) if len(row) > 5 and row[5] else [],
        )

    def count_conversations(self) -> int:
        with self._connect() as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM conversations;").fetchone()
            return count or 0

    def create_conversation(self, title: str | None = None) -> Conversation:
        ts = _now()
        conv_id = uuid4().hex
        if title is None:
            title = f"Conversation {self.count_conversations() + 1}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conv_id, title, ts, ts),
            )
            conn.commit()
        return Conversation(conversation_id=conv_id, title=title, created_at=ts, updated_at=ts)

    def list_conversations(self) -> list[Conversation]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._hydrate_conversation(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return self._hydrate_conversation(row) if row else None

    def get_most_recent(self) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self._hydrate_conversation(row) if row else None

    def ensure_default_conversation(self) -> Conversation:
        existing = self.get_most_recent()
        if existing:
            return existing
        return self.create_conversation()

    def add_message(
        self, conversation_id: str, role: str, content: str, sources: list[dict] | None = None
    ) -> Message:
        ts = _now()
        payload = json.dumps(sources or [], ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at, sources)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, ts, payload),
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE conversation_id = ?
                """,
                (ts, conversation_id),
            )
            conn.commit()
            message_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
        return Message(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=ts,
            sources=sources or [],
        )

    def list_messages(self, conversation_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, conversation_id, role, content, created_at, sources
                FROM messages
                WHERE conversation_id = ?
                ORDER BY message_id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._hydrate_message(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if not existing:
                return False
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            conn.commit()
        return True


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStore:
    return ConversationStore(CONVERSATIONS_DB_PATH)
