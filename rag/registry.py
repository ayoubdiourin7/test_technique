from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentRecord:
    doc_id: str
    original_name: str
    stored_path: str
    ext: str
    chunk_ids: list[str]


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        desired_columns = {"doc_id", "original_name", "stored_path", "ext", "chunk_ids"}
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    ext TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL
                );
                """
            )
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(documents);").fetchall()
            }
            if cols != desired_columns:
                conn.execute("DROP TABLE IF EXISTS documents;")
                conn.execute(
                    """
                    CREATE TABLE documents (
                        doc_id TEXT PRIMARY KEY,
                        original_name TEXT NOT NULL,
                        stored_path TEXT NOT NULL,
                        ext TEXT NOT NULL,
                        chunk_ids TEXT NOT NULL
                    );
                    """
                )
            conn.commit()

    def list(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, original_name, stored_path, ext, chunk_ids
                FROM documents
                ORDER BY rowid DESC
                """
            ).fetchall()
        return [
            DocumentRecord(
                doc_id=row[0],
                original_name=row[1],
                stored_path=row[2],
                ext=row[3],
                chunk_ids=json.loads(row[4]) if row[4] else [],
            )
            for row in rows
        ]

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT doc_id, original_name, stored_path, ext, chunk_ids
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            doc_id=row[0],
            original_name=row[1],
            stored_path=row[2],
            ext=row[3],
            chunk_ids=json.loads(row[4]) if row[4] else [],
        )

    def add(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, original_name, stored_path, ext, chunk_ids
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.doc_id,
                    record.original_name,
                    record.stored_path,
                    record.ext,
                    json.dumps(record.chunk_ids, ensure_ascii=True),
                ),
            )
            conn.commit()

    def remove(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT doc_id, original_name, stored_path, ext, chunk_ids
                FROM documents
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

        return DocumentRecord(
            doc_id=row[0],
            original_name=row[1],
            stored_path=row[2],
            ext=row[3],
            chunk_ids=json.loads(row[4]) if row[4] else [],
        )
