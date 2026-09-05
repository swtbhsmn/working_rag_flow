import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, stored_path TEXT NOT NULL,
                    media_type TEXT NOT NULL, size_bytes INTEGER NOT NULL, status TEXT NOT NULL,
                    extracted_text TEXT, chunk_size INTEGER NOT NULL, chunk_overlap INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0, token_count INTEGER NOT NULL DEFAULT 0,
                    model_id TEXT, embedding_dim INTEGER, fingerprint TEXT,
                    created_at TEXT NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL, page INTEGER, text TEXT NOT NULL,
                    token_count INTEGER NOT NULL, vector BLOB NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, entity_id TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(job_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id, sequence);
                """
            )

    def create_job(self, job_id: str, kind: str, entity_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, 'running', ?)",
                (job_id, kind, entity_id, utc_now()),
            )

    def add_event(self, job_id: str, stage: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS seq FROM events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            sequence = int(row["seq"])
            timestamp = utc_now()
            conn.execute(
                "INSERT INTO events(job_id, sequence, stage, status, timestamp, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, sequence, stage, status, timestamp, json.dumps(payload)),
            )
            if status == "failed":
                conn.execute("UPDATE jobs SET status = 'failed' WHERE id = ?", (job_id,))
            elif stage == "complete" and status == "completed":
                conn.execute("UPDATE jobs SET status = 'completed' WHERE id = ?", (job_id,))
        return {"sequence": sequence, "stage": stage, "status": status, "timestamp": timestamp, "payload": payload}

    def events_after(self, job_id: str, sequence: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT sequence, stage, status, timestamp, payload FROM events WHERE job_id = ? AND sequence > ? ORDER BY sequence",
                (job_id, sequence),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def chunks_for(self, document_ids: list[str]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in document_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.document_id IN ({placeholders}) ORDER BY c.document_id, c.chunk_index",
                document_ids,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["vector"] = np.frombuffer(item["vector"], dtype=np.float32).copy()
            result.append(item)
        return result

    def insert_document(self, values: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO documents
                (id, filename, stored_path, media_type, size_bytes, status, chunk_size, chunk_overlap, created_at)
                VALUES (:id, :filename, :stored_path, :media_type, :size_bytes, :status, :chunk_size, :chunk_overlap, :created_at)""",
                values,
            )

    def update_document(self, document_id: str, **values: Any) -> None:
        if not values:
            return
        columns = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE documents SET {columns} WHERE id = ?",
                (*values.values(), document_id),
            )

    def replace_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """INSERT INTO chunks
                (id, document_id, chunk_index, page, text, token_count, vector)
                VALUES (:id, :document_id, :chunk_index, :page, :text, :token_count, :vector)""",
                chunks,
            )

    def delete_document(self, document_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM jobs WHERE entity_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def latest_job_for_entity(self, entity_id: str, kind: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM jobs WHERE entity_id = ?"
        params: list[Any] = [entity_id]
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None
