"""SQLite checkpointer — durable graph state after every node (LangGraph analog).

A run can crash, wait on a human, or restart the process and resume from the last
successful node because each step is serialized here together with an event log.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from armadacrew.runtime import GraphState


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    current_node TEXT,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    node TEXT NOT NULL,
    status TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, step)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    node TEXT,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id, step);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteCheckpointer:
    """Thread-safe SQLite persistence for run state, checkpoints, and traces."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def upsert_run(self, state: GraphState) -> None:
        payload = state.model_dump_json()
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs (run_id, task, mode, status, created_at, updated_at, current_node, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    current_node=excluded.current_node,
                    state_json=excluded.state_json
                """,
                (
                    state.run_id,
                    state.task,
                    state.mode,
                    state.status,
                    now,
                    now,
                    state.current_node,
                    payload,
                ),
            )
            self._conn.commit()

    def save_checkpoint(self, state: GraphState, node: str, step: int) -> int:
        payload = state.model_dump_json()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO checkpoints (run_id, step, node, status, state_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step) DO UPDATE SET
                    node=excluded.node,
                    status=excluded.status,
                    state_json=excluded.state_json,
                    created_at=excluded.created_at
                """,
                (state.run_id, step, node, state.status, payload, _now()),
            )
            self.upsert_run(state)
            self._conn.commit()
            return int(cur.lastrowid or step)

    def load_run(self, run_id: str) -> GraphState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return GraphState.model_validate_json(row["state_json"])

    def latest_checkpoint(self, run_id: str) -> tuple[int, GraphState] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT step, state_json FROM checkpoints
                WHERE run_id = ? ORDER BY step DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return int(row["step"]), GraphState.model_validate_json(row["state_json"])

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT step, node, status, created_at FROM checkpoints
                WHERE run_id = ? ORDER BY step ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id, task, mode, status, created_at, updated_at, current_node
                FROM runs ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def append_event(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        node: str | None = None,
    ) -> dict[str, Any]:
        blob = json.dumps(payload, default=str)
        ts = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (run_id, ts, kind, node, payload_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, ts, kind, node, blob),
            )
            self._conn.commit()
            event_id = int(cur.lastrowid)
        return {
            "id": event_id,
            "run_id": run_id,
            "ts": ts,
            "kind": kind,
            "node": node,
            "payload": payload,
        }

    def events_since(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, run_id, ts, kind, node, payload_json
                FROM events WHERE run_id = ? AND id > ? ORDER BY id ASC
                """,
                (run_id, after_id),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "ts": row["ts"],
                    "kind": row["kind"],
                    "node": row["node"],
                    "payload": json.loads(row["payload_json"]),
                }
            )
        return out

    def all_events(self, run_id: str) -> list[dict[str, Any]]:
        return self.events_since(run_id, after_id=0)
