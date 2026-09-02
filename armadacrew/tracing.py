"""Event-sourced traces for the run inspector (SSE + REST)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from armadacrew.checkpointer import SqliteCheckpointer


class Tracer:
    """Thin facade over the checkpointer event log."""

    def __init__(self, checkpointer: SqliteCheckpointer) -> None:
        self.checkpointer = checkpointer

    def emit(self, run_id: str, kind: str, payload: dict[str, Any] | None = None, node: str | None = None) -> dict[str, Any]:
        return self.checkpointer.append_event(run_id, kind, payload or {}, node=node)

    def history(self, run_id: str) -> list[dict[str, Any]]:
        return self.checkpointer.all_events(run_id)

    def since(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return self.checkpointer.events_since(run_id, after_id)

    def node_started(self, run_id: str, node: str, step: int) -> dict[str, Any]:
        return self.emit(run_id, "node_start", {"step": step}, node=node)

    def node_finished(self, run_id: str, node: str, step: int, status: str) -> dict[str, Any]:
        return self.emit(run_id, "node_end", {"step": step, "status": status}, node=node)

    def tool_called(
        self,
        run_id: str,
        tool: str,
        args: dict[str, Any],
        ok: bool,
        duration_ms: float,
        retries: int,
        error: str | None = None,
        node: str | None = None,
    ) -> dict[str, Any]:
        return self.emit(
            run_id,
            "tool_call",
            {
                "tool": tool,
                "args": args,
                "ok": ok,
                "duration_ms": round(duration_ms, 2),
                "retries": retries,
                "error": error,
            },
            node=node,
        )

    def interrupted(self, run_id: str, reason: str, pending: dict[str, Any] | None) -> dict[str, Any]:
        return self.emit(run_id, "interrupt", {"reason": reason, "pending": pending}, node="hitl")

    def resumed(self, run_id: str, approved: bool, comment: str) -> dict[str, Any]:
        return self.emit(run_id, "resume", {"approved": approved, "comment": comment}, node="hitl")

    def failed(self, run_id: str, error: str, node: str | None) -> dict[str, Any]:
        return self.emit(run_id, "error", {"error": error}, node=node)


def format_sse(event: dict[str, Any]) -> str:
    return f"id: {event.get('id', 0)}\nevent: {event.get('kind', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"


def iter_sse_once(tracer: Tracer, run_id: str, after_id: int = 0) -> Iterator[str]:
    for event in tracer.since(run_id, after_id):
        yield format_sse(event)


async def sse_poll(
    tracer: Tracer,
    run_id: str,
    is_terminal,
    poll_seconds: float = 0.25,
    after_id: int = 0,
) -> AsyncIterator[str]:
    """Yield SSE frames until the run reaches a terminal status."""

    import asyncio

    last = after_id
    idle_rounds = 0
    while True:
        events = tracer.since(run_id, last)
        if events:
            idle_rounds = 0
            for event in events:
                last = int(event["id"])
                yield format_sse(event)
        else:
            idle_rounds += 1
            yield ": keepalive\n\n"
        if is_terminal(run_id) and not tracer.since(run_id, last):
            yield format_sse({"id": last, "kind": "stream_end", "payload": {"run_id": run_id}, "node": None})
            break
        if idle_rounds > 800:
            break
        await asyncio.sleep(poll_seconds)
