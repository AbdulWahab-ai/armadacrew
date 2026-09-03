"""FastAPI surface: runs, resume, SSE events, health, and the mission-control UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from armadacrew import __version__
from armadacrew.bootstrap import AppContext, build_context
from armadacrew.config import Settings, get_settings
from armadacrew.runtime import GraphState
from armadacrew.tools import tool_catalog
from armadacrew.tracing import sse_poll


class RunCreate(BaseModel):
    task: str = Field(min_length=8, max_length=2000)
    mode: Literal["incident", "customer_ops", "auto"] = "auto"


class ResumeBody(BaseModel):
    approved: bool
    comment: str = ""
    actor: str = "operator"


SAMPLE_TASKS = [
    {
        "id": "latency",
        "title": "Payments p99 spike",
        "mode": "incident",
        "task": "API p99 latency spiked on payments-api in prod",
    },
    {
        "id": "refund",
        "title": "Ada Lovelace refund",
        "mode": "customer_ops",
        "task": "Customer Ada Lovelace wants a $720 refund for duplicate charge",
    },
    {
        "id": "postmortem",
        "title": "Auth outage postmortem",
        "mode": "incident",
        "task": "Draft a postmortem for yesterday's auth outage",
    },
]


def _state_payload(state: GraphState) -> dict[str, Any]:
    return state.model_dump()


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        context = ctx or build_context(cfg)
        app.state.ctx = context
        yield
        context.runs.shutdown()
        context.checkpointer.close()

    app = FastAPI(
        title="ArmadaCrew",
        version=__version__,
        description="Production multi-agent orchestration for operations copilots.",
        lifespan=lifespan,
    )

    static_dir = Path(cfg.static_dir)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def _ctx() -> AppContext:
        return app.state.ctx

    @app.get("/")
    async def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(404, "UI bundle missing")
        return FileResponse(index_path)

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        context = _ctx()
        return {
            "ok": True,
            "version": __version__,
            "llm": context.llm.name,
            "db": str(context.settings.db_path),
            "documents": len(context.store.documents),
            "services": len(context.store.services),
        }

    @app.get("/v1/meta")
    async def meta() -> dict[str, Any]:
        return {
            "samples": SAMPLE_TASKS,
            "tools": tool_catalog(),
            "nodes": ["supervisor", "researcher", "sre", "support", "critic", "writer", "hitl"],
        }

    @app.post("/v1/runs", status_code=202)
    async def create_run(body: RunCreate) -> dict[str, Any]:
        state = _ctx().runs.start(task=body.task, mode=body.mode)
        return {"run_id": state.run_id, "status": state.status, "task": state.task, "mode": body.mode}

    @app.get("/v1/runs")
    async def list_runs(limit: int = Query(default=40, ge=1, le=200)) -> dict[str, Any]:
        items = _ctx().checkpointer.list_runs(limit=limit)
        return {"items": items, "count": len(items)}

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        state = _ctx().runs.get(run_id)
        if state is None:
            raise HTTPException(404, f"Unknown run {run_id}")
        checkpoints = _ctx().checkpointer.list_checkpoints(run_id)
        return {"run": _state_payload(state), "checkpoints": checkpoints}

    @app.post("/v1/runs/{run_id}/resume")
    async def resume_run(run_id: str, body: ResumeBody) -> dict[str, Any]:
        context = _ctx()
        if context.runs.get(run_id) is None:
            raise HTTPException(404, f"Unknown run {run_id}")
        try:
            state = context.runs.resume(run_id, approved=body.approved, comment=body.comment, actor=body.actor)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run_id, "status": state.status}

    @app.get("/v1/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
        context = _ctx()
        if context.runs.get(run_id) is None:
            raise HTTPException(404, f"Unknown run {run_id}")

        def terminal(rid: str) -> bool:
            state = context.runs.get(rid)
            if state is None:
                return True
            if context.runs.is_running(rid):
                return False
            return state.status in {"completed", "failed", "rejected", "interrupted"}

        async def gen():
            async for chunk in sse_poll(
                context.tracer,
                run_id,
                terminal,
                poll_seconds=context.settings.sse_poll_seconds,
                after_id=after,
            ):
                if await request.is_disconnected():
                    break
                yield chunk

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/runs/{run_id}/events/history")
    async def event_history(run_id: str) -> dict[str, Any]:
        context = _ctx()
        if context.runs.get(run_id) is None:
            raise HTTPException(404, f"Unknown run {run_id}")
        return {"items": context.tracer.history(run_id)}

    @app.exception_handler(HTTPException)
    async def http_exc(_, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail, "status": exc.status_code}, status_code=exc.status_code)

    return app


app = create_app()
