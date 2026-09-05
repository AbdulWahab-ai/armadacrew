"""API e2e: health, create/list/get, HITL resume, event history."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from armadacrew.api import create_app
from armadacrew.bootstrap import build_context


def _wait(client: TestClient, run_id: str, timeout: float = 20.0, terminal: set[str] | None = None) -> dict:
    wanted = terminal or {"interrupted", "completed", "failed", "rejected"}
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        res = client.get(f"/v1/runs/{run_id}")
        assert res.status_code == 200
        last = res.json()["run"]
        if last["status"] in wanted:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {run_id} in {wanted}: {last}")


def test_health_and_index(ctx):
    with TestClient(create_app(ctx.settings, ctx)) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ok"] is True
        assert body["documents"] >= 10
        page = client.get("/")
        assert page.status_code == 200
        assert b"ArmadaCrew" in page.content
        meta = client.get("/v1/meta")
        assert len(meta.json()["samples"]) == 3
        assert len(meta.json()["tools"]) == 7


def test_postmortem_e2e(settings):
    context = build_context(settings, reset_domain=True)
    try:
        with TestClient(create_app(settings, context)) as client:
            created = client.post(
                "/v1/runs",
                json={"task": "Draft a postmortem for yesterday's auth outage", "mode": "incident"},
            )
            assert created.status_code == 202
            run_id = created.json()["run_id"]
            run = _wait(client, run_id)
            assert run["status"] == "completed"
            listed = client.get("/v1/runs")
            assert any(item["run_id"] == run_id for item in listed.json()["items"])
            hist = client.get(f"/v1/runs/{run_id}/events/history")
            kinds = {e["kind"] for e in hist.json()["items"]}
            assert "run_start" in kinds and "node_start" in kinds
            assert "tool_call" in kinds
    finally:
        context.runs.shutdown()
        context.checkpointer.close()


def test_refund_interrupt_resume_e2e(settings):
    context = build_context(settings, reset_domain=True)
    try:
        with TestClient(create_app(settings, context)) as client:
            created = client.post(
                "/v1/runs",
                json={
                    "task": "Customer Ada Lovelace wants a $720 refund for duplicate charge",
                    "mode": "customer_ops",
                },
            )
            run_id = created.json()["run_id"]
            run = _wait(client, run_id)
            assert run["status"] == "interrupted"
            assert run["pending_action"]["action_type"] == "refund"
            resumed = client.post(
                f"/v1/runs/{run_id}/resume",
                json={"approved": True, "comment": "ok from api test", "actor": "pytest"},
            )
            assert resumed.status_code == 200
            final = _wait(client, run_id, terminal={"completed", "failed", "rejected"})
            assert final["status"] == "completed"
            assert final["human_decision"]["approved"] is True
    finally:
        context.runs.shutdown()
        context.checkpointer.close()


def test_unknown_run_404(ctx):
    with TestClient(create_app(ctx.settings, ctx)) as client:
        assert client.get("/v1/runs/does-not-exist").status_code == 404
        assert client.post("/v1/runs/does-not-exist/resume", json={"approved": True}).status_code == 404


def test_resume_wrong_status_409(settings):
    context = build_context(settings, reset_domain=True)
    try:
        with TestClient(create_app(settings, context)) as client:
            created = client.post(
                "/v1/runs",
                json={"task": "Draft a postmortem for yesterday's auth outage", "mode": "incident"},
            )
            run_id = created.json()["run_id"]
            _wait(client, run_id)
            conflict = client.post(f"/v1/runs/{run_id}/resume", json={"approved": True})
            assert conflict.status_code == 409
    finally:
        context.runs.shutdown()
        context.checkpointer.close()


def test_validation_on_create(ctx):
    with TestClient(create_app(ctx.settings, ctx)) as client:
        res = client.post("/v1/runs", json={"task": "short", "mode": "auto"})
        assert res.status_code == 422
