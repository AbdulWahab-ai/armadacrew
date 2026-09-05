"""Tool sandbox: schemas, validation, retries, and real domain lookups."""

from __future__ import annotations

import pytest

from armadacrew.tools import ApprovalRequired, ToolError, tool_catalog


def test_catalog_covers_required_tools():
    names = {item["name"] for item in tool_catalog()}
    assert names == {
        "knowledge_search",
        "metrics_query",
        "logs_search",
        "crm_lookup",
        "create_ticket",
        "send_email",
        "pager_ack",
    }
    for item in tool_catalog():
        assert "parameters" in item
        assert item["timeout_seconds"] > 0
        assert "properties" in item["parameters"] or "$defs" in item["parameters"] or item["parameters"]


def test_knowledge_search_finds_runbooks(ctx):
    result = ctx.tools.invoke("knowledge_search", {"query": "payments-api latency p99", "kind": "runbook"})
    assert result["ok"]
    titles = " ".join(hit["title"].lower() for hit in result["data"]["hits"])
    assert "latency" in titles or "payments" in titles
    assert result["data"]["count"] >= 1


def test_knowledge_search_finds_refund_policy(ctx):
    result = ctx.tools.invoke("knowledge_search", {"query": "refund duplicate charge policy", "kind": "kb"})
    assert result["ok"]
    blob = " ".join(hit["excerpt"].lower() for hit in result["data"]["hits"])
    assert "refund" in blob or "duplicate" in blob


def test_metrics_query_spikes_payments(ctx):
    result = ctx.tools.invoke("metrics_query", {"service": "payments-api", "metric": "p99_ms", "window_minutes": 60})
    data = result["data"]
    assert data["latest"] > data["baseline"]
    assert data["slo_breached"] is True
    assert len(data["points"]) >= 10


def test_logs_search_returns_errors(ctx):
    result = ctx.tools.invoke("logs_search", {"service": "payments-api", "query": "error"})
    assert result["data"]["error_count"] >= 1
    assert result["data"]["hits"][0]["service"] == "payments-api"


def test_crm_lookup_ada(ctx):
    result = ctx.tools.invoke("crm_lookup", {"query": "Ada Lovelace"})
    names = [m["name"] for m in result["data"]["matches"]]
    assert "Ada Lovelace" in names
    ada = result["data"]["matches"][0]
    assert ada["duplicate_charge_usd"] == 720


def test_create_ticket_and_pager(ctx):
    ticket = ctx.tools.invoke("create_ticket", {"title": "SEV2 payments", "body": "p99 spike", "severity": "sev2"})
    assert ticket["data"]["id"].startswith("TCK-")
    ack = ctx.tools.invoke("pager_ack", {"incident_key": "inc-pay-latency"})
    assert ack["data"]["ok"] is True
    assert ack["data"]["event"]["acked"] is True


def test_send_email_requires_approval(ctx):
    with pytest.raises(ApprovalRequired):
        ctx.tools.invoke(
            "send_email",
            {"to": "ada@analytical.engine", "subject": "Refund", "body": "We will refund $720."},
        )
    sent = ctx.tools.invoke(
        "send_email",
        {"to": "ada@analytical.engine", "subject": "Refund", "body": "We will refund $720.", "approved": True},
        approved=True,
    )
    assert sent["data"]["status"] == "sent"


def test_invalid_args_are_rejected(ctx):
    with pytest.raises(ToolError, match="Invalid arguments"):
        ctx.tools.invoke("metrics_query", {"service": "x", "window_minutes": 1})
    with pytest.raises(ToolError, match="Unknown tool"):
        ctx.tools.invoke("drop_database", {"confirm": True})


def test_audit_log_records_calls(ctx):
    ctx.tools.invoke("crm_lookup", {"query": "Hopper"})
    assert ctx.tools.audit[-1]["tool"] == "crm_lookup"
    assert ctx.tools.audit[-1]["ok"] is True


def test_schema_validation_email(ctx):
    with pytest.raises(ToolError):
        ctx.tools.invoke("send_email", {"to": "not-an-email", "subject": "Hi", "body": "Hello there"})
