"""Supervisor routing, critic reject loop, and specialist nodes."""

from __future__ import annotations

from armadacrew.agents import critic_node, researcher_node
from armadacrew.runtime import GraphState, Message, PlanStep, apply_update, initial_state
from armadacrew.supervisor import decide_route


def _state(task: str, **kwargs) -> GraphState:
    state = initial_state(task, mode=kwargs.pop("mode", "auto"))
    if kwargs:
        state = apply_update(state, kwargs)
    return state


def test_supervisor_routes_incident_to_researcher():
    state = _state("API p99 latency spiked on payments-api in prod", mode="incident")
    assert decide_route(state) == "researcher"


def test_supervisor_routes_customer_to_support():
    state = _state("Customer Ada Lovelace wants a $720 refund for duplicate charge", mode="customer_ops")
    assert decide_route(state) == "support"


def test_supervisor_advances_after_research():
    state = _state(
        "API p99 latency spiked on payments-api in prod",
        mode="incident",
        metadata={"research_done": True, "classified_mode": "incident"},
        messages=[Message(role="assistant", content="research done", agent="researcher").model_dump()],
        route_history=["supervisor", "researcher"],
    )
    assert decide_route(state) == "sre"


def test_supervisor_sends_approved_plan_to_writer():
    state = _state(
        "Draft a postmortem for yesterday's auth outage",
        mode="incident",
        critic_verdict="approve",
        metadata={"research_done": True, "sre_done": True, "critic_rounds": 1, "classified_mode": "incident"},
        messages=[
            Message(content="r", agent="researcher").model_dump(),
            Message(content="s", agent="sre").model_dump(),
            Message(content="c", agent="critic").model_dump(),
        ],
        route_history=["researcher", "sre", "critic"],
    )
    assert decide_route(state) == "writer"


def test_researcher_uses_knowledge_tool(ctx):
    state = initial_state("API p99 latency spiked on payments-api in prod", mode="incident")
    result = researcher_node(state, ctx.tools, ctx.llm)
    assert result.update["metadata"]["research_done"] is True
    assert any(tc.get("tool") == "knowledge_search" or (tc.get("data") or {}).get("hits") is not None for tc in result.update["tool_calls"])
    assert result.update["artifacts"]


def test_critic_reject_then_approve_loop(ctx):
    state = initial_state("API p99 latency spiked on payments-api in prod", mode="incident")
    state = apply_update(
        state,
        {
            "metadata": {"classified_mode": "incident", "sre_done": True, "research_done": True},
            "plan_steps": [
                PlanStep(id="observe", title="Look at p99", owner="sre", risk="low").model_dump(),
                PlanStep(id="restart", title="Rolling restart payments-api in prod", owner="sre", risk="high").model_dump(),
            ],
            "tool_calls": [
                {"ok": True, "tool": "metrics_query", "data": {"slo_breached": True}},
                {"ok": True, "tool": "logs_search", "data": {"error_count": 8}},
                {"ok": True, "tool": "knowledge_search", "data": {"hits": [{"title": "latency"}]}},
            ],
            "messages": [
                Message(content="research", agent="researcher").model_dump(),
                Message(content="sre plan", agent="sre").model_dump(),
            ],
        },
    )
    first = critic_node(state, ctx.tools, ctx.llm)
    assert first.update["critic_verdict"] == "reject"
    assert first.update["next_agent"] == "sre"
    state = apply_update(state, first.update)
    second = critic_node(state, ctx.tools, ctx.llm)
    assert second.update["critic_verdict"] == "approve"
    assert second.update["next_agent"] == "writer"
    assert second.update["critic_score"] >= ctx.settings.critic_pass_score


def test_full_postmortem_completes(ctx):
    state = ctx.graph.invoke(initial_state("Draft a postmortem for yesterday's auth outage", mode="incident"))
    assert state.status == "completed"
    kinds = {a.kind for a in state.artifacts}
    assert "incident_report" in kinds or any("report" in a.name for a in state.artifacts)
    assert "critic" in state.route_history
    assert state.route_history.count("critic") >= 2  # reject loop then approve
    assert "writer" in state.route_history


def test_latency_run_interrupts_for_prod_restart(ctx):
    state = ctx.graph.invoke(initial_state("API p99 latency spiked on payments-api in prod", mode="incident"))
    assert state.status == "interrupted"
    assert state.pending_action is not None
    assert state.pending_action.action_type == "prod_restart"
    resumed = ctx.graph.resume(state.run_id, approved=True, comment="restart during the window")
    assert resumed.status == "completed"
    assert resumed.human_decision and resumed.human_decision.approved
    assert any(a.kind == "incident_report" for a in resumed.artifacts)


def test_refund_run_interrupts_over_threshold(ctx):
    state = ctx.graph.invoke(
        initial_state("Customer Ada Lovelace wants a $720 refund for duplicate charge", mode="customer_ops")
    )
    assert state.status == "interrupted"
    assert state.pending_action is not None
    assert state.pending_action.action_type == "refund"
    assert float(state.pending_action.payload.get("amount_usd") or 0) >= 720
    resumed = ctx.graph.resume(state.run_id, approved=True, comment="finance signed off")
    assert resumed.status == "completed"
    assert any(a.kind in {"customer_email", "email_draft"} for a in resumed.artifacts)


def test_hitl_reject_stops_run(ctx):
    state = ctx.graph.invoke(
        initial_state("Customer Ada Lovelace wants a $720 refund for duplicate charge", mode="customer_ops")
    )
    resumed = ctx.graph.resume(state.run_id, approved=False, comment="not a duplicate")
    assert resumed.status == "rejected"
