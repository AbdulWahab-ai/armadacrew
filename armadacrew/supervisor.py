"""Supervisor routing + compiled crew graph.

The supervisor is a node, not a hidden if-statement in the API. It inspects graph
state (phase, critic verdict, HITL) and emits `next_agent`, which a conditional
edge maps onto specialist nodes — the same pattern LangGraph uses in production.
"""

from __future__ import annotations

from typing import Any

from armadacrew.agents import detect_mode, make_agent_nodes
from armadacrew.checkpointer import SqliteCheckpointer
from armadacrew.config import Settings, get_settings
from armadacrew.domain import classify_task
from armadacrew.llm import LLMMessage, LLMProvider, get_llm
from armadacrew.runtime import END, GraphState, Message, NodeResult, StateGraph
from armadacrew.tools import ToolRegistry
from armadacrew.tracing import Tracer


SPECIALISTS = ("researcher", "sre", "support", "critic", "writer", "hitl")


def decide_route(state: GraphState) -> str:
    """Pure router used by the supervisor node and by tests."""

    if state.status in {"completed", "failed", "rejected"}:
        return "end"
    if state.human_decision and not state.human_decision.approved:
        return "end"
    if state.approval_required and state.human_decision is None:
        return "hitl"
    if state.current_node == "hitl" and state.human_decision and state.human_decision.approved:
        if state.metadata.get("writer_done"):
            return "end"
        if state.critic_verdict == "approve":
            return "writer"
        return "critic"

    mode = state.resolved_mode
    agents_seen = set(state.route_history)
    # route_history includes the supervisor itself; also look at message authors.
    authors = {m.agent for m in state.messages if m.agent}
    agents_seen |= authors

    has_research = "researcher" in agents_seen or bool(state.metadata.get("research_done"))
    has_sre = "sre" in agents_seen or bool(state.metadata.get("sre_done"))
    has_support = "support" in agents_seen or bool(state.metadata.get("support_done"))
    has_specialist = has_sre if mode == "incident" else has_support
    has_writer = "writer" in agents_seen or bool(state.metadata.get("writer_done"))
    critic_rounds = int(state.metadata.get("critic_rounds") or 0)
    max_rounds = 2

    if state.next_agent in SPECIALISTS:
        # Honor an explicit specialist request except when it would loop forever.
        requested = state.next_agent
        if requested == "hitl" and state.human_decision is None:
            return "hitl"
        if requested == "writer" and (state.critic_verdict == "approve" or critic_rounds >= max_rounds):
            return "writer"
        if requested in {"researcher", "sre", "support", "critic"}:
            # After HITL approval, skip re-entering the same specialist if already done.
            if requested == "sre" and has_sre and state.human_decision:
                return "critic" if state.critic_verdict != "approve" else "writer"
            if requested == "support" and has_support and state.human_decision:
                return "critic" if state.critic_verdict != "approve" else "writer"
            return requested

    if mode == "customer_ops":
        if not has_support:
            return "support"
        if state.critic_verdict == "reject" and critic_rounds < max_rounds:
            return "support"
        if state.critic_verdict != "approve" and critic_rounds < max_rounds:
            return "critic"
        if not has_writer:
            return "writer"
        return "end"

    # incident (and postmortem)
    if not has_research:
        return "researcher"
    if not has_sre:
        return "sre"
    if state.critic_verdict == "reject" and critic_rounds < max_rounds:
        return "sre"
    if state.critic_verdict != "approve" and critic_rounds < max_rounds:
        return "critic"
    if not has_writer:
        return "writer"
    return "end"


def supervisor_node(state: GraphState, llm: LLMProvider | None = None) -> NodeResult:
    provider = llm or get_llm()
    classified = detect_mode(state.task, state.mode)
    # Seed resolved mode into metadata so later nodes agree.
    thought = provider.complete(
        [
            LLMMessage(role="system", content="You are the ArmadaCrew supervisor."),
            LLMMessage(
                role="user",
                content=(
                    f"TASK: {state.task}\nmode={classified}\n"
                    f"route_history={state.route_history}\n"
                    f"critic_verdict={state.critic_verdict}\n"
                    f"approval_required={state.approval_required}\n"
                    f"TRANSCRIPT:\n{state.transcript()}"
                ),
            ),
        ],
        role="supervisor",
    )
    nxt = decide_route(
        state.model_copy(
            update={"metadata": {**state.metadata, "classified_mode": classified}}
        )
    )
    llm_next = thought.parsed.get("next")
    if llm_next in SPECIALISTS + ("end",) and nxt in {llm_next, "critic", "writer", "hitl", "end"}:
        # Prefer the deterministic router; LLM thought is explanatory.
        pass
    summary = thought.parsed.get("thought") or f"Routing to {nxt}"
    return NodeResult(
        update={
            "messages": [Message(role="assistant", content=f"Supervisor: {summary} → {nxt}", agent="supervisor")],
            "next_agent": None if nxt == "end" else nxt,
            "metadata": {"classified_mode": classified, "supervisor_next": nxt},
        }
    )


def route_from_supervisor(state: GraphState) -> str:
    nxt = state.metadata.get("supervisor_next") or state.next_agent or decide_route(state)
    if nxt in SPECIALISTS or nxt == "end":
        return str(nxt)
    return decide_route(state)


def route_from_critic(state: GraphState) -> str:
    if state.critic_verdict == "approve":
        return "writer"
    if state.resolved_mode == "customer_ops":
        return "support"
    return "sre"


def route_from_writer(state: GraphState) -> str:
    if state.approval_required:
        return "hitl"
    return "end"


def route_from_hitl(state: GraphState) -> str:
    if state.status == "rejected":
        return "end"
    if state.approval_required and state.human_decision is None:
        return "end"  # interrupt already handled; safety
    if state.metadata.get("writer_done"):
        return "end"
    if state.critic_verdict == "approve":
        return "writer"
    return "supervisor"


def route_back_to_supervisor(_state: GraphState) -> str:
    return "supervisor"


def build_graph(
    tools: ToolRegistry,
    llm: LLMProvider | None = None,
    checkpointer: SqliteCheckpointer | None = None,
    tracer: Tracer | None = None,
    settings: Settings | None = None,
) -> Any:
    cfg = settings or tools.settings or get_settings()
    provider = llm or get_llm(cfg)
    cp = checkpointer or SqliteCheckpointer(cfg.db_path)
    tr = tracer or Tracer(cp)
    tools.bind(tr, None, None)
    nodes = make_agent_nodes(tools, provider)

    def _supervisor(state: GraphState) -> NodeResult:
        return supervisor_node(state, provider)

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", _supervisor)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "sre": "sre",
            "support": "support",
            "critic": "critic",
            "writer": "writer",
            "hitl": "hitl",
            "end": END,
        },
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_conditional_edges(
        "sre",
        lambda s: "hitl" if s.approval_required else "supervisor",
        {"hitl": "hitl", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "support",
        lambda s: "hitl" if s.approval_required else "supervisor",
        {"hitl": "hitl", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {"writer": "writer", "sre": "sre", "support": "support", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "writer",
        route_from_writer,
        {"hitl": "hitl", "end": END},
    )
    graph.add_conditional_edges(
        "hitl",
        route_from_hitl,
        {"supervisor": "supervisor", "writer": "writer", "end": END},
    )
    return graph.compile(checkpointer=cp, tracer=tr, max_steps=cfg.max_graph_steps)


def classify(task: str) -> str:
    return classify_task(task)
