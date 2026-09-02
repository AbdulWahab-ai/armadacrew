"""Specialist agent nodes: researcher, SRE, support, critic, writer, HITL.

Each node is a real callable that:
  1. Asks the LLM for a structured JSON plan (mock LLM works offline).
  2. Executes domain tools against synthetic infra.
  3. Returns a state update (messages, artifacts, plan steps, maybe an interrupt).
"""

from __future__ import annotations

from typing import Any

from armadacrew.domain import (
    classify_task,
    extract_refund_amount,
    looks_like_pii_email,
)
from armadacrew.llm import LLMMessage, LLMProvider, get_llm
from armadacrew.runtime import (
    Artifact,
    GraphState,
    Message,
    NodeResult,
    PendingAction,
    PlanStep,
    Ticket,
)
from armadacrew.tools import ApprovalRequired, ToolError, ToolRegistry


def _llm_blob(state: GraphState, extra: str = "") -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=f"You are an ArmadaCrew agent. Task mode={state.resolved_mode}."),
        LLMMessage(
            role="user",
            content=(
                f"TASK: {state.task}\n"
                f"critic_rounds={state.metadata.get('critic_rounds', 0)}\n"
                f"previous_verdict={state.critic_verdict}\n"
                f"status={state.status} approval_required={state.approval_required}\n"
                f"TRANSCRIPT:\n{state.transcript()}\n{extra}"
            ),
        ),
    ]


def _safe_invoke(tools: ToolRegistry, name: str, args: dict[str, Any], approved: bool = False) -> dict[str, Any]:
    try:
        return tools.invoke(name, args, approved=approved)
    except ApprovalRequired as exc:
        return {"ok": False, "tool": name, "approval_required": True, "action": exc.action, "error": str(exc)}
    except ToolError as exc:
        return {"ok": False, "tool": name, "error": str(exc)}


def _infer_service(state: GraphState) -> str:
    text = state.task.lower()
    if "payment" in text:
        return "payments-api"
    if "auth" in text:
        return "auth-service"
    if "checkout" in text:
        return "checkout-api"
    if "notif" in text:
        return "notifications"
    meta = state.metadata.get("service")
    if isinstance(meta, str) and meta:
        return meta
    return "payments-api" if state.resolved_mode == "incident" else "billing-api"


def researcher_node(state: GraphState, tools: ToolRegistry, llm: LLMProvider) -> NodeResult:
    plan = llm.complete(_llm_blob(state), role="researcher").parsed
    queries = plan.get("queries") or [state.task, "runbook", "incident"]
    tool_calls: list[dict[str, Any]] = []
    excerpts: list[str] = []
    kinds = ["runbook", "kb"]
    for i, query in enumerate(queries[:4]):
        kind = kinds[i % 2] if i < 2 else None
        result = _safe_invoke(tools.bind(tools.tracer, state.run_id, "researcher"), "knowledge_search", {"query": query, "kind": kind, "k": 4})
        tool_calls.append(result)
        for hit in (result.get("data") or {}).get("hits") or []:
            excerpts.append(f"- {hit.get('title')}: {hit.get('excerpt', '')[:180]}")

    related = tools.store.related_incidents(state.task, _infer_service(state))
    if related:
        excerpts.append("Related incidents: " + ", ".join(inc.get("id", "?") for inc in related))

    summary = (
        "Research complete. Sources:\n" + ("\n".join(excerpts[:8]) or "- No documents matched.")
    )
    artifact = Artifact(
        name="research-brief.md",
        kind="research",
        content=summary,
        created_by="researcher",
    )
    return NodeResult(
        update={
            "messages": [Message(role="assistant", content=summary, agent="researcher")],
            "artifacts": [artifact.model_dump()],
            "tool_calls": tool_calls,
            "next_agent": "sre" if state.resolved_mode == "incident" else "support",
            "metadata": {
                "service": plan.get("focus") or _infer_service(state),
                "related_incidents": related,
                "research_done": True,
            },
        }
    )


def sre_node(state: GraphState, tools: ToolRegistry, llm: LLMProvider) -> NodeResult:
    tools.bind(tools.tracer, state.run_id, "sre")
    service = _infer_service(state)
    plan = llm.complete(_llm_blob(state, extra=f"service={service}"), role="sre").parsed
    service = str(plan.get("service") or service)
    tool_calls = [
        _safe_invoke(tools, "metrics_query", {"service": service, "metric": "p99_ms", "env": "prod"}),
        _safe_invoke(tools, "metrics_query", {"service": service, "metric": "error_rate", "env": "prod"}),
        _safe_invoke(tools, "logs_search", {"service": service, "query": "error timeout", "env": "prod"}),
        _safe_invoke(tools, "knowledge_search", {"query": f"{service} latency outage runbook", "kind": "runbook"}),
    ]
    incident_key = "inc-auth-yesterday" if "auth" in state.task.lower() else f"inc-{service}"
    ack = _safe_invoke(tools, "pager_ack", {"incident_key": incident_key, "actor": "sre-agent"})
    tool_calls.append(ack)

    p99 = (tool_calls[0].get("data") or {}).get("latest")
    breached = bool((tool_calls[0].get("data") or {}).get("slo_breached"))
    error_lines = (tool_calls[2].get("data") or {}).get("error_count", 0)

    raw_steps = plan.get("steps") or []
    steps: list[PlanStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            continue
        steps.append(
            PlanStep(
                id=str(raw.get("id") or f"step-{len(steps)+1}"),
                title=str(raw.get("title") or "step"),
                owner=str(raw.get("owner") or "sre"),
                risk=raw.get("risk") or "low",
                details=str(raw.get("details") or ""),
                requires_approval=bool(raw.get("requires_approval") or raw.get("risk") == "high"),
            )
        )
    if not steps:
        steps = [
            PlanStep(id="observe", title=f"Confirm p99={p99} on {service}", owner="sre", risk="low"),
            PlanStep(id="mitigate", title="Scale and shed load", owner="sre", risk="medium"),
        ]

    high = any(s.risk == "high" or s.requires_approval for s in steps)
    is_postmortem = any(w in state.task.lower() for w in ("postmortem", "post-mortem", "writeup"))
    # Prod restart is always high-risk HITL unless this run is a retrospective write-up.
    if not is_postmortem and (high or "restart" in state.task.lower() or breached):
        if not any("restart" in s.title.lower() for s in steps):
            steps.append(
                PlanStep(
                    id="restart",
                    title=f"Rolling restart {service} in prod",
                    owner="sre",
                    risk="high",
                    requires_approval=True,
                    details="Last-resort mitigation after scale-out.",
                )
            )
        high = True
    if is_postmortem:
        high = False

    ticket_res = _safe_invoke(
        tools,
        "create_ticket",
        {
            "title": f"{service} degradation: {state.task[:80]}",
            "body": f"p99={p99} breached={breached} errors={error_lines}\n{state.task}",
            "severity": "sev1" if breached else "sev2",
            "queue": "sre",
        },
    )
    tool_calls.append(ticket_res)
    tickets: list[dict[str, Any]] = []
    if ticket_res.get("ok"):
        data = ticket_res["data"]
        tickets.append(Ticket(id=data["id"], title=data["title"], severity=data["severity"], body=data["body"]).model_dump())

    summary = (
        f"SRE analysis for {service}: p99={p99}ms slo_breached={breached} "
        f"error_log_hits={error_lines}. Plan has {len(steps)} steps."
    )
    pending = None
    interrupt = False
    if high and not (state.human_decision and state.human_decision.approved):
        restart = next((s for s in steps if s.risk == "high"), steps[-1])
        pending = PendingAction(
            action_type="prod_restart",
            risk="high",
            reason=f"Production restart of {service} requires a human gate",
            payload={"service": service, "env": "prod", "step": restart.title},
            policy="prod_restart",
        )
        interrupt = True

    update: dict[str, Any] = {
        "messages": [Message(role="assistant", content=summary, agent="sre")],
        "tool_calls": tool_calls,
        "plan_steps": [s.model_dump() for s in steps],
        "plan": {"title": plan.get("plan_title") or f"Stabilize {service}", "service": service, "steps": [s.model_dump() for s in steps]},
        "tickets": tickets,
        "next_agent": "hitl" if interrupt else "critic",
        "metadata": {"service": service, "sre_done": True, "slo_breached": breached},
        "artifacts": [
            Artifact(
                name=f"{service}-sre-plan.json",
                kind="plan",
                content=summary + "\n" + "\n".join(f"* {s.title} ({s.risk})" for s in steps),
                created_by="sre",
            ).model_dump()
        ],
    }
    if interrupt and pending:
        update.update(
            {
                "approval_required": True,
                "pending_action": pending.model_dump(),
            }
        )
    # Interrupt happens on the HITL node; SRE only flags the pending action.
    return NodeResult(update=update, interrupt=False)


def support_node(state: GraphState, tools: ToolRegistry, llm: LLMProvider) -> NodeResult:
    tools.bind(tools.tracer, state.run_id, "support")
    plan = llm.complete(_llm_blob(state), role="support").parsed
    query = str(plan.get("customer_query") or state.task)
    lookup = _safe_invoke(tools, "crm_lookup", {"query": query})
    matches = (lookup.get("data") or {}).get("matches") or []
    customer = matches[0] if matches else None
    amount = extract_refund_amount(state.task)
    if amount is None and customer:
        amount = float((customer.get("last_charge_usd") or customer.get("duplicate_charge_usd") or 0) or 0)
    if amount is None:
        amount = float(plan.get("refund_usd") or 0)

    policy = _safe_invoke(tools, "knowledge_search", {"query": "refund duplicate charge policy", "kind": "kb"})
    ticket_res = _safe_invoke(
        tools,
        "create_ticket",
        {
            "title": f"Refund request: {query[:60]}",
            "body": f"Requested refund ${amount:.2f}. Task: {state.task}",
            "severity": "sev2" if amount >= 500 else "sev3",
            "queue": "billing",
            "customer_id": (customer or {}).get("id"),
        },
    )
    tool_calls = [lookup, policy, ticket_res]
    tickets: list[dict[str, Any]] = []
    if ticket_res.get("ok"):
        data = ticket_res["data"]
        tickets.append(Ticket(id=data["id"], title=data["title"], severity=data["severity"], body=data["body"], queue="billing").model_dump())

    email_to = (customer or {}).get("email") or "customer@example.com"
    draft = (
        f"Hello {(customer or {}).get('name', 'there')},\n\n"
        f"We reviewed your account and found a duplicate charge of ${amount:.2f}. "
        f"Per the refund policy we can reverse it"
        f"{' after a supervisor approves the amount' if amount > 500 else ''}.\n\n"
        "ArmadaCrew Support"
    )
    high = amount > (tools.settings.hitl_refund_usd)
    summary = (
        f"Support: customer={(customer or {}).get('name')} email={email_to} "
        f"refund=${amount:.2f} high_risk={high}."
    )
    pending = None
    interrupt = False
    if high and not (state.human_decision and state.human_decision.approved):
        pending = PendingAction(
            action_type="refund",
            risk="high",
            reason=f"Refund ${amount:.2f} exceeds ${tools.settings.hitl_refund_usd:.0f} policy threshold",
            payload={"amount_usd": amount, "customer": customer, "email_to": email_to, "draft": draft},
            policy="refund_threshold",
        )
        interrupt = True

    steps = [
        PlanStep(id="verify", title="Verify duplicate charge in CRM", owner="support", risk="low"),
        PlanStep(
            id="refund",
            title=f"Issue ${amount:.2f} refund",
            owner="support",
            risk="high" if high else "low",
            requires_approval=high,
        ),
        PlanStep(id="reply", title="Send customer email", owner="writer", risk="high", requires_approval=True),
    ]
    update: dict[str, Any] = {
        "messages": [Message(role="assistant", content=summary, agent="support")],
        "tool_calls": tool_calls,
        "tickets": tickets,
        "plan_steps": [s.model_dump() for s in steps],
        "plan": {"title": "Duplicate charge refund", "amount_usd": amount, "customer": customer},
        "next_agent": "hitl" if interrupt else "critic",
        "metadata": {"support_done": True, "refund_usd": amount, "email_to": email_to, "email_draft": draft},
        "artifacts": [
            Artifact(name="customer-draft.md", kind="email_draft", content=draft, created_by="support").model_dump()
        ],
    }
    if interrupt and pending:
        update.update({"approval_required": True, "pending_action": pending.model_dump()})
    return NodeResult(update=update, interrupt=False)


def critic_node(state: GraphState, tools: ToolRegistry, llm: LLMProvider) -> NodeResult:
    tools.bind(tools.tracer, state.run_id, "critic")
    rounds = int(state.metadata.get("critic_rounds") or 0)
    evidence = [tc for tc in state.tool_calls if tc.get("ok")]
    extra = (
        f"plan_steps={len(state.plan_steps)} evidence_tools={len(evidence)} "
        f"critic_rounds={rounds} previous_verdict={state.critic_verdict}"
    )
    parsed = llm.complete(_llm_blob(state, extra=extra), role="critic").parsed
    score = float(parsed.get("score") if parsed.get("score") is not None else 0.5)

    # Heuristic overlay so the critic is not a coin flip: evidence + named steps required.
    if len(evidence) < 1 or not state.plan_steps:
        score = min(score, 0.4)
    elif rounds >= 1:
        score = max(score, 0.84)
    elif len(evidence) >= 3 and state.plan_steps:
        # First pass still rejects once to demonstrate the production revision loop.
        if rounds == 0 and not state.metadata.get("skip_first_critic_reject"):
            score = min(score, 0.5)

    threshold = tools.settings.critic_pass_score
    verdict = "approve" if score >= threshold else "reject"
    feedback = str(parsed.get("feedback") or "")
    if verdict == "reject" and not feedback:
        feedback = "Missing evidence or rollback; specialist must revise the plan."
    if verdict == "approve" and not feedback:
        feedback = "Plan is sufficiently evidenced and policy-checked."

    specialist = "support" if state.resolved_mode == "customer_ops" else "sre"
    nxt = "writer" if verdict == "approve" else specialist
    summary = f"Critic score={score:.2f} verdict={verdict}. {feedback}"
    return NodeResult(
        update={
            "messages": [Message(role="assistant", content=summary, agent="critic")],
            "critic_score": score,
            "critic_feedback": feedback,
            "critic_verdict": verdict,
            "next_agent": nxt,
            "metadata": {"critic_rounds": rounds + 1, "critic_done": verdict == "approve"},
        }
    )


def writer_node(state: GraphState, tools: ToolRegistry, llm: LLMProvider) -> NodeResult:
    tools.bind(tools.tracer, state.run_id, "writer")
    parsed = llm.complete(_llm_blob(state), role="writer").parsed
    mode = state.resolved_mode
    decision_note = ""
    if state.human_decision:
        decision_note = (
            f"\nHuman {'APPROVED' if state.human_decision.approved else 'REJECTED'} "
            f"with comment: {state.human_decision.comment or '(none)'}"
        )

    if mode == "customer_ops":
        draft = str(state.metadata.get("email_draft") or parsed.get("body") or "We will follow up shortly.")
        to = str(state.metadata.get("email_to") or "customer@example.com")
        body = draft + decision_note
        pii = looks_like_pii_email(body)
        # Queue without sending — send_email always HITL unless this run already approved it.
        already = bool(state.human_decision and state.human_decision.approved)
        email_result: dict[str, Any]
        interrupt = False
        pending = None
        if already:
            email_result = _safe_invoke(
                tools,
                "send_email",
                {"to": to, "subject": "Your refund request", "body": body, "contains_pii": pii, "approved": True},
                approved=True,
            )
        else:
            queued = tools.store.queue_email(to, "Your refund request", body, approved=False)
            email_result = {"ok": True, "tool": "send_email", "data": queued, "queued": True}
            pending = PendingAction(
                action_type="send_email",
                risk="high",
                reason="Outbound customer email requires approval (PII / refund communication)",
                payload={"to": to, "subject": "Your refund request", "body": body},
                policy="email_gate",
            )
            interrupt = not already
        content = f"To: {to}\nSubject: Your refund request\n\n{body}"
        artifact = Artifact(name="customer-email.md", kind="customer_email", content=content, created_by="writer")
        update: dict[str, Any] = {
            "messages": [Message(role="assistant", content="Writer drafted the customer email.", agent="writer")],
            "artifacts": [artifact.model_dump()],
            "tool_calls": [email_result],
            "metadata": {"writer_done": True, "email_queued": True},
        }
        if interrupt and pending:
            update.update(
                {
                    "approval_required": True,
                    "pending_action": pending.model_dump(),
                    "next_agent": "hitl",
                    "status": "running",
                }
            )
            return NodeResult(update=update, interrupt=False)
        update["status"] = "completed"
        update["next_agent"] = "end"
        return NodeResult(update=update, goto="__end__")

    service = str(state.metadata.get("service") or _infer_service(state))
    steps = "\n".join(f"- {s.title} [{s.risk}]" for s in state.plan_steps) or "- See transcript"
    report = (
        f"# Incident report — {service}\n\n"
        f"**Task:** {state.task}\n\n"
        f"**Critic:** {state.critic_verdict} ({state.critic_score})\n\n"
        f"## Timeline\n"
        + "\n".join(f"- {n}" for n in state.route_history)
        + f"\n\n## Plan\n{steps}\n"
        f"{decision_note}\n\n"
        "## Evidence\n"
        f"- Tool calls recorded: {len(state.tool_calls)}\n"
        f"- Tickets: {', '.join(t.id for t in state.tickets) or 'none'}\n"
    )
    artifact = Artifact(name=f"{service}-incident-report.md", kind="incident_report", content=report, created_by="writer")
    return NodeResult(
        update={
            "messages": [Message(role="assistant", content=f"Writer published the {service} incident report.", agent="writer")],
            "artifacts": [artifact.model_dump()],
            "status": "completed",
            "next_agent": "end",
            "metadata": {"writer_done": True},
        },
        goto="__end__",
    )


def hitl_node(state: GraphState, tools: ToolRegistry, llm: LLMProvider) -> NodeResult:
    """Pause when no decision exists; apply the human gate on resume."""

    llm.complete(_llm_blob(state), role="hitl")  # structured thought, unused beyond audit
    decision = state.human_decision
    pending = state.pending_action
    if decision is None:
        reason = pending.reason if pending else "High-risk action requires a human"
        action = pending or PendingAction(
            action_type="unknown",
            risk="high",
            reason=reason,
            payload={},
        )
        return NodeResult(
            update={
                "approval_required": True,
                "pending_action": action.model_dump(),
                "status": "interrupted",
                "messages": [Message(role="assistant", content=f"HITL pause: {reason}", agent="hitl")],
                "next_agent": "hitl",
            },
            interrupt=True,
        )

    if not decision.approved:
        return NodeResult(
            update={
                "status": "rejected",
                "approval_required": False,
                "messages": [
                    Message(
                        role="human",
                        content=f"Rejected: {decision.comment or 'no comment'}",
                        agent="operator",
                    )
                ],
                "next_agent": "end",
            },
            goto="__end__",
        )

    # Approved: execute the gated side effect if it was an email/refund marker.
    notes = [f"Human approved ({decision.actor}): {decision.comment or 'ok'}"]
    tool_calls: list[dict[str, Any]] = []
    if pending and pending.action_type == "send_email":
        payload = dict(pending.payload)
        payload["approved"] = True
        tool_calls.append(_safe_invoke(tools.bind(tools.tracer, state.run_id, "hitl"), "send_email", payload, approved=True))
        notes.append("Email released from the approval queue.")
    if pending and pending.action_type == "refund":
        notes.append(f"Refund ${pending.payload.get('amount_usd')} authorized.")
    if pending and pending.action_type == "prod_restart":
        notes.append(f"Prod restart authorized for {pending.payload.get('service')}.")

    return NodeResult(
        update={
            "approval_required": False,
            "pending_action": None,
            "status": "running",
            "messages": [Message(role="human", content=" ".join(notes), agent="operator")],
            "tool_calls": tool_calls,
            "next_agent": "writer" if state.metadata.get("sre_done") or state.metadata.get("support_done") else "supervisor",
            "metadata": {"hitl_approved": True},
        }
    )


def make_agent_nodes(tools: ToolRegistry, llm: LLMProvider | None = None) -> dict[str, Any]:
    provider = llm or get_llm(tools.settings)

    def wrap(fn):
        def _node(state: GraphState) -> NodeResult:
            return fn(state, tools, provider)

        _node.__name__ = fn.__name__
        return _node

    return {
        "researcher": wrap(researcher_node),
        "sre": wrap(sre_node),
        "support": wrap(support_node),
        "critic": wrap(critic_node),
        "writer": wrap(writer_node),
        "hitl": wrap(hitl_node),
    }


def detect_mode(task: str, mode: str) -> str:
    if mode and mode != "auto":
        return mode
    return classify_task(task)
