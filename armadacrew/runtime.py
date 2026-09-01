"""LangGraph-style state graph runtime: nodes, reducers, conditional edges, interrupts.

Hiring-manager mapping
----------------------
LangGraph concept          ArmadaCrew analog
-------------------------  -----------------------------------------------
StateGraph / TypedDict     GraphState (pydantic) + StateGraph
node                       callable registered with add_node
edge / conditional_edge    add_edge / add_conditional_edges
reducer (operator.add)     LIST_APPEND_FIELDS merge in apply_update
checkpointer               SqliteCheckpointer.save_checkpoint after each node
interrupt / HITL           NodeResult.interrupt + status=interrupted
invoke / astream           CompiledGraph.invoke / stream_steps
Send API                   not required here; supervisor routes serially
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


START = "__start__"
END = "__end__"

LIST_APPEND_FIELDS = frozenset(
    {
        "messages",
        "artifacts",
        "tickets",
        "tool_calls",
        "route_history",
        "plan_steps",
    }
)

TERMINAL_STATUSES = frozenset({"completed", "failed", "rejected"})


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    interrupted = "interrupted"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "human"] = "assistant"
    content: str
    agent: str | None = None
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Artifact(BaseModel):
    name: str
    kind: str
    content: str
    created_by: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Ticket(BaseModel):
    id: str
    title: str
    severity: str = "sev2"
    status: str = "open"
    body: str = ""
    queue: str = "sre"


class PendingAction(BaseModel):
    action_type: str
    risk: Literal["low", "medium", "high"] = "high"
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)
    policy: str = "human_required"


class HumanDecision(BaseModel):
    approved: bool
    comment: str = ""
    actor: str = "operator"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PlanStep(BaseModel):
    id: str
    title: str
    owner: str
    risk: Literal["low", "medium", "high"] = "low"
    details: str = ""
    requires_approval: bool = False


class GraphState(BaseModel):
    """Shared graph state. List fields are append-reduced unless replaced explicitly."""

    run_id: str
    task: str
    mode: Literal["incident", "customer_ops", "auto"] = "auto"
    messages: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    tickets: list[Ticket] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    plan_steps: list[PlanStep] = Field(default_factory=list)
    plan: dict[str, Any] = Field(default_factory=dict)
    critic_score: float | None = None
    critic_feedback: str | None = None
    critic_verdict: Literal["approve", "reject", "pending"] = "pending"
    next_agent: str | None = None
    approval_required: bool = False
    pending_action: PendingAction | None = None
    human_decision: HumanDecision | None = None
    status: str = RunStatus.pending.value
    current_node: str | None = None
    last_node: str | None = None
    route_history: list[str] = Field(default_factory=list)
    step: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    llm_provider: str = "mock"

    @field_validator("status", mode="before")
    @classmethod
    def _status_str(cls, value: object) -> str:
        if isinstance(value, RunStatus):
            return value.value
        return str(value)

    @property
    def resolved_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        return str(self.metadata.get("classified_mode") or "incident")

    def transcript(self, limit: int = 12) -> str:
        lines = []
        for msg in self.messages[-limit:]:
            who = msg.agent or msg.role
            lines.append(f"[{who}] {msg.content[:400]}")
        return "\n".join(lines)


class NodeResult(BaseModel):
    update: dict[str, Any] = Field(default_factory=dict)
    interrupt: bool = False
    goto: str | None = None


NodeFn = Callable[[GraphState], NodeResult | dict[str, Any] | GraphState]
RouterFn = Callable[[GraphState], str]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid4().hex[:16]


def initial_state(
    task: str,
    mode: Literal["incident", "customer_ops", "auto"] = "auto",
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraphState:
    rid = run_id or new_run_id()
    return GraphState(
        run_id=rid,
        task=task,
        mode=mode,
        status=RunStatus.pending.value,
        messages=[Message(role="user", content=task, agent="operator")],
        metadata=metadata or {},
        current_node=START,
    )


def apply_update(state: GraphState, update: Mapping[str, Any]) -> GraphState:
    """Merge a node update into state using list-append reducers (LangGraph style)."""

    data = state.model_dump()
    for key, value in update.items():
        if key.startswith("_"):
            continue
        if key in LIST_APPEND_FIELDS and isinstance(value, list):
            existing = list(data.get(key) or [])
            data[key] = existing + value
        elif key == "metadata" and isinstance(value, dict):
            merged = dict(data.get("metadata") or {})
            merged.update(value)
            data[key] = merged
        elif value is not None:
            data[key] = value
        elif key in {"pending_action", "human_decision", "error", "critic_feedback"}:
            data[key] = value
    return GraphState.model_validate(data)


def _normalize_result(raw: NodeResult | dict[str, Any] | GraphState) -> NodeResult:
    if isinstance(raw, NodeResult):
        return raw
    if isinstance(raw, GraphState):
        return NodeResult(update=raw.model_dump())
    if isinstance(raw, dict):
        interrupt = bool(raw.pop("interrupt", False)) if "interrupt" in raw else False
        goto = raw.pop("goto", None) if "goto" in raw else None
        return NodeResult(update=raw, interrupt=interrupt, goto=goto)
    raise TypeError(f"Node returned unsupported type {type(raw)!r}")


class StateGraph:
    """Minimal StateGraph: add_node, add_edge, add_conditional_edges, compile."""

    def __init__(self, state_cls: type[GraphState] = GraphState) -> None:
        self.state_cls = state_cls
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.conditionals: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self.entry: str = START

    def add_node(self, name: str, fn: NodeFn) -> None:
        if name in {START, END}:
            raise ValueError(f"Reserved node name: {name}")
        self.nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> None:
        self.conditionals[src] = (router, mapping)

    def compile(self, checkpointer: Any, tracer: Any, max_steps: int = 18) -> CompiledGraph:
        if self.entry == START or self.entry not in self.nodes:
            raise ValueError("Entry point must be a registered node")
        return CompiledGraph(self, checkpointer=checkpointer, tracer=tracer, max_steps=max_steps)


class CompiledGraph:
    """Executable graph with SQLite checkpoints and HITL interrupts."""

    def __init__(self, graph: StateGraph, checkpointer: Any, tracer: Any, max_steps: int = 18) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.tracer = tracer
        self.max_steps = max_steps

    def next_node(self, state: GraphState, from_node: str) -> str:
        if from_node in self.graph.conditionals:
            router, mapping = self.graph.conditionals[from_node]
            key = router(state)
            dest = mapping.get(key) or mapping.get(str(key))
            if dest is None:
                raise KeyError(f"Router on {from_node} returned {key!r} not in {list(mapping)}")
            return dest
        if from_node in self.graph.edges:
            return self.graph.edges[from_node]
        return END

    def invoke(self, state: GraphState, resume: bool = False) -> GraphState:
        """Run until END, interrupt, failure, or max steps."""

        if not resume:
            state = apply_update(
                state,
                {"status": RunStatus.running.value, "current_node": self.graph.entry, "error": None},
            )
            self.checkpointer.upsert_run(state)
            self.tracer.emit(state.run_id, "run_start", {"task": state.task, "mode": state.mode})
            node_name = self.graph.entry
        else:
            node_name = state.current_node or self.graph.entry
            state = apply_update(state, {"status": RunStatus.running.value})
            self.tracer.resumed(
                state.run_id,
                bool(state.human_decision and state.human_decision.approved),
                state.human_decision.comment if state.human_decision else "",
            )

        steps_this_call = 0
        while node_name not in {END, START} and steps_this_call < self.max_steps:
            if node_name not in self.graph.nodes:
                state = apply_update(
                    state,
                    {
                        "status": RunStatus.failed.value,
                        "error": f"Unknown node {node_name}",
                        "current_node": node_name,
                    },
                )
                self.checkpointer.save_checkpoint(state, node_name, state.step)
                self.tracer.failed(state.run_id, state.error or "unknown node", node_name)
                return state

            fn = self.graph.nodes[node_name]
            step = state.step + 1
            state = apply_update(
                state,
                {
                    "current_node": node_name,
                    "last_node": node_name,
                    "step": step,
                    "status": RunStatus.running.value,
                    "route_history": [node_name],
                },
            )
            self.tracer.node_started(state.run_id, node_name, step)
            try:
                result = _normalize_result(fn(state))
            except Exception as exc:  # noqa: BLE001 — persist failure into run state
                state = apply_update(
                    state,
                    {
                        "status": RunStatus.failed.value,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self.checkpointer.save_checkpoint(state, node_name, step)
                self.tracer.node_finished(state.run_id, node_name, step, state.status)
                self.tracer.failed(state.run_id, state.error or "error", node_name)
                return state

            state = apply_update(state, result.update)
            if result.interrupt:
                state = apply_update(
                    state,
                    {
                        "status": RunStatus.interrupted.value,
                        "approval_required": True,
                        "current_node": node_name,
                    },
                )
                self.checkpointer.save_checkpoint(state, node_name, step)
                self.tracer.node_finished(state.run_id, node_name, step, state.status)
                pending = state.pending_action.model_dump() if state.pending_action else None
                self.tracer.interrupted(
                    state.run_id,
                    (state.pending_action.reason if state.pending_action else "HITL"),
                    pending,
                )
                return state

            self.checkpointer.save_checkpoint(state, node_name, step)
            self.tracer.node_finished(state.run_id, node_name, step, state.status)
            if state.status in TERMINAL_STATUSES:
                self.tracer.emit(state.run_id, "run_end", {"status": state.status})
                return state

            dest = result.goto or self.next_node(state, node_name)
            if dest == END:
                end_step = state.step + 1
                state = apply_update(
                    state,
                    {
                        "status": RunStatus.completed.value,
                        "current_node": END,
                        "step": end_step,
                    },
                )
                self.checkpointer.save_checkpoint(state, END, end_step)
                self.tracer.emit(state.run_id, "run_end", {"status": state.status})
                return state
            node_name = dest
            steps_this_call += 1

        if state.status not in TERMINAL_STATUSES and state.status != RunStatus.interrupted.value:
            state = apply_update(
                state,
                {
                    "status": RunStatus.failed.value,
                    "error": f"Exceeded max_graph_steps={self.max_steps}",
                },
            )
            self.checkpointer.save_checkpoint(state, state.current_node or "unknown", state.step)
            self.tracer.failed(state.run_id, state.error or "max steps", state.current_node)
        return state

    def resume(self, run_id: str, approved: bool, comment: str = "", actor: str = "operator") -> GraphState:
        loaded = self.checkpointer.load_run(run_id)
        if loaded is None:
            raise KeyError(f"Unknown run {run_id}")
        decision = HumanDecision(approved=approved, comment=comment, actor=actor)
        loaded = apply_update(
            loaded,
            {
                "human_decision": decision.model_dump(),
                "approval_required": False,
                "status": RunStatus.running.value,
            },
        )
        self.checkpointer.upsert_run(loaded)
        return self.invoke(loaded, resume=True)

    def stream_steps(self, state: GraphState) -> Iterator[GraphState]:
        """Invoke and yield the final state (checkpoints already persist intermediates)."""

        final = self.invoke(state)
        yield final
