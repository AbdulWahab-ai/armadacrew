"""Sandboxed tools with JSON schema, validation, timeout, retry, and audit logging.

Every tool talks to the in-memory DomainStore (runbooks, synthetic metrics/logs/CRM)
rather than returning hardcoded empty payloads.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from armadacrew.config import Settings, get_settings
from armadacrew.domain import DomainStore, get_store, looks_like_pii_email
from armadacrew.tracing import Tracer


class ToolError(Exception):
    """Raised when a tool fails validation or execution."""


class ApprovalRequired(ToolError):
    """Raised when a side-effecting tool must pause for HITL."""

    def __init__(self, message: str, action: dict[str, Any]) -> None:
        super().__init__(message)
        self.action = action


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    kind: Literal["runbook", "kb"] | None = None
    k: int = Field(default=5, ge=1, le=12)


class MetricsQueryArgs(BaseModel):
    service: str = Field(min_length=2, max_length=80)
    metric: Literal["p99_ms", "p95_ms", "error_rate", "cpu_pct", "rps", "sat_pct"] = "p99_ms"
    window_minutes: int = Field(default=60, ge=5, le=1440)
    env: Literal["prod", "staging", "dev"] = "prod"


class LogsSearchArgs(BaseModel):
    service: str = Field(min_length=2, max_length=80)
    query: str = Field(default="error", min_length=1, max_length=120)
    env: Literal["prod", "staging", "dev"] = "prod"
    limit: int = Field(default=20, ge=1, le=100)


class CrmLookupArgs(BaseModel):
    query: str = Field(min_length=2, max_length=120)


class CreateTicketArgs(BaseModel):
    title: str = Field(min_length=3, max_length=180)
    body: str = Field(min_length=3, max_length=4000)
    severity: Literal["sev1", "sev2", "sev3", "sev4"] = "sev2"
    queue: Literal["sre", "support", "billing", "security"] = "sre"
    customer_id: str | None = None


class SendEmailArgs(BaseModel):
    to: str = Field(min_length=3, max_length=200)
    subject: str = Field(min_length=3, max_length=180)
    body: str = Field(min_length=3, max_length=8000)
    contains_pii: bool = False
    approved: bool = False

    @field_validator("to")
    @classmethod
    def _looks_like_email(cls, value: str) -> str:
        if "@" not in value or "." not in value.split("@")[-1]:
            raise ValueError("to must be an email address")
        return value


class PagerAckArgs(BaseModel):
    incident_key: str = Field(min_length=2, max_length=80)
    actor: str = Field(default="armadacrew", min_length=2, max_length=80)


class ToolSpec(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    timeout_seconds: float = 8.0
    max_retries: int = 2
    requires_approval: bool = False
    side_effect: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return {
            "name": self.name,
            "description": self.description,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "requires_approval": self.requires_approval,
            "side_effect": self.side_effect,
            "parameters": schema,
        }


class ToolRegistry:
    """Validates arguments, enforces timeout/retry, sandboxes handlers, writes audits."""

    def __init__(
        self,
        store: DomainStore | None = None,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        run_id: str | None = None,
        node: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.tracer = tracer
        self.run_id = run_id
        self.node = node
        self.audit: list[dict[str, Any]] = []
        timeout = self.settings.tool_timeout_seconds
        retries = self.settings.tool_max_retries
        self._handlers: dict[str, Callable[[BaseModel], dict[str, Any]]] = {
            "knowledge_search": self._knowledge_search,
            "metrics_query": self._metrics_query,
            "logs_search": self._logs_search,
            "crm_lookup": self._crm_lookup,
            "create_ticket": self._create_ticket,
            "send_email": self._send_email,
            "pager_ack": self._pager_ack,
        }
        self.specs: dict[str, ToolSpec] = {
            "knowledge_search": ToolSpec(
                name="knowledge_search",
                description="Search bundled SRE runbooks and support knowledge articles.",
                args_model=KnowledgeSearchArgs,
                timeout_seconds=timeout,
                max_retries=retries,
            ),
            "metrics_query": ToolSpec(
                name="metrics_query",
                description="Query synthetic service metrics (p99, error rate, CPU, RPS).",
                args_model=MetricsQueryArgs,
                timeout_seconds=timeout,
                max_retries=retries,
            ),
            "logs_search": ToolSpec(
                name="logs_search",
                description="Search synthetic application logs for a service.",
                args_model=LogsSearchArgs,
                timeout_seconds=timeout,
                max_retries=retries,
            ),
            "crm_lookup": ToolSpec(
                name="crm_lookup",
                description="Look up synthetic customers, charges, and entitlements.",
                args_model=CrmLookupArgs,
                timeout_seconds=timeout,
                max_retries=retries,
            ),
            "create_ticket": ToolSpec(
                name="create_ticket",
                description="Open a ticket in the fake ITSM queue.",
                args_model=CreateTicketArgs,
                timeout_seconds=timeout,
                max_retries=retries,
                side_effect=True,
            ),
            "send_email": ToolSpec(
                name="send_email",
                description="Queue or send an email. High-risk (PII / customer mail) requires approval.",
                args_model=SendEmailArgs,
                timeout_seconds=timeout,
                max_retries=retries,
                requires_approval=True,
                side_effect=True,
            ),
            "pager_ack": ToolSpec(
                name="pager_ack",
                description="Acknowledge a pager incident key.",
                args_model=PagerAckArgs,
                timeout_seconds=timeout,
                max_retries=retries,
                side_effect=True,
            ),
        }

    def bind(self, tracer: Tracer | None, run_id: str | None, node: str | None) -> ToolRegistry:
        self.tracer = tracer
        self.run_id = run_id
        self.node = node
        return self

    def list_specs(self) -> list[dict[str, Any]]:
        return [spec.json_schema() for spec in self.specs.values()]

    def invoke(self, name: str, raw_args: dict[str, Any], approved: bool = False) -> dict[str, Any]:
        spec = self.specs.get(name)
        if spec is None:
            raise ToolError(f"Unknown tool {name!r}")
        try:
            args = spec.args_model.model_validate(raw_args)
        except ValidationError as exc:
            record = self._audit(name, raw_args, ok=False, duration_ms=0, retries=0, error=str(exc))
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc

        if name == "send_email":
            payload = args.model_dump()
            pii = bool(payload.get("contains_pii") or looks_like_pii_email(str(payload.get("body", ""))))
            if (pii or True) and not (approved or payload.get("approved")):
                # Customer email always needs HITL; still validate + audit the attempt.
                action = {
                    "action_type": "send_email",
                    "risk": "high",
                    "reason": "Outbound customer email (possible PII) requires approval",
                    "payload": payload,
                    "policy": "email_gate",
                }
                self._audit(name, payload, ok=False, duration_ms=0, retries=0, error="approval_required")
                raise ApprovalRequired("send_email requires human approval", action)

        last_error: Exception | None = None
        attempts = spec.max_retries + 1
        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                result = self._sandboxed(spec, args)
                duration_ms = (time.perf_counter() - t0) * 1000
                self._audit(name, args.model_dump(), ok=True, duration_ms=duration_ms, retries=attempt)
                return {"ok": True, "tool": name, "data": result, "retries": attempt, "duration_ms": duration_ms}
            except ApprovalRequired:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                duration_ms = (time.perf_counter() - t0) * 1000
                if attempt >= attempts - 1:
                    self._audit(
                        name,
                        args.model_dump(),
                        ok=False,
                        duration_ms=duration_ms,
                        retries=attempt,
                        error=str(exc),
                    )
                    raise ToolError(f"{name} failed after {attempt + 1} attempt(s): {exc}") from exc
                time.sleep(0.02 * (attempt + 1))
        raise ToolError(str(last_error) if last_error else f"{name} failed")

    def _sandboxed(self, spec: ToolSpec, args: BaseModel) -> dict[str, Any]:
        handler = self._handlers[spec.name]
        # Copy of arguments only — handlers receive a pydantic model, never caller mutables.
        isolated = spec.args_model.model_validate(args.model_dump())
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{spec.name}") as pool:
            future = pool.submit(handler, isolated)
            try:
                return future.result(timeout=spec.timeout_seconds)
            except FuturesTimeout as exc:
                raise ToolError(f"{spec.name} timed out after {spec.timeout_seconds}s") from exc

    def _audit(
        self,
        name: str,
        args: dict[str, Any],
        ok: bool,
        duration_ms: float,
        retries: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "tool": name,
            "args": args,
            "ok": ok,
            "duration_ms": round(duration_ms, 2),
            "retries": retries,
            "error": error,
            "node": self.node,
        }
        self.audit.append(record)
        if self.tracer and self.run_id:
            self.tracer.tool_called(
                self.run_id,
                name,
                args,
                ok=ok,
                duration_ms=duration_ms,
                retries=retries,
                error=error,
                node=self.node,
            )
        return record

    def _knowledge_search(self, args: KnowledgeSearchArgs) -> dict[str, Any]:
        hits = self.store.search_knowledge(args.query, kind=args.kind, k=args.k)
        return {"hits": hits, "count": len(hits), "query": args.query}

    def _metrics_query(self, args: MetricsQueryArgs) -> dict[str, Any]:
        return self.store.metrics_query(args.service, args.metric, args.window_minutes, args.env)

    def _logs_search(self, args: LogsSearchArgs) -> dict[str, Any]:
        return self.store.logs_search(args.service, args.query, args.env, args.limit)

    def _crm_lookup(self, args: CrmLookupArgs) -> dict[str, Any]:
        return self.store.crm_lookup(args.query)

    def _create_ticket(self, args: CreateTicketArgs) -> dict[str, Any]:
        return self.store.create_ticket(
            title=args.title,
            body=args.body,
            severity=args.severity,
            queue=args.queue,
            customer_id=args.customer_id,
        )

    def _send_email(self, args: SendEmailArgs) -> dict[str, Any]:
        return self.store.queue_email(args.to, args.subject, args.body, approved=True)

    def _pager_ack(self, args: PagerAckArgs) -> dict[str, Any]:
        return self.store.pager_ack(args.incident_key, args.actor)


def tool_catalog() -> list[dict[str, Any]]:
    return ToolRegistry().list_specs()
