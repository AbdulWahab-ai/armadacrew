"""LLM providers: deterministic mock (default) plus optional OpenAI / Anthropic adapters.

The mock still returns structured JSON plans and actions derived from the task and
role. Cloud adapters are thin wrappers around the same JSON contract and are never
required to run the product or the tests.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from armadacrew.config import Settings
from armadacrew.domain import classify_task, extract_refund_amount


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    text: str
    parsed: dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {"value": payload}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return {}
        return {}


class LLMProvider(ABC):
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        role: str = "supervisor",
        schema_hint: str | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class MockLLM(LLMProvider):
    """Offline planner that emits structured JSON from role + task keywords."""

    name = "mock"
    model = "armadacrew-mock-1"

    def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        role: str = "supervisor",
        schema_hint: str | None = None,
    ) -> LLMResponse:
        blob = "\n".join(_msg_content(m) for m in messages)
        task = _infer_task(blob)
        mode = classify_task(task)
        parsed = self._plan(role=role, task=task, mode=mode, blob=blob)
        text = json.dumps(parsed, indent=2)
        return LLMResponse(
            text=text,
            parsed=parsed,
            provider=self.name,
            model=self.model,
            usage={"input_tokens": max(1, len(blob) // 4), "output_tokens": max(1, len(text) // 4)},
        )

    def _plan(self, role: str, task: str, mode: str, blob: str) -> dict[str, Any]:
        lowered = f"{task}\n{blob}".lower()
        refund = extract_refund_amount(task) or extract_refund_amount(blob)
        high_refund = refund is not None and refund > 500
        wants_restart = any(w in lowered for w in ("restart", "reboot", "roll prod", "prod restart"))
        is_postmortem = any(w in lowered for w in ("postmortem", "post-mortem", "writeup", "yesterday"))
        critic_round = 0
        if "critic_rounds" in lowered:
            match = re.search(r"critic_rounds[\"']?\s*[:=]\s*(\d+)", lowered)
            if match:
                critic_round = int(match.group(1))
        has_research = "researcher" in lowered and "completed research" in lowered or "[researcher]" in lowered
        has_specialist = "[sre]" in lowered or "[support]" in lowered
        has_critic_reject = "verdict" in lowered and "reject" in lowered

        if role == "supervisor":
            nxt = _supervisor_next(mode, has_research, has_specialist, has_critic_reject, is_postmortem, blob)
            return {
                "thought": f"Routing a {mode} task; next specialist is {nxt}.",
                "next": nxt,
                "mode": mode,
                "risk": "high" if high_refund or wants_restart else "medium",
                "rationale": "Keyword policy plus conversation phase.",
            }

        if role == "researcher":
            queries = _research_queries(task, mode)
            return {
                "thought": "Gather runbooks, KB articles, and related incidents before specialists act.",
                "queries": queries,
                "focus": "payments-api" if "payment" in lowered else ("auth-service" if "auth" in lowered else "platform"),
                "next": "sre" if mode == "incident" else "support",
            }

        if role == "sre":
            service = "payments-api" if "payment" in lowered else ("auth-service" if "auth" in lowered else "edge-gateway")
            steps = [
                {"id": "ack", "title": "Acknowledge pager / open incident", "owner": "sre", "risk": "low"},
                {"id": "observe", "title": f"Confirm SLO burn on {service}", "owner": "sre", "risk": "low"},
                {
                    "id": "mitigate",
                    "title": "Scale replicas and shed non-critical traffic",
                    "owner": "sre",
                    "risk": "medium",
                },
            ]
            if wants_restart or "latency" in lowered or "outage" in lowered:
                steps.append(
                    {
                        "id": "restart",
                        "title": f"Rolling restart {service} in prod",
                        "owner": "sre",
                        "risk": "high",
                        "requires_approval": True,
                    }
                )
            return {
                "thought": f"Correlate metrics and logs for {service}, then propose a runbook sequence.",
                "service": service,
                "plan_title": f"Stabilize {service}",
                "steps": steps,
                "risk": "high" if any(s.get("risk") == "high" for s in steps) else "medium",
                "approval_required": any(s.get("requires_approval") for s in steps),
            }

        if role == "support":
            amount = refund or 0.0
            return {
                "thought": "Look up the customer, confirm duplicate charge, draft a reply, queue refund.",
                "customer_query": _customer_query(task),
                "refund_usd": amount,
                "approval_required": high_refund,
                "risk": "high" if high_refund else "low",
                "email_subject": "Your refund request",
                "next": "hitl" if high_refund else "critic",
            }

        if role == "critic":
            weak = "no evidence" in lowered or "empty plan" in lowered or critic_round == 0 and "plan_steps" in blob and "[]" in blob
            # First pass on incident plans is stricter unless evidence is clearly present.
            has_evidence = "tool_call" in lowered or "slo_breached" in lowered or "customer match" in lowered
            score = 0.88 if has_evidence and critic_round >= 1 else (0.44 if not has_evidence or critic_round == 0 else 0.81)
            if weak:
                score = 0.31
            # After a prior reject, accept a revised plan with evidence.
            if "previous_verdict=reject" in lowered or critic_round >= 1:
                score = max(score, 0.84)
            verdict = "approve" if score >= 0.62 else "reject"
            return {
                "thought": "Score completeness, evidence, blast radius, and policy compliance.",
                "score": score,
                "verdict": verdict,
                "feedback": (
                    "Need tool evidence and a named rollback before execution."
                    if verdict == "reject"
                    else "Evidence and policy checks look sufficient."
                ),
                "next": "writer" if verdict == "approve" else "specialist",
            }

        if role == "writer":
            kind = "customer_email" if mode == "customer_ops" else "incident_report"
            return {
                "thought": "Write the operator-facing artifact from the approved plan and tool evidence.",
                "artifact_kind": kind,
                "title": "Customer reply" if kind == "customer_email" else "Incident report",
                "send_email": mode == "customer_ops",
            }

        if role == "hitl":
            return {
                "thought": "Pause for a human because the pending action is high-risk.",
                "approval_required": True,
            }

        return {"thought": f"Unhandled role {role}", "next": "writer"}


def _msg_content(message: LLMMessage | dict[str, str]) -> str:
    if isinstance(message, LLMMessage):
        return message.content
    return str(message.get("content", ""))


def _infer_task(blob: str) -> str:
    match = re.search(r"(?:TASK|User task|task:)\s*(.+)", blob, re.I)
    if match:
        return match.group(1).splitlines()[0].strip()
    first = blob.strip().splitlines()
    return first[0][:400] if first else blob[:400]


def _research_queries(task: str, mode: str) -> list[str]:
    queries = [task]
    if mode == "incident":
        queries.extend(["latency runbook", "error budget", "rollback", "pager"])
    else:
        queries.extend(["refund policy", "duplicate charge", "billing SLA"])
    if re.search(r"auth", task, re.I):
        queries.append("auth outage runbook")
    if re.search(r"payment", task, re.I):
        queries.append("payments-api latency")
    return queries[:5]


def _customer_query(task: str) -> str:
    match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", task)
    if match:
        return match.group(1)
    match = re.search(r"[\w.]+@[\w.]+", task)
    if match:
        return match.group(0)
    return task


def _supervisor_next(
    mode: str,
    has_research: bool,
    has_specialist: bool,
    has_critic_reject: bool,
    is_postmortem: bool,
    blob: str,
) -> str:
    lowered = blob.lower()
    if "approval_required" in lowered and "true" in lowered and "approved" not in lowered:
        return "hitl"
    if "[writer]" in lowered and "final" in lowered:
        return "end"
    if mode == "customer_ops":
        if not has_specialist:
            return "support"
        if has_critic_reject:
            return "support"
        if "verdict" in lowered and "approve" in lowered:
            return "writer"
        return "critic"
    if not has_research:
        return "researcher"
    if not has_specialist:
        return "sre"
    if has_critic_reject:
        return "sre"
    if "verdict" in lowered and "approve" in lowered:
        return "writer"
    if is_postmortem and has_research and has_specialist and "[critic]" in lowered:
        return "writer"
    return "critic"


class OpenAIAdapter(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        role: str = "supervisor",
        schema_hint: str | None = None,
    ) -> LLMResponse:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return MockLLM().complete(messages, role=role, schema_hint=schema_hint)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install armadacrew[openai] to use the OpenAI adapter") from exc
        client = OpenAI(api_key=api_key)
        payload = [_as_openai(m) for m in messages]
        if schema_hint:
            payload.append({"role": "system", "content": f"Respond with JSON matching: {schema_hint}"})
        resp = client.chat.completions.create(model=self.model, messages=payload, temperature=0.1)
        text = resp.choices[0].message.content or "{}"
        usage = {
            "input_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
        }
        return LLMResponse(text=text, parsed=_extract_json(text), provider=self.name, model=self.model, usage=usage)


class AnthropicAdapter(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-20241022") -> None:
        self.model = model

    def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        role: str = "supervisor",
        schema_hint: str | None = None,
    ) -> LLMResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return MockLLM().complete(messages, role=role, schema_hint=schema_hint)
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install armadacrew[anthropic] to use the Anthropic adapter") from exc
        client = Anthropic(api_key=api_key)
        system = f"You are the {role} agent in ArmadaCrew. Reply with compact JSON only."
        if schema_hint:
            system += f" Schema: {schema_hint}"
        payload = []
        for message in messages:
            role_name = message.role if isinstance(message, LLMMessage) else message.get("role", "user")
            content = _msg_content(message)
            if role_name == "system":
                system += "\n" + content
            else:
                payload.append({"role": "assistant" if role_name == "assistant" else "user", "content": content})
        if not payload:
            payload = [{"role": "user", "content": "Produce the JSON plan."}]
        resp = client.messages.create(model=self.model, max_tokens=1200, system=system, messages=payload)
        text = "".join(getattr(block, "text", "") for block in resp.content)
        usage = {
            "input_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
        }
        return LLMResponse(text=text, parsed=_extract_json(text), provider=self.name, model=self.model, usage=usage)


def _as_openai(message: LLMMessage | dict[str, str]) -> dict[str, str]:
    if isinstance(message, LLMMessage):
        return {"role": message.role, "content": message.content}
    return {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}


def get_llm(settings: Settings | None = None) -> LLMProvider:
    from armadacrew.config import get_settings

    cfg = settings or get_settings()
    if cfg.llm_provider == "openai":
        return OpenAIAdapter(model=cfg.openai_model)
    if cfg.llm_provider == "anthropic":
        return AnthropicAdapter(model=cfg.anthropic_model)
    return MockLLM()
