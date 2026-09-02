"""Domain corpus: runbooks, knowledge articles, synthetic customers, services, and telemetry.

Tools query this in-memory infra rather than empty stubs. Telemetry is deterministic
for a given (service, metric, window) so tests and the mock LLM stay reproducible.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from armadacrew.config import Settings, get_settings


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    path: str
    kind: str
    body: str
    tags: list[str] = field(default_factory=list)

    @property
    def excerpt(self) -> str:
        text = " ".join(self.body.strip().split())
        return text[:420] + ("…" if len(text) > 420 else "")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DomainStore:
    """Loads bundled markdown + JSON fixtures and serves synthetic ops data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.documents: list[Document] = []
        self.customers: list[dict[str, Any]] = []
        self.services: list[dict[str, Any]] = []
        self.incidents: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self.email_queue: list[dict[str, Any]] = []
        self.pager_events: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        self.documents = []
        self.documents.extend(self._load_markdown(self.settings.runbooks_dir, "runbook"))
        self.documents.extend(self._load_markdown(self.settings.kb_dir, "kb"))
        fixtures = self.settings.fixtures_dir
        self.customers = self._load_json(fixtures / "customers.json")
        self.services = self._load_json(fixtures / "services.json")
        self.incidents = self._load_json(fixtures / "incidents.json")
        self.tickets = []
        self.email_queue = []
        self.pager_events = [
            {
                "incident_key": inc["id"],
                "service": inc.get("service"),
                "severity": inc.get("severity"),
                "acked": False,
                "title": inc.get("title"),
            }
            for inc in self.incidents
            if inc.get("status") in {"open", "investigating"}
        ]

    def reload(self) -> None:
        self._load()

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "customers", "services", "incidents"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        return []

    @staticmethod
    def _load_markdown(folder: Path, kind: str) -> list[Document]:
        docs: list[Document] = []
        if not folder.exists():
            return docs
        for path in sorted(folder.glob("*.md")):
            body = path.read_text(encoding="utf-8")
            title = path.stem.replace("-", " ").title()
            first = body.strip().splitlines()[0] if body.strip() else title
            if first.startswith("#"):
                title = first.lstrip("#").strip()
            tags = re.findall(r"`([a-z0-9\-]+)`", body.lower())[:12]
            docs.append(
                Document(
                    doc_id=f"{kind}:{path.stem}",
                    title=title,
                    path=str(path),
                    kind=kind,
                    body=body,
                    tags=list(dict.fromkeys(tags)),
                )
            )
        return docs

    def search_knowledge(self, query: str, kind: str | None = None, k: int = 5) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        q_set = set(q_tokens)
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            if kind and doc.kind != kind:
                continue
            tokens = _tokenize(f"{doc.title} {doc.body} {' '.join(doc.tags)}")
            if not tokens:
                continue
            overlap = len(q_set.intersection(tokens))
            if overlap == 0:
                continue
            tf = overlap / max(len(q_set), 1)
            title_bonus = 0.35 if any(t in doc.title.lower() for t in q_set) else 0.0
            kind_bonus = 0.1 if doc.kind == "runbook" and any(
                w in query.lower() for w in ("latency", "outage", "error", "cpu", "restart")
            ) else 0.0
            score = tf + title_bonus + kind_bonus
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, doc in scored[:k]:
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "kind": doc.kind,
                    "score": round(score, 4),
                    "tags": doc.tags,
                    "excerpt": doc.excerpt,
                    "path": doc.path,
                }
            )
        return results

    def find_service(self, name: str) -> dict[str, Any] | None:
        needle = name.lower().strip()
        for svc in self.services:
            aliases = [svc.get("name", ""), svc.get("id", ""), *svc.get("aliases", [])]
            if any(needle == str(a).lower() or needle in str(a).lower() for a in aliases):
                return svc
        return None

    def infer_service(self, task: str) -> dict[str, Any] | None:
        lowered = task.lower()
        ranked: list[tuple[int, dict[str, Any]]] = []
        for svc in self.services:
            names = [svc.get("name", ""), svc.get("id", ""), *svc.get("aliases", [])]
            hits = sum(1 for n in names if n and str(n).lower() in lowered)
            if hits:
                ranked.append((hits, svc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1] if ranked else None

    def metrics_query(
        self,
        service: str,
        metric: str = "p99_ms",
        window_minutes: int = 60,
        env: str = "prod",
    ) -> dict[str, Any]:
        svc = self.find_service(service) or {"id": service, "name": service, "slo": {}}
        seed = abs(hash((svc.get("id"), metric, env, window_minutes))) % (2**32)
        rng = random.Random(seed)
        now = _utc_now().replace(second=0, microsecond=0)
        incident = self._active_incident_for(str(svc.get("id", service)))
        points: list[dict[str, Any]] = []
        baseline = self._baseline(metric, svc)
        spike = self._should_spike(svc, metric, incident)
        for i in range(window_minutes):
            ts = now - timedelta(minutes=window_minutes - i)
            noise = rng.gauss(0, baseline * 0.08)
            value = max(0.01, baseline + noise)
            if spike and i >= window_minutes - 18:
                ramp = (i - (window_minutes - 18)) / 18
                value = baseline * (1 + 7.5 * ramp) + abs(rng.gauss(0, baseline * 0.2))
            points.append({"ts": ts.isoformat(), "value": round(float(value), 2)})
        values = [p["value"] for p in points]
        latest = values[-1] if values else 0.0
        p95 = sorted(values)[max(0, math.floor(0.95 * (len(values) - 1)))] if values else 0.0
        slo = (svc.get("slo") or {}).get(metric)
        breached = bool(slo is not None and latest > float(slo))
        return {
            "service": svc.get("id", service),
            "env": env,
            "metric": metric,
            "window_minutes": window_minutes,
            "baseline": baseline,
            "latest": latest,
            "p95": round(p95, 2),
            "slo": slo,
            "slo_breached": breached,
            "incident": incident["id"] if incident else None,
            "points": points[-40:],
            "summary": (
                f"{svc.get('id', service)} {metric} latest={latest} p95={round(p95, 2)} "
                f"slo={slo} breached={breached}"
            ),
        }

    def logs_search(
        self,
        service: str,
        query: str = "error",
        env: str = "prod",
        limit: int = 25,
    ) -> dict[str, Any]:
        svc = self.find_service(service) or {"id": service, "name": service}
        svc_id = str(svc.get("id", service))
        seed = abs(hash((svc_id, query, env))) % (2**32)
        rng = random.Random(seed)
        now = _utc_now()
        incident = self._active_incident_for(svc_id)
        templates = self._log_templates(svc_id, query.lower())
        lines: list[dict[str, Any]] = []
        n = max(8, min(limit, 40))
        for i in range(n):
            ts = now - timedelta(seconds=rng.randint(5, 3600))
            level = "ERROR" if incident or "error" in query.lower() or i % 4 == 0 else "INFO"
            if incident and i < n // 2:
                level = "ERROR"
            message = templates[i % len(templates)]
            if incident:
                message = f"{message} correlated_incident={incident['id']}"
            lines.append(
                {
                    "ts": ts.isoformat(),
                    "service": svc_id,
                    "env": env,
                    "level": level,
                    "trace_id": f"tr-{rng.randint(10000, 99999)}",
                    "message": message,
                }
            )
        lines.sort(key=lambda row: row["ts"], reverse=True)
        error_count = sum(1 for row in lines if row["level"] == "ERROR")
        return {
            "service": svc_id,
            "query": query,
            "env": env,
            "hits": lines[:limit],
            "error_count": error_count,
            "summary": f"{len(lines[:limit])} log lines for {svc_id} ({error_count} ERROR)",
        }

    def crm_lookup(self, query: str) -> dict[str, Any]:
        needle = query.lower().strip()
        matches: list[dict[str, Any]] = []
        for customer in self.customers:
            blob = " ".join(
                str(customer.get(k, ""))
                for k in ("id", "name", "email", "plan", "company")
            ).lower()
            if needle in blob or any(tok in blob for tok in needle.split() if len(tok) > 2):
                matches.append(customer)
        if not matches and self.customers:
            # Fuzzy: last-name token
            for customer in self.customers:
                if any(tok in customer.get("name", "").lower() for tok in needle.split()):
                    matches.append(customer)
        return {
            "query": query,
            "matches": matches[:8],
            "count": len(matches),
            "summary": f"{len(matches)} customer match(es) for {query!r}",
        }

    def create_ticket(
        self,
        title: str,
        body: str,
        severity: str = "sev2",
        queue: str = "sre",
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        ticket_id = f"TCK-{len(self.tickets) + 1001}"
        ticket = {
            "id": ticket_id,
            "title": title,
            "body": body,
            "severity": severity,
            "queue": queue,
            "customer_id": customer_id,
            "status": "open",
            "created_at": _utc_now().isoformat(),
        }
        self.tickets.append(ticket)
        return ticket

    def queue_email(self, to: str, subject: str, body: str, approved: bool = False) -> dict[str, Any]:
        item = {
            "id": f"EML-{len(self.email_queue) + 4001}",
            "to": to,
            "subject": subject,
            "body": body,
            "status": "sent" if approved else "queued_pending_approval",
            "approved": approved,
            "created_at": _utc_now().isoformat(),
        }
        self.email_queue.append(item)
        return item

    def pager_ack(self, incident_key: str, actor: str = "armadacrew") -> dict[str, Any]:
        for event in self.pager_events:
            if event["incident_key"] == incident_key or incident_key.lower() in str(event.get("title", "")).lower():
                event["acked"] = True
                event["acked_by"] = actor
                event["acked_at"] = _utc_now().isoformat()
                return {"ok": True, "event": event}
        created = {
            "incident_key": incident_key,
            "acked": True,
            "acked_by": actor,
            "acked_at": _utc_now().isoformat(),
            "title": incident_key,
        }
        self.pager_events.append(created)
        return {"ok": True, "event": created, "note": "synthetic ack created"}

    def related_incidents(self, task: str, service_id: str | None = None) -> list[dict[str, Any]]:
        tokens = set(_tokenize(task))
        hits: list[tuple[int, dict[str, Any]]] = []
        for inc in self.incidents:
            blob = _tokenize(json.dumps(inc))
            score = len(tokens.intersection(blob))
            if service_id and inc.get("service") == service_id:
                score += 3
            if score:
                hits.append((score, inc))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [inc for _, inc in hits[:5]]

    def _active_incident_for(self, service_id: str) -> dict[str, Any] | None:
        for inc in self.incidents:
            if inc.get("service") == service_id and inc.get("status") in {"open", "investigating", "resolved"}:
                return inc
        return None

    @staticmethod
    def _baseline(metric: str, svc: dict[str, Any]) -> float:
        slo = (svc.get("slo") or {}).get(metric)
        defaults = {
            "p99_ms": 120.0,
            "p95_ms": 80.0,
            "error_rate": 0.4,
            "cpu_pct": 42.0,
            "rps": 850.0,
            "sat_pct": 55.0,
        }
        if slo is not None:
            return float(slo) * 0.45
        return defaults.get(metric, 10.0)

    @staticmethod
    def _should_spike(svc: dict[str, Any], metric: str, incident: dict[str, Any] | None) -> bool:
        svc_id = str(svc.get("id", "")).lower()
        if incident:
            return True
        if "payments" in svc_id and metric in {"p99_ms", "p95_ms", "error_rate"}:
            return True
        if "auth" in svc_id and metric in {"error_rate", "p99_ms"}:
            return True
        return False

    @staticmethod
    def _log_templates(service: str, query: str) -> list[str]:
        common = [
            f"{service} request completed status=200 latency_ms=41",
            f"{service} upstream timeout from checkout-db after 2000ms",
            f"{service} circuit breaker half-open on dependency=redis-primary",
            f"{service} retry exhausted for charge_id=ch_4291",
            f"{service} authz cache miss user_hash=sha256:9f3a",
            f"{service} pod memory working_set_bytes=1.8e9",
            f"{service} SLO burn rate 6h window=3.4",
            f"{service} handler /v1/charges raised DeadlineExceeded",
        ]
        if "auth" in service:
            common.extend(
                [
                    f"{service} jwt verify failed kid=legacy-2023",
                    f"{service} idp latency_ms=2400 issuer=okta",
                ]
            )
        if "payment" in service:
            common.extend(
                [
                    f"{service} stripe 429 rate_limited idempotency_key=dup-720",
                    f"{service} duplicate capture suspected payment_intent=pi_ada_720",
                ]
            )
        if "refund" in query or "charge" in query:
            common.append(f"{service} duplicate charge detector fired amount_usd=720")
        return common


_STORE: DomainStore | None = None


def get_store(settings: Settings | None = None) -> DomainStore:
    global _STORE
    if _STORE is None or (settings is not None and settings is not _STORE.settings):
        _STORE = DomainStore(settings)
    return _STORE


def reset_store(settings: Settings | None = None) -> DomainStore:
    global _STORE
    _STORE = DomainStore(settings)
    return _STORE


def classify_task(task: str) -> str:
    """Return incident | customer_ops | mixed from free-text."""

    text = task.lower()
    customer_hits = sum(
        1
        for w in ("refund", "customer", "charge", "billing", "invoice", "email", "duplicate", "support")
        if w in text
    )
    incident_hits = sum(
        1
        for w in (
            "latency",
            "p99",
            "outage",
            "incident",
            "spike",
            "error",
            "restart",
            "cpu",
            "auth",
            "postmortem",
            "post-mortem",
            "5xx",
            "slo",
        )
        if w in text
    )
    if customer_hits and not incident_hits:
        return "customer_ops"
    if incident_hits and not customer_hits:
        return "incident"
    if customer_hits >= incident_hits and customer_hits:
        return "customer_ops"
    return "incident"


def looks_like_pii_email(body: str) -> bool:
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", body):
        return True
    if re.search(r"\b(?:\d[ -]*?){13,19}\b", body):
        return True
    if re.search(r"\b\d{4,5}\s+\w+\s+(street|st|ave|road|rd)\b", body, re.I):
        return True
    return False


def extract_refund_amount(text: str) -> float | None:
    match = re.search(r"\$\s?(\d{1,6}(?:\.\d{1,2})?)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d{1,6}(?:\.\d{1,2})?)\s*(?:usd|dollars)", text, re.I)
    if match:
        return float(match.group(1))
    return None


def iter_named_entities(task: str) -> Iterable[str]:
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", task):
        yield match.group(1)
