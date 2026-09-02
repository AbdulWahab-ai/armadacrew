"""Process bootstrap: settings, domain store, checkpointer, compiled graph, run manager."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from armadacrew.checkpointer import SqliteCheckpointer
from armadacrew.config import Settings, get_settings
from armadacrew.domain import DomainStore, get_store, reset_store
from armadacrew.llm import LLMProvider, get_llm
from armadacrew.runtime import GraphState, initial_state
from armadacrew.supervisor import build_graph
from armadacrew.tools import ToolRegistry
from armadacrew.tracing import Tracer


@dataclass
class AppContext:
    settings: Settings
    store: DomainStore
    tools: ToolRegistry
    llm: LLMProvider
    checkpointer: SqliteCheckpointer
    tracer: Tracer
    graph: Any
    runs: "RunManager"


class RunManager:
    """Background executor so the API can stream events while the graph is in motion."""

    def __init__(self, graph: Any, checkpointer: SqliteCheckpointer, max_workers: int = 4) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="armada-run")
        self._lock = threading.Lock()
        self._futures: dict[str, Future[GraphState]] = {}

    def start(
        self,
        task: str,
        mode: Literal["incident", "customer_ops", "auto"] = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> GraphState:
        state = initial_state(task=task, mode=mode, metadata=metadata)
        self.checkpointer.upsert_run(state)
        self._submit(state.run_id, lambda: self.graph.invoke(state))
        return state

    def resume(self, run_id: str, approved: bool, comment: str = "", actor: str = "operator") -> GraphState:
        current = self.checkpointer.load_run(run_id)
        if current is None:
            raise KeyError(run_id)
        if current.status != "interrupted":
            raise RuntimeError(f"Run {run_id} is {current.status}, not interrupted")
        self._submit(run_id, lambda: self.graph.resume(run_id, approved=approved, comment=comment, actor=actor))
        refreshed = self.checkpointer.load_run(run_id)
        return refreshed or current

    def invoke_sync(self, task: str, mode: Literal["incident", "customer_ops", "auto"] = "auto") -> GraphState:
        state = initial_state(task=task, mode=mode)
        return self.graph.invoke(state)

    def get(self, run_id: str) -> GraphState | None:
        return self.checkpointer.load_run(run_id)

    def wait(self, run_id: str, timeout: float | None = 30.0) -> GraphState:
        with self._lock:
            fut = self._futures.get(run_id)
        if fut is None:
            loaded = self.get(run_id)
            if loaded is None:
                raise KeyError(run_id)
            return loaded
        return fut.result(timeout=timeout)

    def is_terminal(self, run_id: str) -> bool:
        state = self.get(run_id)
        if state is None:
            return False
        return state.status in {"completed", "failed", "rejected", "interrupted"} and not self.is_running(run_id)

    def is_running(self, run_id: str) -> bool:
        with self._lock:
            fut = self._futures.get(run_id)
        return bool(fut and not fut.done())

    def _submit(self, run_id: str, fn) -> None:
        fut = self._pool.submit(fn)
        with self._lock:
            self._futures[run_id] = fut

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def build_context(settings: Settings | None = None, reset_domain: bool = False) -> AppContext:
    cfg = settings or get_settings()
    store = reset_store(cfg) if reset_domain else get_store(cfg)
    tools = ToolRegistry(store=store, settings=cfg)
    llm = get_llm(cfg)
    checkpointer = SqliteCheckpointer(cfg.db_path)
    tracer = Tracer(checkpointer)
    graph = build_graph(tools=tools, llm=llm, checkpointer=checkpointer, tracer=tracer, settings=cfg)
    runs = RunManager(graph, checkpointer)
    return AppContext(
        settings=cfg,
        store=store,
        tools=tools,
        llm=llm,
        checkpointer=checkpointer,
        tracer=tracer,
        graph=graph,
        runs=runs,
    )
