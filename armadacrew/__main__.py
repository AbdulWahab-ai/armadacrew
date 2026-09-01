"""CLI: `python -m armadacrew` serves the UI; `python -m armadacrew run "..."` is a one-shot graph."""

from __future__ import annotations

import argparse
import json
import sys

from armadacrew.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="armadacrew", description="ArmadaCrew operations copilot")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Run the FastAPI mission-control server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")

    run = sub.add_parser("run", help="Execute a single task synchronously")
    run.add_argument("task", help="Natural language ops task")
    run.add_argument("--mode", choices=["incident", "customer_ops", "auto"], default="auto")
    run.add_argument("--approve", action="store_true", help="Auto-approve HITL if the run interrupts")

    sub.add_parser("health", help="Print local health JSON without serving")

    args = parser.parse_args(argv)
    cmd = args.cmd or "serve"

    if cmd == "serve":
        import uvicorn

        settings = get_settings()
        host = args.host or settings.host
        port = args.port or settings.port
        uvicorn.run("armadacrew.api:app", host=host, port=port, reload=args.reload)
        return 0

    if cmd == "health":
        from armadacrew.bootstrap import build_context

        ctx = build_context()
        print(
            json.dumps(
                {
                    "ok": True,
                    "llm": ctx.llm.name,
                    "documents": len(ctx.store.documents),
                    "db": str(ctx.settings.db_path),
                },
                indent=2,
            )
        )
        ctx.runs.shutdown()
        ctx.checkpointer.close()
        return 0

    if cmd == "run":
        from armadacrew.bootstrap import build_context

        ctx = build_context()
        state = ctx.runs.invoke_sync(args.task, mode=args.mode)
        if state.status == "interrupted" and args.approve:
            state = ctx.graph.resume(state.run_id, approved=True, comment="cli --approve", actor="cli")
        print(json.dumps(state.model_dump(), indent=2, default=str))
        ctx.runs.shutdown()
        ctx.checkpointer.close()
        return 0 if state.status in {"completed", "interrupted"} else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
