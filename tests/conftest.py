from __future__ import annotations

from pathlib import Path

import pytest

from armadacrew.bootstrap import AppContext, build_context
from armadacrew.config import Settings

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "armada-test.db",
        data_dir=ROOT / "data",
        static_dir=ROOT / "static",
        llm_provider="mock",
        max_graph_steps=20,
        tool_timeout_seconds=5,
        tool_max_retries=1,
    )


@pytest.fixture()
def ctx(settings: Settings) -> AppContext:
    context = build_context(settings, reset_domain=True)
    yield context
    context.runs.shutdown()
    context.checkpointer.close()
