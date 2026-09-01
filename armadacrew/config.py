"""Runtime settings. Defaults are fully offline (mock LLM, local SQLite, bundled data)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Process-wide configuration loaded from env / `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="ARMADACREW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    db_path: Path = Field(default=Path("./armadacrew.db"))
    data_dir: Path = Field(default=_project_root() / "data")
    static_dir: Path = Field(default=_project_root() / "static")
    llm_provider: Literal["mock", "openai", "anthropic"] = "mock"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-20241022"
    log_level: str = "INFO"
    max_graph_steps: int = Field(default=18, ge=4, le=64)
    tool_timeout_seconds: float = Field(default=8.0, ge=0.5, le=60.0)
    tool_max_retries: int = Field(default=2, ge=0, le=6)
    hitl_refund_usd: float = Field(default=500.0, ge=0)
    critic_pass_score: float = Field(default=0.62, ge=0.0, le=1.0)
    max_critic_rounds: int = Field(default=2, ge=1, le=5)
    sse_poll_seconds: float = Field(default=0.25, ge=0.05, le=2.0)

    @field_validator("data_dir", "static_dir", "db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (_project_root() / path).resolve()
        return path

    @property
    def runbooks_dir(self) -> Path:
        return self.data_dir / "runbooks"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def fixtures_dir(self) -> Path:
        return self.data_dir / "fixtures"


def get_settings() -> Settings:
    """Build settings from the current environment."""

    return Settings()
