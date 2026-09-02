from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    deepseek_api_key: str | None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_timeout_seconds: float = 45.0
    deepseek_enabled: bool = True
    default_top_k: int = 5
    max_top_k: int = 20
    chroma_dir: str = "output/chroma_db"
    chroma_collection: str = "bjut_schedule"
    max_tool_rounds: int = 4
    max_chat_history_messages: int = 20

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        load_dotenv(root / ".env", override=False)
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
        return cls(
            project_root=root,
            deepseek_api_key=api_key,
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            deepseek_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/"),
            deepseek_timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "45")),
            deepseek_enabled=_env_bool("DEEPSEEK_ENABLED", True),
            default_top_k=int(os.getenv("RAG_DEFAULT_TOP_K", "5")),
            max_top_k=int(os.getenv("RAG_MAX_TOP_K", "20")),
            chroma_dir=os.getenv("RAG_CHROMA_DIR", "output/chroma_db").strip(),
            chroma_collection=os.getenv("RAG_CHROMA_COLLECTION", "bjut_schedule").strip(),
            max_tool_rounds=int(os.getenv("RAG_MAX_TOOL_ROUNDS", "4")),
            max_chat_history_messages=int(
                os.getenv("RAG_MAX_CHAT_HISTORY_MESSAGES", "20")
            ),
        )
