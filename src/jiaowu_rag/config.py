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


DEFAULT_EMAIL_DOMAINS = ("emails.bjut.edu.cn", "bjut.edu.cn", "illinois.edu")


def _env_str(name: str) -> str | None:
    return os.getenv(name, "").strip() or None


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
    session_db: str = "output/sessions.sqlite"
    usage_db: str = "output/usage.sqlite"
    daily_token_budget: int = 3_000_000
    semester_start: str | None = None
    auth_required: bool = True
    auth_db: str = "output/auth.sqlite"
    allowed_email_domains: tuple[str, ...] = DEFAULT_EMAIL_DOMAINS
    user_daily_questions: int = 30
    user_questions_per_minute: int = 5
    smtp_host: str | None = None
    smtp_port: int = 465
    smtp_security: str = "ssl"
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None

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
            session_db=os.getenv("RAG_SESSION_DB", "output/sessions.sqlite").strip(),
            usage_db=os.getenv("RAG_USAGE_DB", "output/usage.sqlite").strip(),
            daily_token_budget=int(os.getenv("RAG_DAILY_TOKEN_BUDGET", "3000000")),
            semester_start=_env_str("RAG_SEMESTER_START"),
            auth_required=_env_bool("RAG_AUTH_REQUIRED", True),
            auth_db=os.getenv("RAG_AUTH_DB", "output/auth.sqlite").strip(),
            allowed_email_domains=tuple(
                domain.strip().lower()
                for domain in os.getenv(
                    "RAG_ALLOWED_EMAIL_DOMAINS", ",".join(DEFAULT_EMAIL_DOMAINS)
                ).split(",")
                if domain.strip()
            ),
            user_daily_questions=int(os.getenv("RAG_USER_DAILY_QUESTIONS", "30")),
            user_questions_per_minute=int(os.getenv("RAG_USER_QUESTIONS_PER_MINUTE", "5")),
            smtp_host=_env_str("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "465")),
            smtp_security=os.getenv("SMTP_SECURITY", "ssl").strip().lower(),
            smtp_username=_env_str("SMTP_USERNAME"),
            smtp_password=_env_str("SMTP_PASSWORD"),
            smtp_from=_env_str("SMTP_FROM"),
        )

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path
