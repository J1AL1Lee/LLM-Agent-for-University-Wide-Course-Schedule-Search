from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .models import TokenUsage


# Budgets reset at midnight Beijing time, when students' days actually start.
_BEIJING = timezone(timedelta(hours=8))


def beijing_today() -> date:
    return datetime.now(_BEIJING).date()


def today() -> str:
    return beijing_today().isoformat()


class UsageLedger:
    """Per-day model token totals, persisted so the daily budget survives restarts."""

    def __init__(self, path: Path, daily_token_budget: int) -> None:
        if daily_token_budget < 0:
            raise ValueError("RAG_DAILY_TOKEN_BUDGET cannot be negative")
        self.daily_token_budget = daily_token_budget
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_usage (
                    day TEXT PRIMARY KEY,
                    requests INTEGER NOT NULL DEFAULT 0,
                    model_calls INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def record(self, usage: TokenUsage, day: str | None = None) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO daily_usage
                    (day, requests, model_calls, input_tokens, cached_input_tokens, output_tokens)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    requests = requests + 1,
                    model_calls = model_calls + excluded.model_calls,
                    input_tokens = input_tokens + excluded.input_tokens,
                    cached_input_tokens = cached_input_tokens + excluded.cached_input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens
                """,
                (
                    day or today(),
                    usage.model_calls,
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                ),
            )

    def tokens_used(self, day: str | None = None) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT input_tokens + output_tokens FROM daily_usage WHERE day = ?",
                (day or today(),),
            ).fetchone()
        return int(row[0]) if row else 0

    def budget_exhausted(self) -> bool:
        return bool(self.daily_token_budget) and self.tokens_used() >= self.daily_token_budget

    def close(self) -> None:
        with self._lock:
            self._connection.close()
