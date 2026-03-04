from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    base_url: str = "https://jwglxt.bjut.edu.cn"
    timeout_ms: int = 45_000


def load_settings(base_url_override: str | None = None) -> Settings:
    load_dotenv()
    base_url = (base_url_override or os.getenv("JW_BASE_URL") or "https://jwglxt.bjut.edu.cn").rstrip("/")
    timeout_ms = int(os.getenv("JW_TIMEOUT_MS", "45000"))
    return Settings(base_url=base_url, timeout_ms=timeout_ms)
