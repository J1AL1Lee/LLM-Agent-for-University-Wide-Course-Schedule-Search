from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
import secrets
import smtplib
import sqlite3
import ssl
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from .config import Settings
from .usage import beijing_day


logger = logging.getLogger("jiaowu_rag.auth")

CODE_TTL_SECONDS = 10 * 60
CODE_MAX_ATTEMPTS = 5
CODE_RESEND_SECONDS = 60
CODES_PER_EMAIL_PER_DAY = 5
CODES_PER_IP_PER_DAY = 20
TOKEN_TTL_SECONDS = 30 * 24 * 3600
_EMAIL = re.compile(r"^[a-z0-9._%+-]{1,64}@([a-z0-9-]+(?:\.[a-z0-9-]+)+)$")


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class User:
    id: int
    email: str


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    title: str
    created_at: float
    updated_at: float


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Mailer(Protocol):
    async def send_login_code(self, email: str, code: str) -> None: ...


class LogMailer:
    """Development mailer: writes the code to the server log instead of sending it."""

    async def send_login_code(self, email: str, code: str) -> None:
        logger.warning("DEV MODE (no SMTP configured): login code for %s is %s", email, code)


class SmtpMailer:
    def __init__(self, settings: Settings) -> None:
        if not (settings.smtp_host and settings.smtp_username and settings.smtp_password):
            raise ValueError("SMTP_HOST, SMTP_USERNAME and SMTP_PASSWORD are required")
        if settings.smtp_security not in {"ssl", "starttls"}:
            raise ValueError("SMTP_SECURITY must be ssl or starttls")
        self.settings = settings

    def _send(self, email: str, code: str) -> None:
        settings = self.settings
        message = EmailMessage()
        message["Subject"] = f"课表助手登录验证码 {code}"
        message["From"] = settings.smtp_from or settings.smtp_username
        message["To"] = email
        minutes = CODE_TTL_SECONDS // 60
        message.set_content(
            f"你的登录验证码是：{code}\n\n{minutes} 分钟内有效。如果不是你本人操作，请忽略这封邮件。\n\n"
            f"Your login code is {code}. It expires in {minutes} minutes. "
            "If you did not request it, ignore this email.\n"
        )
        context = ssl.create_default_context()
        timeout = 20
        if settings.smtp_security == "ssl":
            with smtplib.SMTP_SSL(
                settings.smtp_host, settings.smtp_port, context=context, timeout=timeout
            ) as client:
                client.login(settings.smtp_username, settings.smtp_password)
                client.send_message(message)
        else:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=timeout) as client:
                client.starttls(context=context)
                client.login(settings.smtp_username, settings.smtp_password)
                client.send_message(message)

    async def send_login_code(self, email: str, code: str) -> None:
        await asyncio.to_thread(self._send, email, code)


def create_mailer(settings: Settings) -> Mailer:
    if settings.smtp_host:
        return SmtpMailer(settings)
    logger.warning("SMTP_HOST is not set; login codes will be written to the log (development only)")
    return LogMailer()


class AuthStore:
    """Users, emailed login codes, bearer tokens, per-user quotas and session ownership."""

    def __init__(
        self,
        path: Path,
        allowed_domains: tuple[str, ...],
        daily_questions: int,
        questions_per_minute: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not allowed_domains:
            raise ValueError("RAG_ALLOWED_EMAIL_DOMAINS cannot be empty")
        self.allowed_domains = tuple(domain.lower() for domain in allowed_domains)
        self.daily_questions = daily_questions
        self.questions_per_minute = questions_per_minute
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recent_questions: dict[int, deque[float]] = {}
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS login_codes (
                    email TEXT PRIMARY KEY,
                    code_hash TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS code_sends (
                    email TEXT NOT NULL,
                    ip TEXT NOT NULL,
                    sent_at REAL NOT NULL,
                    day TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS code_sends_email ON code_sends (email, day);
                CREATE INDEX IF NOT EXISTS code_sends_ip ON code_sends (ip, day);
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users (id),
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_questions (
                    user_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    questions INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, day)
                );
                CREATE TABLE IF NOT EXISTS session_owners (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS session_owners_user ON session_owners (user_id, updated_at);
                """
            )

    # ---- login -------------------------------------------------------------

    def normalize_email(self, email: str) -> str:
        normalized = email.strip().lower()
        match = _EMAIL.match(normalized)
        if not match:
            raise AuthError(400, "邮箱格式不正确")
        # Exact domain match: "x@evil-bjut.edu.cn" and "x@bjut.edu.cn.evil.com" are rejected.
        if match.group(1) not in self.allowed_domains:
            raise AuthError(400, "只支持以下邮箱注册：" + "、".join(self.allowed_domains))
        return normalized

    def issue_code(self, email: str, ip: str) -> tuple[str, str]:
        """Return (normalized email, code) after enforcing the send limits."""
        email = self.normalize_email(email)
        now = self.clock()
        day = beijing_day(now)
        with self._lock, self._db:
            last = self._db.execute(
                "SELECT MAX(sent_at) FROM code_sends WHERE email = ?", (email,)
            ).fetchone()[0]
            if last is not None and now - last < CODE_RESEND_SECONDS:
                wait = int(CODE_RESEND_SECONDS - (now - last)) + 1
                raise AuthError(429, f"验证码发送太频繁，请 {wait} 秒后再试", wait)
            sent_today = self._db.execute(
                "SELECT COUNT(*) FROM code_sends WHERE email = ? AND day = ?", (email, day)
            ).fetchone()[0]
            if sent_today >= CODES_PER_EMAIL_PER_DAY:
                raise AuthError(429, "该邮箱今天获取验证码的次数已达上限，请明天再试")
            ip_today = self._db.execute(
                "SELECT COUNT(*) FROM code_sends WHERE ip = ? AND day = ?", (ip, day)
            ).fetchone()[0]
            if ip_today >= CODES_PER_IP_PER_DAY:
                raise AuthError(429, "当前网络今天获取验证码的次数已达上限，请明天再试")

            code = f"{secrets.randbelow(1_000_000):06d}"
            self._db.execute(
                """
                INSERT INTO login_codes (email, code_hash, expires_at, attempts) VALUES (?, ?, ?, 0)
                ON CONFLICT(email) DO UPDATE SET
                    code_hash = excluded.code_hash, expires_at = excluded.expires_at, attempts = 0
                """,
                (email, _hash(f"{email}:{code}"), now + CODE_TTL_SECONDS),
            )
            self._db.execute(
                "INSERT INTO code_sends (email, ip, sent_at, day) VALUES (?, ?, ?, ?)",
                (email, ip, now, day),
            )
        return email, code

    def verify_code(self, email: str, code: str) -> tuple[User, str, float]:
        """Return (user, bearer token, expires_at); creates the account on first login."""
        email = self.normalize_email(email)
        now = self.clock()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT code_hash, expires_at, attempts FROM login_codes WHERE email = ?", (email,)
            ).fetchone()
            if row is None or row[1] < now:
                raise AuthError(400, "验证码无效或已过期，请重新获取")
            code_hash, _expires_at, attempts = row
            if attempts >= CODE_MAX_ATTEMPTS:
                raise AuthError(429, "验证码错误次数过多，请重新获取")
            correct = hmac.compare_digest(code_hash, _hash(f"{email}:{code.strip()}"))
            if not correct:
                self._db.execute(
                    "UPDATE login_codes SET attempts = attempts + 1 WHERE email = ?", (email,)
                )
        # Raised outside the transaction so the failed attempt is committed, not rolled back.
        if not correct:
            raise AuthError(400, "验证码错误")
        with self._lock, self._db:
            deleted = self._db.execute(
                "DELETE FROM login_codes WHERE email = ? AND code_hash = ?", (email, code_hash)
            ).rowcount
            if not deleted:
                raise AuthError(400, "验证码无效或已过期，请重新获取")
            self._db.execute(
                "INSERT OR IGNORE INTO users (email, created_at) VALUES (?, ?)", (email, now)
            )
            user_id, disabled = self._db.execute(
                "SELECT id, disabled FROM users WHERE email = ?", (email,)
            ).fetchone()
            if disabled:
                raise AuthError(403, "该账号已被停用")
            token = secrets.token_urlsafe(32)
            expires_at = now + TOKEN_TTL_SECONDS
            self._db.execute(
                "INSERT INTO tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (_hash(token), user_id, now, expires_at),
            )
        return User(id=user_id, email=email), token, expires_at

    def user_for_token(self, token: str) -> User | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT users.id, users.email FROM tokens JOIN users ON users.id = tokens.user_id
                WHERE tokens.token_hash = ? AND tokens.expires_at > ? AND users.disabled = 0
                """,
                (_hash(token), self.clock()),
            ).fetchone()
        return User(id=row[0], email=row[1]) if row else None

    def revoke_token(self, token: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM tokens WHERE token_hash = ?", (_hash(token),))

    # ---- quotas ------------------------------------------------------------

    def consume_question(self, user: User) -> None:
        now = self.clock()
        day = beijing_day(now)
        with self._lock, self._db:
            recent = self._recent_questions.setdefault(user.id, deque())
            while recent and now - recent[0] >= 60:
                recent.popleft()
            if self.questions_per_minute and len(recent) >= self.questions_per_minute:
                wait = int(60 - (now - recent[0])) + 1
                raise AuthError(429, f"提问太快了，请 {wait} 秒后再试", wait)
            used = self.questions_today(user, _locked=True, day=day)
            if self.daily_questions and used >= self.daily_questions:
                raise AuthError(429, f"今天的 {self.daily_questions} 次提问额度已用完，明天再来吧")
            recent.append(now)
            self._db.execute(
                """
                INSERT INTO user_questions (user_id, day, questions) VALUES (?, ?, 1)
                ON CONFLICT(user_id, day) DO UPDATE SET questions = questions + 1
                """,
                (user.id, day),
            )

    def questions_today(self, user: User, _locked: bool = False, day: str | None = None) -> int:
        query = "SELECT questions FROM user_questions WHERE user_id = ? AND day = ?"
        params = (user.id, day or beijing_day(self.clock()))
        if _locked:
            row = self._db.execute(query, params).fetchone()
        else:
            with self._lock:
                row = self._db.execute(query, params).fetchone()
        return int(row[0]) if row else 0

    # ---- session ownership -------------------------------------------------

    def claim_session(self, session_id: str, user: User, title: str) -> bool:
        """Attach a session to its first user; False if another user owns it."""
        now = self.clock()
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT OR IGNORE INTO session_owners (session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user.id, title[:60], now, now),
            )
            owner = self._db.execute(
                "SELECT user_id FROM session_owners WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            if owner != user.id:
                return False
            self._db.execute(
                "UPDATE session_owners SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )
        return True

    def owns_session(self, session_id: str, user: User) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM session_owners WHERE session_id = ? AND user_id = ?",
                (session_id, user.id),
            ).fetchone()
        return row is not None

    def list_sessions(self, user: User, limit: int = 50) -> list[SessionSummary]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT session_id, title, created_at, updated_at FROM session_owners
                WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?
                """,
                (user.id, limit),
            ).fetchall()
        return [SessionSummary(*row) for row in rows]

    def forget_session(self, session_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM session_owners WHERE session_id = ?", (session_id,))

    def close(self) -> None:
        with self._lock:
            self._db.close()
