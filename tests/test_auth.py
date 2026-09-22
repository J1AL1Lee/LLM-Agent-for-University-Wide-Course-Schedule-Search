from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from jiaowu_rag.api import create_app
from jiaowu_rag.auth import CODE_MAX_ATTEMPTS, AuthError, AuthStore
from jiaowu_rag.config import DEFAULT_EMAIL_DOMAINS, Settings
from jiaowu_rag.models import SearchFilters, SessionMessage, TokenUsage, ToolLoopResult
from jiaowu_rag.retriever import ChromaScheduleRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
START = 1_790_000_000.0  # a fixed instant; the tests move the clock by hand


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


class CapturingMailer:
    def __init__(self) -> None:
        self.codes: dict[str, str] = {}
        self.fail = False

    async def send_login_code(self, email: str, code: str) -> None:
        if self.fail:
            raise ConnectionError("smtp down")
        self.codes[email] = code


class EchoAgent:
    """Stands in for the LangChain agent and remembers conversations per session."""

    model_name = "echo"

    def __init__(self) -> None:
        self.sessions: dict[str, list[SessionMessage]] = {}

    async def run(
        self,
        question: str,
        result_limit: int,
        required_filters: SearchFilters | None = None,
        session_id: str | None = None,
        usage: TokenUsage | None = None,
    ) -> ToolLoopResult:
        answer = f"回答：{question}"
        self.sessions.setdefault(session_id, []).extend(
            [SessionMessage(role="user", content=question), SessionMessage(role="assistant", content=answer)]
        )
        return ToolLoopResult(answer=answer)

    async def get_history(self, session_id: str) -> list[SessionMessage] | None:
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def aclose(self) -> None:
        return None


class AuthStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.clock = Clock()
        self.store = AuthStore(
            Path(self._tmp.name) / "auth.sqlite", DEFAULT_EMAIL_DOMAINS, 30, 5, clock=self.clock
        )
        self.addCleanup(self.store.close)

    def assertAuthError(self, status: int, func, *args):
        with self.assertRaises(AuthError) as caught:
            func(*args)
        self.assertEqual(caught.exception.status_code, status)
        return caught.exception

    def test_only_exact_allowed_domains_can_sign_up(self) -> None:
        self.assertEqual(self.store.normalize_email(" Student@Emails.BJUT.edu.cn "), "student@emails.bjut.edu.cn")
        self.assertEqual(self.store.normalize_email("t@bjut.edu.cn"), "t@bjut.edu.cn")
        self.assertEqual(self.store.normalize_email("netid@illinois.edu"), "netid@illinois.edu")
        for email in (
            "x@evil-bjut.edu.cn",
            "x@bjut.edu.cn.evil.com",
            "x@gmail.com",
            "x@cs.illinois.edu",
            "not-an-email",
            "a@b@bjut.edu.cn",
        ):
            self.assertAuthError(400, self.store.normalize_email, email)

    def test_code_login_creates_one_account_per_email(self) -> None:
        email, code = self.store.issue_code("s@emails.bjut.edu.cn", "1.1.1.1")
        user, token, expires_at = self.store.verify_code(email, code)
        self.assertEqual(self.store.user_for_token(token), user)
        self.assertGreater(expires_at, START)

        self.clock.now += 120
        _, code = self.store.issue_code("S@emails.bjut.edu.cn", "1.1.1.1")
        again, second_token, _ = self.store.verify_code("s@emails.bjut.edu.cn", code)
        self.assertEqual(again.id, user.id)
        self.assertNotEqual(second_token, token)

        self.store.revoke_token(token)
        self.assertIsNone(self.store.user_for_token(token))
        self.assertEqual(self.store.user_for_token(second_token), user)

    def test_codes_are_single_use_and_expire(self) -> None:
        email, code = self.store.issue_code("s@bjut.edu.cn", "ip")
        self.store.verify_code(email, code)
        self.assertAuthError(400, self.store.verify_code, email, code)

        self.clock.now += 120
        _, code = self.store.issue_code(email, "ip")
        self.clock.now += 10 * 60 + 1
        self.assertAuthError(400, self.store.verify_code, email, code)

    def test_wrong_codes_lock_the_code(self) -> None:
        email, code = self.store.issue_code("s@bjut.edu.cn", "ip")
        wrong = "000000" if code != "000000" else "111111"
        for _ in range(CODE_MAX_ATTEMPTS):
            self.assertAuthError(400, self.store.verify_code, email, wrong)
        self.assertAuthError(429, self.store.verify_code, email, code)

    def test_code_sending_is_rate_limited(self) -> None:
        email = "s@bjut.edu.cn"
        self.store.issue_code(email, "ip")
        error = self.assertAuthError(429, self.store.issue_code, email, "ip")
        self.assertGreater(error.retry_after, 0)
        for _ in range(4):
            self.clock.now += 61
            self.store.issue_code(email, "ip")
        self.clock.now += 61
        self.assertAuthError(429, self.store.issue_code, email, "ip")

        for index in range(20):
            self.store.issue_code(f"u{index}@bjut.edu.cn", "shared-ip")
        self.assertAuthError(429, self.store.issue_code, "one-more@bjut.edu.cn", "shared-ip")

    def test_question_quotas(self) -> None:
        email, code = self.store.issue_code("s@bjut.edu.cn", "ip")
        user, _, _ = self.store.verify_code(email, code)
        for _ in range(5):
            self.store.consume_question(user)
        error = self.assertAuthError(429, self.store.consume_question, user)
        self.assertTrue(1 <= error.retry_after <= 61)
        self.clock.now += 61
        for _ in range(25):
            self.clock.now += 13
            self.store.consume_question(user)
        self.assertEqual(self.store.questions_today(user), 30)
        self.clock.now += 61
        self.assertAuthError(429, self.store.consume_question, user)

    def test_sessions_belong_to_their_first_user(self) -> None:
        users = []
        for name in ("a", "b"):
            email, code = self.store.issue_code(f"{name}@bjut.edu.cn", "ip")
            users.append(self.store.verify_code(email, code)[0])
        alice, bob = users
        self.assertTrue(self.store.claim_session("session-1", alice, "第一个问题"))
        self.assertFalse(self.store.claim_session("session-1", bob, "偷看"))
        self.assertTrue(self.store.owns_session("session-1", alice))
        self.assertFalse(self.store.owns_session("session-1", bob))
        self.assertEqual([s.title for s in self.store.list_sessions(alice)], ["第一个问题"])
        self.assertEqual(self.store.list_sessions(bob), [])


class AuthApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = ChromaScheduleRetriever(PROJECT_ROOT)

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = Settings(
            project_root=PROJECT_ROOT,
            deepseek_api_key=None,
            usage_db=str(Path(tmp.name) / "usage.sqlite"),
        )
        self.clock = Clock()
        store = AuthStore(Path(tmp.name) / "auth.sqlite", DEFAULT_EMAIL_DOMAINS, 30, 5, clock=self.clock)
        self.mailer = CapturingMailer()
        self.agent = EchoAgent()
        app = create_app(settings, self.retriever, self.agent, auth_store=store, mailer=self.mailer)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def login(self, email: str) -> dict[str, str]:
        sent = self.client.post("/v1/auth/request-code", json={"email": email})
        self.assertEqual(sent.status_code, 202, sent.text)
        verified = self.client.post(
            "/v1/auth/verify", json={"email": email, "code": self.mailer.codes[email.lower()]}
        )
        self.assertEqual(verified.status_code, 200, verified.text)
        return {"Authorization": f"Bearer {verified.json()['token']}"}

    def test_login_flow_and_session_ownership(self) -> None:
        self.assertEqual(self.client.post("/v1/query", json={"question": "hi"}).status_code, 401)
        self.assertEqual(
            self.client.post("/v1/query", json={"question": "hi"}, headers={"Authorization": "Bearer nope"}).status_code,
            401,
        )
        alice = self.login("alice@emails.bjut.edu.cn")
        bob = self.login("bob@illinois.edu")

        me = self.client.get("/v1/me", headers=alice).json()
        self.assertEqual(me, {"email": "alice@emails.bjut.edu.cn", "questions_today": 0, "daily_question_limit": 30})

        first = self.client.post("/v1/query", json={"question": "230101班周二有什么课"}, headers=alice)
        self.assertEqual(first.status_code, 200, first.text)
        session_id = first.json()["session_id"]
        follow_up = self.client.post(
            "/v1/query", json={"question": "那周三呢", "session_id": session_id}, headers=alice
        )
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(self.client.get("/v1/me", headers=alice).json()["questions_today"], 2)

        sessions = self.client.get("/v1/sessions", headers=alice).json()["sessions"]
        self.assertEqual([(s["session_id"], s["title"]) for s in sessions], [(session_id, "230101班周二有什么课")])
        history = self.client.get(f"/v1/sessions/{session_id}", headers=alice).json()["messages"]
        self.assertEqual(len(history), 4)

        # Bob cannot read, continue, or delete Alice's conversation, even with its ID.
        self.assertEqual(self.client.get(f"/v1/sessions/{session_id}", headers=bob).status_code, 404)
        self.assertEqual(
            self.client.post("/v1/query", json={"question": "x", "session_id": session_id}, headers=bob).status_code,
            404,
        )
        self.assertEqual(self.client.delete(f"/v1/sessions/{session_id}", headers=bob).status_code, 404)
        self.assertEqual(self.client.get("/v1/sessions", headers=bob).json()["sessions"], [])
        self.assertEqual(len(self.agent.sessions[session_id]), 4)

        self.assertEqual(self.client.delete(f"/v1/sessions/{session_id}", headers=alice).status_code, 204)
        self.assertEqual(self.client.get("/v1/sessions", headers=alice).json()["sessions"], [])

        self.assertEqual(self.client.post("/v1/auth/logout", headers=alice).status_code, 204)
        self.assertEqual(self.client.get("/v1/me", headers=alice).status_code, 401)

    def test_rejects_other_domains_and_reports_rate_limits(self) -> None:
        rejected = self.client.post("/v1/auth/request-code", json={"email": "x@gmail.com"})
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("emails.bjut.edu.cn", rejected.json()["detail"])

        headers = self.login("carol@bjut.edu.cn")
        for index in range(5):
            response = self.client.post("/v1/query", json={"question": f"问题{index}"}, headers=headers)
            self.assertEqual(response.status_code, 200)
        limited = self.client.post("/v1/query", json={"question": "太快了"}, headers=headers)
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)

    def test_mail_failure_is_reported(self) -> None:
        self.mailer.fail = True
        response = self.client.post("/v1/auth/request-code", json={"email": "d@bjut.edu.cn"})
        self.assertEqual(response.status_code, 502)


if __name__ == "__main__":
    unittest.main()
