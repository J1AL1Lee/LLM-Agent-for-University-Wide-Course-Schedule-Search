from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from jiaowu_rag.agent import LangChainScheduleAgent, date_note
from jiaowu_rag.api import create_app
from jiaowu_rag.config import Settings
from jiaowu_rag.models import (
    QueryRequest,
    SearchFilters,
    TokenUsage,
    ToolCallRecord,
    ToolLoopResult,
)
from jiaowu_rag.retriever import ChromaScheduleRetriever
from jiaowu_rag.service import ToolCallingRAGService
from jiaowu_rag.tools import SQL_TOOL_NAME, VECTOR_TOOL_NAME, ScheduleToolbox
from jiaowu_rag.usage import UsageLedger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 40},
        },
    )


class ScriptedChatModel(BaseChatModel):
    """Replays scripted AI messages and records what each model call received."""

    script: list[AIMessage]
    seen: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tool_names=[tool.name for tool in tools], tool_choice=tool_choice)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(
            {
                "roles": [message.type for message in messages],
                "messages": [message.text for message in messages],
                "tool_names": kwargs.get("tool_names"),
                "tool_choice": kwargs.get("tool_choice"),
            }
        )
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


class FakeToolCallingAssistant:
    model_name = "fake-deepseek"

    def __init__(self, toolbox: ScheduleToolbox) -> None:
        self.toolbox = toolbox

    async def run(
        self,
        question: str,
        result_limit: int,
        required_filters: SearchFilters | None = None,
        session_id: str | None = None,
        usage: TokenUsage | None = None,
    ) -> ToolLoopResult:
        self.calls = getattr(self, "calls", 0) + 1
        if usage is not None:
            usage.model_calls += 2
            usage.input_tokens += 1000
            usage.output_tokens += 200
        toolbox = self.toolbox
        arguments = {
            "sql": (
                "SELECT id, class_no, weekday, actual_period, course_name, teacher, location "
                "FROM courses WHERE class_no='230101' AND weekday='星期二' "
                "AND actual_period='1-2节' ORDER BY id"
            ),
            "max_rows": result_limit,
        }
        result = toolbox.execute(SQL_TOOL_NAME, arguments, result_limit)
        return ToolLoopResult(
            answer=f"通过 SQL 工具找到 {result.result_count} 条记录 [id:{result.courses[0].id}]",
            courses=result.courses,
            calls=[
                ToolCallRecord(
                    name=SQL_TOOL_NAME,
                    arguments=arguments,
                    result_count=result.result_count,
                )
            ],
            rounds=1,
        )

    async def aclose(self) -> None:
        return None


class RAGBackendTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.retriever = ChromaScheduleRetriever(PROJECT_ROOT)
        cls.settings = Settings(
            project_root=PROJECT_ROOT,
            deepseek_api_key=None,
            default_top_k=3,
            max_top_k=10,
            usage_db=str(Path(cls._tmp.name) / "usage.sqlite"),
        )
        cls.toolbox = ScheduleToolbox(cls.retriever, max_results=10)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _ledger(self, budget: int) -> UsageLedger:
        ledger = UsageLedger(Path(self._tmp.name) / f"ledger-{self.id()}.sqlite", budget)
        self.addCleanup(ledger.close)
        return ledger

    def test_usage_ledger_tracks_daily_budget(self) -> None:
        ledger = self._ledger(budget=1500)
        self.assertFalse(ledger.budget_exhausted())
        ledger.record(TokenUsage(model_calls=2, input_tokens=900, cached_input_tokens=400, output_tokens=100))
        ledger.record(TokenUsage(model_calls=1, input_tokens=400, output_tokens=50), day="2000-01-01")
        self.assertEqual(ledger.tokens_used(), 1000)
        self.assertFalse(ledger.budget_exhausted())
        ledger.record(TokenUsage(model_calls=1, input_tokens=500))
        self.assertTrue(ledger.budget_exhausted())
        self.assertFalse(self._ledger(budget=0).budget_exhausted())

    async def test_service_records_usage_and_stops_at_budget(self) -> None:
        assistant = FakeToolCallingAssistant(self.toolbox)
        ledger = self._ledger(budget=1500)
        service = ToolCallingRAGService(self.settings, self.retriever, assistant, ledger)
        question = QueryRequest(question="230101班星期二第一二节有什么课")

        first = await service.query(question)
        self.assertEqual(first.mode, "tool_calling")
        self.assertEqual(first.diagnostics.token_usage.total_tokens, 1200)
        self.assertEqual(ledger.tokens_used(), 1200)

        await service.query(question)
        self.assertTrue(ledger.budget_exhausted())
        third = await service.query(question)
        self.assertEqual(third.mode, "local_only")
        self.assertIn("额度", third.warnings[0])
        self.assertEqual(assistant.calls, 2)

    def test_chroma_collection_is_loaded(self) -> None:
        self.assertEqual(self.retriever.manifest["storage"], "chromadb")
        self.assertEqual(self.retriever.collection.name, "bjut_schedule")
        self.assertEqual(self.retriever.count, 11115)

    def test_chroma_retrieval_uses_exact_metadata_filters(self) -> None:
        result = self.retriever.search("230101班星期二第一二节有什么课", top_k=3)
        self.assertEqual(result.filters.class_no, "230101")
        self.assertEqual(result.filters.weekday, "星期二")
        self.assertEqual(result.filters.period, "1-2节")
        self.assertEqual(len(result.results), 3)
        self.assertTrue(all(item.class_no == "230101" for item in result.results))
        self.assertTrue(all(item.retrieval_lanes == ["vector"] for item in result.results))

    def test_text_to_sql_tool_executes_read_only_select(self) -> None:
        result = self.toolbox.execute(
            SQL_TOOL_NAME,
            {
                "sql": (
                    "SELECT id, course_name FROM courses "
                    "WHERE class_no='230101' AND weekday='星期二' AND actual_period='1-2节' "
                    "ORDER BY id"
                ),
                "max_rows": 3,
            },
            request_limit=3,
        )
        payload = json.loads(result.content)
        self.assertEqual(payload["row_count"], 3)
        self.assertEqual(len(result.courses), 3)
        self.assertTrue(all(item.retrieval_lanes == ["sql"] for item in result.courses))

    def test_text_to_sql_tool_rejects_writes_and_other_tables(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only SELECT"):
            self.toolbox.execute(SQL_TOOL_NAME, {"sql": "DELETE FROM courses"}, 3)
        with self.assertRaisesRegex(ValueError, "only read the courses table"):
            self.toolbox.execute(SQL_TOOL_NAME, {"sql": "SELECT name FROM sqlite_master"}, 3)
        with self.assertRaises(sqlite3.DatabaseError):
            self.toolbox.execute(
                SQL_TOOL_NAME,
                {"sql": "SELECT id, load_extension('untrusted') FROM courses"},
                3,
            )

    def test_text_to_sql_tool_flags_truncated_results(self) -> None:
        sql = "SELECT id FROM courses WHERE class_no='230101' ORDER BY id"
        truncated = json.loads(self.toolbox.execute(SQL_TOOL_NAME, {"sql": sql, "max_rows": 3}, 3).content)
        self.assertEqual(truncated["row_count"], 3)
        self.assertTrue(truncated["truncated"])
        complete = json.loads(
            self.toolbox.execute(SQL_TOOL_NAME, {"sql": sql + " LIMIT 2", "max_rows": 3}, 3).content
        )
        self.assertNotIn("truncated", complete)

    def test_date_note_resolves_weekday_and_teaching_week(self) -> None:
        monday = date(2026, 9, 21)
        self.assertIn("2026-09-21 星期一", date_note(monday, None))
        self.assertIn("教学周未知", date_note(monday, None))
        self.assertIn("第 3 教学周", date_note(monday, date(2026, 9, 7)))
        self.assertIn("尚未开始", date_note(monday, date(2026, 9, 28)))

    def test_text_to_sql_tool_allows_safe_aggregation(self) -> None:
        result = self.toolbox.execute(
            SQL_TOOL_NAME,
            {"sql": "SELECT COUNT(*) AS total FROM courses WHERE record_type='course'"},
            3,
        )
        self.assertEqual(json.loads(result.content)["rows"][0]["total"], 10566)

    def test_vector_tool_uses_chroma(self) -> None:
        result = self.toolbox.execute(
            VECTOR_TOOL_NAME,
            {"query": "机器人相关课程", "top_k": 2, "record_type": "course"},
            request_limit=2,
        )
        payload = json.loads(result.content)
        self.assertEqual(payload["match_count"], 2)
        self.assertEqual(len(result.courses), 2)

    def test_sql_tool_description_contains_schedule_business_rules(self) -> None:
        sql_definition = next(
            item for item in self.toolbox.definitions if item["function"]["name"] == SQL_TOOL_NAME
        )
        description = sql_definition["function"]["description"]
        self.assertIn("class_no LIKE", description)
        self.assertIn("target_classes LIKE", description)
        self.assertIn("teacher LIKE", description)
        self.assertIn("1-2节", description)

    async def test_service_falls_back_to_chroma_without_deepseek(self) -> None:
        service = ToolCallingRAGService(self.settings, self.retriever, assistant=None)
        response = await service.query(QueryRequest(question="230101班星期二第一二节有什么课"))
        self.assertEqual(response.mode, "local_only")
        self.assertTrue(response.results)
        self.assertTrue(response.warnings)
        self.assertIn("ChromaDB", response.answer)

    async def test_service_uses_tool_calling_result(self) -> None:
        service = ToolCallingRAGService(
            self.settings, self.retriever, assistant=FakeToolCallingAssistant(self.toolbox)
        )
        response = await service.query(QueryRequest(question="230101班星期二第一二节有什么课"))
        self.assertEqual(response.mode, "tool_calling")
        self.assertEqual(response.deepseek_model, "fake-deepseek")
        self.assertEqual(response.tool_calls[0].name, SQL_TOOL_NAME)
        self.assertTrue(response.results)
        self.assertEqual(response.results[0].retrieval_lanes, ["sql"])
        self.assertTrue(response.session_id)

    def _agent(
        self, script: list[AIMessage], max_tool_rounds: int = 3
    ) -> tuple[LangChainScheduleAgent, ScriptedChatModel]:
        settings = Settings(
            project_root=PROJECT_ROOT,
            deepseek_api_key=None,
            max_tool_rounds=max_tool_rounds,
        )
        model = ScriptedChatModel(script=script)
        agent = LangChainScheduleAgent(
            settings, self.toolbox, model, InMemorySaver(), model_name="fake-langchain"
        )
        return agent, model

    async def test_langchain_agent_routes_tools_and_persists_session(self) -> None:
        agent, model = self._agent(
            [
                _tool_call(
                    "call_sql_1",
                    SQL_TOOL_NAME,
                    {
                        "sql": "SELECT id, course_name FROM courses WHERE class_no='230101' ORDER BY id",
                        "max_rows": 2,
                    },
                ),
                _tool_call("call_vector_2", VECTOR_TOOL_NAME, {"query": "机器人相关课程", "top_k": 2}),
                AIMessage(
                    content="SQL 工具返回了两条课程记录 [id:1] [id:2]。",
                    usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                    response_metadata={"token_usage": {"prompt_cache_hit_tokens": 40}},
                ),
                _tool_call(
                    "call_sql_3",
                    SQL_TOOL_NAME,
                    {"sql": "SELECT id FROM courses WHERE class_no='230101' AND weekday='星期三'"},
                ),
                AIMessage(content="周三也有课。"),
            ]
        )
        usage = TokenUsage()
        result = await agent.run(
            "查询230101班的课程", result_limit=2, session_id="session-a1", usage=usage
        )
        self.assertEqual(usage.model_calls, 3)
        self.assertEqual(usage.input_tokens, 300)
        self.assertEqual(usage.cached_input_tokens, 120)
        self.assertEqual(usage.output_tokens, 60)
        self.assertEqual([call.name for call in result.calls], [SQL_TOOL_NAME, VECTOR_TOOL_NAME])
        self.assertTrue(all(call.error is None for call in result.calls))
        self.assertEqual(result.rounds, 2)
        self.assertEqual(len(result.courses), 2)
        self.assertEqual(result.answer, "SQL 工具返回了两条课程记录 [id:1] [id:2]。")

        first, second, third = model.seen[:3]
        self.assertEqual(first["tool_choice"], "required")
        self.assertEqual(set(first["tool_names"]), {SQL_TOOL_NAME, VECTOR_TOOL_NAME})
        self.assertIsNone(second["tool_choice"])
        self.assertIn("row_count", second["messages"][-1])
        self.assertEqual(third["roles"][-2:], ["ai", "tool"])
        self.assertIn("match_count", third["messages"][-1])

        follow_up = await agent.run("那周三呢", result_limit=2, session_id="session-a1")
        self.assertEqual(follow_up.answer, "周三也有课。")
        self.assertEqual(follow_up.rounds, 1)
        # Earlier turns reach the model as question/answer text only.
        self.assertEqual(model.seen[3]["roles"], ["system", "human", "ai", "human"])
        self.assertEqual(model.seen[3]["messages"][2], "SQL 工具返回了两条课程记录 [id:1] [id:2]。")
        self.assertEqual(model.seen[3]["tool_choice"], "required")

        history = await agent.get_history("session-a1")
        self.assertEqual(
            [(item.role, item.content) for item in history],
            [
                ("user", "查询230101班的课程"),
                ("assistant", "SQL 工具返回了两条课程记录 [id:1] [id:2]。"),
                ("user", "那周三呢"),
                ("assistant", "周三也有课。"),
            ],
        )
        self.assertIsNone(await agent.get_history("session-other"))
        await agent.delete_session("session-a1")
        self.assertIsNone(await agent.get_history("session-a1"))

    async def test_langchain_agent_limits_rounds_and_reports_tool_errors(self) -> None:
        agent, model = self._agent(
            [
                _tool_call("call_bad", SQL_TOOL_NAME, {"sql": "DELETE FROM courses"}),
                AIMessage(content="无法执行写操作，证据不足。"),
            ],
            max_tool_rounds=1,
        )
        result = await agent.run(
            "删除课程",
            result_limit=3,
            required_filters=SearchFilters(class_no="230101"),
            session_id="session-b1",
        )
        self.assertEqual(result.rounds, 1)
        self.assertIn("Only SELECT", result.calls[0].error)
        self.assertIn("Only SELECT", model.seen[1]["messages"][-1])
        self.assertIn("230101", model.seen[0]["messages"][0])
        # After the round limit the model is called without tools for the final answer.
        self.assertIsNone(model.seen[1]["tool_names"])

    async def test_langchain_agent_requires_tool_evidence(self) -> None:
        agent, _ = self._agent([AIMessage(content="我猜有课。")])
        with self.assertRaisesRegex(ValueError, "did not call a retrieval tool"):
            await agent.run("230101班有什么课", result_limit=3, session_id="session-c1")

    def test_fastapi_session_endpoints(self) -> None:
        agent, _ = self._agent(
            [
                _tool_call("call_sql_1", SQL_TOOL_NAME, {"sql": "SELECT id FROM courses LIMIT 1"}),
                AIMessage(content="找到一条记录 [id:1]。"),
            ]
        )
        app = create_app(settings=self.settings, retriever=self.retriever, assistant=agent)
        with TestClient(app) as client:
            response = client.post("/v1/query", json={"question": "随便查一条课"})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["mode"], "tool_calling")
            session_id = body["session_id"]
            self.assertRegex(session_id, r"^[0-9a-f]{32}$")

            history = client.get(f"/v1/sessions/{session_id}")
            self.assertEqual(history.status_code, 200)
            self.assertEqual(
                [item["role"] for item in history.json()["messages"]], ["user", "assistant"]
            )
            self.assertEqual(client.get("/v1/sessions/unknown-session").status_code, 404)
            self.assertEqual(client.get("/v1/sessions/bad%20id").status_code, 422)
            self.assertEqual(client.delete(f"/v1/sessions/{session_id}").status_code, 204)
            self.assertEqual(client.get(f"/v1/sessions/{session_id}").status_code, 404)
            bad = client.post("/v1/query", json={"question": "x", "session_id": "../etc"})
            self.assertEqual(bad.status_code, 422)

    def test_fastapi_health_and_local_query(self) -> None:
        app = create_app(settings=self.settings, retriever=self.retriever)
        with TestClient(app) as client:
            root = client.get("/", follow_redirects=False)
            self.assertEqual(root.status_code, 307)
            self.assertEqual(root.headers["location"], "/docs")
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["indexed_records"], 11115)
            response = client.post(
                "/v1/query",
                json={
                    "question": "230101班星期二第一二节有什么课",
                    "top_k": 2,
                    "use_deepseek": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "local_only")
            self.assertEqual(len(response.json()["results"]), 2)


if __name__ == "__main__":
    unittest.main()
