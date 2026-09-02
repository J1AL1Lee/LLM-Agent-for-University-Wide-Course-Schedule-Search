from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from jiaowu_rag.api import create_app
from jiaowu_rag.config import Settings
from jiaowu_rag.deepseek import DeepSeekToolCallingAssistant
from jiaowu_rag.models import QueryRequest, SearchFilters, ToolCallRecord, ToolLoopResult
from jiaowu_rag.retriever import ChromaScheduleRetriever
from jiaowu_rag.service import ToolCallingRAGService
from jiaowu_rag.tools import SQL_TOOL_NAME, VECTOR_TOOL_NAME, ScheduleToolbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeToolCallingAssistant:
    model_name = "fake-deepseek"

    async def run(
        self,
        question: str,
        toolbox: ScheduleToolbox,
        result_limit: int,
        required_filters: SearchFilters | None = None,
    ) -> ToolLoopResult:
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
        cls.retriever = ChromaScheduleRetriever(PROJECT_ROOT)
        cls.settings = Settings(
            project_root=PROJECT_ROOT,
            deepseek_api_key=None,
            default_top_k=3,
            max_top_k=10,
        )
        cls.toolbox = ScheduleToolbox(cls.retriever, max_results=10)

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
            self.settings, self.retriever, assistant=FakeToolCallingAssistant()
        )
        response = await service.query(QueryRequest(question="230101班星期二第一二节有什么课"))
        self.assertEqual(response.mode, "tool_calling")
        self.assertEqual(response.deepseek_model, "fake-deepseek")
        self.assertEqual(response.tool_calls[0].name, SQL_TOOL_NAME)
        self.assertTrue(response.results)
        self.assertEqual(response.results[0].retrieval_lanes, ["sql"])

    async def test_deepseek_loop_sends_tools_and_tool_results(self) -> None:
        settings = Settings(
            project_root=PROJECT_ROOT,
            deepseek_api_key="test-key",
            max_tool_rounds=3,
        )
        assistant = DeepSeekToolCallingAssistant(settings)
        await assistant.aclose()
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/chat/completions")
            self.assertEqual(payload["model"], "deepseek-v4-flash")
            if request_count == 1:
                self.assertEqual(payload["tool_choice"], "required")
                self.assertEqual(
                    [message["role"] for message in payload["messages"][:4]],
                    ["system", "user", "assistant", "user"],
                )
                self.assertEqual(payload["messages"][1]["content"], "上次查的是230101班。")
                self.assertEqual(payload["messages"][2]["content"], "已记录这个班级。")
                tool_names = {item["function"]["name"] for item in payload["tools"]}
                self.assertEqual(tool_names, {SQL_TOOL_NAME, VECTOR_TOOL_NAME})
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_sql_1",
                                            "type": "function",
                                            "function": {
                                                "name": SQL_TOOL_NAME,
                                                "arguments": json.dumps(
                                                    {
                                                        "sql": (
                                                            "SELECT id, course_name FROM courses "
                                                            "WHERE class_no='230101' ORDER BY id"
                                                        ),
                                                        "max_rows": 2,
                                                    },
                                                    ensure_ascii=False,
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                )
            self.assertEqual(payload["tool_choice"], "auto")
            tool_messages = [item for item in payload["messages"] if item["role"] == "tool"]
            if request_count == 2:
                self.assertEqual(len(tool_messages), 1)
                self.assertEqual(tool_messages[0]["tool_call_id"], "call_sql_1")
                self.assertIn("row_count", tool_messages[0]["content"])
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_vector_2",
                                            "type": "function",
                                            "function": {
                                                "name": VECTOR_TOOL_NAME,
                                                "arguments": json.dumps(
                                                    {"query": "机器人相关课程", "top_k": 2},
                                                    ensure_ascii=False,
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                )
            self.assertEqual(len(tool_messages), 2)
            self.assertEqual(tool_messages[1]["tool_call_id"], "call_vector_2")
            self.assertIn("match_count", tool_messages[1]["content"])
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "SQL 工具返回了两条课程记录 [id:1] [id:2]。",
                            }
                        }
                    ]
                },
            )

        assistant._client = httpx.AsyncClient(  # noqa: SLF001 - transport injection
            base_url="https://api.deepseek.com",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer test-key"},
        )
        try:
            result = await assistant.run(
                "查询它的课程",
                self.toolbox,
                result_limit=2,
                conversation_history=[
                    {"role": "user", "content": "上次查的是230101班。"},
                    {"role": "assistant", "content": "已记录这个班级。"},
                ],
            )
            self.assertEqual(request_count, 3)
            self.assertEqual(result.calls[0].name, SQL_TOOL_NAME)
            self.assertEqual(result.calls[1].name, VECTOR_TOOL_NAME)
            self.assertEqual(result.rounds, 2)
            self.assertEqual(len(result.courses), 2)
        finally:
            await assistant.aclose()

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
