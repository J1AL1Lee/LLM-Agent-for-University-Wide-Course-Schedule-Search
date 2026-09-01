from __future__ import annotations

import json
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from jiaowu_rag.api import create_app
from jiaowu_rag.config import Settings
from jiaowu_rag.deepseek import LangChainDeepSeekAssistant
from jiaowu_rag.models import QueryPlan, QueryRequest, SearchFilters
from jiaowu_rag.retriever import LocalScheduleRetriever
from jiaowu_rag.service import DualLaneRAGService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeAssistant:
    model_name = "fake-deepseek"

    async def plan(self, question: str) -> QueryPlan:
        return QueryPlan(
            rewritten_query="230101班 星期二 1-2节 课程",
            filters=SearchFilters(class_no="230101", weekday="星期二", period="1-2节"),
        )

    async def answer(self, question: str, results: list) -> str:
        return f"找到 {len(results)} 条有依据的课程记录 [id:{results[0].id}]"


class RAGBackendTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = LocalScheduleRetriever(PROJECT_ROOT)
        cls.settings = Settings(
            project_root=PROJECT_ROOT,
            deepseek_api_key=None,
            default_top_k=3,
            max_top_k=10,
        )

    def test_local_retrieval_uses_exact_filters(self) -> None:
        result = self.retriever.search("230101班星期二第一二节有什么课", top_k=3)
        self.assertEqual(result.filters.class_no, "230101")
        self.assertEqual(result.filters.weekday, "星期二")
        self.assertEqual(result.filters.period, "1-2节")
        self.assertEqual(len(result.results), 3)
        self.assertTrue(all(item.class_no == "230101" for item in result.results))

    async def test_service_falls_back_without_deepseek(self) -> None:
        service = DualLaneRAGService(self.settings, self.retriever, assistant=None)
        response = await service.query(QueryRequest(question="230101班星期二第一二节有什么课"))
        self.assertEqual(response.mode, "local_only")
        self.assertTrue(response.results)
        self.assertTrue(response.warnings)
        self.assertIn("本地知识库命中", response.answer)

    async def test_service_fuses_direct_and_deepseek_lanes(self) -> None:
        service = DualLaneRAGService(self.settings, self.retriever, assistant=FakeAssistant())
        response = await service.query(QueryRequest(question="230101班星期二第一二节有什么课"))
        self.assertEqual(response.mode, "dual")
        self.assertEqual(response.deepseek_model, "fake-deepseek")
        self.assertEqual(response.rewritten_query, "230101班 星期二 1-2节 课程")
        self.assertTrue(response.results)
        self.assertIn("direct", response.results[0].retrieval_lanes)
        self.assertIn("deepseek", response.results[0].retrieval_lanes)

    async def test_langchain_deepseek_pipeline_uses_official_api_shape(self) -> None:
        settings = Settings(project_root=PROJECT_ROOT, deepseek_api_key="test-key")
        assistant = LangChainDeepSeekAssistant(settings)
        await assistant.aclose()

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/chat/completions")
            self.assertEqual(payload["model"], "deepseek-v4-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            if payload.get("response_format"):
                content = json.dumps(
                    {
                        "rewritten_query": "230101班 星期二 1-2节 课程",
                        "filters": {"class_no": "230101", "weekday": "星期二", "period": "1-2节"},
                    },
                    ensure_ascii=False,
                )
            else:
                content = "机器人基础在1教428 [id:4]"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": content}}]},
            )

        assistant._client = httpx.AsyncClient(  # noqa: SLF001 - transport injection for contract test
            base_url="https://api.deepseek.com",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer test-key"},
        )
        try:
            plan = await assistant.plan("230101班周二第一二节是什么课")
            self.assertEqual(plan.filters.period, "1-2节")
            result = self.retriever.search("230101班星期二第一二节有什么课", top_k=1)
            answer = await assistant.answer("在哪里上课", result.results)
            self.assertIn("[id:4]", answer)
        finally:
            await assistant.aclose()

    def test_fastapi_health_and_local_query(self) -> None:
        app = create_app(settings=self.settings, retriever=self.retriever)
        with TestClient(app) as client:
            root = client.get("/", follow_redirects=False)
            self.assertEqual(root.status_code, 307)
            self.assertEqual(root.headers["location"], "/docs")
            self.assertEqual(client.get("/favicon.ico").status_code, 204)
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
