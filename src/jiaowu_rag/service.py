from __future__ import annotations

import asyncio
import logging
import time
import uuid

from .agent import ScheduleAgent
from .config import Settings
from .models import (
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    RetrievedCourse,
    SearchFilters,
    SessionMessage,
    TokenUsage,
    ToolCallRecord,
)
from .retriever import ChromaScheduleRetriever
from .usage import UsageLedger


logger = logging.getLogger("jiaowu_rag.query")


def _elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _local_answer(results: list[RetrievedCourse]) -> str:
    if not results:
        return "没有在 ChromaDB 课表向量库中检索到匹配记录。"
    lines = ["ChromaDB 本地检索命中："]
    for item in results:
        lines.append(
            f"- {item.class_no}班，{item.weekday}{item.actual_period}《{item.course_name}》，"
            f"{item.teacher}，{item.location}，{item.weeks} [id:{item.id}]"
        )
    return "\n".join(lines)


def _normalize_ranks(results: list[RetrievedCourse], top_k: int) -> list[RetrievedCourse]:
    normalized = []
    seen: set[int] = set()
    for item in results:
        if item.id in seen:
            continue
        seen.add(item.id)
        normalized.append(item.model_copy(deep=True))
        if len(normalized) >= top_k:
            break
    for rank, item in enumerate(normalized, start=1):
        item.rank = rank
    return normalized


class ToolCallingRAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: ChromaScheduleRetriever,
        assistant: ScheduleAgent | None = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.assistant = assistant
        self.ledger = ledger

    async def _local_query(
        self, request: QueryRequest, top_k: int
    ) -> tuple[list[RetrievedCourse], SearchFilters, int]:
        started = time.perf_counter()
        bundle = await asyncio.to_thread(
            self.retriever.search,
            request.question,
            top_k,
            request.filters,
            True,
        )
        return bundle.results, bundle.filters, _elapsed_ms(started)

    async def query(self, request: QueryRequest) -> QueryResponse:
        total_start = time.perf_counter()
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        warnings: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        tool_loop_ms: int | None = None
        local_vector_ms: int | None = None
        tool_rounds = 0
        session_id = request.session_id
        usage: TokenUsage | None = None

        if (
            request.use_deepseek
            and self.assistant is not None
            and self.ledger is not None
            and self.ledger.budget_exhausted()
        ):
            warnings.append("今日模型调用额度已用完，已降级为 ChromaDB 本地检索。")
        elif request.use_deepseek and self.assistant is not None:
            session_id = session_id or uuid.uuid4().hex
            usage = TokenUsage()
            tool_start = time.perf_counter()
            try:
                # Tools get the full row budget so one class's day is never cut off;
                # top_k only limits the course list returned to the client.
                outcome = await self.assistant.run(
                    request.question,
                    self.settings.max_top_k,
                    request.filters,
                    session_id,
                    usage,
                )
                tool_loop_ms = _elapsed_ms(tool_start)
                tool_calls = outcome.calls
                if not tool_calls:
                    warnings.append("这条回答没有查询课表数据；涉及具体课程安排时请以查询结果为准。")
                tool_rounds = outcome.rounds
                results = _normalize_ranks(outcome.courses, top_k)
                response = QueryResponse(
                    question=request.question,
                    session_id=session_id,
                    answer=outcome.answer,
                    mode="tool_calling",
                    deepseek_model=self.assistant.model_name,
                    applied_filters=request.filters,
                    results=results,
                    tool_calls=tool_calls,
                    warnings=warnings,
                    diagnostics=QueryDiagnostics(
                        total_ms=_elapsed_ms(total_start),
                        tool_loop_ms=tool_loop_ms,
                        tool_rounds=tool_rounds,
                        token_usage=usage,
                    ),
                )
                self._log(response)
                return response
            except Exception as exc:
                tool_loop_ms = _elapsed_ms(tool_start)
                logger.warning("agent failed session=%s error=%r", session_id, exc)
                warnings.append(
                    f"LangChain 工具调用循环失败，已降级为 ChromaDB 本地检索：{type(exc).__name__}"
                    "（本轮回答未写入会话历史）"
                )
            finally:
                if self.ledger is not None and usage.model_calls:
                    self.ledger.record(usage)
        elif request.use_deepseek:
            warnings.append("大模型未配置，已降级为 ChromaDB 本地检索。")

        results, applied_filters, local_vector_ms = await self._local_query(request, top_k)
        results = _normalize_ranks(results, top_k)
        response = QueryResponse(
            question=request.question,
            session_id=session_id,
            answer=_local_answer(results),
            mode="local_only",
            deepseek_model=None,
            applied_filters=applied_filters,
            results=results,
            tool_calls=tool_calls,
            warnings=warnings,
            diagnostics=QueryDiagnostics(
                total_ms=_elapsed_ms(total_start),
                local_vector_ms=local_vector_ms,
                tool_loop_ms=tool_loop_ms,
                tool_rounds=tool_rounds,
                token_usage=usage,
            ),
        )
        self._log(response)
        return response

    @staticmethod
    def _log(response: QueryResponse) -> None:
        usage = response.diagnostics.token_usage or TokenUsage()
        logger.info(
            "query session=%s mode=%s tools=%d rounds=%d model_calls=%d "
            "input=%d cached=%d output=%d ms=%d",
            response.session_id,
            response.mode,
            len(response.tool_calls),
            response.diagnostics.tool_rounds,
            usage.model_calls,
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            response.diagnostics.total_ms,
        )

    def _require_sessions(self) -> ScheduleAgent:
        if self.assistant is None:
            raise LookupError("Sessions require a configured LLM agent")
        return self.assistant

    async def get_session_history(self, session_id: str) -> list[SessionMessage] | None:
        return await self._require_sessions().get_history(session_id)

    async def delete_session(self, session_id: str) -> None:
        await self._require_sessions().delete_session(session_id)
