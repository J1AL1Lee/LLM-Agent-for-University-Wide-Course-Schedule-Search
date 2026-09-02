from __future__ import annotations

import asyncio
import time

from .config import Settings
from .deepseek import DeepSeekAssistant
from .models import (
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    RetrievedCourse,
    SearchFilters,
    ToolCallRecord,
)
from .retriever import ChromaScheduleRetriever
from .tools import ScheduleToolbox


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
        assistant: DeepSeekAssistant | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.assistant = assistant
        self.toolbox = ScheduleToolbox(retriever, max_results=settings.max_top_k)

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

        if request.use_deepseek and self.assistant is not None:
            tool_start = time.perf_counter()
            try:
                outcome = await self.assistant.run(
                    request.question,
                    self.toolbox,
                    top_k,
                    request.filters,
                )
                tool_loop_ms = _elapsed_ms(tool_start)
                tool_calls = outcome.calls
                tool_rounds = outcome.rounds
                results = _normalize_ranks(outcome.courses, top_k)
                return QueryResponse(
                    question=request.question,
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
                    ),
                )
            except Exception as exc:
                tool_loop_ms = _elapsed_ms(tool_start)
                warnings.append(
                    f"DeepSeek 工具调用循环失败，已降级为 ChromaDB 本地检索：{type(exc).__name__}"
                )
        elif request.use_deepseek:
            warnings.append("DeepSeek 未配置，已降级为 ChromaDB 本地检索。")

        results, applied_filters, local_vector_ms = await self._local_query(request, top_k)
        results = _normalize_ranks(results, top_k)
        return QueryResponse(
            question=request.question,
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
            ),
        )
