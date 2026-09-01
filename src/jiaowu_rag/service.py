from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from .config import Settings
from .deepseek import DeepSeekAssistant
from .models import (
    QueryDiagnostics,
    QueryPlan,
    QueryRequest,
    QueryResponse,
    RetrievedCourse,
    SearchBundle,
)
from .retriever import LocalScheduleRetriever


def _elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _fuse_results(
    direct: Iterable[RetrievedCourse],
    assisted: Iterable[RetrievedCourse],
    top_k: int,
    rank_constant: int = 60,
) -> list[RetrievedCourse]:
    by_id: dict[int, RetrievedCourse] = {}
    fused: dict[int, float] = {}
    lanes: dict[int, list[str]] = {}
    for lane, items in (("direct", direct), ("deepseek", assisted)):
        for rank, item in enumerate(items, start=1):
            by_id.setdefault(item.id, item.model_copy(deep=True))
            fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (rank_constant + rank)
            lanes.setdefault(item.id, []).append(lane)
    ordered = sorted(by_id.values(), key=lambda item: (-fused[item.id], item.id))[:top_k]
    for rank, item in enumerate(ordered, start=1):
        item.rank = rank
        item.fused_score = round(fused[item.id], 8)
        item.retrieval_lanes = lanes[item.id]  # type: ignore[assignment]
    return ordered


def _local_answer(results: list[RetrievedCourse]) -> str:
    if not results:
        return "没有在本地课表知识库中检索到匹配记录。"
    lines = ["本地知识库命中："]
    for item in results:
        lines.append(
            f"- {item.class_no}班，{item.weekday}{item.actual_period}《{item.course_name}》，"
            f"{item.teacher}，{item.location}，{item.weeks} [id:{item.id}]"
        )
    return "\n".join(lines)


class DualLaneRAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: LocalScheduleRetriever,
        assistant: DeepSeekAssistant | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.assistant = assistant

    async def query(self, request: QueryRequest) -> QueryResponse:
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        warnings: list[str] = []
        plan: QueryPlan | None = None
        direct_start = time.perf_counter()
        direct_task = asyncio.create_task(
            asyncio.to_thread(
                self.retriever.search,
                request.question,
                top_k,
                request.filters,
                True,
            )
        )

        plan_task: asyncio.Task[QueryPlan] | None = None
        plan_start: float | None = None
        if request.use_deepseek and self.assistant is not None:
            plan_start = time.perf_counter()
            plan_task = asyncio.create_task(self.assistant.plan(request.question))
        elif request.use_deepseek:
            warnings.append("DeepSeek 未配置，已降级为本地向量检索。")

        direct_bundle = await direct_task
        direct_ms = _elapsed_ms(direct_start)
        plan_ms: int | None = None
        if plan_task is not None and plan_start is not None:
            try:
                plan = await plan_task
                plan_ms = _elapsed_ms(plan_start)
            except Exception as exc:
                plan_ms = _elapsed_ms(plan_start)
                warnings.append(f"DeepSeek 查询规划失败，已保留本地结果：{type(exc).__name__}")

        assisted_bundle = SearchBundle(query=request.question, filters=direct_bundle.filters, results=[])
        assisted_ms: int | None = None
        if plan is not None:
            assisted_start = time.perf_counter()
            assisted_filters = plan.filters.merged_over(direct_bundle.filters)
            assisted_bundle = await asyncio.to_thread(
                self.retriever.search,
                plan.rewritten_query,
                top_k,
                assisted_filters,
                False,
            )
            assisted_ms = _elapsed_ms(assisted_start)

        results = _fuse_results(direct_bundle.results, assisted_bundle.results, top_k)
        answer = _local_answer(results)
        answer_ms: int | None = None
        answer_used = False
        if plan is not None and self.assistant is not None:
            answer_start = time.perf_counter()
            try:
                generated = await self.assistant.answer(request.question, results)
                if generated:
                    answer = generated
                    answer_used = True
                else:
                    warnings.append("DeepSeek 返回了空答案，已使用本地结果摘要。")
            except Exception as exc:
                warnings.append(f"DeepSeek 答案生成失败，已使用本地结果摘要：{type(exc).__name__}")
            answer_ms = _elapsed_ms(answer_start)

        mode = "dual" if plan is not None else "local_only"
        return QueryResponse(
            question=request.question,
            answer=answer,
            mode=mode,
            deepseek_model=self.assistant.model_name if answer_used and self.assistant else None,
            applied_filters=assisted_bundle.filters if plan is not None else direct_bundle.filters,
            rewritten_query=plan.rewritten_query if plan is not None else None,
            results=results,
            warnings=warnings,
            diagnostics=QueryDiagnostics(
                direct_ms=direct_ms,
                deepseek_plan_ms=plan_ms,
                assisted_retrieval_ms=assisted_ms,
                deepseek_answer_ms=answer_ms,
            ),
        )
