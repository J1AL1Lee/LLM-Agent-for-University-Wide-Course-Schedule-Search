from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Path as PathParam, Request, Response
from fastapi.responses import RedirectResponse

from .agent import LangChainScheduleAgent, ScheduleAgent
from .config import Settings
from .models import (
    SESSION_ID_PATTERN,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionHistoryResponse,
)
from .retriever import ChromaScheduleRetriever
from .service import ToolCallingRAGService
from .tools import ScheduleToolbox
from .usage import UsageLedger


def _configure_logging() -> None:
    package_logger = logging.getLogger("jiaowu_rag")
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.INFO)


def create_app(
    settings: Settings | None = None,
    retriever: ChromaScheduleRetriever | None = None,
    assistant: ScheduleAgent | None = None,
) -> FastAPI:
    configured_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _configure_logging()
        chroma_path = Path(configured_settings.chroma_dir)
        if not chroma_path.is_absolute():
            chroma_path = configured_settings.project_root / chroma_path
        active_retriever = retriever or ChromaScheduleRetriever(
            configured_settings.project_root,
            persist_dir=chroma_path,
            collection_name=configured_settings.chroma_collection,
        )
        active_assistant = assistant
        if (
            active_assistant is None
            and configured_settings.deepseek_enabled
            and configured_settings.deepseek_api_key
        ):
            active_assistant = await LangChainScheduleAgent.create(
                configured_settings,
                ScheduleToolbox(active_retriever, max_results=configured_settings.max_top_k),
            )
        ledger = UsageLedger(
            configured_settings.resolve_path(configured_settings.usage_db),
            configured_settings.daily_token_budget,
        )
        app.state.rag_service = ToolCallingRAGService(
            configured_settings, active_retriever, active_assistant, ledger
        )
        try:
            yield
        finally:
            ledger.close()
            if active_assistant is not None:
                await active_assistant.aclose()

    app = FastAPI(
        title="BJUT Schedule Tool-Calling RAG API",
        version="1.0.0",
        description=(
            "LangChain agent routing between read-only Text-to-SQL and ChromaDB retrieval, "
            "with conversations persisted per session_id."
        ),
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        service: ToolCallingRAGService = request.app.state.rag_service
        return HealthResponse(
            vector_index_ready=True,
            indexed_records=service.retriever.count,
            deepseek_configured=service.assistant is not None,
            deepseek_model=service.assistant.model_name if service.assistant else None,
            tokens_used_today=service.ledger.tokens_used() if service.ledger else 0,
            daily_token_budget=service.ledger.daily_token_budget if service.ledger else 0,
        )

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(payload: QueryRequest, request: Request) -> QueryResponse:
        service: ToolCallingRAGService = request.app.state.rag_service
        try:
            return await service.query(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id_param = PathParam(pattern=SESSION_ID_PATTERN)

    @app.get("/v1/sessions/{session_id}", response_model=SessionHistoryResponse)
    async def session_history(
        request: Request, session_id: str = session_id_param
    ) -> SessionHistoryResponse:
        service: ToolCallingRAGService = request.app.state.rag_service
        try:
            messages = await service.get_session_history(session_id)
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if messages is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionHistoryResponse(session_id=session_id, messages=messages)

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(request: Request, session_id: str = session_id_param) -> Response:
        service: ToolCallingRAGService = request.app.state.rag_service
        try:
            await service.delete_session(session_id)
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(status_code=204)

    return app


app = create_app()
