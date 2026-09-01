from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from .config import Settings
from .deepseek import LangChainDeepSeekAssistant
from .models import HealthResponse, QueryRequest, QueryResponse
from .retriever import LocalScheduleRetriever
from .service import DualLaneRAGService


def create_app(
    settings: Settings | None = None,
    retriever: LocalScheduleRetriever | None = None,
    assistant: LangChainDeepSeekAssistant | None = None,
) -> FastAPI:
    configured_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_retriever = retriever or LocalScheduleRetriever(configured_settings.project_root)
        active_assistant = assistant
        if (
            active_assistant is None
            and configured_settings.deepseek_enabled
            and configured_settings.deepseek_api_key
        ):
            active_assistant = LangChainDeepSeekAssistant(configured_settings)
        app.state.rag_service = DualLaneRAGService(
            configured_settings, active_retriever, active_assistant
        )
        try:
            yield
        finally:
            if active_assistant is not None:
                await active_assistant.aclose()

    app = FastAPI(
        title="BJUT Schedule Dual-Lane RAG API",
        version="1.0.0",
        description="Local dense retrieval with optional DeepSeek query planning and grounded answers.",
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
        service: DualLaneRAGService = request.app.state.rag_service
        return HealthResponse(
            vector_index_ready=True,
            indexed_records=service.retriever.count,
            deepseek_configured=service.assistant is not None,
            deepseek_model=service.assistant.model_name if service.assistant else None,
        )

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(payload: QueryRequest, request: Request) -> QueryResponse:
        service: DualLaneRAGService = request.app.state.rag_service
        try:
            return await service.query(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
