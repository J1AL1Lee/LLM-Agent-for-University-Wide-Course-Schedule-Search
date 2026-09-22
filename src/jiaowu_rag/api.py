from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .agent import LangChainScheduleAgent, ScheduleAgent
from .auth import CODE_TTL_SECONDS, AuthError, AuthStore, Mailer, User, create_mailer
from .config import Settings
from .models import (
    SESSION_ID_PATTERN,
    HealthResponse,
    LoginCodeRequest,
    LoginCodeResponse,
    MeResponse,
    QueryRequest,
    QueryResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionSummaryResponse,
    TokenResponse,
    VerifyCodeRequest,
)
from .retriever import ChromaScheduleRetriever
from .service import ToolCallingRAGService
from .tools import ScheduleToolbox
from .usage import UsageLedger


logger = logging.getLogger("jiaowu_rag.auth")
CHAT_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
_bearer = HTTPBearer(auto_error=False, description="Token from POST /v1/auth/verify")


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
    auth_store: AuthStore | None = None,
    mailer: Mailer | None = None,
) -> FastAPI:
    configured_settings = settings or Settings.from_env()
    auth_enabled = configured_settings.auth_required

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
            and configured_settings.llm_enabled
            and configured_settings.llm_api_key
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
        active_auth = None
        if auth_enabled:
            active_auth = auth_store or AuthStore(
                configured_settings.resolve_path(configured_settings.auth_db),
                configured_settings.allowed_email_domains,
                configured_settings.user_daily_questions,
                configured_settings.user_questions_per_minute,
            )
            app.state.mailer = mailer or create_mailer(configured_settings)
        app.state.auth = active_auth
        try:
            yield
        finally:
            ledger.close()
            if active_auth is not None:
                active_auth.close()
            if active_assistant is not None:
                await active_assistant.aclose()

    app = FastAPI(
        title="BJUT Schedule Tool-Calling RAG API",
        version="1.0.0",
        description=(
            "LangChain agent routing between read-only Text-to-SQL and ChromaDB retrieval, "
            "with conversations persisted per session_id. Log in with a school email code "
            "(POST /v1/auth/request-code, then /v1/auth/verify) and send the token as "
            "`Authorization: Bearer <token>`."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(AuthError)
    async def auth_error_handler(_request: Request, exc: AuthError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=headers)

    async def current_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> User | None:
        """None when authentication is disabled; otherwise a valid user or 401."""
        store: AuthStore | None = request.app.state.auth
        if store is None:
            return None
        user = store.user_for_token(credentials.credentials) if credentials else None
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="请先用学校邮箱登录",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    def require_owned_session(request: Request, session_id: str, user: User | None) -> None:
        store: AuthStore | None = request.app.state.auth
        # Someone else's session looks exactly like a missing one.
        if store is not None and user is not None and not store.owns_session(session_id, user):
            raise HTTPException(status_code=404, detail="Session not found")

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        # no-cache: students get a changed page on their next visit, not a stale copy.
        return FileResponse(CHAT_PAGE, headers={"Cache-Control": "no-cache"})

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

    if auth_enabled:

        @app.post("/v1/auth/request-code", response_model=LoginCodeResponse, status_code=202)
        async def request_code(payload: LoginCodeRequest, request: Request) -> LoginCodeResponse:
            store: AuthStore = request.app.state.auth
            ip = request.client.host if request.client else "unknown"
            email, code = store.issue_code(payload.email, ip)
            try:
                await request.app.state.mailer.send_login_code(email, code)
            except Exception as exc:
                logger.error("sending login code to %s failed: %r", email, exc)
                raise HTTPException(status_code=502, detail="验证码邮件发送失败，请稍后再试") from exc
            return LoginCodeResponse(email=email, expires_in_seconds=CODE_TTL_SECONDS)

        @app.post("/v1/auth/verify", response_model=TokenResponse)
        async def verify_code(payload: VerifyCodeRequest, request: Request) -> TokenResponse:
            store: AuthStore = request.app.state.auth
            user, token, expires_at = store.verify_code(payload.email, payload.code)
            return TokenResponse(token=token, email=user.email, expires_at=expires_at)

        @app.post("/v1/auth/logout", status_code=204)
        async def logout(
            request: Request,
            credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
            _user: User | None = Depends(current_user),
        ) -> Response:
            request.app.state.auth.revoke_token(credentials.credentials)
            return Response(status_code=204)

        @app.get("/v1/me", response_model=MeResponse)
        async def me(request: Request, user: User | None = Depends(current_user)) -> MeResponse:
            store: AuthStore = request.app.state.auth
            return MeResponse(
                email=user.email,
                questions_today=store.questions_today(user),
                daily_question_limit=store.daily_questions,
            )

        @app.get("/v1/sessions", response_model=SessionListResponse)
        async def list_sessions(
            request: Request, user: User | None = Depends(current_user)
        ) -> SessionListResponse:
            store: AuthStore = request.app.state.auth
            return SessionListResponse(
                sessions=[
                    SessionSummaryResponse(
                        session_id=summary.session_id,
                        title=summary.title,
                        created_at=summary.created_at,
                        updated_at=summary.updated_at,
                    )
                    for summary in store.list_sessions(user)
                ]
            )

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(
        payload: QueryRequest, request: Request, user: User | None = Depends(current_user)
    ) -> QueryResponse:
        service: ToolCallingRAGService = request.app.state.rag_service
        store: AuthStore | None = request.app.state.auth
        if store is not None and user is not None:
            if payload.session_id is None:
                payload = payload.model_copy(update={"session_id": uuid.uuid4().hex})
            if not store.claim_session(payload.session_id, user, payload.question):
                raise HTTPException(status_code=404, detail="Session not found")
            store.consume_question(user)
        try:
            return await service.query(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id_param = PathParam(pattern=SESSION_ID_PATTERN)

    @app.get("/v1/sessions/{session_id}", response_model=SessionHistoryResponse)
    async def session_history(
        request: Request,
        session_id: str = session_id_param,
        user: User | None = Depends(current_user),
    ) -> SessionHistoryResponse:
        require_owned_session(request, session_id, user)
        service: ToolCallingRAGService = request.app.state.rag_service
        try:
            messages = await service.get_session_history(session_id)
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if messages is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionHistoryResponse(session_id=session_id, messages=messages)

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(
        request: Request,
        session_id: str = session_id_param,
        user: User | None = Depends(current_user),
    ) -> Response:
        require_owned_session(request, session_id, user)
        service: ToolCallingRAGService = request.app.state.rag_service
        try:
            await service.delete_session(session_id)
        except LookupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if request.app.state.auth is not None:
            request.app.state.auth.forget_session(session_id)
        return Response(status_code=204)

    return app


app = create_app()
