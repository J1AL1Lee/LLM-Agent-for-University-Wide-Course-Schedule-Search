from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,128}$"

class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grade: str | None = None
    class_no: str | None = None
    weekday: Literal["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"] | None = None
    period: str | None = None
    daytime: Literal["上午", "下午", "晚上"] | None = None
    record_type: Literal["course", "block_placeholder"] | None = None
    course_name: str | None = None

    @field_validator("grade", "class_no", "period", "course_name", mode="before")
    @classmethod
    def blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    def without_none(self) -> dict[str, str]:
        return self.model_dump(exclude_none=True)

    def merged_over(self, fallback: "SearchFilters") -> "SearchFilters":
        return SearchFilters.model_validate({**fallback.without_none(), **self.without_none()})


class RetrievedCourse(BaseModel):
    id: int
    rank: int
    score: float
    retrieval_lanes: list[Literal["vector", "sql"]] = Field(default_factory=list)
    document: str
    semester: str
    grade: str
    major: str
    class_no: str
    schedule_label: str
    weekday: str
    actual_period: str
    course_name: str
    weeks: str
    location: str
    teacher: str
    course_code: str
    target_classes: str
    record_type: str
    source_file: str


class SearchBundle(BaseModel):
    query: str
    filters: SearchFilters
    results: list[RetrievedCourse]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    use_deepseek: bool = True
    filters: SearchFilters = Field(default_factory=SearchFilters)
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Continue an existing conversation; omit to start a new one.",
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class TokenUsage(BaseModel):
    model_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class QueryDiagnostics(BaseModel):
    total_ms: int
    local_vector_ms: int | None = None
    tool_loop_ms: int | None = None
    tool_rounds: int = 0
    token_usage: TokenUsage | None = None


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    result_count: int = 0
    error: str | None = None


class ToolLoopResult(BaseModel):
    answer: str
    courses: list[RetrievedCourse] = Field(default_factory=list)
    calls: list[ToolCallRecord] = Field(default_factory=list)
    rounds: int = 0


class QueryResponse(BaseModel):
    question: str
    session_id: str | None = None
    answer: str
    mode: Literal["tool_calling", "local_only"]
    deepseek_model: str | None = None
    applied_filters: SearchFilters
    results: list[RetrievedCourse]
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: QueryDiagnostics


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    vector_index_ready: bool
    indexed_records: int
    deepseek_configured: bool
    deepseek_model: str | None = None
    tokens_used_today: int = 0
    daily_token_budget: int = 0


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]
