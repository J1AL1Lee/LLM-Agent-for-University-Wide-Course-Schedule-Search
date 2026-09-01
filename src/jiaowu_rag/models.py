from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(min_length=1, max_length=500)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class RetrievedCourse(BaseModel):
    id: int
    rank: int
    score: float
    fused_score: float = 0.0
    retrieval_lanes: list[Literal["direct", "deepseek"]] = Field(default_factory=list)
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

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class QueryDiagnostics(BaseModel):
    direct_ms: int
    deepseek_plan_ms: int | None = None
    assisted_retrieval_ms: int | None = None
    deepseek_answer_ms: int | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["dual", "local_only"]
    deepseek_model: str | None = None
    applied_filters: SearchFilters
    rewritten_query: str | None = None
    results: list[RetrievedCourse]
    warnings: list[str] = Field(default_factory=list)
    diagnostics: QueryDiagnostics


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    vector_index_ready: bool
    indexed_records: int
    deepseek_configured: bool
    deepseek_model: str | None = None
