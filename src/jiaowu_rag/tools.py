from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .models import RetrievedCourse, SearchFilters
from .retriever import ChromaScheduleRetriever


SQL_TOOL_NAME = "query_schedule_sql"
VECTOR_TOOL_NAME = "search_schedule_vectors"
_FORBIDDEN_SQL = re.compile(
    r"\b(attach|detach|pragma|insert|update|delete|replace|create|drop|alter|"
    r"vacuum|reindex|analyze|begin|commit|rollback|savepoint|release)\b",
    re.IGNORECASE,
)
_TABLE_REFERENCE = re.compile(r"\b(?:from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)", re.IGNORECASE)
_ALLOWED_SQL_FUNCTIONS = {
    "abs",
    "avg",
    "coalesce",
    "count",
    "ifnull",
    "length",
    "like",
    "lower",
    "max",
    "min",
    "nullif",
    "round",
    "sum",
    "total",
    "trim",
    "upper",
}


@dataclass(slots=True)
class ToolResult:
    content: str
    courses: list[RetrievedCourse]
    result_count: int


class ScheduleToolbox:
    def __init__(self, retriever: ChromaScheduleRetriever, max_results: int = 20) -> None:
        self.retriever = retriever
        self.max_results = max(1, max_results)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": SQL_TOOL_NAME,
                    "description": (
                        "在只读 SQLite 课表库上执行 Text-to-SQL。适合精确筛选、计数、分组、"
                        "比较、教师/教室/班级等结构化问题。SQL 必须是单条 SELECT，只能读取 courses 表。"
                        "查询某班课程时必须使用 (class_no LIKE '%班号%' OR target_classes LIKE '%班号%')，"
                        "不得在 course_name 中搜索班号；"
                        "查询教师时按完整姓名精确匹配：(',' || teacher || ',') LIKE '%,姓名,%'"
                        "（teacher 是逗号分隔的多位教师），只有用户给出的姓名不完整时才用 teacher LIKE '%片段%'，"
                        "并在回答中区分返回的不同教师；"
                        "actual_period 使用 1-2节、3-4节等数字规范值。"
                        "合班课程会在每个班的课表中各出现一次，返回课程明细时必须去重："
                        "SELECT MIN(id) AS id, course_name, weekday, actual_period, weeks, location, teacher "
                        "... GROUP BY course_name, weekday, actual_period, weeks, location, teacher。"
                        "结果带 truncated=true 时说明还有未返回的行，必须缩小条件或分组汇总后重查，不能据此断言“没有其他课”。"
                        "courses 字段：id, semester, grade, major, class_no, schedule_label, weekday, "
                        "daytime, period_original, actual_period, course_name, weeks, location, teacher, "
                        "course_code, target_classes, record_type, source_file。返回明细时务必选择 id。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "只访问 courses 表的单条 SQLite SELECT 查询。",
                            },
                            "max_rows": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "最多返回多少行，默认 10。",
                            },
                        },
                        "required": ["sql"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": VECTOR_TOOL_NAME,
                    "description": (
                        "在 ChromaDB 的 BGE 中文向量 collection 中进行语义检索。适合模糊课程描述、"
                        "自然语言近义表达或用户不知道精确字段值的情况；也可附加精确元数据过滤条件。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "用于向量检索的中文查询。"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                            "grade": {"type": "string"},
                            "class_no": {"type": "string"},
                            "weekday": {
                                "type": "string",
                                "enum": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
                            },
                            "period": {"type": "string", "description": "例如 1-2节。"},
                            "daytime": {"type": "string", "enum": ["上午", "下午", "晚上"]},
                            "record_type": {"type": "string", "enum": ["course", "block_placeholder"]},
                            "course_name": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    @staticmethod
    def _validate_sql(sql: str) -> str:
        statement = sql.strip()
        if not statement or len(statement) > 4000:
            raise ValueError("SQL must contain between 1 and 4000 characters")
        if statement.endswith(";"):
            statement = statement[:-1].rstrip()
        if ";" in statement:
            raise ValueError("Only one SQL statement is allowed")
        if not re.match(r"^select\b", statement, re.IGNORECASE):
            raise ValueError("Only SELECT statements are allowed")
        if "--" in statement or "/*" in statement or "*/" in statement:
            raise ValueError("SQL comments are not allowed")
        forbidden = _FORBIDDEN_SQL.search(statement)
        if forbidden:
            raise ValueError(f"Forbidden SQL keyword: {forbidden.group(1)}")
        tables = {
            match.group(1).strip("`\"[]").lower().split(".")[-1]
            for match in _TABLE_REFERENCE.finditer(statement)
        }
        if not tables or tables != {"courses"}:
            raise ValueError("SQL may only read the courses table")
        return statement

    def _execute_sql(self, arguments: dict[str, Any], request_limit: int) -> ToolResult:
        raw_sql = arguments.get("sql")
        if not isinstance(raw_sql, str):
            raise ValueError("sql must be a string")
        sql = self._validate_sql(raw_sql)
        requested = arguments.get("max_rows", 10)
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise ValueError("max_rows must be an integer")
        max_rows = min(max(1, requested), 20, request_limit, self.max_results)
        # Fetch one extra row so the model can be told the result was cut off.
        bounded_sql = f"SELECT * FROM ({sql}) AS _tool_query LIMIT {max_rows + 1}"

        connection = sqlite3.connect(
            f"file:{self.retriever.db_path.as_posix()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        progress_calls = 0

        def authorize(
            action: int,
            arg1: str | None,
            arg2: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and (arg1 or "").lower() == "courses":
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                function_name = (arg2 or arg1 or "").lower()
                if function_name in _ALLOWED_SQL_FUNCTIONS:
                    return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        def stop_expensive_query() -> int:
            nonlocal progress_calls
            progress_calls += 1
            return int(progress_calls > 20_000)

        connection.set_authorizer(authorize)
        connection.set_progress_handler(stop_expensive_query, 1000)
        try:
            rows = [dict(row) for row in connection.execute(bounded_sql).fetchall()]
        finally:
            connection.close()
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        course_ids = []
        for row in rows:
            value = row.get("id", row.get("course_id"))
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                course_ids.append(int(value))
        courses = self.retriever.fetch_courses(course_ids, lane="sql")
        payload: dict[str, Any] = {"sql": sql, "row_count": len(rows), "rows": rows}
        if truncated:
            payload["truncated"] = True
            payload["note"] = (
                f"结果超过 {max_rows} 行已被截断。请用 GROUP BY 去重、增加筛选条件或改用 COUNT 汇总后重新查询。"
            )
        content = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        return ToolResult(content=content, courses=courses, result_count=len(rows))

    def _execute_vector(self, arguments: dict[str, Any], request_limit: int) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        requested = arguments.get("top_k", request_limit)
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise ValueError("top_k must be an integer")
        top_k = min(max(1, requested), request_limit, self.max_results)
        filters = SearchFilters.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key
                in {
                    "grade",
                    "class_no",
                    "weekday",
                    "period",
                    "daytime",
                    "record_type",
                    "course_name",
                }
            }
        )
        bundle = self.retriever.search(
            query=query.strip(),
            top_k=top_k,
            filters=filters,
            auto_filter=False,
        )
        matches = [
            {
                "id": item.id,
                "score": item.score,
                "class_no": item.class_no,
                "weekday": item.weekday,
                "actual_period": item.actual_period,
                "course_name": item.course_name,
                "teacher": item.teacher,
                "location": item.location,
                "weeks": item.weeks,
                "document": item.document,
            }
            for item in bundle.results
        ]
        content = json.dumps(
            {
                "query": bundle.query,
                "filters": bundle.filters.without_none(),
                "match_count": len(matches),
                "matches": matches,
            },
            ensure_ascii=False,
        )
        return ToolResult(content=content, courses=bundle.results, result_count=len(matches))

    def execute(self, name: str, arguments: dict[str, Any], request_limit: int) -> ToolResult:
        effective_limit = min(max(1, request_limit), self.max_results)
        if name == SQL_TOOL_NAME:
            return self._execute_sql(arguments, effective_limit)
        if name == VECTOR_TOOL_NAME:
            return self._execute_vector(arguments, effective_limit)
        raise ValueError(f"Unknown tool: {name}")
