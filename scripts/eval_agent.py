from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
import sqlite3
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jiaowu_rag.agent import LangChainScheduleAgent  # noqa: E402
from jiaowu_rag.config import Settings  # noqa: E402
from jiaowu_rag.models import QueryRequest, QueryResponse, TokenUsage  # noqa: E402
from jiaowu_rag.retriever import ChromaScheduleRetriever  # noqa: E402
from jiaowu_rag.service import ToolCallingRAGService  # noqa: E402
from jiaowu_rag.tools import ScheduleToolbox  # noqa: E402
from jiaowu_rag.usage import beijing_today  # noqa: E402


DEFAULT_CASES = PROJECT_ROOT / "evals" / "schedule_questions.json"
# Default prices per 1M tokens (deepseek-flash peak, USD, 2026-09); pass --price-* for your model.
PRICE_INPUT_MISS = 0.30
PRICE_INPUT_HIT = 0.006
PRICE_OUTPUT = 1.20
WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
NONE_PHRASES = (
    "没有", "未找到", "未查到", "没查到", "查不到", "找不到", "不存在",
    "无课", "无相关", "未检索到", "没有检索到", "无法找到", "0 条", "0条",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run realistic schedule questions through the real agent and grade the answers."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--only", nargs="*", default=None, help="Case ids or categories to run.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--top-k", type=int, default=None, help="Omit to use the server default, like the chat page."
    )
    parser.add_argument("--price-input", type=float, default=PRICE_INPUT_MISS, help="Per 1M uncached input tokens.")
    parser.add_argument("--price-cached", type=float, default=PRICE_INPUT_HIT, help="Per 1M cached input tokens.")
    parser.add_argument("--price-output", type=float, default=PRICE_OUTPUT, help="Per 1M output tokens.")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", text)


def value_in_answer(value: str, answer: str) -> bool:
    candidates = {normalize(value)}
    if value.startswith("星期"):
        day = value[2:]
        candidates |= {f"周{day}", f"礼拜{day}"}
        if day == "日":
            candidates |= {"周天", "星期天"}
    return any(candidate in answer for candidate in candidates)


def number_in_answer(number: int, answer: str) -> bool:
    return re.search(rf"(?<![\d.]){number}(?![\d.])", answer) is not None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grade(
    case: dict[str, Any], response: QueryResponse, gold: list[Any], db_path: Path, db_hash: str
) -> tuple[bool | None, str]:
    if response.mode != "tool_calling":
        return False, f"fell back to local search: {response.warnings}"
    answer = normalize(response.answer)
    check = case["check"]

    if check == "all_in_answer":
        if not gold:
            return False, "gold query returned nothing; fix the case"
        missing = [str(value) for value in gold if not value_in_answer(str(value), answer)]
        wrong = [word for word in case.get("forbidden_in_answer", []) if normalize(word) in answer]
        if missing or wrong:
            return False, f"missing {missing} wrongly included {wrong}"
        return True, f"all {len(gold)} present"

    if check == "number_in_answer":
        expected = int(gold[0])
        found = number_in_answer(expected, answer)
        return found, f"expected {expected}" + ("" if found else " not in answer")

    if check == "keywords_in_answer":
        # gold holds every real course name matching the keywords; count those the answer names.
        hits = sorted(name for name in gold if normalize(name) in answer)
        return len(hits) >= case["min_hits"], f"{len(hits)} matching courses {hits[:6]}"

    if check == "says_none":
        said_none = any(normalize(phrase) in answer for phrase in NONE_PHRASES)
        return said_none, "said none" if said_none else "did not say there were no results"

    if check == "database_unchanged":
        unchanged = sha256(db_path) == db_hash
        leaked = [word for word in case.get("forbidden_in_answer", []) if normalize(word) in answer]
        passed = unchanged and not leaked
        return passed, f"db_unchanged={unchanged} leaked={leaked}"

    if check == "manual":
        return None, case.get("note", "review manually")
    raise ValueError(f"Unknown check: {check}")


def keyword_courses_sql(case: dict[str, Any]) -> tuple[str, list[str]]:
    keywords = case["keywords"]
    sql = "SELECT DISTINCT course_name FROM courses WHERE (" + " OR ".join(
        "course_name LIKE ?" for _ in keywords
    ) + ")"
    params = [f"%{word}%" for word in keywords]
    if case.get("class_scope"):
        sql += " AND (class_no = ? OR target_classes LIKE ?)"
        params += [case["class_scope"], f"%{case['class_scope']}%"]
    return sql, params


PRICES = {"input": PRICE_INPUT_MISS, "cached": PRICE_INPUT_HIT, "output": PRICE_OUTPUT}


def cost_usd(usage: TokenUsage) -> float:
    miss = usage.input_tokens - usage.cached_input_tokens
    return (
        miss * PRICES["input"]
        + usage.cached_input_tokens * PRICES["cached"]
        + usage.output_tokens * PRICES["output"]
    ) / 1_000_000


async def run_case(
    case: dict[str, Any],
    service: ToolCallingRAGService,
    top_k: int,
    db_path: Path,
    db_hash: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        tomorrow = WEEKDAYS[(beijing_today() + timedelta(days=1)).weekday()]
        gold_sql = case.get("gold_sql", "").replace("{tomorrow_weekday}", tomorrow)
        if case["check"] == "keywords_in_answer":
            gold_sql, params = keyword_courses_sql(case)
            gold = [row[0] for row in connection.execute(gold_sql, params)]
        else:
            gold = [row[0] for row in connection.execute(gold_sql)] if gold_sql else []
    finally:
        connection.close()

    turns = case.get("turns") or [case["question"]]
    session_id = f"eval-{case['id']}-{int(time.time())}"
    usage = TokenUsage()
    transcript = []
    async with semaphore:
        started = time.perf_counter()
        for question in turns:
            response = await service.query(
                QueryRequest(question=question, top_k=top_k, session_id=session_id)
            )
            turn_usage = response.diagnostics.token_usage or TokenUsage()
            for field in ("model_calls", "input_tokens", "cached_input_tokens", "output_tokens"):
                setattr(usage, field, getattr(usage, field) + getattr(turn_usage, field))
            transcript.append(
                {
                    "question": question,
                    "answer": response.answer,
                    "mode": response.mode,
                    "warnings": response.warnings,
                    "tool_calls": [call.model_dump() for call in response.tool_calls],
                    "rounds": response.diagnostics.tool_rounds,
                }
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000)

    passed, reason = grade(case, response, gold, db_path, db_hash)
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "reason": reason,
        "gold": gold,
        "turns": transcript,
        "elapsed_ms": elapsed_ms,
        "usage": usage.model_dump(),
        "cost_usd": cost_usd(usage),
    }


async def main_async(args: argparse.Namespace) -> int:
    PRICES.update(input=args.price_input, cached=args.price_cached, output=args.price_output)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case["id"] in wanted or case["category"] in wanted]
    if not cases:
        print("No matching cases.", file=sys.stderr)
        return 2

    output_dir = PROJECT_ROOT / "output" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env(PROJECT_ROOT)
    if not settings.llm_api_key:
        print("LLM_API_KEY is not configured.", file=sys.stderr)
        return 2
    # Keep evaluation sessions out of the service's session store and daily budget.
    settings = dataclasses.replace(
        settings,
        session_db=str(output_dir / "eval_sessions.sqlite"),
        max_top_k=max(settings.max_top_k, args.top_k or 0),
    )
    retriever = ChromaScheduleRetriever(
        PROJECT_ROOT,
        persist_dir=settings.resolve_path(settings.chroma_dir),
        collection_name=settings.chroma_collection,
    )
    agent = await LangChainScheduleAgent.create(
        settings, ScheduleToolbox(retriever, max_results=settings.max_top_k)
    )
    service = ToolCallingRAGService(settings, retriever, agent)
    db_hash = sha256(retriever.db_path)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    print(f"Running {len(cases)} cases with {agent.model_name} ...")
    try:
        results = await asyncio.gather(
            *[
                run_case(case, service, args.top_k, retriever.db_path, db_hash, semaphore)
                for case in cases
            ]
        )
    finally:
        await agent.aclose()

    by_category: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        mark = {True: "PASS", False: "FAIL", None: "REVIEW"}[result["passed"]]
        print(f"[{mark:6}] {result['id']:<26} {result['elapsed_ms'] / 1000:5.1f}s  {result['reason']}")
        if result["passed"] is not None:
            by_category[result["category"]].append(result["passed"])
    for result in results:
        if result["passed"] is None:
            print(f"\n--- {result['id']} (review manually) ---\n{result['turns'][-1]['answer']}")

    graded = [passed for values in by_category.values() for passed in values]
    total_usage = TokenUsage()
    for result in results:
        for field, value in result["usage"].items():
            setattr(total_usage, field, getattr(total_usage, field) + value)
    total_cost = sum(result["cost_usd"] for result in results)
    questions = sum(len(result["turns"]) for result in results)
    summary = {
        "model": agent.model_name,
        "passed": sum(graded),
        "graded": len(graded),
        "by_category": {key: f"{sum(v)}/{len(v)}" for key, v in sorted(by_category.items())},
        "questions": questions,
        "usage": total_usage.model_dump(),
        "estimated_cost": round(total_cost, 4),
        "average_cost_per_question": round(total_cost / questions, 5),
        "average_tokens_per_question": round(total_usage.total_tokens / questions),
        "average_seconds_per_case": round(
            sum(result["elapsed_ms"] for result in results) / len(results) / 1000, 1
        ),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    report = output_dir / f"report-{datetime.now():%Y%m%d-%H%M%S}.json"
    report.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Full report: {report}")
    return 0 if all(graded) else 1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
