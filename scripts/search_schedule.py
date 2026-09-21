from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jiaowu_rag.models import SearchFilters  # noqa: E402
from jiaowu_rag.retriever import ChromaScheduleRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the persistent ChromaDB schedule collection.")
    parser.add_argument("query", help="Natural-language schedule question")
    parser.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "output" / "chroma_db")
    parser.add_argument("--collection", default="bjut_schedule")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "output" / "model_cache")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--grade")
    parser.add_argument("--class-no")
    parser.add_argument(
        "--weekday",
        choices=["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
    )
    parser.add_argument("--period", help="Canonical period such as 1-2节 or 5-6节")
    parser.add_argument("--daytime", choices=["上午", "下午", "晚上"])
    parser.add_argument("--record-type", choices=["course", "block_placeholder"])
    parser.add_argument("--course-name")
    parser.add_argument("--no-auto-filter", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    retriever = ChromaScheduleRetriever(
        PROJECT_ROOT,
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        model_cache=args.model_cache,
        verify_hashes=not args.no_verify,
    )
    filters = SearchFilters(
        grade=args.grade,
        class_no=args.class_no,
        weekday=args.weekday,
        period=args.period,
        daytime=args.daytime,
        record_type=args.record_type,
        course_name=args.course_name,
    )
    bundle = retriever.search(
        args.query,
        top_k=args.top_k,
        filters=filters,
        auto_filter=not args.no_auto_filter,
    )
    if args.json:
        print(json.dumps(bundle.model_dump(), ensure_ascii=False, indent=2))
        return

    if bundle.filters.without_none():
        print(
            "Filters: "
            + ", ".join(f"{key}={value}" for key, value in bundle.filters.without_none().items())
        )
    for item in bundle.results:
        print(
            f"[{item.rank}] score={item.score:.4f} id={item.id} "
            f"{item.class_no} {item.weekday} {item.actual_period} 《{item.course_name}》"
        )
        print(f"    {item.document}")


if __name__ == "__main__":
    main()
