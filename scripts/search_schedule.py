from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding


WEEKDAY_ALIASES = {
    "星期一": "星期一",
    "周一": "星期一",
    "星期二": "星期二",
    "周二": "星期二",
    "星期三": "星期三",
    "周三": "星期三",
    "星期四": "星期四",
    "周四": "星期四",
    "星期五": "星期五",
    "周五": "星期五",
    "星期六": "星期六",
    "周六": "星期六",
    "星期日": "星期日",
    "星期天": "星期日",
    "周日": "星期日",
    "周天": "星期日",
}
PERIOD_ALIASES = {
    "第一二节": "1-2节",
    "一二节": "1-2节",
    "第三四节": "3-4节",
    "三四节": "3-4节",
    "第五六节": "5-6节",
    "五六节": "5-6节",
    "第七八节": "7-8节",
    "七八节": "7-8节",
    "第九十节": "9-10节",
    "九十节": "9-10节",
    "第十一十二节": "11-12节",
    "十一十二节": "11-12节",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_filters(
    query: str, connection: sqlite3.Connection
) -> tuple[str | None, str | None, str | None, str | None]:
    class_no = None
    for candidate in re.findall(r"(?<!\d)(\d{6,8})(?!\d)", query):
        exists = connection.execute("SELECT 1 FROM courses WHERE class_no = ? LIMIT 1", (candidate,)).fetchone()
        if exists:
            class_no = candidate
            break

    weekday = None
    for alias, canonical in WEEKDAY_ALIASES.items():
        if alias in query:
            weekday = canonical
            break

    period = None
    numeric_period = re.search(r"(?<!\d)(\d{1,2})\s*[-到至~～]\s*(\d{1,2})\s*节", query)
    if numeric_period:
        period = f"{int(numeric_period.group(1))}-{int(numeric_period.group(2))}节"
    else:
        for alias, canonical in PERIOD_ALIASES.items():
            if alias in query:
                period = canonical
                break

    daytime = next((value for value in ("上午", "下午", "晚上") if value in query), None)
    return class_no, weekday, period, daytime


def eligible_ids(
    connection: sqlite3.Connection,
    grade: str | None,
    class_no: str | None,
    weekday: str | None,
    period: str | None,
    daytime: str | None,
    record_type: str | None,
    course_name: str | None,
) -> np.ndarray | None:
    clauses: list[str] = []
    values: list[str] = []
    for column, value in (
        ("grade", grade),
        ("class_no", class_no),
        ("weekday", weekday),
        ("actual_period", period),
        ("daytime", daytime),
        ("record_type", record_type),
    ):
        if value:
            clauses.append(f"{column} = ?")
            values.append(value)
    if course_name:
        clauses.append("course_name LIKE ?")
        values.append(f"%{course_name}%")
    if not clauses:
        return None
    sql = "SELECT id FROM courses WHERE " + " AND ".join(clauses)
    return np.asarray([int(row[0]) for row in connection.execute(sql, values)], dtype=np.int64)


def fetch_records(connection: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, Any]]:
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"""
        SELECT id, semester, grade, major, class_no, schedule_label, weekday,
               actual_period, course_name, weeks, location, teacher, course_code,
               target_classes, record_type, source_file
        FROM courses WHERE id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return {int(row["id"]): dict(row) for row in rows}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Search the local schedule vector index.")
    parser.add_argument("query", help="Natural-language schedule question")
    parser.add_argument("--index-dir", type=Path, default=project_root / "output" / "vector_store")
    parser.add_argument("--model-cache", type=Path, default=project_root / "output" / "model_cache")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--grade")
    parser.add_argument("--class-no")
    parser.add_argument("--weekday", choices=sorted(set(WEEKDAY_ALIASES.values())))
    parser.add_argument("--period", help="Canonical period such as 1-2节 or 5-6节")
    parser.add_argument("--daytime", choices=["上午", "下午", "晚上"])
    parser.add_argument("--record-type", choices=["course", "block_placeholder"])
    parser.add_argument("--course-name")
    parser.add_argument("--no-auto-filter", action="store_true", help="Disable class/weekday extraction from the query")
    parser.add_argument("--no-verify", action="store_true", help="Skip database/corpus hash verification")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    project_root = Path(__file__).resolve().parents[1]
    index_dir = args.index_dir.resolve()
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Vector index not found: {manifest_path}. Run build_vector_index.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    db_path = (project_root / manifest["database"]).resolve()
    corpus_path = (project_root / manifest["corpus"]).resolve()

    if not args.no_verify:
        if sha256_file(db_path) != manifest["database_sha256"]:
            raise ValueError("SQLite database changed after vectorization; rebuild the vector index")
        if sha256_file(corpus_path) != manifest["corpus_sha256"]:
            raise ValueError("RAG corpus changed after vectorization; rebuild the vector index")

    vectors = np.load(index_dir / manifest["embeddings_file"], mmap_mode="r", allow_pickle=False)
    vector_ids = np.load(index_dir / manifest["ids_file"], mmap_mode="r", allow_pickle=False)
    documents = corpus_path.read_text(encoding="utf-8").splitlines()
    if vectors.shape != (manifest["count"], manifest["dimensions"]):
        raise ValueError(f"Vector file shape does not match manifest: {vectors.shape}")
    if len(vector_ids) != len(documents) or len(vector_ids) != len(vectors):
        raise ValueError("Vector IDs, documents, and embeddings are misaligned")

    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        inferred_class, inferred_weekday, inferred_period, inferred_daytime = (None, None, None, None)
        if not args.no_auto_filter:
            inferred_class, inferred_weekday, inferred_period, inferred_daytime = infer_filters(args.query, connection)
        class_no = args.class_no or inferred_class
        weekday = args.weekday or inferred_weekday
        period = args.period or inferred_period
        daytime = args.daytime or inferred_daytime
        allowed_ids = eligible_ids(
            connection,
            grade=args.grade,
            class_no=class_no,
            weekday=weekday,
            period=period,
            daytime=daytime,
            record_type=args.record_type,
            course_name=args.course_name,
        )
        if allowed_ids is not None and len(allowed_ids) == 0:
            raise ValueError("No schedule records match the selected filters")

        model = TextEmbedding(
            model_name=manifest["model"],
            cache_dir=str(args.model_cache.resolve()),
            local_files_only=True,
        )
        query_vector = np.asarray(list(model.query_embed(args.query))[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            raise ValueError("Embedding model produced a zero-length query vector")
        query_vector /= norm

        scores = np.asarray(vectors @ query_vector, dtype=np.float32)
        if allowed_ids is not None:
            mask = np.isin(vector_ids, allowed_ids, assume_unique=False)
            scores[~mask] = -np.inf
            available = int(mask.sum())
        else:
            available = len(scores)
        result_count = min(args.top_k, available)
        if result_count == 0:
            raise ValueError("No searchable records remain after filtering")

        if result_count == len(scores):
            positions = np.argsort(scores)[::-1]
        else:
            candidates = np.argpartition(scores, -result_count)[-result_count:]
            positions = candidates[np.argsort(scores[candidates])[::-1]]
        positions = positions[:result_count]
        result_ids = [int(vector_ids[position]) for position in positions]
        records = fetch_records(connection, result_ids)

        results = []
        for rank, position in enumerate(positions, start=1):
            course_id = int(vector_ids[position])
            results.append(
                {
                    "rank": rank,
                    "score": round(float(scores[position]), 6),
                    "document": documents[position],
                    **records[course_id],
                }
            )
    finally:
        connection.close()

    filters = {
        key: value
        for key, value in {
            "grade": args.grade,
            "class_no": class_no,
            "weekday": weekday,
            "period": period,
            "daytime": daytime,
            "record_type": args.record_type,
            "course_name": args.course_name,
        }.items()
        if value
    }
    if args.json:
        print(json.dumps({"query": args.query, "filters": filters, "results": results}, ensure_ascii=False, indent=2))
        return

    if filters:
        print("Filters: " + ", ".join(f"{key}={value}" for key, value in filters.items()))
    for item in results:
        print(
            f"[{item['rank']}] score={item['score']:.4f} id={item['id']} "
            f"{item['class_no']} {item['weekday']} {item['actual_period']} 《{item['course_name']}》"
        )
        print(f"    {item['document']}")


if __name__ == "__main__":
    main()
