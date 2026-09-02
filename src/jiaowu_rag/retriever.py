from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

import chromadb
import numpy as np
from fastembed import TextEmbedding

from .models import RetrievedCourse, SearchBundle, SearchFilters


WEEKDAY_ALIASES = {
    "星期一": "星期一", "周一": "星期一",
    "星期二": "星期二", "周二": "星期二",
    "星期三": "星期三", "周三": "星期三",
    "星期四": "星期四", "周四": "星期四",
    "星期五": "星期五", "周五": "星期五",
    "星期六": "星期六", "周六": "星期六",
    "星期日": "星期日", "星期天": "星期日", "周日": "星期日", "周天": "星期日",
}
PERIOD_ALIASES = {
    "第一二节": "1-2节", "一二节": "1-2节",
    "第三四节": "3-4节", "三四节": "3-4节",
    "第五六节": "5-6节", "五六节": "5-6节",
    "第七八节": "7-8节", "七八节": "7-8节",
    "第九十节": "9-10节", "九十节": "9-10节",
    "第十一十二节": "11-12节", "十一十二节": "11-12节",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChromaScheduleRetriever:
    """Persistent ChromaDB retrieval with SQLite-backed schedule records."""

    def __init__(
        self,
        project_root: Path,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        model_cache: Path | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.persist_dir = (persist_dir or self.project_root / "output" / "chroma_db").resolve()
        self.model_cache = (model_cache or self.project_root / "output" / "model_cache").resolve()
        manifest_path = self.persist_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Chroma manifest not found: {manifest_path}. Run build_vector_index.py first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("storage") != "chromadb":
            raise ValueError("The configured index is not a ChromaDB index; rebuild it")
        self.collection_name = collection_name or str(self.manifest["collection"])
        if self.collection_name != self.manifest["collection"]:
            raise ValueError(
                f"Chroma collection mismatch: configured={self.collection_name}, "
                f"manifest={self.manifest['collection']}"
            )
        self.db_path = (self.project_root / self.manifest["database"]).resolve()
        self.corpus_path = (self.project_root / self.manifest["corpus"]).resolve()

        if verify_hashes:
            if _sha256_file(self.db_path) != self.manifest["database_sha256"]:
                raise ValueError("SQLite database changed after vectorization; rebuild the Chroma index")
            if _sha256_file(self.corpus_path) != self.manifest["corpus_sha256"]:
                raise ValueError("RAG corpus changed after vectorization; rebuild the Chroma index")

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=None,
        )
        if self.collection.count() != int(self.manifest["count"]):
            raise ValueError(
                f"Chroma collection count does not match manifest: "
                f"collection={self.collection.count()}, manifest={self.manifest['count']}"
            )
        self.model = TextEmbedding(
            model_name=self.manifest["model"],
            cache_dir=str(self.model_cache),
            local_files_only=True,
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def infer_filters(self, query: str, connection: sqlite3.Connection) -> SearchFilters:
        class_no = None
        for candidate in re.findall(r"(?<!\d)(\d{6,8})(?!\d)", query):
            exists = connection.execute(
                "SELECT 1 FROM courses WHERE class_no = ? LIMIT 1", (candidate,)
            ).fetchone()
            if exists:
                class_no = candidate
                break

        weekday = next((canonical for alias, canonical in WEEKDAY_ALIASES.items() if alias in query), None)
        numeric_period = re.search(r"(?<!\d)(\d{1,2})\s*[-到至~～]\s*(\d{1,2})\s*节", query)
        if numeric_period:
            period = f"{int(numeric_period.group(1))}-{int(numeric_period.group(2))}节"
        else:
            period = next((canonical for alias, canonical in PERIOD_ALIASES.items() if alias in query), None)
        daytime = next((value for value in ("上午", "下午", "晚上") if value in query), None)
        return SearchFilters(class_no=class_no, weekday=weekday, period=period, daytime=daytime)

    @staticmethod
    def _where(filters: SearchFilters) -> dict[str, Any] | None:
        metadata_fields = {
            "grade": "grade",
            "class_no": "class_no",
            "weekday": "weekday",
            "period": "actual_period",
            "daytime": "daytime",
            "record_type": "record_type",
            "course_name": "course_name",
        }
        conditions = [
            {metadata_fields[field]: {"$eq": value}}
            for field, value in filters.without_none().items()
            if field in metadata_fields
        ]
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _fetch_records(
        connection: sqlite3.Connection, ids: list[int]
    ) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
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

    def fetch_courses(
        self,
        ids: list[int],
        lane: Literal["vector", "sql"],
    ) -> list[RetrievedCourse]:
        unique_ids = list(dict.fromkeys(int(value) for value in ids))
        if not unique_ids:
            return []
        stored = self.collection.get(ids=[str(value) for value in unique_ids], include=["documents"])
        document_by_id = {
            int(course_id): document
            for course_id, document in zip(stored["ids"], stored.get("documents") or [])
        }
        connection = self._connect()
        try:
            records = self._fetch_records(connection, unique_ids)
        finally:
            connection.close()
        courses = []
        for rank, course_id in enumerate(unique_ids, start=1):
            record = records.get(course_id)
            document = document_by_id.get(course_id)
            if record is None or document is None:
                continue
            courses.append(
                RetrievedCourse(
                    rank=rank,
                    score=0.0,
                    retrieval_lanes=[lane],
                    document=document,
                    **record,
                )
            )
        return courses

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        auto_filter: bool = True,
    ) -> SearchBundle:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        explicit = filters or SearchFilters()
        connection = self._connect()
        try:
            inferred = self.infer_filters(query, connection) if auto_filter else SearchFilters()
            applied = explicit.merged_over(inferred)
        finally:
            connection.close()

        query_vector = np.asarray(list(self.model.query_embed(query))[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            raise ValueError("Embedding model produced a zero-length query vector")
        query_vector /= norm

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector.tolist()],
            "n_results": min(top_k, self.count),
            "include": ["documents", "distances"],
        }
        where = self._where(applied)
        if where is not None:
            kwargs["where"] = where
        raw = self.collection.query(**kwargs)
        ids = raw["ids"][0] if raw.get("ids") else []
        documents = (raw.get("documents") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        result_ids = [int(value) for value in ids]

        connection = self._connect()
        try:
            records = self._fetch_records(connection, result_ids)
        finally:
            connection.close()
        results = []
        for rank, (course_id, document, distance) in enumerate(
            zip(result_ids, documents, distances), start=1
        ):
            record = records.get(course_id)
            if record is None or document is None:
                continue
            results.append(
                RetrievedCourse(
                    rank=rank,
                    score=round(1.0 - float(distance), 6),
                    retrieval_lanes=["vector"],
                    document=document,
                    **record,
                )
            )
        return SearchBundle(query=query, filters=applied, results=results)
