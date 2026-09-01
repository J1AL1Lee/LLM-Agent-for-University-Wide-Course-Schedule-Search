from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

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


class LocalScheduleRetriever:
    """Load the existing dense index once and provide filtered cosine search."""

    def __init__(
        self,
        project_root: Path,
        index_dir: Path | None = None,
        model_cache: Path | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.index_dir = (index_dir or self.project_root / "output" / "vector_store").resolve()
        self.model_cache = (model_cache or self.project_root / "output" / "model_cache").resolve()
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Vector index not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.db_path = (self.project_root / self.manifest["database"]).resolve()
        self.corpus_path = (self.project_root / self.manifest["corpus"]).resolve()

        if verify_hashes:
            if _sha256_file(self.db_path) != self.manifest["database_sha256"]:
                raise ValueError("SQLite database changed after vectorization; rebuild the vector index")
            if _sha256_file(self.corpus_path) != self.manifest["corpus_sha256"]:
                raise ValueError("RAG corpus changed after vectorization; rebuild the vector index")

        self.vectors = np.load(
            self.index_dir / self.manifest["embeddings_file"], mmap_mode="r", allow_pickle=False
        )
        self.vector_ids = np.load(
            self.index_dir / self.manifest["ids_file"], mmap_mode="r", allow_pickle=False
        )
        self.documents = self.corpus_path.read_text(encoding="utf-8").splitlines()
        expected_shape = (self.manifest["count"], self.manifest["dimensions"])
        if self.vectors.shape != expected_shape:
            raise ValueError(f"Vector file shape does not match manifest: {self.vectors.shape}")
        if len(self.vector_ids) != len(self.documents) or len(self.vector_ids) != len(self.vectors):
            raise ValueError("Vector IDs, documents, and embeddings are misaligned")
        self.model = TextEmbedding(
            model_name=self.manifest["model"],
            cache_dir=str(self.model_cache),
            local_files_only=True,
        )

    @property
    def count(self) -> int:
        return int(self.manifest["count"])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
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
    def _eligible_ids(connection: sqlite3.Connection, filters: SearchFilters) -> np.ndarray | None:
        clauses: list[str] = []
        values: list[str] = []
        data = filters.without_none()
        column_map = {"period": "actual_period"}
        for field in ("grade", "class_no", "weekday", "period", "daytime", "record_type"):
            value = data.get(field)
            if value:
                clauses.append(f"{column_map.get(field, field)} = ?")
                values.append(value)
        if data.get("course_name"):
            clauses.append("course_name LIKE ?")
            values.append(f"%{data['course_name']}%")
        if not clauses:
            return None
        sql = "SELECT id FROM courses WHERE " + " AND ".join(clauses)
        return np.asarray([int(row[0]) for row in connection.execute(sql, values)], dtype=np.int64)

    @staticmethod
    def _fetch_records(connection: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, Any]]:
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
            allowed_ids = self._eligible_ids(connection, applied)
            if allowed_ids is not None and len(allowed_ids) == 0:
                return SearchBundle(query=query, filters=applied, results=[])

            query_vector = np.asarray(list(self.model.query_embed(query))[0], dtype=np.float32)
            norm = float(np.linalg.norm(query_vector))
            if norm == 0:
                raise ValueError("Embedding model produced a zero-length query vector")
            query_vector /= norm
            scores = np.asarray(self.vectors @ query_vector, dtype=np.float32)
            if allowed_ids is not None:
                mask = np.isin(self.vector_ids, allowed_ids, assume_unique=False)
                scores[~mask] = -np.inf
                available = int(mask.sum())
            else:
                available = len(scores)
            result_count = min(top_k, available)
            if result_count == 0:
                return SearchBundle(query=query, filters=applied, results=[])
            if result_count == len(scores):
                positions = np.argsort(scores)[::-1]
            else:
                candidates = np.argpartition(scores, -result_count)[-result_count:]
                positions = candidates[np.argsort(scores[candidates])[::-1]]
            positions = positions[:result_count]
            result_ids = [int(self.vector_ids[position]) for position in positions]
            records = self._fetch_records(connection, result_ids)
            results = []
            for rank, position in enumerate(positions, start=1):
                course_id = int(self.vector_ids[position])
                results.append(
                    RetrievedCourse(
                        rank=rank,
                        score=round(float(scores[position]), 6),
                        document=self.documents[position],
                        **records[course_id],
                    )
                )
            return SearchBundle(query=query, filters=applied, results=results)
        finally:
            connection.close()
