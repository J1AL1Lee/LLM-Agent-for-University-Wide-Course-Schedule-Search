from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from fastembed import TextEmbedding


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION = "bjut_schedule"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_data(
    db_path: Path, corpus_path: Path
) -> tuple[list[dict[str, Any]], list[str], Counter[str]]:
    documents = corpus_path.read_text(encoding="utf-8").splitlines()
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, grade, class_no, weekday, daytime, actual_period,
                   course_name, teacher, location, record_type
            FROM courses ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    records = [dict(row) for row in rows]
    if len(records) != len(documents):
        raise ValueError(
            f"Database/corpus count mismatch: database={len(records)}, corpus={len(documents)}. "
            "Rebuild the structured corpus before vectorizing."
        )
    if not records:
        raise ValueError("No schedule records found")
    if any(not document.strip() for document in documents):
        raise ValueError("Corpus contains empty documents")
    ids = [int(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Database contains duplicate course IDs")
    record_types = Counter(str(record["record_type"]) for record in records)
    return records, documents, record_types


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model produced a zero-length vector")
    return vectors / norms


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the persistent ChromaDB schedule index.")
    parser.add_argument("--db", type=Path, default=project_root / "class_schedule.db")
    parser.add_argument("--corpus", type=Path, default=project_root / "output" / "schedule_rag_corpus.txt")
    parser.add_argument("--output-dir", type=Path, default=project_root / "output" / "chroma_db")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-cache", type=Path, default=project_root / "output" / "model_cache")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="FastEmbed data-parallel workers; omit to use ONNX Runtime threading.",
    )
    return parser.parse_args()


def _metadata(record: dict[str, Any]) -> dict[str, str | int]:
    return {
        "course_id": int(record["id"]),
        "grade": str(record.get("grade") or ""),
        "class_no": str(record.get("class_no") or ""),
        "weekday": str(record.get("weekday") or ""),
        "daytime": str(record.get("daytime") or ""),
        "actual_period": str(record.get("actual_period") or ""),
        "course_name": str(record.get("course_name") or ""),
        "teacher": str(record.get("teacher") or ""),
        "location": str(record.get("location") or ""),
        "record_type": str(record.get("record_type") or ""),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    db_path = args.db.resolve()
    corpus_path = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    model_cache = args.model_cache.resolve()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"RAG corpus not found: {corpus_path}")

    records, documents, record_types = load_source_data(db_path, corpus_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {args.model}")
    model = TextEmbedding(
        model_name=args.model,
        cache_dir=str(model_cache),
        local_files_only=True,
    )
    client = chromadb.PersistentClient(path=str(output_dir))
    existing = {item.name for item in client.list_collections()}
    if args.collection in existing:
        print(f"Replacing Chroma collection: {args.collection}")
        client.delete_collection(args.collection)
    collection = client.create_collection(
        name=args.collection,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": args.model,
            "description": "BJUT class schedule records",
        },
    )

    print(f"Embedding and inserting documents into ChromaDB: {len(documents)}")
    dimensions: int | None = None
    for start in range(0, len(documents), args.batch_size):
        end = min(start + args.batch_size, len(documents))
        batch_documents = documents[start:end]
        vectors = np.asarray(
            list(
                model.passage_embed(
                    batch_documents,
                    batch_size=args.batch_size,
                    parallel=args.parallel,
                )
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(batch_documents):
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values")
        vectors = normalize_rows(vectors).astype(np.float32, copy=False)
        dimensions = dimensions or int(vectors.shape[1])
        if vectors.shape[1] != dimensions:
            raise ValueError("Embedding dimensions changed while building the collection")

        batch_records = records[start:end]
        collection.add(
            ids=[str(record["id"]) for record in batch_records],
            embeddings=vectors.tolist(),
            documents=batch_documents,
            metadatas=[_metadata(record) for record in batch_records],
        )
        print(f"Inserted {end}/{len(documents)}")

    if collection.count() != len(documents) or dimensions is None:
        raise ValueError(
            f"Chroma collection count mismatch: expected={len(documents)}, actual={collection.count()}"
        )

    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "format_version": 2,
        "storage": "chromadb",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": args.collection,
        "model": args.model,
        "dimensions": dimensions,
        "count": collection.count(),
        "normalized": True,
        "similarity": "cosine",
        "embedding_method": "passage_embed",
        "database": os.path.relpath(db_path, project_root).replace("\\", "/"),
        "database_sha256": sha256_file(db_path),
        "corpus": os.path.relpath(corpus_path, project_root).replace("\\", "/"),
        "corpus_sha256": sha256_file(corpus_path),
        "record_types": dict(sorted(record_types.items())),
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    print(f"ChromaDB: {output_dir}")
    print(f"Collection: {args.collection}")
    print(f"Records: {collection.count()} x {dimensions}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
