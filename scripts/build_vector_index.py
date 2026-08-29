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

import numpy as np
from fastembed import TextEmbedding


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_data(db_path: Path, corpus_path: Path) -> tuple[list[int], list[str], Counter[str]]:
    documents = corpus_path.read_text(encoding="utf-8").splitlines()

    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT id, record_type FROM courses ORDER BY id").fetchall()
    finally:
        connection.close()

    ids = [int(row[0]) for row in rows]
    record_types = Counter(str(row[1]) for row in rows)
    if len(ids) != len(documents):
        raise ValueError(
            f"Database/corpus count mismatch: database={len(ids)}, corpus={len(documents)}. "
            "Rebuild the structured corpus before vectorizing."
        )
    if not ids:
        raise ValueError("No schedule records found")
    if any(not document.strip() for document in documents):
        raise ValueError("Corpus contains empty documents")
    if len(ids) != len(set(ids)):
        raise ValueError("Database contains duplicate course IDs")
    return ids, documents, record_types


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model produced a zero-length vector")
    return vectors / norms


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a local dense-vector index for schedule RAG retrieval.")
    parser.add_argument("--db", type=Path, default=project_root / "class_schedule.db")
    parser.add_argument("--corpus", type=Path, default=project_root / "output" / "schedule_rag_corpus.txt")
    parser.add_argument("--output-dir", type=Path, default=project_root / "output" / "vector_store")
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

    ids, documents, record_types = load_source_data(db_path, corpus_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {args.model}")
    model = TextEmbedding(model_name=args.model, cache_dir=str(model_cache))
    print(f"Embedding documents: {len(documents)}")
    vectors = np.asarray(
        list(
            model.passage_embed(
                documents,
                batch_size=args.batch_size,
                parallel=args.parallel,
            )
        ),
        dtype=np.float32,
    )

    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    vectors = normalize_rows(vectors).astype(np.float32, copy=False)
    id_array = np.asarray(ids, dtype=np.int64)

    embeddings_path = output_dir / "schedule_embeddings.npy"
    ids_path = output_dir / "schedule_ids.npy"
    manifest_path = output_dir / "manifest.json"
    save_npy_atomic(embeddings_path, vectors)
    save_npy_atomic(ids_path, id_array)

    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dimensions": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "dtype": str(vectors.dtype),
        "normalized": True,
        "similarity": "cosine_via_dot_product",
        "embedding_method": "passage_embed",
        "database": os.path.relpath(db_path, project_root).replace("\\", "/"),
        "database_sha256": sha256_file(db_path),
        "corpus": os.path.relpath(corpus_path, project_root).replace("\\", "/"),
        "corpus_sha256": sha256_file(corpus_path),
        "embeddings_file": embeddings_path.name,
        "ids_file": ids_path.name,
        "record_types": dict(sorted(record_types.items())),
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    print(f"Vector index: {embeddings_path}")
    print(f"Shape: {vectors.shape[0]} x {vectors.shape[1]}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
