#!/usr/bin/env python3
"""Build the local SQLite RAG index from data/rag/corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_rag import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP, DEFAULT_CORPUS_DIR, DEFAULT_DB_PATH, build_index, dumps_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local SQLite FTS RAG index")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR, help="Corpus directory")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB output path")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help="Target chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_index(
        args.corpus,
        args.db,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )
    print(dumps_json(stats), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
