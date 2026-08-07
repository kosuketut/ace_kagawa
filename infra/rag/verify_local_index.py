#!/usr/bin/env python3
"""Fail when the local SQLite RAG index is stale or corrupted."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_rag import DEFAULT_CORPUS_DIR, DEFAULT_DB_PATH, dumps_json, verify_index_freshness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local RAG corpus/index freshness")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(dumps_json(verify_index_freshness(args.corpus, args.db)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
