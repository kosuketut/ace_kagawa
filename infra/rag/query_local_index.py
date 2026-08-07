#!/usr/bin/env python3
"""Query the local SQLite RAG index."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_rag import DEFAULT_DB_PATH, dumps_json, hits_to_json_payload, search_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a local SQLite FTS RAG index")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2800,
        help="Maximum formatted context characters",
    )
    parser.add_argument("query", help="Search query")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hits = search_index(args.db, args.query, top_k=args.top_k)
    print(dumps_json(hits_to_json_payload(args.query, hits, max_context_chars=args.max_context_chars)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
