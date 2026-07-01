#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ROOT = Path("/home2/ko66/ace-sandbox")


def build_directories(root: Path) -> tuple[Path, ...]:
    return (
        root / "nim-cache" / "asr",
        root / "nim-cache" / "tts",
        root / "logs",
        root / "logs" / "asr",
        root / "logs" / "tts",
        root / "audio",
        root / "ue-ddc",
        root / "docker",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize persistent storage for the ACE sandbox")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("ACE_SANDBOX_ROOT", DEFAULT_ROOT)))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for path in build_directories(args.root):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SystemExit(f"failed to create {path}: {exc}") from exc
        print(path)


if __name__ == "__main__":
    main()
