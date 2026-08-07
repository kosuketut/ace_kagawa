#!/usr/bin/env python3
"""Verify and import the curated TEU ZIP into the local Markdown RAG corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlparse
import zipfile


DEFAULT_ARCHIVE = Path("data/teu_rag_dataset.zip")
DEFAULT_CORPUS_DIR = Path("data/rag/corpus")
DEFAULT_OUTPUT_NAME = "02_teu_faculty_admissions_dataset.md"
DEFAULT_SOURCE_DIR = Path("data/rag/sources/teu/current")
DEFAULT_BACKUP_DIR = Path("data/rag/backups/teu_import")
LEGACY_CORPUS_NAMES = ("02_hachioji_faculties.md", "04_admissions.md")
BAD_TUITION_DOC_ID = "doc_1437162e0087211f"
BAD_TUITION_URL = "https://www.teu.ac.jp/entrance/006272.html"
CURRENT_TUITION_PDF_URL = "https://www.teu.ac.jp/entrance/info/sougousenbatsu2027youkou.pdf"


class DatasetValidationError(RuntimeError):
    """Raised when the archive cannot be trusted for import."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DatasetValidationError(f"unsafe archive member: {name!r}")
    return path


def _archive_root(names: set[str]) -> str:
    candidates = [name[: -len("checksums.sha256")].rstrip("/") for name in names if name.endswith("checksums.sha256")]
    if len(candidates) != 1:
        raise DatasetValidationError("archive must contain exactly one checksums.sha256")
    return candidates[0]


def verify_archive(archive_path: Path) -> tuple[str, dict[str, bytes]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"TEU dataset archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise DatasetValidationError(f"ZIP CRC check failed: {bad_member}")
        names = set()
        for info in archive.infolist():
            safe_path = _safe_archive_name(info.filename)
            if not info.is_dir():
                names.add(safe_path.as_posix())
        root = _archive_root(names)
        checksum_member = f"{root}/checksums.sha256" if root else "checksums.sha256"
        checksum_bytes = archive.read(checksum_member)
        verified: dict[str, bytes] = {"checksums.sha256": checksum_bytes}
        for line_number, raw_line in enumerate(checksum_bytes.decode("utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            parts = raw_line.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                raise DatasetValidationError(f"invalid checksum line {line_number}")
            expected, relative_name = parts[0], parts[1].strip().lstrip("*")
            relative_path = _safe_archive_name(relative_name)
            member = f"{root}/{relative_path.as_posix()}" if root else relative_path.as_posix()
            if member not in names:
                raise DatasetValidationError(f"checksummed member is missing: {relative_name}")
            content = archive.read(member)
            actual = _sha256_bytes(content)
            if actual != expected:
                raise DatasetValidationError(f"checksum mismatch for {relative_name}: {actual} != {expected}")
            verified[relative_path.as_posix()] = content
    return root, verified


def _validated_chunks(jsonl_bytes: bytes) -> list[dict]:
    chunks: list[dict] = []
    seen_ids: set[str] = set()
    required = {
        "chunk_id",
        "doc_id",
        "title",
        "section",
        "category",
        "source_url",
        "source_format",
        "page_number",
        "effective_year",
        "temporal_status",
        "retrieved_at",
        "text",
        "char_count",
        "content_hash_sha256",
        "keywords",
    }
    for line_number, raw_line in enumerate(jsonl_bytes.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"invalid JSON on chunks_plaintext.jsonl line {line_number}") from exc
        missing = required.difference(chunk)
        if missing:
            raise DatasetValidationError(f"line {line_number} is missing fields: {sorted(missing)}")
        chunk_id = str(chunk["chunk_id"])
        if not chunk_id or chunk_id in seen_ids:
            raise DatasetValidationError(f"invalid or duplicate chunk_id on line {line_number}: {chunk_id!r}")
        seen_ids.add(chunk_id)
        text = str(chunk["text"]).strip()
        if not 30 <= len(text) <= 1000:
            raise DatasetValidationError(f"chunk {chunk_id} has invalid text length: {len(text)}")
        if int(chunk["char_count"]) != len(text):
            raise DatasetValidationError(f"chunk {chunk_id} char_count mismatch")
        if _sha256_bytes(text.encode("utf-8")) != chunk["content_hash_sha256"]:
            raise DatasetValidationError(f"chunk {chunk_id} content hash mismatch")
        source_url = str(chunk["source_url"])
        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "www.teu.ac.jp":
            raise DatasetValidationError(f"chunk {chunk_id} has a non-official source URL: {source_url}")
        if chunk["section"] not in {"entrance", "gakubu", "grad"}:
            raise DatasetValidationError(f"chunk {chunk_id} has unsupported section: {chunk['section']!r}")
        if chunk["source_format"] not in {"html", "pdf"}:
            raise DatasetValidationError(f"chunk {chunk_id} has unsupported source_format")
        chunk["text"] = text
        chunks.append(chunk)

    if not chunks:
        raise DatasetValidationError("chunks_plaintext.jsonl contains no chunks")
    return chunks


def _is_bad_tuition_metadata(chunk: dict) -> bool:
    return (
        chunk["doc_id"] == BAD_TUITION_DOC_ID
        or chunk["source_url"] == BAD_TUITION_URL
    ) and chunk["effective_year"] == 2027


def _require_current_tuition_pdf(chunks: list[dict]) -> None:
    if not any(
        chunk["source_url"] == CURRENT_TUITION_PDF_URL
        and chunk["page_number"] == 18
        and chunk["effective_year"] == 2027
        and chunk["temporal_status"] == "current_2027"
        and "学費" in chunk["text"]
        for chunk in chunks
    ):
        raise DatasetValidationError("verified 2027 tuition evidence from official PDF page 18 is missing")


def _single_line(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _record_type(section: str) -> str:
    if section == "entrance":
        return "admission"
    if section == "grad":
        return "graduate_profile"
    return "faculty_profile"


def _source_priority(chunk: dict) -> str:
    if chunk["source_format"] == "pdf" and chunk["temporal_status"] == "current_2027":
        return "P0"
    return "P1"


def render_markdown(chunks: list[dict], *, archive_sha256: str, generated_at: str) -> str:
    lines = [
        "# 東京工科大学 学部・大学院・入試 RAGデータ",
        "",
        f"generated_at: {generated_at}",
        "dataset_source: data/teu_rag_dataset.zip",
        f"dataset_sha256: {archive_sha256}",
        "dataset_input: chunks_plaintext.jsonl",
        "",
    ]
    for chunk in chunks:
        title = _single_line(chunk["title"])
        heading = _single_line(f"{chunk['chunk_id']} {title}").replace("#", "＃")
        keywords = [
            *[str(keyword) for keyword in chunk.get("keywords", [])],
            str(chunk["doc_id"]),
            str(chunk["section"]),
            str(chunk["category"]),
            str(chunk["temporal_status"]),
        ]
        if chunk["effective_year"] is not None:
            keywords.append(str(chunk["effective_year"]))
        unique_keywords = list(dict.fromkeys(_single_line(keyword) for keyword in keywords if _single_line(keyword)))
        source_type = f"teu_dataset_{chunk['source_format']}"
        body = re.sub(r"(?m)^##\s+", "＃＃ ", chunk["text"])
        lines.extend(
            [
                f"## {heading}",
                "",
                f"source_title: {title}",
                f"source_url: {_single_line(chunk['source_url'])}",
                "publisher: 東京工科大学",
                f"accessed_date: {_single_line(chunk['retrieved_at'])}",
                f"source_type: {source_type}",
                f"source_format: {_single_line(chunk['source_format'])}",
                f"page_number: {_single_line(chunk['page_number'])}",
                f"effective_year: {_single_line(chunk['effective_year'])}",
                f"temporal_status: {_single_line(chunk['temporal_status'])}",
                f"source_priority: {_source_priority(chunk)}",
                f"record_type: {_record_type(str(chunk['section']))}",
                f"chunk_id: {_single_line(chunk['chunk_id'])}",
                f"keywords: {', '.join(unique_keywords)}",
                "",
                body,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def import_dataset(
    archive_path: Path,
    corpus_dir: Path,
    source_dir: Path,
    backup_root: Path,
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict:
    archive_path = archive_path.resolve()
    corpus_dir = corpus_dir.resolve()
    source_dir = source_dir.resolve()
    backup_root = backup_root.resolve()
    _root, verified = verify_archive(archive_path)
    if "chunks_plaintext.jsonl" not in verified:
        raise DatasetValidationError("checksums do not cover chunks_plaintext.jsonl")

    all_chunks = _validated_chunks(verified["chunks_plaintext.jsonl"])
    _require_current_tuition_pdf(all_chunks)
    skipped = [chunk for chunk in all_chunks if _is_bad_tuition_metadata(chunk)]
    chunks = [chunk for chunk in all_chunks if not _is_bad_tuition_metadata(chunk)]
    if len(skipped) != 1:
        raise DatasetValidationError(
            f"expected exactly one known tuition metadata exclusion, found {len(skipped)}"
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    archive_sha256 = _sha256_file(archive_path)
    markdown = render_markdown(chunks, archive_sha256=archive_sha256, generated_at=generated_at)
    output_path = corpus_dir / output_name

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = backup_root / timestamp
    backup_candidates = [corpus_dir / name for name in LEGACY_CORPUS_NAMES]
    if output_path not in backup_candidates:
        backup_candidates.append(output_path)
    existing_candidates = [path for path in backup_candidates if path.is_file()]
    backup_files: list[str] = []
    if existing_candidates:
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in existing_candidates:
            destination = backup_dir / path.name
            path.replace(destination)
            backup_files.append(str(destination))

    _atomic_write(output_path, markdown.encode("utf-8"))
    _atomic_write(source_dir / "chunks_plaintext.jsonl", verified["chunks_plaintext.jsonl"])
    _atomic_write(source_dir / "checksums.sha256", verified["checksums.sha256"])
    if "README.md" in verified:
        _atomic_write(source_dir / "dataset_README.md", verified["README.md"])

    manifest = {
        "schema_version": 1,
        "imported_at": generated_at,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "checksums_verified": len(verified) - 1,
        "input": "chunks_plaintext.jsonl",
        "input_chunks": len(all_chunks),
        "imported_chunks": len(chunks),
        "output_path": str(output_path),
        "output_sha256": _sha256_bytes(markdown.encode("utf-8")),
        "backup_files": backup_files,
        "excluded_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_url": chunk["source_url"],
                "reason": "HTML page describes 2026 tuition but dataset metadata marks it as 2027; use official 2027 PDF page 18",
            }
            for chunk in skipped
        ],
        "current_tuition_evidence": {
            "source_url": CURRENT_TUITION_PDF_URL,
            "page_number": 18,
            "effective_year": 2027,
        },
    }
    _atomic_write(
        source_dir / "import_manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = import_dataset(
        args.archive,
        args.corpus_dir,
        args.source_dir,
        args.backup_dir,
        output_name=args.output_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
