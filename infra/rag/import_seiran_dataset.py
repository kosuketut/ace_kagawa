#!/usr/bin/env python3
"""Validate and import the curated SEIRAN ZIP into the local Markdown RAG corpus."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlparse
import zipfile


DEFAULT_ARCHIVE = Path("data/tut_seiran_rag_dataset.zip")
DEFAULT_CORPUS_DIR = Path("data/rag/corpus")
DEFAULT_OUTPUT_NAME = "04_tut_seiran.md"
DEFAULT_SOURCE_DIR = Path("data/rag/sources/seiran/current")
DEFAULT_BACKUP_DIR = Path("data/rag/backups/seiran_import")
ARCHIVE_ROOT = "tut_seiran_rag"
CHUNKS_NAME = "tut_seiran_chunks.jsonl"
FAQ_NAME = "tut_seiran_faq.jsonl"
MANIFEST_NAME = "source_manifest.json"
REQUIRED_MEMBERS = {
    "README.md",
    "source_manifest.csv",
    MANIFEST_NAME,
    CHUNKS_NAME,
    FAQ_NAME,
    "tut_seiran_knowledge.md",
}
ALLOWED_FACT_STATUSES = {"verified", "derived", "not_publicly_confirmed", "guidance"}
ALLOWED_SOURCE_HOSTS = {"www.teu.ac.jp", "www.aitc.teu.ac.jp", "top500.org", "www.top500.org"}
MAX_ARCHIVE_MEMBERS = 20
MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024


class DatasetValidationError(RuntimeError):
    """Raised when the SEIRAN archive is unsafe or internally inconsistent."""


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


def verify_archive(archive_path: Path) -> dict[str, bytes]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"SEIRAN dataset archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise DatasetValidationError(f"ZIP CRC check failed: {bad_member}")
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise DatasetValidationError(f"archive contains too many files: {len(infos)}")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise DatasetValidationError("archive exceeds the uncompressed size limit")

        members: dict[str, bytes] = {}
        for info in infos:
            path = _safe_archive_name(info.filename)
            if len(path.parts) != 2 or path.parts[0] != ARCHIVE_ROOT:
                raise DatasetValidationError(f"unexpected archive layout: {info.filename!r}")
            relative_name = path.parts[1]
            if relative_name in members:
                raise DatasetValidationError(f"duplicate archive member: {relative_name}")
            members[relative_name] = archive.read(info)

    missing = REQUIRED_MEMBERS.difference(members)
    if missing:
        raise DatasetValidationError(f"archive is missing required files: {sorted(missing)}")
    return members


def _official_url(value: object) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in ALLOWED_SOURCE_HOSTS:
        raise DatasetValidationError(f"non-official or invalid source URL: {url!r}")
    return url


def _validated_sources(data: bytes) -> tuple[str, dict[str, dict]]:
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("source_manifest.json is invalid") from exc
    retrieved_at = str(payload.get("retrieved_at") or "").strip()
    try:
        retrieved_date = date.fromisoformat(retrieved_at)
    except ValueError as exc:
        raise DatasetValidationError(f"invalid manifest retrieved_at: {retrieved_at!r}") from exc
    if retrieved_date > date.today():
        raise DatasetValidationError(f"manifest retrieved_at is in the future: {retrieved_at}")

    source_map: dict[str, dict] = {}
    for source in payload.get("sources") or []:
        source_id = str(source.get("source_id") or "").strip()
        if not re.fullmatch(r"S\d{2}", source_id) or source_id in source_map:
            raise DatasetValidationError(f"invalid or duplicate source_id: {source_id!r}")
        normalized = dict(source)
        normalized["url"] = _official_url(source.get("url"))
        if not str(source.get("title") or "").strip() or not str(source.get("organization") or "").strip():
            raise DatasetValidationError(f"source {source_id} is missing title or organization")
        source_map[source_id] = normalized
    if not source_map:
        raise DatasetValidationError("source manifest contains no sources")
    return retrieved_at, source_map


def _jsonl_rows(data: bytes, name: str) -> list[dict]:
    rows: list[dict] = []
    for line_number, raw_line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"invalid JSON in {name} line {line_number}") from exc
        if not isinstance(row, dict):
            raise DatasetValidationError(f"non-object row in {name} line {line_number}")
        rows.append(row)
    if not rows:
        raise DatasetValidationError(f"{name} contains no rows")
    return rows


def _validate_source_links(row: dict, source_map: dict[str, dict], *, row_id: str) -> None:
    source_ids = row.get("source_ids")
    source_urls = row.get("source_urls")
    if not isinstance(source_ids, list) or not source_ids or not isinstance(source_urls, list):
        raise DatasetValidationError(f"{row_id} has invalid source_ids/source_urls")
    if len(source_ids) != len(source_urls) or len(source_ids) != len(set(source_ids)):
        raise DatasetValidationError(f"{row_id} has inconsistent source references")
    for source_id, source_url in zip(source_ids, source_urls):
        if source_id not in source_map:
            raise DatasetValidationError(f"{row_id} references unknown source {source_id!r}")
        normalized_url = _official_url(source_url)
        if normalized_url != source_map[source_id]["url"]:
            raise DatasetValidationError(f"{row_id} source URL does not match manifest source {source_id}")


def validated_chunks(data: bytes, source_map: dict[str, dict]) -> list[dict]:
    rows = _jsonl_rows(data, CHUNKS_NAME)
    seen_ids: set[str] = set()
    required = {
        "id",
        "title",
        "content",
        "category",
        "keywords",
        "questions",
        "source_ids",
        "source_urls",
        "fact_status",
        "temporal_scope",
        "answer_constraints",
        "retrieved_at",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise DatasetValidationError(f"chunk is missing fields: {sorted(missing)}")
        row_id = str(row["id"])
        if not re.fullmatch(r"tut-seiran-\d{3}", row_id) or row_id in seen_ids:
            raise DatasetValidationError(f"invalid or duplicate chunk id: {row_id!r}")
        seen_ids.add(row_id)
        if not 40 <= len(str(row["content"]).strip()) <= 2000:
            raise DatasetValidationError(f"{row_id} has invalid content length")
        if row["fact_status"] not in ALLOWED_FACT_STATUSES:
            raise DatasetValidationError(f"{row_id} has invalid fact_status: {row['fact_status']!r}")
        if not isinstance(row["keywords"], list) or not isinstance(row["questions"], list):
            raise DatasetValidationError(f"{row_id} has invalid keywords/questions")
        if not isinstance(row["answer_constraints"], list):
            raise DatasetValidationError(f"{row_id} has invalid answer_constraints")
        _validate_source_links(row, source_map, row_id=row_id)
    return rows


def validated_faq(data: bytes, source_map: dict[str, dict]) -> tuple[list[dict], list[str]]:
    rows = _jsonl_rows(data, FAQ_NAME)
    seen_ids: set[str] = set()
    inferred_verified: list[str] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        if not re.fullmatch(r"faq-\d{3}", row_id) or row_id in seen_ids:
            raise DatasetValidationError(f"invalid or duplicate FAQ id: {row_id!r}")
        seen_ids.add(row_id)
        if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
            raise DatasetValidationError(f"{row_id} has an empty question or answer")
        _validate_source_links(row, source_map, row_id=row_id)
        status = row.get("fact_status")
        if status is None:
            inferred_verified.append(row_id)
        elif status not in ALLOWED_FACT_STATUSES:
            raise DatasetValidationError(f"{row_id} has invalid fact_status: {status!r}")
    return rows, inferred_verified


def _single_line(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _effective_year(scope: object) -> str:
    years = re.findall(r"20\d{2}", str(scope or ""))
    return years[-1] if years else ""


def _temporal_status(chunk: dict) -> str:
    status = str(chunk["fact_status"])
    scope = str(chunk.get("temporal_scope") or "")
    year = _effective_year(scope)
    if status == "verified" and year:
        return f"current_{year}" if year == str(date.today().year) else f"historical_{year}"
    if status == "verified":
        return "current_or_undated"
    return status


def _source_priority(status: str) -> str:
    if status == "verified":
        return "P0"
    if status in {"derived", "not_publicly_confirmed"}:
        return "P1"
    return "P2"


def render_markdown(
    chunks: list[dict],
    source_map: dict[str, dict],
    *,
    archive_sha256: str,
    generated_at: str,
) -> str:
    lines = [
        "# 東京工科大学 スーパーコンピュータ 青嵐 RAGデータ",
        "",
        f"generated_at: {generated_at}",
        "dataset_source: data/tut_seiran_rag_dataset.zip",
        f"dataset_sha256: {archive_sha256}",
        f"dataset_input: {CHUNKS_NAME}",
        "",
    ]
    status_labels = {
        "verified": "確認済み",
        "derived": "公開値からの派生値",
        "not_publicly_confirmed": "一般公開情報では未確認",
        "guidance": "回答運用上の注意",
    }
    for chunk in chunks:
        source = source_map[chunk["source_ids"][0]]
        title = _single_line(chunk["title"])
        heading = _single_line(f"{chunk['id']} {title}").replace("#", "＃")
        keywords = [
            *chunk["keywords"],
            *chunk["questions"],
            *chunk["source_ids"],
            chunk["category"],
            chunk["fact_status"],
        ]
        unique_keywords = list(dict.fromkeys(_single_line(value) for value in keywords if _single_line(value)))
        body_lines = [
            f"事実状態: {status_labels[chunk['fact_status']]}。",
            _single_line(chunk["content"]),
        ]
        temporal_scope = _single_line(chunk.get("temporal_scope"))
        if temporal_scope:
            body_lines.append(f"有効時点: {temporal_scope}。")
        constraints = [_single_line(value) for value in chunk.get("answer_constraints") or [] if _single_line(value)]
        if constraints:
            body_lines.append("回答制約: " + " ".join(constraints))
        notes = _single_line(chunk.get("notes"))
        if notes:
            body_lines.append(f"注記: {notes}")
        lines.extend(
            [
                f"## {heading}",
                "",
                f"source_title: {title}",
                f"source_url: {_single_line(source['url'])}",
                f"publisher: {_single_line(source['organization'])}",
                f"published_date: {_single_line(source.get('published_at'))}",
                f"accessed_date: {_single_line(chunk['retrieved_at'])}",
                "source_type: curated_official_seiran_dataset",
                "source_format: jsonl",
                "page_number:",
                f"effective_year: {_effective_year(chunk.get('temporal_scope'))}",
                f"temporal_status: {_temporal_status(chunk)}",
                f"source_priority: {_source_priority(chunk['fact_status'])}",
                f"record_type: seiran_{chunk['fact_status']}",
                f"chunk_id: {_single_line(chunk['id'])}",
                f"keywords: {', '.join(unique_keywords)}",
                "",
                "\n".join(body_lines),
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
    members = verify_archive(archive_path)
    retrieved_at, source_map = _validated_sources(members[MANIFEST_NAME])
    chunks = validated_chunks(members[CHUNKS_NAME], source_map)
    faqs, inferred_verified_faqs = validated_faq(members[FAQ_NAME], source_map)

    archive_sha256 = _sha256_file(archive_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    markdown = render_markdown(
        chunks,
        source_map,
        archive_sha256=archive_sha256,
        generated_at=generated_at,
    )
    output_path = corpus_dir / output_name
    backup_files: list[str] = []
    if output_path.is_file():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = backup_root / timestamp
        backup_dir.mkdir(parents=True, exist_ok=False)
        backup_path = backup_dir / output_path.name
        output_path.replace(backup_path)
        backup_files.append(str(backup_path))

    _atomic_write(output_path, markdown.encode("utf-8"))
    for name, content in members.items():
        _atomic_write(source_dir / name, content)

    member_hashes = {name: _sha256_bytes(content) for name, content in sorted(members.items())}
    quality_warnings = [
        "The source archive has ZIP CRC protection but no publisher-provided per-file SHA-256 manifest; the import pins the archive and member hashes.",
    ]
    if inferred_verified_faqs:
        quality_warnings.append(
            f"{len(inferred_verified_faqs)} FAQ rows omit fact_status; FAQ rows are retained as source material but are not indexed to avoid duplicate evidence."
        )
    manifest = {
        "schema_version": 1,
        "imported_at": generated_at,
        "dataset_retrieved_at": retrieved_at,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "member_sha256": member_hashes,
        "source_count": len(source_map),
        "validated_chunks": len(chunks),
        "validated_faqs": len(faqs),
        "indexed_input": CHUNKS_NAME,
        "indexed_chunks": len(chunks),
        "faq_rows_indexed": 0,
        "output_path": str(output_path),
        "output_sha256": _sha256_bytes(markdown.encode("utf-8")),
        "backup_files": backup_files,
        "quality_warnings": quality_warnings,
        "fact_status_counts": {
            status: sum(chunk["fact_status"] == status for chunk in chunks)
            for status in sorted(ALLOWED_FACT_STATUSES)
        },
        "faq_missing_fact_status": inferred_verified_faqs,
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
