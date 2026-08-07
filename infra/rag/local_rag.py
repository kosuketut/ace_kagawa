#!/usr/bin/env python3
"""Dependency-free local RAG index based on SQLite FTS5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Iterable
import unicodedata


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".pdf"}
DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_PROMPT_CONTEXT_CHARS = 2800
DEFAULT_DB_PATH = Path("data/rag/local/local_rag.sqlite")
DEFAULT_CORPUS_DIR = Path("data/rag/corpus")
INDEX_SCHEMA_VERSION = "5"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LocalRagHit:
    path: str
    title: str
    section_title: str
    source_title: str
    source_url: str
    publisher: str
    published_date: str
    accessed_date: str
    source_type: str
    source_priority: str
    record_type: str
    chunk_id: str
    keywords: str
    chunk_index: int
    text: str
    score: float
    source_format: str = ""
    page_number: str = ""
    effective_year: str = ""
    temporal_status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QueryRouteDecision:
    route_candidate: bool
    domain: str
    reason: str
    intents: tuple[str, ...]


@dataclass(frozen=True)
class LocalRagAssessment:
    route_candidate: bool
    accepted: bool
    domain: str
    confidence: float
    reason: str
    intents: tuple[str, ...]


class StaleIndexError(RuntimeError):
    """Raised when a local index does not match its current corpus."""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_KANJI_DIGITS = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_KANJI_YEAR_RE = re.compile(r"[〇零一二三四五六七八九十百千]{2,8}(?=年度|年)")
_SEIRAN_ASR_TOKEN = r"(?:青嵐|セーラン|セーラ|セイラン|セイラ|セラン|せいらん|せいら|西ラ)"
_QUERY_ASR_REPLACEMENTS = (
    (re.compile(r"コンピューター?センス"), "コンピュータサイエンス"),
    (re.compile(r"コンピューターサイエンス"), "コンピュータサイエンス"),
    (re.compile(r"トップ(?:五百|500)", re.IGNORECASE), "TOP500"),
    (
        re.compile(rf"{_SEIRAN_ASR_TOKEN}(?:[\s　、,・]*{_SEIRAN_ASR_TOKEN})*"),
        "青嵐",
    ),
)


def _kanji_year_to_ascii(match: re.Match[str]) -> str:
    token = match.group(0)
    if all(character in _KANJI_DIGITS for character in token):
        value = int("".join(str(_KANJI_DIGITS[character]) for character in token))
    else:
        value = 0
        pending_digit = 0
        for character in token:
            if character in _KANJI_DIGITS:
                pending_digit = _KANJI_DIGITS[character]
                continue
            unit = {"千": 1000, "百": 100, "十": 10}.get(character)
            if unit is None:
                return token
            value += (pending_digit or 1) * unit
            pending_digit = 0
        value += pending_digit
    return str(value) if 1900 <= value <= 2199 else token


def normalize_query(query: str) -> str:
    """Normalize stable ASR variants without changing the user's intent."""

    normalized = unicodedata.normalize("NFKC", str(query))
    normalized = _KANJI_YEAR_RE.sub(_kanji_year_to_ascii, normalized)
    for pattern, replacement in _QUERY_ASR_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return _normalize_text(normalized)


def _read_pdf(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _normalize_text(_read_pdf(path))
    return _normalize_text(path.read_text(encoding="utf-8", errors="replace"))


def document_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem.replace("_", " ")


def iter_corpus_files(corpus_dir: Path, suffixes: Iterable[str] = SUPPORTED_SUFFIXES) -> list[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    return sorted(path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def _split_long_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    step = max(1, chunk_chars - chunk_overlap)
    while start < len(text):
        chunk = text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_text(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive")
    if chunk_overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be smaller than chunk_chars")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, chunk_chars, chunk_overlap))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if chunk_overlap and chunks:
            prefix = chunks[-1][-chunk_overlap:].strip()
            current = f"{prefix}\n\n{paragraph}" if prefix else paragraph
            if len(current) > chunk_chars:
                current = paragraph
        else:
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


_SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_GENERATED_AT_RE = re.compile(r"^generated_at:\s*(.+?)\s*$", re.MULTILINE)
_METADATA_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$")
_METADATA_KEYS = {
    "source_title",
    "source_url",
    "publisher",
    "published_date",
    "accessed_date",
    "source_type",
    "source_format",
    "page_number",
    "effective_year",
    "temporal_status",
    "source_priority",
    "record_type",
    "chunk_id",
    "keywords",
}


def _empty_chunk_metadata() -> dict[str, str]:
    return {
        "section_title": "",
        "source_title": "",
        "source_url": "",
        "publisher": "",
        "published_date": "",
        "accessed_date": "",
        "source_type": "",
        "source_format": "",
        "page_number": "",
        "effective_year": "",
        "temporal_status": "",
        "source_priority": "",
        "record_type": "",
        "chunk_id": "",
        "keywords": "",
    }


def _parse_section_metadata(section_text: str) -> tuple[dict[str, str], str]:
    metadata = _empty_chunk_metadata()
    body_lines: list[str] = []
    reading_metadata = True

    for line in section_text.splitlines():
        stripped = line.strip()
        match = _METADATA_RE.match(stripped)
        if reading_metadata and match and match.group(1) in _METADATA_KEYS:
            key, value = match.group(1), match.group(2).strip()
            if key in metadata:
                metadata[key] = value
            continue
        if reading_metadata and not stripped:
            continue
        reading_metadata = False
        body_lines.append(line)

    return metadata, _normalize_text("\n".join(body_lines))


def _iter_markdown_sections(text: str) -> list[tuple[str, dict[str, str], str]]:
    matches = list(_SECTION_HEADING_RE.finditer(text))
    sections: list[tuple[str, dict[str, str], str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_title = match.group(1).strip()
        metadata, body = _parse_section_metadata(text[start:end])
        if not body:
            continue
        metadata["section_title"] = section_title
        if not metadata["chunk_id"]:
            metadata["chunk_id"] = section_title
        sections.append((section_title, metadata, body))
    return sections


def _document_accessed_date(text: str) -> str:
    match = _GENERATED_AT_RE.search(text)
    return match.group(1).strip() if match else ""


def _publisher_from_source_url(source_url: str) -> str:
    normalized = source_url.lower()
    if "teu.ac.jp" in normalized:
        return "東京工科大学"
    return ""


def iter_document_chunks(
    path: Path,
    text: str,
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[tuple[dict[str, str], str]]:
    if path.suffix.lower() in {".md", ".markdown"}:
        sections = _iter_markdown_sections(text)
        if sections:
            document_chunks: list[tuple[dict[str, str], str]] = []
            document_accessed_date = _document_accessed_date(text)
            for _section_title, metadata, body in sections:
                section_chunks = chunk_text(body, chunk_chars, chunk_overlap)
                for part_index, chunk in enumerate(section_chunks):
                    chunk_metadata = dict(metadata)
                    if not chunk_metadata["accessed_date"]:
                        chunk_metadata["accessed_date"] = document_accessed_date
                    if not chunk_metadata["publisher"]:
                        chunk_metadata["publisher"] = _publisher_from_source_url(chunk_metadata["source_url"])
                    if part_index > 0 and chunk_metadata["chunk_id"]:
                        chunk_metadata["chunk_id"] = f"{chunk_metadata['chunk_id']}__part_{part_index + 1:02d}"
                    document_chunks.append((chunk_metadata, chunk))
            return document_chunks

    metadata = _empty_chunk_metadata()
    return [(metadata, chunk) for chunk in chunk_text(text, chunk_chars, chunk_overlap)]


def _create_schema(conn: sqlite3.Connection) -> str:
    conn.executescript(
        """
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            section_title TEXT NOT NULL DEFAULT '',
            source_title TEXT NOT NULL DEFAULT '',
            source_url TEXT NOT NULL DEFAULT '',
            publisher TEXT NOT NULL DEFAULT '',
            published_date TEXT NOT NULL DEFAULT '',
            accessed_date TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT '',
            source_format TEXT NOT NULL DEFAULT '',
            page_number TEXT NOT NULL DEFAULT '',
            effective_year TEXT NOT NULL DEFAULT '',
            temporal_status TEXT NOT NULL DEFAULT '',
            source_priority TEXT NOT NULL DEFAULT '',
            record_type TEXT NOT NULL DEFAULT '',
            chunk_id TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '',
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        """
    )
    tokenizer = "trigram"
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "text, title, section_title, source_title, publisher, source_type, source_format, "
            "effective_year, temporal_status, keywords, record_type, chunk_id, path, "
            "content='chunks', content_rowid='id', tokenize='trigram'"
            ")"
        )
    except sqlite3.OperationalError:
        tokenizer = "unicode61"
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "text, title, section_title, source_title, publisher, source_type, source_format, "
            "effective_year, temporal_status, keywords, record_type, chunk_id, path, "
            "content='chunks', content_rowid='id'"
            ")"
        )
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema_version", INDEX_SCHEMA_VERSION))
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("tokenizer", tokenizer))
    return tokenizer


def index_manifest_path(db_path: Path) -> Path:
    return db_path.with_suffix(".manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_manifest(
    corpus_dir: Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    corpus_dir = corpus_dir.resolve()
    files = iter_corpus_files(corpus_dir)
    fingerprint = hashlib.sha256()
    file_entries = []
    for path in files:
        relative_path = str(path.relative_to(corpus_dir))
        stat = path.stat()
        content_sha256 = _sha256_file(path)
        fingerprint.update(relative_path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(str(stat.st_size).encode("ascii"))
        fingerprint.update(b"\0")
        fingerprint.update(content_sha256.encode("ascii"))
        fingerprint.update(b"\0")
        file_entries.append(
            {
                "path": relative_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": content_sha256,
            }
        )
    fingerprint.update(f"chunk_chars={chunk_chars}\nchunk_overlap={chunk_overlap}\n".encode("ascii"))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_dir": str(corpus_dir),
        "corpus_fingerprint": fingerprint.hexdigest(),
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "files": file_entries,
    }


def load_index_manifest(db_path: Path) -> dict:
    manifest_path = index_manifest_path(db_path)
    if not manifest_path.is_file():
        raise StaleIndexError(f"local RAG manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleIndexError(f"local RAG manifest is unreadable: {manifest_path}") from exc
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise StaleIndexError(f"unsupported local RAG manifest schema: {payload.get('schema_version')!r}")
    return payload


def verify_index_bundle(db_path: Path) -> dict:
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise StaleIndexError(f"local RAG DB not found: {db_path}")
    saved = load_index_manifest(db_path)
    actual_db_sha256 = _sha256_file(db_path)
    if saved.get("index_sha256") != actual_db_sha256:
        raise StaleIndexError("local RAG index hash differs from its manifest")
    with _connect(db_path) as conn:
        metadata = {str(row["key"]): str(row["value"]) for row in conn.execute("SELECT key, value FROM metadata")}
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity.lower() != "ok":
        raise StaleIndexError(f"local RAG DB integrity check failed: {integrity}")
    if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise StaleIndexError(f"local RAG DB schema is stale: {metadata.get('schema_version')!r}")
    if metadata.get("corpus_fingerprint") != saved.get("corpus_fingerprint"):
        raise StaleIndexError("local RAG DB metadata fingerprint differs from its manifest")
    return {
        "fresh": True,
        "db_path": str(db_path),
        "manifest_path": str(index_manifest_path(db_path)),
        "index_sha256": actual_db_sha256,
        "corpus_fingerprint": metadata["corpus_fingerprint"],
        "files_indexed": int(metadata.get("files_indexed", "0")),
        "chunks_indexed": int(metadata.get("chunks_indexed", "0")),
        "created_at": metadata.get("created_at", ""),
    }


def verify_index_freshness(corpus_dir: Path, db_path: Path) -> dict:
    corpus_dir = corpus_dir.resolve()
    db_path = db_path.resolve()
    bundle = verify_index_bundle(db_path)
    saved = load_index_manifest(db_path)
    current = corpus_manifest(
        corpus_dir,
        chunk_chars=int(saved.get("chunk_chars", DEFAULT_CHUNK_CHARS)),
        chunk_overlap=int(saved.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
    )
    if saved.get("corpus_fingerprint") != current["corpus_fingerprint"]:
        raise StaleIndexError(
            "local RAG index is stale: corpus fingerprint differs; rebuild with infra/rag/build_local_index.py"
        )
    with _connect(db_path) as conn:
        metadata = {str(row["key"]): str(row["value"]) for row in conn.execute("SELECT key, value FROM metadata")}
    if metadata.get("corpus_fingerprint") != current["corpus_fingerprint"]:
        raise StaleIndexError("local RAG DB metadata fingerprint differs from the corpus")
    return bundle


def _build_index_file(
    corpus_dir: Path,
    db_path: Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[dict, dict]:
    corpus_dir = corpus_dir.resolve()
    db_path = db_path.resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    files = iter_corpus_files(corpus_dir)
    manifest = corpus_manifest(corpus_dir, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    created_at = datetime.now(timezone.utc).isoformat()
    skipped_files = 0
    chunks_indexed = 0

    with _connect(db_path) as conn:
        tokenizer = _create_schema(conn)
        for path in files:
            text = read_document(path)
            if not text:
                skipped_files += 1
                continue
            title = document_title(path, text)
            relative_path = str(path.relative_to(corpus_dir))
            for chunk_index, (chunk_metadata, chunk) in enumerate(
                iter_document_chunks(path, text, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
            ):
                cursor = conn.execute(
                    """
                    INSERT INTO chunks(
                        path,
                        title,
                        section_title,
                        source_title,
                        source_url,
                        publisher,
                        published_date,
                        accessed_date,
                        source_type,
                        source_format,
                        page_number,
                        effective_year,
                        temporal_status,
                        source_priority,
                        record_type,
                        chunk_id,
                        keywords,
                        chunk_index,
                        text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relative_path,
                        title,
                        chunk_metadata["section_title"],
                        chunk_metadata["source_title"],
                        chunk_metadata["source_url"],
                        chunk_metadata["publisher"],
                        chunk_metadata["published_date"],
                        chunk_metadata["accessed_date"],
                        chunk_metadata["source_type"],
                        chunk_metadata["source_format"],
                        chunk_metadata["page_number"],
                        chunk_metadata["effective_year"],
                        chunk_metadata["temporal_status"],
                        chunk_metadata["source_priority"],
                        chunk_metadata["record_type"],
                        chunk_metadata["chunk_id"],
                        chunk_metadata["keywords"],
                        chunk_index,
                        chunk,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunks_fts(
                        rowid,
                        text,
                        title,
                        section_title,
                        source_title,
                        publisher,
                        source_type,
                        source_format,
                        effective_year,
                        temporal_status,
                        keywords,
                        record_type,
                        chunk_id,
                        path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        chunk,
                        title,
                        chunk_metadata["section_title"],
                        chunk_metadata["source_title"],
                        chunk_metadata["publisher"],
                        chunk_metadata["source_type"],
                        chunk_metadata["source_format"],
                        chunk_metadata["effective_year"],
                        chunk_metadata["temporal_status"],
                        chunk_metadata["keywords"],
                        chunk_metadata["record_type"],
                        chunk_metadata["chunk_id"],
                        relative_path,
                    ),
                )
                chunks_indexed += 1
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("files_indexed", str(len(files) - skipped_files)))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("chunks_indexed", str(chunks_indexed)))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("created_at", created_at))
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("corpus_fingerprint", manifest["corpus_fingerprint"]),
        )
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("chunk_chars", str(chunk_chars)))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("chunk_overlap", str(chunk_overlap)))

    with _connect(db_path) as conn:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity.lower() != "ok":
        raise RuntimeError(f"new local RAG DB failed integrity check: {integrity}")

    stats = {
        "db_path": str(db_path),
        "corpus_dir": str(corpus_dir),
        "files_seen": len(files),
        "files_indexed": len(files) - skipped_files,
        "files_skipped": skipped_files,
        "chunks_indexed": chunks_indexed,
        "tokenizer": tokenizer,
        "created_at": created_at,
        "corpus_fingerprint": manifest["corpus_fingerprint"],
    }
    return stats, manifest


def build_index(
    corpus_dir: Path,
    db_path: Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Build and validate a temporary DB, then atomically replace the live index."""

    corpus_dir = corpus_dir.resolve()
    db_path = db_path.resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = index_manifest_path(db_path)
    nonce = f"{os.getpid()}.{time.time_ns()}"
    temporary_db = db_path.with_name(f".{db_path.name}.{nonce}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{nonce}.tmp")
    try:
        stats, manifest = _build_index_file(
            corpus_dir,
            temporary_db,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        index_sha256 = _sha256_file(temporary_db)
        manifest.update(
            {
                "created_at": stats["created_at"],
                "db_path": str(db_path),
                "index_sha256": index_sha256,
                "files_indexed": stats["files_indexed"],
                "files_skipped": stats["files_skipped"],
                "chunks_indexed": stats["chunks_indexed"],
                "tokenizer": stats["tokenizer"],
            }
        )
        temporary_manifest.write_text(dumps_json(manifest), encoding="utf-8")
        os.replace(temporary_db, db_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for temporary_path in (temporary_db, temporary_manifest):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    result = dict(stats)
    result["db_path"] = str(db_path)
    result["manifest_path"] = str(manifest_path)
    result["index_sha256"] = index_sha256
    return result


_JAPANESE_SPLIT_RE = re.compile(
    r"(?:について|とは|ですか|でしょうか|ください|教えて|何|どんな|どの|その|この|"
    r"[\s　、。,.!?！？:：;；「」『』（）()【】\[\]]+|"
    r"や|の|は|を|に|で|と|が|も|へ|から|まで|より)"
)
_ASCII_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+-]*")
_QUERY_ALIASES = {
    "八王子": ["hachioji"],
    "蒲田": ["kamata"],
    "アクセス": ["access"],
    "交通": ["access"],
    "キャンパス": ["campus"],
    "入試": ["admissions"],
    "学部": ["faculties"],
    "学科": ["faculties", "faculty_profile"],
    "専攻": ["faculties", "faculty_profile"],
    "大学院": ["faculties", "graduate_program"],
    "研究科": ["faculties", "graduate_program"],
    "学費": ["tuition", "admissions"],
    "入学金": ["tuition", "admissions"],
    "奨学金": ["scholarship", "admissions"],
    "所属": ["香川豊", "profile"],
    "研究": ["research", "研究テーマ", "研究分野"],
    "プロジェクト": ["research_project", "研究課題", "KAKEN"],
    "学生支援": ["student_support", "school_profile"],
    "大学概要": ["university_profile"],
    "大学案内": ["pamphlet"],
    "パンフレット": ["pamphlet"],
    "オープンキャンパス": ["open_campus"],
    "スパコン": ["スーパーコンピュータ", "AIスパコン", "青嵐", "SEIRAN"],
    "スーパーコンピュータ": ["スパコン", "AIスパコン", "青嵐", "SEIRAN"],
    "青嵐": ["SEIRAN", "AIスパコン", "スーパーコンピュータ"],
}


def extract_query_terms(query: str) -> list[str]:
    normalized = normalize_query(query)
    terms: list[str] = []
    compact = re.sub(r"\s+", "", normalized)
    if 2 <= len(compact) <= 32:
        terms.append(compact)

    terms.extend(_ASCII_TERM_RE.findall(normalized))
    for part in _JAPANESE_SPLIT_RE.split(normalized):
        candidate = part.strip()
        if len(candidate) >= 2:
            terms.append(candidate)

    if any(term in compact for term in ("香川先生", "香川豊先生", "香川さん")):
        terms.append("香川豊")
    if "専門分野" in compact:
        terms.extend(["材料強度学", "複合材料", "高信頼性材料"])
    if "年齢" in compact or "何歳" in compact:
        terms.extend(["生年月日", "1952年9月19日", "東京生まれ"])
    if "研究内容" in compact:
        terms.extend(["研究テーマ", "研究分野", "research"])
    if "論文" in compact or "業績" in compact:
        terms.extend(["publication", "paper", "award"])
    if "経歴" in compact:
        terms.extend(["career", "学歴", "職歴"])
    if "ebc" in compact.lower():
        terms.extend(["耐環境コーティング", "environmental barrier coating"])
    if "cmc" in compact.lower():
        terms.append("セラミック基複合材料")
    if "東京工科大学" in compact and any(term in compact for term in ("概要", "特徴", "どんな大学")):
        terms.extend(["実学主義", "university_profile", "school_profile"])
    if "パンフレット" in compact or "大学案内" in compact:
        terms.extend(["pamphlet", "大学案内2027", "デジタルパンフレット"])
    if "学部" in compact:
        terms.extend(["faculty_profile", "設置学部"])
    if "コンピュータサイエンス" in compact:
        terms.extend(["computer science", "faculty_profile"])
    for needle, aliases in _QUERY_ALIASES.items():
        if needle in compact:
            terms.extend(aliases)

    seen = set()
    unique_terms = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            unique_terms.append(term)
            seen.add(key)
    return unique_terms


def _quote_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _fts_query(query: str) -> str:
    terms = extract_query_terms(query)
    if not terms:
        return _quote_fts_term(query)
    return " OR ".join(_quote_fts_term(term) for term in terms[:12])


_CHUNK_SELECT_COLUMNS = """
    c.path,
    c.title,
    c.section_title,
    c.source_title,
    c.source_url,
    c.publisher,
    c.published_date,
    c.accessed_date,
    c.source_type,
    c.source_format,
    c.page_number,
    c.effective_year,
    c.temporal_status,
    c.source_priority,
    c.record_type,
    c.chunk_id,
    c.keywords,
    c.chunk_index,
    c.text
"""


def _row_text(row: sqlite3.Row, key: str) -> str:
    if key not in row.keys() or row[key] is None:
        return ""
    return str(row[key])


def _row_hit(row: sqlite3.Row, score: float) -> LocalRagHit:
    return LocalRagHit(
        path=_row_text(row, "path"),
        title=_row_text(row, "title"),
        section_title=_row_text(row, "section_title"),
        source_title=_row_text(row, "source_title"),
        source_url=_row_text(row, "source_url"),
        publisher=_row_text(row, "publisher"),
        published_date=_row_text(row, "published_date"),
        accessed_date=_row_text(row, "accessed_date"),
        source_type=_row_text(row, "source_type"),
        source_priority=_row_text(row, "source_priority"),
        record_type=_row_text(row, "record_type"),
        chunk_id=_row_text(row, "chunk_id"),
        keywords=_row_text(row, "keywords"),
        chunk_index=int(row["chunk_index"]),
        text=_row_text(row, "text"),
        score=score,
        source_format=_row_text(row, "source_format"),
        page_number=_row_text(row, "page_number"),
        effective_year=_row_text(row, "effective_year"),
        temporal_status=_row_text(row, "temporal_status"),
    )


def _rows_to_hits(rows: Iterable[sqlite3.Row]) -> list[LocalRagHit]:
    return [_row_hit(row, float(row["score"]) if "score" in row.keys() else 0.0) for row in rows]


def _query_intents(query: str) -> set[str]:
    compact = re.sub(r"\s+", "", normalize_query(query)).lower()
    intents: set[str] = set()
    if any(term in compact for term in ("アクセス", "行き方", "交通", "駅", "バス")):
        intents.add("access")
    if any(
        term in compact
        for term in (
            "入試",
            "受験",
            "選抜",
            "出願",
            "学費",
            "入学金",
            "授業料",
            "奨学金",
            "募集人員",
            "指定2教科",
            "試験科目",
            "選抜方法",
            "合格発表",
            "受験票",
            "入学試験日",
            "試験日",
            "入学手続期限",
        )
    ):
        intents.add("admissions")
    if any(term in compact for term in ("総合型選抜", "ao入試", "ao")):
        intents.add("admission_ao")
    if "共通テスト" in compact:
        intents.add("admission_common_test")
    if "奨学生入試" in compact:
        intents.add("admission_scholarship_exam")
    if "admissions" in intents and any(
        term in compact
        for term in (
            "日程",
            "カレンダー",
            "スケジュール",
            "いつ",
            "出願期間",
            "入学試験日",
            "試験日",
            "合格発表",
            "入学手続期限",
        )
    ):
        intents.add("admission_calendar")
    if "admissions" in intents and any(term in compact for term in ("入試情報", "入試について", "入試を詳しく")):
        intents.add("admission_overview")
    if "admissions" in intents and any(term in compact for term in ("募集人員", "募集定員")):
        intents.add("admission_capacity")
    if "admissions" in intents and any(term in compact for term in ("指定2教科", "試験科目", "基礎学力試験", "数学・英語")):
        intents.add("admission_subjects")
    if "admissions" in intents and any(term in compact for term in ("選抜方法", "配点", "面接試験", "プレゼンテーション")):
        intents.add("admission_selection_method")
    if any(term in compact for term in ("学費", "入学金", "授業料")):
        intents.add("admission_tuition")
    if any(term in compact for term in ("奨学金", "奨学生")):
        intents.add("admission_scholarship")
    if any(term in compact for term in ("オープンキャンパス", "説明会", "体験講義", "キャンパスツアー")):
        intents.add("open_campus")
    if any(term in compact for term in ("学部", "学科", "専攻", "大学院", "研究科")):
        intents.add("faculties")
    if any(term in compact for term in ("大学院", "研究科")):
        intents.add("graduate")
    if any(term in compact for term in ("パンフレット", "大学案内", "入試案内", "デジタルパンフレット")):
        intents.add("pamphlet")
    if any(
        term in compact
        for term in (
            "スパコン",
            "スーパーコンピュータ",
            "青嵐",
            "seiran",
            "dgxb200",
            "dgx b200",
            "top500",
            "hpcg",
        )
    ):
        intents.add("seiran")
    if "seiran" in intents and any(term in compact for term in ("gpu", "dgx", "b200", "ノード", "何基", "何台")):
        intents.add("seiran_hardware")
    if "seiran" in intents and any(
        term in compact for term in ("top500", "hpcg", "順位", "ランキング", "rmax", "rpeak")
    ):
        intents.add("seiran_ranking")
    if "seiran" in intents and any(
        term in compact for term in ("利用", "申請", "料金", "slurm", "ジョブ", "キュー", "アカウント", "外部利用")
    ):
        intents.add("seiran_usage")
    if "東京工科大学" in compact and any(term in compact for term in ("概要", "特徴", "どんな大学", "教育方針")):
        intents.add("university")
    if any(term in compact for term in ("学歴", "卒業", "修了", "博士前期", "博士後期", "理学修士", "工学博士", "学位")):
        intents.add("education")
    if any(term in compact for term in ("経歴", "職歴", "略歴", "就任", "所属歴")):
        intents.add("career")
    if any(term in compact for term in ("役職", "職名", "現職", "現在の役職", "今の役職", "所属", "肩書", "誰ですか")):
        intents.add("current_role")
    if any(term in compact for term in ("年齢", "何歳", "生年月日", "誕生日", "生まれ")):
        intents.add("birth_profile")
    if any(term in compact for term in ("論文", "文献", "著書", "出版", "業績", "発表", "特許")):
        intents.add("publication")
    if any(term in compact for term in ("受賞", "賞")):
        intents.add("award")
    if any(term in compact for term in ("専門分野", "専門は", "専攻")):
        intents.add("profile_fields")
    if any(
        term in compact
        for term in (
            "研究内容",
            "研究",
            "研究テーマ",
            "研究分野",
            "プロジェクト",
            "研究課題",
            "専門分野",
            "ebc",
            "cmc",
            "sic/sic",
            "非破壊",
            "複合材料",
            "材料強度",
            "界面",
        )
    ):
        intents.add("research")
    if "東京工科大学" in compact and any(
        term in compact
        for term in (
            "概要",
            "特徴",
            "どんな大学",
            "教育方針",
            "学生支援",
            "実学主義",
            "施設",
            "大学について",
        )
    ):
        intents.add("university")
    if (
        "香川" in compact
        and not intents.intersection(
            {"profile_fields", "career", "education", "birth_profile", "current_role", "research", "publication", "award"}
        )
        and any(term in compact for term in ("について教えて", "について説明", "どんな人", "プロフィール"))
    ):
        intents.add("current_role")
    return intents


def _query_locations(query: str) -> set[str]:
    compact = re.sub(r"\s+", "", normalize_query(query)).lower()
    locations: set[str] = set()
    if "八王子" in compact or "hachioji" in compact:
        locations.add("hachioji")
    if "蒲田" in compact or "kamata" in compact:
        locations.add("kamata")
    return locations


def _fields_match_domain(record_type: str, chunk_id: str, path: str, domain: str) -> bool:
    record_type = record_type.lower()
    chunk_id = chunk_id.lower()
    path = path.lower()
    if domain == "kagawa_profile":
        return record_type in {"kagawa_profile", "kagawa_career", "kagawa_education", "kagawa_president"}
    if domain == "kagawa_research":
        return record_type.startswith("kagawa_") and record_type != "kagawa_rag_todo"
    if domain == "access":
        return record_type == "access_route" or "access" in chunk_id or "access" in path
    if domain == "graduate_admissions":
        return record_type == "graduate_profile"
    if domain == "admissions":
        return record_type in {"admission", "admissions"} or "admission" in chunk_id
    if domain == "open_campus":
        return record_type in {"oc_event", "oc_program", "open_campus"} or chunk_id.startswith("chunk_oc_")
    if domain == "faculties":
        return record_type in {"faculty_profile", "graduate_profile"} or chunk_id == "chunk_school_hachioji_campus_overview"
    if domain == "university":
        return record_type == "school_profile" and "university" in chunk_id
    if domain == "pamphlet":
        return record_type == "pamphlet" or "pamphlet" in chunk_id
    if domain == "seiran":
        return record_type.startswith("seiran_") or chunk_id.startswith("tut-seiran-")
    return False


def _intent_boost(row: sqlite3.Row, query: str) -> float:
    intents = _query_intents(query)
    path = _row_text(row, "path").lower()
    section_title = _row_text(row, "section_title").lower()
    source_title = _row_text(row, "source_title").lower()
    record_type = _row_text(row, "record_type").lower()
    chunk_id = _row_text(row, "chunk_id").lower()
    keywords = _row_text(row, "keywords").lower()
    text = _row_text(row, "text").lower()
    combined_meta = " ".join((path, section_title, source_title, record_type, chunk_id, keywords))
    boost = 0.0

    compact_query = re.sub(r"\s+", "", normalize_query(query)).lower()
    primary_domain = _domain_from_intents(compact_query, intents)
    if primary_domain:
        if _fields_match_domain(record_type, chunk_id, path, primary_domain):
            boost += 260
        elif record_type or chunk_id:
            boost -= 180

    if "todo" in record_type or "todo" in chunk_id or "調査メモ" in source_title:
        boost -= 1000

    if not intents:
        return boost

    if "profile_fields" in intents:
        if record_type == "kagawa_profile" and (
            "専門分野" in text
            or "材料強度学" in combined_meta
            or "複合材料" in combined_meta
            or "高信頼性材料" in combined_meta
        ):
            boost += 155
        elif record_type == "kagawa_research_keywords":
            boost += 20

    if "current_role" in intents:
        if record_type == "kagawa_profile":
            boost += 115
        elif record_type == "kagawa_president":
            boost += 85
        elif record_type == "kagawa_career" and ("学長" in text or "現任" in text or "役職" in combined_meta):
            boost += 45

    if "education" in intents:
        if record_type == "kagawa_education":
            boost += 150
        elif record_type == "kagawa_profile":
            boost += 45
        elif record_type == "kagawa_career" and "career" not in intents:
            boost -= 60

    if "birth_profile" in intents:
        if record_type == "kagawa_profile" and (
            "birth_degree_fields" in chunk_id
            or "生年月日" in combined_meta
            or "1952年9月19日" in text
            or "東京生まれ" in text
        ):
            boost += 145
        elif record_type == "kagawa_profile":
            boost += 35

    if "research" in intents:
        if record_type in {"kagawa_research_keywords", "kagawa_research_project"}:
            boost += 110
        elif record_type in {"kagawa_research_significance", "kagawa_cmc_center"}:
            boost += 85
        elif record_type in {"kagawa_industry_academia", "kagawa_publication"}:
            boost += 35
        elif record_type == "kagawa_president":
            boost -= 45

    if "publication" in intents:
        if record_type == "kagawa_publication":
            boost += 120
        elif record_type == "kagawa_award":
            boost += 75
        elif record_type == "kagawa_conference_activity":
            boost += 65
        elif record_type == "kagawa_research_project":
            boost += 30
        elif record_type == "kagawa_president":
            boost -= 45

    if "award" in intents:
        if record_type == "kagawa_award":
            boost += 120
        elif record_type in {"kagawa_publication", "kagawa_conference_activity"}:
            boost += 35

    if "career" in intents:
        if record_type == "kagawa_career":
            boost += 120
        elif record_type == "kagawa_education":
            boost += 95
        elif record_type == "kagawa_profile":
            boost += 65
        elif record_type == "kagawa_researcher_id":
            boost += 30
        elif record_type == "kagawa_president":
            boost -= 25

    if "access" in intents:
        if record_type == "access_route":
            boost += 120
        elif record_type == "spoken_answer" and "access" in combined_meta:
            boost += 90
        if "access" in path or "access" in section_title or "access" in chunk_id:
            boost += 55

    if "admissions" in intents:
        if record_type in {"admission", "admissions"}:
            boost += 110
        elif record_type == "spoken_answer" and "admission" in combined_meta:
            boost += 75
        elif record_type == "oc_program":
            boost -= 20

    if "admission_overview" in intents and record_type in {"admission", "admissions"}:
        if any(term in combined_meta or term in text for term in ("募集人員", "入試日程", "選抜方法")):
            boost += 135
        if any(term in combined_meta or term in text for term in ("全学部ao入試", "学部特色入試")):
            boost += 75
        if text.lstrip().startswith(("総合型選抜", "共通事項")):
            boost += 35

    if "admission_capacity" in intents and record_type in {"admission", "admissions"}:
        if "募集人員" in combined_meta or "募集人員" in text:
            boost += 150

    if "admission_subjects" in intents and record_type in {"admission", "admissions"}:
        if any(term in combined_meta or term in text for term in ("指定2教科", "基礎学力試験", "試験内容")):
            boost += 160

    if "admission_selection_method" in intents and record_type in {"admission", "admissions"}:
        if any(term in combined_meta or term in text for term in ("選抜方法", "配点", "面接試験", "プレゼンテーション")):
            boost += 150

    if "admission_ao" in intents:
        if record_type in {"admission", "admissions"} and ("ao" in combined_meta or "総合型選抜" in text):
            boost += 90
        elif record_type == "oc_program":
            boost -= 80
        ao_detail_terms = (
            "日程",
            "募集人員",
            "選抜方法",
            "基礎学力",
            "面接",
            "プレゼン",
            "出願",
            "検定料",
            "合格発表",
            "学費",
            "奨学金",
            "併願",
            "q&a",
            "qa",
        )
        if not any(term in query.lower() for term in ao_detail_terms):
            if "2方式の概要" in source_title or "admission_method" in keywords:
                boost += 190
            if "q&a" in source_title:
                boost -= 160

    if "admission_common_test" in intents:
        if record_type in {"admission", "admissions"} and (
            "common_test" in combined_meta or "共通テスト" in text
        ):
            boost += 110
        elif record_type == "visitor_page" and "共通テスト" in text:
            boost += 45

    if "admission_scholarship_exam" in intents:
        if "shogakusei" in combined_meta or "奨学生入試" in text:
            boost += 260
        else:
            boost -= 250
        if record_type in {"admission", "admissions"} and "scholarship" in combined_meta and "奨学生入試" not in text:
            boost -= 120

    if "admission_calendar" in intents:
        if record_type in {"admission", "admissions"} and (
            "calendar" in combined_meta or "カレンダー" in text or "日程" in text
        ):
            boost += 95
        if record_type in {"admission", "admissions"} and any(
            unrelated in combined_meta for unrelated in ("tuition", "scholarship")
        ):
            boost -= 70

    if "admission_tuition" in intents:
        if record_type in {"admission", "admissions"} and (
            "tuition" in combined_meta or "学費" in combined_meta or "学費" in text
        ):
            boost += 280
            if any(term in text for term in ("単位:円", "入学金", "授業料", "学費等納入金(正規化表)")):
                boost += 110
        elif record_type in {"admission", "admissions"}:
            boost -= 140

    if "admission_scholarship" in intents:
        if record_type in {"admission", "admissions"} and (
            "scholarship" in combined_meta or "奨学" in combined_meta or "奨学" in text
        ):
            boost += 220
        elif record_type in {"admission", "admissions"}:
            boost -= 120

    if "open_campus" in intents:
        if record_type in {"oc_event", "open_campus"}:
            boost += 175
            if any(term in query for term in ("いつ", "次", "日程", "開催")):
                boost += 80
        elif record_type == "oc_program":
            boost += 115
        elif record_type == "spoken_answer" and "open_campus" in combined_meta:
            boost += 85

    if "faculties" in intents and primary_domain == "faculties":
        if record_type == "school_profile" and "campus_overview" in chunk_id:
            boost += 230
        elif record_type == "faculty_profile":
            boost += 150
        elif record_type == "oc_program":
            boost -= 120

    if "graduate" in intents:
        if record_type == "graduate_profile":
            boost += 250
            is_calendar_record = "admission_calendar" in combined_meta or "入試日程" in combined_meta
            if "admission_calendar" in intents:
                boost += 180 if is_calendar_record else -80
            elif "admission_overview" in intents:
                if any(
                    term in combined_meta or term in text
                    for term in ("募集課程", "出願資格", "選抜方法", "研究指導希望", "提出書類")
                ):
                    boost += 180
                if is_calendar_record:
                    boost -= 100
            elif any(
                term in compact_query for term in ("出願資格", "選抜方法", "学費", "入学金", "授業料", "奨学金")
            ):
                if any(
                    term in combined_meta or term in text
                    for term in ("募集課程", "出願資格", "選抜方法", "研究指導希望", "提出書類")
                ):
                    boost += 140
                if is_calendar_record:
                    boost -= 100
            if any(term in query for term in ("一覧", "研究科", "専攻")) and (
                "overview" in keywords or "navigation" in keywords or source_title == "大学院"
            ):
                boost += 140
        elif record_type == "faculty_profile":
            boost -= 180

    if "university" in intents:
        if record_type == "school_profile" and "university" in chunk_id:
            boost += 180
        elif record_type == "school_profile":
            boost += 100
        elif record_type.startswith("kagawa_"):
            boost -= 100

    if "pamphlet" in intents:
        if record_type == "pamphlet":
            boost += 220
        elif path.startswith("07_visitor_pages"):
            boost -= 100

    if "seiran_hardware" in intents and primary_domain == "seiran":
        if any(term in combined_meta or term in text for term in ("gpu数", "96 gpu", "ノードとgpuの構成")):
            boost += 360
        elif any(term in combined_meta or term in text for term in ("gpu", "dgx", "b200", "ノード")):
            boost += 60

    if "seiran_ranking" in intents and primary_domain == "seiran":
        if any(term in combined_meta for term in ("top500", "hpcg", "ランキング")):
            boost += 230
            if "最新" in compact_query and _row_text(row, "effective_year") == "2026":
                boost += 100

    if "seiran_usage" in intents and primary_domain == "seiran":
        if any(term in combined_meta for term in ("利用申請", "料金", "slurm", "ジョブスケジューラ", "外部利用")):
            boost += 300

    if intents.intersection({"seiran_hardware", "seiran_ranking", "seiran_usage"}) and chunk_id == "tut-seiran-001":
        boost -= 80

    row_location_text = " ".join((section_title, source_title, chunk_id, keywords, text))
    for location in _query_locations(query):
        if location == "hachioji" and "八王子" not in row_location_text and "hachioji" not in row_location_text:
            boost -= 800
        if location == "kamata" and "蒲田" not in row_location_text and "kamata" not in row_location_text:
            boost -= 800

    if path.startswith("07_visitor_pages"):
        boost -= 35
    return boost


def _priority_boost(row: sqlite3.Row) -> float:
    priority = _row_text(row, "source_priority").upper()
    if priority == "P0":
        return 4.0
    if priority == "P1":
        return 2.0
    return 0.0


_QUERY_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _temporal_boost(row: sqlite3.Row, query: str) -> float:
    """Prefer applicable records while retaining explicit historical/planned queries."""

    query = normalize_query(query)
    query_years = {match.group(1) for match in _QUERY_YEAR_RE.finditer(query)}
    effective_year = _row_text(row, "effective_year").strip()
    status = _row_text(row, "temporal_status").strip().lower()
    compact = re.sub(r"\s+", "", query).lower()

    if query_years:
        if effective_year in query_years:
            year_boost = 170.0
        elif effective_year:
            year_boost = -220.0
        else:
            year_boost = -20.0
    else:
        year_boost = 0.0

    if re.fullmatch(r"current_20\d{2}", status):
        return year_boost + (35.0 if not query_years else 10.0)
    if status == "current_or_undated":
        return year_boost + 15.0
    if status == "planned_subject_to_change":
        explicitly_requested = (
            effective_year in query_years
            or "デジタルエンターテインメント" in compact
            or "設置構想" in compact
            or "新学部" in compact
        )
        return year_boost + (130.0 if explicitly_requested else -180.0)
    if status.startswith("recruitment_closed"):
        explicitly_requested = "アントレプレナー" in compact or "募集停止" in compact
        return year_boost + (120.0 if explicitly_requested else -180.0)
    if status == "historical_result":
        if query_years:
            return year_boost
        if "入試結果" in compact or "結果" in compact:
            recency = max(0, int(effective_year) - 2024) if effective_year.isdigit() else 0
            return 10.0 + recency
        return -70.0
    if status.startswith("historical"):
        return year_boost if query_years else -80.0
    if status == "versioned_page_verify_year":
        return year_boost - 50.0
    return year_boost


def _score_row(row: sqlite3.Row, query: str, terms: list[str], fts_score: float = 0.0) -> float:
    path = str(row["path"]).lower()
    title = str(row["title"]).lower()
    section_title = _row_text(row, "section_title").lower()
    source_title = _row_text(row, "source_title").lower()
    record_type = _row_text(row, "record_type").lower()
    chunk_id = _row_text(row, "chunk_id").lower()
    keywords = _row_text(row, "keywords").lower()
    text = str(row["text"]).lower()
    score = fts_score * 0.25 + _intent_boost(row, query) + _priority_boost(row) + _temporal_boost(row, query)
    for term in terms:
        needle = term.lower()
        if not needle:
            continue
        weight = max(1, min(len(term), 12))
        if needle in path:
            score += weight * 5
        if needle in title:
            score += weight * 4
        if needle in section_title:
            score += weight * 6
        if needle in source_title:
            score += weight * 4
        if needle in record_type:
            score += weight * 6
        if needle in chunk_id:
            score += weight * 5
        if needle in keywords:
            score += weight * 8
        if needle in text:
            score += weight
    return score


def _evidence_group_key(hit: LocalRagHit) -> tuple[str, str, str]:
    if hit.page_number:
        return (hit.source_url or hit.path, "page", hit.page_number)
    base_chunk_id = re.sub(r"__part_\d+$", "", hit.chunk_id)
    return (hit.path, "chunk", base_chunk_id or str(hit.chunk_index))


def _dedupe_hits(
    hits: Iterable[LocalRagHit],
    top_k: int,
    *,
    domain: str = "",
) -> list[LocalRagHit]:
    best: dict[tuple[str, int], LocalRagHit] = {}
    for hit in hits:
        key = (hit.path, hit.chunk_index)
        if key not in best or hit.score > best[key].score:
            best[key] = hit
    ranked = sorted(best.values(), key=lambda hit: (-hit.score, hit.path, hit.chunk_index))
    if domain:
        domain_hits = [hit for hit in ranked if _hit_matches_domain(hit, domain)]
        if domain_hits:
            ranked = domain_hits

    diversified: list[LocalRagHit] = []
    deferred: list[LocalRagHit] = []
    seen_groups: set[tuple[str, str, str]] = set()
    for hit in ranked:
        group = _evidence_group_key(hit)
        if group in seen_groups:
            deferred.append(hit)
            continue
        diversified.append(hit)
        seen_groups.add(group)
        if len(diversified) == top_k:
            return diversified
    diversified.extend(deferred[: max(0, top_k - len(diversified))])
    return diversified[:top_k]


def _rerank_rows(rows: Iterable[sqlite3.Row], query: str) -> list[LocalRagHit]:
    terms = extract_query_terms(query)
    hits = []
    for row in rows:
        fts_score = float(row["score"]) if "score" in row.keys() else 0.0
        score = _score_row(row, query, terms, fts_score)
        if score > 0:
            hits.append(_row_hit(row, score))
    return hits


def _short_lexical_rows(
    conn: sqlite3.Connection,
    terms: Iterable[str],
    *,
    domain: str = "",
) -> list[sqlite3.Row]:
    short_terms = []
    seen = set()
    for term in terms:
        normalized = term.strip().lower()
        if len(normalized) != 2 or normalized in seen:
            continue
        seen.add(normalized)
        short_terms.append(normalized)
    if not short_terms:
        return []

    searchable_columns = (
        "c.text",
        "c.title",
        "c.section_title",
        "c.source_title",
        "c.publisher",
        "c.source_type",
        "c.source_format",
        "c.effective_year",
        "c.temporal_status",
        "c.keywords",
        "c.record_type",
        "c.chunk_id",
        "c.path",
    )
    clauses = []
    parameters: list[str] = []
    for term in short_terms[:8]:
        clauses.append("(" + " OR ".join(f"instr(lower({column}), ?) > 0" for column in searchable_columns) + ")")
        parameters.extend([term] * len(searchable_columns))
    rows = conn.execute(
        f"SELECT {_CHUNK_SELECT_COLUMNS}, 0.0 AS score FROM chunks c WHERE " + " OR ".join(clauses),
        parameters,
    ).fetchall()
    if not domain:
        return rows
    return [
        row
        for row in rows
        if _fields_match_domain(
            _row_text(row, "record_type"),
            _row_text(row, "chunk_id"),
            _row_text(row, "path"),
            domain,
        )
        or (not _row_text(row, "record_type") and not _row_text(row, "chunk_id"))
    ]


def _domain_rows(conn: sqlite3.Connection, domain: str) -> list[sqlite3.Row]:
    """Guarantee that strong lexical matches from another domain cannot crowd out the target domain."""

    if not domain:
        return []
    rows = conn.execute(f"SELECT {_CHUNK_SELECT_COLUMNS}, 0.0 AS score FROM chunks c").fetchall()
    return [
        row
        for row in rows
        if _fields_match_domain(
            _row_text(row, "record_type"),
            _row_text(row, "chunk_id"),
            _row_text(row, "path"),
            domain,
        )
    ]


def search_index(db_path: Path, query: str, *, top_k: int = 3, domain: str = "") -> list[LocalRagHit]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = normalize_query(query)
    if not query.strip():
        return []
    if not db_path.exists():
        raise FileNotFoundError(f"local RAG DB not found: {db_path}")

    terms = extract_query_terms(query)
    if not domain:
        route = classify_query(query)
        domain = route.domain if route.route_candidate else ""
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                {_CHUNK_SELECT_COLUMNS},
                -bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (_fts_query(query), max(top_k * 30, 100)),
        ).fetchall()
        rows.extend(_short_lexical_rows(conn, terms, domain=domain))
        rows.extend(_domain_rows(conn, domain))
        return _dedupe_hits(_rerank_rows(rows, query), top_k, domain=domain)


_GREETING_RE = re.compile(r"^(?:おはよう(?:ございます)?|こんにちは|こんばんは|はじめまして|ありがとう(?:ございます)?)\W*$")
_PERSON_NAME_RE = re.compile(r"([一-龥]{2,6})(?:先生|教授|准教授|氏|さん)")
_KNOWN_PERSON_NAMES = {"香川", "香川豊"}
_DOMAIN_ANCHORS = {
    "kagawa_profile": (
        "香川",
        "専門分野",
        "経歴",
        "学歴",
        "職歴",
        "年齢",
        "生年月日",
        "役職",
        "学位",
        "プロフィール",
    ),
    "kagawa_research": (
        "香川",
        "研究",
        "論文",
        "業績",
        "受賞",
        "ebc",
        "cmc",
        "sic/sic",
        "非破壊",
        "複合材料",
    ),
    "access": ("アクセス", "行き方", "交通", "八王子", "蒲田", "スクールバス"),
    "graduate_admissions": ("大学院", "研究科", "入試", "受験", "出願", "選抜"),
    "admissions": ("入試", "受験", "選抜", "出願", "総合型", "共通テスト", "奨学生"),
    "open_campus": ("オープンキャンパス", "説明会", "体験講義", "キャンパスツアー"),
    "faculties": ("学部", "学科", "専攻", "大学院", "研究科", "八王子", "蒲田", "設置学部"),
    "university": ("東京工科大学", "概要", "特徴", "実学主義", "教育方針"),
    "pamphlet": ("パンフレット", "大学案内", "入試案内", "デジタルパンフレット"),
    "seiran": (
        "スパコン",
        "スーパーコンピュータ",
        "青嵐",
        "seiran",
        "dgxb200",
        "dgx b200",
        "top500",
        "hpcg",
    ),
}


def _domain_from_intents(compact: str, intent_set: set[str]) -> str:
    has_kagawa = "香川" in compact
    admission_intents = {
        "admissions",
        "admission_ao",
        "admission_common_test",
        "admission_scholarship_exam",
        "admission_calendar",
        "admission_tuition",
        "admission_scholarship",
    }
    profile_intents = {"profile_fields", "career", "education", "birth_profile", "current_role"}
    research_intents = {"research", "publication", "award"}

    if "access" in intent_set:
        return "access"
    if "open_campus" in intent_set:
        return "open_campus"
    if "graduate" in intent_set and intent_set.intersection(admission_intents):
        return "graduate_admissions"
    if intent_set.intersection(admission_intents):
        return "admissions"
    if "pamphlet" in intent_set:
        return "pamphlet"
    if "seiran" in intent_set:
        return "seiran"
    if has_kagawa and intent_set.intersection(profile_intents):
        return "kagawa_profile"
    if has_kagawa and intent_set.intersection(research_intents):
        return "kagawa_research"
    if "faculties" in intent_set:
        return "faculties"
    if "university" in intent_set or (
        "東京工科大学" in compact and intent_set.intersection(research_intents)
    ):
        return "university"
    if intent_set.intersection(research_intents):
        return "kagawa_research"
    if intent_set.intersection(profile_intents):
        return "kagawa_profile"
    return ""


_GENERIC_ROUTE_KEYWORDS = {"詳細", "詳しく", "根拠", "資料", "ドキュメント", "引用", "出典"}


def _configured_route_domain(compact: str, route_keywords: Iterable[str]) -> tuple[str, str]:
    for raw_keyword in route_keywords:
        keyword = _normalize_text(str(raw_keyword))
        normalized_keyword = re.sub(r"\s+", "", keyword).lower()
        if not normalized_keyword or normalized_keyword not in compact:
            continue
        keyword_domain = _domain_from_intents(normalized_keyword, _query_intents(keyword))
        if keyword_domain:
            return keyword_domain, keyword
        if keyword in _GENERIC_ROUTE_KEYWORDS:
            continue
        if "香川" in compact:
            return "kagawa_research", keyword
        if "東京工科大学" in compact:
            return "university", keyword
    return "", ""


def classify_query(query: str, route_keywords: Iterable[str] = ()) -> QueryRouteDecision:
    normalized = normalize_query(query)
    compact = re.sub(r"\s+", "", normalized).lower()
    intents = tuple(sorted(_query_intents(normalized)))

    if not compact:
        return QueryRouteDecision(False, "", "empty_query", intents)
    if _GREETING_RE.fullmatch(normalized):
        return QueryRouteDecision(False, "", "greeting", intents)
    if any(term in compact for term in ("天気", "気温", "降水", "雨予報", "台風")):
        return QueryRouteDecision(False, "", "weather", intents)
    if any(term in compact for term in ("夕食", "晩ごはん", "晩御飯", "献立", "レシピ")):
        return QueryRouteDecision(False, "", "general_chitchat", intents)
    if any(term in compact for term in ("あなたの名前", "君の名前", "お名前は")) and "香川" not in compact:
        return QueryRouteDecision(False, "", "assistant_name", intents)

    domain = _domain_from_intents(compact, set(intents))
    if domain:
        return QueryRouteDecision(True, domain, "supported_domain", intents)

    configured_domain, matched_keyword = _configured_route_domain(compact, route_keywords)
    if configured_domain:
        return QueryRouteDecision(
            True,
            configured_domain,
            f"configured_route_keyword:{matched_keyword}",
            intents,
        )
    return QueryRouteDecision(False, "", "outside_supported_domains", intents)


_FOLLOWUP_RE = re.compile(
    r"^(?:それ|その(?:点|内容|件|研究|入試|学部)?|こちら)?(?:について)?"
    r"(?:もう少し|さらに|もっと)?(?:詳しく|具体的に)?"
    r"(?:教えて|説明して|話して|知りたい|お願いします|ください)?[。.!！?？]*$"
)
_FOLLOWUP_TERMS = (
    "もう少し",
    "さらに詳しく",
    "もっと詳しく",
    "具体的に",
    "続きを",
    "他には",
    "それはいつ",
    "それはどこ",
    "その理由",
    "何位",
)


def _looks_like_followup(query: str) -> bool:
    normalized = normalize_query(query)
    compact = re.sub(r"\s+", "", normalized)
    if not compact or len(compact) > 40:
        return False
    return bool(_FOLLOWUP_RE.fullmatch(compact)) or any(term in compact for term in _FOLLOWUP_TERMS)


_YEAR_PREFIX_FRAGMENT_RE = re.compile(r"^20\d{2}(?:年度|年)?(?:の)?$")


def resolve_conversation_query(user_messages: Iterable[str], route_keywords: Iterable[str] = ()) -> str:
    messages = [normalize_query(str(message)) for message in user_messages if normalize_query(str(message))]
    if not messages:
        return ""
    current = messages[-1]
    if len(messages) >= 2 and not _QUERY_YEAR_RE.search(current):
        previous = messages[-2]
        if _YEAR_PREFIX_FRAGMENT_RE.fullmatch(re.sub(r"\s+", "", previous)):
            current = f"{previous}{current}"
    current_route = classify_query(current, route_keywords)
    if current_route.route_candidate or current_route.reason in {
        "greeting",
        "weather",
        "general_chitchat",
        "assistant_name",
    }:
        return current
    if not _looks_like_followup(current):
        return current
    for previous in reversed(messages[:-1]):
        if classify_query(previous, route_keywords).route_candidate:
            return f"{previous}\nフォローアップ: {current}"
    return current


def _hit_searchable_text(hit: LocalRagHit) -> str:
    return " ".join(
        (
            hit.path,
            hit.title,
            hit.section_title,
            hit.source_title,
            hit.source_url,
            hit.publisher,
            hit.published_date,
            hit.accessed_date,
            hit.source_type,
            hit.source_format,
            hit.page_number,
            hit.effective_year,
            hit.temporal_status,
            hit.record_type,
            hit.chunk_id,
            hit.keywords,
            hit.text,
        )
    ).lower()


def _hit_matches_domain(hit: LocalRagHit, domain: str) -> bool:
    if domain == "graduate_admissions":
        metadata = " ".join(
            (
                hit.section_title,
                hit.source_title,
                hit.keywords,
                hit.chunk_id,
            )
        ).lower()
        return hit.record_type.lower() == "graduate_profile" and (
            "graduate_admission" in metadata or "入試" in metadata
        )
    return _fields_match_domain(hit.record_type, hit.chunk_id, hit.path, domain)


def _unknown_person(query: str) -> str:
    compact = re.sub(r"\s+", "", normalize_query(query))
    for name in _PERSON_NAME_RE.findall(compact):
        if name not in _KNOWN_PERSON_NAMES:
            return name
    return ""


def _unknown_admission_qualifier(query: str, hits: list[LocalRagHit]) -> str:
    compact = re.sub(r"\s+", "", normalize_query(query))
    combined = " ".join(_hit_searchable_text(hit) for hit in hits)
    for qualifier in re.findall(r"([一-龥ぁ-んァ-ン]{2,12})(?=選抜)", compact):
        for separator in ("について", "における", "年度", "の"):
            if separator in qualifier:
                qualifier = qualifier.rsplit(separator, 1)[-1]
        for prefix in ("東京工科大学", "本学", "大学"):
            if qualifier.startswith(prefix):
                qualifier = qualifier[len(prefix) :]
        qualifier = qualifier.strip()
        if not qualifier:
            continue
        if qualifier not in combined:
            return qualifier
    return ""


def filter_hits_for_domain(hits: Iterable[LocalRagHit], domain: str) -> list[LocalRagHit]:
    return [hit for hit in hits if _hit_matches_domain(hit, domain)]


def assess_local_rag_query(
    query: str,
    hits: list[LocalRagHit],
    route_keywords: Iterable[str] = (),
) -> LocalRagAssessment:
    query = normalize_query(query)
    route = classify_query(query, route_keywords)
    if not route.route_candidate:
        return LocalRagAssessment(False, False, route.domain, 0.0, route.reason, route.intents)
    if not hits:
        return LocalRagAssessment(True, False, route.domain, 0.0, "no_hit", route.intents)

    unknown_person = _unknown_person(query)
    if unknown_person:
        return LocalRagAssessment(True, False, route.domain, 0.0, f"unknown_person:{unknown_person}", route.intents)
    if route.domain == "admissions":
        unknown_qualifier = _unknown_admission_qualifier(query, hits)
        if unknown_qualifier:
            return LocalRagAssessment(
                True,
                False,
                route.domain,
                0.15,
                f"unmatched_admission_qualifier:{unknown_qualifier}",
                route.intents,
            )

    domain_hits = filter_hits_for_domain(hits, route.domain)
    if not domain_hits:
        return LocalRagAssessment(True, False, route.domain, 0.15, "domain_mismatch", route.intents)
    if hits[0].score < 12.0:
        return LocalRagAssessment(True, False, route.domain, 0.2, "score_below_threshold", route.intents)

    compact_query = re.sub(r"\s+", "", query).lower()
    anchors = [anchor for anchor in _DOMAIN_ANCHORS[route.domain] if anchor in compact_query]
    combined_domain_text = " ".join(_hit_searchable_text(hit) for hit in domain_hits)
    matched_anchors = [anchor for anchor in anchors if anchor in combined_domain_text]
    if route.domain == "access" and "アクセス" in anchors and "access" in combined_domain_text:
        matched_anchors.append("アクセス")
    if anchors and not matched_anchors:
        return LocalRagAssessment(True, False, route.domain, 0.25, "query_anchor_not_found", route.intents)

    domain_ratio = len(domain_hits) / len(hits)
    anchor_ratio = len(matched_anchors) / len(anchors) if anchors else 0.0
    priority_quality = 1.0 if any(hit.source_priority.upper() in {"P0", "P1"} for hit in domain_hits) else 0.0
    metadata_quality = 1.0 if any(hit.source_url and hit.chunk_id and hit.path for hit in domain_hits) else 0.0
    confidence = min(
        1.0,
        0.45 + 0.25 * domain_ratio + 0.15 * anchor_ratio + 0.1 * priority_quality + 0.05 * metadata_quality,
    )
    accepted = confidence >= 0.68
    reason = "accepted" if accepted else "confidence_below_threshold"
    return LocalRagAssessment(True, accepted, route.domain, round(confidence, 4), reason, route.intents)


def grounded_direct_reply(
    query: str,
    hits: list[LocalRagHit],
    *,
    today: date | None = None,
) -> str | None:
    """Return a short extractive reply for a narrow, well-grounded overview."""

    route = classify_query(query)
    if not hits or any(not _hit_matches_domain(hit, route.domain) for hit in hits):
        return None

    compact = re.sub(r"\s+", "", normalize_query(query))
    evidence = " ".join(hit.text for hit in hits)
    if route.domain == "access":
        if "八王子" in compact:
            return next((hit.text.strip() for hit in hits if "八王子" in hit.text), None)
        if "蒲田" in compact:
            return next((hit.text.strip() for hit in hits if "蒲田" in hit.text), None)
        return "アクセスは八王子キャンパスと蒲田キャンパスで異なります。キャンパス名を指定してください。"

    if route.domain == "kagawa_profile" and "current_role" in route.intents:
        if "東京工科大学の学長" in evidence and "セラミックス複合材料センター長" in evidence:
            return "香川豊は東京工科大学の学長・教授で、セラミックス複合材料センター長です。"

    if route.domain == "open_campus" and any(term in compact for term in ("次", "いつ", "日程", "開催")):
        reference_date = today or date.today()
        upcoming: list[tuple[date, str]] = []
        for hit in hits:
            location = "八王子キャンパス" if "八王子" in hit.text else "蒲田キャンパス" if "蒲田" in hit.text else ""
            if not location:
                continue
            for year, month, day in re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", hit.text):
                try:
                    event_date = date(int(year), int(month), int(day))
                except ValueError:
                    continue
                if event_date >= reference_date:
                    upcoming.append((event_date, location))
        if upcoming:
            upcoming.sort()
            next_date, next_location = upcoming[0]
            other = next(
                (
                    (event_date, location)
                    for event_date, location in upcoming[1:]
                    if location != next_location
                ),
                None,
            )
            reply = f"次回は{next_date.month}月{next_date.day}日の{next_location}です。"
            if other:
                other_date, other_location = other
                reply += f"{other_location}は{other_date.month}月{other_date.day}日です。"
            return reply
        return None

    if route.domain != "graduate_admissions":
        return None

    if "admission_selection_method" in route.intents:
        return "大学院入試の選抜方法は研究科・課程で異なります。研究科名と課程を指定してください。"
    if "出願資格" in compact:
        return "大学院入試の出願資格は研究科・課程で異なります。研究科名と課程を指定してください。"
    if "admission_tuition" in route.intents:
        return "現在の参照情報では、大学院の学費の正確な金額を確認できません。"
    if "admission_scholarship" in route.intents:
        return "現在の参照情報では、大学院向け奨学金の詳細を確認できません。"

    if "admission_calendar" in route.intents:
        if "合格発表" in compact and "2026年9月8日" in evidence and "2027年2月16日" in evidence:
            return "2027年度大学院入試の合格発表日は、A日程が2026年9月8日、B日程が2027年2月16日です。"
        if "入学手続期限" in compact and "2026年10月1日" in evidence and "2027年2月24日" in evidence:
            return "2027年度大学院入試の入学手続期限は、A日程が2026年10月1日、B日程が2027年2月24日です。"
        if "出願期間" in compact:
            if "デザイン研究科" in compact and "2026年6月29日" in evidence and "12月10日" in evidence:
                return "デザイン研究科の出願期間は、A日程が2026年6月29日から7月2日、B日程が12月7日から12月10日です。"
            if "医療技術学研究科" in compact and "1月8日" in evidence:
                return "医療技術学研究科の出願期間は、A日程が2026年7月28日から7月30日、B日程が2027年1月6日から1月8日です。"
            if any(term in compact for term in ("工学研究科", "バイオ・情報メディア研究科")) and "1月7日" in evidence:
                return "出願期間は、A日程が2026年7月28日から7月30日、B日程が2027年1月5日から1月7日です。"
            return "大学院入試の出願期間は研究科により異なります。研究科名を指定してお尋ねください。"
        if "2026年8月29日" in evidence and "2027年1月30日" in evidence:
            return (
                "2027年度大学院入試の試験日は、A日程が2026年8月29日、"
                "B日程が2027年1月30日です。出願期間は研究科により異なります。"
            )
        return None
    if "admission_overview" not in route.intents:
        return None

    fields = [
        field
        for field in ("募集課程", "出願資格", "日程", "選抜方法", "研究指導希望", "提出書類")
        if field in evidence
    ]
    if len(fields) < 3:
        return None
    return f"大学院入試では、{'、'.join(fields)}などをご案内しています。"


def citations_for_hits(hits: list[LocalRagHit]) -> list[dict]:
    citations = []
    for hit in hits:
        published_date_is_known = hit.published_date.strip().lower() not in {"", "未確認", "不明", "unknown"}
        citations.append(
            {
                "document_id": f"{hit.path}#{hit.chunk_id or 'chunk'}:{hit.chunk_index}",
                "source_title": hit.source_title or hit.title,
                "path": hit.path,
                "source_url": hit.source_url,
                "chunk_id": hit.chunk_id or f"{hit.path}:{hit.chunk_index}",
                "publisher": hit.publisher,
                "published_date": hit.published_date,
                "accessed_date": hit.accessed_date,
                "date": hit.published_date if published_date_is_known else hit.accessed_date,
                "date_type": "published" if published_date_is_known else "accessed",
                "source_type": hit.source_type,
                "source_format": hit.source_format,
                "page_number": hit.page_number,
                "effective_year": hit.effective_year,
                "temporal_status": hit.temporal_status,
                "source_priority": hit.source_priority,
                "record_type": hit.record_type,
                "score": hit.score,
            }
        )
    return citations


_CURRENT_EVIDENCE_STATUSES = {"", "current_2026", "current_2027", "current_or_undated"}
_NONCURRENT_EVIDENCE_STATUSES = {
    "historical_or_versioned",
    "historical_reference",
    "historical_result",
    "planned_subject_to_change",
    "recruitment_closed_since_2023",
    "versioned_page_verify_year",
}
_SOURCE_GUIDANCE_ACTIONS = ("参照", "確認", "照合", "優先", "使用")
_SOURCE_GUIDANCE_MARKERS = (
    "公式",
    "募集要項",
    "当該年度",
    "最新年度",
    "元ページ",
    "原表",
    "ページ表示上",
    "出願時",
    "本データセット",
    "RAG回答",
)
_SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")


def _is_source_guidance_sentence(sentence: str) -> bool:
    return any(action in sentence for action in _SOURCE_GUIDANCE_ACTIONS) and any(
        marker in sentence for marker in _SOURCE_GUIDANCE_MARKERS
    )


def _clean_prompt_body(text: str, *, temporal_status: str = "") -> str:
    body_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if _METADATA_RE.match(stripped):
            continue
        body_lines.append(stripped)
    body = " ".join(body_lines)
    body = re.sub(r"RAGでは[^。]*(?:。|$)", "", body)
    body = re.sub(r"RAG化[^。]*(?:。|$)", "", body)
    if temporal_status.strip().lower() in _CURRENT_EVIDENCE_STATUSES:
        body = "".join(
            sentence
            for sentence in _SENTENCE_RE.findall(body)
            if not _is_source_guidance_sentence(sentence)
        )
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _prompt_hits_for_query(hits: list[LocalRagHit], query: str) -> list[LocalRagHit]:
    """Keep current evidence clean while retaining requested historical/planned evidence."""

    selected = list(hits)
    if not selected:
        return selected
    query_years = {match.group(1) for match in _QUERY_YEAR_RE.finditer(normalize_query(query))}
    has_current = any(hit.temporal_status.strip().lower() in _CURRENT_EVIDENCE_STATUSES for hit in selected)
    if has_current and not query_years:
        current_only = [
            hit
            for hit in selected
            if hit.temporal_status.strip().lower() not in _NONCURRENT_EVIDENCE_STATUSES
        ]
        if current_only:
            selected = current_only
    if any(hit.temporal_status.strip().lower() != "versioned_page_verify_year" for hit in selected):
        selected = [
            hit
            for hit in selected
            if hit.temporal_status.strip().lower() != "versioned_page_verify_year"
        ]
    return selected


def _hit_header(index: int, hit: LocalRagHit) -> str:
    title = hit.source_title or hit.section_title or hit.title
    chunk_id = hit.chunk_id or f"{hit.path}:{hit.chunk_index}"
    metadata = [f"chunk_id={chunk_id}"]
    if hit.effective_year:
        metadata.append(f"effective_year={hit.effective_year}")
    if hit.temporal_status:
        metadata.append(f"temporal_status={hit.temporal_status}")
    if hit.page_number:
        metadata.append(f"page={hit.page_number}")
    return f"[source {index}] {title} [{', '.join(metadata)}]"


def format_hits_for_prompt_with_sources(
    hits: list[LocalRagHit],
    *,
    max_chars: int = DEFAULT_PROMPT_CONTEXT_CHARS,
    query: str = "",
) -> tuple[str, list[LocalRagHit]]:
    if not hits:
        return "", []
    prompt_hits = _prompt_hits_for_query(hits, query)
    if not prompt_hits:
        return "", []
    lines = [
        "参考情報（回答本文ではURL・ファイルパス・chunk_idを読み上げず、内容だけを自然に説明してください）:"
    ]
    lines.append(
        "注意: 表の日程・試験場・募集人員は、同じデータ行または対象学部に"
        "明記された対応関係だけを使ってください。"
    )
    if all(hit.temporal_status.strip().lower() in _CURRENT_EVIDENCE_STATUSES for hit in prompt_hits):
        lines.append(
            "回答方針: 現在有効な参考情報に答えがある場合は内容を直接答え、"
            "出典の再確認を促す定型文を追加しないでください。"
        )
    else:
        lines.append(
            "回答方針: 過年度・未確定・変更予定の参考情報に限り、"
            "その時点または不確実性を必要最小限の一文で示してください。"
        )
    used = sum(len(line) + 1 for line in lines)
    included_hits: list[LocalRagHit] = []
    for index, hit in enumerate(prompt_hits, start=1):
        header = _hit_header(index, hit)
        body = _clean_prompt_body(hit.text, temporal_status=hit.temporal_status)
        if not body:
            continue
        remaining = max_chars - used - len(header) - 8
        if remaining <= 0:
            break
        if len(body) > remaining:
            body = body[: max(0, remaining - 1)].rstrip() + "..."
        lines.extend([header, body])
        included_hits.append(hit)
        used += len(header) + len(body) + 2
    if not included_hits:
        return "", []
    return "\n".join(lines), included_hits


def format_hits_for_prompt(
    hits: list[LocalRagHit],
    *,
    max_chars: int = DEFAULT_PROMPT_CONTEXT_CHARS,
    query: str = "",
) -> str:
    context, _included_hits = format_hits_for_prompt_with_sources(hits, max_chars=max_chars, query=query)
    return context


def hits_to_json_payload(
    query: str,
    hits: list[LocalRagHit],
    *,
    max_context_chars: int = DEFAULT_PROMPT_CONTEXT_CHARS,
) -> dict:
    context, included_hits = format_hits_for_prompt_with_sources(
        hits,
        max_chars=max_context_chars,
        query=query,
    )
    return {
        "query": query,
        "hits": [hit.to_dict() for hit in hits],
        "citations": citations_for_hits(included_hits),
        "context": context,
    }


def dumps_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
