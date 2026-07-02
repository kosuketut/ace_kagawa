#!/usr/bin/env python3
"""Dependency-free local RAG index based on SQLite FTS5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Iterable


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".pdf"}
DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_DB_PATH = Path("data/rag/local/local_rag.sqlite")
DEFAULT_CORPUS_DIR = Path("data/rag/corpus")


@dataclass(frozen=True)
class LocalRagHit:
    path: str
    title: str
    section_title: str
    source_title: str
    source_url: str
    published_date: str
    source_priority: str
    record_type: str
    chunk_id: str
    keywords: str
    chunk_index: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
_METADATA_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$")
_METADATA_KEYS = {
    "source_title",
    "source_url",
    "publisher",
    "published_date",
    "accessed_date",
    "source_type",
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
        "published_date": "",
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
            for _section_title, metadata, body in sections:
                for chunk in chunk_text(body, chunk_chars, chunk_overlap):
                    document_chunks.append((metadata, chunk))
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
            published_date TEXT NOT NULL DEFAULT '',
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
            "text, title, section_title, source_title, keywords, record_type, chunk_id, path, "
            "content='chunks', content_rowid='id', tokenize='trigram'"
            ")"
        )
    except sqlite3.OperationalError:
        tokenizer = "unicode61"
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "text, title, section_title, source_title, keywords, record_type, chunk_id, path, "
            "content='chunks', content_rowid='id'"
            ")"
        )
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema_version", "2"))
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("tokenizer", tokenizer))
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("created_at", str(time.time())))
    return tokenizer


def build_index(
    corpus_dir: Path,
    db_path: Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    corpus_dir = corpus_dir.resolve()
    db_path = db_path.resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    files = iter_corpus_files(corpus_dir)
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
                        published_date,
                        source_priority,
                        record_type,
                        chunk_id,
                        keywords,
                        chunk_index,
                        text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relative_path,
                        title,
                        chunk_metadata["section_title"],
                        chunk_metadata["source_title"],
                        chunk_metadata["source_url"],
                        chunk_metadata["published_date"],
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
                        keywords,
                        record_type,
                        chunk_id,
                        path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        chunk,
                        title,
                        chunk_metadata["section_title"],
                        chunk_metadata["source_title"],
                        chunk_metadata["keywords"],
                        chunk_metadata["record_type"],
                        chunk_metadata["chunk_id"],
                        relative_path,
                    ),
                )
                chunks_indexed += 1
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("files_indexed", str(len(files) - skipped_files)))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("chunks_indexed", str(chunks_indexed)))

    return {
        "db_path": str(db_path),
        "corpus_dir": str(corpus_dir),
        "files_seen": len(files),
        "files_indexed": len(files) - skipped_files,
        "files_skipped": skipped_files,
        "chunks_indexed": chunks_indexed,
        "tokenizer": tokenizer,
    }


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
}


def extract_query_terms(query: str) -> list[str]:
    normalized = _normalize_text(query)
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
    c.published_date,
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
        published_date=_row_text(row, "published_date"),
        source_priority=_row_text(row, "source_priority"),
        record_type=_row_text(row, "record_type"),
        chunk_id=_row_text(row, "chunk_id"),
        keywords=_row_text(row, "keywords"),
        chunk_index=int(row["chunk_index"]),
        text=_row_text(row, "text"),
        score=score,
    )


def _rows_to_hits(rows: Iterable[sqlite3.Row]) -> list[LocalRagHit]:
    return [_row_hit(row, float(row["score"]) if "score" in row.keys() else 0.0) for row in rows]


def _query_intents(query: str) -> set[str]:
    compact = re.sub(r"\s+", "", query).lower()
    intents: set[str] = set()
    if any(term in compact for term in ("アクセス", "行き方", "交通", "駅", "バス")):
        intents.add("access")
    if any(term in compact for term in ("入試", "受験", "選抜", "出願")):
        intents.add("admissions")
    if any(term in compact for term in ("総合型選抜", "ao入試", "ao")):
        intents.add("admission_ao")
    if "共通テスト" in compact:
        intents.add("admission_common_test")
    if "奨学生入試" in compact:
        intents.add("admission_scholarship_exam")
    if any(term in compact for term in ("日程", "カレンダー", "スケジュール", "いつ")):
        intents.add("admission_calendar")
    if any(term in compact for term in ("学費", "入学金", "授業料")):
        intents.add("admission_tuition")
    if any(term in compact for term in ("奨学金", "奨学生")):
        intents.add("admission_scholarship")
    if any(term in compact for term in ("オープンキャンパス", "説明会", "体験講義", "キャンパスツアー")):
        intents.add("open_campus")
    if any(term in compact for term in ("学歴", "卒業", "修了", "博士前期", "博士後期", "理学修士", "工学博士", "学位")):
        intents.add("education")
    if any(term in compact for term in ("経歴", "職歴", "略歴", "就任", "所属歴")):
        intents.add("career")
    if any(term in compact for term in ("役職", "職名", "現職", "現在の役職", "今の役職", "誰ですか")):
        intents.add("current_role")
    if any(term in compact for term in ("年齢", "何歳", "生年月日", "誕生日", "生まれ")):
        intents.add("birth_profile")
    if any(term in compact for term in ("論文", "文献", "著書", "出版", "業績", "発表")):
        intents.add("publication")
    if any(term in compact for term in ("受賞", "賞")):
        intents.add("award")
    if any(term in compact for term in ("専門分野", "専門は", "専攻")):
        intents.add("profile_fields")
    if any(
        term in compact
        for term in (
            "研究内容",
            "研究テーマ",
            "研究分野",
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
    return intents


def _query_locations(query: str) -> set[str]:
    compact = re.sub(r"\s+", "", query).lower()
    locations: set[str] = set()
    if "八王子" in compact or "hachioji" in compact:
        locations.add("hachioji")
    if "蒲田" in compact or "kamata" in compact:
        locations.add("kamata")
    return locations


def _intent_boost(row: sqlite3.Row, query: str) -> float:
    intents = _query_intents(query)
    path = _row_text(row, "path").lower()
    section_title = _row_text(row, "section_title").lower()
    source_title = _row_text(row, "source_title").lower()
    record_type = _row_text(row, "record_type").lower()
    chunk_id = _row_text(row, "chunk_id").lower()
    keywords = _row_text(row, "keywords").lower()
    text = _row_text(row, "text").lower()
    combined_meta = " ".join((path, section_title, record_type, chunk_id, keywords))
    boost = 0.0

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

    if "admission_ao" in intents:
        if record_type in {"admission", "admissions"} and ("ao" in combined_meta or "総合型選抜" in text):
            boost += 90
        elif record_type == "oc_program":
            boost -= 80

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
        if record_type in {"admission", "admissions"} and "tuition" in combined_meta:
            boost += 90

    if "admission_scholarship" in intents:
        if record_type in {"admission", "admissions"} and "scholarship" in combined_meta:
            boost += 90

    if "open_campus" in intents:
        if record_type in {"oc_program", "open_campus"}:
            boost += 115
        elif record_type == "spoken_answer" and "open_campus" in combined_meta:
            boost += 85

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


def _score_row(row: sqlite3.Row, query: str, terms: list[str], fts_score: float = 0.0) -> float:
    path = str(row["path"]).lower()
    title = str(row["title"]).lower()
    section_title = _row_text(row, "section_title").lower()
    source_title = _row_text(row, "source_title").lower()
    record_type = _row_text(row, "record_type").lower()
    chunk_id = _row_text(row, "chunk_id").lower()
    keywords = _row_text(row, "keywords").lower()
    text = str(row["text"]).lower()
    score = fts_score * 0.25 + _intent_boost(row, query) + _priority_boost(row)
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


def _dedupe_hits(hits: Iterable[LocalRagHit], top_k: int) -> list[LocalRagHit]:
    best: dict[tuple[str, int], LocalRagHit] = {}
    for hit in hits:
        key = (hit.path, hit.chunk_index)
        if key not in best or hit.score > best[key].score:
            best[key] = hit
    return sorted(best.values(), key=lambda hit: (-hit.score, hit.path, hit.chunk_index))[:top_k]


def _rerank_rows(rows: Iterable[sqlite3.Row], query: str) -> list[LocalRagHit]:
    terms = extract_query_terms(query)
    hits = []
    for row in rows:
        fts_score = float(row["score"]) if "score" in row.keys() else 0.0
        score = _score_row(row, query, terms, fts_score)
        if score > 0:
            hits.append(_row_hit(row, score))
    return hits


def _like_search(conn: sqlite3.Connection, query: str, top_k: int) -> list[LocalRagHit]:
    terms = extract_query_terms(query)
    rows = conn.execute(f"SELECT {_CHUNK_SELECT_COLUMNS} FROM chunks c").fetchall()
    scored = []
    for row in rows:
        score = _score_row(row, query, terms)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], str(item[1]["path"]), int(item[1]["chunk_index"])))
    return [_row_hit(row, score) for score, row in scored[:top_k]]


def search_index(db_path: Path, query: str, *, top_k: int = 3) -> list[LocalRagHit]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not query.strip():
        return []
    if not db_path.exists():
        raise FileNotFoundError(f"local RAG DB not found: {db_path}")

    with _connect(db_path) as conn:
        try:
            rows = conn.execute(
                """
                SELECT
                    c.path,
                    c.title,
                    c.section_title,
                    c.source_title,
                    c.source_url,
                    c.published_date,
                    c.source_priority,
                    c.record_type,
                    c.chunk_id,
                    c.keywords,
                    c.chunk_index,
                    c.text,
                    -bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (_fts_query(query), max(top_k * 10, 50)),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        hits = [*_rerank_rows(rows, query), *_like_search(conn, query, max(top_k * 5, 20))]
        return _dedupe_hits(hits, top_k)


def _clean_prompt_body(text: str) -> str:
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
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _hit_header(index: int, hit: LocalRagHit) -> str:
    title = hit.source_title or hit.section_title or hit.title
    locator = hit.path
    if hit.chunk_id:
        locator = f"{locator}#{hit.chunk_id}"
    return f"[{index}] {title} ({locator})"


def format_hits_for_prompt(hits: list[LocalRagHit], *, max_chars: int = 1800) -> str:
    if not hits:
        return ""
    lines = ["参考情報:"]
    used = len(lines[0])
    for index, hit in enumerate(hits, start=1):
        header = _hit_header(index, hit)
        body = _clean_prompt_body(hit.text)
        if not body:
            continue
        remaining = max_chars - used - len(header) - 8
        if remaining <= 0:
            break
        if len(body) > remaining:
            body = body[: max(0, remaining - 1)].rstrip() + "..."
        lines.extend([header, body])
        used += len(header) + len(body) + 2
    return "\n".join(lines)


def hits_to_json_payload(query: str, hits: list[LocalRagHit], *, max_context_chars: int = 1800) -> dict:
    return {
        "query": query,
        "hits": [hit.to_dict() for hit in hits],
        "context": format_hits_for_prompt(hits, max_chars=max_context_chars),
    }


def dumps_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
