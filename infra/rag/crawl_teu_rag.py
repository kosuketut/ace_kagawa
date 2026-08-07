#!/usr/bin/env python3
"""Crawl focused Tokyo University of Technology pages into RAG Markdown.

The crawler starts from the official faculty and admissions landing pages,
follows only topic-local links on official ``teu.ac.jp`` hosts, extracts the
main page body, and writes metadata-rich Markdown understood by
``infra/rag/local_rag.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_FACULTY_SEED = "https://www.teu.ac.jp/gakubu/index.html"
DEFAULT_ADMISSIONS_SEED = "https://www.teu.ac.jp/entrance/index.html"
DEFAULT_FACULTY_OUTPUT = Path("data/rag/corpus/02_hachioji_faculties.md")
DEFAULT_ADMISSIONS_OUTPUT = Path("data/rag/corpus/04_admissions.md")
DEFAULT_MANIFEST = Path("data/rag/crawl/teu_faculty_admissions_manifest.json")
DEFAULT_BACKUP_ROOT = Path("data/rag/backups/teu_faculty_admissions")

USER_AGENT = "ACE-Kagawa-RAG-Crawler/1.0 (+https://www.teu.ac.jp/)"
MAX_HTML_BYTES = 8 * 1024 * 1024
MAX_PDF_BYTES = 32 * 1024 * 1024
MIN_BODY_CHARS = 120

_SPACE_RE = re.compile(r"[\t\r\f\v ]+")
_BLANK_RE = re.compile(r"\n{3,}")
_UNSAFE_HEADING_RE = re.compile(r"[^0-9A-Za-zぁ-んァ-ヶ一-龠々ー._ -]+")
_NEWS_PATH_RE = re.compile(r"/gakubu/20\d{2}\.html$")
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}
_STATIC_SUFFIXES = {
    ".css",
    ".js",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".mp4",
    ".zip",
    ".xlsx",
    ".xls",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_SKIP_TAGS = {"aside", "canvas", "footer", "form", "nav", "noscript", "script", "style", "svg", "template"}
_SKIP_MARKERS = {
    "breadcrumb",
    "category-nav",
    "cookie",
    "footer",
    "global-nav",
    "header-nav",
    "local-nav",
    "menu",
    "nav-category",
    "pagination",
    "share",
    "side-nav",
    "sidenav",
    "slidemenu",
    "sns",
}


@dataclass(frozen=True)
class CrawlTarget:
    url: str
    depth: int
    label: str = ""


@dataclass
class PageRecord:
    category: str
    url: str
    depth: int
    title: str
    body: str
    content_type: str
    sha256: str
    link_count: int


@dataclass
class CrawlFailure:
    category: str
    url: str
    depth: int
    error: str


@dataclass
class _Frame:
    tag: str
    skip: bool
    is_main: bool


class TeuMainContentParser(HTMLParser):
    """Extract text and links from the page's semantic main-content region."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.frames: list[_Frame] = []
        self.main_level: int | None = None
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.anchor: dict[str, object] | None = None

    @staticmethod
    def _attrs_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.lower(): (value or "") for key, value in attrs}

    @staticmethod
    def _is_main(tag: str, attrs: dict[str, str]) -> bool:
        element_id = attrs.get("id", "").lower()
        role = attrs.get("role", "").lower()
        return tag == "main" or role == "main" or element_id in {"bodyarea", "main", "main-content", "content"}

    @staticmethod
    def _is_skipped(tag: str, attrs: dict[str, str]) -> bool:
        if tag in _SKIP_TAGS:
            return True
        marker_text = " ".join((attrs.get("id", ""), attrs.get("class", ""))).lower()
        tokens = set(re.split(r"[^a-z0-9_-]+", marker_text))
        return any(marker in marker_text or marker in tokens for marker in _SKIP_MARKERS)

    def _active(self) -> bool:
        return self.main_level is not None and not any(frame.skip for frame in self.frames[self.main_level - 1 :])

    def _separator(self) -> None:
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = self._attrs_dict(attrs)
        is_main = self._is_main(tag, attrs_dict)
        inherited_skip = self.frames[-1].skip if self.frames else False
        skip = inherited_skip or self._is_skipped(tag, attrs_dict)

        if tag == "title":
            self.in_title = True

        if tag in _VOID_TAGS:
            if tag in {"br", "hr"} and self._active():
                self._separator()
            return

        self.frames.append(_Frame(tag=tag, skip=skip, is_main=is_main))
        if is_main and self.main_level is None:
            self.main_level = len(self.frames)

        if self._active() and tag in _BLOCK_TAGS:
            self._separator()

        if tag == "a" and self._active():
            href = attrs_dict.get("href", "").strip()
            if href:
                self.anchor = {"depth": len(self.frames), "href": href, "text": []}

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self.in_title = False

        if tag in _VOID_TAGS or not self.frames:
            return

        matching_index = next((index for index in range(len(self.frames) - 1, -1, -1) if self.frames[index].tag == tag), None)
        if matching_index is None:
            return

        if tag == "a" and self.anchor is not None:
            label = _normalize_inline(" ".join(self.anchor["text"]))
            self.links.append((str(self.anchor["href"]), label))
            self.anchor = None

        was_active = self._active()
        if was_active and tag in _BLOCK_TAGS:
            self._separator()

        removed = self.frames[matching_index:]
        del self.frames[matching_index:]
        if self.main_level is not None and any(frame.is_main for frame in removed):
            self.main_level = None

    def handle_data(self, data: str) -> None:
        normalized = _normalize_inline(data)
        if not normalized:
            return
        if self.in_title:
            self.title_parts.append(normalized)
        if not self._active():
            return
        self.parts.append(normalized)
        self.parts.append(" ")
        if self.anchor is not None:
            self.anchor["text"].append(normalized)

    def title(self) -> str:
        return _normalize_inline(" ".join(self.title_parts))

    def text(self) -> str:
        raw = "".join(self.parts)
        lines: list[str] = []
        previous = ""
        for line in raw.splitlines():
            line = _normalize_inline(line)
            if not line or line == previous:
                continue
            lines.append(line)
            previous = line
        return "\n\n".join(lines).strip()


def _normalize_inline(value: str) -> str:
    return _SPACE_RE.sub(" ", value.replace("\u3000", " ")).strip()


def _normalize_document(value: str) -> str:
    lines: list[str] = []
    previous = ""
    for raw_line in value.replace("\x0c", "\n").splitlines():
        line = _normalize_inline(raw_line)
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return _BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


def canonicalize_url(url: str, base_url: str = "") -> str:
    absolute = urljoin(base_url, url.strip())
    parsed = urlsplit(absolute)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    netloc = host
    if parsed.port and not ((parsed.scheme == "http" and parsed.port == 80) or (parsed.scheme == "https" and parsed.port == 443)):
        netloc = f"{host}:{parsed.port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in _TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items))
    return urlunsplit((parsed.scheme.lower(), netloc, path, query, ""))


def is_official_teu_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host == "teu.ac.jp" or host.endswith(".teu.ac.jp")


def is_crawlable_url(url: str, category: str, *, label: str = "") -> bool:
    if not url or not is_official_teu_url(url):
        return False
    parsed = urlsplit(url)
    path = parsed.path.lower()
    suffix = Path(path).suffix.lower()
    if suffix in _STATIC_SUFFIXES:
        return False
    if suffix and suffix not in {".html", ".htm", ".pdf"}:
        return False

    host = (parsed.hostname or "").lower()
    if category == "faculties":
        if host == "www.teu.ac.jp" and path.startswith("/gakubu/"):
            return True
        return suffix == ".pdf" and "gakubu" in path

    if category == "admissions":
        if host == "www.teu.ac.jp" and path.startswith("/entrance/"):
            return True
        if suffix == ".pdf" and any(term in f"{path} {label}" for term in ("入試", "募集", "選抜", "出願", "admission", "entrance")):
            return True
        if host == "jyuken.teu.ac.jp":
            return any(term in f"{path} {label}" for term in ("入試", "募集", "選抜", "出願", "受験", "jyuken"))
    return False


def extract_html(html_text: str) -> tuple[str, str, list[tuple[str, str]]]:
    parser = TeuMainContentParser()
    parser.feed(html_text)
    parser.close()
    title = parser.title() or "東京工科大学"
    return title, parser.text(), parser.links


def _decode_html(payload: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, re.IGNORECASE)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8", "cp932", "shift_jis"])
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def _read_limited(response, limit: int) -> bytes:
    payload = response.read(limit + 1)
    if len(payload) > limit:
        raise RuntimeError(f"response exceeds {limit} bytes")
    return payload


def fetch_url(url: str, *, timeout: float) -> tuple[str, str, bytes]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1"})
    with urlopen(request, timeout=timeout) as response:
        final_url = canonicalize_url(response.geturl())
        content_type = response.headers.get("Content-Type", "").lower()
        limit = MAX_PDF_BYTES if "pdf" in content_type or final_url.lower().endswith(".pdf") else MAX_HTML_BYTES
        payload = _read_limited(response, limit)
    return final_url, content_type, payload


def extract_pdf(payload: bytes) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise RuntimeError("pdftotext is required to extract linked PDF files")
    with tempfile.TemporaryDirectory(prefix="teu-rag-pdf-") as tmp:
        pdf_path = Path(tmp) / "source.pdf"
        text_path = Path(tmp) / "source.txt"
        pdf_path.write_bytes(payload)
        completed = subprocess.run(
            [pdftotext, "-layout", "-enc", "UTF-8", str(pdf_path), str(text_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit {completed.returncode}"
            raise RuntimeError(f"pdftotext failed: {detail}")
        return _normalize_document(text_path.read_text(encoding="utf-8", errors="replace"))


def crawl_category(
    *,
    category: str,
    seed_url: str,
    max_depth: int,
    max_pages: int,
    delay: float,
    timeout: float,
) -> tuple[list[PageRecord], list[CrawlFailure]]:
    seed = canonicalize_url(seed_url)
    queue: deque[CrawlTarget] = deque([CrawlTarget(seed, 0, "")])
    queued = {seed}
    visited: set[str] = set()
    pages: list[PageRecord] = []
    failures: list[CrawlFailure] = []

    while queue and len(visited) < max_pages:
        target = queue.popleft()
        if target.url in visited:
            continue
        visited.add(target.url)
        if pages or failures:
            time.sleep(delay)

        try:
            final_url, content_type, payload = fetch_url(target.url, timeout=timeout)
            if not is_crawlable_url(final_url, category, label=target.label):
                raise RuntimeError(f"redirected outside allowed {category} scope: {final_url}")

            digest = hashlib.sha256(payload).hexdigest()
            is_pdf = "pdf" in content_type or final_url.lower().endswith(".pdf")
            if is_pdf:
                body = extract_pdf(payload)
                title = target.label or Path(urlsplit(final_url).path).name or "PDF document"
                links: list[tuple[str, str]] = []
                normalized_content_type = "application/pdf"
            else:
                html_text = _decode_html(payload, content_type)
                title, body, links = extract_html(html_text)
                normalized_content_type = "text/html"

            body = _normalize_document(body)
            if len(body) < MIN_BODY_CHARS:
                raise RuntimeError(f"extracted body is too short ({len(body)} chars)")

            pages.append(
                PageRecord(
                    category=category,
                    url=final_url,
                    depth=target.depth,
                    title=title,
                    body=body,
                    content_type=normalized_content_type,
                    sha256=digest,
                    link_count=len(links),
                )
            )

            if target.depth >= max_depth or is_pdf:
                continue

            for href, label in links:
                linked = canonicalize_url(href, final_url)
                if linked in queued or linked in visited:
                    continue
                if not is_crawlable_url(linked, category, label=label):
                    continue
                queued.add(linked)
                queue.append(CrawlTarget(linked, target.depth + 1, label))
        except (HTTPError, URLError, TimeoutError, subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
            failures.append(CrawlFailure(category=category, url=target.url, depth=target.depth, error=str(exc)))

    if not any(page.depth == 0 for page in pages):
        raise RuntimeError(f"failed to crawl required {category} seed: {seed}")
    return pages, failures


def _safe_heading(value: str) -> str:
    cleaned = _UNSAFE_HEADING_RE.sub(" ", value)
    return _normalize_inline(cleaned)[:160] or "東京工科大学"


def _chunk_id(category: str, url: str) -> str:
    parsed = urlsplit(url)
    stem = re.sub(r"[^a-z0-9]+", "_", f"{parsed.hostname}_{parsed.path}", flags=re.IGNORECASE).strip("_")
    stem = stem[-72:] or category
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    prefix = "faculty" if category == "faculties" else "admission"
    return f"chunk_teu_{prefix}_{stem}_{digest}".lower()


def _keywords(category: str, title: str) -> str:
    base = ["東京工科大学"]
    if category == "faculties":
        base.extend(["学部", "学科", "専攻", "カリキュラム"])
    else:
        base.extend(["入試", "入学", "選抜", "出願", "募集要項", "学費", "奨学金"])
    title_terms = [term for term in re.split(r"[|｜・／/\s]+", title) if 1 < len(term) <= 40]
    return ",".join(dict.fromkeys(base + title_terms))


def render_markdown(category: str, pages: Iterable[PageRecord], *, generated_at: str) -> str:
    title = "Tokyo University of Technology Faculty Pages" if category == "faculties" else "Tokyo University of Technology Admissions Pages"
    blocks = [f"# {title}", "", f"generated_at: {generated_at}"]
    for page in sorted(pages, key=lambda item: (item.depth, item.url)):
        priority = "P0" if page.depth == 0 else "P1"
        record_type = "faculty_profile" if category == "faculties" else "admission"
        section = _safe_heading(page.title)
        blocks.extend(
            [
                "",
                f"## {section}",
                "",
                f"source_title: {section}",
                f"source_url: {page.url}",
                "publisher: 東京工科大学",
                f"accessed_date: {generated_at}",
                f"source_type: {category}",
                f"source_priority: {priority}",
                f"record_type: {record_type}",
                f"chunk_id: {_chunk_id(category, page.url)}",
                f"keywords: {_keywords(category, page.title)}",
                "",
                page.body,
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _backup_existing(paths: Iterable[Path], backup_root: Path, *, timestamp: str) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    safe_timestamp = timestamp.replace(":", "-")
    backup_dir = backup_root / safe_timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faculty-seed", default=DEFAULT_FACULTY_SEED)
    parser.add_argument("--admissions-seed", default=DEFAULT_ADMISSIONS_SEED)
    parser.add_argument("--faculty-output", type=Path, default=DEFAULT_FACULTY_OUTPUT)
    parser.add_argument("--admissions-output", type=Path, default=DEFAULT_ADMISSIONS_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages-per-category", type=int, default=120)
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_depth < 0:
        raise SystemExit("--max-depth must be non-negative")
    if args.max_pages_per_category < 1:
        raise SystemExit("--max-pages-per-category must be at least 1")
    if args.delay < 0:
        raise SystemExit("--delay must be non-negative")

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    all_pages: dict[str, list[PageRecord]] = {}
    all_failures: list[CrawlFailure] = []
    for category, seed in (("faculties", args.faculty_seed), ("admissions", args.admissions_seed)):
        pages, failures = crawl_category(
            category=category,
            seed_url=seed,
            max_depth=args.max_depth,
            max_pages=args.max_pages_per_category,
            delay=args.delay,
            timeout=args.timeout,
        )
        all_pages[category] = pages
        all_failures.extend(failures)

    faculty_markdown = render_markdown("faculties", all_pages["faculties"], generated_at=generated_at)
    admissions_markdown = render_markdown("admissions", all_pages["admissions"], generated_at=generated_at)
    backup_dir = _backup_existing(
        (args.faculty_output, args.admissions_output),
        args.backup_root,
        timestamp=generated_at,
    )
    _atomic_write(args.faculty_output, faculty_markdown)
    _atomic_write(args.admissions_output, admissions_markdown)

    manifest = {
        "generated_at": generated_at,
        "seeds": {"faculties": args.faculty_seed, "admissions": args.admissions_seed},
        "crawl": {
            "max_depth": args.max_depth,
            "max_pages_per_category": args.max_pages_per_category,
            "official_domain_only": True,
            "user_agent": USER_AGENT,
        },
        "outputs": {
            "faculties": str(args.faculty_output),
            "admissions": str(args.admissions_output),
            "backup_dir": str(backup_dir) if backup_dir else "",
        },
        "counts": {category: len(pages) for category, pages in all_pages.items()},
        "content_type_counts": {
            category: {
                content_type: sum(page.content_type == content_type for page in pages)
                for content_type in sorted({page.content_type for page in pages})
            }
            for category, pages in all_pages.items()
        },
        "failures": [asdict(failure) for failure in all_failures],
        "pages": {category: [asdict(page) | {"body": ""} for page in pages] for category, pages in all_pages.items()},
    }
    _atomic_write(args.manifest, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    summary = {
        "generated_at": generated_at,
        "counts": manifest["counts"],
        "content_type_counts": manifest["content_type_counts"],
        "failure_count": len(all_failures),
        "outputs": manifest["outputs"],
        "manifest": str(args.manifest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
