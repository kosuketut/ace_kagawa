from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


IMPORTER = load_module("teu_rag_import_test_module", ROOT / "infra/rag/import_teu_dataset.py")
LOCAL_RAG = load_module("teu_local_rag_test_module", ROOT / "infra/rag/local_rag.py")


def make_chunk(
    chunk_id: str,
    doc_id: str,
    title: str,
    text: str,
    *,
    source_url: str,
    section: str = "entrance",
    category: str = "tuition",
    source_format: str = "html",
    page_number=None,
    effective_year=None,
    temporal_status: str = "current_or_undated",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_index": 0,
        "title": title,
        "section": section,
        "category": category,
        "source_url": source_url,
        "source_format": source_format,
        "page_number": page_number,
        "effective_year": effective_year,
        "temporal_status": temporal_status,
        "language": "ja",
        "retrieved_at": "2026-07-17T00:00:00+00:00",
        "text": text,
        "char_count": len(text),
        "estimated_tokens_ja": len(text),
        "content_hash_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "asset_ids": [],
        "keywords": [category, str(effective_year or "")],
        "representation": "plaintext",
        "table_handling": "narrative",
    }


def write_fixture_archive(path: Path) -> None:
    chunks = [
        make_chunk(
            "chunk_bad_tuition",
            IMPORTER.BAD_TUITION_DOC_ID,
            "学費・入学金について",
            "このHTMLは2026年4月入学者向けの学費ページであり、2027年度の根拠には使用しません。",
            source_url=IMPORTER.BAD_TUITION_URL,
            effective_year=2027,
            temporal_status="current_2027",
        ),
        make_chunk(
            "chunk_current_tuition",
            "doc_current_tuition",
            "2027年度学費等納入金",
            "2027年度入学者の学費は、学部と学科、学年、前期・後期によって異なります。入学金と授業料を確認してください。",
            source_url=IMPORTER.CURRENT_TUITION_PDF_URL,
            source_format="pdf",
            page_number=18,
            effective_year=2027,
            temporal_status="current_2027",
        ),
        make_chunk(
            "chunk_historical_tuition",
            "doc_historical_tuition",
            "2026年度学費",
            "2026年度入学者の学費に関する履歴情報です。2027年度の金額として使用してはいけません。",
            source_url="https://www.teu.ac.jp/entrance/history/tuition2026.html",
            effective_year=2026,
            temporal_status="historical_reference",
        ),
        make_chunk(
            "chunk_graduate",
            "doc_graduate",
            "大学院の研究科・専攻",
            "東京工科大学大学院には研究科と専攻があり、高度な教育研究と研究指導を行います。",
            source_url="https://www.teu.ac.jp/grad/index.html",
            section="grad",
            category="graduate_program",
            temporal_status="current_or_undated",
        ),
    ]
    jsonl = ("\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n").encode("utf-8")
    members = {
        "README.md": b"fixture\n",
        "chunks_plaintext.jsonl": jsonl,
    }
    checksum_text = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in members.items()
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(f"teu_rag_dataset/{name}", content)
        archive.writestr("teu_rag_dataset/checksums.sha256", checksum_text)


class TeuRagImportTests(unittest.TestCase):
    def test_import_verifies_filters_backs_up_and_preserves_temporal_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "dataset.zip"
            corpus = root / "corpus"
            sources = root / "sources"
            backups = root / "backups"
            corpus.mkdir()
            (corpus / "02_hachioji_faculties.md").write_text("legacy faculty", encoding="utf-8")
            (corpus / "04_admissions.md").write_text("legacy admissions", encoding="utf-8")
            write_fixture_archive(archive)

            manifest = IMPORTER.import_dataset(archive, corpus, sources, backups)

            self.assertEqual(manifest["input_chunks"], 4)
            self.assertEqual(manifest["imported_chunks"], 3)
            self.assertEqual(len(manifest["excluded_chunks"]), 1)
            self.assertFalse((corpus / "02_hachioji_faculties.md").exists())
            self.assertFalse((corpus / "04_admissions.md").exists())
            self.assertEqual(len(manifest["backup_files"]), 2)
            markdown = (corpus / IMPORTER.DEFAULT_OUTPUT_NAME).read_text(encoding="utf-8")
            self.assertNotIn("chunk_bad_tuition", markdown)
            self.assertIn("chunk_current_tuition", markdown)
            self.assertIn("page_number: 18", markdown)
            self.assertIn("effective_year: 2027", markdown)
            self.assertIn("temporal_status: current_2027", markdown)
            self.assertTrue((sources / "chunks_plaintext.jsonl").is_file())
            self.assertTrue((sources / "import_manifest.json").is_file())

            db_path = root / "local_rag.sqlite"
            LOCAL_RAG.build_index(corpus, db_path)
            current_hits = LOCAL_RAG.search_index(db_path, "学費", top_k=3)
            self.assertEqual(current_hits[0].chunk_id, "chunk_current_tuition")
            self.assertEqual(current_hits[0].effective_year, "2027")
            self.assertEqual(current_hits[0].page_number, "18")
            historical_hits = LOCAL_RAG.search_index(db_path, "2026年度の学費", top_k=3)
            self.assertEqual(historical_hits[0].chunk_id, "chunk_historical_tuition")
            graduate_hits = LOCAL_RAG.search_index(db_path, "大学院の研究科と専攻", top_k=3)
            self.assertEqual(graduate_hits[0].chunk_id, "chunk_graduate")
            citation = LOCAL_RAG.citations_for_hits([current_hits[0]])[0]
            self.assertEqual(citation["page_number"], "18")
            self.assertEqual(citation["effective_year"], "2027")
            self.assertEqual(citation["temporal_status"], "current_2027")

    def test_graduate_query_routes_to_faculties(self) -> None:
        decision = LOCAL_RAG.classify_query("大学院の研究科と専攻を教えてください")
        self.assertTrue(decision.route_candidate)
        self.assertEqual(decision.domain, "faculties")

    def test_archive_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../checksums.sha256", "")
            with self.assertRaises(IMPORTER.DatasetValidationError):
                IMPORTER.verify_archive(archive_path)


if __name__ == "__main__":
    unittest.main()
