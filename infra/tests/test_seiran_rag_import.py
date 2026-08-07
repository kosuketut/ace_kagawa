from __future__ import annotations

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


IMPORTER = load_module("seiran_rag_import_test_module", ROOT / "infra/rag/import_seiran_dataset.py")
LOCAL_RAG = load_module("seiran_local_rag_test_module", ROOT / "infra/rag/local_rag.py")


def write_fixture_archive(path: Path, *, mismatched_url: bool = False) -> None:
    source_url = "https://www.teu.ac.jp/information/2025.html?id=234"
    manifest = {
        "retrieved_at": "2026-07-19",
        "sources": [
            {
                "source_id": "S01",
                "title": "青嵐公式情報",
                "organization": "東京工科大学",
                "url": source_url,
                "published_at": "2025-10-07",
                "source_type": "大学公式お知らせ",
                "notes": "fixture",
            }
        ],
    }
    chunks = [
        {
            "id": "tut-seiran-001",
            "title": "青嵐の概要",
            "content": "青嵐は東京工科大学八王子キャンパスに設置されたGPUスーパーコンピュータで、研究・教育を支える計算基盤です。",
            "category": "概要",
            "keywords": ["青嵐", "AIスパコン"],
            "questions": ["スパコンについて教えてください"],
            "source_ids": ["S01"],
            "source_urls": ["https://top500.org/system/180436/" if mismatched_url else source_url],
            "fact_status": "verified",
            "temporal_scope": "",
            "answer_constraints": [],
            "notes": "",
            "retrieved_at": "2026-07-19",
        },
        {
            "id": "tut-seiran-029",
            "title": "利用申請の公開状況",
            "content": "一般公開ページでは利用申請、料金、ジョブキューの具体的な手順は確認できず、学内の最新案内が必要です。",
            "category": "利用",
            "keywords": ["利用申請", "料金"],
            "questions": ["利用料金を教えてください"],
            "source_ids": ["S01"],
            "source_urls": [source_url],
            "fact_status": "not_publicly_confirmed",
            "temporal_scope": "2026-07-19時点の一般公開情報",
            "answer_constraints": ["料金や申請URLを推測しない。"],
            "notes": "",
            "retrieved_at": "2026-07-19",
        },
        {
            "id": "tut-seiran-006",
            "title": "ノードとGPUの構成",
            "content": "青嵐はDGX B200を12ノード、NVIDIA B200 GPUを合計96基備えます。",
            "category": "ハードウェア",
            "keywords": ["DGX B200", "12ノード", "96 GPU", "GPU数"],
            "questions": ["青嵐にはGPUが何基ありますか？"],
            "source_ids": ["S01"],
            "source_urls": [source_url],
            "fact_status": "verified",
            "temporal_scope": "",
            "answer_constraints": [],
            "notes": "",
            "retrieved_at": "2026-07-19",
        },
    ]
    faqs = [
        {
            "id": "faq-001",
            "question": "東京工科大学のスパコン名は？",
            "answer": "青嵐です。",
            "source_ids": ["S01"],
            "source_urls": [source_url],
            "retrieved_at": "2026-07-19",
        }
    ]
    members = {
        "README.md": b"fixture\n",
        "source_manifest.csv": b"source_id,title\nS01,seiran\n",
        "source_manifest.json": (json.dumps(manifest, ensure_ascii=False) + "\n").encode(),
        "tut_seiran_chunks.jsonl": ("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks) + "\n").encode(),
        "tut_seiran_faq.jsonl": ("\n".join(json.dumps(row, ensure_ascii=False) for row in faqs) + "\n").encode(),
        "tut_seiran_knowledge.md": b"# fixture\n",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(f"tut_seiran_rag/{name}", content)


class SeiranRagImportTests(unittest.TestCase):
    def test_import_validates_normalizes_and_indexes_chunks_without_duplicate_faqs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "seiran.zip"
            corpus = root / "corpus"
            sources = root / "sources"
            backups = root / "backups"
            corpus.mkdir()
            write_fixture_archive(archive)

            manifest = IMPORTER.import_dataset(archive, corpus, sources, backups)

            self.assertEqual(manifest["validated_chunks"], 3)
            self.assertEqual(manifest["validated_faqs"], 1)
            self.assertEqual(manifest["faq_rows_indexed"], 0)
            self.assertEqual(manifest["faq_missing_fact_status"], ["faq-001"])
            markdown = (corpus / IMPORTER.DEFAULT_OUTPUT_NAME).read_text(encoding="utf-8")
            self.assertIn("record_type: seiran_verified", markdown)
            self.assertIn("record_type: seiran_not_publicly_confirmed", markdown)
            self.assertIn("料金や申請URLを推測しない", markdown)

            db_path = root / "local_rag.sqlite"
            LOCAL_RAG.build_index(corpus, db_path)
            decision = LOCAL_RAG.classify_query("スパコンについて教えてください")
            self.assertTrue(decision.route_candidate)
            self.assertEqual(decision.domain, "seiran")
            hits = LOCAL_RAG.search_index(db_path, "スパコンについて教えてください", top_k=3, domain="seiran")
            self.assertEqual(hits[0].chunk_id, "tut-seiran-001")
            gpu_hits = LOCAL_RAG.search_index(db_path, "青嵐のGPUは何基ですか", top_k=3, domain="seiran")
            self.assertEqual(gpu_hits[0].chunk_id, "tut-seiran-006")
            usage_hits = LOCAL_RAG.search_index(
                db_path,
                "青嵐の利用申請方法や料金を教えてください",
                top_k=3,
                domain="seiran",
            )
            self.assertEqual(usage_hits[0].chunk_id, "tut-seiran-029")

    def test_import_rejects_source_url_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "seiran.zip"
            write_fixture_archive(archive, mismatched_url=True)
            with self.assertRaisesRegex(IMPORTER.DatasetValidationError, "does not match"):
                IMPORTER.import_dataset(archive, root / "corpus", root / "sources", root / "backups")

    def test_archive_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../README.md", "unsafe")
            with self.assertRaises(IMPORTER.DatasetValidationError):
                IMPORTER.verify_archive(archive_path)


if __name__ == "__main__":
    unittest.main()
