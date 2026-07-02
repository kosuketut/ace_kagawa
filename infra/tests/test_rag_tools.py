from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RagToolTests(unittest.TestCase):
    def write_kagawa_profile_corpus(self, corpus_dir: Path) -> None:
        (corpus_dir / "kagawa_yutaka_rag_current.md").write_text(
            "# Kagawa Yutaka RAG Current\n\n"
            "## kagawa_yutaka_profile_basic\n\n"
            "source_title: 香川 豊 | 片柳研究所 | 教員紹介 | 大学・大学院案内\n"
            "source_priority: P0\n"
            "record_type: kagawa_profile\n"
            "chunk_id: chunk_kagawa_yutaka_profile_basic\n"
            "keywords: 香川豊, KAGAWA Yutaka, 東京工科大学, 片柳研究所, 学長, CMCセンター長\n\n"
            "香川豊先生は東京工科大学の学長であり、片柳研究所に所属する教授、"
            "セラミックス複合材料センター長である。学位は工学博士、専門分野は"
            "材料強度学、複合材料、高信頼性材料とされている。\n\n"
            "## kagawa_yutaka_birth_degree_fields\n\n"
            "source_title: 学長 香川 豊 | 片柳研究所 | 教員紹介 | 大学・大学院案内 | 東京工科大学\n"
            "source_priority: P0\n"
            "record_type: kagawa_profile\n"
            "chunk_id: chunk_kagawa_yutaka_birth_degree_fields\n"
            "keywords: 生年月日, 東京生まれ, 工学博士, 材料強度学, 複合材料, 高信頼性材料\n\n"
            "東京工科大学の学長プロフィールでは、香川豊先生は1952年9月19日東京生まれ、"
            "工学博士とされる。専門分野は材料強度学、複合材料、高信頼性材料である。\n\n"
            "## kagawa_yutaka_education\n\n"
            "source_title: 学長 香川 豊 | 片柳研究所 | 教員紹介 | 大学・大学院案内 | 東京工科大学\n"
            "source_priority: P0\n"
            "record_type: kagawa_education\n"
            "chunk_id: chunk_kagawa_yutaka_education\n"
            "keywords: 学歴, 早稲田大学, 理工学部, 博士前期課程, 博士後期課程, 理学修士, 工学博士\n\n"
            "香川豊先生の学歴は、東京工科大学公式プロフィールで早稲田大学を中心に確認できる。"
            "1976年に早稲田大学理工学部を卒業し、1978年に早稲田大学大学院理工学研究科"
            "博士前期課程を修了して理学修士となった。その後、1984年に同研究科博士後期課程を"
            "修了し、工学博士となっている。\n\n"
            "## kagawa_yutaka_career_tut_roles\n\n"
            "source_title: 学長 香川 豊 | 片柳研究所 | 教員紹介 | 大学・大学院案内 | 東京工科大学\n"
            "source_priority: P0\n"
            "record_type: kagawa_career\n"
            "chunk_id: chunk_kagawa_yutaka_career_tut_roles\n"
            "keywords: 東京工科大学, 片柳研究所長, セラミックス複合材料センター長, 副学長, 学長, 東京大学名誉教授, 役職\n\n"
            "東京工科大学公式プロフィールでは、2017年に香川豊先生が東京工科大学教授、"
            "片柳研究所長、セラミックス複合材料センター長となり、同年に東京大学名誉教授となった"
            "ことが示されている。2019年には東京工科大学副学長、2023年には東京工科大学学長に"
            "就任している。現在の教員紹介では、教授／学長、セラミックス複合材料センター長が"
            "確認できる。\n\n"
            "## kagawa_yutaka_kaken_keywords\n\n"
            "source_title: KAKEN — 研究者をさがす | 香川 豊 (50152591)\n"
            "source_priority: P1\n"
            "record_type: kagawa_research_keywords\n"
            "chunk_id: chunk_kagawa_yutaka_kaken_keywords\n"
            "keywords: 複合材料, 界面力学特性, SiC繊維, CMC, EBC, 非破壊検査, 電磁波, 損傷\n\n"
            "KAKENの研究者ページでは、香川豊先生の研究分野として複合材料・物性、無機材料・物性、"
            "金属材料、材料加工・処理、複合材料および界面関連などが確認できる。キーワードには、"
            "複合材料、界面力学特性、SiC繊維、界面せん断滑り応力、コーティング、界面剥離、"
            "CMC、SiC/SiC、EBC、耐環境コーティング、非破壊検査、電磁波、損傷、誘電率などが"
            "含まれる。\n\n"
            "## kagawa_yutaka_additional_research_todo\n\n"
            "source_title: 調査メモ: 香川豊先生RAG追加調査項目\n"
            "source_priority: P3\n"
            "record_type: kagawa_rag_todo\n"
            "chunk_id: chunk_kagawa_yutaka_additional_research_todo\n"
            "keywords: 追加調査, DOI, 被引用数, researchmap, ORCID, 東京大学, 受賞歴, 学会役職\n\n"
            "RAG品質を上げるには、代表論文のDOIと被引用数、researchmapページ、ORCID、"
            "東京大学時代の公式プロフィール、受賞歴と学会役職を追加確認する必要がある。\n",
            encoding="utf-8",
        )

    def test_ingest_script_can_reset_collection_before_reingest(self) -> None:
        script = (ROOT / "infra" / "rag" / "ingest_local_corpus.sh").read_text(encoding="utf-8")

        self.assertIn("--reset-collection", script)
        self.assertIn("delete_collection", script)
        self.assertIn("-X DELETE", script)
        self.assertIn('/v1/collections"', script)

    def test_local_rag_index_finds_japanese_corpus_chunks(self) -> None:
        local_rag = load_module("local_rag_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "kagawa.md").write_text(
                "# 香川豊\n\n"
                "香川豊先生の専門分野は材料強度学、複合材料、CMC、SiC/SiC複合材料です。\n"
                "研究テーマにはEBCと非破壊評価が含まれます。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            stats = local_rag.build_index(corpus_dir, db_path, chunk_chars=80, chunk_overlap=10)
            hits = local_rag.search_index(db_path, "香川先生の専門分野は何ですか", top_k=2)
            context = local_rag.format_hits_for_prompt(hits, max_chars=400)

            self.assertEqual(stats["files_indexed"], 1)
            self.assertGreaterEqual(stats["chunks_indexed"], 1)
            self.assertTrue(db_path.is_file())
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("複合材料", hits[0].text)
            self.assertIn("参考情報", context)
            self.assertIn("kagawa.md", context)

    def test_local_rag_cli_builds_and_queries_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "school.md").write_text(
                "# 東京工科大学\n\n八王子キャンパスには応用生物学部と工学部があります。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            build = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "infra" / "rag" / "build_local_index.py"),
                    "--corpus",
                    str(corpus_dir),
                    "--db",
                    str(db_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            query = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "infra" / "rag" / "query_local_index.py"),
                    "--db",
                    str(db_path),
                    "--top-k",
                    "1",
                    "八王子キャンパスの学部",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            build_payload = json.loads(build.stdout)
            query_payload = json.loads(query.stdout)
            self.assertEqual(build_payload["files_indexed"], 1)
            self.assertEqual(query_payload["query"], "八王子キャンパスの学部")
            self.assertEqual(len(query_payload["hits"]), 1)
            self.assertIn("応用生物学部", query_payload["hits"][0]["text"])

    def test_local_rag_reranks_dedicated_access_document_above_noisy_matches(self) -> None:
        local_rag = load_module("local_rag_rerank_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "01_hachioji_access.md").write_text(
                "# Hachioji Access And Locations\n\n"
                "八王子キャンパスへは、JR中央線「八王子」駅からスクールバス約10分です。\n",
                encoding="utf-8",
            )
            (corpus_dir / "07_visitor_pages.md").write_text(
                "# Visitor Pages\n\n"
                "電子ブックは学外からアクセスできます。八王子キャンパスのお知らせもあります。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=120, chunk_overlap=10)
            hits = local_rag.search_index(db_path, "八王子キャンパスのアクセス", top_k=2)

            self.assertGreaterEqual(len(hits), 1)
            self.assertEqual(hits[0].path, "01_hachioji_access.md")

    def test_local_rag_filters_access_sections_for_requested_campus(self) -> None:
        local_rag = load_module("local_rag_access_filter_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "01_hachioji_access.md").write_text(
                "# Hachioji Access And Locations\n\n"
                "## access_kamata_from_kamata_station\n\n"
                "record_type: access_route\n"
                "chunk_id: chunk_access_kamata_from_kamata_station\n\n"
                "蒲田キャンパスへは、JR蒲田駅西口から徒歩2分です。\n\n"
                "## access_hachioji_from_hachioji_station\n\n"
                "record_type: access_route\n"
                "chunk_id: chunk_access_hachioji_from_hachioji_station\n\n"
                "八王子キャンパスへは、JR八王子駅からスクールバス約10分です。\n\n"
                "## access_hachioji_from_minamino_station\n\n"
                "record_type: access_route\n"
                "chunk_id: chunk_access_hachioji_from_minamino_station\n\n"
                "八王子キャンパスへは、JR八王子みなみ野駅からスクールバス約5分です。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=200, chunk_overlap=0)
            hits = local_rag.search_index(db_path, "八王子キャンパスへのアクセスは？", top_k=3)

            self.assertEqual(len(hits), 2)
            self.assertTrue(all("八王子" in hit.text for hit in hits))

    def test_local_rag_prefers_admission_sections_for_selection_and_calendar_queries(self) -> None:
        local_rag = load_module("local_rag_admission_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "04_admissions.md").write_text(
                "# Admissions\n\n"
                "## admission_2027_calendar\n\n"
                "source_type: admissions\n"
                "source_priority: P1\n"
                "record_type: admission\n"
                "chunk_id: chunk_admission_2027_calendar\n\n"
                "2027年度入試カレンダーが入試・入学案内ページに掲載されています。\n\n"
                "## admission_2027_ao_all\n\n"
                "source_type: admissions\n"
                "source_priority: P1\n"
                "record_type: admission\n"
                "chunk_id: chunk_admission_2027_ao_all\n\n"
                "2027年度の総合型選抜（全学部AO入試）が入試・入学案内ページに掲載されています。\n\n"
                "## admission_2027_tuition\n\n"
                "source_type: admissions\n"
                "source_priority: P1\n"
                "record_type: admission\n"
                "chunk_id: chunk_admission_2027_tuition\n\n"
                "学費・入学金についての入口が入試・入学案内ページに掲載されています。\n",
                encoding="utf-8",
            )
            (corpus_dir / "00_hachioji_open_campus_current.md").write_text(
                "# Hachioji Open Campus Current\n\n"
                "## oc_selection_event\n\n"
                "source_type: open_campus_html\n"
                "source_priority: P0\n"
                "record_type: oc_program\n"
                "chunk_id: chunk_oc_selection_event\n\n"
                "オープンキャンパスでは総合型選抜対策講座が予定されています。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=300, chunk_overlap=0)
            selection_hit = local_rag.search_index(db_path, "総合型選抜について教えてください", top_k=1)[0]
            calendar_hit = local_rag.search_index(db_path, "出願や入試日程は？", top_k=1)[0]

            self.assertEqual(selection_hit.chunk_id, "chunk_admission_2027_ao_all")
            self.assertEqual(calendar_hit.chunk_id, "chunk_admission_2027_calendar")

    def test_local_rag_prefers_common_test_and_scholarship_exam_sections(self) -> None:
        local_rag = load_module("local_rag_admission_specific_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "04_admissions.md").write_text(
                "# Admissions\n\n"
                "## admission_2027_common_test\n\n"
                "source_type: admissions\n"
                "source_priority: P1\n"
                "record_type: admission\n"
                "chunk_id: chunk_admission_2027_common_test\n\n"
                "共通テスト利用試験は前期・中期・後期の入口が入試・入学案内ページに掲載されています。\n\n"
                "## admission_2027_scholarship\n\n"
                "source_type: admissions\n"
                "source_priority: P1\n"
                "record_type: admission\n"
                "chunk_id: chunk_admission_2027_scholarship\n\n"
                "奨学金についての入口が入試・入学案内ページに掲載されています。\n",
                encoding="utf-8",
            )
            (corpus_dir / "07_visitor_pages.md").write_text(
                "# Visitor Pages\n\n"
                "## visitor_admission_shogakusei_01\n\n"
                "source_type: visitor\n"
                "source_priority: P1\n"
                "record_type: visitor_page\n"
                "chunk_id: chunk_visitor_admission_shogakusei_01\n\n"
                "2027年度入試情報の奨学生入試ページでは、対象学部、入試日程、"
                "試験場、選抜方法、試験教科・科目が案内されています。\n\n"
                "## visitor_tuition_01\n\n"
                "source_type: visitor\n"
                "source_priority: P1\n"
                "record_type: visitor_page\n"
                "chunk_id: chunk_visitor_tuition_01\n\n"
                "学費・入学金と奨学金について案内しています。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=500, chunk_overlap=0)
            common_hit = local_rag.search_index(db_path, "共通テスト利用試験について教えてください", top_k=1)[0]
            shogakusei_hit = local_rag.search_index(db_path, "奨学生入試について教えてください", top_k=1)[0]

            self.assertEqual(common_hit.chunk_id, "chunk_admission_2027_common_test")
            self.assertEqual(shogakusei_hit.chunk_id, "chunk_visitor_admission_shogakusei_01")

    def test_local_rag_prefers_matching_kagawa_research_publication_and_career_sections(self) -> None:
        local_rag = load_module("local_rag_intent_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "kagawa_yutaka_rag_current.md").write_text(
                "# Kagawa Yutaka RAG Current\n\n"
                "## kagawa_yutaka_president_appointment\n\n"
                "source_title: 学長就任\n"
                "source_priority: P0\n"
                "record_type: kagawa_president\n"
                "chunk_id: chunk_kagawa_yutaka_president_appointment\n"
                "keywords: 学長就任, 東京工科大学, 実学\n\n"
                "香川豊先生は2023年4月1日に東京工科大学学長に就任した。"
                "大学の教育研究を推進している。\n\n"
                "## kagawa_yutaka_kaken_keywords\n\n"
                "source_title: KAKEN 研究キーワード\n"
                "source_priority: P1\n"
                "record_type: kagawa_research_keywords\n"
                "chunk_id: chunk_kagawa_yutaka_kaken_keywords\n"
                "keywords: 複合材料, CMC, EBC, 非破壊検査, 界面力学特性\n\n"
                "香川豊先生の研究分野には複合材料、CMC、EBC、非破壊検査、"
                "界面力学特性が含まれる。RAGでは研究内容の質問でこの情報を優先する。\n\n"
                "## kagawa_yutaka_additional_research_todo\n\n"
                "source_title: 調査メモ\n"
                "source_priority: P3\n"
                "record_type: kagawa_rag_todo\n"
                "chunk_id: chunk_kagawa_yutaka_additional_research_todo\n"
                "keywords: 研究内容, 複合材料, CMC, EBC, research, researchmap, 代表論文, 受賞歴\n\n"
                "香川豊先生の研究内容、複合材料、CMC、EBCに関するRAG品質を上げるには、"
                "代表論文、researchmap、受賞歴を追加調査する必要がある。\n\n"
                "## kagawa_yutaka_publication\n\n"
                "source_title: 代表論文\n"
                "source_priority: P1\n"
                "record_type: kagawa_publication\n"
                "chunk_id: chunk_kagawa_yutaka_publication\n"
                "keywords: 論文, 業績, Journal of Composites Science, SiC/SiC\n\n"
                "代表的な業績として、SiC/SiC複合材料の非破壊評価に関する論文がある。\n\n"
                "## kagawa_yutaka_career\n\n"
                "source_title: 公式プロフィール\n"
                "source_priority: P0\n"
                "record_type: kagawa_career\n"
                "chunk_id: chunk_kagawa_yutaka_career\n"
                "keywords: 経歴, 東京大学, 東京工科大学, 副学長, 学長\n\n"
                "香川豊先生は東京大学で教授を務めた後、東京工科大学で副学長、"
                "学長を務めている。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=500, chunk_overlap=0)

            research_hit = local_rag.search_index(db_path, "香川先生の研究内容について教えてください", top_k=1)[0]
            publication_hit = local_rag.search_index(db_path, "香川先生の論文や業績を教えてください", top_k=1)[0]
            career_hit = local_rag.search_index(db_path, "香川先生の経歴を教えてください", top_k=1)[0]
            ebc_hit = local_rag.search_index(db_path, "EBCとは何ですか", top_k=1)[0]

            self.assertEqual(research_hit.record_type, "kagawa_research_keywords")
            self.assertEqual(publication_hit.record_type, "kagawa_publication")
            self.assertEqual(career_hit.record_type, "kagawa_career")
            self.assertEqual(ebc_hit.record_type, "kagawa_research_keywords")
            self.assertEqual(research_hit.chunk_id, "chunk_kagawa_yutaka_kaken_keywords")
            self.assertNotEqual(research_hit.record_type, "kagawa_rag_todo")

    def test_local_rag_prefers_kagawa_education_for_school_history_query(self) -> None:
        local_rag = load_module("local_rag_kagawa_education_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            self.write_kagawa_profile_corpus(corpus_dir)
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=900, chunk_overlap=0)
            hit = local_rag.search_index(db_path, "香川先生の学歴を教えてください", top_k=1)[0]

            self.assertEqual(hit.chunk_id, "chunk_kagawa_yutaka_education")

    def test_local_rag_excludes_kagawa_todo_from_current_role_queries(self) -> None:
        local_rag = load_module("local_rag_kagawa_role_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            self.write_kagawa_profile_corpus(corpus_dir)
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=900, chunk_overlap=0)
            hits = local_rag.search_index(db_path, "香川先生の現在の役職は？", top_k=3)

            self.assertEqual(hits[0].chunk_id, "chunk_kagawa_yutaka_profile_basic")
            self.assertTrue(all(hit.record_type != "kagawa_rag_todo" for hit in hits))

    def test_local_rag_prefers_official_profile_for_kagawa_field_query(self) -> None:
        local_rag = load_module("local_rag_kagawa_field_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            self.write_kagawa_profile_corpus(corpus_dir)
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=900, chunk_overlap=0)
            hit = local_rag.search_index(db_path, "香川先生の専門分野は何ですか", top_k=1)[0]

            self.assertEqual(hit.chunk_id, "chunk_kagawa_yutaka_birth_degree_fields")

    def test_local_rag_prefers_birth_profile_for_kagawa_age_query(self) -> None:
        local_rag = load_module("local_rag_kagawa_age_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            self.write_kagawa_profile_corpus(corpus_dir)
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=900, chunk_overlap=0)
            hit = local_rag.search_index(db_path, "香川先生の年齢は？", top_k=1)[0]

            self.assertEqual(hit.chunk_id, "chunk_kagawa_yutaka_birth_degree_fields")

    def test_local_rag_prompt_context_uses_metadata_without_raw_operational_notes(self) -> None:
        local_rag = load_module("local_rag_context_test", ROOT / "infra" / "rag" / "local_rag.py")

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "kagawa.md").write_text(
                "# Kagawa\n\n"
                "## research\n\n"
                "source_title: KAKEN 研究キーワード\n"
                "source_url: https://example.invalid/kaken\n"
                "published_date: 未確認\n"
                "source_priority: P1\n"
                "record_type: kagawa_research_keywords\n"
                "chunk_id: chunk_research\n"
                "keywords: CMC, EBC\n\n"
                "香川豊先生はCMCとEBCを研究している。RAGでは検索用の運用文を保持する。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"

            local_rag.build_index(corpus_dir, db_path, chunk_chars=500, chunk_overlap=0)
            hits = local_rag.search_index(db_path, "EBCとは何ですか", top_k=1)
            context = local_rag.format_hits_for_prompt(hits, max_chars=400)
            payload = local_rag.hits_to_json_payload("EBCとは何ですか", hits, max_context_chars=400)

            self.assertIn("KAKEN 研究キーワード", context)
            self.assertIn("香川豊先生はCMCとEBCを研究している。", context)
            self.assertNotIn("source_title:", context)
            self.assertNotIn("RAGでは", context)
            self.assertEqual(payload["hits"][0]["record_type"], "kagawa_research_keywords")
            self.assertEqual(payload["hits"][0]["source_url"], "https://example.invalid/kaken")


if __name__ == "__main__":
    unittest.main()
