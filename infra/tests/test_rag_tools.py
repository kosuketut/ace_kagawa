from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from unittest import mock
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

    def test_evaluation_suite_covers_reported_routing_boundaries(self) -> None:
        payload = json.loads((ROOT / "infra" / "rag" / "evaluation_cases.json").read_text(encoding="utf-8"))
        case_ids = {case["id"] for case in payload["cases"]}
        evaluator = (ROOT / "infra" / "rag" / "evaluate_local_rag.py").read_text(encoding="utf-8")

        self.assertTrue(
            {
                "kagawa_research_generic",
                "kagawa_research_project",
                "kagawa_affiliation",
                "tuition_short_term",
                "scholarship_without_admission_word",
                "qualified_ao_admission",
                "faculty_major",
                "cs_admissions_2027_detailed",
                "cs_admissions_asr_without_year",
                "cs_admissions_fragmented_asr_year",
                "cs_admission_subjects",
                "student_support",
                "research_followup",
            }.issubset(case_ids)
        )
        self.assertIn("must_rag_domain_accuracy", evaluator)
        self.assertIn("must_rag_retrieval_success_rate", evaluator)
        self.assertIn("citation_id_duplicate_cases", evaluator)
        self.assertIn("coverage_limitations", evaluator)

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
            tuition_hit = local_rag.search_index(db_path, "東京工科大学の学費を教えてください", top_k=1)[0]

            self.assertEqual(selection_hit.chunk_id, "chunk_admission_2027_ao_all")
            self.assertEqual(calendar_hit.chunk_id, "chunk_admission_2027_calendar")
            self.assertEqual(tuition_hit.chunk_id, "chunk_admission_2027_tuition")

    def test_local_rag_keeps_graduate_admissions_separate_from_undergraduate_admissions(self) -> None:
        local_rag = load_module(
            "local_rag_graduate_admission_test",
            ROOT / "infra" / "rag" / "local_rag.py",
        )

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "admissions.md").write_text(
                "# Admissions\n\n"
                "## undergraduate_admission\n\n"
                "source_priority: P0\n"
                "record_type: admission\n"
                "chunk_id: chunk_undergraduate_admission\n"
                "keywords: 入試, 総合型選抜, 募集人員\n\n"
                "学部の総合型選抜について、募集人員、日程、選抜方法を案内します。\n\n"
                "## graduate_admission\n\n"
                "source_priority: P1\n"
                "record_type: graduate_profile\n"
                "chunk_id: chunk_graduate_admission\n"
                "keywords: graduate_admission, 大学院, 入試\n\n"
                "大学院入試について、募集課程、出願資格、日程、選抜方法を案内します。\n\n"
                "## graduate_admission_calendar\n\n"
                "source_priority: P0\n"
                "record_type: graduate_profile\n"
                "chunk_id: chunk_graduate_admission_calendar\n"
                "keywords: graduate_admission, admission_calendar, 大学院, 入試日程\n\n"
                "2027年度大学院入試のA日程の入学試験日は2026年8月29日、"
                "B日程の入学試験日は2027年1月30日です。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path, chunk_chars=500, chunk_overlap=0)

            query = "大学院の入試について教えて"
            route = local_rag.classify_query(query)
            hits = local_rag.search_index(db_path, query, top_k=3)
            assessment = local_rag.assess_local_rag_query(query, hits)

            self.assertEqual(route.domain, "graduate_admissions")
            self.assertTrue(assessment.accepted)
            self.assertEqual({hit.record_type for hit in hits}, {"graduate_profile"})
            self.assertEqual(hits[0].chunk_id, "chunk_graduate_admission")
            self.assertEqual(
                local_rag.grounded_direct_reply(query, hits),
                "大学院入試では、募集課程、出願資格、日程、選抜方法などをご案内しています。",
            )
            calendar_hit = local_rag.LocalRagHit(
                path="graduate_calendar.md",
                title="2027年度大学院入試日程",
                section_title="大学院入試日程",
                source_title="大学院入試概要",
                source_url="https://www.teu.ac.jp/grad/entrance/index.html",
                publisher="東京工科大学",
                published_date="",
                accessed_date="2026-07-28",
                source_type="teu_official_html",
                source_priority="P0",
                record_type="graduate_profile",
                chunk_id="chunk_graduate_admission_2027_calendar",
                keywords="graduate_admission, admission_calendar, 大学院, 入試日程, 工学研究科",
                chunk_index=0,
                text=(
                    "2027年度大学院入試のA日程の出願期間は2026年7月28日から7月30日、"
                    "入学試験日は2026年8月29日、合格発表日は2026年9月8日、"
                    "入学手続期限は2026年10月1日。B日程の出願期間は"
                    "2027年1月5日から1月7日、入学試験日は2027年1月30日、"
                    "合格発表日は2027年2月16日、入学手続期限は2027年2月24日。"
                ),
                score=900.0,
                effective_year="2027",
                temporal_status="current_2027",
            )
            self.assertEqual(
                local_rag.grounded_direct_reply(
                    "大学院入試の日程について教えてください",
                    [calendar_hit],
                ),
                (
                    "2027年度大学院入試の試験日は、A日程が2026年8月29日、"
                    "B日程が2027年1月30日です。出願期間は研究科により異なります。"
                ),
            )
            self.assertEqual(
                local_rag.grounded_direct_reply(
                    "大学院入試の合格発表日はいつですか",
                    [calendar_hit],
                ),
                "2027年度大学院入試の合格発表日は、A日程が2026年9月8日、B日程が2027年2月16日です。",
            )
            self.assertEqual(
                local_rag.grounded_direct_reply(
                    "工学研究科の出願期間を教えてください",
                    [calendar_hit],
                ),
                "出願期間は、A日程が2026年7月28日から7月30日、B日程が2027年1月5日から1月7日です。",
            )
            self.assertEqual(
                local_rag.grounded_direct_reply("大学院入試の選抜方法を教えてください", hits),
                "大学院入試の選抜方法は研究科・課程で異なります。研究科名と課程を指定してください。",
            )
            self.assertEqual(
                local_rag.grounded_direct_reply("大学院入試の出願資格を教えてください", hits),
                "大学院入試の出願資格は研究科・課程で異なります。研究科名と課程を指定してください。",
            )
            self.assertEqual(
                local_rag.grounded_direct_reply("大学院の学費について教えてください", hits),
                "現在の参照情報では、大学院の学費の正確な金額を確認できません。",
            )
            self.assertIsNone(local_rag.grounded_direct_reply("大学院の専攻を教えて", hits))

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

    def test_grounded_direct_reply_covers_current_open_campus_and_common_profile_routes(self) -> None:
        local_rag = load_module("local_rag_direct_reply_test", ROOT / "infra" / "rag" / "local_rag.py")

        def hit(**overrides):
            payload = {
                "path": "official.md",
                "title": "Official",
                "section_title": "Official",
                "source_title": "公式情報",
                "source_url": "https://example.invalid/official",
                "publisher": "東京工科大学",
                "published_date": "",
                "accessed_date": "2026-07-29",
                "source_type": "official_html",
                "source_priority": "P0",
                "record_type": "oc_event",
                "chunk_id": "chunk_oc",
                "keywords": "open_campus",
                "chunk_index": 0,
                "text": "",
                "score": 200.0,
            }
            payload.update(overrides)
            return local_rag.LocalRagHit(**payload)

        open_campus_hits = [
            hit(
                chunk_id="chunk_oc_kamata",
                text="蒲田キャンパスでは2026年8月1日と2026年8月16日に開催予定です。",
            ),
            hit(
                chunk_id="chunk_oc_hachioji",
                text="八王子キャンパスでは2026年8月2日と2026年8月23日に開催予定です。",
            ),
        ]
        self.assertEqual(
            local_rag.grounded_direct_reply(
                "次のオープンキャンパスはいつですか",
                open_campus_hits,
                today=date(2026, 7, 29),
            ),
            "次回は8月1日の蒲田キャンパスです。八王子キャンパスは8月2日です。",
        )

        access_hit = hit(
            record_type="access_route",
            chunk_id="chunk_access_hachioji",
            keywords="access",
            text="八王子キャンパスへは、八王子駅からスクールバスで約10分です。",
        )
        self.assertEqual(
            local_rag.grounded_direct_reply("八王子キャンパスへのアクセスを教えてください", [access_hit]),
            access_hit.text,
        )

        profile_hit = hit(
            record_type="kagawa_profile",
            chunk_id="chunk_kagawa_profile",
            keywords="香川豊,プロフィール",
            text="香川豊は東京工科大学の学長・教授で、セラミックス複合材料センター長である。",
        )
        self.assertEqual(
            local_rag.grounded_direct_reply("香川先生について教えてください", [profile_hit]),
            "香川豊は東京工科大学の学長・教授で、セラミックス複合材料センター長です。",
        )

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
                "publisher: 日本学術振興会\n"
                "published_date: 未確認\n"
                "accessed_date: 2026-07-16\n"
                "source_type: official_database\n"
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
            self.assertEqual(payload["citations"][0]["source_title"], "KAKEN 研究キーワード")
            self.assertEqual(payload["citations"][0]["path"], "kagawa.md")
            self.assertEqual(payload["citations"][0]["source_url"], "https://example.invalid/kaken")
            self.assertEqual(payload["citations"][0]["chunk_id"], "chunk_research")
            self.assertEqual(payload["citations"][0]["publisher"], "日本学術振興会")
            self.assertEqual(payload["citations"][0]["accessed_date"], "2026-07-16")
            self.assertEqual(payload["citations"][0]["date"], "2026-07-16")
            self.assertEqual(payload["citations"][0]["date_type"], "accessed")
            self.assertEqual(payload["citations"][0]["source_type"], "official_database")
            self.assertEqual(payload["citations"][0]["document_id"], "kagawa.md#chunk_research:0")

    def test_current_prompt_context_removes_source_guidance_and_unverified_hit(self) -> None:
        local_rag = load_module("local_rag_current_guidance_test", ROOT / "infra" / "rag" / "local_rag.py")
        current_hit = local_rag.LocalRagHit(
            path="admissions.md",
            title="Admissions",
            section_title="2027年度総合型選抜",
            source_title="2027年度総合型選抜",
            source_url="https://example.invalid/current.pdf",
            publisher="東京工科大学",
            published_date="",
            accessed_date="2026-07-17",
            source_type="teu_dataset_pdf",
            source_priority="P0",
            record_type="admission",
            chunk_id="chunk_current_2027",
            keywords="総合型選抜",
            chunk_index=0,
            text=(
                "2027年度は書類審査、基礎学力試験、面接試験で選抜する。"
                "主要数値と日程は公式募集要項PDFの表テキストを参照する。"
            ),
            score=300.0,
            effective_year="2027",
            temporal_status="current_2027",
        )
        unverified_hit = local_rag.LocalRagHit(
            path="admissions.md",
            title="Admissions",
            section_title="学校推薦型選抜",
            source_title="学校推薦型選抜",
            source_url="https://example.invalid/recommend_2025.html",
            publisher="東京工科大学",
            published_date="",
            accessed_date="2026-07-17",
            source_type="teu_dataset_html",
            source_priority="P1",
            record_type="admission",
            chunk_id="chunk_versioned",
            keywords="学校推薦型選抜",
            chunk_index=1,
            text="URLの年度と募集要項を必ず確認し、RAG回答では年度を混同しない。",
            score=250.0,
            temporal_status="versioned_page_verify_year",
        )

        context, included = local_rag.format_hits_for_prompt_with_sources(
            [current_hit, unverified_hit],
            query="学校推薦型選抜について教えてください",
        )

        self.assertEqual([hit.chunk_id for hit in included], ["chunk_current_2027"])
        self.assertIn("2027年度は書類審査、基礎学力試験、面接試験で選抜する。", context)
        self.assertIn("定型文を追加しない", context)
        self.assertNotIn("公式募集要項PDFの表テキストを参照する", context)
        self.assertNotIn("必ず確認", context)
        self.assertNotIn("RAG回答", context)

    def test_historical_prompt_context_keeps_necessary_temporal_caveat(self) -> None:
        local_rag = load_module("local_rag_historical_guidance_test", ROOT / "infra" / "rag" / "local_rag.py")
        historical_hit = local_rag.LocalRagHit(
            path="admissions.md",
            title="Admissions",
            section_title="2026年度入試結果",
            source_title="2026年度入試結果",
            source_url="https://example.invalid/2026-results.html",
            publisher="東京工科大学",
            published_date="",
            accessed_date="2026-07-17",
            source_type="teu_dataset_html",
            source_priority="P1",
            record_type="admission",
            chunk_id="chunk_results_2026",
            keywords="入試結果",
            chunk_index=0,
            text=(
                "2026年度入試の志願者数と合格者数を示す履歴データ。"
                "現在の募集条件には使用せず、当該年度募集要項を優先する。"
            ),
            score=300.0,
            effective_year="2026",
            temporal_status="historical_result",
        )

        context, included = local_rag.format_hits_for_prompt_with_sources(
            [historical_hit],
            query="2026年度の入試結果を教えてください",
        )

        self.assertEqual([hit.chunk_id for hit in included], ["chunk_results_2026"])
        self.assertIn("当該年度募集要項を優先する", context)
        self.assertIn("不確実性を必要最小限", context)

    def test_citation_document_ids_remain_unique_when_section_splits(self) -> None:
        local_rag = load_module("local_rag_citation_id_test", ROOT / "infra" / "rag" / "local_rag.py")
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            repeated = "EBCとCMCの研究成果です。" * 30
            (corpus_dir / "research.md").write_text(
                "# Research\n\ngenerated_at: 2026-07-16T12:00:00+09:00\n\n## projects\n\n"
                "source_url: https://www.teu.ac.jp/research\n"
                "source_priority: P1\n"
                "record_type: kagawa_research_project\n"
                "chunk_id: chunk_projects\n\n"
                f"{repeated}\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path, chunk_chars=120, chunk_overlap=10)

            hits = local_rag.search_index(db_path, "EBCとCMCの研究成果", top_k=3)
            document_ids = [citation["document_id"] for citation in local_rag.citations_for_hits(hits)]
            chunk_ids = [hit.chunk_id for hit in hits]
            payload = local_rag.hits_to_json_payload("EBCとCMCの研究成果", hits, max_context_chars=220)

            self.assertGreaterEqual(len(document_ids), 2)
            self.assertEqual(len(document_ids), len(set(document_ids)))
            self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
            self.assertEqual(len(payload["citations"]), payload["context"].count("[source "))
            self.assertTrue(all(hit.accessed_date == "2026-07-16T12:00:00+09:00" for hit in hits))
            self.assertTrue(all(hit.publisher == "東京工科大学" for hit in hits))

    def test_query_classifier_covers_supported_domains_without_generic_detail_routing(self) -> None:
        local_rag = load_module("local_rag_classifier_test", ROOT / "infra" / "rag" / "local_rag.py")
        must_rag = {
            "香川先生の専門分野は何ですか": "kagawa_profile",
            "香川先生の経歴を教えてください": "kagawa_profile",
            "香川先生の所属を教えてください": "kagawa_profile",
            "香川先生について教えてください": "kagawa_profile",
            "香川先生の研究について教えてください": "kagawa_research",
            "香川先生の研究プロジェクトを教えてください": "kagawa_research",
            "EBCとは何ですか": "kagawa_research",
            "八王子キャンパスへのアクセスは": "access",
            "総合型選抜について教えてください": "admissions",
            "次のオープンキャンパスはいつですか": "open_campus",
            "八王子キャンパスにはどの学部がありますか": "faculties",
            "東京工科大学の概要を教えてください": "university",
            "東京工科大学の学生支援について教えてください": "university",
            "2027年度の学費について教えてください": "admissions",
            "奨学金について教えてください": "admissions",
            "コンピュータサイエンス学部の専攻を教えてください": "faculties",
            "大学案内パンフレットはどこで見られますか": "pamphlet",
            "スパコンについて教えてください": "seiran",
            "青嵐のGPUは何基ですか": "seiran",
            "SEIRANのTOP500順位は何位ですか": "seiran",
        }
        for query, domain in must_rag.items():
            with self.subTest(query=query):
                decision = local_rag.classify_query(query)
                self.assertTrue(decision.route_candidate)
                self.assertEqual(decision.domain, domain)

        must_not_rag = {
            "おはようございます": "greeting",
            "あなたの名前は": "assistant_name",
            "今日の天気は": "weather",
            "今日の天気を詳しく教えてください": "weather",
            "おすすめの夕食を教えてください": "general_chitchat",
            "詳しく教えてください": "outside_supported_domains",
        }
        for query, reason in must_not_rag.items():
            with self.subTest(query=query):
                decision = local_rag.classify_query(query)
                self.assertFalse(decision.route_candidate)
                self.assertEqual(decision.reason, reason)

        configured = local_rag.classify_query("香川先生の共同事業について", ["共同事業"])
        self.assertTrue(configured.route_candidate)
        self.assertEqual(configured.domain, "kagawa_research")
        self.assertTrue(configured.reason.startswith("configured_route_keyword:"))

    def test_confidence_rejects_unknown_person_and_low_relevance(self) -> None:
        local_rag = load_module("local_rag_confidence_test", ROOT / "infra" / "rag" / "local_rag.py")
        hit = local_rag.LocalRagHit(
            path="kagawa.md",
            title="香川豊",
            section_title="経歴",
            source_title="公式プロフィール",
            source_url="https://example.invalid/kagawa",
            publisher="東京工科大学",
            published_date="2026-01-01",
            accessed_date="2026-07-16",
            source_type="official_profile",
            source_priority="P0",
            record_type="kagawa_career",
            chunk_id="chunk_kagawa_career",
            keywords="香川豊,経歴",
            chunk_index=0,
            text="香川豊先生の経歴です。",
            score=220.0,
        )

        accepted = local_rag.assess_local_rag_query("香川先生の経歴を教えてください", [hit])
        unknown = local_rag.assess_local_rag_query("山田太郎教授の経歴を教えてください", [hit])
        no_hit = local_rag.assess_local_rag_query("未来創造特別選抜制度について教えてください", [])

        self.assertTrue(accepted.accepted)
        self.assertFalse(unknown.accepted)
        self.assertTrue(unknown.reason.startswith("unknown_person:"))
        self.assertFalse(no_hit.accepted)
        self.assertEqual(no_hit.reason, "no_hit")

        access_hit = local_rag.LocalRagHit(
            path="access.md",
            title="Campus Access",
            section_title="access_hachioji",
            source_title="交通案内",
            source_url="https://example.invalid/access",
            publisher="東京工科大学",
            published_date="2026-01-01",
            accessed_date="2026-07-16",
            source_type="official_access",
            source_priority="P0",
            record_type="access_route",
            chunk_id="chunk_access_hachioji",
            keywords="access,八王子",
            chunk_index=0,
            text="八王子キャンパスへは八王子駅からスクールバスで約10分です。",
            score=220.0,
        )
        access = local_rag.assess_local_rag_query("アクセスを教えてください", [access_hit])
        self.assertTrue(access.accepted)

    def test_confidence_accepts_qualified_known_admission_name(self) -> None:
        local_rag = load_module("local_rag_admission_qualifier_test", ROOT / "infra" / "rag" / "local_rag.py")
        hit = local_rag.LocalRagHit(
            path="admissions.md",
            title="Admissions",
            section_title="総合型選抜",
            source_title="2027年度入試情報",
            source_url="https://example.invalid/admissions",
            publisher="東京工科大学",
            published_date="2026-07-01",
            accessed_date="2026-07-16",
            source_type="admissions",
            source_priority="P1",
            record_type="admission",
            chunk_id="chunk_admission_2027_ao_all",
            keywords="総合型選抜,AO入試",
            chunk_index=0,
            text="東京工科大学の2027年度総合型選抜を案内します。",
            score=200.0,
        )

        assessment = local_rag.assess_local_rag_query(
            "東京工科大学の総合型選抜について教えてください",
            [hit],
        )

        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.domain, "admissions")

    def test_conversation_query_resolves_supported_followup(self) -> None:
        local_rag = load_module("local_rag_followup_test", ROOT / "infra" / "rag" / "local_rag.py")

        resolved = local_rag.resolve_conversation_query(
            ["香川先生の研究内容を教えてください", "もう少し詳しく教えてください"]
        )
        weather = local_rag.resolve_conversation_query(
            ["香川先生の研究内容を教えてください", "今日の天気を教えてください"]
        )

        self.assertIn("香川先生の研究内容", resolved)
        self.assertIn("フォローアップ", resolved)
        self.assertEqual(local_rag.classify_query(resolved).domain, "kagawa_research")
        self.assertEqual(weather, "今日の天気を教えてください")

    def test_query_normalization_handles_asr_faculty_name_and_japanese_year(self) -> None:
        local_rag = load_module("local_rag_query_normalization_test", ROOT / "infra" / "rag" / "local_rag.py")

        normalized = local_rag.normalize_query(
            "２０２７年度のコンピューターセンス学部の入試情報"
        )
        resolved = local_rag.resolve_conversation_query(
            ["二千二十七年度の", "コンピューターセンス学部の入試情報について詳しく教えてください"]
        )

        self.assertEqual(normalized, "2027年度のコンピュータサイエンス学部の入試情報")
        self.assertTrue(resolved.startswith("2027年度のコンピュータサイエンス学部"))
        self.assertEqual(local_rag.classify_query(resolved).domain, "admissions")

    def test_query_normalization_routes_seiran_asr_aliases_and_fragmented_top500_followup(self) -> None:
        local_rag = load_module("local_rag_seiran_asr_test", ROOT / "infra" / "rag" / "local_rag.py")

        self.assertEqual(
            local_rag.normalize_query("セーラのトップ五百について教えてください"),
            "青嵐のTOP500について教えてください",
        )
        self.assertEqual(
            local_rag.normalize_query("西ラの利用申請料金について教えてください"),
            "青嵐の利用申請料金について教えてください",
        )
        for observed in (
            "セーランセランのトップ五百について",
            "セーラン セランのトップ五百について",
            "せいらんせいらんのトップ五百について",
            "青嵐、青嵐のトップ五百について",
        ):
            self.assertEqual(
                local_rag.normalize_query(observed),
                "青嵐のTOP500について",
            )
        resolved = local_rag.resolve_conversation_query(
            ["スパコンについて教えてください", "トップ五百では", "ええ", "何位でしたか"]
        )

        self.assertIn("TOP500では", resolved)
        self.assertIn("フォローアップ: 何位でしたか", resolved)
        self.assertEqual(local_rag.classify_query(resolved).domain, "seiran")

    def test_mixed_faculty_admission_query_keeps_admission_domain_and_diverse_pages(self) -> None:
        local_rag = load_module("local_rag_mixed_domain_test", ROOT / "infra" / "rag" / "local_rag.py")
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            sections = []
            for index in range(5):
                sections.append(
                    f"## faculty_{index}\n\nrecord_type: faculty_profile\n"
                    f"chunk_id: chunk_faculty_{index}\n\n"
                    "コンピュータサイエンス学部の専攻・研究室・学修内容を詳しく紹介します。\n"
                )
            for page, detail in (("6", "募集人員"), ("7", "指定2教科"), ("8", "学部特色入試")):
                sections.append(
                    f"## admission_{page}\n\nsource_url: https://example.invalid/guide.pdf\n"
                    f"page_number: {page}\neffective_year: 2027\ntemporal_status: current_2027\n"
                    f"record_type: admission\nchunk_id: chunk_admission_{page}\n\n"
                    f"2027年度のコンピュータサイエンス学部の{detail}です。\n"
                )
            (corpus_dir / "mixed.md").write_text("# Mixed\n\n" + "\n".join(sections), encoding="utf-8")
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path)

            query = "コンピューターセンス学部の入試情報について詳しく教えてください"
            hits = local_rag.search_index(db_path, query, top_k=3)
            assessment = local_rag.assess_local_rag_query(query, hits)

            self.assertTrue(assessment.accepted)
            self.assertEqual(assessment.domain, "admissions")
            self.assertEqual({hit.record_type for hit in hits}, {"admission"})
            self.assertEqual({hit.page_number for hit in hits}, {"6", "7", "8"})

    def test_search_does_not_return_priority_only_hits_for_unrelated_query(self) -> None:
        local_rag = load_module("local_rag_no_scan_fallback_test", ROOT / "infra" / "rag" / "local_rag.py")
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "official.md").write_text(
                "# Official\n\n## profile\n\nsource_priority: P0\nrecord_type: school_profile\n"
                "chunk_id: chunk_official\n\n東京工科大学の教育方針です。\n",
                encoding="utf-8",
            )
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path)

            self.assertEqual(local_rag.search_index(db_path, "今日の天気は", top_k=3), [])

    def test_atomic_build_preserves_previous_index_when_rebuild_fails(self) -> None:
        local_rag = load_module("local_rag_atomic_test", ROOT / "infra" / "rag" / "local_rag.py")
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            source_path = corpus_dir / "school.md"
            source_path.write_text("# School\n\n東京工科大学の概要です。\n", encoding="utf-8")
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path)
            manifest_path = local_rag.index_manifest_path(db_path)
            original_db = db_path.read_bytes()
            original_manifest = manifest_path.read_bytes()

            with mock.patch.object(local_rag, "read_document", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    local_rag.build_index(corpus_dir, db_path, chunk_chars=500)

            self.assertEqual(db_path.read_bytes(), original_db)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertTrue(local_rag.verify_index_freshness(corpus_dir, db_path)["fresh"])

    def test_manifest_detects_corpus_changes(self) -> None:
        local_rag = load_module("local_rag_manifest_test", ROOT / "infra" / "rag" / "local_rag.py")
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            source_path = corpus_dir / "school.md"
            source_path.write_text("# School\n\n東京工科大学の概要です。\n", encoding="utf-8")
            db_path = Path(tmp) / "local_rag.sqlite"
            local_rag.build_index(corpus_dir, db_path)

            self.assertTrue(local_rag.verify_index_freshness(corpus_dir, db_path)["fresh"])
            source_path.write_text("# School\n\n東京工科大学の概要を更新しました。\n", encoding="utf-8")
            with self.assertRaisesRegex(local_rag.StaleIndexError, "fingerprint"):
                local_rag.verify_index_freshness(corpus_dir, db_path)


if __name__ == "__main__":
    unittest.main()
