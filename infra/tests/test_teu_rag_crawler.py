from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_crawler():
    path = ROOT / "infra" / "rag" / "crawl_teu_rag.py"
    spec = importlib.util.spec_from_file_location("crawl_teu_rag_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TeuRagCrawlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.crawler = load_crawler()

    def test_extracts_main_content_and_ignores_navigation(self) -> None:
        html = """
        <html><head><title>学部案内 | 東京工科大学</title></head><body>
        <header><a href="/gakubu/noise.html">不要なヘッダー</a></header>
        <div id="bodyarea" role="main">
          <h1>コンピュータサイエンス学部</h1>
          <p>先進情報専攻と社会情報専攻について学びます。</p>
          <a href="/gakubu/cs/course.html">専攻紹介</a>
          <aside class="nav-category"><a href="/gakubu/noise2.html">不要なメニュー</a></aside>
        </div>
        <footer>不要なフッター</footer>
        </body></html>
        """
        title, body, links = self.crawler.extract_html(html)

        self.assertEqual(title, "学部案内 | 東京工科大学")
        self.assertIn("コンピュータサイエンス学部", body)
        self.assertIn("先進情報専攻", body)
        self.assertNotIn("不要", body)
        self.assertEqual(links, [("/gakubu/cs/course.html", "専攻紹介")])

    def test_url_scope_keeps_only_official_topic_pages(self) -> None:
        canonical = self.crawler.canonicalize_url(
            "/entrance/info/AO/index.html?utm_source=test#detail",
            "https://www.teu.ac.jp/entrance/index.html",
        )
        self.assertEqual(canonical, "https://www.teu.ac.jp/entrance/info/AO/index.html")
        self.assertTrue(self.crawler.is_crawlable_url(canonical, "admissions"))
        self.assertTrue(
            self.crawler.is_crawlable_url(
                "https://www.teu.ac.jp/gakubu/cs/index.html",
                "faculties",
            )
        )
        self.assertFalse(self.crawler.is_crawlable_url("https://example.com/gakubu/index.html", "faculties"))
        self.assertFalse(self.crawler.is_crawlable_url("https://www.teu.ac.jp/campus/index.html", "faculties"))
        self.assertFalse(self.crawler.is_crawlable_url("https://www.teu.ac.jp/gakubu/image.jpg", "faculties"))

    def test_rendered_markdown_has_local_rag_metadata(self) -> None:
        page = self.crawler.PageRecord(
            category="admissions",
            url="https://www.teu.ac.jp/entrance/info/AO/index.html",
            depth=1,
            title="総合型選抜（全学部AO入試） | 東京工科大学",
            body="総合型選抜の案内です。" * 10,
            content_type="text/html",
            sha256="abc",
            link_count=2,
        )
        rendered = self.crawler.render_markdown(
            "admissions",
            [page],
            generated_at="2026-07-17T12:00:00+09:00",
        )

        self.assertIn("source_url: https://www.teu.ac.jp/entrance/info/AO/index.html", rendered)
        self.assertIn("publisher: 東京工科大学", rendered)
        self.assertIn("source_type: admissions", rendered)
        self.assertIn("source_priority: P1", rendered)
        self.assertIn("record_type: admission", rendered)
        self.assertIn("chunk_id: chunk_teu_admission_", rendered)
        self.assertIn("総合型選抜の案内です。", rendered)


if __name__ == "__main__":
    unittest.main()
