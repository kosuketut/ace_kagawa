from __future__ import annotations

import unittest

from app.text import SentenceChunker


class SentenceChunkerTests(unittest.TestCase):
    def test_chunker_emits_complete_sentences(self) -> None:
        chunker = SentenceChunker()
        first = chunker.push("今日は")
        second = chunker.push("晴れです。明日")
        third = chunker.push("も晴れるでしょう。")
        self.assertEqual(first, [])
        self.assertEqual(second, ["今日は晴れです。"])
        self.assertEqual(third, ["明日も晴れるでしょう。"])


if __name__ == "__main__":
    unittest.main()

