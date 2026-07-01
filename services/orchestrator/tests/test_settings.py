from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.settings import Settings


EXPECTED_ASR_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
EXPECTED_NIM_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EXPECTED_NIM_LLM_MODEL = "stockmark/stockmark-2-100b-instruct"


class SettingsTests(unittest.TestCase):
    def test_default_asr_model_uses_nemotron_streaming(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.asr_model, EXPECTED_ASR_MODEL)

    def test_default_llm_uses_stockmark_nim(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.nim_llm_base_url, EXPECTED_NIM_LLM_BASE_URL)
        self.assertEqual(settings.nim_llm_model, EXPECTED_NIM_LLM_MODEL)

    def test_default_system_prompt_uses_standard_japanese(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertIn("標準語", settings.system_prompt)
        self.assertIn("香川", settings.system_prompt)
        self.assertIn("40から120文字", settings.system_prompt)
        self.assertIn("対話型バーチャルアシスタント", settings.system_prompt)
        self.assertNotIn("大阪弁", settings.system_prompt)
        self.assertNotIn("大藪", settings.system_prompt)
        self.assertNotIn("/no_think", settings.system_prompt)


if __name__ == "__main__":
    unittest.main()
