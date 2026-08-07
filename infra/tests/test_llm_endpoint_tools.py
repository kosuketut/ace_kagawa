from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_llm = load_module("check_llm_endpoint", ROOT / "infra" / "llm" / "check_llm_endpoint.py")
merge_lora = load_module("merge_osaka_swallow_lora", ROOT / "infra" / "llm" / "merge_osaka_swallow_lora.py")


class LlmEndpointToolTests(unittest.TestCase):
    def test_endpoint_check_defaults_to_nemotron_ultra(self) -> None:
        self.assertEqual(check_llm.DEFAULT_MODEL, EXPECTED_DEFAULT_MODEL)

    def test_arg_parser_reuses_tokkio_nvidia_api_key(self) -> None:
        with patch.dict(os.environ, {"TOKKIO_NVIDIA_API_KEY": "tokkio-key"}, clear=True):
            args = check_llm.build_arg_parser().parse_args([])

        self.assertEqual(args.api_key, "tokkio-key")

    def test_normalize_base_url_adds_v1_suffix_once(self) -> None:
        self.assertEqual(check_llm.normalize_openai_base_url("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(check_llm.normalize_openai_base_url("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1")

    def test_parse_chat_sse_deltas_ignores_done_and_non_delta_lines(self) -> None:
        lines = [
            b": keepalive\n",
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            'data: {"choices":[{"delta":{"content":"まいど"}}]}\n'.encode("utf-8"),
            'data: {"choices":[{"delta":{"content":"。"}}]}\n'.encode("utf-8"),
            b"data: [DONE]\n",
        ]

        self.assertEqual(list(check_llm.parse_chat_sse_deltas(lines)), ["まいど", "。"])

    def test_chat_payload_can_disable_nemotron_reasoning(self) -> None:
        payload = check_llm.build_chat_payload(
            model=EXPECTED_DEFAULT_MODEL,
            prompt="確認",
            max_tokens=64,
            disable_thinking=True,
        )

        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_summarize_token_timings_reports_ttft_and_itl(self) -> None:
        timings = check_llm.summarize_token_timings(start_time=10.0, token_times=[10.25, 10.35, 10.50])

        self.assertEqual(timings["token_count"], 3)
        self.assertAlmostEqual(timings["ttft_ms"], 250.0)
        self.assertAlmostEqual(timings["avg_itl_ms"], 125.0)

    def test_disable_thinking_by_default_makes_qwen_generation_prompt_unconditional(self) -> None:
        original = r"""{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

        updated = merge_lora.disable_thinking_by_default(original)

        self.assertIn("<think>", updated)
        self.assertNotIn("enable_thinking is defined", updated)


if __name__ == "__main__":
    unittest.main()
