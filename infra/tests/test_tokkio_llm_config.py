from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ASR_MODEL = "conformer-unified-ja-JP-asr-streaming-asr-bls-ensemble"
EXPECTED_ASR_RMIR = "nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0"
EXPECTED_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EXPECTED_LLM_MODEL = "stockmark/stockmark-2-100b-instruct"
EXPECTED_RAG_SERVER_URL = "http://10.209.1.12:8081/v1"
EXPECTED_RAG_COLLECTION = "ace_kagawa"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_tokkio_workspace", ROOT / "infra" / "tokkio" / "prepare_tokkio_workspace.py")
customize = load_module("customize_tokkio_japanese", ROOT / "infra" / "tokkio" / "customize_tokkio_japanese.py")


class TokkioLlmConfigTests(unittest.TestCase):
    def test_resolve_llm_settings_defaults_to_stockmark_nim_endpoint(self) -> None:
        settings = prepare.resolve_llm_settings(
            {
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_NVIDIA_API_KEY": "nvidia-key",
            }
        )

        self.assertEqual(settings.base_url, EXPECTED_LLM_BASE_URL)
        self.assertEqual(settings.model, EXPECTED_LLM_MODEL)
        self.assertEqual(settings.api_key, "nvidia-key")

    def test_generated_env_exports_llm_api_key_for_local_endpoint(self) -> None:
        generated = prepare.build_generated_env(
            {
                "TOKKIO_NVIDIA_API_KEY": "real-nvidia-key",
                "TOKKIO_NGC_CLI_API_KEY": "real-ngc-key",
                "TOKKIO_OPENAI_API_KEY": "",
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_APP_HOST_SSH_USER": "kyano",
                "TOKKIO_LLM_API_KEY": "local-key",
            }
        )

        self.assertIn('export NVIDIA_LLM_API_KEY="local-key"', generated)

    def test_generated_env_reuses_nvidia_api_key_for_hosted_nim(self) -> None:
        generated = prepare.build_generated_env(
            {
                "TOKKIO_NVIDIA_API_KEY": "real-nvidia-key",
                "TOKKIO_NGC_CLI_API_KEY": "real-ngc-key",
                "TOKKIO_OPENAI_API_KEY": "",
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_APP_HOST_SSH_USER": "kyano",
            }
        )

        self.assertIn('export NVIDIA_LLM_API_KEY="real-nvidia-key"', generated)

    def test_resolve_rag_settings_defaults_to_disabled_host_side_service(self) -> None:
        settings = prepare.resolve_rag_settings({"TOKKIO_APP_HOST_IPV4_ADDR": "10.209.1.12"})

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.server_url, EXPECTED_RAG_SERVER_URL)
        self.assertEqual(settings.collection_name, EXPECTED_RAG_COLLECTION)
        self.assertTrue(settings.use_knowledge_base)
        self.assertEqual(settings.max_tokens, 1000)

    def test_resolve_rag_settings_normalizes_url_and_bool_values(self) -> None:
        settings = prepare.resolve_rag_settings(
            {
                "TOKKIO_RAG_ENABLED": "true",
                "TOKKIO_RAG_SERVER_URL": "http://192.0.2.10:8081",
                "TOKKIO_RAG_COLLECTION_NAME": "manuals",
                "TOKKIO_RAG_USE_KNOWLEDGE_BASE": "false",
                "TOKKIO_RAG_MAX_TOKENS": "512",
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.server_url, "http://192.0.2.10:8081/v1")
        self.assertEqual(settings.collection_name, "manuals")
        self.assertFalse(settings.use_knowledge_base)
        self.assertEqual(settings.max_tokens, 512)

    def test_custom_config_yaml_uses_stockmark_nim_endpoint_and_model(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn(f'base_url: "{EXPECTED_LLM_BASE_URL}"', yaml_text)
        self.assertIn(f'model: "{EXPECTED_LLM_MODEL}"', yaml_text)
        self.assertIn("標準語で自然かつ簡潔", yaml_text)
        self.assertIn("40から120文字", yaml_text)
        self.assertIn('llm_processor: "NvidiaLLMService"', yaml_text)

    def test_custom_config_yaml_can_enable_rag_processor(self) -> None:
        yaml_text = customize.build_config_yaml(
            rag_enabled=True,
            rag_server_url="http://10.209.1.12:8081",
            rag_collection_name="ace_kagawa",
            rag_max_tokens=512,
        )

        self.assertIn('llm_processor: "NvidiaRAGService"', yaml_text)
        self.assertIn(f'rag_server_url: "{EXPECTED_RAG_SERVER_URL}"', yaml_text)
        self.assertIn(f'collection_name: "{EXPECTED_RAG_COLLECTION}"', yaml_text)
        self.assertIn("max_tokens: 512", yaml_text)

    def test_custom_config_yaml_uses_standard_japanese_prompt(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn('name: "香川"', yaml_text)
        self.assertIn('tts_processor: "IrodoriTTSService"', yaml_text)
        self.assertIn("IrodoriTTSService:", yaml_text)
        self.assertIn('base_url: "http://10.209.1.12:8021"', yaml_text)
        self.assertIn('voice: "kagawa"', yaml_text)
        self.assertIn("あなたは「{name}」という名前", yaml_text)
        self.assertIn("自然な標準語", yaml_text)
        self.assertNotIn("大阪弁", yaml_text)
        self.assertNotIn("大阪・住之江", yaml_text)
        self.assertNotIn("大藪", yaml_text)
        self.assertNotIn("/no_think", yaml_text)

    def test_custom_config_yaml_uses_riva_asr_model(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn(f'model: "{EXPECTED_ASR_MODEL}"', yaml_text)

    def test_patch_riva_values_uses_japanese_asr_rmir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-no-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(
                "riva-api:\n"
                "  modelRepoGenerator:\n"
                "    ngcModelConfigs:\n"
                "      triton0:\n"
                "        models:\n"
                "          - nvidia/riva/rmir_asr_parakeet_1-1b_en_us_str_silero:2.19.0.1\n",
                encoding="utf-8",
            )

            customize.patch_riva_values(ace_repo_dir)

            self.assertIn(EXPECTED_ASR_RMIR, values_path.read_text(encoding="utf-8"))

    def test_patch_riva_values_migrates_previous_japanese_asr_rmir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-no-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(
                "riva-api:\n"
                "  modelRepoGenerator:\n"
                "    ngcModelConfigs:\n"
                "      triton0:\n"
                "        models:\n"
                "          - nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0\n",
                encoding="utf-8",
            )

            customize.patch_riva_values(ace_repo_dir)

            self.assertIn(EXPECTED_ASR_RMIR, values_path.read_text(encoding="utf-8"))

    def test_bot_snippet_prefers_dedicated_llm_api_key(self) -> None:
        self.assertIn('os.getenv("NVIDIA_LLM_API_KEY")', customize.NEW_LLM_SNIPPET)
        self.assertIn('or os.getenv("NVIDIA_API_KEY")', customize.NEW_LLM_SNIPPET)
        self.assertNotIn("tensorrt_llm", customize.NEW_LLM_SNIPPET)

    def test_patch_llm_snippet_updates_previous_tensorrt_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.PREVIOUS_LLM_SNIPPET, encoding="utf-8")

            customize.patch_llm_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_LLM_SNIPPET, updated)
            self.assertNotIn("tensorrt_llm", updated)

    def test_config_py_supports_irodori_tts_processor(self) -> None:
        self.assertIn('"IrodoriTTSService"', customize.CONFIG_PY)
        self.assertIn("class IrodoriTTSServiceConfig", customize.CONFIG_PY)
        self.assertIn("IrodoriTTSService: IrodoriTTSServiceConfig", customize.CONFIG_PY)

    def test_bot_snippet_builds_irodori_tts_service(self) -> None:
        self.assertIn("IrodoriTTSService", customize.NEW_BOT_SNIPPET)
        self.assertIn("config.IrodoriTTSService.base_url", customize.NEW_BOT_SNIPPET)
        self.assertIn("response_format=config.IrodoriTTSService.response_format", customize.NEW_BOT_SNIPPET)

    def test_apply_patch_updates_profile_ace_controller_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            llm_rag_dir = ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "src" / "llm-rag"
            (llm_rag_dir / "src").mkdir(parents=True)
            (llm_rag_dir / "configs").mkdir(parents=True)
            (llm_rag_dir / "src" / "config.py").write_text("# old config\n", encoding="utf-8")
            (llm_rag_dir / "configs" / "config.yaml").write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            (llm_rag_dir / "src" / "bot.py").write_text(
                customize.OLD_LLM_SNIPPET + "\n" + customize.OLD_BOT_SNIPPET,
                encoding="utf-8",
            )
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-with-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(f"- {customize.ENGLISH_ASR_RMIR}\n", encoding="utf-8")
            profile_config_path = values_path.parent / "config" / "ace-controller" / "config.yaml"
            profile_config_path.parent.mkdir(parents=True)
            profile_config_path.write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")

            customize.apply_patch(
                ace_repo_dir,
                llm_base_url=EXPECTED_LLM_BASE_URL,
                llm_model=EXPECTED_LLM_MODEL,
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            self.assertIn('name: "香川"', profile_config)
            self.assertIn(f'base_url: "{EXPECTED_LLM_BASE_URL}"', profile_config)
            self.assertIn(f'model: "{EXPECTED_LLM_MODEL}"', profile_config)
            self.assertIn('tts_processor: "IrodoriTTSService"', profile_config)
            self.assertIn('base_url: "http://10.209.1.12:8021"', profile_config)
            self.assertTrue((llm_rag_dir / "src" / "tokkio_irodori_tts.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "tokkio_rag.py").is_file())
            self.assertIn(
                '"collection_names": [self.collection_name]',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )

    def test_apply_patch_can_enable_rag_in_profile_ace_controller_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ace_repo_dir = Path(tmp)
            llm_rag_dir = ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "src" / "llm-rag"
            (llm_rag_dir / "src").mkdir(parents=True)
            (llm_rag_dir / "configs").mkdir(parents=True)
            (llm_rag_dir / "src" / "config.py").write_text("# old config\n", encoding="utf-8")
            (llm_rag_dir / "configs" / "config.yaml").write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")
            (llm_rag_dir / "src" / "bot.py").write_text(
                customize.OLD_LLM_SNIPPET + "\n" + customize.OLD_BOT_SNIPPET,
                encoding="utf-8",
            )
            values_path = (
                ace_repo_dir
                / "workflows"
                / "tokkio"
                / "5.0.0-ga"
                / "llm-rag"
                / "tokkio-1stream-with-ui"
                / "values.yaml"
            )
            values_path.parent.mkdir(parents=True)
            values_path.write_text(f"- {customize.ENGLISH_ASR_RMIR}\n", encoding="utf-8")
            profile_config_path = values_path.parent / "config" / "ace-controller" / "config.yaml"
            profile_config_path.parent.mkdir(parents=True)
            profile_config_path.write_text("OpenAILLMContext:\n  name: Aki\n", encoding="utf-8")

            customize.apply_patch(
                ace_repo_dir,
                rag_enabled=True,
                rag_server_url="http://10.209.1.12:8081",
                rag_collection_name="ace_kagawa",
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            self.assertIn('llm_processor: "NvidiaRAGService"', profile_config)
            self.assertIn(f'rag_server_url: "{EXPECTED_RAG_SERVER_URL}"', profile_config)
            self.assertIn(f'collection_name: "{EXPECTED_RAG_COLLECTION}"', profile_config)

    def test_build_config_yaml_accepts_custom_irodori_tts_url(self) -> None:
        yaml_text = customize.build_config_yaml(irodori_tts_base_url="http://192.0.2.10:8021")

        self.assertIn('base_url: "http://192.0.2.10:8021"', yaml_text)


if __name__ == "__main__":
    unittest.main()
