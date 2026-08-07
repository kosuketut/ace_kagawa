from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ASR_MODEL = "conformer-unified-ja-JP-asr-streaming-asr-bls-ensemble"
EXPECTED_ASR_RMIR = "nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0"
EXPECTED_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EXPECTED_LLM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
EXPECTED_RAG_SERVER_URL = "http://10.209.1.12:8081/v1"
EXPECTED_RAG_COLLECTION = "ace_kagawa"
EXPECTED_RAG_MAX_TOKENS = 64
EXPECTED_RAG_VDB_TOP_K = 12
EXPECTED_RAG_RERANKER_TOP_K = 5
EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K = 10
EXPECTED_RAG_MODE = "auto"
EXPECTED_RAG_ROUTE_KEYWORDS = ["論文", "資料", "EBC", "SiC/SiC"]
EXPECTED_KAGAWA_PROFILE_ROUTE_KEYWORDS = [
    "専門分野",
    "学歴",
    "職歴",
    "役職",
    "現職",
    "学位",
    "所属",
    "生年月日",
    "年齢",
    "誰ですか",
]
EXPECTED_EXPANDED_ROUTE_KEYWORDS = [
    "専攻",
    "学費",
    "入学金",
    "授業料",
    "奨学金",
    "学生支援",
    "研究",
    "スパコン",
    "スーパーコンピュータ",
    "青嵐",
    "SEIRAN",
    "DGX B200",
]
EXPECTED_RAG_PROVIDER = "local"
EXPECTED_LOCAL_RAG_CORPUS = "data/rag/corpus"
EXPECTED_LOCAL_RAG_DB = "data/rag/local/local_rag.sqlite"
EXPECTED_LOCAL_RAG_RUNTIME_DB = "/code/configs/local_rag.sqlite"
EXPECTED_LOCAL_RAG_TOP_K = 3
EXPECTED_LOCAL_RAG_MAX_CONTEXT_CHARS = 2800


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


def install_stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    if "." not in name:
        module.__path__ = []
    sys.modules[name] = module
    return module


def load_tokkio_llm_with_stubs():
    class Logger:
        def debug(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    class Frame:
        def __init__(self, content=""):
            self.content = content

    class BaseLLMService:
        pass

    loguru = install_stub_module("loguru")
    loguru.logger = Logger()

    install_stub_module("nvidia_pipecat")
    install_stub_module("nvidia_pipecat.services")
    nvidia_llm = install_stub_module("nvidia_pipecat.services.nvidia_llm")
    nvidia_llm.NvidiaLLMService = BaseLLMService

    install_stub_module("pipecat")
    install_stub_module("pipecat.frames")
    frames = install_stub_module("pipecat.frames.frames")
    frames.TextFrame = Frame
    frames.TTSSpeakFrame = Frame
    install_stub_module("pipecat.processors")
    install_stub_module("pipecat.processors.aggregators")
    openai_context = install_stub_module("pipecat.processors.aggregators.openai_llm_context")
    openai_context.OpenAILLMContext = object
    install_stub_module("pipecat.services")
    install_stub_module("pipecat.services.openai")
    openai_llm = install_stub_module("pipecat.services.openai.llm")
    openai_llm.OpenAILLMService = BaseLLMService

    return load_module("tokkio_llm_fast_reply_test", ROOT / "infra" / "tokkio" / "tokkio_llm.py")


class TokkioLlmConfigTests(unittest.TestCase):
    def test_resolve_llm_settings_defaults_to_nemotron_ultra_nim_endpoint(self) -> None:
        settings = prepare.resolve_llm_settings(
            {
                "TOKKIO_APP_HOST_IPV4_ADDR": "10.0.0.42",
                "TOKKIO_NVIDIA_API_KEY": "nvidia-key",
            }
        )

        self.assertEqual(settings.base_url, EXPECTED_LLM_BASE_URL)
        self.assertEqual(settings.model, EXPECTED_LLM_MODEL)
        self.assertEqual(settings.api_key, "nvidia-key")

    def test_nemotron_ultra_uses_realtime_non_reasoning_parameters(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        params = tokkio_llm.nvidia_model_input_params(EXPECTED_LLM_MODEL)

        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["max_tokens"], 512)
        self.assertEqual(
            params["extra"],
            {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            },
        )
        self.assertEqual(tokkio_llm.nvidia_model_input_params("mistralai/mistral-nemotron"), {})

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
        self.assertEqual(settings.max_tokens, EXPECTED_RAG_MAX_TOKENS)
        self.assertEqual(settings.vdb_top_k, EXPECTED_RAG_VDB_TOP_K)
        self.assertEqual(settings.reranker_top_k, EXPECTED_RAG_RERANKER_TOP_K)
        self.assertEqual(settings.multimodal_reranker_top_k, EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K)
        self.assertTrue(settings.enable_reranker)
        self.assertEqual(settings.mode, EXPECTED_RAG_MODE)
        self.assertIn("論文", settings.route_keywords)
        self.assertIn("SiC/SiC", settings.route_keywords)
        for keyword in ("研究", "プロジェクト", "専攻", "学費", "奨学金", "学生支援"):
            self.assertIn(keyword, settings.route_keywords)
        for keyword in EXPECTED_KAGAWA_PROFILE_ROUTE_KEYWORDS:
            self.assertIn(keyword, settings.route_keywords)
        self.assertFalse(settings.fallback_to_llm_on_error)
        self.assertEqual(settings.provider, EXPECTED_RAG_PROVIDER)
        self.assertEqual(settings.local_corpus_path, EXPECTED_LOCAL_RAG_CORPUS)
        self.assertEqual(settings.local_db_path, EXPECTED_LOCAL_RAG_DB)
        self.assertEqual(settings.local_runtime_db_path, EXPECTED_LOCAL_RAG_RUNTIME_DB)
        self.assertEqual(settings.local_top_k, EXPECTED_LOCAL_RAG_TOP_K)
        self.assertEqual(settings.local_max_context_chars, EXPECTED_LOCAL_RAG_MAX_CONTEXT_CHARS)

    def test_resolve_rag_settings_normalizes_url_and_bool_values(self) -> None:
        settings = prepare.resolve_rag_settings(
            {
                "TOKKIO_RAG_ENABLED": "true",
                "TOKKIO_RAG_PROVIDER": "nvidia",
                "TOKKIO_RAG_SERVER_URL": "http://192.0.2.10:8081",
                "TOKKIO_RAG_COLLECTION_NAME": "manuals",
                "TOKKIO_RAG_USE_KNOWLEDGE_BASE": "false",
                "TOKKIO_RAG_MAX_TOKENS": "512",
                "TOKKIO_RAG_VDB_TOP_K": "16",
                "TOKKIO_RAG_RERANKER_TOP_K": "6",
                "TOKKIO_RAG_MULTIMODAL_RERANKER_TOP_K": "10",
                "TOKKIO_RAG_ENABLE_RERANKER": "false",
                "TOKKIO_RAG_MODE": "always",
                "TOKKIO_RAG_ROUTE_KEYWORDS": "論文, 資料, EBC, SiC/SiC",
                "TOKKIO_RAG_FALLBACK_TO_LLM_ON_ERROR": "false",
                "TOKKIO_LOCAL_RAG_DB": "/tmp/custom-local.sqlite",
                "TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH": "/code/configs/custom-local.sqlite",
                "TOKKIO_LOCAL_RAG_TOP_K": "4",
                "TOKKIO_LOCAL_RAG_MAX_CONTEXT_CHARS": "1200",
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.mode, "always")
        self.assertEqual(settings.provider, "nvidia")
        self.assertEqual(settings.server_url, "http://192.0.2.10:8081/v1")
        self.assertEqual(settings.collection_name, "manuals")
        self.assertFalse(settings.use_knowledge_base)
        self.assertEqual(settings.max_tokens, 512)
        self.assertEqual(settings.vdb_top_k, 16)
        self.assertEqual(settings.reranker_top_k, 6)
        self.assertEqual(settings.multimodal_reranker_top_k, 10)
        self.assertFalse(settings.enable_reranker)
        self.assertEqual(settings.route_keywords, EXPECTED_RAG_ROUTE_KEYWORDS)
        self.assertFalse(settings.fallback_to_llm_on_error)
        self.assertEqual(settings.local_db_path, "/tmp/custom-local.sqlite")
        self.assertEqual(settings.local_runtime_db_path, "/code/configs/custom-local.sqlite")
        self.assertEqual(settings.local_top_k, 4)
        self.assertEqual(settings.local_max_context_chars, 1200)

    def test_resolve_rag_settings_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "TOKKIO_RAG_MODE"):
            prepare.resolve_rag_settings({"TOKKIO_RAG_ENABLED": "true", "TOKKIO_RAG_MODE": "sometimes"})

    def test_resolve_rag_settings_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "TOKKIO_RAG_PROVIDER"):
            prepare.resolve_rag_settings({"TOKKIO_RAG_ENABLED": "true", "TOKKIO_RAG_PROVIDER": "remote"})

    def test_prepare_freshness_gate_rejects_changed_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            corpus_dir.mkdir()
            source_path = corpus_dir / "school.md"
            source_path.write_text("# School\n\n東京工科大学の概要です。\n", encoding="utf-8")
            db_path = Path(tmp) / "local_rag.sqlite"
            subprocess.run(
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
            settings = prepare.resolve_rag_settings(
                {
                    "TOKKIO_RAG_ENABLED": "true",
                    "TOKKIO_RAG_PROVIDER": "local",
                    "TOKKIO_LOCAL_RAG_CORPUS": str(corpus_dir),
                    "TOKKIO_LOCAL_RAG_DB": str(db_path),
                }
            )

            self.assertTrue(prepare.verify_local_rag_index(settings)["fresh"])
            source_path.write_text("# School\n\n東京工科大学の概要を更新しました。\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.verify_local_rag_index(settings)

    def test_custom_config_yaml_uses_nemotron_ultra_nim_endpoint_and_model(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn(f'base_url: "{EXPECTED_LLM_BASE_URL}"', yaml_text)
        self.assertIn(f'model: "{EXPECTED_LLM_MODEL}"', yaml_text)
        self.assertIn("標準語で自然かつ簡潔", yaml_text)
        self.assertIn("40から60文字", yaml_text)
        self.assertIn("原則1文", yaml_text)
        self.assertIn("100文字以内", yaml_text)
        self.assertIn("最大2文", yaml_text)
        self.assertNotIn("80文字以内", yaml_text)
        self.assertIn("time_delay: 2.5", yaml_text)
        self.assertIn('- "確認しています"', yaml_text)
        self.assertNotIn("少々お待ちください", yaml_text)
        self.assertIn('llm_processor: "NvidiaLLMService"', yaml_text)

    def test_custom_config_yaml_can_enable_rag_processor(self) -> None:
        yaml_text = customize.build_config_yaml(
            rag_enabled=True,
            rag_mode="always",
            rag_server_url="http://10.209.1.12:8081",
            rag_collection_name="ace_kagawa",
            rag_max_tokens=512,
        )

        self.assertIn('llm_processor: "NvidiaRAGService"', yaml_text)
        self.assertIn(f'rag_server_url: "{EXPECTED_RAG_SERVER_URL}"', yaml_text)
        self.assertIn(f'collection_name: "{EXPECTED_RAG_COLLECTION}"', yaml_text)
        self.assertIn("max_tokens: 512", yaml_text)
        self.assertIn(f"vdb_top_k: {EXPECTED_RAG_VDB_TOP_K}", yaml_text)
        self.assertIn(f"reranker_top_k: {EXPECTED_RAG_RERANKER_TOP_K}", yaml_text)
        self.assertIn(f"multimodal_reranker_top_k: {EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K}", yaml_text)
        self.assertIn("enable_reranker: true", yaml_text)

    def test_custom_config_yaml_can_enable_auto_rag_router(self) -> None:
        yaml_text = customize.build_config_yaml(
            rag_enabled=True,
            rag_mode="auto",
            rag_route_keywords=EXPECTED_RAG_ROUTE_KEYWORDS,
        )

        self.assertIn('llm_processor: "NvidiaLLMRAGRouterService"', yaml_text)
        self.assertIn("NvidiaRAGRouterService:", yaml_text)
        self.assertIn('provider: "local"', yaml_text)
        self.assertIn(f'local_db_path: "{EXPECTED_LOCAL_RAG_RUNTIME_DB}"', yaml_text)
        self.assertIn(f"local_top_k: {EXPECTED_LOCAL_RAG_TOP_K}", yaml_text)
        self.assertIn(f"local_max_context_chars: {EXPECTED_LOCAL_RAG_MAX_CONTEXT_CHARS}", yaml_text)
        route_section = yaml_text.split("    route_keywords:\n", 1)[1].split(
            "    fallback_to_llm_on_error:", 1
        )[0]
        for keyword in EXPECTED_RAG_ROUTE_KEYWORDS:
            self.assertIn(f'        - "{keyword}"', route_section)
        self.assertNotIn('        - "香川先生"', route_section)
        self.assertNotIn(customize.RAG_ROUTE_KEYWORDS_TEMPLATE_MARKER, yaml_text)
        self.assertIn("fallback_to_llm_on_error: false", yaml_text)

    def test_grounded_rag_failures_never_use_direct_llm_fallback(self) -> None:
        source = (ROOT / "infra" / "tokkio" / "tokkio_rag.py").read_text(encoding="utf-8")
        local_rag_block = source.split(
            "    async def _stream_local_rag_response", 1
        )[1].split("    async def _stream_rag_response", 1)[0]
        router_error_block = source.split(
            '            logger.error(f"RAG router failed, Error: {exc!r}")', 1
        )[1]

        self.assertIn("GROUNDED_ANSWER_UNAVAILABLE_REPLY", source)
        self.assertEqual(local_rag_block.count("await self._push_grounded_answer_unavailable()"), 2)
        self.assertNotIn("await super()._process_context(context)", local_rag_block)
        self.assertNotIn("await super()._process_context(context)", router_error_block)
        self.assertIn(
            "await self._push_grounded_answer_unavailable(start_metrics=False)",
            source,
        )
        self.assertIn(
            "await self.push_frame(TTSSpeakFrame(GROUNDED_ANSWER_UNAVAILABLE_REPLY))",
            router_error_block,
        )

    def test_custom_config_yaml_uses_early_cached_filler_delay(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn('        - "確認しています"', yaml_text)
        self.assertIn("    time_delay: 2.5", yaml_text)

    def test_custom_config_yaml_routes_kagawa_profile_detail_questions_by_default(self) -> None:
        yaml_text = customize.build_config_yaml(rag_enabled=True, rag_mode="auto")

        for keyword in EXPECTED_KAGAWA_PROFILE_ROUTE_KEYWORDS:
            self.assertIn(f'        - "{keyword}"', yaml_text)
        for keyword in EXPECTED_EXPANDED_ROUTE_KEYWORDS:
            self.assertIn(f'        - "{keyword}"', yaml_text)

    def test_custom_config_yaml_can_force_rag_off_even_when_enabled(self) -> None:
        yaml_text = customize.build_config_yaml(rag_enabled=True, rag_mode="off")

        self.assertIn('llm_processor: "NvidiaLLMService"', yaml_text)

    def test_tokkio_llm_fast_replies_cover_greetings_name_and_current_age(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        self.assertEqual(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "おはようございます。"}]),
            "おはようございます。",
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "あなたの名前は？"}]),
            "私は香川豊です。",
        )
        self.assertIsNone(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "香川先生の論文の名前は？"}])
        )
        self.assertIsNone(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "香川先生は誰ですか"}])
        )
        self.assertIsNone(
            tokkio_llm.get_fast_profile_reply([{"role": "user", "content": "香川先生の名前と専門分野は？"}])
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply(
                [{"role": "user", "content": "香川先生の年齢は？"}],
                today=__import__("datetime").date(2026, 7, 19),
            ),
            "私は1952年9月19日生まれで、現在73歳です。",
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply(
                [{"role": "user", "content": "香川先生は何歳ですか？"}],
                today=__import__("datetime").date(2026, 9, 19),
            ),
            "私は1952年9月19日生まれで、現在74歳です。",
        )
        self.assertEqual(
            tokkio_llm.get_fast_profile_reply(
                [{"role": "user", "content": "先生はおいくつですか？"}],
                today=__import__("datetime").date(2026, 7, 19),
            ),
            "私は1952年9月19日生まれで、現在73歳です。",
        )

    def test_tokkio_llm_open_campus_greeting_uses_explicit_cues_and_full_script(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        script = tokkio_llm.get_scripted_speech_reply(
            [{"role": "user", "content": "オープンキャンパスのご挨拶をお願いします。"}]
        )

        self.assertEqual(script, tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT)
        self.assertEqual(script[0], "皆さん、こんにちは。")
        self.assertIn("USJとほぼ同じ広さ", "".join(script))
        self.assertIn("私はAI学長です。", "".join(script))
        self.assertEqual(
            script[-1],
            "ここで、開発した学生と私とのリアルなやり取りをご覧ください。",
        )
        self.assertIsNone(
            tokkio_llm.get_scripted_speech_reply(
                [{"role": "user", "content": "オープンキャンパスはいつですか？"}]
            )
        )

    def test_tokkio_llm_open_campus_greeting_pushes_sentence_segments_without_llm(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_script():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                filler = ["確認しています"]
                time_delay = 60.0
                first_response_timeout_s = 0.01

                def __init__(self) -> None:
                    self.frames = []
                    self.metrics_started = 0
                    self.metrics_stopped = 0

                async def start_ttfb_metrics(self) -> None:
                    self.metrics_started += 1

                async def stop_ttfb_metrics(self) -> None:
                    self.metrics_stopped += 1

                async def push_frame(self, frame) -> None:
                    self.frames.append(frame)

            async def unexpected_stream(_context):
                raise AssertionError("scripted speech must not call the LLM")

            context = types.SimpleNamespace(
                get_messages=lambda: [{"role": "user", "content": "台本を読んでください"}]
            )
            harness = Harness()
            await harness._process_context_common(context, unexpected_stream)
            return harness

        harness = __import__("asyncio").run(exercise_script())

        self.assertEqual(
            [frame.content for frame in harness.frames],
            list(tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT),
        )
        self.assertEqual(harness.metrics_started, 1)
        self.assertEqual(harness.metrics_stopped, 1)

    def test_tokkio_llm_open_campus_closing_uses_separate_fixed_script(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        script = tokkio_llm.get_scripted_speech_reply(
            [{"role": "user", "content": "オープンキャンパスの締めの言葉をお願いします。"}]
        )

        self.assertEqual(script, tokkio_llm.OPEN_CAMPUS_CLOSING_SCRIPT)
        self.assertEqual(
            script,
            (
                "ね。このようにAI香川豊、しっかり対応できるんですよ。",
                "今後も様々なシーンでお目にかかるかと思います。",
                "4月にみなさんとお会いできることを楽しみにしています。",
                "今日は一日、楽しんでいってください。",
            ),
        )
        self.assertNotIn(
            "ここで、開発した学生と私とのリアルなやり取りをご覧ください。",
            script,
        )

    def test_tokkio_llm_uses_bounded_first_response_timeout(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        self.assertEqual(tokkio_llm._DEFAULT_FIRST_RESPONSE_TIMEOUT_S, 20.0)
        self.assertIn("もう一度お尋ねください", tokkio_llm._LLM_TIMEOUT_REPLY)

    def test_tokkio_llm_speaks_japanese_fallback_when_first_text_times_out(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_timeout():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                filler = ["確認しています"]
                time_delay = 60.0
                first_response_timeout_s = 0.01

                def __init__(self) -> None:
                    self.frames = []
                    self.metrics_stopped = 0

                async def start_ttfb_metrics(self) -> None:
                    pass

                async def stop_ttfb_metrics(self) -> None:
                    self.metrics_stopped += 1

                async def push_frame(self, frame) -> None:
                    self.frames.append(frame)

            async def slow_stream(_context):
                async def chunks():
                    await __import__("asyncio").sleep(0.05)
                    yield types.SimpleNamespace(content="遅れて届いた回答です。")

                return chunks()

            context = types.SimpleNamespace(get_messages=lambda: [])
            harness = Harness()
            await harness._process_context_common(context, slow_stream)
            return harness

        harness = __import__("asyncio").run(exercise_timeout())
        self.assertEqual(harness.metrics_stopped, 1)
        self.assertEqual([frame.content for frame in harness.frames], [tokkio_llm._LLM_TIMEOUT_REPLY])

    def test_tokkio_llm_retries_transient_worker_capacity_error_once(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()

        async def exercise_retry():
            class Harness(tokkio_llm.TokkioLLMServiceMixin):
                filler = []
                time_delay = 60.0
                first_response_timeout_s = 1.0
                llm_resource_retry_delay_s = 0.0

                def __init__(self) -> None:
                    self.frames = []
                    self.calls = 0

                async def start_ttfb_metrics(self) -> None:
                    pass

                async def stop_ttfb_metrics(self) -> None:
                    pass

                async def push_frame(self, frame) -> None:
                    self.frames.append(frame)

            harness = Harness()

            async def transient_stream(_context):
                harness.calls += 1
                if harness.calls == 1:
                    raise RuntimeError("ResourceExhausted: Worker local total request limit reached (38/32)")

                async def chunks():
                    yield types.SimpleNamespace(content="再試行で応答しました。")

                return chunks()

            context = types.SimpleNamespace(get_messages=lambda: [])
            await harness._process_context_common(context, transient_stream)
            return harness

        harness = __import__("asyncio").run(exercise_retry())
        self.assertEqual(harness.calls, 2)
        self.assertEqual([frame.content for frame in harness.frames], ["再試行で応答しました。"])
        self.assertIn("混み合っています", tokkio_llm._LLM_BUSY_REPLY)

    def test_speech_segment_buffer_emits_sentences_as_soon_as_punctuation_arrives(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer()

        self.assertEqual(buffer.feed("東京工科大学の"), [])
        self.assertEqual(buffer.feed("学長です。次に"), ["東京工科大学の学長です。"])
        self.assertEqual(buffer.flush(), ["次に"])

    def test_speech_segment_buffer_splits_long_clause_at_japanese_comma(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(soft_max_chars=18, hard_max_chars=30)

        segments = buffer.feed("材料強度学、複合材料、高信頼性材料について研究しています")

        self.assertEqual(segments, ["材料強度学、複合材料、"])
        self.assertEqual(buffer.flush(), ["高信頼性材料について研究しています"])

    def test_speech_segment_buffer_hard_splits_when_no_natural_boundary_exists(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(soft_max_chars=10, hard_max_chars=12)

        segments = buffer.feed("abcdefghijklmnopqrstuvwxyz")

        self.assertEqual(segments, ["abcdefghijkl", "mnopqrstuvwx"])
        self.assertEqual(buffer.flush(), ["yz"])

    def test_speech_segment_buffer_keeps_short_comma_clause_until_more_text_arrives(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(
            soft_max_chars=10,
            hard_max_chars=24,
            min_segment_chars=8,
        )

        self.assertEqual(buffer.feed("特に、"), [])
        self.assertEqual(buffer.feed("材料強度学について、次に"), ["特に、材料強度学について、"])
        self.assertEqual(buffer.flush(), ["次に"])

    def test_speech_segment_buffer_does_not_emit_punctuation_only_segments(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer(
            soft_max_chars=100,
            hard_max_chars=5,
            min_segment_chars=2,
        )

        self.assertEqual(buffer.feed("abcde"), ["abcde"])
        self.assertEqual(buffer.feed("。"), [])
        self.assertEqual(buffer.flush(), [])

    def test_speech_segment_buffer_converts_streamed_markdown_lists_to_spoken_sentences(self) -> None:
        tokkio_llm = load_tokkio_llm_with_stubs()
        buffer = tokkio_llm.SpeechSegmentBuffer()
        response = (
            "**学部特色入試** * 先進情報専攻:3名、社会情報専攻:2名 "
            "*試験日:9月27日 **全学部AO入試** *指定2教科は数学・英語"
        )

        segments = buffer.feed(response[:35]) + buffer.feed(response[35:]) + buffer.flush()
        spoken = "".join(segments)

        self.assertNotIn("*", spoken)
        self.assertNotIn("**", spoken)
        self.assertIn("学部特色入試。", spoken)
        self.assertIn("全学部AO入試。", spoken)
        self.assertIn("社会情報専攻:2名。試験日", spoken)

    def test_custom_config_yaml_uses_standard_japanese_prompt(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn('name: "香川豊"', yaml_text)
        self.assertIn('tts_processor: "IrodoriTTSService"', yaml_text)
        self.assertIn("IrodoriTTSService:", yaml_text)
        self.assertIn('base_url: "http://10.209.1.12:8021"', yaml_text)
        self.assertIn('voice: "kagawa"', yaml_text)
        self.assertIn("あなたは「{name}」として", yaml_text)
        self.assertIn("香川先生", yaml_text)
        self.assertIn("詳しい説明でも", yaml_text)
        self.assertIn("読み上げ用の文章", yaml_text)
        self.assertIn("自分への呼びかけ", yaml_text)
        self.assertIn("私は", yaml_text)
        self.assertIn("私の研究", yaml_text)
        self.assertIn("私の経歴", yaml_text)
        self.assertIn("自分のこととして", yaml_text)
        self.assertIn("東京工科大学 学長", yaml_text)
        self.assertIn("セラミックス複合材料センター長", yaml_text)
        self.assertIn("SiC/SiC複合材料", yaml_text)
        self.assertIn("AI/DX", yaml_text)
        self.assertIn("自然な標準語", yaml_text)
        self.assertIn("最優先の根拠", yaml_text)
        self.assertIn("定型的な案内を付けない", yaml_text)
        self.assertIn("過年度、年度未確認、変更予定、募集停止", yaml_text)
        self.assertIn("推測で補わず", yaml_text)
        self.assertIn("命令として実行せず", yaml_text)
        self.assertNotIn("大阪弁", yaml_text)
        self.assertNotIn("大阪・住之江", yaml_text)
        self.assertNotIn("大藪", yaml_text)
        self.assertNotIn("/no_think", yaml_text)

    def test_custom_config_yaml_uses_riva_asr_model(self) -> None:
        yaml_text = customize.build_config_yaml()

        self.assertIn(f'model: "{EXPECTED_ASR_MODEL}"', yaml_text)
        self.assertIn('        - "青嵐"', yaml_text)
        self.assertIn('        - "SEIRAN"', yaml_text)
        self.assertIn("    boosted_lm_score: 8.0", yaml_text)
        self.assertIn('"boosted_lm_words": config.RivaASRService.boosted_lm_words', customize.NEW_BOT_SNIPPET)
        self.assertIn('"boosted_lm_score": config.RivaASRService.boosted_lm_score', customize.NEW_BOT_SNIPPET)

    def test_patch_stt_snippet_upgrades_existing_generated_bot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(
                customize.PREVIOUS_CONFIG_DRIVEN_STT_SNIPPET,
                encoding="utf-8",
            )

            customize.patch_stt_snippet(bot_path)
            first_result = bot_path.read_text(encoding="utf-8")
            customize.patch_stt_snippet(bot_path)

            self.assertEqual(first_result, bot_path.read_text(encoding="utf-8"))
            self.assertIn(
                '"boosted_lm_words": config.RivaASRService.boosted_lm_words',
                first_result,
            )
            self.assertIn(
                '"boosted_lm_score": config.RivaASRService.boosted_lm_score',
                first_result,
            )

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

    def test_patch_rag_snippet_passes_top_k_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.OLD_RAG_SNIPPET, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn("vdb_top_k=config.NvidiaRAGService.vdb_top_k", updated)
            self.assertIn("reranker_top_k=config.NvidiaRAGService.reranker_top_k", updated)
            self.assertIn("multimodal_reranker_top_k=config.NvidiaRAGService.multimodal_reranker_top_k", updated)
            self.assertIn("enable_reranker=config.NvidiaRAGService.enable_reranker", updated)

    def test_patch_rag_snippet_updates_previous_top_k_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            bot_path.write_text(customize.PREVIOUS_RAG_SNIPPET, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_RAG_SNIPPET, updated)
            self.assertIn("multimodal_reranker_top_k=config.NvidiaRAGService.multimodal_reranker_top_k", updated)

    def test_patch_rag_snippet_updates_top_k_only_branch_to_router_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            top_k_only_snippet = customize.NEW_RAG_SNIPPET.split(
                "\n        if config.Pipeline.llm_processor == 'NvidiaLLMRAGRouterService':"
            )[0] + "\n"
            bot_path.write_text(top_k_only_snippet, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_RAG_SNIPPET, updated)
            self.assertIn("TokkioNvidiaLLMRAGRouterService", updated)

    def test_patch_rag_snippet_replaces_previous_router_branch_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot_path = Path(tmp) / "bot.py"
            previous_router_snippet = customize.NEW_RAG_SNIPPET
            for line in (
                "                provider=config.NvidiaRAGRouterService.provider,\n",
                "                local_db_path=config.NvidiaRAGRouterService.local_db_path,\n",
                "                local_top_k=config.NvidiaRAGRouterService.local_top_k,\n",
                "                local_max_context_chars=config.NvidiaRAGRouterService.local_max_context_chars,\n",
            ):
                previous_router_snippet = previous_router_snippet.replace(line, "")
            bot_path.write_text(previous_router_snippet, encoding="utf-8")

            customize.patch_rag_snippet(bot_path)

            updated = bot_path.read_text(encoding="utf-8")
            self.assertIn(customize.NEW_RAG_SNIPPET, updated)
            self.assertEqual(updated.count("if config.Pipeline.llm_processor == 'NvidiaLLMRAGRouterService':"), 1)

    def test_config_py_supports_irodori_tts_processor(self) -> None:
        self.assertIn('"IrodoriTTSService"', customize.CONFIG_PY)
        self.assertIn("class IrodoriTTSServiceConfig", customize.CONFIG_PY)
        self.assertIn("IrodoriTTSService: IrodoriTTSServiceConfig", customize.CONFIG_PY)
        self.assertIn("vdb_top_k: int = 12", customize.CONFIG_PY)
        self.assertIn("reranker_top_k: int = 5", customize.CONFIG_PY)
        self.assertIn("multimodal_reranker_top_k: int = 10", customize.CONFIG_PY)
        self.assertIn("enable_reranker: bool = True", customize.CONFIG_PY)
        self.assertIn('"NvidiaLLMRAGRouterService"', customize.CONFIG_PY)
        self.assertIn("class NvidiaRAGRouterService", customize.CONFIG_PY)
        self.assertIn('provider: Literal["nvidia", "local"]', customize.CONFIG_PY)
        self.assertIn("local_db_path: str", customize.CONFIG_PY)
        self.assertIn("local_top_k: int = 3", customize.CONFIG_PY)
        self.assertIn("route_keywords: list[str]", customize.CONFIG_PY)

    def test_bot_snippet_builds_irodori_tts_service(self) -> None:
        self.assertIn("IrodoriTTSService", customize.NEW_BOT_SNIPPET)
        self.assertIn("config.IrodoriTTSService.base_url", customize.NEW_BOT_SNIPPET)
        self.assertIn("response_format=config.IrodoriTTSService.response_format", customize.NEW_BOT_SNIPPET)

    def test_bot_snippet_builds_auto_rag_router_service(self) -> None:
        self.assertIn("TokkioNvidiaLLMRAGRouterService", customize.NEW_RAG_SNIPPET)
        self.assertIn("provider=config.NvidiaRAGRouterService.provider", customize.NEW_RAG_SNIPPET)
        self.assertIn("local_db_path=config.NvidiaRAGRouterService.local_db_path", customize.NEW_RAG_SNIPPET)
        self.assertIn("local_top_k=config.NvidiaRAGRouterService.local_top_k", customize.NEW_RAG_SNIPPET)
        self.assertIn("route_keywords=config.NvidiaRAGRouterService.route_keywords", customize.NEW_RAG_SNIPPET)
        self.assertIn(
            "fallback_to_llm_on_error=config.NvidiaRAGRouterService.fallback_to_llm_on_error",
            customize.NEW_RAG_SNIPPET,
        )

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
            self.assertIn('name: "香川豊"', profile_config)
            self.assertIn("東京工科大学 学長", profile_config)
            self.assertIn(f'base_url: "{EXPECTED_LLM_BASE_URL}"', profile_config)
            self.assertIn(f'model: "{EXPECTED_LLM_MODEL}"', profile_config)
            self.assertIn('tts_processor: "IrodoriTTSService"', profile_config)
            self.assertIn('base_url: "http://10.209.1.12:8021"', profile_config)
            self.assertTrue((llm_rag_dir / "src" / "tokkio_irodori_tts.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "tokkio_llm.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "local_rag.py").is_file())
            self.assertTrue((llm_rag_dir / "src" / "tokkio_rag.py").is_file())
            self.assertIn(
                "get_fast_profile_reply",
                (llm_rag_dir / "src" / "tokkio_llm.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"collection_names": [self.collection_name]',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"vdb_top_k": vdb_top_k',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"reranker_top_k": reranker_top_k',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"stop": self.stop_words or []',
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "if self.filler and not first_chunk_received",
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
                rag_mode="always",
                rag_server_url="http://10.209.1.12:8081",
                rag_collection_name="ace_kagawa",
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            self.assertIn('llm_processor: "NvidiaRAGService"', profile_config)
            self.assertIn(f'rag_server_url: "{EXPECTED_RAG_SERVER_URL}"', profile_config)
            self.assertIn(f'collection_name: "{EXPECTED_RAG_COLLECTION}"', profile_config)
            self.assertIn(f"max_tokens: {EXPECTED_RAG_MAX_TOKENS}", profile_config)
            self.assertIn(f"vdb_top_k: {EXPECTED_RAG_VDB_TOP_K}", profile_config)
            self.assertIn(f"reranker_top_k: {EXPECTED_RAG_RERANKER_TOP_K}", profile_config)
            self.assertIn(f"multimodal_reranker_top_k: {EXPECTED_RAG_MULTIMODAL_RERANKER_TOP_K}", profile_config)
            self.assertIn("enable_reranker: true", profile_config)

    def test_apply_patch_can_enable_auto_rag_router_in_profile_ace_controller_config(self) -> None:
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
            source_db_path = Path(tmp) / "local_rag.sqlite"
            source_manifest_path = Path(tmp) / "local_rag.manifest.json"
            source_db_path.write_bytes(b"sqlite-index")
            source_manifest_path.write_text('{"schema_version": 1}\n', encoding="utf-8")

            customize.apply_patch(
                ace_repo_dir,
                rag_enabled=True,
                rag_mode="auto",
                rag_provider="local",
                local_rag_db_path=str(Path(tmp) / "local_rag.sqlite"),
                rag_route_keywords=EXPECTED_RAG_ROUTE_KEYWORDS,
            )

            profile_config = profile_config_path.read_text(encoding="utf-8")
            self.assertIn('llm_processor: "NvidiaLLMRAGRouterService"', profile_config)
            self.assertIn("NvidiaRAGRouterService:", profile_config)
            self.assertIn('provider: "local"', profile_config)
            self.assertIn(f'local_db_path: "{EXPECTED_LOCAL_RAG_RUNTIME_DB}"', profile_config)
            self.assertIn('        - "EBC"', profile_config)
            self.assertIn("fallback_to_llm_on_error: false", profile_config)
            self.assertIn(
                "class TokkioNvidiaLLMRAGRouterService",
                (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "def search_index",
                (llm_rag_dir / "src" / "local_rag.py").read_text(encoding="utf-8"),
            )
            self.assertEqual((llm_rag_dir / "configs" / "local_rag.sqlite").read_bytes(), b"sqlite-index")
            self.assertEqual(
                (llm_rag_dir / "configs" / "local_rag.manifest.json").read_text(encoding="utf-8"),
                '{"schema_version": 1}\n',
            )
            generated_router = (llm_rag_dir / "src" / "tokkio_rag.py").read_text(encoding="utf-8")
            self.assertIn("assess_local_rag_query", generated_router)
            self.assertIn("resolve_conversation_query", generated_router)
            self.assertIn("format_hits_for_prompt_with_sources", generated_router)
            self.assertIn("grounded_direct_reply", generated_router)
            self.assertIn("ユーザー質問(検索用に正規化)", generated_router)
            self.assertIn("NvidiaRAGCitationsFrame", generated_router)

    def test_build_config_yaml_accepts_custom_irodori_tts_url(self) -> None:
        yaml_text = customize.build_config_yaml(irodori_tts_base_url="http://192.0.2.10:8021")

        self.assertIn('base_url: "http://192.0.2.10:8021"', yaml_text)

    def test_manage_tokkio_auto_rag_health_is_optional(self) -> None:
        script = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")

        self.assertIn('RAG_MODE="${TOKKIO_RAG_MODE:-auto}"', script)
        self.assertIn('RAG_PROVIDER="${TOKKIO_RAG_PROVIDER:-local}"', script)
        self.assertIn("is_rag_required()", script)
        self.assertIn("Local RAG provider selected; skipping NVIDIA RAG health check.", script)
        self.assertIn("RAG auto mode or local provider", script)
        self.assertIn("first_pod_name_by_prefix()", script)
        self.assertNotIn("awk -v prefix", script)
        self.assertIn('"local_rag.py"', script)
        self.assertIn("local_rag.sqlite", script)
        self.assertIn('"tokkio_llm.py"', script)
        self.assertIn("local_rag_manifest_name", script)
        self.assertIn("Staging generated ace-controller files", script)
        self.assertIn("Publishing the verified controller bundle", script)
        self.assertIn("publish_prepare_commands", script)
        self.assertIn("publish_commit_commands", script)
        self.assertIn(".${sync_id}.tmp", script)
        self.assertIn("mv -f", script)
        self.assertIn("Local RAG DB hashes and runtime bundle verification passed", script)
        self.assertIn("verify_index_bundle", script)
        self.assertIn("sha256sum", script)
        self.assertNotIn("sync_controller_runtime_files || true", script)


if __name__ == "__main__":
    unittest.main()
