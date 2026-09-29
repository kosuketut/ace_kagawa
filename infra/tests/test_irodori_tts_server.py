from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import fastapi
except ModuleNotFoundError:
    fastapi = None


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


irodori_server = load_module("irodori_server", ROOT / "infra" / "tts" / "irodori_server.py")


class IrodoriTtsServerTests(unittest.TestCase):
    def test_settings_defaults_use_data_storage_gpu0_and_kagawa_reference(self) -> None:
        settings = irodori_server.IrodoriSettings.from_env({})

        self.assertEqual(settings.data_root, Path("/data/ACE/irodori"))
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 8021)
        self.assertEqual(settings.hf_checkpoint, "Aratako/Irodori-TTS-v4.1-Small")
        self.assertEqual(settings.model_device, "cuda:0")
        self.assertEqual(settings.codec_device, "cuda:0")
        self.assertIsNone(settings.num_steps)
        self.assertIsNone(settings.max_ref_seconds)
        self.assertEqual(settings.duration_cap_base_seconds, 0.7)
        self.assertEqual(settings.duration_cap_mora_per_second, 6.0)
        self.assertEqual(settings.duration_scale, 1.0)
        self.assertEqual(settings.short_duration_max_chars, 16)
        self.assertEqual(settings.short_duration_scale, 0.8)
        self.assertEqual(settings.response_sample_rate_hz, 16000)
        self.assertEqual(settings.voice, "kagawa")
        self.assertEqual(settings.reference_source, ROOT / "Irodori-TTS" / "data" / "kagawa_voice.m4a")
        self.assertEqual(settings.reference_wav, Path("/data/ACE/irodori/reference/kagawa_voice_ref_48k_mono.wav"))
        self.assertEqual(
            settings.fixed_open_campus_greeting_pcm,
            Path("/data/ACE/irodori/fixed-phrases/open_campus_greeting_16k_mono.pcm"),
        )
        self.assertIs(getattr(settings, "short_cache_enabled", None), False)
        self.assertEqual(getattr(settings, "short_cache_max_chars", None), 40)
        self.assertEqual(getattr(settings, "short_cache_max_entries", None), 128)
        self.assertEqual(getattr(settings, "short_cache_prewarm_texts", None), ())
        self.assertIsNone(getattr(settings, "short_cache_prewarm_num_steps", None))
        self.assertEqual(getattr(settings, "stream_chunk_bytes", None), 3200)
        self.assertIs(getattr(settings, "progressive_stream_enabled", None), False)
        self.assertEqual(getattr(settings, "progressive_max_segments", None), 2)

    def test_settings_from_env_parses_short_cache_options(self) -> None:
        settings = irodori_server.IrodoriSettings.from_env(
            {
                "IRODORI_TTS_SHORT_CACHE_ENABLED": "true",
                "IRODORI_TTS_SHORT_CACHE_MAX_CHARS": "24",
                "IRODORI_TTS_SHORT_CACHE_MAX_ENTRIES": "32",
                "IRODORI_TTS_SHORT_CACHE_PREWARM_TEXTS": " こんにちは。 | | ありがとうございました。 ",
                "IRODORI_TTS_SHORT_CACHE_PREWARM_NUM_STEPS": "40",
                "IRODORI_TTS_SHORT_DURATION_MAX_CHARS": "12",
                "IRODORI_TTS_SHORT_DURATION_SCALE": "0.75",
                "IRODORI_TTS_FIXED_OPEN_CAMPUS_GREETING_PCM": "/tmp/fixed-greeting.pcm",
                "IRODORI_TTS_STREAM_CHUNK_BYTES": "6400",
                "IRODORI_TTS_PROGRESSIVE_STREAM_ENABLED": "true",
                "IRODORI_TTS_PROGRESSIVE_MAX_SEGMENTS": "4",
            }
        )

        self.assertIs(getattr(settings, "short_cache_enabled", None), True)
        self.assertEqual(getattr(settings, "short_cache_max_chars", None), 24)
        self.assertEqual(getattr(settings, "short_cache_max_entries", None), 32)
        self.assertEqual(
            getattr(settings, "short_cache_prewarm_texts", None),
            ("こんにちは。", "ありがとうございました。"),
        )
        self.assertEqual(getattr(settings, "short_cache_prewarm_num_steps", None), 40)
        self.assertEqual(settings.short_duration_max_chars, 12)
        self.assertEqual(settings.short_duration_scale, 0.75)
        self.assertEqual(settings.fixed_open_campus_greeting_pcm, Path("/tmp/fixed-greeting.pcm"))
        self.assertEqual(getattr(settings, "stream_chunk_bytes", None), 6400)
        self.assertIs(getattr(settings, "progressive_stream_enabled", None), True)
        self.assertEqual(getattr(settings, "progressive_max_segments", None), 4)

    def test_settings_from_env_parses_checkpoint_steps_and_reference_length(self) -> None:
        settings = irodori_server.IrodoriSettings.from_env(
            {
                "IRODORI_TTS_HF_CHECKPOINT": "Aratako/Irodori-TTS-v4.1-Small-MF",
                "IRODORI_TTS_NUM_STEPS": "8",
                "IRODORI_TTS_MAX_REF_SECONDS": "60",
                "IRODORI_TTS_DURATION_CAP_BASE_SECONDS": "1.2",
                "IRODORI_TTS_DURATION_CAP_MORA_PER_SECOND": "0",
            }
        )

        self.assertEqual(settings.hf_checkpoint, "Aratako/Irodori-TTS-v4.1-Small-MF")
        self.assertEqual(settings.num_steps, 8)
        self.assertEqual(settings.max_ref_seconds, 60.0)
        self.assertEqual(settings.duration_cap_base_seconds, 1.2)
        self.assertEqual(settings.duration_cap_mora_per_second, 0.0)

    def test_load_runtime_downloads_checkpoint_with_bundled_tokenizer(self) -> None:
        downloaded: list[str] = []
        runtime_keys: list[object] = []

        class FakeRuntimeKey:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class FakeInferenceRuntime:
            @classmethod
            def from_key(cls, key):
                runtime_keys.append(key)
                return cls()

        def fake_download_hf_checkpoint(source: str) -> str:
            downloaded.append(source)
            return "/cache/int8-weight-only/model.safetensors"

        fake_package = types.ModuleType("irodori_tts")
        fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
        fake_inference_runtime.InferenceRuntime = FakeInferenceRuntime
        fake_inference_runtime.RuntimeKey = FakeRuntimeKey
        fake_inference_runtime.download_hf_checkpoint = fake_download_hf_checkpoint
        previous_package = sys.modules.get("irodori_tts")
        previous_inference_runtime = sys.modules.get("irodori_tts.inference_runtime")
        sys.modules["irodori_tts"] = fake_package
        sys.modules["irodori_tts.inference_runtime"] = fake_inference_runtime
        try:
            settings = irodori_server.IrodoriSettings(
                hf_checkpoint="Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only",
            )
            synthesizer = irodori_server.IrodoriSynthesizer(settings)

            synthesizer.load_runtime()

            self.assertEqual(downloaded, ["Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only"])
            self.assertEqual(runtime_keys[0].checkpoint, "/cache/int8-weight-only/model.safetensors")
        finally:
            if previous_package is None:
                sys.modules.pop("irodori_tts", None)
            else:
                sys.modules["irodori_tts"] = previous_package
            if previous_inference_runtime is None:
                sys.modules.pop("irodori_tts.inference_runtime", None)
            else:
                sys.modules["irodori_tts.inference_runtime"] = previous_inference_runtime

    def test_audio_to_pcm16_resamples_clips_and_returns_little_endian_pcm(self) -> None:
        audio = np.linspace(-1.5, 1.5, num=480, dtype=np.float32)

        pcm = irodori_server.audio_to_pcm16(audio, source_sample_rate=48000, target_sample_rate=16000)

        self.assertEqual(len(pcm), 160 * 2)
        decoded = np.frombuffer(pcm, dtype="<i2")
        self.assertEqual(decoded.shape, (160,))
        self.assertEqual(decoded[0], -32768)
        self.assertEqual(decoded[-1], 32767)

    def test_wav_bytes_wrap_pcm16_audio(self) -> None:
        pcm = np.array([0, 1000, -1000], dtype="<i2").tobytes()

        wav = irodori_server.pcm16_to_wav_bytes(pcm, sample_rate_hz=16000)

        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:16])
        self.assertGreater(len(wav), len(pcm))

    def test_iter_pcm_chunks_keeps_16_bit_samples_intact(self) -> None:
        pcm = bytes(range(14))

        chunks = list(irodori_server.iter_pcm_chunks(pcm, chunk_bytes=5))

        self.assertEqual(chunks, [bytes(range(4)), bytes(range(4, 8)), bytes(range(8, 12)), bytes(range(12, 14))])

    def test_split_progressive_speech_input_prefers_sentence_boundaries(self) -> None:
        segments = irodori_server.split_progressive_speech_input(
            "香川豊です。材料強度を研究しています。よろしくお願いします。"
        )

        self.assertEqual(
            segments,
            ["香川ゆたかです。", "材料強度を研究しています。", "よろしくお願いします。"],
        )

    def test_split_progressive_speech_input_splits_long_clause_at_soft_boundary(self) -> None:
        segments = irodori_server.split_progressive_speech_input(
            "材料強度学、複合材料、高信頼性材料について研究しています",
            soft_max_chars=18,
            hard_max_chars=30,
            min_segment_chars=8,
        )

        self.assertEqual(segments[0], "材料強度学、複合材料、")
        self.assertEqual("".join(segments), "材料強度学、複合材料、高信頼性材料について研究しています")

    def test_coalesce_progressive_segments_keeps_first_segment_then_merges_tail(self) -> None:
        segments = irodori_server.coalesce_progressive_segments(
            ["最初です。", "次です。", "最後です。"],
            max_segments=2,
        )

        self.assertEqual(segments, ["最初です。", "次です。最後です。"])

    def test_normalize_speech_input_applies_kagawa_yutaka_pronunciation_hint(self) -> None:
        text = "香川豊さん、こんにちは。香川 豊の自己紹介です。"

        normalized = irodori_server.normalize_speech_input(text)

        self.assertEqual(normalized, "香川ゆたかさん、こんにちは。香川ゆたかの自己紹介です。")

    def test_normalize_speech_input_preserves_ai_and_reads_usj_as_japanese_initialism(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "私はAI学長です。いわゆるUSJとほぼ同じ広さです。OpenAIは別の語です。"
        )

        self.assertEqual(
            normalized,
            "私はAI学長です。いわゆるユーエスジェイとほぼ同じ広さです。"
            "OpenAIは別の語です。",
        )

    def test_normalize_speech_input_applies_seiran_pronunciation_without_duplicate_reading(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "青嵐はGPUスーパーコンピュータです。青嵐（せいらん、SEIRAN）の順位は398位です。"
        )

        self.assertEqual(
            normalized,
            "せいらんはGPUスーパーコンピュータです。せいらんの順位はさんびゃくきゅうじゅうはちいです。",
        )
        self.assertNotIn("青嵐", normalized)

    def test_normalize_speech_input_removes_farewell_period_for_stable_ending(self) -> None:
        normalized = irodori_server.normalize_speech_input(" ありがとうございました。 ")

        self.assertEqual(normalized, "ありがとうございました")

    def test_normalize_speech_input_removes_script_greeting_period_for_stable_ending(self) -> None:
        normalized = irodori_server.normalize_speech_input(" 皆さん、こんにちは。 ")

        self.assertEqual(normalized, "皆さん、こんにちは")

    def test_normalize_speech_input_verbalizes_rag_year_and_tuition_amount(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "2027年度にメディア学部へ入学する場合、"
            "1年目前期の合計は**976,300円**です。"
        )

        self.assertEqual(
            normalized,
            "にせんにじゅうななねんどにメディア学部へ入学する場合、"
            "いちねんめ前期の合計はきゅうじゅうななまんろくせんさんびゃくえんです。",
        )

    def test_normalize_speech_input_verbalizes_multiple_tuition_amounts(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "入学金250,000円、授業料703,000円、諸会費23,300円です。"
        )

        self.assertEqual(
            normalized,
            "入学金にじゅうごまんえん、授業料ななじゅうまんさんぜんえん、"
            "諸会費にまんさんぜんさんびゃくえんです。",
        )

    def test_normalize_speech_input_handles_dates_percentages_and_phone_numbers(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "期限は2026年12月15日、利用者は103名、全体の40%、電話は042-637-2011です。"
        )

        self.assertEqual(
            normalized,
            "期限はにせんにじゅうろくねんじゅうにがつじゅうごにち、"
            "利用者はひゃくさんめい、全体のよんじゅうぱーせんと、"
            "電話はぜろよんにのろくさんななのにぜろいちいちです。",
        )

    def test_normalize_speech_input_reads_age_rank_and_hardware_counters(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "私は現在73歳です。青嵐はTOP500で398位、GPUは8基、96基、100基、ノードは12台です。"
        )

        self.assertEqual(
            normalized,
            "私は現在ななじゅうさんさいです。せいらんはトップごひゃくで"
            "さんびゃくきゅうじゅうはちい、GPUははっき、きゅうじゅうろっき、"
            "ひゃっき、ノードはじゅうにだいです。",
        )

    def test_normalize_speech_input_reads_seiran_performance_and_capacity_units(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "性能は3.66392 PFLOPS、98.4203 TFLOPS、容量は17.28TB、CPUは15,552 coresです。"
        )

        self.assertEqual(
            normalized,
            "性能はさんてんろくろくさんきゅうにペタフロップス、"
            "きゅうじゅうはちてんよんにぜろさんテラフロップス、"
            "容量はじゅうななてんにはちテラバイト、CPUは"
            "いちまんごせんごひゃくごじゅうにコアです。",
        )

    def test_normalize_speech_input_reads_multi_dot_software_versions(self) -> None:
        normalized = irodori_server.normalize_speech_input("CUDA 13.2.1.1、Ubuntu 24.04.3 LTSです。")

        self.assertEqual(
            normalized,
            "CUDA じゅうさんてんにてんいちてんいち、Ubuntu にじゅうよんてんぜろよんてんさん LTSです。",
        )

    def test_normalize_speech_input_does_not_rewrite_product_name_digits(self) -> None:
        normalized = irodori_server.normalize_speech_input("RTX4090とCC2020です。")

        self.assertEqual(normalized, "RTX4090とCC2020です。")

    def test_normalize_speech_input_naturalizes_rag_markdown_weekdays_and_times(self) -> None:
        normalized = irodori_server.normalize_speech_input(
            "**全学部AO入試** *先進情報専攻:22名 "
            "*試験日:9月27日(日) *当日は9:00~、試験は10:30~11:10、"
            "数学・英語の2教科です。"
        )

        self.assertNotIn("*", normalized)
        self.assertNotIn(":", normalized)
        self.assertNotIn("~", normalized)
        self.assertNotIn("・", normalized)
        self.assertIn("全学部エーオー入試。", normalized)
        self.assertIn("先進情報専攻は、にじゅうにめい", normalized)
        self.assertIn("くじから", normalized)
        self.assertIn("じゅうじさんじゅっぷんからじゅういちじじゅっぷんまで", normalized)
        self.assertIn("日曜日", normalized)
        self.assertIn("数学、英語のにきょうか", normalized)

    def test_normalize_speech_input_reads_counter_sound_changes(self) -> None:
        cases = {
            "14時に集合です。": "じゅうよじに集合です。",
            "17時と19時です。": "じゅうしちじとじゅうくじです。",
            "14:30から始まります。": "じゅうよじさんじゅっぷんから始まります。",
            "0時です。": "れいじです。",
            "1時間半、4時間、7時間、9時間です。": "いちじかん半、よじかん、ななじかん、くじかんです。",
            "1人、2人、4人、14人です。": "ひとり、ふたり、よにん、じゅうよにんです。",
            "20歳です。": "はたちです。",
            "1つ目の理由は2つあります。": "ひとつ目の理由はふたつあります。",
            "3つ、4つ、10個です。": "みっつ、よっつ、じゅっこです。",
            "4年生と14年前、9年間です。": "よねん生とじゅうよねん前、きゅうねんかんです。",
            "令和8年です。": "令和はちねんです。",
            "3階と13階です。": "さんがいとじゅうさんがいです。",
            "1週間と8週間です。": "いっしゅうかんとはっしゅうかんです。",
            "3か月、1ヶ月、6カ月、2か所です。": "さんかげつ、いっかげつ、ろっかげつ、にかしょです。",
            "1点と10点です。": "いってんとじゅってんです。",
            "1期生です。": "いっき生です。",
            "1泊2日です。": "いっぱくふつかです。",
            "60周年です。": "ろくじゅっしゅうねんです。",
            "100歳、600点、100個、100本です。": "ひゃくさい、ろっぴゃくてん、ひゃっこ、ひゃっぽんです。",
            "3分野と2分の1です。": "さん分野とにぶんのいちです。",
            "3〜4割です。": "さん割からよん割です。",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(irodori_server.normalize_speech_input(text), expected)

    def test_normalize_speech_input_reads_dates_days_and_ranges(self) -> None:
        cases = {
            "4/1（火）から開始します。": "しがつついたち、火曜日、から開始します。",
            "2026/4/1です。": "にせんにじゅうろくねんしがつついたちです。",
            "24/7です。": "にじゅうよん/ななです。",
            "4月1日と毎月1日です。": "しがつついたちと毎月ついたちです。",
            "1日目と7日間、1日に2回です。": "いちにちめとなのかかん、いちにちににかいです。",
            "2〜3人、1〜3月です。": "ふたりからさんにん、いちがつからさんがつです。",
            "9〜17時です。": "くじからじゅうしちじです。",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(irodori_server.normalize_speech_input(text), expected)

    def test_normalize_speech_input_reads_money_decimals_and_units(self) -> None:
        cases = {
            "1000万円と1億円です。": "いっせん万円といち億円です。",
            "10000000円です。": "いっせんまんえんです。",
            "¥5,000です。": "ごせんえんです。",
            "0.5秒です。": "れいてんご秒です。",
            "36.5℃、100m、5km、10kgです。": (
                "さんじゅうろくてんごど、ひゃくメートル、ごキロメートル、じゅうキログラムです。"
            ),
            "5Gと3Dです。": "5Gと3Dです。",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(irodori_server.normalize_speech_input(text), expected)

    def test_synthesize_passes_pronunciation_normalized_text_to_runtime(self) -> None:
        class FakeSamplingRequest:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class FakeResult:
            audio = np.array([0.0], dtype=np.float32)
            sample_rate = 16000

        class FakeRuntime:
            def __init__(self) -> None:
                self.request_text: str | None = None

            def synthesize(self, request, log_fn=None):
                self.request_text = request.text
                return FakeResult()

        fake_package = types.ModuleType("irodori_tts")
        fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
        fake_inference_runtime.SamplingRequest = FakeSamplingRequest
        previous_package = sys.modules.get("irodori_tts")
        previous_inference_runtime = sys.modules.get("irodori_tts.inference_runtime")
        sys.modules["irodori_tts"] = fake_package
        sys.modules["irodori_tts.inference_runtime"] = fake_inference_runtime
        try:
            settings = irodori_server.IrodoriSettings(
                reference_source=Path(__file__),
                reference_wav=Path(__file__),
            )
            runtime = FakeRuntime()
            synthesizer = irodori_server.IrodoriSynthesizer(settings)
            synthesizer._runtime = runtime

            synthesizer.synthesize(" 香川豊です。 ", response_format="pcm")

            self.assertEqual(runtime.request_text, "香川ゆたかです。")
        finally:
            if previous_package is None:
                sys.modules.pop("irodori_tts", None)
            else:
                sys.modules["irodori_tts"] = previous_package
            if previous_inference_runtime is None:
                sys.modules.pop("irodori_tts.inference_runtime", None)
            else:
                sys.modules["irodori_tts.inference_runtime"] = previous_inference_runtime

    def test_synthesize_caches_repeated_short_text_when_enabled(self) -> None:
        class FakeSamplingRequest:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class FakeResult:
            def __init__(self, value: float) -> None:
                self.audio = np.array([value], dtype=np.float32)
                self.sample_rate = 16000

        class FakeRuntime:
            def __init__(self) -> None:
                self.calls = 0
                self.duration_scales: list[float] = []

            def synthesize(self, request, log_fn=None):
                self.calls += 1
                self.duration_scales.append(request.duration_scale)
                return FakeResult(0.1 * self.calls)

        fake_package = types.ModuleType("irodori_tts")
        fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
        fake_inference_runtime.SamplingRequest = FakeSamplingRequest
        previous_package = sys.modules.get("irodori_tts")
        previous_inference_runtime = sys.modules.get("irodori_tts.inference_runtime")
        sys.modules["irodori_tts"] = fake_package
        sys.modules["irodori_tts.inference_runtime"] = fake_inference_runtime
        try:
            settings = irodori_server.IrodoriSettings(
                reference_source=Path(__file__),
                reference_wav=Path(__file__),
                duration_scale=1.25,
                short_duration_max_chars=16,
                short_duration_scale=0.8,
                short_cache_enabled=True,
                short_cache_max_chars=20,
                short_cache_max_entries=128,
            )
            runtime = FakeRuntime()
            synthesizer = irodori_server.IrodoriSynthesizer(settings)
            synthesizer._runtime = runtime

            first = synthesizer.synthesize("香川豊です。", response_format="pcm")
            second = synthesizer.synthesize(" 香川 豊です。 ", response_format="pcm")
            synthesizer.synthesize(
                "これは短文補正を使わない十分に長い文章です。",
                response_format="pcm",
            )

            disabled_settings = irodori_server.IrodoriSettings(
                reference_source=Path(__file__),
                reference_wav=Path(__file__),
                duration_scale=1.25,
                short_duration_max_chars=0,
                short_duration_scale=0.8,
            )
            disabled_synthesizer = irodori_server.IrodoriSynthesizer(disabled_settings)
            disabled_synthesizer._runtime = runtime
            disabled_synthesizer.synthesize("こんにちは", response_format="pcm")

            self.assertEqual(first, second)
            self.assertEqual(runtime.calls, 3)
            self.assertEqual(runtime.duration_scales, [0.8, 1.25, 1.25])
        finally:
            if previous_package is None:
                sys.modules.pop("irodori_tts", None)
            else:
                sys.modules["irodori_tts"] = previous_package
            if previous_inference_runtime is None:
                sys.modules.pop("irodori_tts.inference_runtime", None)
            else:
                sys.modules["irodori_tts.inference_runtime"] = previous_inference_runtime

    def test_estimate_speech_mora_counts_kana_kanji_and_ascii(self) -> None:
        self.assertEqual(irodori_server.estimate_speech_mora("はい。"), 2.0)
        # A pause at internal punctuation counts as two mora; the final "。" does not.
        self.assertEqual(irodori_server.estimate_speech_mora("ええ、そうです。"), 8.0)
        self.assertEqual(irodori_server.estimate_speech_mora("しょうしょう"), 4.0)
        self.assertAlmostEqual(irodori_server.estimate_speech_mora("私はAI"), 1.7 + 1.0 + 3.0)

    def test_synthesize_caps_predicted_duration_by_estimated_mora(self) -> None:
        class FakeSamplingRequest:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class FakeResult:
            audio = np.array([0.0], dtype=np.float32)
            sample_rate = 16000

        class FakeRuntime:
            def __init__(self) -> None:
                self.max_seconds: list[float] = []

            def synthesize(self, request, log_fn=None):
                self.max_seconds.append(request.max_seconds)
                return FakeResult()

        fake_package = types.ModuleType("irodori_tts")
        fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
        fake_inference_runtime.SamplingRequest = FakeSamplingRequest
        previous_package = sys.modules.get("irodori_tts")
        previous_inference_runtime = sys.modules.get("irodori_tts.inference_runtime")
        sys.modules["irodori_tts"] = fake_package
        sys.modules["irodori_tts.inference_runtime"] = fake_inference_runtime
        try:
            runtime = FakeRuntime()
            for cap_mora_per_second in (6.0, 0.0):
                settings = irodori_server.IrodoriSettings(
                    reference_source=Path(__file__),
                    reference_wav=Path(__file__),
                    duration_cap_base_seconds=0.7,
                    duration_cap_mora_per_second=cap_mora_per_second,
                )
                synthesizer = irodori_server.IrodoriSynthesizer(settings)
                synthesizer._runtime = runtime
                synthesizer.synthesize("はい。", response_format="pcm")
                synthesizer.synthesize("少々お待ちください。", response_format="pcm")

            # "はい。" hits the 1.2 s floor; 11.1 mora -> 0.7 + 11.1 / 6.
            self.assertEqual(runtime.max_seconds, [1.2, 2.55, 30.0, 30.0])
        finally:
            if previous_package is None:
                sys.modules.pop("irodori_tts", None)
            else:
                sys.modules["irodori_tts"] = previous_package
            if previous_inference_runtime is None:
                sys.modules.pop("irodori_tts.inference_runtime", None)
            else:
                sys.modules["irodori_tts.inference_runtime"] = previous_inference_runtime

    def test_prewarm_short_cache_synthesizes_fixed_phrases_only_once(self) -> None:
        class FakeSamplingRequest:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class FakeResult:
            audio = np.array([0.1], dtype=np.float32)
            sample_rate = 16000

        class FakeRuntime:
            def __init__(self) -> None:
                self.texts: list[str] = []
                self.num_steps: list[int] = []

            def synthesize(self, request, log_fn=None):
                self.texts.append(request.text)
                self.num_steps.append(request.num_steps)
                return FakeResult()

        fake_package = types.ModuleType("irodori_tts")
        fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
        fake_inference_runtime.SamplingRequest = FakeSamplingRequest
        previous_package = sys.modules.get("irodori_tts")
        previous_inference_runtime = sys.modules.get("irodori_tts.inference_runtime")
        sys.modules["irodori_tts"] = fake_package
        sys.modules["irodori_tts.inference_runtime"] = fake_inference_runtime
        try:
            settings = irodori_server.IrodoriSettings(
                reference_source=Path(__file__),
                reference_wav=Path(__file__),
                short_cache_enabled=True,
                short_cache_max_chars=20,
                short_cache_max_entries=128,
            )
            runtime = FakeRuntime()
            synthesizer = irodori_server.IrodoriSynthesizer(settings)
            synthesizer._runtime = runtime

            warmed = synthesizer.prewarm_short_cache(
                ("こんにちは。", "ありがとうございました。"),
                num_steps=40,
            )
            cached = synthesizer.synthesize("ありがとうございました。", response_format="pcm")

            self.assertEqual(warmed, 2)
            self.assertTrue(cached[0])
            self.assertEqual(runtime.texts, ["こんにちは。", "ありがとうございました"])
            self.assertEqual(runtime.num_steps, [40, 40])
            self.assertEqual(synthesizer.short_cache_size(), 2)
        finally:
            if previous_package is None:
                sys.modules.pop("irodori_tts", None)
            else:
                sys.modules["irodori_tts"] = previous_package
            if previous_inference_runtime is None:
                sys.modules.pop("irodori_tts.inference_runtime", None)
            else:
                sys.modules["irodori_tts.inference_runtime"] = previous_inference_runtime

    def test_synthesize_uses_prerecorded_pcm_for_open_campus_greeting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixed_path = Path(tmp) / "greeting.pcm"
            pcm = np.array([0, 1000, -1000, 500], dtype="<i2").tobytes()
            fixed_path.write_bytes(pcm)
            settings = irodori_server.IrodoriSettings(
                fixed_open_campus_greeting_pcm=fixed_path,
                response_sample_rate_hz=16000,
            )
            synthesizer = irodori_server.IrodoriSynthesizer(settings)

            prerecorded = synthesizer.synthesize("皆さん、こんにちは。", response_format="pcm")
            prerecorded_wav = synthesizer.synthesize("皆さん、こんにちは。", response_format="wav")

            self.assertEqual(prerecorded, (pcm, "audio/L16", 16000))
            self.assertTrue(prerecorded_wav[0].startswith(b"RIFF"))
            self.assertEqual(prerecorded_wav[1:], ("audio/wav", 16000))
            self.assertIsNone(synthesizer._runtime)

    def test_iter_pcm_stream_synthesizes_progressive_segments_in_order(self) -> None:
        class FakeSamplingRequest:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class FakeResult:
            def __init__(self, value: float) -> None:
                self.audio = np.array([value], dtype=np.float32)
                self.sample_rate = 16000

        class FakeRuntime:
            def __init__(self) -> None:
                self.texts: list[str] = []

            def synthesize(self, request, log_fn=None):
                self.texts.append(request.text)
                return FakeResult(0.1 * len(self.texts))

        fake_package = types.ModuleType("irodori_tts")
        fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
        fake_inference_runtime.SamplingRequest = FakeSamplingRequest
        previous_package = sys.modules.get("irodori_tts")
        previous_inference_runtime = sys.modules.get("irodori_tts.inference_runtime")
        sys.modules["irodori_tts"] = fake_package
        sys.modules["irodori_tts.inference_runtime"] = fake_inference_runtime
        try:
            settings = irodori_server.IrodoriSettings(
                reference_source=Path(__file__),
                reference_wav=Path(__file__),
                short_cache_enabled=False,
            )
            runtime = FakeRuntime()
            synthesizer = irodori_server.IrodoriSynthesizer(settings)
            synthesizer._runtime = runtime

            chunks = list(
                synthesizer.iter_pcm_stream(
                    "香川豊です。材料強度を研究しています。",
                    chunk_bytes=3200,
                )
            )

            self.assertEqual(runtime.texts, ["香川ゆたかです。", "材料強度を研究しています。"])
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all(chunks))
        finally:
            if previous_package is None:
                sys.modules.pop("irodori_tts", None)
            else:
                sys.modules["irodori_tts"] = previous_package
            if previous_inference_runtime is None:
                sys.modules.pop("irodori_tts.inference_runtime", None)
            else:
                sys.modules["irodori_tts.inference_runtime"] = previous_inference_runtime

    @unittest.skipIf(fastapi is None, "fastapi is not installed in this Python environment")
    def test_speech_endpoint_treats_request_as_json_body(self) -> None:
        class FakeSynthesizer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def synthesize(self, text: str, *, response_format: str = "pcm"):
                self.calls.append((text, response_format))
                return b"\x01\x00", "audio/L16", 16000

        settings = irodori_server.IrodoriSettings()
        synthesizer = FakeSynthesizer()
        app = irodori_server.create_app(settings, synthesizer)
        speech_route = next(route for route in app.routes if getattr(route, "path", "") == "/v1/audio/speech")

        self.assertEqual([param.name for param in speech_route.dependant.body_params], ["request"])
        self.assertNotIn("request", [param.name for param in speech_route.dependant.query_params])

    def test_progressive_pcm_response_iterator_uses_stream_generator_lazily(self) -> None:
        class FakeSynthesizer:
            def __init__(self) -> None:
                self.synthesize_calls = 0
                self.stream_calls: list[tuple[str, int]] = []

            def synthesize(self, text: str, *, response_format: str = "pcm"):
                self.synthesize_calls += 1
                raise AssertionError("stream=true must not synthesize the full request before responding")

            def iter_pcm_stream(self, text: str, *, chunk_bytes: int):
                self.stream_calls.append((text, chunk_bytes))
                yield b"\x01\x00"
                yield b"\x02\x00"

            def short_cache_size(self) -> int:
                return 0

        synthesizer = FakeSynthesizer()

        body_iter = irodori_server.iter_progressive_pcm_response(
            synthesizer,
            "香川豊です。",
            chunk_bytes=4,
        )

        self.assertEqual(synthesizer.stream_calls, [])
        self.assertEqual(b"".join(body_iter), b"\x01\x00\x02\x00")
        self.assertEqual(synthesizer.synthesize_calls, 0)
        self.assertEqual(synthesizer.stream_calls, [("香川豊です。", 4)])

    def test_streaming_pcm_response_iterator_defaults_to_full_synthesis_chunks(self) -> None:
        class FakeSynthesizer:
            def __init__(self) -> None:
                self.synthesize_calls: list[tuple[str, str]] = []
                self.stream_calls = 0

            def synthesize(self, text: str, *, response_format: str = "pcm"):
                self.synthesize_calls.append((text, response_format))
                return b"\x01\x00\x02\x00\x03\x00", "audio/L16", 16000

            def iter_pcm_stream(self, text: str, *, chunk_bytes: int):
                self.stream_calls += 1
                raise AssertionError("progressive stream must be opt-in")

        synthesizer = FakeSynthesizer()
        settings = irodori_server.IrodoriSettings(stream_chunk_bytes=4)

        body = b"".join(
            irodori_server.iter_streaming_pcm_response(
                synthesizer,
                "香川豊です。",
                settings=settings,
            )
        )

        self.assertEqual(body, b"\x01\x00\x02\x00\x03\x00")
        self.assertEqual(synthesizer.synthesize_calls, [("香川豊です。", "pcm")])
        self.assertEqual(synthesizer.stream_calls, 0)

    def test_streaming_pcm_response_iterator_uses_progressive_when_enabled(self) -> None:
        class FakeSynthesizer:
            def __init__(self) -> None:
                self.synthesize_calls = 0
                self.stream_calls: list[tuple[str, int]] = []

            def synthesize(self, text: str, *, response_format: str = "pcm"):
                self.synthesize_calls += 1
                raise AssertionError("progressive stream should not synthesize full text first")

            def iter_pcm_stream(self, text: str, *, chunk_bytes: int):
                self.stream_calls.append((text, chunk_bytes))
                yield b"\x01\x00"
                yield b"\x02\x00"

        synthesizer = FakeSynthesizer()
        settings = irodori_server.IrodoriSettings(
            stream_chunk_bytes=4,
            progressive_stream_enabled=True,
        )

        body_iter = irodori_server.iter_streaming_pcm_response(
            synthesizer,
            "香川豊です。",
            settings=settings,
        )

        self.assertEqual(synthesizer.stream_calls, [])
        self.assertEqual(b"".join(body_iter), b"\x01\x00\x02\x00")
        self.assertEqual(synthesizer.synthesize_calls, 0)
        self.assertEqual(synthesizer.stream_calls, [("香川豊です。", 4)])


if __name__ == "__main__":
    unittest.main()
