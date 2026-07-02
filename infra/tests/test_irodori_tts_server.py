from __future__ import annotations

import importlib.util
import sys
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
        self.assertEqual(settings.hf_checkpoint, "Aratako/Irodori-TTS-500M-v3")
        self.assertEqual(settings.model_device, "cuda:0")
        self.assertEqual(settings.codec_device, "cuda:0")
        self.assertEqual(settings.num_steps, 40)
        self.assertEqual(settings.duration_scale, 1.0)
        self.assertEqual(settings.response_sample_rate_hz, 16000)
        self.assertEqual(settings.voice, "kagawa")
        self.assertEqual(settings.reference_source, ROOT / "Irodori-TTS" / "data" / "kagawa_voice.m4a")
        self.assertEqual(settings.reference_wav, Path("/data/ACE/irodori/reference/kagawa_voice_ref_48k_mono.wav"))
        self.assertIs(getattr(settings, "short_cache_enabled", None), False)
        self.assertEqual(getattr(settings, "short_cache_max_chars", None), 40)
        self.assertEqual(getattr(settings, "short_cache_max_entries", None), 128)
        self.assertEqual(getattr(settings, "stream_chunk_bytes", None), 3200)

    def test_settings_from_env_parses_short_cache_options(self) -> None:
        settings = irodori_server.IrodoriSettings.from_env(
            {
                "IRODORI_TTS_SHORT_CACHE_ENABLED": "true",
                "IRODORI_TTS_SHORT_CACHE_MAX_CHARS": "24",
                "IRODORI_TTS_SHORT_CACHE_MAX_ENTRIES": "32",
                "IRODORI_TTS_STREAM_CHUNK_BYTES": "6400",
            }
        )

        self.assertIs(getattr(settings, "short_cache_enabled", None), True)
        self.assertEqual(getattr(settings, "short_cache_max_chars", None), 24)
        self.assertEqual(getattr(settings, "short_cache_max_entries", None), 32)
        self.assertEqual(getattr(settings, "stream_chunk_bytes", None), 6400)

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

    def test_normalize_speech_input_applies_kagawa_yutaka_pronunciation_hint(self) -> None:
        text = "香川豊さん、こんにちは。香川 豊の自己紹介です。"

        normalized = irodori_server.normalize_speech_input(text)

        self.assertEqual(normalized, "香川ゆたかさん、こんにちは。香川ゆたかの自己紹介です。")

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

            def synthesize(self, request, log_fn=None):
                self.calls += 1
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
                short_cache_enabled=True,
                short_cache_max_chars=20,
                short_cache_max_entries=128,
            )
            runtime = FakeRuntime()
            synthesizer = irodori_server.IrodoriSynthesizer(settings)
            synthesizer._runtime = runtime

            first = synthesizer.synthesize("香川豊です。", response_format="pcm")
            second = synthesizer.synthesize(" 香川 豊です。 ", response_format="pcm")

            self.assertEqual(first, second)
            self.assertEqual(runtime.calls, 1)
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


if __name__ == "__main__":
    unittest.main()
