from __future__ import annotations

import importlib.util
import sys
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
