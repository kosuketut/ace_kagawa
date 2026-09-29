from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import httpx

try:
    import fastapi
    from fastapi.testclient import TestClient
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


elevenlabs_server = load_module("elevenlabs_server", ROOT / "infra" / "tts" / "elevenlabs_server.py")


def configured_settings(**overrides):
    values = {"api_key": "test-key", "voice_id": "voice123"}
    values.update(overrides)
    return elevenlabs_server.ElevenLabsSettings(**values)


class ElevenLabsTtsServerTests(unittest.TestCase):
    def test_settings_defaults_use_v4_turbo_japanese_pcm16k_on_port_8022(self) -> None:
        settings = elevenlabs_server.ElevenLabsSettings.from_env({})

        self.assertEqual(settings.port, 8022)
        self.assertEqual(settings.voice, "kagawa")
        self.assertEqual(settings.model_id, "eleven_v4_turbo")
        self.assertEqual(settings.language_code, "ja")
        self.assertEqual(settings.sample_rate_hz, 16000)
        self.assertEqual(settings.voice_id, "")
        self.assertEqual(settings.api_key, "")
        self.assertIs(settings.normalize_text, True)
        self.assertIsNone(settings.stability)

    def test_settings_read_tokkio_env_names_and_hide_api_key(self) -> None:
        settings = elevenlabs_server.ElevenLabsSettings.from_env(
            {
                "TOKKIO_ELEVENLABS_API_KEY": "secret-value",
                "TOKKIO_ELEVENLABS_VOICE_ID": "abc",
                "ELEVENLABS_TTS_MODEL_ID": "eleven_v4",
                "ELEVENLABS_TTS_STABILITY": "0.5",
                "ELEVENLABS_TTS_SPEED": "0.95",
                "ELEVENLABS_TTS_PORT": "18022",
            }
        )

        self.assertEqual(settings.api_key, "secret-value")
        self.assertEqual(settings.voice_id, "abc")
        self.assertEqual(settings.model_id, "eleven_v4")
        self.assertEqual(settings.stability, 0.5)
        self.assertEqual(settings.speed, 0.95)
        self.assertEqual(settings.port, 18022)
        self.assertNotIn("secret-value", repr(settings))

    def test_explicit_elevenlabs_env_names_take_precedence(self) -> None:
        settings = elevenlabs_server.ElevenLabsSettings.from_env(
            {
                "ELEVENLABS_API_KEY": "primary",
                "TOKKIO_ELEVENLABS_API_KEY": "fallback",
                "ELEVENLABS_VOICE_ID": "primary-voice",
                "TOKKIO_ELEVENLABS_VOICE_ID": "fallback-voice",
            }
        )

        self.assertEqual(settings.api_key, "primary")
        self.assertEqual(settings.voice_id, "primary-voice")

    def test_build_speech_request_normalizes_japanese_numbers(self) -> None:
        url, payload = elevenlabs_server.build_speech_request(
            configured_settings(),
            "授業料は703,000円で、14時に集合です。",
        )

        self.assertEqual(
            url,
            "https://api.elevenlabs.io/v1/text-to-speech/voice123/stream?output_format=pcm_16000",
        )
        self.assertEqual(payload["text"], "授業料はななじゅうまんさんぜんえんで、じゅうよじに集合です。")
        self.assertEqual(payload["model_id"], "eleven_v4_turbo")
        self.assertEqual(payload["language_code"], "ja")
        self.assertNotIn("voice_settings", payload)

    def test_build_speech_request_passes_only_configured_voice_settings(self) -> None:
        _url, payload = elevenlabs_server.build_speech_request(
            configured_settings(stability=0.5, speed=0.9, normalize_text=False),
            "703,000円",
        )

        self.assertEqual(payload["text"], "703,000円")
        self.assertEqual(payload["voice_settings"], {"stability": 0.5, "speed": 0.9})

    def test_iter_aligned_pcm_never_splits_16_bit_samples(self) -> None:
        async def upstream():
            for chunk in (b"\x01", b"\x02\x03\x04\x05", b"\x06\x07", b"\x08\x09"):
                yield chunk

        async def collect():
            return [chunk async for chunk in elevenlabs_server.iter_aligned_pcm(upstream(), chunk_bytes=4)]

        chunks = asyncio.run(collect())

        self.assertTrue(all(len(chunk) % 2 == 0 for chunk in chunks))
        self.assertEqual(b"".join(chunks), bytes(range(1, 9)))


@unittest.skipIf(fastapi is None, "fastapi is not installed")
class ElevenLabsTtsEndpointTests(unittest.TestCase):
    def make_client(self, settings, handler):
        app = elevenlabs_server.create_app(settings, transport=httpx.MockTransport(handler))
        return TestClient(app)

    def test_speech_streams_upstream_pcm_with_normalized_text(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["api_key"] = request.headers.get("xi-api-key")
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, content=b"\x00\x01\x02\x03\x04")

        with self.make_client(configured_settings(), handler) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "4人です。", "voice": "kagawa", "response_format": "pcm", "stream": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Sample-Rate-Hz"], "16000")
        self.assertEqual(response.content, b"\x00\x01\x02\x03")
        self.assertEqual(seen["api_key"], "test-key")
        self.assertEqual(seen["payload"]["text"], "よにんです。")

    def test_speech_returns_wav_when_requested(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"\x00\x01" * 8)

        with self.make_client(configured_settings(), handler) as client:
            response = client.post("/v1/audio/speech", json={"input": "はい。", "response_format": "wav"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"RIFF"))

    def test_upstream_error_becomes_bad_gateway_with_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": {"message": "invalid api key"}})

        with self.make_client(configured_settings(), handler) as client:
            response = client.post("/v1/audio/speech", json={"input": "はい。", "stream": True})

        self.assertEqual(response.status_code, 502)
        self.assertIn("401", response.json()["detail"])
        self.assertIn("invalid api key", response.json()["detail"])

    def test_unknown_voice_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("upstream must not be called")

        with self.make_client(configured_settings(), handler) as client:
            response = client.post("/v1/audio/speech", json={"input": "はい。", "voice": "oyabu"})

        self.assertEqual(response.status_code, 400)

    def test_healthz_reports_unconfigured_voice_as_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("health check must not call ElevenLabs")

        with self.make_client(configured_settings(voice_id=""), handler) as client:
            unconfigured = client.get("/healthz")
        with self.make_client(configured_settings(), handler) as client:
            configured = client.get("/healthz")

        self.assertEqual(unconfigured.status_code, 503)
        self.assertIs(unconfigured.json()["voice_id_configured"], False)
        self.assertEqual(configured.status_code, 200)
        self.assertEqual(configured.json()["model_id"], "eleven_v4_turbo")
        self.assertNotIn("test-key", configured.text)


if __name__ == "__main__":
    unittest.main()
