from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Frame:
    pass


class TTSAudioRawFrame(Frame):
    def __init__(self, *, audio: bytes, sample_rate: int, num_channels: int) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.num_channels = num_channels


class TTSStartedFrame(Frame):
    pass


class TTSStoppedFrame(Frame):
    pass


class TTSTextFrame(Frame):
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTTSService:
    def __init__(self, *, sample_rate: int, push_text_frames: bool, push_stop_frames: bool, **kwargs) -> None:
        self._sample_rate = sample_rate
        self.push_text_frames = push_text_frames
        self.push_stop_frames = push_stop_frames
        self.voice = None
        self.ttfb_stops = 0
        self.usage_texts: list[str] = []

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    async def start_ttfb_metrics(self) -> None:
        pass

    async def stop_ttfb_metrics(self) -> None:
        self.ttfb_stops += 1

    async def start_tts_usage_metrics(self, text: str) -> None:
        self.usage_texts.append(text)


def load_tokkio_irodori_tts():
    httpx_module = types.ModuleType("httpx")
    httpx_module.Timeout = lambda value: value

    class FakeLogger:
        def debug(self, *args, **kwargs) -> None:
            pass

    loguru_module = types.ModuleType("loguru")
    loguru_module.logger = FakeLogger()

    frames_module = types.ModuleType("pipecat.frames.frames")
    frames_module.Frame = Frame
    frames_module.TTSAudioRawFrame = TTSAudioRawFrame
    frames_module.TTSStartedFrame = TTSStartedFrame
    frames_module.TTSStoppedFrame = TTSStoppedFrame
    frames_module.TTSTextFrame = TTSTextFrame

    tts_service_module = types.ModuleType("pipecat.services.tts_service")
    tts_service_module.TTSService = FakeTTSService

    previous = {
        name: sys.modules.get(name)
        for name in (
            "pipecat",
            "pipecat.frames",
            "pipecat.frames.frames",
            "pipecat.services",
            "pipecat.services.tts_service",
            "httpx",
            "loguru",
        )
    }
    sys.modules["httpx"] = httpx_module
    sys.modules["loguru"] = loguru_module
    sys.modules["pipecat"] = types.ModuleType("pipecat")
    sys.modules["pipecat.frames"] = types.ModuleType("pipecat.frames")
    sys.modules["pipecat.frames.frames"] = frames_module
    sys.modules["pipecat.services"] = types.ModuleType("pipecat.services")
    sys.modules["pipecat.services.tts_service"] = tts_service_module

    spec = importlib.util.spec_from_file_location(
        "tokkio_irodori_tts_under_test",
        ROOT / "infra" / "tokkio" / "tokkio_irodori_tts.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load tokkio_irodori_tts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tokkio_irodori_tts_under_test"] = module
    spec.loader.exec_module(module)

    return module, previous


def restore_modules(previous: dict[str, types.ModuleType | None]) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TokkioIrodoriTtsTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_tts_streams_pcm_chunks_as_audio_frames(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            class FakeResponse:
                headers = {"X-Sample-Rate-Hz": "22050"}

                def __init__(self) -> None:
                    self.chunk_size: int | None = None

                def raise_for_status(self) -> None:
                    pass

                async def aiter_bytes(self, chunk_size: int | None = None):
                    self.chunk_size = chunk_size
                    yield b"\x01\x00\x02\x00"
                    yield b"\x03\x00"

            class FakeStreamContext:
                def __init__(self, response: FakeResponse) -> None:
                    self.response = response

                async def __aenter__(self) -> FakeResponse:
                    return self.response

                async def __aexit__(self, exc_type, exc, tb) -> None:
                    pass

            class FakeAsyncClient:
                last_instance: "FakeAsyncClient | None" = None

                def __init__(self, *, timeout) -> None:
                    self.timeout = timeout
                    self.requests: list[dict[str, object]] = []
                    self.response = FakeResponse()
                    FakeAsyncClient.last_instance = self

                async def __aenter__(self) -> "FakeAsyncClient":
                    return self

                async def __aexit__(self, exc_type, exc, tb) -> None:
                    pass

                def stream(self, method: str, url: str, *, json: dict[str, object]):
                    self.requests.append({"method": method, "url": url, "json": json})
                    return FakeStreamContext(self.response)

            module.httpx.AsyncClient = FakeAsyncClient
            service = module.IrodoriTTSService(
                base_url="http://tts.local",
                voice="kagawa",
                sample_rate=16000,
                response_format="pcm",
                stream_audio=True,
                stream_chunk_bytes=4096,
            )

            self.assertFalse(service.push_stop_frames)
            frames = [frame async for frame in service.run_tts("こんにちは。")]

            audio_frames = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
            self.assertEqual([frame.audio for frame in audio_frames], [b"\x01\x00\x02\x00", b"\x03\x00"])
            self.assertEqual([frame.sample_rate for frame in audio_frames], [22050, 22050])
            self.assertEqual([frame.num_channels for frame in audio_frames], [1, 1])
            self.assertIsInstance(frames[0], TTSStartedFrame)
            self.assertIsInstance(frames[1], TTSTextFrame)
            self.assertIsInstance(frames[-1], TTSStoppedFrame)
            self.assertEqual(service.ttfb_stops, 1)
            self.assertEqual(service.usage_texts, ["こんにちは。"])

            client = FakeAsyncClient.last_instance
            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(
                client.requests,
                [
                    {
                        "method": "POST",
                        "url": "http://tts.local/v1/audio/speech",
                        "json": {
                            "input": "こんにちは。",
                            "voice": "kagawa",
                            "response_format": "pcm",
                            "stream": True,
                        },
                    }
                ],
            )
            self.assertEqual(client.response.chunk_size, 4096)
        finally:
            restore_modules(previous)


if __name__ == "__main__":
    unittest.main()
