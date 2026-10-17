from __future__ import annotations

import asyncio
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


class ErrorFrame(Frame):
    def __init__(self, error: str, fatal: bool = False) -> None:
        self.error = error
        self.fatal = fatal


class FakeResampler:
    last_instance: "FakeResampler | None" = None

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int, int]] = []
        FakeResampler.last_instance = self

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        self.calls.append((audio, in_rate, out_rate))
        return b"\x10\x00\x20\x00"


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

        def error(self, *args, **kwargs) -> None:
            pass

    loguru_module = types.ModuleType("loguru")
    loguru_module.logger = FakeLogger()

    audio_utils_module = types.ModuleType("pipecat.audio.utils")
    audio_utils_module.create_default_resampler = FakeResampler

    frames_module = types.ModuleType("pipecat.frames.frames")
    frames_module.Frame = Frame
    frames_module.ErrorFrame = ErrorFrame
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
            "pipecat.audio",
            "pipecat.audio.utils",
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
    sys.modules["pipecat.audio"] = types.ModuleType("pipecat.audio")
    sys.modules["pipecat.audio.utils"] = audio_utils_module
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


class FailingResponse:
    def __init__(
        self,
        *,
        status_error: BaseException | None,
        read_error: BaseException | None,
        chunks: tuple[bytes, ...] = (b"\x01\x00\x02\x00",),
        sample_rate: str = "16000",
    ) -> None:
        self.headers = {"X-Sample-Rate-Hz": sample_rate}
        self.status_error = status_error
        self.read_error = read_error
        self.chunks = chunks

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    async def aiter_bytes(self, chunk_size: int | None = None):
        for chunk in self.chunks:
            yield chunk
        if self.read_error is not None:
            raise self.read_error


class FailingStreamContext:
    def __init__(self, client: "FailingAsyncClient") -> None:
        self.client = client

    async def __aenter__(self) -> FailingResponse:
        if self.client.connect_error is not None:
            raise self.client.connect_error
        return FailingResponse(
            status_error=self.client.status_error,
            read_error=self.client.read_error,
            chunks=self.client.chunks,
            sample_rate=self.client.sample_rate,
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass


class FailingAsyncClient:
    connect_error: BaseException | None = None
    status_error: BaseException | None = None
    read_error: BaseException | None = None
    chunks: tuple[bytes, ...] = (b"\x01\x00\x02\x00",)
    sample_rate = "16000"

    def __init__(self, *, timeout) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FailingAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    def stream(self, method: str, url: str, *, json: dict[str, object]) -> FailingStreamContext:
        return FailingStreamContext(self)


def failing_service(
    module,
    *,
    connect_error: BaseException | None = None,
    status_error: BaseException | None = None,
    read_error: BaseException | None = None,
    chunks: tuple[bytes, ...] = (b"\x01\x00\x02\x00",),
    sample_rate: str = "16000",
):
    client_class = type(
        "ScenarioAsyncClient",
        (FailingAsyncClient,),
        {
            "connect_error": connect_error,
            "status_error": status_error,
            "read_error": read_error,
            "chunks": chunks,
            "sample_rate": sample_rate,
        },
    )
    module.httpx.AsyncClient = client_class
    return module.IrodoriTTSService(base_url="http://tts.local", sample_rate=16000)


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
            self.assertEqual([frame.audio for frame in audio_frames], [b"\x10\x00\x20\x00"])
            self.assertEqual([frame.sample_rate for frame in audio_frames], [16000])
            self.assertEqual([frame.num_channels for frame in audio_frames], [1])
            self.assertEqual(
                FakeResampler.last_instance.calls,
                [(b"\x01\x00\x02\x00\x03\x00", 22050, 16000)],
            )
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

    async def test_run_tts_reports_http_error_frame_without_opening_audio_stream(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(
                module, status_error=RuntimeError("Server error '500 Internal Server Error'")
            )

            # The exception must not escape run_tts: in pipecat it would end the
            # TTS input task, muting every later sentence of the session.
            frames = [frame async for frame in service.run_tts("最初の文です。")]

            self.assertEqual([type(frame) for frame in frames], [ErrorFrame])
            self.assertIn("500", frames[0].error)
            self.assertEqual(service.ttfb_stops, 1)
            self.assertEqual(service.usage_texts, [])
        finally:
            restore_modules(previous)

    async def test_run_tts_reports_connect_error_frame(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(module, connect_error=ConnectionError("connection refused"))

            frames = [frame async for frame in service.run_tts("最初の文です。")]

            self.assertEqual([type(frame) for frame in frames], [ErrorFrame])
            self.assertIn("connection refused", frames[0].error)
            self.assertEqual(service.ttfb_stops, 1)
        finally:
            restore_modules(previous)

    async def test_run_tts_closes_started_audio_stream_after_mid_stream_read_error(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(module, read_error=RuntimeError("peer closed connection"))

            frames = [frame async for frame in service.run_tts("最初の文です。")]

            # Audio2Face was opened by TTSStartedFrame, so it must be closed again.
            self.assertEqual(
                [type(frame) for frame in frames],
                [TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, ErrorFrame, TTSStoppedFrame],
            )
            self.assertIn("peer closed connection", frames[3].error)
            self.assertEqual(service.ttfb_stops, 1)
            self.assertEqual(service.usage_texts, [])
        finally:
            restore_modules(previous)

    async def test_run_tts_empty_body_opens_no_audio_stream(self) -> None:
        # The server answers 200 with no audio for text with nothing speakable.
        # TTSStartedFrame would open an Audio2Face stream that never gets audio.
        for sample_rate in ("16000", "22050"):
            for chunks in ((), (b"", b"")):
                module, previous = load_tokkio_irodori_tts()
                try:
                    service = failing_service(module, chunks=chunks, sample_rate=sample_rate)

                    frames = [frame async for frame in service.run_tts("※。")]

                    with self.subTest(sample_rate=sample_rate, chunks=chunks):
                        self.assertEqual(frames, [])
                        self.assertEqual(service.ttfb_stops, 1)
                        self.assertEqual(service.usage_texts, [])
                        self.assertEqual(FakeResampler.last_instance.calls, [])
                finally:
                    restore_modules(previous)

    async def test_run_tts_read_error_before_first_audio_reports_only_error(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(module, chunks=(), read_error=RuntimeError("peer closed connection"))

            frames = [frame async for frame in service.run_tts("最初の文です。")]

            self.assertEqual([type(frame) for frame in frames], [ErrorFrame])
            self.assertIn("peer closed connection", frames[0].error)
            self.assertEqual(service.ttfb_stops, 1)
            self.assertEqual(service.usage_texts, [])
        finally:
            restore_modules(previous)

    async def test_run_tts_first_audio_chunk_is_streamed_after_started_frame(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(module, chunks=(b"", b"\x01\x00", b"\x02\x00"))

            frames = [frame async for frame in service.run_tts("最初の文です。")]

            self.assertEqual(
                [type(frame) for frame in frames],
                [TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, TTSAudioRawFrame, TTSStoppedFrame],
            )
            self.assertEqual([frame.audio for frame in frames[2:4]], [b"\x01\x00", b"\x02\x00"])
            self.assertEqual(service.ttfb_stops, 1)
            self.assertEqual(service.usage_texts, ["最初の文です。"])
        finally:
            restore_modules(previous)

    async def test_run_tts_resamples_the_first_audio_chunk_too(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(module, chunks=(b"", b"\x01\x00", b"\x02\x00"), sample_rate="22050")

            frames = [frame async for frame in service.run_tts("最初の文です。")]

            self.assertEqual(
                [type(frame) for frame in frames],
                [TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, TTSStoppedFrame],
            )
            self.assertEqual(FakeResampler.last_instance.calls, [(b"\x01\x00\x02\x00", 22050, 16000)])
            self.assertEqual(service.ttfb_stops, 1)
        finally:
            restore_modules(previous)

    async def test_run_tts_lets_barge_in_cancellation_propagate(self) -> None:
        module, previous = load_tokkio_irodori_tts()
        try:
            service = failing_service(module, read_error=asyncio.CancelledError())

            with self.assertRaises(asyncio.CancelledError):
                [frame async for frame in service.run_tts("最初の文です。")]

            frames = service.run_tts("最初の文です。")
            self.assertIsInstance(await frames.__anext__(), TTSStartedFrame)
            # Closing the generator mid-stream (interruption) must not raise.
            await frames.aclose()
        finally:
            restore_modules(previous)


if __name__ == "__main__":
    unittest.main()
