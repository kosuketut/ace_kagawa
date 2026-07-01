from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import httpx

from app.adapters.base import AsrClient, AsrEvent, StreamingAsrSession
from app.service_status import ServiceStatus

if TYPE_CHECKING:
    from app.settings import Settings


_END = object()


class MockAsrStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AsrEvent | object] = asyncio.Queue()
        self._audio_bytes = 0
        self._sent_partial = False

    async def push_audio(self, chunk: bytes) -> None:
        self._audio_bytes += len(chunk)
        if self._audio_bytes >= 3200 and not self._sent_partial:
            self._sent_partial = True
            await self._queue.put(AsrEvent(kind="partial", text="モックの途中結果です。"))

    async def end(self) -> None:
        await self._queue.put(AsrEvent(kind="final", text="モックの最終認識結果です。"))
        await self._queue.put(_END)

    async def cancel(self) -> None:
        await self._queue.put(_END)

    async def events(self) -> AsyncIterator[AsrEvent]:
        while True:
            item = await self._queue.get()
            if item is _END:
                break
            yield item


class MockAsrClient:
    def open_stream(self) -> StreamingAsrSession:
        return MockAsrStream()

    async def healthcheck(self) -> ServiceStatus:
        return ServiceStatus(name="asr", ok=True, detail="mock ASR enabled")


class RivaAsrStream:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loop = asyncio.get_running_loop()
        self._audio_queue: queue.Queue[bytes | object] = queue.Queue()
        self._event_queue: asyncio.Queue[AsrEvent | Exception | object] = asyncio.Queue()
        self._worker = asyncio.create_task(asyncio.to_thread(self._run_blocking))

    async def push_audio(self, chunk: bytes) -> None:
        self._audio_queue.put(chunk)

    async def end(self) -> None:
        self._audio_queue.put(_END)
        await self._worker

    async def cancel(self) -> None:
        self._audio_queue.put(_END)
        if not self._worker.done():
            self._worker.cancel()

    async def events(self) -> AsyncIterator[AsrEvent]:
        while True:
            item = await self._event_queue.get()
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _run_blocking(self) -> None:
        try:
            import riva.client

            auth = riva.client.Auth(uri=self._settings.asr_server, use_ssl=False)
            asr_service = riva.client.ASRService(auth)
            config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    language_code=self._settings.asr_language_code,
                    model=self._settings.asr_model,
                    max_alternatives=1,
                    profanity_filter=False,
                    enable_automatic_punctuation=True,
                    verbatim_transcripts=False,
                ),
                interim_results=True,
            )
            responses = asr_service.streaming_response_generator(
                audio_chunks=self._audio_chunks(),
                streaming_config=config,
            )
            for response in responses:
                for result in getattr(response, "results", []):
                    alternatives = getattr(result, "alternatives", [])
                    if not alternatives:
                        continue
                    transcript = alternatives[0].transcript.strip()
                    if not transcript:
                        continue
                    event = AsrEvent(
                        kind="final" if getattr(result, "is_final", False) else "partial",
                        text=transcript,
                        metadata={"stability": getattr(result, "stability", None)},
                    )
                    self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)
        except Exception as exc:
            self._loop.call_soon_threadsafe(self._event_queue.put_nowait, exc)
        finally:
            self._loop.call_soon_threadsafe(self._event_queue.put_nowait, _END)

    def _audio_chunks(self) -> Iterator[bytes]:
        while True:
            chunk = self._audio_queue.get()
            if chunk is _END:
                break
            assert isinstance(chunk, bytes)
            yield chunk


class RivaAsrClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def open_stream(self) -> StreamingAsrSession:
        return RivaAsrStream(self._settings)

    async def healthcheck(self) -> ServiceStatus:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                response = await client.get(f"{self._settings.asr_http_url.rstrip('/')}/v1/health/ready")
                response.raise_for_status()
            return ServiceStatus(
                name="asr",
                ok=True,
                detail="ASR NIM is ready",
                meta={"grpc": self._settings.asr_server, "http": self._settings.asr_http_url},
            )
        except Exception as exc:
            return ServiceStatus(
                name="asr",
                ok=False,
                detail=str(exc),
                meta={"grpc": self._settings.asr_server, "http": self._settings.asr_http_url},
            )


def build_asr_client(settings: "Settings") -> AsrClient:
    if settings.mock_asr:
        return MockAsrClient()
    return RivaAsrClient(settings)
