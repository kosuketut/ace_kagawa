from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

from app.service_status import ServiceStatus

if TYPE_CHECKING:
    from app.settings import Settings


class MockTtsClient:
    async def stream_text(self, text: str) -> AsyncIterator[bytes]:
        silence = b"\x00\x00" * 960
        for _ in range(3):
            yield silence
            await asyncio.sleep(0)

    async def healthcheck(self) -> ServiceStatus:
        return ServiceStatus(name="tts", ok=True, detail="mock TTS enabled")


class RivaTtsClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    async def stream_text(self, text: str) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()

        def run_blocking() -> None:
            try:
                import riva.client
                from riva.client.proto.riva_audio_pb2 import AudioEncoding

                auth = riva.client.Auth(uri=self._settings.tts_server, use_ssl=False)
                service = riva.client.SpeechSynthesisService(auth)
                responses = service.synthesize_online(
                    [text],
                    self._settings.tts_voice,
                    self._settings.tts_language_code,
                    sample_rate_hz=self._settings.tts_sample_rate_hz,
                    encoding=(
                        AudioEncoding.OGGOPUS
                        if self._settings.tts_encoding == "OGGOPUS"
                        else AudioEncoding.LINEAR_PCM
                    ),
                )
                for response in responses:
                    loop.call_soon_threadsafe(queue.put_nowait, response.audio)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(run_blocking))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await worker

    async def healthcheck(self) -> ServiceStatus:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                response = await client.get(f"{self._settings.tts_http_url.rstrip('/')}/v1/health/ready")
                response.raise_for_status()
            return ServiceStatus(
                name="tts",
                ok=True,
                detail="TTS NIM is ready",
                meta={"grpc": self._settings.tts_server, "http": self._settings.tts_http_url},
            )
        except Exception as exc:
            return ServiceStatus(
                name="tts",
                ok=False,
                detail=str(exc),
                meta={"grpc": self._settings.tts_server, "http": self._settings.tts_http_url},
            )


def build_tts_client(settings: "Settings") -> MockTtsClient | RivaTtsClient:
    if settings.mock_tts:
        return MockTtsClient()
    return RivaTtsClient(settings)
