from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)

try:
    from pipecat.services.tts_service import TTSService
except ImportError:
    from pipecat.services.ai_services import TTSService


class IrodoriTTSService(TTSService):
    def __init__(
        self,
        *,
        base_url: str,
        voice: str = "kagawa",
        sample_rate: int = 16000,
        timeout_s: float = 180.0,
        response_format: str = "pcm",
        stream_audio: bool = True,
        stream_chunk_bytes: int = 3200,
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=False,
            # run_tts emits an explicit TTSStoppedFrame after the HTTP audio
            # stream is exhausted.  Enabling the base service's idle timeout
            # as well can close Audio2Face while Irodori is still synthesizing
            # a long sentence (the live timeout is two seconds).
            push_stop_frames=False,
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._configured_sample_rate = sample_rate
        self._timeout_s = timeout_s
        self._response_format = response_format
        self._stream_audio = stream_audio
        self._stream_chunk_bytes = stream_chunk_bytes
        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        text = text.lstrip("\n")
        if not text.strip():
            return
        if self._response_format != "pcm":
            raise ValueError("IrodoriTTSService requires response_format='pcm' for TTSAudioRawFrame")

        logger.debug(f"Generating Irodori TTS: [{text}]")
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()
        yield TTSTextFrame(text)

        first_audio_chunk = True
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout_s)) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    "response_format": self._response_format,
                    "stream": self._stream_audio,
                },
            ) as response:
                response.raise_for_status()
                response_sample_rate = int(
                    response.headers.get("X-Sample-Rate-Hz")
                    or self._sample_rate
                    or self._configured_sample_rate
                )
                async for chunk in response.aiter_bytes(chunk_size=self._stream_chunk_bytes):
                    if not chunk:
                        continue
                    if first_audio_chunk:
                        first_audio_chunk = False
                        await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=response_sample_rate,
                        num_channels=1,
                    )

        if first_audio_chunk:
            await self.stop_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()
