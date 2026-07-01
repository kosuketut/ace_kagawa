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
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=False,
            push_stop_frames=True,
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._configured_sample_rate = sample_rate
        self._timeout_s = timeout_s
        self._response_format = response_format
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

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout_s)) as client:
            response = await client.post(
                f"{self._base_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    "response_format": self._response_format,
                },
            )
            response.raise_for_status()

        await self.stop_ttfb_metrics()
        if response.content:
            yield TTSAudioRawFrame(
                audio=response.content,
                sample_rate=self._sample_rate or self._configured_sample_rate,
                num_channels=1,
            )
        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()
