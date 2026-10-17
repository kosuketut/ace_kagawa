from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator

import httpx
from loguru import logger
from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    ErrorFrame,
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


async def _prepend_chunk(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    yield first
    async for chunk in rest:
        yield chunk


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
        self._resampler = create_default_resampler()
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

        started = False
        first_audio_chunk = True
        # An exception escaping run_tts ends pipecat's TTS input task, which
        # mutes every later sentence and blocks EndFrame.  Report it as an
        # ErrorFrame instead.  CancelledError/GeneratorExit (barge-in) are not
        # Exceptions and still propagate; never yield from a finally here.
        try:
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
                    target_sample_rate = self._configured_sample_rate
                    audio_chunks = response.aiter_bytes(chunk_size=self._stream_chunk_bytes)
                    first_chunk = b""
                    async for chunk in audio_chunks:
                        if chunk:
                            first_chunk = chunk
                            break
                    if not first_chunk:
                        # 200 with no audio: the text had nothing speakable.
                        # TTSStartedFrame would open an Audio2Face stream that
                        # never receives audio.
                        logger.debug(f"Irodori TTS returned no audio for [{text}]")
                        await self.stop_ttfb_metrics()
                        return
                    # Open the Audio2Face stream only once audio is arriving.
                    yield TTSStartedFrame()
                    started = True
                    yield TTSTextFrame(text)
                    response_audio = bytearray()
                    async for chunk in _prepend_chunk(first_chunk, audio_chunks):
                        if not chunk:
                            continue
                        if response_sample_rate != target_sample_rate:
                            response_audio.extend(chunk)
                            continue
                        if first_audio_chunk:
                            first_audio_chunk = False
                            await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=target_sample_rate,
                            num_channels=1,
                        )

                    if response_audio:
                        logger.debug(
                            f"Resampling TTS audio from {response_sample_rate} Hz "
                            f"to {target_sample_rate} Hz"
                        )
                        audio = await self._resampler.resample(
                            bytes(response_audio), response_sample_rate, target_sample_rate
                        )
                        chunk_bytes = max(2, self._stream_chunk_bytes - self._stream_chunk_bytes % 2)
                        for offset in range(0, len(audio), chunk_bytes):
                            if first_audio_chunk:
                                first_audio_chunk = False
                                await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(
                                audio=audio[offset : offset + chunk_bytes],
                                sample_rate=target_sample_rate,
                                num_channels=1,
                            )
        except Exception as exc:
            logger.error(f"{self} TTS request failed for [{text}]: {exc}")
            if first_audio_chunk:
                await self.stop_ttfb_metrics()
            yield ErrorFrame(error=f"{self} TTS request failed: {exc}")
            if started:
                yield TTSStoppedFrame()
            return

        if first_audio_chunk:
            await self.stop_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()
