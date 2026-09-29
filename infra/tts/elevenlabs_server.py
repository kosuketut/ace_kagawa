#!/usr/bin/env python3
"""ElevenLabs gateway with the same /v1/audio/speech contract as irodori_server.py.

Tokkio's IrodoriTTSService client can point at this service unchanged. The
gateway applies the same Japanese reading normalization as the Irodori service
and calls ElevenLabs' HTTP streaming endpoint, because the v3/v4 models are not
available on the text-to-speech WebSocket that pipecat's ElevenLabs service uses.
"""

import argparse
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from irodori_server import (  # noqa: E402
    _env_bool,
    _env_float,
    _env_int,
    _env_optional_float,
    normalize_speech_input,
    pcm16_to_wav_bytes,
)


ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
DEFAULT_MODEL_ID = "eleven_v4_turbo"
DEFAULT_VOICE = "kagawa"


def _env_first(env: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = env.get(key, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class ElevenLabsSettings:
    host: str = "0.0.0.0"
    port: int = 8022
    voice: str = DEFAULT_VOICE
    voice_id: str = ""
    api_key: str = field(default="", repr=False)
    model_id: str = DEFAULT_MODEL_ID
    language_code: str = "ja"
    sample_rate_hz: int = 16000
    # None leaves the setting to the voice's defaults on ElevenLabs.
    stability: float | None = None
    similarity_boost: float | None = None
    speed: float | None = None
    normalize_text: bool = True
    timeout_s: float = 60.0
    stream_chunk_bytes: int = 3200
    api_base: str = ELEVENLABS_API_BASE

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ElevenLabsSettings":
        values = os.environ if env is None else env
        return cls(
            host=values.get("ELEVENLABS_TTS_HOST", "").strip() or "0.0.0.0",
            port=_env_int(values, "ELEVENLABS_TTS_PORT", 8022),
            voice=values.get("ELEVENLABS_TTS_VOICE", "").strip() or DEFAULT_VOICE,
            voice_id=_env_first(values, "ELEVENLABS_VOICE_ID", "TOKKIO_ELEVENLABS_VOICE_ID"),
            api_key=_env_first(values, "ELEVENLABS_API_KEY", "TOKKIO_ELEVENLABS_API_KEY"),
            model_id=_env_first(values, "ELEVENLABS_TTS_MODEL_ID", "TOKKIO_ELEVENLABS_MODEL_ID")
            or DEFAULT_MODEL_ID,
            language_code=values.get("ELEVENLABS_TTS_LANGUAGE_CODE", "").strip() or "ja",
            sample_rate_hz=_env_int(values, "ELEVENLABS_TTS_SAMPLE_RATE_HZ", 16000),
            stability=_env_optional_float(values, "ELEVENLABS_TTS_STABILITY"),
            similarity_boost=_env_optional_float(values, "ELEVENLABS_TTS_SIMILARITY_BOOST"),
            speed=_env_optional_float(values, "ELEVENLABS_TTS_SPEED"),
            normalize_text=_env_bool(values, "ELEVENLABS_TTS_NORMALIZE_TEXT", True),
            timeout_s=_env_float(values, "ELEVENLABS_TTS_TIMEOUT_S", 60.0),
            stream_chunk_bytes=_env_int(values, "ELEVENLABS_TTS_STREAM_CHUNK_BYTES", 3200),
            api_base=values.get("ELEVENLABS_API_BASE", "").strip().rstrip("/") or ELEVENLABS_API_BASE,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.voice_id)


def build_speech_request(settings: ElevenLabsSettings, text: str) -> tuple[str, dict[str, object]]:
    """Return the ElevenLabs streaming URL and JSON body for one utterance."""
    spoken = normalize_speech_input(text) if settings.normalize_text else text.strip()
    payload: dict[str, object] = {
        "text": spoken,
        "model_id": settings.model_id,
        "language_code": settings.language_code,
    }
    voice_settings = {
        key: value
        for key, value in (
            ("stability", settings.stability),
            ("similarity_boost", settings.similarity_boost),
            ("speed", settings.speed),
        )
        if value is not None
    }
    if voice_settings:
        payload["voice_settings"] = voice_settings
    url = (
        f"{settings.api_base}/v1/text-to-speech/{settings.voice_id}/stream"
        f"?output_format=pcm_{settings.sample_rate_hz}"
    )
    return url, payload


async def iter_aligned_pcm(chunks: AsyncIterable[bytes], *, chunk_bytes: int) -> AsyncIterator[bytes]:
    """Re-chunk an HTTP byte stream so no 16-bit sample is split across chunks."""
    size = max(2, chunk_bytes - chunk_bytes % 2)
    buffer = bytearray()
    async for chunk in chunks:
        buffer.extend(chunk)
        while len(buffer) >= size:
            yield bytes(buffer[:size])
            del buffer[:size]
    tail = len(buffer) - len(buffer) % 2
    if tail:
        yield bytes(buffer[:tail])


def _upstream_error_message(body: bytes) -> str:
    try:
        detail = json.loads(body).get("detail", {})
    except (ValueError, AttributeError):
        return body[:200].decode(errors="replace")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)[:300]
    return str(detail)[:300]


def create_app(settings: ElevenLabsSettings | None = None, transport=None):
    import httpx
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field

    resolved = settings or ElevenLabsSettings.from_env()

    class SpeechRequest(BaseModel):
        input: str = Field(min_length=1)
        voice: str = resolved.voice
        response_format: str = "pcm"
        stream: bool = False

    app = FastAPI(title="ElevenLabs TTS Gateway")
    app.state.settings = resolved
    app.state.http_client = None

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(resolved.timeout_s),
            transport=transport,
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if app.state.http_client is not None:
            await app.state.http_client.aclose()

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            status_code=200 if resolved.configured else 503,
            content={
                "status": "ok" if resolved.configured else "unconfigured",
                "backend": "elevenlabs",
                "voice": resolved.voice,
                "model_id": resolved.model_id,
                "language_code": resolved.language_code,
                "sample_rate_hz": resolved.sample_rate_hz,
                "voice_id_configured": bool(resolved.voice_id),
                "api_key_configured": bool(resolved.api_key),
                "normalize_text": resolved.normalize_text,
            },
        )

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if request.voice != resolved.voice:
            raise HTTPException(status_code=400, detail=f"unsupported voice: {request.voice}")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")
        if request.stream and request.response_format != "pcm":
            raise HTTPException(status_code=400, detail="stream=true requires response_format=pcm")
        if not resolved.configured:
            raise HTTPException(status_code=503, detail="ElevenLabs API key or voice id is not configured")

        url, payload = build_speech_request(resolved, request.input)
        if not str(payload["text"]).strip():
            raise HTTPException(status_code=400, detail="input text must be non-empty")

        client = app.state.http_client
        try:
            upstream = await client.send(
                client.build_request("POST", url, json=payload, headers={"xi-api-key": resolved.api_key}),
                stream=True,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ElevenLabs request failed: {exc}") from exc
        if upstream.status_code != 200:
            body = await upstream.aread()
            await upstream.aclose()
            raise HTTPException(
                status_code=502,
                detail=f"ElevenLabs HTTP {upstream.status_code}: {_upstream_error_message(body)}",
            )

        headers = {
            "X-Sample-Rate-Hz": str(resolved.sample_rate_hz),
            "X-Audio-Channels": "1",
            "X-Audio-Encoding": "pcm_s16le" if request.response_format == "pcm" else "wav",
        }
        pcm_chunks = iter_aligned_pcm(upstream.aiter_bytes(), chunk_bytes=resolved.stream_chunk_bytes)
        if request.stream:

            async def body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in pcm_chunks:
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(body(), media_type="audio/L16", headers=headers)

        try:
            pcm = b"".join([chunk async for chunk in pcm_chunks])
        finally:
            await upstream.aclose()
        if request.response_format == "wav":
            return Response(
                pcm16_to_wav_bytes(pcm, sample_rate_hz=resolved.sample_rate_hz),
                media_type="audio/wav",
                headers=headers,
            )
        return Response(pcm, media_type="audio/L16", headers=headers)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ElevenLabs TTS gateway")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    import uvicorn

    settings = ElevenLabsSettings.from_env()
    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
