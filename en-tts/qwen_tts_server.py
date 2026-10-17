#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for the English Kagawa Qwen3-TTS voice clone."""

import argparse
import asyncio
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
from threading import Lock
from typing import Iterator, Mapping
import wave

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "data/models/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REFERENCE_AUDIO = ROOT / "voice/kagawa_EN.MP3"
DEFAULT_REFERENCE_TEXT = ROOT / "voice/qwen_tts_voice_clone_english_script.md"
DEFAULT_PREWARM_TEXT = ROOT / "voice/symposium_opening_script.md"


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    return int(raw) if raw else default


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = env.get(key, "").strip()
    return Path(raw).expanduser() if raw else default


def markdown_body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def sentence_texts(path: Path) -> tuple[str, ...]:
    text = " ".join(path.read_text(encoding="utf-8").split())
    return tuple(part for part in re.split(r"(?<=[.!?])\s+", text) if part)


@dataclass(frozen=True)
class QwenTtsSettings:
    host: str = "0.0.0.0"
    port: int = 8025
    model_dir: Path = DEFAULT_MODEL_DIR
    device: str = "cuda:1"
    voice: str = "kagawa_en"
    language: str = "Auto"
    reference_audio: Path = DEFAULT_REFERENCE_AUDIO
    reference_text: Path = DEFAULT_REFERENCE_TEXT
    prewarm_text: Path = DEFAULT_PREWARM_TEXT
    response_sample_rate_hz: int = 24000
    stream_chunk_bytes: int = 3200
    preload_model: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "QwenTtsSettings":
        values = os.environ if env is None else env
        return cls(
            host=values.get("QWEN_TTS_HOST", cls.host).strip() or cls.host,
            port=_env_int(values, "QWEN_TTS_PORT", cls.port),
            model_dir=_env_path(values, "QWEN_TTS_MODEL_DIR", DEFAULT_MODEL_DIR),
            device=values.get("QWEN_TTS_DEVICE", cls.device).strip() or cls.device,
            voice=values.get("QWEN_TTS_VOICE", cls.voice).strip() or cls.voice,
            language=values.get("QWEN_TTS_LANGUAGE", cls.language).strip() or cls.language,
            reference_audio=_env_path(values, "QWEN_TTS_REFERENCE_AUDIO", DEFAULT_REFERENCE_AUDIO),
            reference_text=_env_path(values, "QWEN_TTS_REFERENCE_TEXT", DEFAULT_REFERENCE_TEXT),
            prewarm_text=_env_path(values, "QWEN_TTS_PREWARM_TEXT", DEFAULT_PREWARM_TEXT),
            response_sample_rate_hz=_env_int(
                values, "QWEN_TTS_RESPONSE_SAMPLE_RATE_HZ", cls.response_sample_rate_hz
            ),
            stream_chunk_bytes=_env_int(values, "QWEN_TTS_STREAM_CHUNK_BYTES", cls.stream_chunk_bytes),
            preload_model=values.get("QWEN_TTS_PRELOAD_MODEL", "1").strip().lower()
            not in {"0", "false", "no", "off"},
        )


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio
    target_length = max(1, round(audio.size * target_rate / source_rate))
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0, audio.size - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def audio_to_pcm16(audio: np.ndarray, *, source_rate: int, target_rate: int) -> bytes:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    mono = _resample_linear(mono, source_rate, target_rate)
    clipped = np.clip(mono, -1.0, 1.0)
    return np.where(clipped >= 0, clipped * 32767, clipped * 32768).astype("<i2").tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def iter_pcm_chunks(pcm: bytes, chunk_bytes: int) -> Iterator[bytes]:
    chunk_bytes = max(2, chunk_bytes - chunk_bytes % 2)
    for start in range(0, len(pcm), chunk_bytes):
        yield pcm[start : start + chunk_bytes]


class QwenTtsSynthesizer:
    def __init__(self, settings: QwenTtsSettings) -> None:
        self.settings = settings
        self.model = None
        self.voice_clone_prompt = None
        self.prewarmed_pcm: dict[str, bytes] = {}
        self.load_lock = Lock()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def prewarm_entries(self) -> int:
        return len(self.prewarmed_pcm)

    def load_model(self):
        if self.model is not None:
            return self.model
        with self.load_lock:
            if self.model is not None:
                return self.model

            import soundfile as sf
            import torch
            from qwen_tts import Qwen3TTSModel

            reference_audio, sample_rate = sf.read(self.settings.reference_audio, dtype="float32")
            if reference_audio.ndim == 2:
                reference_audio = reference_audio.mean(axis=1)
            model = Qwen3TTSModel.from_pretrained(
                str(self.settings.model_dir),
                device_map=self.settings.device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=(reference_audio, sample_rate),
                ref_text=markdown_body(self.settings.reference_text),
                x_vector_only_mode=False,
            )
            self.model = model
        return self.model

    def _generate_pcm(self, text: str) -> bytes:
        model = self.load_model()
        wavs, source_rate = model.generate_voice_clone(
            text=text,
            language=self.settings.language,
            voice_clone_prompt=self.voice_clone_prompt,
        )
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio")
        return audio_to_pcm16(
            wavs[0],
            source_rate=int(source_rate),
            target_rate=self.settings.response_sample_rate_hz,
        )

    def prewarm(self) -> None:
        for text in sentence_texts(self.settings.prewarm_text):
            self.prewarmed_pcm[text] = self._generate_pcm(text)

    def synthesize(self, text: str, response_format: str = "pcm") -> tuple[bytes, str, int]:
        text = text.strip()
        if not text:
            raise ValueError("input text must be non-empty")
        if response_format not in {"pcm", "wav"}:
            raise ValueError("response_format must be pcm or wav")

        pcm = self.prewarmed_pcm.get(text) or self._generate_pcm(text)
        if response_format == "wav":
            return pcm16_to_wav(pcm, self.settings.response_sample_rate_hz), "audio/wav", self.settings.response_sample_rate_hz
        return pcm, "audio/L16", self.settings.response_sample_rate_hz


def create_app(settings: QwenTtsSettings | None = None, synthesizer: QwenTtsSynthesizer | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response, StreamingResponse
    from pydantic import BaseModel, Field

    settings = settings or QwenTtsSettings.from_env()
    synthesizer = synthesizer or QwenTtsSynthesizer(settings)

    class SpeechRequest(BaseModel):
        input: str = Field(min_length=1)
        voice: str = settings.voice
        response_format: str = "pcm"
        stream: bool = False

    app = FastAPI(title="Qwen TTS Voice Clone Service")
    app.state.synthesis_lock = None

    @app.on_event("startup")
    async def startup() -> None:
        app.state.synthesis_lock = asyncio.Lock()
        if settings.preload_model:
            await asyncio.to_thread(synthesizer.prewarm)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "qwen-tts-clone",
            "voice": settings.voice,
            "model_loaded": synthesizer.model_loaded,
            "reference_audio_exists": settings.reference_audio.is_file(),
            "reference_text_exists": settings.reference_text.is_file(),
            "prewarm_text_exists": settings.prewarm_text.is_file(),
            "prewarm_entries": synthesizer.prewarm_entries,
            "sample_rate_hz": settings.response_sample_rate_hz,
        }

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest):
        if request.voice != settings.voice:
            raise HTTPException(status_code=400, detail=f"unsupported voice: {request.voice}")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")
        if request.stream and request.response_format != "pcm":
            raise HTTPException(status_code=400, detail="stream=true requires response_format=pcm")

        if app.state.synthesis_lock is None:
            app.state.synthesis_lock = asyncio.Lock()
        async with app.state.synthesis_lock:
            try:
                content, media_type, sample_rate = await asyncio.to_thread(
                    synthesizer.synthesize, request.input, request.response_format
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        headers = {
            "X-Sample-Rate-Hz": str(sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Encoding": "pcm_s16le" if request.response_format == "pcm" else "wav",
        }
        if request.stream:
            return StreamingResponse(
                iter_pcm_chunks(content, settings.stream_chunk_bytes),
                media_type=media_type,
                headers=headers,
            )
        return Response(content=content, media_type=media_type, headers=headers)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Qwen3-TTS English voice clone service")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    import uvicorn

    settings = QwenTtsSettings.from_env()
    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
