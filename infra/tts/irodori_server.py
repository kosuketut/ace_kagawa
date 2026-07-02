#!/usr/bin/env python3
"""HTTP wrapper for Irodori-TTS inference.

The module is intentionally cheap to import. FastAPI, torch, and Irodori are
loaded only when the app or runtime is created, so repo-level tests can validate
configuration and audio conversion without loading model dependencies.
"""

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
from threading import Lock
from typing import Mapping
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/data/ACE/irodori")
DEFAULT_HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v3"
DEFAULT_REFERENCE_SOURCE = ROOT / "Irodori-TTS" / "data" / "kagawa_voice.m4a"
DEFAULT_REFERENCE_WAV = DEFAULT_DATA_ROOT / "reference" / "kagawa_voice_ref_48k_mono.wav"
DEFAULT_VOICE = "kagawa"
PRONUNCIATION_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"香川\s*豊"), "香川ゆたか"),
)


def normalize_speech_input(text: str) -> str:
    normalized = text.strip()
    for pattern, replacement in PRONUNCIATION_HINTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    return int(raw) if raw else default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    return float(raw) if raw else default


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = env.get(key, "").strip()
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class IrodoriSettings:
    host: str = "0.0.0.0"
    port: int = 8021
    data_root: Path = DEFAULT_DATA_ROOT
    hf_checkpoint: str = DEFAULT_HF_CHECKPOINT
    checkpoint: Path | None = None
    lora_adapter: Path | None = None
    reference_source: Path = DEFAULT_REFERENCE_SOURCE
    reference_wav: Path = DEFAULT_REFERENCE_WAV
    voice: str = DEFAULT_VOICE
    caption: str | None = None
    model_device: str = "cuda:0"
    model_precision: str = "bf16"
    codec_device: str = "cuda:0"
    codec_precision: str = "fp32"
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    num_steps: int = 40
    duration_scale: float = 1.0
    max_ref_seconds: float = 30.0
    response_sample_rate_hz: int = 16000
    preload_model: bool = False
    short_cache_enabled: bool = False
    short_cache_max_chars: int = 40
    short_cache_max_entries: int = 128

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "IrodoriSettings":
        values = os.environ if env is None else env
        checkpoint_raw = values.get("IRODORI_TTS_CHECKPOINT", "").strip()
        lora_raw = values.get("IRODORI_TTS_LORA_ADAPTER", "").strip()
        caption_raw = values.get("IRODORI_TTS_CAPTION", "").strip()
        return cls(
            host=values.get("IRODORI_TTS_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=_env_int(values, "IRODORI_TTS_PORT", 8021),
            data_root=_env_path(values, "IRODORI_TTS_DATA_ROOT", DEFAULT_DATA_ROOT),
            hf_checkpoint=values.get("IRODORI_TTS_HF_CHECKPOINT", DEFAULT_HF_CHECKPOINT).strip()
            or DEFAULT_HF_CHECKPOINT,
            checkpoint=Path(checkpoint_raw).expanduser() if checkpoint_raw else None,
            lora_adapter=Path(lora_raw).expanduser() if lora_raw else None,
            reference_source=_env_path(values, "IRODORI_TTS_REFERENCE_SOURCE", DEFAULT_REFERENCE_SOURCE),
            reference_wav=_env_path(values, "IRODORI_TTS_REFERENCE_WAV", DEFAULT_REFERENCE_WAV),
            voice=values.get("IRODORI_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE,
            caption=caption_raw or None,
            model_device=values.get("IRODORI_TTS_MODEL_DEVICE", "cuda:0").strip() or "cuda:0",
            model_precision=values.get("IRODORI_TTS_MODEL_PRECISION", "bf16").strip() or "bf16",
            codec_device=values.get("IRODORI_TTS_CODEC_DEVICE", "cuda:0").strip() or "cuda:0",
            codec_precision=values.get("IRODORI_TTS_CODEC_PRECISION", "fp32").strip() or "fp32",
            codec_repo=values.get(
                "IRODORI_TTS_CODEC_REPO",
                "Aratako/Semantic-DACVAE-Japanese-32dim",
            ).strip()
            or "Aratako/Semantic-DACVAE-Japanese-32dim",
            num_steps=_env_int(values, "IRODORI_TTS_NUM_STEPS", 40),
            duration_scale=_env_float(values, "IRODORI_TTS_DURATION_SCALE", 1.0),
            max_ref_seconds=_env_float(values, "IRODORI_TTS_MAX_REF_SECONDS", 30.0),
            response_sample_rate_hz=_env_int(values, "IRODORI_TTS_RESPONSE_SAMPLE_RATE_HZ", 16000),
            preload_model=_env_bool(values, "IRODORI_TTS_PRELOAD_MODEL", False),
            short_cache_enabled=_env_bool(values, "IRODORI_TTS_SHORT_CACHE_ENABLED", False),
            short_cache_max_chars=_env_int(values, "IRODORI_TTS_SHORT_CACHE_MAX_CHARS", 40),
            short_cache_max_entries=_env_int(values, "IRODORI_TTS_SHORT_CACHE_MAX_ENTRIES", 128),
        )

    def ensure_reference_wav(self) -> Path:
        if self.reference_wav.is_file():
            return self.reference_wav
        if not self.reference_source.is_file():
            raise FileNotFoundError(f"reference audio not found: {self.reference_source}")
        self.reference_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(self.reference_source),
                "-ac",
                "1",
                "-ar",
                "48000",
                str(self.reference_wav),
            ],
            check=True,
        )
        return self.reference_wav


def _to_mono_float32(audio: object) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def _resample_linear(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or audio.size == 0:
        return audio
    target_len = max(1, int(round(audio.size * target_sample_rate / source_sample_rate)))
    if audio.size == 1:
        return np.repeat(audio, target_len)
    old_positions = np.linspace(0.0, float(audio.size - 1), num=audio.size, dtype=np.float64)
    new_positions = np.linspace(0.0, float(audio.size - 1), num=target_len, dtype=np.float64)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def audio_to_pcm16(audio: object, *, source_sample_rate: int, target_sample_rate: int) -> bytes:
    mono = _to_mono_float32(audio)
    resampled = _resample_linear(mono, source_sample_rate, target_sample_rate)
    clipped = np.clip(resampled, -1.0, 1.0)
    int_audio = np.where(clipped >= 0, clipped * 32767.0, clipped * 32768.0).astype("<i2")
    return int_audio.tobytes()


def pcm16_to_wav_bytes(pcm: bytes, *, sample_rate_hz: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class IrodoriSynthesizer:
    def __init__(self, settings: IrodoriSettings) -> None:
        self._settings = settings
        self._runtime = None
        self._short_cache: OrderedDict[tuple[str, str], tuple[bytes, str, int]] = OrderedDict()
        self._short_cache_lock = Lock()

    def load_runtime(self):
        if self._runtime is not None:
            return self._runtime

        from huggingface_hub import hf_hub_download
        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

        if self._settings.checkpoint is not None:
            checkpoint = str(self._settings.checkpoint)
        else:
            checkpoint = hf_hub_download(
                repo_id=self._settings.hf_checkpoint,
                filename="model.safetensors",
            )

        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint,
                model_device=self._settings.model_device,
                codec_repo=self._settings.codec_repo,
                model_precision=self._settings.model_precision,
                codec_device=self._settings.codec_device,
                codec_precision=self._settings.codec_precision,
            )
        )
        return self._runtime

    def _synthesize_normalized(self, normalized_text: str, *, response_format: str) -> tuple[bytes, str, int]:
        from irodori_tts.inference_runtime import SamplingRequest

        ref_wav = self._settings.ensure_reference_wav()
        runtime = self.load_runtime()
        result = runtime.synthesize(
            SamplingRequest(
                text=normalized_text,
                caption=self._settings.caption,
                ref_wav=str(ref_wav),
                no_ref=False,
                num_steps=self._settings.num_steps,
                duration_scale=self._settings.duration_scale,
                max_ref_seconds=self._settings.max_ref_seconds,
                lora_adapter=None
                if self._settings.lora_adapter is None
                else str(self._settings.lora_adapter),
            ),
            log_fn=None,
        )
        pcm = audio_to_pcm16(
            result.audio,
            source_sample_rate=int(result.sample_rate),
            target_sample_rate=self._settings.response_sample_rate_hz,
        )
        if response_format == "wav":
            return (
                pcm16_to_wav_bytes(pcm, sample_rate_hz=self._settings.response_sample_rate_hz),
                "audio/wav",
                self._settings.response_sample_rate_hz,
            )
        return pcm, "audio/L16", self._settings.response_sample_rate_hz

    def _is_short_cacheable(self, normalized_text: str) -> bool:
        return (
            self._settings.short_cache_enabled
            and self._settings.short_cache_max_entries > 0
            and len(normalized_text) <= self._settings.short_cache_max_chars
        )

    def _get_short_cache(self, key: tuple[str, str]) -> tuple[bytes, str, int] | None:
        with self._short_cache_lock:
            cached = self._short_cache.get(key)
            if cached is None:
                return None
            self._short_cache.move_to_end(key)
            return cached

    def _put_short_cache(self, key: tuple[str, str], value: tuple[bytes, str, int]) -> None:
        with self._short_cache_lock:
            self._short_cache[key] = value
            self._short_cache.move_to_end(key)
            while len(self._short_cache) > self._settings.short_cache_max_entries:
                self._short_cache.popitem(last=False)

    def short_cache_size(self) -> int:
        with self._short_cache_lock:
            return len(self._short_cache)

    def synthesize(self, text: str, *, response_format: str = "pcm") -> tuple[bytes, str, int]:
        normalized_text = normalize_speech_input(text)
        if not normalized_text:
            raise ValueError("input text must be non-empty")
        if response_format not in {"pcm", "wav"}:
            raise ValueError("response_format must be one of: pcm, wav")

        cache_key = (response_format, normalized_text)
        if self._is_short_cacheable(normalized_text):
            cached = self._get_short_cache(cache_key)
            if cached is not None:
                return cached

        result = self._synthesize_normalized(normalized_text, response_format=response_format)
        if self._is_short_cacheable(normalized_text):
            self._put_short_cache(cache_key, result)
        return result


def create_app(settings: IrodoriSettings | None = None, synthesizer: IrodoriSynthesizer | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    resolved_settings = settings or IrodoriSettings.from_env()
    resolved_synthesizer = synthesizer or IrodoriSynthesizer(resolved_settings)

    class SpeechRequest(BaseModel):
        input: str = Field(min_length=1)
        voice: str = resolved_settings.voice
        response_format: str = "pcm"

    app = FastAPI(title="Irodori TTS Service")
    app.state.settings = resolved_settings
    app.state.synthesizer = resolved_synthesizer
    app.state.synthesis_lock = None

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.synthesis_lock = asyncio.Lock()
        if resolved_settings.preload_model:
            await asyncio.to_thread(resolved_synthesizer.load_runtime)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "voice": resolved_settings.voice,
            "model_loaded": resolved_synthesizer._runtime is not None,
            "reference_source_exists": resolved_settings.reference_source.is_file(),
            "reference_wav_exists": resolved_settings.reference_wav.is_file(),
            "sample_rate_hz": resolved_settings.response_sample_rate_hz,
            "short_cache_enabled": resolved_settings.short_cache_enabled,
            "short_cache_entries": resolved_synthesizer.short_cache_size()
            if hasattr(resolved_synthesizer, "short_cache_size")
            else 0,
            "short_cache_max_chars": resolved_settings.short_cache_max_chars,
        }

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if request.voice != resolved_settings.voice:
            raise HTTPException(status_code=400, detail=f"unsupported voice: {request.voice}")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")
        if app.state.synthesis_lock is None:
            app.state.synthesis_lock = asyncio.Lock()
        async with app.state.synthesis_lock:
            try:
                content, media_type, sample_rate = await asyncio.to_thread(
                    app.state.synthesizer.synthesize,
                    request.input,
                    response_format=request.response_format,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "X-Sample-Rate-Hz": str(sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Encoding": "pcm_s16le" if request.response_format == "pcm" else "wav",
            },
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Irodori-TTS HTTP service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    import uvicorn

    settings = IrodoriSettings.from_env()
    host = args.host or settings.host
    port = args.port or settings.port
    uvicorn.run(create_app(settings), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
