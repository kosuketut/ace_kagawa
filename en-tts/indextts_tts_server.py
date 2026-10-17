#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP wrapper for the Kagawa IndexTTS-2.5 voice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from threading import RLock
from typing import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import wave

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = Path("/data/ACE/en-tts/repos/IndexTTS")
DEFAULT_MODEL_DIR = Path("/data/ACE/en-tts/models/IndexTTS-2.5")
DEFAULT_REFERENCE_AUDIO = Path("/data/ACE/en-tts/reference/kagawa_en_sentence1.wav")
DEFAULT_PREWARM_TEXT = ROOT / "voice/symposium_opening_script.md"
DEFAULT_VOICE = "kagawa_en"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_SAMPLE_RATE_HZ = 22050
DEFAULT_TRAILING_SILENCE_MS = 500
PRONUNCIATION_SUBSTITUTIONS = {
    "Hachioji": "HH AA1 . CH IY0 . OW0 . JH IY0",
    "Hachiōji": "HH AA1 . CH IY0 . OW0 . JH IY0",
    "Seiran": "S EY1 . R AA0 N",
    "supercomputer": "S UW2 . P ER0 . K AH0 M . P Y UW1 . T ER0",
    "supercomputers": "S UW2 . P ER0 . K AH0 M . P Y UW1 . T ER0 Z",
}
# Terms IndexTTS's English normalizer misreads ("24/7" -> "twenty four sevenths").
NORMALIZATION_SUBSTITUTIONS = (
    (re.compile(r"(?<![\w/])24/7(?![\w/])"), "twenty-four seven"),
    (re.compile(r"\bProf\.(?=\s)"), "Professor"),
)
# The normalizer's contraction rule reads every "...it's"/"...he's" as "is" ("It's been" ->
# "It is been", "MIT's" -> "MIT is"); a typographic apostrophe skips it and is mapped back to "'".
ASCII_APOSTROPHE_S = re.compile(r"(?<=\w)'(?=[sS]\b)")
# Units WeText reads as a measure ("1,000 km" -> "one thousand kilometers"); see tn/english/data/measure.
MEASURE_UNITS = (
    "kbps kcal mbps amu atm bar cwt ghz gpa gwh kgf khz kpa kwh lbs mhz min mol mpa mph rad rpm"
    " au bq cc cd cm cu da db dm ds eb ev ft gb gl gw gy ha hl hp hs hz kb kg kl km kn kt kv kw lb lm"
    " ma mb mc mf mg mi ml mm ms mv mw ng nm ns oz pa pb pg ps sq sv tb tj tl yb yd zb μg μm μs ω f g h m s v"
)
_BARE_NUMBER_START = r"(?<![^\s(\[\"“‘])"
_MEASURE_END = r"\s?(?:[°º℃℉%]|(?:" + "|".join(MEASURE_UNITS.split()) + r")(?![A-Za-z]))"
# WeText drops the "one" of a bare 1,000-1,999 ("1,500" -> "thousand five hundred"). Money, measures,
# decimals, ordinals, compounds and larger numbers are read correctly and keep their digits.
BARE_ONE_THOUSAND = re.compile(
    _BARE_NUMBER_START + r"1,(\d{3})(?!\d|,\d|\.\d|[-A-Za-z]|" + _MEASURE_END + ")", re.IGNORECASE
)
# It also reads a bare 1000 as the year "ten hundred". "1000-year-old" is misread too (unlike
# "1,000-year-old" -> "thousand-year-old"), so only a range hyphen is left alone here.
BARE_1000 = re.compile(
    _BARE_NUMBER_START + r"1000(?!\d|[,.]\d|-\d|[A-Za-z]|" + _MEASURE_END + ")", re.IGNORECASE
)
PRONUNCIATION_ANNOTATION = re.compile(r"<([^|>\n]+)\|[^>\n]+>")
CJK_TEXT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    return int(raw) if raw else default


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = env.get(key, "").strip()
    return Path(raw).expanduser() if raw else default


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    return default if raw is None else raw.strip().lower() not in {"0", "false", "no", "off"}


def markdown_body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def sentence_texts(path: Path) -> tuple[str, ...]:
    text = " ".join(markdown_body(path).split())
    return tuple(part for part in re.split(r"(?<=[.!?])\s+", text) if part)


def apply_pronunciation_substitutions(text: str) -> str:
    for word, pronunciation in PRONUNCIATION_SUBSTITUTIONS.items():
        pattern = rf"(?<![A-Za-z0-9_<|]){re.escape(word)}(?![A-Za-z0-9_])"
        text = re.sub(
            pattern,
            lambda match: f"<{match.group(0)}|{pronunciation}>",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _spell_bare_one_thousand(match: re.Match[str]) -> str:
    rest = int(match.group(1))
    return f"one thousand {rest}" if rest else "one thousand"


def _normalizer_input(text: str) -> str:
    """Work around TextNormalizer misreadings; only for text that goes through normalize()."""
    text = BARE_ONE_THOUSAND.sub(_spell_bare_one_thousand, text)
    text = BARE_1000.sub("one thousand", text)
    # The normalizer leaves the "+" in "one thousand+" unread; spell it out.
    text = re.sub(r"\b(one thousand)\+", r"\1 plus", text)
    return ASCII_APOSTROPHE_S.sub("’", text)


def normalize_english_text(text: str, normalizer: object | None) -> str:
    """Expand numbers, times and abbreviations before IndexTTS maps ':' to ','."""
    if normalizer is None:
        return text
    for pattern, replacement in NORMALIZATION_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    # TextNormalizer.normalize() switches to its Chinese normalizer for text without
    # Latin letters ("2026." -> Chinese numerals); send such text unnormalized.
    plain = PRONUNCIATION_ANNOTATION.sub(r"\1", text)
    if not re.search(r"[A-Za-z]", plain) or CJK_TEXT.search(plain) or normalizer.use_chinese(plain):
        return text
    normalized = normalizer.normalize(_normalizer_input(text)).strip()
    if not normalized:
        return text
    if text[-1] in ".!?" and normalized[-1] not in ".!?…":
        normalized += text[-1]
    return normalized


def audio_to_pcm16(audio: object) -> bytes:
    """Convert IndexTTS's mono PCM-scale array to little-endian PCM16 bytes."""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    source = audio
    if hasattr(source, "detach"):
        source = source.detach()
    if hasattr(source, "cpu"):
        source = source.cpu()
    if hasattr(source, "numpy"):
        source = source.numpy()

    samples = np.asarray(source).reshape(-1)
    if samples.size == 0:
        return b""
    values = np.nan_to_num(samples.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if samples.dtype.kind == "f" and np.abs(values).max() <= 1.0:
        values *= 32767.0
    return np.clip(np.rint(values), -32768, 32767).astype("<i2").tobytes()


def resample_pcm16(pcm: bytes, source_rate_hz: int, target_rate_hz: int) -> bytes:
    if source_rate_hz == target_rate_hz or not pcm:
        return pcm
    # Same band-limited resampler (soxr VHQ) that the pipecat client would otherwise use.
    import soxr

    samples = np.frombuffer(pcm, dtype="<i2")
    return soxr.resample(samples, source_rate_hz, target_rate_hz, quality="VHQ").astype("<i2").tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate_hz: int) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)
    return output.getvalue()


def iter_pcm_chunks(pcm: bytes, chunk_bytes: int) -> Iterator[bytes]:
    chunk_bytes = max(2, chunk_bytes - chunk_bytes % 2)
    for start in range(0, len(pcm), chunk_bytes):
        yield pcm[start : start + chunk_bytes]


class IndexTtsSettings:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = 8023,
        repo_dir: Path = DEFAULT_REPO_DIR,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: str = "cuda:0",
        voice: str = DEFAULT_VOICE,
        reference_audio: Path = DEFAULT_REFERENCE_AUDIO,
        prewarm_text: Path = DEFAULT_PREWARM_TEXT,
        response_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        stream_chunk_bytes: int = 3200,
        trailing_silence_ms: int = DEFAULT_TRAILING_SILENCE_MS,
        preload_model: bool = True,
        cuda_visible_devices: str = "1",
    ) -> None:
        if voice != DEFAULT_VOICE:
            raise ValueError(f"voice must be {DEFAULT_VOICE}")
        self.host = host
        self.port = port
        self.repo_dir = repo_dir
        self.model_dir = model_dir
        self.device = device
        self.voice = voice
        self.reference_audio = reference_audio
        self.prewarm_text = prewarm_text
        self.response_sample_rate_hz = response_sample_rate_hz
        self.stream_chunk_bytes = stream_chunk_bytes
        self.trailing_silence_ms = max(0, trailing_silence_ms)
        self.preload_model = preload_model
        self.cuda_visible_devices = cuda_visible_devices

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "IndexTtsSettings":
        values = os.environ if env is None else env
        return cls(
            host=values.get("INDEXTTS_TTS_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
            port=_env_int(values, "INDEXTTS_TTS_PORT", 8023),
            repo_dir=_env_path(values, "INDEXTTS_TTS_REPO_DIR", DEFAULT_REPO_DIR),
            model_dir=_env_path(values, "INDEXTTS_TTS_MODEL_DIR", DEFAULT_MODEL_DIR),
            device=values.get("INDEXTTS_TTS_DEVICE", "cuda:0").strip() or "cuda:0",
            voice=values.get("INDEXTTS_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE,
            reference_audio=_env_path(
                values, "INDEXTTS_TTS_REFERENCE_AUDIO", DEFAULT_REFERENCE_AUDIO
            ),
            prewarm_text=_env_path(values, "INDEXTTS_TTS_PREWARM_TEXT", DEFAULT_PREWARM_TEXT),
            response_sample_rate_hz=_env_int(
                values,
                "INDEXTTS_TTS_RESPONSE_SAMPLE_RATE_HZ",
                DEFAULT_SAMPLE_RATE_HZ,
            ),
            stream_chunk_bytes=_env_int(values, "INDEXTTS_TTS_STREAM_CHUNK_BYTES", 3200),
            trailing_silence_ms=_env_int(
                values,
                "INDEXTTS_TTS_TRAILING_SILENCE_MS",
                DEFAULT_TRAILING_SILENCE_MS,
            ),
            preload_model=_env_bool(values, "INDEXTTS_TTS_PRELOAD_MODEL", True),
            cuda_visible_devices=values.get(
                "INDEXTTS_TTS_CUDA_VISIBLE_DEVICES",
                values.get("CUDA_VISIBLE_DEVICES", "1"),
            ).strip()
            or "1",
        )


class IndexTtsSynthesizer:
    def __init__(self, settings: IndexTtsSettings) -> None:
        self.settings = settings
        self.model = None
        self.prewarmed_pcm: dict[str, bytes] = {}
        self.prewarmed_sample_rates: dict[str, int] = {}
        # ponytail: one global inference lock; use per-GPU workers only if throughput requires it.
        self.synthesis_lock = RLock()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def prewarm_entries(self) -> int:
        return len(self.prewarmed_pcm)

    def load_model(self):
        with self.synthesis_lock:
            if self.model is not None:
                return self.model
            if not self.settings.repo_dir.is_dir():
                raise RuntimeError(f"IndexTTS repo not found: {self.settings.repo_dir}")
            if not self.settings.reference_audio.is_file():
                raise RuntimeError(f"reference audio not found: {self.settings.reference_audio}")

            os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.cuda_visible_devices
            os.environ["HF_HUB_CACHE"] = str(self.settings.model_dir / "hf_cache")
            repo = str(self.settings.repo_dir)
            if repo not in sys.path:
                sys.path.insert(0, repo)
            # Keep the GPU/model import lazy so unit tests can run on CPU-only hosts.
            from indextts.infer_v2_5 import IndexTTS2

            os.environ["HF_HUB_CACHE"] = str(self.settings.model_dir / "hf_cache")
            self.model = IndexTTS2(
                cfg_path=str(self.settings.model_dir / "config.yaml"),
                model_dir=str(self.settings.model_dir),
                use_bf16=True,
                device=self.settings.device,
                use_cuda_kernel=False,
                use_deepspeed=False,
                use_accel=False,
                use_torch_compile=False,
                use_qwen_emo=False,
            )
            return self.model

    def _generate_audio(self, text: str) -> tuple[bytes, int]:
        model = self.load_model()
        result = model.infer(
            spk_audio_prompt=str(self.settings.reference_audio),
            text=normalize_english_text(
                apply_pronunciation_substitutions(text), getattr(model, "text_process", None)
            ),
            output_path=None,
            lang="EN",
            use_random=False,
            interval_silence=350,
            duration_factor=1.0,
            text_normalization=False,
            verbose=False,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("IndexTTS returned no audio")
        sample_rate_hz, audio = result
        pcm = audio_to_pcm16(audio)
        if not pcm:
            raise RuntimeError("IndexTTS returned empty audio")
        output_rate_hz = self.settings.response_sample_rate_hz
        return resample_pcm16(pcm, int(sample_rate_hz), output_rate_hz), output_rate_hz

    def prewarm(self) -> None:
        with self.synthesis_lock:
            for text in sentence_texts(self.settings.prewarm_text):
                pcm, sample_rate_hz = self._generate_audio(text)
                self.prewarmed_sample_rates[text] = sample_rate_hz
                self.prewarmed_pcm[text] = pcm

    def synthesize(self, text: str, response_format: str = "pcm") -> tuple[bytes, str, int]:
        text = text.strip()
        if not text:
            raise ValueError("input text must be non-empty")
        if response_format not in {"pcm", "wav"}:
            raise ValueError("response_format must be pcm or wav")

        # Cached sentences must not wait for a running (possibly abandoned) synthesis.
        if text in self.prewarmed_pcm:
            pcm = self.prewarmed_pcm[text]
            sample_rate_hz = self.prewarmed_sample_rates[text]
        else:
            with self.synthesis_lock:
                pcm, sample_rate_hz = self._generate_audio(text)
        pcm += bytes(sample_rate_hz * self.settings.trailing_silence_ms // 1000 * 2)
        if response_format == "wav":
            return (
                pcm16_to_wav(pcm, sample_rate_hz),
                "audio/wav",
                sample_rate_hz,
            )
        return pcm, "audio/L16", sample_rate_hz


def _health_payload(settings: IndexTtsSettings, synthesizer: IndexTtsSynthesizer) -> dict[str, object]:
    return {
        "status": "ok",
        "service": "indextts-tts",
        "voice": settings.voice,
        "model_loaded": synthesizer.model_loaded,
        "reference_audio_exists": settings.reference_audio.is_file(),
        "prewarm_text_exists": settings.prewarm_text.is_file(),
        "prewarm_entries": synthesizer.prewarm_entries,
        "sample_rate_hz": settings.response_sample_rate_hz,
        "trailing_silence_ms": settings.trailing_silence_ms,
    }


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class _TtsRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def settings(self) -> IndexTtsSettings:
        return self.server.settings  # type: ignore[attr-defined]

    @property
    def synthesizer(self) -> IndexTtsSynthesizer:
        return self.server.synthesizer  # type: ignore[attr-defined]

    def _send_bytes(
        self,
        status: int,
        content: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(content)

    def _send_error_json(self, status: int, detail: str) -> None:
        self._send_bytes(status, _json_bytes({"detail": detail}), "application/json")

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/healthz":
            self._send_bytes(200, _json_bytes(_health_payload(self.settings, self.synthesizer)), "application/json")
            return
        self._send_error_json(404, "not found")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/audio/speech":
            self._send_error_json(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > 1024 * 1024:
                raise ValueError("request body must have a Content-Length <= 1048576")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            text = request.get("input")
            if not isinstance(text, str) or not text:
                raise ValueError("input must be a non-empty string")
            voice = request.get("voice", self.settings.voice)
            if voice != self.settings.voice:
                raise ValueError(f"unsupported voice: {voice}")
            response_format = request.get("response_format", "pcm")
            if response_format not in {"pcm", "wav"}:
                raise ValueError("response_format must be pcm or wav")
            stream = request.get("stream", False)
            if not isinstance(stream, bool):
                raise ValueError("stream must be a boolean")
            if stream and response_format != "pcm":
                raise ValueError("stream=true requires response_format=pcm")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_error_json(400, str(exc))
            return

        try:
            content, media_type, sample_rate_hz = self.synthesizer.synthesize(
                text, response_format=response_format
            )
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        except Exception as exc:
            self._send_error_json(500, str(exc))
            return

        headers = {
            "X-Sample-Rate-Hz": str(sample_rate_hz),
            "X-Audio-Channels": "1",
            "X-Audio-Encoding": "pcm_s16le" if response_format == "pcm" else "wav",
        }
        try:
            self.send_response(200)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(content)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if stream:
                for chunk in iter_pcm_chunks(content, self.settings.stream_chunk_bytes):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError) as exc:
            # Tokkio closes the stream on Stop/barge-in; this is not a server error.
            self.close_connection = True
            self.log_message("client disconnected before the audio was sent (%s)", type(exc).__name__)


class IndexTtsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        settings: IndexTtsSettings,
        synthesizer: IndexTtsSynthesizer,
    ) -> None:
        self.settings = settings
        self.synthesizer = synthesizer
        super().__init__(address, _TtsRequestHandler)


def create_server(
    settings: IndexTtsSettings | None = None,
    synthesizer: IndexTtsSynthesizer | None = None,
) -> IndexTtsHTTPServer:
    settings = settings or IndexTtsSettings.from_env()
    synthesizer = synthesizer or IndexTtsSynthesizer(settings)
    return IndexTtsHTTPServer((settings.host, settings.port), settings, synthesizer)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the IndexTTS-2.5 English voice service")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    settings = IndexTtsSettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
    synthesizer = IndexTtsSynthesizer(settings)
    if settings.preload_model:
        synthesizer.prewarm()
    server = create_server(settings, synthesizer)
    try:
        print(f"IndexTTS HTTP service listening on {settings.host}:{settings.port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
