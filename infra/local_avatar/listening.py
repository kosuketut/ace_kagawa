"""Turn-taking for the microphone: VAD endpointing and early recognition.

Ported from ELF-SR2app (mars: scripts/auto_voice.py, voice_chat.py,
local_asr.py). The browser sends 16 kHz mono PCM16 in chunks of any size;
FrameChunker cuts it into the 30 ms frames WebRTC VAD needs.
"""

from __future__ import annotations

import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RATE = 16000
FRAME_MS = 30
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
END_SILENCE_MS = 700  # Silence that ends a turn.
EARLY_ASR_MS = 300  # Silence after which recognition starts while the endpoint is still being confirmed.
MAX_UTTERANCE_MS = 19980

# Whisper's stock outputs for noise/silence (subtitle credits). Never sent to the LLM.
HALLUCINATIONS = {"ご視聴ありがとうございました", "ご視聴ありがとうございます", "ご清聴ありがとうございました",
                  "チャンネル登録よろしくお願いします", "チャンネル登録お願いします"}


class Discarded(Exception):
    """This utterance is dropped and listening continues (nothing was sent)."""


class Unheard(ValueError):
    """Nothing usable was recognized; the turn is dropped without generating a reply."""


def check_transcript(text: str) -> str:
    text = text.strip()
    if not text or re.sub(r"[\s。、．，,.!！?？…]", "", text) in HALLUCINATIONS:
        raise Unheard("聞き取れませんでした。もう一度話してください。")
    if len(text) > 1000:
        raise ValueError("音声認識結果が長すぎます。")
    return text


class FrameChunker:
    """Re-cut a byte stream of PCM16 into fixed 30 ms frames."""

    def __init__(self) -> None:
        self._pending = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        self._pending += data
        frames = []
        while len(self._pending) >= FRAME_BYTES:
            frames.append(bytes(self._pending[:FRAME_BYTES]))
            del self._pending[:FRAME_BYTES]
        return frames

    def reset(self) -> None:
        self._pending.clear()


class Utterance:
    """VAD endpointing. on_pause(pcm) fires once a speech utterance has been silent for early_ms
    (so recognition can start early); on_resume() fires when speech starts again after that point."""

    def __init__(self, end_silence_ms=END_SILENCE_MS, on_pause=None, on_resume=None, early_ms=EARLY_ASR_MS):
        if not 300 <= end_silence_ms <= 3000:
            raise ValueError("end_silence_ms は300〜3000にしてください。")
        self.end_frames = end_silence_ms // FRAME_MS
        self.early_frames = early_ms // FRAME_MS if early_ms > 0 else 0
        self.on_pause, self.on_resume = on_pause, on_resume
        self.reset()

    def reset(self):
        self.preroll = deque(maxlen=10)
        self.frames = []
        self.voiced = 0
        self.silent = 0
        self.active = False

    def feed(self, pcm, speech):
        if len(pcm) != FRAME_BYTES:
            raise ValueError("Unexpected microphone frame size")
        if not self.active:
            self.preroll.append(pcm)
            self.voiced = self.voiced + 1 if speech else 0
            if self.voiced >= 5:  # 150 ms sustained speech, with 300 ms pre-roll.
                self.active = True
                self.frames = list(self.preroll)
            return None
        self.frames.append(pcm)
        self.voiced += int(speech)
        early = 0 < self.early_frames < self.end_frames
        if speech:
            if early and self.silent >= self.early_frames and self.on_resume:
                self.on_resume()
            self.silent = 0
        else:
            self.silent += 1
        if len(self.frames) * FRAME_MS >= MAX_UTTERANCE_MS:
            self.reset()
            if self.on_resume:
                self.on_resume()
            raise Discarded("20秒続けて音を拾ったため送信しませんでした。短く区切って話してください。")
        if early and self.silent == self.early_frames and self.voiced >= 10 and self.on_pause:
            self.on_pause(b"".join(self.frames))
        if self.silent >= self.end_frames:
            if self.voiced < 10:  # Reject isolated short sounds; resume listening.
                self.reset()
                return None
            return b"".join(self.frames)
        return None


class EarlyASR:
    """Recognize an utterance while its end-of-speech silence is still being confirmed.

    pause(pcm) after a short silence, resume() if speech continues. At the endpoint the final PCM is
    the paused PCM plus silence only, so the early transcript is used when it is still current.
    One worker thread, so the ASR model never runs two recognitions at once.
    """

    def __init__(self, asr) -> None:
        self._asr = asr
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="early-asr")
        self._current = None  # (paused PCM length, future)
        self.used = 0

    def _recognize(self, pcm: bytes) -> str:
        return check_transcript(self._asr.transcribe(pcm))

    def pause(self, pcm: bytes) -> None:
        self._current = (len(pcm), self._pool.submit(self._recognize, pcm))

    def resume(self) -> None:
        self._current = None

    def result(self, pcm: bytes) -> str:
        current, self._current = self._current, None
        if current is not None and len(pcm) >= current[0]:
            self.used += 1
            return current[1].result()
        return self._pool.submit(self._recognize, pcm).result()

    def close(self) -> None:
        self._current = None
        self._pool.shutdown(wait=False, cancel_futures=True)


class WhisperASR:
    """faster-whisper (CTranslate2) with a local model directory; nothing is downloaded at runtime."""

    def __init__(self, model_dir: str, device: str = "cuda", device_index: int = 0,
                 compute_type: str = "float16", language: str = "ja", beam_size: int = 5) -> None:
        path = Path(model_dir)
        if not (path / "model.bin").is_file():
            raise ValueError(f"ASRモデルが見つかりません: {path}")
        from faster_whisper import WhisperModel

        self.language = language
        self.beam_size = beam_size
        self.model = WhisperModel(str(path), device=device, device_index=device_index,
                                  compute_type=compute_type, local_files_only=True)
        # Warm up CUDA kernels so the first real turn is not slower.
        self.transcribe(b"\x00\x00" * (RATE // 2))

    def transcribe(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        # condition_on_previous_text=False limits repetition on short utterances.
        segments, _info = self.model.transcribe(
            audio, language=self.language, task="transcribe", beam_size=self.beam_size,
            condition_on_previous_text=False, without_timestamps=True)
        return "".join(segment.text for segment in segments).strip()
