"""One utterance of face animation and audio, cut into fixed-rate frames.

Frames and audio may arrive in pieces (for example from Audio2Face-3D while it
is still inferring). Frame ``k`` owns the audio samples
``[round(k * rate / fps), round((k + 1) * rate / fps))`` so frame lengths
alternate (533/534 at 16 kHz and 30 fps) and never drift from the clock.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

import numpy as np


def resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    dst_len = int(round(len(samples) * dst_rate / src_rate))
    src_t = np.arange(len(samples)) / src_rate
    dst_t = np.arange(dst_len) / dst_rate
    out = np.interp(dst_t, src_t, samples.astype(np.float32))
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class SpeechBuffer:
    def __init__(self, blendshape_names: Sequence[str], fps: int, sample_rate: int) -> None:
        self.names = list(blendshape_names)
        self.fps = fps
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._weights: list[np.ndarray] = []
        self._audio: list[np.ndarray] = []
        self._audio_len = 0
        self._audio_cache: np.ndarray | None = None
        self._source_index: list[int | None] = list(range(len(self.names)))
        self._source_rate = sample_rate
        self._finished = False

    def set_source(self, names: Sequence[str], audio_rate: int) -> None:
        position = {name: i for i, name in enumerate(names)}
        with self._lock:
            self._source_index = [position.get(name) for name in self.names]
            self._source_rate = audio_rate

    def add_weights(self, values: Sequence[float]) -> None:
        row = np.zeros(len(self.names), dtype=np.float32)
        for target, source in enumerate(self._source_index):
            if source is not None and source < len(values):
                row[target] = values[source]
        with self._lock:
            self._weights.append(row)

    def add_audio(self, pcm16: bytes) -> None:
        chunk = np.frombuffer(pcm16, dtype=np.int16)
        chunk = resample_int16(chunk, self._source_rate, self.sample_rate)
        with self._lock:
            self._audio.append(chunk)
            self._audio_len += len(chunk)
            self._audio_cache = None

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def audio_samples(self) -> int:
        with self._lock:
            return self._audio_len

    def _span(self, k: int) -> tuple[int, int]:
        return (round(k * self.sample_rate / self.fps), round((k + 1) * self.sample_rate / self.fps))

    def total_frames(self) -> int:
        with self._lock:
            audio_frames = math.ceil(self._audio_len * self.fps / self.sample_rate)
            return max(len(self._weights), audio_frames)

    def available(self, k: int) -> bool:
        with self._lock:
            have_weights = k < len(self._weights)
            have_audio = self._span(k)[1] <= self._audio_len
            if have_weights and have_audio:
                return True
            if not self._finished:
                return False
        return k < self.total_frames()

    def exhausted(self, k: int) -> bool:
        return self.finished and k >= self.total_frames()

    def ready_to_start(self, prebuffer_frames: int) -> bool:
        if self.finished:
            return True
        return self.available(max(prebuffer_frames - 1, 0))

    def frame(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if k < len(self._weights):
                weights = self._weights[k]
            else:
                weights = np.zeros(len(self.names), dtype=np.float32)
            if self._audio_cache is None:
                self._audio_cache = (
                    np.concatenate(self._audio) if self._audio else np.zeros(0, dtype=np.int16)
                )
            a0, a1 = self._span(k)
            return weights, self._audio_cache[a0:a1]
