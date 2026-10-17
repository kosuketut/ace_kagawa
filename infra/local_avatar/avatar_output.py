"""Speak WAV segments through Audio2Face-3D into the relay, in order.

This replaces ELF-SR2app's file mailbox to Unreal: each sentence WAV goes to
Audio2Face-3D and its animation and audio are pushed to mini_animgraph.py,
which plays utterances back to back. idle() is the counterpart of the UE
playback acknowledgement: nothing queued here and the relay not speaking.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import wave
from collections.abc import Callable
from urllib import request

import numpy as np

from speech_buffer import resample_int16

A2F_RATE = 16000


def wav_to_pcm16k(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("TTS must return PCM16 WAV")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return resample_int16(pcm, w.getframerate(), A2F_RATE).tobytes()


def relay_status(url: str, timeout: float = 2.0) -> dict:
    with request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


class AvatarVoice:
    def __init__(self, speak: Callable[[bytes], dict], status: Callable[[], dict]) -> None:
        self._speak = speak
        self._status = status
        self._jobs: queue.Queue[bytes | None] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self.error: Exception | None = None
        self._worker = threading.Thread(target=self._run, daemon=True, name="avatar-voice")
        self._worker.start()

    def say(self, wav: bytes) -> None:
        pcm = wav_to_pcm16k(wav)
        with self._lock:
            self._pending += 1
        self._jobs.put(pcm)

    def _run(self) -> None:
        while (pcm := self._jobs.get()) is not None:
            try:
                self._speak(pcm)
            except Exception as exc:  # Reported through idle()/error; the next turn may recover.
                self.error = exc
            finally:
                with self._lock:
                    self._pending -= 1

    def idle(self) -> bool:
        with self._lock:
            if self._pending:
                return False
        status = self._status()
        return not status["speaking"] and status["queued"] == 0

    def wait_idle(self, timeout: float = 180.0, poll: float = 0.1) -> None:
        deadline = time.monotonic() + timeout
        while not self.idle():
            if time.monotonic() >= deadline:
                raise TimeoutError("アバターの再生完了を確認できません。")
            time.sleep(poll)

    def close(self) -> None:
        self._jobs.put(None)
        self._worker.join(timeout=5)
