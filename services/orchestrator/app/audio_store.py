from __future__ import annotations

import wave
from pathlib import Path
from typing import Any
from uuid import UUID


class TurnAudioArtifactStore:
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def write_turn_audio(
        self,
        *,
        session_id: UUID,
        turn_id: UUID,
        mic_audio: bytes,
        tts_audio: bytes,
        tts_sample_rate_hz: int,
    ) -> dict[str, Any]:
        turn_dir = self._root_dir / str(session_id)
        turn_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {}
        if mic_audio:
            path = turn_dir / f"{turn_id}-mic.wav"
            self._write_pcm16_wav(path, sample_rate_hz=16000, payload=mic_audio)
            result["mic_wav"] = str(path)
        if tts_audio:
            path = turn_dir / f"{turn_id}-tts.wav"
            self._write_pcm16_wav(path, sample_rate_hz=tts_sample_rate_hz, payload=tts_audio)
            result["tts_wav"] = str(path)
        return result

    @staticmethod
    def _write_pcm16_wav(path: Path, *, sample_rate_hz: int, payload: bytes) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate_hz)
            handle.writeframes(payload)

