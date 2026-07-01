from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VadEvent:
    is_speech: bool
    speech_started: bool
    end_of_utterance: bool


class TurnDetector:
    def __init__(
        self,
        *,
        sample_rate_hz: int,
        frame_ms: int,
        eos_silence_ms: int,
        aggressiveness: int,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._frame_ms = frame_ms
        self._eos_silence_ms = eos_silence_ms
        self._silence_ms = 0
        self._in_speech = False
        self._fallback_threshold = 350
        try:
            import webrtcvad  # type: ignore

            self._vad = webrtcvad.Vad(aggressiveness)
        except Exception:
            self._vad = None

    def reset(self) -> None:
        self._silence_ms = 0
        self._in_speech = False

    def feed(self, pcm16: bytes) -> VadEvent:
        if not pcm16:
            return VadEvent(is_speech=False, speech_started=False, end_of_utterance=False)
        is_speech = self._is_speech_frame(pcm16)
        speech_started = False
        end_of_utterance = False

        if is_speech:
            self._silence_ms = 0
            if not self._in_speech:
                self._in_speech = True
                speech_started = True
        elif self._in_speech:
            self._silence_ms += self._frame_ms
            if self._silence_ms >= self._eos_silence_ms:
                end_of_utterance = True
                self.reset()

        return VadEvent(
            is_speech=is_speech,
            speech_started=speech_started,
            end_of_utterance=end_of_utterance,
        )

    def _is_speech_frame(self, pcm16: bytes) -> bool:
        if self._vad is not None:
            return bool(self._vad.is_speech(pcm16, self._sample_rate_hz))
        return self._fallback_energy_vad(pcm16)

    def _fallback_energy_vad(self, pcm16: bytes) -> bool:
        if len(pcm16) < 2:
            return False
        sample_count = len(pcm16) // 2
        total = 0
        for index in range(0, len(pcm16), 2):
            value = int.from_bytes(pcm16[index : index + 2], byteorder="little", signed=True)
            total += abs(value)
        return (total / sample_count) >= self._fallback_threshold

