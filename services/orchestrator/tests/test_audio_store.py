from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from uuid import uuid4

from app.audio_store import TurnAudioArtifactStore


class AudioStoreTests(unittest.TestCase):
    def test_write_turn_audio_creates_wav_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TurnAudioArtifactStore(Path(tmp))
            session_id = uuid4()
            turn_id = uuid4()
            artifacts = store.write_turn_audio(
                session_id=session_id,
                turn_id=turn_id,
                mic_audio=b"\x00\x00" * 160,
                tts_audio=b"\x01\x00" * 240,
                tts_sample_rate_hz=24000,
            )
            self.assertIn("mic_wav", artifacts)
            self.assertIn("tts_wav", artifacts)
            with wave.open(artifacts["tts_wav"], "rb") as handle:
                self.assertEqual(handle.getframerate(), 24000)
                self.assertEqual(handle.getnchannels(), 1)


if __name__ == "__main__":
    unittest.main()

