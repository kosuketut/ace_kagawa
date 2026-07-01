from __future__ import annotations

import unittest
from uuid import uuid4

from app.protocol import AudioFrameKind, pack_audio_frame, unpack_audio_frame


class ProtocolTests(unittest.TestCase):
    def test_audio_frame_round_trip(self) -> None:
        turn_id = uuid4()
        payload = b"\x01\x02\x03\x04"
        encoded = pack_audio_frame(
            kind=AudioFrameKind.TTS,
            sample_rate_hz=24000,
            channels=1,
            payload=payload,
            turn_id=turn_id,
        )
        decoded = unpack_audio_frame(encoded)
        self.assertEqual(decoded.kind, AudioFrameKind.TTS)
        self.assertEqual(decoded.sample_rate_hz, 24000)
        self.assertEqual(decoded.channels, 1)
        self.assertEqual(decoded.payload, payload)
        self.assertEqual(decoded.turn_id, turn_id)


if __name__ == "__main__":
    unittest.main()

