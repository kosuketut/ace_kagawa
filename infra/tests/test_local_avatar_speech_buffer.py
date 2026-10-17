from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "infra" / "local_avatar"))

from speech_buffer import SpeechBuffer  # noqa: E402

NAMES = ["JawOpen", "MouthClose", "EyeBlinkLeft"]


def pcm(samples: list[int] | np.ndarray) -> bytes:
    return np.asarray(samples, dtype=np.int16).tobytes()


class SpeechBufferTest(unittest.TestCase):
    def make(self, **kw) -> SpeechBuffer:
        buf = SpeechBuffer(NAMES, fps=30, sample_rate=16000)
        buf.set_source(kw.get("names", NAMES), kw.get("audio_rate", 16000))
        return buf

    def test_frame_audio_spans_cover_one_second_exactly(self):
        buf = self.make()
        buf.add_audio(pcm(np.arange(16000) % 1000))
        for _ in range(30):
            buf.add_weights([0.0, 0.0, 0.0])
        buf.finish()
        lengths = [len(buf.frame(k)[1]) for k in range(30)]
        self.assertEqual(sum(lengths), 16000)
        self.assertEqual(set(lengths), {533, 534})
        # Spans are contiguous: frame 1 starts where frame 0 ended.
        self.assertEqual(int(buf.frame(1)[1][0]), lengths[0] % 1000)

    def test_weights_are_mapped_by_name_and_missing_names_are_zero(self):
        buf = self.make(names=["EyeBlinkLeft", "Unknown", "JawOpen"])
        buf.add_weights([0.25, 0.9, 0.5])
        buf.add_audio(pcm(np.zeros(600)))
        weights, _ = buf.frame(0)
        np.testing.assert_allclose(weights, [0.5, 0.0, 0.25])

    def test_frame_waits_for_both_weights_and_audio_until_finished(self):
        buf = self.make()
        buf.add_weights([0.1, 0.0, 0.0])
        self.assertFalse(buf.available(0))
        buf.add_audio(pcm(np.zeros(533)))
        self.assertTrue(buf.available(0))
        self.assertFalse(buf.available(1))
        buf.finish()
        self.assertFalse(buf.available(1))
        self.assertTrue(buf.exhausted(1))

    def test_finished_buffer_plays_until_the_longer_of_audio_and_weights(self):
        buf = self.make()
        buf.add_weights([0.3, 0.0, 0.0])
        buf.add_audio(pcm(np.ones(16000 // 10)))  # 0.1 s = 3 frames
        buf.finish()
        self.assertEqual(buf.total_frames(), 3)
        weights, audio = buf.frame(2)
        np.testing.assert_allclose(weights, [0.0, 0.0, 0.0])
        self.assertEqual(len(audio), 533)
        self.assertTrue(buf.exhausted(3))

    def test_audio_at_another_rate_is_resampled_to_the_stream_rate(self):
        buf = self.make(audio_rate=24000)
        buf.add_audio(pcm(np.zeros(24000)))
        buf.finish()
        self.assertEqual(buf.audio_samples(), 16000)

    def test_ready_to_start_needs_prebuffer_or_finish(self):
        buf = self.make()
        for _ in range(3):
            buf.add_weights([0.0, 0.0, 0.0])
        buf.add_audio(pcm(np.zeros(1600)))
        self.assertFalse(buf.ready_to_start(prebuffer_frames=5))
        self.assertTrue(buf.ready_to_start(prebuffer_frames=3))
        buf.finish()
        self.assertTrue(buf.ready_to_start(prebuffer_frames=5))


if __name__ == "__main__":
    unittest.main()
