"""Tests for the Tokkio-free conversation loop.

The cases for sentence splitting, the first-clause cut and VAD endpointing are
the ones from ELF-SR2app (mars: scripts/test_chat_to_wav.py, test_auto_voice.py),
whose logic infra/local_avatar ports.
"""

from __future__ import annotations

import io
import sys
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "infra" / "local_avatar"))

import conversation as chat  # noqa: E402
import listening as auto  # noqa: E402

FRAME = b"\1\0" * (auto.FRAME_BYTES // 2)
SILENT = bytes(auto.FRAME_BYTES)


def tone_wav(samples: int, rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\1\0" * samples)
    return out.getvalue()


class SplitTests(unittest.TestCase):
    def test_split_sentences(self):
        self.assertEqual(chat.split_sentences("こんにちは。元気？まだ"), (["こんにちは。", "元気？"], "まだ"))
        self.assertEqual(chat.split_sentences("「はい。」次"), (["「はい。」"], "次"))
        self.assertEqual(chat.split_sentences("。 \n"), ([], ""))
        self.assertEqual(chat.split_sentences("終わり", final=True), (["終わり"], ""))
        long = "あ" * 50 + "、" + "い" * 20
        self.assertEqual(chat.split_sentences(long), (["あ" * 50 + "、"], "い" * 20))

    def test_first_clause(self):
        self.assertEqual(chat.first_clause("そうですね、何か食べ"), ("そうですね、", "何か食べ"))
        self.assertEqual(chat.first_clause("あ、そうですね、次"), ("あ、そうですね、", "次"))
        self.assertEqual(chat.first_clause("はい。そうですね、次"), (None, "はい。そうですね、次"))
        self.assertEqual(chat.first_clause("そうですね"), (None, "そうですね"))


class ReplyStreamTests(unittest.TestCase):
    profile = {
        "llm_base_url": "http://llm.invalid/v1", "llm_model": "m",
        "tts_base_url": "http://tts.invalid", "tts_voice": "kagawa",
        "temperature": 0.0, "max_tokens": 256, "max_reply_chars": 400,
    }

    def run_stream(self, deltas, split_first_clause=True):
        spoken, published = [], []

        def tts(text):
            spoken.append(text)
            return tone_wav(1600)

        streamer = chat.ReplyStreamer(
            dict(self.profile, split_first_clause=split_first_clause),
            stream=lambda payload: iter(deltas), synthesize=tts,
        )
        reply, meta = streamer.run([{"role": "user", "content": "こんにちは"}],
                                   on_segment=lambda audio, text: published.append(text))
        return spoken, published, reply, meta

    def test_only_the_first_sentence_is_cut_at_a_comma(self):
        deltas = [("そうですね、何か", None), ("食べたい。次も、", None), ("です。", None), ("", "stop")]
        for split, expected in ((True, ["そうですね、", "何か食べたい。", "次も、です。"]),
                                (False, ["そうですね、何か食べたい。", "次も、です。"])):
            with self.subTest(split=split):
                spoken, published, reply, meta = self.run_stream(deltas, split)
                self.assertEqual(spoken, expected)
                self.assertEqual(published, expected)
                self.assertEqual(reply, "そうですね、何か食べたい。次も、です。")
                self.assertEqual(meta["segments"], len(expected))

    def test_reply_without_stop_is_an_error_after_speaking(self):
        deltas = [("途中まで。", None), ("", "length")]
        with self.assertRaises(ValueError):
            self.run_stream(deltas)

    def test_tts_failure_stops_the_reply(self):
        def broken(text):
            raise ConnectionError("tts down")

        streamer = chat.ReplyStreamer(self.profile, stream=lambda p: iter([("はい。", None), ("", "stop")]),
                                      synthesize=broken)
        with self.assertRaises(ConnectionError):
            streamer.run([], on_segment=lambda a, t: None)

    def test_reasoning_markup_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_stream([("<think>", None), ("", "stop")])


class UtteranceTests(unittest.TestCase):
    def test_silence_is_bounded(self):
        detector = auto.Utterance()
        for _ in range(2000):
            self.assertIsNone(detector.feed(FRAME, False))
        self.assertFalse(detector.active)
        self.assertEqual(len(detector.preroll), 10)
        self.assertEqual(detector.frames, [])

    def test_speech_and_short_pause_then_endpoint(self):
        for silence_ms, frames in ((auto.END_SILENCE_MS, 23), (900, 30)):
            detector = auto.Utterance(silence_ms)
            for speech in [False] * 20 + [True] * 15 + [False] * 10 + [True] * 10 + [False] * (frames - 1):
                self.assertIsNone(detector.feed(FRAME, speech))
            self.assertIsInstance(detector.feed(FRAME, False), bytes)

    def test_early_pause_and_resume_callbacks(self):
        paused, resumed = [], []
        detector = auto.Utterance(700, on_pause=paused.append, on_resume=lambda: resumed.append(1), early_ms=300)
        for speech in [True] * 15 + [False] * 10:
            self.assertIsNone(detector.feed(FRAME if speech else SILENT, speech))
        self.assertEqual(len(paused), 1)
        self.assertEqual(paused[0], b"".join(detector.frames))
        self.assertIsNone(detector.feed(FRAME, True))
        self.assertEqual(resumed, [1])
        for _ in range(22):
            self.assertIsNone(detector.feed(SILENT, False))
        self.assertEqual(len(paused), 2)
        audio = detector.feed(SILENT, False)
        self.assertIsInstance(audio, bytes)
        self.assertTrue(audio.startswith(paused[1]))
        self.assertFalse(any(audio[len(paused[1]):]))

    def test_no_early_callback_for_short_noise_or_when_disabled(self):
        paused = []
        detector = auto.Utterance(700, on_pause=paused.append, early_ms=300)
        for speech in [True] * 6 + [False] * 30:
            detector.feed(FRAME, speech)
        self.assertEqual(paused, [])
        detector = auto.Utterance(700, on_pause=paused.append, early_ms=0)
        for speech in [True] * 15 + [False] * 30:
            detector.feed(FRAME, speech)
        self.assertEqual(paused, [])

    def test_end_silence_bounds(self):
        for silence_ms in (299, 3001):
            with self.subTest(silence_ms=silence_ms), self.assertRaises(ValueError):
                auto.Utterance(silence_ms)

    def test_short_noise_discarded(self):
        detector = auto.Utterance()
        for speech in [True] * 5 + [False] * 30:
            self.assertIsNone(detector.feed(FRAME, speech))
        self.assertFalse(detector.active)

    def test_continuous_speech_discarded_at_cap_and_reset(self):
        detector = auto.Utterance()
        with self.assertRaises(auto.Discarded):
            for _ in range(700):
                detector.feed(FRAME, True)
        self.assertFalse(detector.active)
        self.assertEqual(detector.frames, [])

    def test_frame_validation(self):
        with self.assertRaises(ValueError):
            auto.Utterance().feed(b"", True)


class FrameChunkerTests(unittest.TestCase):
    def test_arbitrary_chunks_become_30ms_frames(self):
        chunker = auto.FrameChunker()
        frames = chunker.push(b"\0" * 700) + chunker.push(b"\0" * 1000)
        self.assertEqual([len(f) for f in frames], [auto.FRAME_BYTES] * 1)
        frames = chunker.push(b"\0" * (auto.FRAME_BYTES * 2))
        self.assertEqual(len(frames), 2)

    def test_odd_byte_is_kept_for_the_next_chunk(self):
        chunker = auto.FrameChunker()
        self.assertEqual(chunker.push(b"\0" * (auto.FRAME_BYTES - 1)), [])
        self.assertEqual(len(chunker.push(b"\0")), 1)


class EarlyASRTests(unittest.TestCase):
    def test_early_asr_result_is_used_unless_speech_resumed(self):
        seen = []

        def transcribe(pcm):
            seen.append(pcm)
            return f"認識{len(seen)}"

        early = auto.EarlyASR(NS(transcribe=transcribe))
        paused = b"\1\0" * 1600
        early.pause(paused)
        self.assertEqual(early.result(paused + b"\0\0" * 6400), "認識1")
        self.assertEqual(seen, [paused])
        early.pause(paused)
        early.resume()
        final = paused * 2
        self.assertEqual(early.result(final), "認識3")
        self.assertEqual(seen, [paused, paused, final])
        self.assertEqual(early.used, 1)
        early.close()

    def test_early_asr_propagates_unheard(self):
        early = auto.EarlyASR(NS(transcribe=Mock(return_value="")))
        early.pause(b"\1\0" * 1600)
        with self.assertRaises(auto.Unheard):
            early.result(b"\1\0" * 1600)
        early.close()

    def test_subtitle_credits_are_unheard(self):
        for text in ("ご視聴ありがとうございました。", "  ", "チャンネル登録お願いします"):
            with self.subTest(text=text), self.assertRaises(auto.Unheard):
                auto.check_transcript(text)
        self.assertEqual(auto.check_transcript(" 学長室はどこですか "), "学長室はどこですか")


if __name__ == "__main__":
    unittest.main()
