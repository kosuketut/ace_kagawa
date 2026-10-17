import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys
import wave

import numpy as np


MODULE_PATH = Path(__file__).with_name("qwen_tts_server.py")
SPEC = importlib.util.spec_from_file_location("qwen_tts_server_tested", MODULE_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class QwenTtsServerTests(unittest.TestCase):
    def test_settings_read_variant_environment(self) -> None:
        settings = server.QwenTtsSettings.from_env(
            {
                "QWEN_TTS_PORT": "9000",
                "QWEN_TTS_DEVICE": "cuda:0",
                "QWEN_TTS_VOICE": "test_voice",
                "QWEN_TTS_PRELOAD_MODEL": "false",
            }
        )
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.device, "cuda:0")
        self.assertEqual(settings.voice, "test_voice")
        self.assertFalse(settings.preload_model)

    def test_markdown_body_drops_heading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.md"
            path.write_text("# Heading\n\nSpoken reference.\n", encoding="utf-8")
            self.assertEqual(server.markdown_body(path), "Spoken reference.")

    def test_symposium_prewarm_file_matches_seven_runtime_segments(self) -> None:
        segments = server.sentence_texts(server.DEFAULT_PREWARM_TEXT)
        self.assertEqual(len(segments), 7)
        self.assertEqual(segments[0], "Good afternoon, everyone.")
        self.assertTrue(segments[-1].endswith("Tokyo University of Technology."))

    def test_audio_conversion_resamples_full_signal_and_writes_mono_pcm16_wav(self) -> None:
        source = np.linspace(-1, 1, 24000, dtype=np.float32)
        pcm = server.audio_to_pcm16(source, source_rate=24000, target_rate=16000)
        wav_bytes = server.pcm16_to_wav(pcm, 16000)

        with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        self.assertEqual(len(samples), 16000)
        self.assertLess(samples[0], -32000)
        self.assertGreater(samples[-1], 32000)


if __name__ == "__main__":
    unittest.main()
