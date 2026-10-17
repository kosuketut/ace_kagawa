import importlib.util
from array import array
import json
from pathlib import Path
import sys
from threading import Thread
import unittest
import urllib.request
import wave
from io import BytesIO


MODULE_PATH = Path(__file__).with_name("indextts_tts_server.py")
SPEC = importlib.util.spec_from_file_location("indextts_tts_server_tested", MODULE_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class FakeIndexTTS:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return 22050, array("h", [0, 1000, -1000, 32767])


class IndexTtsServerTests(unittest.TestCase):
    def test_settings_defaults_match_deployed_index_tts_setup(self) -> None:
        settings = server.IndexTtsSettings.from_env({})

        self.assertEqual(settings.port, 8023)
        self.assertEqual(settings.repo_dir, Path("/data/ACE/en-tts/repos/IndexTTS"))
        self.assertEqual(settings.model_dir, Path("/data/ACE/en-tts/models/IndexTTS-2.5"))
        self.assertEqual(settings.reference_audio, Path("/data/ACE/en-tts/reference/kagawa_en_sentence1.wav"))
        self.assertEqual(settings.device, "cuda:0")
        self.assertEqual(settings.cuda_visible_devices, "1")
        self.assertEqual(settings.voice, "kagawa_en")
        self.assertEqual(settings.response_sample_rate_hz, 22050)
        self.assertEqual(settings.trailing_silence_ms, 500)

    def test_symposium_prewarm_file_has_exactly_seven_sentences(self) -> None:
        sentences = server.sentence_texts(server.DEFAULT_PREWARM_TEXT)

        self.assertEqual(len(sentences), 7)
        self.assertEqual(sentences[0], "Good afternoon, everyone.")
        self.assertEqual(
            sentences[-1],
            "And now, please welcome Professor Kagawa, the President of the Tokyo University of Technology.",
        )

    def test_pronunciation_substitutions_match_index_tts_annotation_syntax(self) -> None:
        text = server.apply_pronunciation_substitutions(
            "Our supercomputer is Seiran in Hachioji, the City near the establishment."
        )

        self.assertEqual(
            text,
            "Our <supercomputer|S UW2 . P ER0 . K AH0 M . P Y UW1 . T ER0> is "
            "<Seiran|S EY1 . R AA0 N> in <Hachioji|HH AA1 . CH IY0 . OW0 . JH IY0>, "
            "the City near the establishment.",
        )

    def test_audio_conversion_and_wav_are_mono_pcm16(self) -> None:
        pcm = server.audio_to_pcm16(array("h", [-32768, 0, 32767]))
        with wave.open(BytesIO(server.pcm16_to_wav(pcm, 22050)), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 22050))
            self.assertEqual(wav.readframes(3), pcm)

    def test_synthesize_uses_proven_index_tts_arguments(self) -> None:
        settings = server.IndexTtsSettings(preload_model=False)
        synthesizer = server.IndexTtsSynthesizer(settings)
        model = FakeIndexTTS()
        synthesizer.model = model

        pcm, media_type, sample_rate = synthesizer.synthesize("Hachioji.")
        self.assertEqual((media_type, sample_rate), ("audio/L16", 22050))
        self.assertEqual(len(pcm), 8 + 22050)
        self.assertEqual(pcm[-22050:], bytes(22050))
        call = model.calls[0]
        self.assertEqual(call["lang"], "EN")
        self.assertFalse(call["use_random"])
        self.assertEqual(call["interval_silence"], 350)
        self.assertEqual(call["duration_factor"], 1.0)
        self.assertFalse(call["text_normalization"])
        self.assertEqual(call["text"], "<Hachioji|HH AA1 . CH IY0 . OW0 . JH IY0>.")

    def test_prewarm_populates_all_seven_entries_and_reuses_them(self) -> None:
        settings = server.IndexTtsSettings(preload_model=False)
        synthesizer = server.IndexTtsSynthesizer(settings)
        model = FakeIndexTTS()
        synthesizer.model = model

        synthesizer.prewarm()
        self.assertEqual(synthesizer.prewarm_entries, 7)
        self.assertEqual(len(model.calls), 7)
        first_text = server.sentence_texts(settings.prewarm_text)[0]
        synthesizer.synthesize(first_text)
        self.assertEqual(len(model.calls), 7)

    def test_http_server_returns_health_and_openai_compatible_audio(self) -> None:
        settings = server.IndexTtsSettings(host="127.0.0.1", port=0, preload_model=False)
        synthesizer = server.IndexTtsSynthesizer(settings)
        synthesizer.model = FakeIndexTTS()
        http_server = server.create_server(settings, synthesizer)
        thread = Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{http_server.server_address[1]}"
        try:
            with urllib.request.urlopen(f"{base_url}/healthz") as response:
                health = json.load(response)
            self.assertEqual(health["service"], "indextts-tts")
            self.assertEqual(health["voice"], "kagawa_en")
            self.assertTrue(health["model_loaded"])
            self.assertEqual(health["prewarm_entries"], 0)
            self.assertEqual(health["trailing_silence_ms"], 500)

            request = urllib.request.Request(
                f"{base_url}/v1/audio/speech",
                data=json.dumps(
                    {
                        "input": "Good afternoon, everyone.",
                        "voice": "kagawa_en",
                        "response_format": "pcm",
                        "stream": True,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                audio = response.read()
                self.assertEqual(response.headers["X-Sample-Rate-Hz"], "22050")
                self.assertEqual(response.headers["X-Audio-Channels"], "1")
                self.assertEqual(response.headers["X-Audio-Encoding"], "pcm_s16le")
            self.assertEqual(len(audio), 8 + 22050)
            self.assertEqual(audio[-22050:], bytes(22050))
        finally:
            http_server.shutdown()
            http_server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
