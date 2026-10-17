import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
from threading import Event, Thread
import time
import types
import unittest
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parent
INDEXTTS_REPO_DIR = Path("/data/ACE/en-tts/repos/IndexTTS")
HACHIOJI = "HH AA1 . CH IY0 . OW0 . JH IY0"
SEIRAN = "S EY1 . R AA0 N"
SUPERCOMPUTER = "S UW2 . P ER0 . K AH0 M . P Y UW1 . T ER0"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


server = load_module("indextts_tts_server_fixes_tested", "indextts_tts_server.py")
qwen_server = load_module("qwen_tts_server_fixes_tested", "qwen_tts_server.py")


def unit_environment(filename: str) -> dict[str, str]:
    text = (ROOT / filename).read_text(encoding="utf-8")
    return dict(re.findall(r"^Environment=([A-Z0-9_]+)=(.*)$", text, flags=re.MULTILINE))


def sine(frequency_hz: float, sample_rate_hz: int, seconds: float, amplitude: int = 10000) -> np.ndarray:
    t = np.arange(int(sample_rate_hz * seconds)) / sample_rate_hz
    return np.rint(amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.int16)


class NumpyIndexTTS:
    """Returns audio like IndexTTS2.infer: an int16 ndarray of shape (N, 1)."""

    def __init__(self, samples=(0, 1000, -1000, 32767), sample_rate_hz: int = 22050, text_process=None) -> None:
        self.samples = np.asarray(samples, dtype=np.int16)
        self.sample_rate_hz = sample_rate_hz
        self.calls: list[dict[str, object]] = []
        if text_process is not None:
            self.text_process = text_process

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return self.sample_rate_hz, self.samples.reshape(-1, 1)


class BlockingIndexTTS(NumpyIndexTTS):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def infer(self, **kwargs):
        self.entered.set()
        self.release.wait(timeout=10)
        return super().infer(**kwargs)


class FakeTextNormalizer:
    def __init__(self, replacements: dict[str, str] | None = None, chinese: bool = False) -> None:
        self.replacements = replacements or {}
        self.chinese = chinese
        self.inputs: list[str] = []

    def use_chinese(self, text: str) -> bool:
        return self.chinese

    def normalize(self, text: str) -> str:
        self.inputs.append(text)
        for old, new in self.replacements.items():
            text = text.replace(old, new)
        return text


def synthesizer_with(model, **settings_kwargs):
    settings = server.IndexTtsSettings(preload_model=False, **settings_kwargs)
    synthesizer = server.IndexTtsSynthesizer(settings)
    synthesizer.model = model
    return synthesizer


class AudioConversionTests(unittest.TestCase):
    def test_converts_real_index_tts_int16_column_array(self) -> None:
        samples = np.array([[0], [1000], [-1000], [32767], [-32768]], dtype=np.int16)

        self.assertEqual(server.audio_to_pcm16(samples), samples.reshape(-1).astype("<i2").tobytes())

    def test_scales_float_audio_and_zeroes_non_finite_values(self) -> None:
        pcm = server.audio_to_pcm16(np.array([0.0, 0.5, -1.0, np.nan], dtype=np.float32))

        self.assertEqual(np.frombuffer(pcm, dtype="<i2").tolist(), [0, 16384, -32767, 0])
        self.assertEqual(server.audio_to_pcm16([0.25, float("inf")]), np.array([8192, 0], dtype="<i2").tobytes())
        self.assertEqual(server.audio_to_pcm16([1, -2, 40000]), np.array([1, -2, 32767], dtype="<i2").tobytes())
        self.assertEqual(server.audio_to_pcm16(np.zeros((0, 1), dtype=np.int16)), b"")

    def test_conversion_is_vectorized(self) -> None:
        samples = np.zeros((22050 * 30, 1), dtype=np.int16)

        started = time.perf_counter()
        server.audio_to_pcm16(samples)
        self.assertLess(time.perf_counter() - started, 0.25)


class ResponseSampleRateTests(unittest.TestCase):
    def test_deployed_unit_returns_audio2face_rate(self) -> None:
        self.assertEqual(unit_environment("ace-indextts-tts.service")["INDEXTTS_TTS_RESPONSE_SAMPLE_RATE_HZ"], "16000")

    def test_fresh_audio_is_resampled_to_configured_rate(self) -> None:
        synthesizer = synthesizer_with(NumpyIndexTTS(sine(1000, 22050, 1.0)), response_sample_rate_hz=16000)

        pcm, media_type, sample_rate = synthesizer.synthesize("Hello there.")

        self.assertEqual((media_type, sample_rate), ("audio/L16", 16000))
        trailing_silence = 16000 * 500 // 1000 * 2
        self.assertEqual(pcm[-trailing_silence:], bytes(trailing_silence))
        speech = np.frombuffer(pcm[:-trailing_silence], dtype="<i2")
        self.assertAlmostEqual(len(speech), 16000, delta=2)
        spectrum = np.abs(np.fft.rfft(speech.astype(np.float64)))
        self.assertAlmostEqual(np.argmax(spectrum) * 16000 / len(speech), 1000, delta=2)

    def test_resampler_is_band_limited(self) -> None:
        source = sine(10000, 22050, 1.0)

        resampled = np.frombuffer(server.resample_pcm16(source.astype("<i2").tobytes(), 22050, 16000), dtype="<i2")

        # A 10 kHz tone is above the 8 kHz output Nyquist rate; linear interpolation would alias it.
        self.assertLess(np.sqrt(np.mean(resampled.astype(np.float64) ** 2)), 0.01 * 10000 / np.sqrt(2))

    def test_native_rate_is_passed_through_unchanged(self) -> None:
        model = NumpyIndexTTS(sine(1000, 22050, 0.1))
        synthesizer = synthesizer_with(model, response_sample_rate_hz=22050, trailing_silence_ms=0)

        pcm, _media_type, sample_rate = synthesizer.synthesize("Hello there.")

        self.assertEqual(sample_rate, 22050)
        self.assertEqual(pcm, model.samples.astype("<i2").tobytes())

    def test_prewarm_cache_is_stored_at_response_rate(self) -> None:
        model = NumpyIndexTTS(sine(1000, 22050, 0.1))
        synthesizer = synthesizer_with(model, response_sample_rate_hz=16000)
        synthesizer.prewarm()

        first_text = server.sentence_texts(synthesizer.settings.prewarm_text)[0]
        pcm, _media_type, sample_rate = synthesizer.synthesize(first_text)

        self.assertEqual(len(model.calls), 7)
        self.assertEqual(sample_rate, 16000)
        self.assertAlmostEqual(len(pcm) - 16000, 1600 * 2, delta=4)

    def test_http_header_and_health_report_the_same_output_rate(self) -> None:
        settings = server.IndexTtsSettings(
            host="127.0.0.1", port=0, preload_model=False, response_sample_rate_hz=16000
        )
        synthesizer = server.IndexTtsSynthesizer(settings)
        synthesizer.model = NumpyIndexTTS(sine(1000, 22050, 0.1))
        http_server = server.create_server(settings, synthesizer)
        thread = Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{http_server.server_address[1]}"
        try:
            with urllib.request.urlopen(f"{base_url}/healthz") as response:
                health = json.load(response)
            request = urllib.request.Request(
                f"{base_url}/v1/audio/speech",
                data=json.dumps({"input": "Hello there.", "voice": "kagawa_en", "stream": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                response.read()
                header_rate = response.headers["X-Sample-Rate-Hz"]
        finally:
            http_server.shutdown()
            http_server.server_close()
            thread.join(timeout=5)

        self.assertEqual((health["sample_rate_hz"], header_rate), (16000, "16000"))


class PronunciationSubstitutionTests(unittest.TestCase):
    def test_the_is_left_for_the_model_to_pronounce(self) -> None:
        text = "the university, the hour and the NVIDIA lab"

        self.assertEqual(server.apply_pronunciation_substitutions(text), text)

    def test_substitutions_ignore_case_and_keep_the_original_spelling(self) -> None:
        cases = {
            "Our supercomputer is called SEIRAN.": f"Our <supercomputer|{SUPERCOMPUTER}> is called <SEIRAN|{SEIRAN}>.",
            "Supercomputers like Seiran run AI.": (
                f"<Supercomputers|{SUPERCOMPUTER} Z> like <Seiran|{SEIRAN}> run AI."
            ),
            "Supercomputer Seiran is in Hachiōji.": (
                f"<Supercomputer|{SUPERCOMPUTER}> <Seiran|{SEIRAN}> is in <Hachiōji|{HACHIOJI}>."
            ),
            "HACHIOJI, Tokyo.": f"<HACHIOJI|{HACHIOJI}>, Tokyo.",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(server.apply_pronunciation_substitutions(text), expected)


class TextNormalizationTests(unittest.TestCase):
    def test_english_is_normalized_after_pronunciation_tags_are_added(self) -> None:
        normalizer = FakeTextNormalizer({"1986": "nineteen eighty six"})
        model = NumpyIndexTTS(text_process=normalizer)

        synthesizer_with(model).synthesize("Hachioji was founded in 1986.")

        self.assertEqual(normalizer.inputs, [f"<Hachioji|{HACHIOJI}> was founded in 1986."])
        self.assertEqual(model.calls[0]["text"], f"<Hachioji|{HACHIOJI}> was founded in nineteen eighty six.")
        self.assertFalse(model.calls[0]["text_normalization"])

    def test_text_the_normalizer_would_treat_as_chinese_is_sent_raw(self) -> None:
        for text, chinese in (("2026.", False), ("東京工科大学は1986年に設立されました。", False), ("Win10 in 2020.", True)):
            with self.subTest(text=text):
                normalizer = FakeTextNormalizer({"20": "two zero"}, chinese=chinese)
                model = NumpyIndexTTS(text_process=normalizer)

                synthesizer_with(model).synthesize(text)

                self.assertEqual(normalizer.inputs, [])
                self.assertEqual(model.calls[0]["text"], text)

    def test_sentence_end_dropped_by_the_normalizer_is_restored(self) -> None:
        normalizer = FakeTextNormalizer({"10:00 a.m.": "ten AM"})
        model = NumpyIndexTTS(text_process=normalizer)

        synthesizer_with(model).synthesize("The lab opened at 10:00 a.m.")

        self.assertEqual(model.calls[0]["text"], "The lab opened at ten AM.")

    def test_domain_terms_the_normalizer_misreads_are_spelled_out_first(self) -> None:
        normalizer = FakeTextNormalizer()
        model = NumpyIndexTTS(text_process=normalizer)

        synthesizer_with(model).synthesize("Prof. Kagawa says Seiran runs 24/7.")

        self.assertEqual(normalizer.inputs, [f"Professor Kagawa says <Seiran|{SEIRAN}> runs twenty-four seven."])

    def test_normalizer_input_dodges_its_contraction_and_thousands_misreadings(self) -> None:
        cases = {
            # The normalizer's contraction rule would read these 's as "is".
            "It's been an honor.": "It’s been an honor.",
            "IT'S BEEN AN HONOR.": "IT’S BEEN AN HONOR.",
            "What's the summit's main theme?": "What’s the summit’s main theme?",
            "Let's go, we're here, don't stop.": "Let’s go, we're here, don't stop.",
            # Bare 1,000-1,999 would lose their "one".
            "We have 1,500 researchers.": "We have one thousand 500 researchers.",
            "It has 1,000 trees, (1,050) desks and 1,001, and more.": (
                "It has one thousand trees, (one thousand 50) desks and one thousand 1, and more."
            ),
            # Measures, money, ordinals, decimals and larger numbers are already read correctly.
            "It is 1,000 km, 1,001 kg, 1,000°C, 1,000 GB, 1,500 MW and 1,500%.": (
                "It is 1,000 km, 1,001 kg, 1,000°C, 1,000 GB, 1,500 MW and 1,500%."
            ),
            "It costs $1,500 or €1,500 for the 1,000th of 11,500 or 1,500,000 seats.": (
                "It costs $1,500 or €1,500 for the 1,000th of 11,500 or 1,500,000 seats."
            ),
            "It is 1,000.5 or -1,500 for a 1,000-year-old tree.": "It is 1,000.5 or -1,500 for a 1,000-year-old tree.",
            # A bare 1000 would be read as the year "ten hundred".
            "The campus has 1000 trees, (1000) desks and a 1000-year-old gate.": (
                "The campus has one thousand trees, (one thousand) desks and a one thousand-year-old gate."
            ),
            "It is 1000 km, 1000%, the 1000th of 1000s, $1000, 1000.5, 10000 and 1000-2000 in 1999.": (
                "It is 1000 km, 1000%, the 1000th of 1000s, $1000, 1000.5, 10000 and 1000-2000 in 1999."
            ),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                normalizer = FakeTextNormalizer()

                server.normalize_english_text(text, normalizer)

                self.assertEqual(normalizer.inputs, [expected])

    def test_text_that_skips_the_normalizer_is_returned_unchanged(self) -> None:
        empty = types.SimpleNamespace(use_chinese=lambda text: False, normalize=lambda text: " ")
        self.assertEqual(server.normalize_english_text("It's been 1,500 days.", empty), "It's been 1,500 days.")
        chinese = FakeTextNormalizer(chinese=True)
        self.assertEqual(server.normalize_english_text("Win10 has 1,500 users.", chinese), "Win10 has 1,500 users.")

    @unittest.skipUnless((INDEXTTS_REPO_DIR / "indextts/utils/front.py").is_file(), "IndexTTS repo not installed")
    def test_real_index_tts_normalizer_expands_numbers_and_keeps_the_script(self) -> None:
        dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # the IndexTTS repo under /data is read-only for tests
        try:
            if str(INDEXTTS_REPO_DIR) not in sys.path:
                sys.path.insert(0, str(INDEXTTS_REPO_DIR))
            from indextts.utils.front import TextNormalizer

            normalizer = TextNormalizer(enable_glossary=True)
            normalizer.load()
        except ImportError as exc:
            self.skipTest(f"IndexTTS normalizer unavailable: {exc}")
        finally:
            sys.dont_write_bytecode = dont_write_bytecode

        self.assertEqual(
            server.normalize_english_text("The symposium starts at 1:30 p.m. on October 8, 2026.", normalizer),
            "The symposium starts at one thirty PM on the eighth of october , twenty twenty six.",
        )
        self.assertEqual(server.normalize_english_text("Seiran runs 24/7.", normalizer), "Seiran runs twenty-four seven.")
        self.assertEqual(server.normalize_english_text("2026.", normalizer), "2026.")
        cases = {
            # 's stays as written, also for "has" and possessives.
            "It's been an honor to welcome you here today.": "It's been an honor to welcome you here today.",
            "IT'S BEEN AN HONOR.": "IT'S BEEN AN HONOR.",
            "There's been a lot of progress since 2019.": "There's been a lot of progress since twenty nineteen.",
            "What's the summit's main theme?": "What's the summit's main theme?",
            "MIT's research is impressive.": "MIT's research is impressive.",
            "He's got a point.": "He's got a point.",
            "Hachioji's campus is green.": f"<Hachioji|{HACHIOJI}>'s campus is green.",
            "Let's meet in the '90s at 5 o'clock.": "Let's meet in the nineties at five o'clock.",
            # Bare 1,000-1,999 keep their "one".
            "We have about 1,500 researchers.": "We have about one thousand five hundred researchers.",
            "The campus has 1,000 trees.": "The campus has one thousand trees.",
            "It has 1,050 rooms and 1,001 desks.": "It has one thousand fifty rooms and one thousand one desks.",
            "The lab has 1,024 nodes.": "The lab has one thousand twenty four nodes.",
            "We have 1,500, and more.": "We have one thousand five hundred, and more.",
            "The count (1,200) is right.": "The count 'one thousand two hundred' is right.",
            "From 1,500 to 2,000 people.": "From one thousand five hundred to two thousand people.",
            "The campus has 1000 trees.": "The campus has one thousand trees.",
            # "plus" survives the thousand rewrite.
            "We have 1,000+ papers and 1000+ alumni.": "We have one thousand plus papers and one thousand plus alumni.",
            "From 1000 to 2000, in 1999 and 2026.": (
                "From one thousand to two thousand, in nineteen ninety nine and twenty twenty six."
            ),
            # Readings that were already right stay right.
            "It costs €1,500, £1,500, ¥1,500 or $1,500.": (
                "It costs one thousand five hundred euros, one thousand five hundred pounds,"
                " one thousand five hundred yen or one thousand five hundred dollars."
            ),
            "It reached 1,400°C and 1,000°C.": (
                "It reached one thousand four hundred degrees Celsius and one thousand degrees Celsius."
            ),
            "It is 1,000 km and 1,001 km.": "It is one thousand kilometers and one thousand and one kilometers.",
            "It ran 1000 km.": "It ran one thousand kilometers.",
            "It stores 1,000 GB.": "It stores one thousand gigabytes.",
            "We have 11,500 rooms and 1,500,000 books.": (
                "We have eleven thousand five hundred rooms and one million five hundred thousand books."
            ),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                tagged = server.apply_pronunciation_substitutions(text)
                self.assertEqual(server.normalize_english_text(tagged, normalizer), expected)
        for sentence in server.sentence_texts(server.DEFAULT_PREWARM_TEXT):
            tagged = server.apply_pronunciation_substitutions(sentence)
            with self.subTest(sentence=sentence):
                self.assertEqual(server.normalize_english_text(tagged, normalizer), tagged)


class DisconnectTests(unittest.TestCase):
    def test_cached_sentence_does_not_wait_for_a_running_synthesis(self) -> None:
        model = BlockingIndexTTS()
        synthesizer = synthesizer_with(model)
        synthesizer.prewarmed_pcm["Good afternoon, everyone."] = b"\x01\x00"
        synthesizer.prewarmed_sample_rates["Good afternoon, everyone."] = 22050
        running = Thread(target=synthesizer.synthesize, args=("An uncached sentence.",), daemon=True)
        running.start()
        try:
            self.assertTrue(model.entered.wait(timeout=5))
            cached = Thread(target=synthesizer.synthesize, args=("Good afternoon, everyone.",), daemon=True)
            cached.start()
            cached.join(timeout=1)
            self.assertFalse(cached.is_alive())
        finally:
            model.release.set()
            running.join(timeout=5)

    def test_client_disconnect_is_logged_once_without_raising(self) -> None:
        for stream in (True, False):
            with self.subTest(stream=stream):
                settings = server.IndexTtsSettings(preload_model=False)
                synthesizer = server.IndexTtsSynthesizer(settings)
                synthesizer.model = NumpyIndexTTS()
                messages: list[str] = []

                class Handler(server._TtsRequestHandler):
                    def log_message(self, format, *args):
                        messages.append(format % args)

                body = json.dumps({"input": "Hello there.", "voice": "kagawa_en", "stream": stream}).encode()
                server_side, client_side = socket.socketpair()
                client_side.sendall(
                    b"POST /v1/audio/speech HTTP/1.1\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                client_side.close()
                try:
                    Handler(server_side, ("127.0.0.1", 0), types.SimpleNamespace(settings=settings, synthesizer=synthesizer))
                finally:
                    server_side.close()

                self.assertEqual(len([message for message in messages if "disconnected" in message]), 1)


class ServiceConfigTests(unittest.TestCase):
    def test_indextts_installer_starts_without_enabling_at_boot(self) -> None:
        script = (ROOT / "install_indextts_tts_service.sh").read_text(encoding="utf-8")

        self.assertIn('sudo systemctl start "${SERVICE_NAME}"', script)
        self.assertNotIn("systemctl enable", script)

    def test_qwen_comparison_backend_does_not_share_the_elevenlabs_port(self) -> None:
        self.assertEqual(qwen_server.QwenTtsSettings.from_env({}).port, 8025)
        self.assertEqual(unit_environment("ace-qwen-tts.service")["QWEN_TTS_PORT"], "8025")


if __name__ == "__main__":
    unittest.main()
