from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest import mock

import numpy as np

try:
    import scipy.signal as scipy_signal
except ModuleNotFoundError:
    scipy_signal = None

try:
    import fastapi
    from fastapi.testclient import TestClient
except (ModuleNotFoundError, RuntimeError):
    fastapi = None
    TestClient = None


ROOT = Path(__file__).resolve().parents[2]
NO_SCIPY = {"scipy": None, "scipy.signal": None}
GAIN = 10 ** (-0.5 / 20)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


irodori_server = load_module("irodori_server", ROOT / "infra" / "tts" / "irodori_server.py")


def tone(freq_hz: float, seconds: float, sample_rate: int, amplitude: float) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def noise(seconds: float, sample_rate: int, level_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(round(seconds * sample_rate))
    return (rng.standard_normal(count) * 10 ** (level_db / 20)).astype(np.float32)


def padded_speech(sample_rate: int) -> np.ndarray:
    """0.25 s noise bed, 0.5 s speech, 0.45 s internal pause, 0.3 s speech, 0.45 s noise bed."""
    return np.concatenate(
        [
            noise(0.25, sample_rate, -45, 1),
            tone(440, 0.5, sample_rate, 0.25) + noise(0.5, sample_rate, -45, 2),
            noise(0.45, sample_rate, -45, 3),
            tone(660, 0.3, sample_rate, 0.25) + noise(0.3, sample_rate, -45, 4),
            noise(0.45, sample_rate, -45, 5),
        ]
    )


def band_noise(
    seconds: float, sample_rate: int, level_db: float, seed: int, low_hz: float, high_hz: float
) -> np.ndarray:
    """Noise limited to [low_hz, high_hz] by an FFT mask (the system Python has no scipy)."""
    count = int(round(seconds * sample_rate))
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1 / sample_rate)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    shaped = np.fft.irfft(spectrum, count)
    return (shaped / np.sqrt(np.mean(shaped**2)) * 10 ** (level_db / 20)).astype(np.float32)


def fricative_speech(sample_rate: int) -> np.ndarray:
    """0.25 s pad, 0.12 s fricative, 0.3 s vowel, 0.2 s devoiced fricative, 0.4 s pad.

    The 3.5-7.5 kHz fricatives sit at -35 dBFS over a -40 dBFS low-frequency floor, below
    the broadband speech threshold like the s/sh/h of real Irodori output.
    """
    speech = np.concatenate(
        [
            np.zeros(int(0.25 * sample_rate), dtype=np.float32),
            band_noise(0.12, sample_rate, -35, 11, 3500, 7500),
            tone(220, 0.3, sample_rate, 0.25),
            band_noise(0.2, sample_rate, -35, 12, 3500, 7500),
            np.zeros(int(0.4 * sample_rate), dtype=np.float32),
        ]
    )
    return speech + band_noise(speech.size / sample_rate, sample_rate, -40, 13, 0, 1000)


def tone_level_db(audio: np.ndarray, sample_rate: int, freq_hz: float) -> float:
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window)) * 2 / window.sum()
    index = int(round(freq_hz * audio.size / sample_rate))
    return float(20 * np.log10(spectrum[index - 2 : index + 3].max() + 1e-12))


def decode(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


class FakeSamplingRequest:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class FakeResult:
    def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
        self.audio = audio
        self.sample_rate = sample_rate


class FakeRuntime:
    def __init__(self, audio: np.ndarray | None = None, sample_rate: int = 48000, error: Exception | None = None):
        self.audio = padded_speech(sample_rate) if audio is None else audio
        self.sample_rate = sample_rate
        self.error = error
        self.texts: list[str] = []

    def synthesize(self, request, log_fn=None):
        self.texts.append(request.text)
        if self.error is not None:
            raise self.error
        return FakeResult(self.audio, self.sample_rate)


@contextmanager
def fake_irodori_package() -> Iterator[None]:
    fake_package = types.ModuleType("irodori_tts")
    fake_inference_runtime = types.ModuleType("irodori_tts.inference_runtime")
    fake_inference_runtime.SamplingRequest = FakeSamplingRequest
    with mock.patch.dict(
        sys.modules,
        {"irodori_tts": fake_package, "irodori_tts.inference_runtime": fake_inference_runtime},
    ):
        yield


def make_synthesizer(runtime: FakeRuntime | None = None, **overrides):
    settings = irodori_server.IrodoriSettings(
        reference_source=Path(__file__),
        reference_wav=Path(__file__),
        **overrides,
    )
    synthesizer = irodori_server.IrodoriSynthesizer(settings)
    synthesizer._runtime = runtime or FakeRuntime()
    return synthesizer


class ResampleTests(unittest.TestCase):
    def assert_band_limited_48k_to_16k(self) -> None:
        alias = decode(
            irodori_server.audio_to_pcm16(
                tone(10000, 1.0, 48000, 0.5),
                source_sample_rate=48000,
                target_sample_rate=16000,
            )
        ) / 32768.0
        passband = decode(
            irodori_server.audio_to_pcm16(
                tone(1000, 1.0, 48000, 0.5),
                source_sample_rate=48000,
                target_sample_rate=16000,
            )
        ) / 32768.0

        # A 10 kHz tone folds to 6 kHz at 16 kHz; it must be filtered out, not aliased.
        self.assertLessEqual(tone_level_db(alias, 16000, 6000) - 20 * np.log10(0.5), -40.0)
        self.assertAlmostEqual(tone_level_db(passband, 16000, 1000), 20 * np.log10(0.5), delta=0.5)
        self.assertEqual(alias.size, 16000)
        odd_length = irodori_server.audio_to_pcm16(
            np.zeros(4801, dtype=np.float32),
            source_sample_rate=48000,
            target_sample_rate=16000,
        )
        self.assertEqual(len(odd_length), 1600 * 2)

    @unittest.skipIf(scipy_signal is None, "scipy is not installed in this Python environment")
    def test_audio_to_pcm16_band_limits_with_scipy(self) -> None:
        self.assert_band_limited_48k_to_16k()

    def test_audio_to_pcm16_band_limits_with_numpy_fallback(self) -> None:
        with mock.patch.dict(sys.modules, NO_SCIPY):
            self.assert_band_limited_48k_to_16k()

    def test_resample_fallback_keeps_round_length_for_non_integer_ratio(self) -> None:
        with mock.patch.dict(sys.modules, NO_SCIPY):
            resampled = irodori_server.resample_audio(tone(440, 1.0, 44100, 0.5)[:44101], 44100, 16000)

        self.assertEqual(resampled.size, 16000)
        self.assertEqual(resampled.dtype, np.float32)

    def test_resample_keeps_single_sample_and_same_rate_input(self) -> None:
        single = np.array([0.25], dtype=np.float32)

        self.assertEqual(irodori_server.resample_audio(single, 48000, 16000).size, 1)
        self.assertIs(irodori_server.resample_audio(single, 16000, 16000), single)


class TrimTests(unittest.TestCase):
    def test_trim_keeps_lead_and_sentence_or_clause_trail_around_speech(self) -> None:
        clip = padded_speech(16000)

        sentence = irodori_server.trim_speech_padding(clip, sample_rate=16000, lead_ms=60, trail_ms=300)
        clause = irodori_server.trim_speech_padding(clip, sample_rate=16000, lead_ms=60, trail_ms=120)

        # Speech runs from 4000 to 24000 samples; the internal pause stays because only the ends are cut.
        np.testing.assert_array_equal(sentence, clip[4000 - 960 : 24000 + 4800])
        np.testing.assert_array_equal(clause, clip[4000 - 960 : 24000 + 1920])

    def test_trim_never_pads_and_leaves_clips_without_speech_frames_alone(self) -> None:
        short_tail = np.concatenate([noise(0.2, 16000, -45, 6), tone(440, 0.3, 16000, 0.25)])
        flat_noise = noise(0.5, 16000, -45, 7)
        silence = np.zeros(1600, dtype=np.float32)
        single = np.array([0.0], dtype=np.float32)

        trimmed = irodori_server.trim_speech_padding(short_tail, sample_rate=16000, lead_ms=60, trail_ms=300)

        np.testing.assert_array_equal(trimmed, short_tail[3200 - 960 :])
        for clip in (flat_noise, silence, single):
            with self.subTest(size=clip.size):
                result = irodori_server.trim_speech_padding(clip, sample_rate=16000, lead_ms=60, trail_ms=300)
                np.testing.assert_array_equal(result, clip)

    def test_trim_keeps_quiet_fricatives_at_both_ends(self) -> None:
        clip = fricative_speech(16000)

        trimmed = irodori_server.trim_speech_padding(clip, sample_rate=16000, lead_ms=60, trail_ms=120)

        # The fricatives run from 4000 to 5920 and from 10720 to 13920 samples. Broadband energy
        # alone starts speech at the vowel (5920) and ends it at 10720, inside the fricatives.
        np.testing.assert_array_equal(trimmed, clip[4000 - 960 : 13920 + 1920])

    def test_trim_does_not_follow_isolated_high_frequency_bursts_in_the_pad(self) -> None:
        clip = fricative_speech(16000)
        burst = band_noise(0.02, 16000, -40, 14, 3500, 7500)
        clip[800:1120] += burst
        clip[17600:17920] += burst

        trimmed = irodori_server.trim_speech_padding(clip, sample_rate=16000, lead_ms=60, trail_ms=120)

        np.testing.assert_array_equal(trimmed, clip[4000 - 960 : 13920 + 1920])


class FadeAndGainTests(unittest.TestCase):
    def test_edge_fades_are_raised_cosine_and_clamped_to_clip_length(self) -> None:
        faded = irodori_server.apply_edge_fades(np.ones(1000, dtype=np.float32), sample_rate=16000, fade_ms=10)

        self.assertEqual(faded[0], 0.0)
        self.assertEqual(faded[-1], 0.0)
        self.assertAlmostEqual(float(faded[80]), 0.5, places=6)
        self.assertTrue(np.all(np.diff(faded[:160]) > 0))
        np.testing.assert_array_equal(faded[160:-160], 1.0)
        np.testing.assert_array_equal(
            irodori_server.apply_edge_fades(np.ones(3, dtype=np.float32), sample_rate=16000, fade_ms=10),
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_array_equal(
            irodori_server.apply_edge_fades(np.array([0.5], dtype=np.float32), sample_rate=16000, fade_ms=10),
            [0.5],
        )

    def test_audio_to_pcm16_rounds_instead_of_truncating(self) -> None:
        audio = np.array([1.6 / 32767, -1.6 / 32768, 0.4 / 32767], dtype=np.float32)

        pcm = irodori_server.audio_to_pcm16(audio, source_sample_rate=16000, target_sample_rate=16000)

        self.assertEqual(decode(pcm).tolist(), [2, -2, 0])

    def test_synthesized_audio_gets_headroom_so_codec_overshoot_does_not_clip(self) -> None:
        overshoot = tone(440, 1.0, 48000, 1.003)
        with fake_irodori_package():
            synthesizer = make_synthesizer(FakeRuntime(overshoot))
            pcm, _media_type, _sample_rate = synthesizer.synthesize("香川豊です。", response_format="pcm")

        samples = decode(pcm)
        self.assertEqual(int(np.sum((samples == 32767) | (samples == -32768))), 0)
        self.assertAlmostEqual(20 * np.log10(np.abs(samples).max() / 32767), -0.5 + 20 * np.log10(1.003), delta=0.1)


class SynthesizerPostprocessTests(unittest.TestCase):
    def test_synthesized_speech_is_trimmed_by_sentence_end_and_faded(self) -> None:
        with fake_irodori_package():
            synthesizer = make_synthesizer()
            sentence, _media_type, sample_rate = synthesizer.synthesize("香川豊です。", response_format="pcm")
            clause, _media_type, _sample_rate = synthesizer.synthesize("確認しています", response_format="pcm")

        self.assertEqual(sample_rate, 16000)
        for pcm, trail_samples in ((sentence, 4800), (clause, 1920)):
            samples = decode(pcm)
            with self.subTest(trail_samples=trail_samples):
                self.assertLessEqual(abs(samples.size - (960 + 20000 + trail_samples)), 160)
                self.assertEqual(samples[0], 0)
                self.assertEqual(samples[-1], 0)

    def test_trim_fade_and_gain_follow_settings_and_env(self) -> None:
        defaults = irodori_server.IrodoriSettings.from_env({})
        configured = irodori_server.IrodoriSettings.from_env(
            {
                "IRODORI_TTS_TRIM_ENABLED": "false",
                "IRODORI_TTS_TRIM_LEAD_MS": "40",
                "IRODORI_TTS_SENTENCE_TRAIL_MS": "250",
                "IRODORI_TTS_CLAUSE_TRAIL_MS": "100",
                "IRODORI_TTS_FADE_MS": "0",
                "IRODORI_TTS_OUTPUT_GAIN_DB": "0",
            }
        )

        self.assertEqual(
            (
                defaults.trim_enabled,
                defaults.trim_lead_ms,
                defaults.sentence_trail_ms,
                defaults.clause_trail_ms,
                defaults.fade_ms,
                defaults.output_gain_db,
            ),
            (True, 60.0, 300.0, 120.0, 10.0, -0.5),
        )
        self.assertEqual(
            (
                configured.trim_enabled,
                configured.trim_lead_ms,
                configured.sentence_trail_ms,
                configured.clause_trail_ms,
                configured.fade_ms,
                configured.output_gain_db,
            ),
            (False, 40.0, 250.0, 100.0, 0.0, 0.0),
        )
        with fake_irodori_package():
            synthesizer = make_synthesizer(
                FakeRuntime(padded_speech(16000), sample_rate=16000),
                trim_enabled=False,
                fade_ms=0.0,
                output_gain_db=0.0,
            )
            pcm, _media_type, _sample_rate = synthesizer.synthesize("香川豊です。", response_format="pcm")
        np.testing.assert_array_equal(
            decode(pcm),
            decode(irodori_server.audio_to_pcm16(padded_speech(16000), source_sample_rate=16000, target_sample_rate=16000)),
        )

    def test_stream_chunk_bytes_must_be_positive(self) -> None:
        for raw in ("0", "-2"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                irodori_server.IrodoriSettings.from_env({"IRODORI_TTS_STREAM_CHUNK_BYTES": raw})

    def test_short_cache_stores_processed_pcm(self) -> None:
        with fake_irodori_package():
            runtime = FakeRuntime()
            synthesizer = make_synthesizer(runtime, short_cache_enabled=True)
            first = synthesizer.synthesize("確認しています", response_format="pcm")
            second = synthesizer.synthesize("確認しています", response_format="pcm")

        self.assertEqual(runtime.texts, ["確認しています"])
        self.assertIs(first, second)
        self.assertEqual(decode(first[0])[0], 0)

    def test_prewarmed_phrases_survive_short_cache_eviction(self) -> None:
        live_texts = ("はい。", "そうです。", "いいえ。")
        with fake_irodori_package():
            runtime = FakeRuntime()
            synthesizer = make_synthesizer(
                runtime,
                short_cache_enabled=True,
                short_cache_max_entries=2,
                short_cache_prewarm_texts=("確認しています",),
            )
            synthesizer.prewarm_short_cache(("確認しています",))
            for text in live_texts:
                synthesizer.synthesize(text, response_format="pcm")
            ready = synthesizer.lookup_ready_audio("確認しています", response_format="pcm")
            synthesizer.synthesize("確認しています", response_format="pcm")

        # The endpoint check and the filler stay GPU-free however much live speech went through.
        self.assertIsNotNone(ready)
        self.assertEqual(runtime.texts, ["確認しています", *live_texts])
        self.assertIsNone(synthesizer.lookup_ready_audio("はい。", response_format="pcm"))
        self.assertEqual(synthesizer.short_cache_size(), 3)


class FixedGreetingTests(unittest.TestCase):
    def write_greeting(self, directory: str, samples: np.ndarray) -> Path:
        path = Path(directory) / "greeting.pcm"
        path.write_bytes(samples.astype("<i2").tobytes())
        return path

    def greeting_samples(self) -> np.ndarray:
        audio = np.concatenate(
            [
                noise(0.3, 16000, -45, 8),
                tone(440, 0.6, 16000, 0.25) + noise(0.6, 16000, -45, 9),
                noise(0.6, 16000, -45, 10),
            ]
        )
        return np.round(audio * 32767)

    def test_greeting_is_served_through_trim_fade_and_gain(self) -> None:
        raw = self.greeting_samples()
        with tempfile.TemporaryDirectory() as tmp:
            synthesizer = make_synthesizer(fixed_open_campus_greeting_pcm=self.write_greeting(tmp, raw))
            pcm, media_type, sample_rate = synthesizer.synthesize("皆さん、こんにちは。", response_format="pcm")

        samples = decode(pcm)
        self.assertEqual((media_type, sample_rate), ("audio/L16", 16000))
        self.assertEqual(synthesizer._runtime.texts, [])
        # "皆さん、こんにちは" has no final 。 after normalization, so it keeps the clause trail.
        self.assertLessEqual(abs(samples.size - (960 + 9600 + 1920)), 160)
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[-1], 0)
        self.assertLessEqual(abs(int(np.abs(samples).max()) - np.abs(raw).max() * GAIN), 2)

    def test_greeting_is_resampled_only_when_the_response_rate_differs(self) -> None:
        raw = np.array([0, 32767, -32768, 1000, -1000, 12345], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = self.write_greeting(tmp, raw)
            untouched = make_synthesizer(
                fixed_open_campus_greeting_pcm=raw_path,
                trim_enabled=False,
                fade_ms=0.0,
                output_gain_db=0.0,
            ).synthesize("皆さん、こんにちは", response_format="pcm")
            greeting_path = self.write_greeting(tmp, self.greeting_samples())
            at_16k = make_synthesizer(fixed_open_campus_greeting_pcm=greeting_path).synthesize("皆さん、こんにちは")
            at_24k = make_synthesizer(
                fixed_open_campus_greeting_pcm=greeting_path,
                response_sample_rate_hz=24000,
            ).synthesize("皆さん、こんにちは")

        self.assertEqual(untouched[0], raw.astype("<i2").tobytes())
        self.assertEqual(at_24k[2], 24000)
        self.assertAlmostEqual(len(at_24k[0]) / 2 / 24000, len(at_16k[0]) / 2 / 16000, delta=0.011)

    def test_invalid_greeting_file_falls_back_to_synthesis(self) -> None:
        for content in (b"", b"\x01\x02\x03"):
            with self.subTest(size=len(content)), tempfile.TemporaryDirectory() as tmp, fake_irodori_package():
                path = Path(tmp) / "greeting.pcm"
                path.write_bytes(content)
                synthesizer = make_synthesizer(fixed_open_campus_greeting_pcm=path)

                with self.assertLogs("irodori_server", level="WARNING"):
                    pcm, _media_type, _sample_rate = synthesizer.synthesize("皆さん、こんにちは。")

                self.assertEqual(synthesizer._runtime.texts, ["皆さん、こんにちは"])
                self.assertTrue(pcm)


class SpeakableTextTests(unittest.TestCase):
    def test_unspeakable_text_returns_empty_audio_without_the_runtime(self) -> None:
        with fake_irodori_package():
            synthesizer = make_synthesizer()
            for text in ("！", "？", "……", "……。", "「」。"):
                with self.subTest(text=text):
                    self.assertEqual(synthesizer.synthesize(text, response_format="pcm"), (b"", "audio/L16", 16000))
            wav, media_type, _sample_rate = synthesizer.synthesize("！", response_format="wav")

        self.assertEqual(synthesizer._runtime.texts, [])
        self.assertEqual(media_type, "audio/wav")
        self.assertEqual(wav, irodori_server.pcm16_to_wav_bytes(b"", sample_rate_hz=16000))

    def test_text_that_normalizes_to_nothing_returns_empty_audio(self) -> None:
        with fake_irodori_package():
            synthesizer = make_synthesizer()
            for text in ("https://www.teu.ac.jp/", "（https://www.teu.ac.jp/）。"):
                with self.subTest(text=text):
                    self.assertEqual(synthesizer.synthesize(text, response_format="pcm"), (b"", "audio/L16", 16000))
            for text in ("", "   "):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    synthesizer.synthesize(text, response_format="pcm")

        self.assertEqual(synthesizer._runtime.texts, [])

    def test_width_folded_latin_kana_and_kanji_zero_are_speakable(self) -> None:
        with fake_irodori_package():
            synthesizer = make_synthesizer()
            for text in ("ＡＩ", "ｶﾀｶﾅ", "〇"):
                synthesizer.synthesize(text, response_format="pcm")

        # normalize_speech_input folds full-width Latin before synthesis.
        self.assertEqual(synthesizer._runtime.texts, ["AI", "ｶﾀｶﾅ", "〇"])


class MoraEstimateTests(unittest.TestCase):
    def test_estimate_speech_mora_reads_letter_names_and_width_variants(self) -> None:
        cases = {
            "AI": 4.0,
            "ＡＩ": 4.0,
            "GPU": 6.0,
            "DX": 6.0,
            "WWW": 12.0,
            "HRZ": 9.0,
            "Web": 6.0,
            "2026": 6.0,
            "〇": 2.0,
            "ｶﾀｶﾅ": 4.0,
            "ｶﾞｯｺｳ": 4.0,
            "ｷｬﾝﾊﾟｽ": 4.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(irodori_server.estimate_speech_mora(text), expected)


class ProgressiveStreamTests(unittest.TestCase):
    def test_progressive_stream_serves_prewarmed_whole_text_before_splitting(self) -> None:
        with fake_irodori_package():
            runtime = FakeRuntime()
            synthesizer = make_synthesizer(runtime, short_cache_enabled=True)
            synthesizer.prewarm_short_cache(("こんにちは。ご用件をどうぞ。",))
            cached, _media_type, _sample_rate = synthesizer.synthesize("こんにちは。ご用件をどうぞ。")

            streamed = b"".join(synthesizer.iter_pcm_stream("こんにちは。ご用件をどうぞ。", chunk_bytes=3200))

        self.assertEqual(runtime.texts, ["こんにちは。ご用件をどうぞ。"])
        self.assertEqual(streamed, cached)

    def test_progressive_response_logs_failures_after_headers_are_sent(self) -> None:
        class FailingSynthesizer:
            def iter_pcm_stream(self, text: str, *, chunk_bytes: int):
                yield b"\x01\x00"
                raise RuntimeError("CUDA out of memory")

        body = irodori_server.iter_progressive_pcm_response(FailingSynthesizer(), "香川豊です。", chunk_bytes=4)

        with self.assertLogs("irodori_server", level="ERROR") as logs, self.assertRaises(RuntimeError):
            list(body)
        self.assertIn("CUDA out of memory", "\n".join(logs.output))


def asgi_post(app, payload: dict[str, object], *, disconnect: bool = False):
    body = json.dumps(payload).encode()
    pending = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict[str, object]] = []

    async def receive():
        if pending:
            return pending.pop(0)
        if disconnect:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    async def run() -> tuple[int, bytes]:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/speech",
            "raw_path": b"/v1/audio/speech",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8021),
        }
        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        content = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return status, content

    return run()


@unittest.skipIf(fastapi is None, "fastapi is not installed in this Python environment")
class SpeechEndpointTests(unittest.TestCase):
    def make_app(self, runtime: FakeRuntime | None = None, **overrides):
        synthesizer = make_synthesizer(runtime, **overrides)
        return irodori_server.create_app(synthesizer._settings, synthesizer), synthesizer

    def test_cache_hit_does_not_wait_for_the_synthesis_lock(self) -> None:
        with fake_irodori_package():
            app, synthesizer = self.make_app(short_cache_enabled=True)
            cached, _media_type, _sample_rate = synthesizer.synthesize("確認しています")

            async def scenario() -> tuple[int, bytes]:
                lock = asyncio.Lock()
                app.state.synthesis_lock = lock
                await lock.acquire()
                try:
                    return await asyncio.wait_for(asgi_post(app, {"input": "確認しています"}), timeout=2.0)
                finally:
                    lock.release()

            status, body = asyncio.run(scenario())

        self.assertEqual(status, 200)
        self.assertEqual(body, cached)
        self.assertEqual(synthesizer._runtime.texts, ["確認しています"])

    def test_queued_request_from_a_disconnected_client_is_not_synthesized(self) -> None:
        with fake_irodori_package():
            app, synthesizer = self.make_app()
            status, body = asyncio.run(asgi_post(app, {"input": "香川豊です。"}, disconnect=True))

        self.assertEqual(synthesizer._runtime.texts, [])
        self.assertEqual(body, b"")
        self.assertNotEqual(status, 200)

    def test_stream_true_without_progressive_mode(self) -> None:
        with fake_irodori_package():
            app, synthesizer = self.make_app(stream_chunk_bytes=3200)
            client = TestClient(app)

            speech = client.post("/v1/audio/speech", json={"input": "香川豊です。", "stream": True})
            empty = client.post("/v1/audio/speech", json={"input": "   ", "stream": True})
            period = client.post("/v1/audio/speech", json={"input": "。", "stream": True})
            punctuation = client.post("/v1/audio/speech", json={"input": "！", "stream": True})
            expected, _media_type, _sample_rate = synthesizer.synthesize("香川豊です。")

        self.assertEqual(speech.status_code, 200)
        self.assertEqual(speech.headers["x-sample-rate-hz"], "16000")
        self.assertEqual(speech.content, expected)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual((period.status_code, period.content), (200, b""))
        self.assertEqual((punctuation.status_code, punctuation.content), (200, b""))
        self.assertEqual(synthesizer._runtime.texts, ["香川ゆたかです。", "香川ゆたかです。"])

    def test_stream_true_with_progressive_mode(self) -> None:
        with fake_irodori_package():
            app, synthesizer = self.make_app(progressive_stream_enabled=True)
            client = TestClient(app)

            speech = client.post(
                "/v1/audio/speech",
                json={"input": "香川豊です。材料強度を研究しています。", "stream": True},
            )
            empty = client.post("/v1/audio/speech", json={"input": "   ", "stream": True})
            period = client.post("/v1/audio/speech", json={"input": "。", "stream": True})
            punctuation = client.post("/v1/audio/speech", json={"input": "！", "stream": True})
            first, _media_type, _sample_rate = synthesizer.synthesize("香川豊です。")
            second, _media_type, _sample_rate = synthesizer.synthesize("材料強度を研究しています。")

        self.assertEqual(speech.status_code, 200)
        self.assertEqual(speech.content, first + second)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual((period.status_code, period.content), (200, b""))
        self.assertEqual((punctuation.status_code, punctuation.content), (200, b""))
        self.assertEqual(
            synthesizer._runtime.texts[:2],
            ["香川ゆたかです。", "材料強度を研究しています。"],
        )


if __name__ == "__main__":
    unittest.main()
