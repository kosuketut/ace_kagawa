from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

try:
    import fastapi
except ModuleNotFoundError:
    fastapi = None


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


irodori_server = load_module("irodori_server", ROOT / "infra" / "tts" / "irodori_server.py")

# Speaks the `speech worker` JSON-lines protocol of speech.cpp 0.8.7: one ready line, then for each
# request base64 48 kHz s16le chunks and an end line. Text containing FAIL answers with an error,
# DIE exits mid-request. Every argv and request is appended to FAKE_SPEECH_LOG.
FAKE_SPEECH = textwrap.dedent(
    """\
    import base64, json, math, os, struct, sys, time
    log = open(os.environ["FAKE_SPEECH_LOG"], "a")
    log.write(json.dumps({"argv": sys.argv[1:]}) + "\\n"); log.flush()
    if sys.argv[1] != "worker":
        sys.exit(2)
    print(json.dumps({"type": "ready", "version": "0.8.7"}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        log.write(json.dumps({"request": request}, ensure_ascii=False) + "\\n"); log.flush()
        if "DIE" in request["text"]:
            sys.exit(3)
        if "FAIL" in request["text"]:
            print(json.dumps({"type": "error", "id": request["id"],
                              "error": {"code": "out_of_range", "message": "too long"}}), flush=True)
            continue
        time.sleep(0.02)
        samples = 24000
        pcm = b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * 220 * i / 48000))) for i in range(samples))
        for start in range(0, len(pcm), 16000):
            print(json.dumps({"type": "chunk", "id": request["id"],
                              "pcm": base64.b64encode(pcm[start:start + 16000]).decode()}), flush=True)
        print(json.dumps({"type": "end", "id": request["id"], "seed": 7, "samples": samples,
                          "stop": "complete"}), flush=True)
    """
)


class SpeechCppBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.log = root / "speech.log"
        self.speech = root / "speech"
        self.speech.write_text(f"#!{sys.executable}\n" + FAKE_SPEECH)
        self.speech.chmod(0o755)
        self.model = root / "Irodori-TTS-859M-v4.1-F16.gguf"
        self.model.write_bytes(b"gguf")
        self.voice = root / "kagawa.voice.gguf"
        self.voice.write_bytes(b"voice")
        env = mock.patch.dict(os.environ, {"FAKE_SPEECH_LOG": str(self.log)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)
        self.synthesizers: list[object] = []

    def settings(self, **overrides):
        values = {
            "IRODORI_TTS_BACKEND": "speechcpp",
            "IRODORI_TTS_SPEECHCPP_BIN": str(self.speech),
            "IRODORI_TTS_SPEECHCPP_MODEL": str(self.model),
            "IRODORI_TTS_SPEECHCPP_VOICE": str(self.voice),
            "IRODORI_TTS_SPEECHCPP_DEVICE": "Vulkan1",
            "IRODORI_TTS_DATA_ROOT": self.tmp.name,
        }
        values.update(overrides)
        return irodori_server.IrodoriSettings.from_env(values)

    def synthesizer(self, settings=None, **kwargs):
        synth = irodori_server.IrodoriSynthesizer(settings or self.settings(), **kwargs)
        self.addCleanup(synth.close)
        return synth

    def entries(self, key: str) -> list[object]:
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [row[key] for row in rows if key in row]

    def test_settings_default_to_python_backend(self) -> None:
        settings = irodori_server.IrodoriSettings.from_env({})

        self.assertEqual(settings.backend, "python")
        self.assertEqual(settings.speechcpp_bin, Path("/data/ACE/speech-cpp/runtime/v0.8.7/speech"))
        self.assertEqual(settings.speechcpp_model, Path("/data/ACE/speech-cpp/models/Irodori-TTS-859M-v4.1-F16.gguf"))
        self.assertEqual(
            settings.speechcpp_voice,
            Path("/data/ACE/irodori/speech-cpp/kagawa_voice_ref_48k_mono.voice.gguf"),
        )
        self.assertIsNone(settings.speechcpp_gpu_uuid)
        self.assertEqual(settings.speechcpp_threads, 8)

    def test_settings_read_speechcpp_environment(self) -> None:
        settings = self.settings(
            IRODORI_TTS_SPEECHCPP_GPU_UUID="GPU-abc",
            IRODORI_TTS_SPEECHCPP_THREADS="4",
            IRODORI_TTS_SPEECHCPP_TIMEOUT_SECONDS="12.5",
        )

        self.assertEqual(settings.backend, "speechcpp")
        self.assertEqual(settings.speechcpp_bin, self.speech)
        self.assertEqual(settings.speechcpp_device, "Vulkan1")
        self.assertEqual(settings.speechcpp_gpu_uuid, "GPU-abc")
        self.assertEqual(settings.speechcpp_threads, 4)
        self.assertEqual(settings.speechcpp_timeout_seconds, 12.5)

    def test_settings_reject_unknown_backend_and_lora_with_speechcpp(self) -> None:
        with self.assertRaisesRegex(ValueError, "IRODORI_TTS_BACKEND"):
            irodori_server.IrodoriSettings.from_env({"IRODORI_TTS_BACKEND": "onnx"})
        with self.assertRaisesRegex(ValueError, "LoRA"):
            self.settings(IRODORI_TTS_LORA_ADAPTER="/tmp/adapter.safetensors")

    def test_short_text_gets_the_mora_cap_as_fixed_length(self) -> None:
        synth = self.synthesizer()

        pcm, media_type, sample_rate = synth.synthesize("少々お待ちください。", response_format="pcm")

        normalized = irodori_server.normalize_speech_input("少々お待ちください。")
        (request,) = self.entries("request")
        self.assertEqual(request["type"], "synthesize")
        self.assertEqual(request["voice"], "kagawa")
        self.assertEqual(request["text"], normalized)
        self.assertEqual(request["seconds"], synth._max_seconds_for(normalized))
        self.assertNotIn("duration_scale", request)
        self.assertNotIn("steps", request)
        self.assertNotIn("max_seconds", request)
        self.assertEqual((media_type, sample_rate), ("audio/L16", 16000))
        # 0.5 s of 48 kHz model audio comes back as at most 0.5 s of 16 kHz PCM after trimming.
        self.assertGreater(len(pcm), 0)
        self.assertLessEqual(len(pcm) // 2, 8000)

        (argv,) = self.entries("argv")
        self.assertEqual(argv[:2], ["worker", str(self.model)])
        self.assertEqual(argv[argv.index("--device") + 1], "Vulkan1")
        self.assertEqual(argv[argv.index("--threads") + 1], "8")
        self.assertEqual(argv[argv.index("--add-voice") + 1], f"kagawa={self.voice}")

    def test_long_text_keeps_the_predicted_length_and_passes_steps_and_caption(self) -> None:
        settings = replace(self.settings(IRODORI_TTS_NUM_STEPS="16"), caption="落ち着いた声で")
        synth = self.synthesizer(settings)
        text = "東京工科大学では、専門分野の知識だけでなく、実践的な学びを大切にしています。"

        synth.synthesize(text)

        (request,) = self.entries("request")
        self.assertNotIn("seconds", request)
        self.assertEqual(request["steps"], 16)
        self.assertEqual(request["instructions"], "落ち着いた声で")

    def test_long_text_passes_a_non_default_duration_scale(self) -> None:
        synth = self.synthesizer(self.settings(IRODORI_TTS_DURATION_SCALE="1.1"))

        synth.synthesize("八王子キャンパスへは八王子駅からスクールバスで来られます。")

        (request,) = self.entries("request")
        self.assertEqual(request["duration_scale"], 1.1)
        self.assertNotIn("seconds", request)

    def test_worker_error_raises_and_the_worker_stays_usable(self) -> None:
        synth = self.synthesizer()

        with self.assertRaisesRegex(RuntimeError, "out_of_range.*too long"):
            synth.synthesize("FAILする文章です。")
        synth.synthesize("次の文章です。")

        self.assertEqual(len(self.entries("argv")), 1)
        self.assertEqual(len(self.entries("request")), 2)

    def test_dead_worker_is_restarted_on_the_next_request(self) -> None:
        synth = self.synthesizer()

        with self.assertRaisesRegex(RuntimeError, "exited"):
            synth.synthesize("DIEする文章です。")
        pcm, _media_type, _rate = synth.synthesize("もう一度お願いします。")

        self.assertGreater(len(pcm), 0)
        self.assertEqual(len(self.entries("argv")), 2)

    def test_worker_on_another_gpu_is_refused(self) -> None:
        settings = self.settings(IRODORI_TTS_SPEECHCPP_GPU_UUID="GPU-expected")
        synth = self.synthesizer(settings, gpu_probe=lambda pid: "GPU-other")

        with self.assertRaisesRegex(RuntimeError, "GPU-other.*GPU-expected"):
            synth.load_runtime()
        self.assertFalse(synth.model_loaded())

    def test_worker_on_the_expected_gpu_reports_its_uuid(self) -> None:
        settings = self.settings(IRODORI_TTS_SPEECHCPP_GPU_UUID="GPU-expected")
        synth = self.synthesizer(settings, gpu_probe=lambda pid: "GPU-expected")

        synth.load_runtime()

        self.assertTrue(synth.model_loaded())
        self.assertEqual(synth.runtime_gpu_uuid(), "GPU-expected")

    def test_missing_voice_file_names_the_command_that_builds_it(self) -> None:
        synth = self.synthesizer(self.settings(IRODORI_TTS_SPEECHCPP_VOICE=f"{self.tmp.name}/none.voice.gguf"))

        with self.assertRaisesRegex(FileNotFoundError, "speech voice .*--device cpu"):
            synth.load_runtime()
        self.assertEqual(self.entries("argv"), [])

    def test_concurrent_requests_are_serialized_on_the_worker(self) -> None:
        synth = self.synthesizer()
        errors: list[BaseException] = []

        def speak(index: int) -> None:
            try:
                synth.synthesize(f"{index}番目の質問にお答えします。")
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=speak, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [])
        self.assertEqual(len(self.entries("request")), 6)
        self.assertEqual(len(self.entries("argv")), 1)

    @unittest.skipIf(fastapi is None, "fastapi is not installed")
    def test_healthz_reports_the_backend_and_worker_state(self) -> None:
        from fastapi.testclient import TestClient

        settings = self.settings()
        synth = self.synthesizer(settings, gpu_probe=lambda pid: "GPU-seen")
        client = TestClient(irodori_server.create_app(settings, synth))

        before = client.get("/healthz").json()
        response = client.post("/v1/audio/speech", json={"input": "こんにちは。"})
        after = client.get("/healthz").json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Sample-Rate-Hz"], "16000")
        self.assertEqual(before["backend"], "speechcpp")
        self.assertFalse(before["model_loaded"])
        self.assertTrue(after["model_loaded"])
        self.assertEqual(after["checkpoint"], str(self.model))
        self.assertEqual(after["speechcpp_device"], "Vulkan1")
        self.assertEqual(after["speechcpp_gpu_uuid"], "GPU-seen")
        self.assertFalse(after["watermark"])


if __name__ == "__main__":
    unittest.main()
