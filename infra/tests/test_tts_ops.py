from __future__ import annotations

import http.server
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
MANAGE_SCRIPT = ROOT / "infra" / "tokkio" / "manage_tokkio.sh"
INSTALL_SCRIPT = ROOT / "infra" / "tts" / "install_irodori_tts_service.sh"
# en-tts/ (English IndexTTS) is optional in a checkout; its cases skip without it.
HAS_EN_TTS = (ROOT / "en-tts").is_dir()
TTS_UNITS =tuple(sorted((ROOT / "infra" / "tts").glob("*.service"))) + tuple(
    sorted((ROOT / "en-tts").glob("*.service"))
)
# Keys outside these prefixes (PYTHONPATH, HF_HOME, CUDA_VISIBLE_DEVICES, ...) are read by libraries.
APP_ENV_PREFIXES = ("IRODORI_", "ELEVENLABS_", "INDEXTTS_", "QWEN_TTS_", "TOKKIO_")
ENV_FIRST_CALL_RE = re.compile(r"_env_first\(\s*\w+\s*,((?:\s*\"[A-Z0-9_]+\"\s*,?)+)\)")
BASH = shutil.which("bash") or "/bin/bash"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


endpoints = load_module("check_tokkio_endpoints_ops", ROOT / "infra" / "tokkio" / "check_tokkio_endpoints.py")


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@dataclass
class Unit:
    path: Path
    environment: dict[str, str] = field(default_factory=dict)
    environment_files: list[str] = field(default_factory=list)
    exec_start: list[str] = field(default_factory=list)


def parse_unit(path: Path) -> Unit:
    unit = Unit(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Environment="):
            for assignment in shlex.split(line[len("Environment=") :]):
                key, _, value = assignment.partition("=")
                unit.environment[key] = value
        elif line.startswith("EnvironmentFile="):
            unit.environment_files.append(line[len("EnvironmentFile=") :].lstrip("-"))
        elif line.startswith("ExecStart="):
            unit.exec_start = shlex.split(line[len("ExecStart=") :])
    return unit


def repo_path_for(absolute: str) -> Path:
    parts = Path(absolute).parts
    for start in range(1, len(parts)):
        candidate = ROOT.joinpath(*parts[start:])
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{absolute} has no counterpart under {ROOT}")


def server_source(unit: Unit) -> str:
    return repo_path_for(unit.exec_start[1]).read_text(encoding="utf-8")


def string_literals(source: str) -> set[str]:
    return set(re.findall(r"[\"']([A-Z][A-Z0-9_]*)[\"']", source))


class TtsUnitConsistencyTests(unittest.TestCase):
    def test_units_are_discovered(self) -> None:
        names = {path.name for path in TTS_UNITS}

        self.assertIn("ace-irodori-tts.service", names)
        self.assertIn("ace-elevenlabs-tts.service", names)

    @unittest.skipUnless(HAS_EN_TTS, "en-tts/ not present")
    def test_english_tts_unit_is_discovered(self) -> None:
        self.assertIn("ace-indextts-tts.service", {path.name for path in TTS_UNITS})

    def test_unit_environment_keys_are_read_by_the_server_they_start(self) -> None:
        for unit_path in TTS_UNITS:
            unit = parse_unit(unit_path)
            literals = string_literals(server_source(unit))
            unread = sorted(key for key in unit.environment if key.startswith(APP_ENV_PREFIXES) and key not in literals)

            with self.subTest(unit=unit_path.name):
                self.assertEqual(unread, [])

    def test_unit_environment_does_not_shadow_tokkio_env_file_aliases(self) -> None:
        checked = 0
        for unit_path in TTS_UNITS:
            unit = parse_unit(unit_path)
            if not unit.environment_files:
                continue
            for match in ENV_FIRST_CALL_RE.finditer(server_source(unit)):
                keys = re.findall(r"\"([A-Z0-9_]+)\"", match.group(1))
                aliases = [key for key in keys if key.startswith("TOKKIO_")]
                if not aliases:
                    continue
                checked += 1
                preferred = keys[: keys.index(aliases[0])]
                shadowing = [key for key in preferred if key in unit.environment]

                with self.subTest(unit=unit_path.name, keys=keys):
                    self.assertEqual(
                        shadowing,
                        [],
                        f"{unit_path.name} sets {shadowing}, which hides {aliases} from {unit.environment_files}",
                    )

        self.assertGreater(checked, 0)

    def test_tts_units_listen_on_distinct_ports(self) -> None:
        owners: dict[str, list[str]] = {}
        for unit_path in TTS_UNITS:
            for key, value in parse_unit(unit_path).environment.items():
                if key.endswith("_PORT"):
                    owners.setdefault(value, []).append(unit_path.name)

        duplicates = {port: names for port, names in owners.items() if len(names) > 1}
        self.assertEqual(duplicates, {})

    def test_manage_default_health_urls_use_the_unit_ports(self) -> None:
        script = MANAGE_SCRIPT.read_text(encoding="utf-8")
        cases = (
            ("DEFAULT_IRODORI_TTS_HEALTH_URL", ROOT / "infra" / "tts" / "ace-irodori-tts.service", "IRODORI_TTS_PORT"),
            ("DEFAULT_QWEN_TTS_HEALTH_URL", ROOT / "en-tts" / "ace-indextts-tts.service", "INDEXTTS_TTS_PORT"),
        )
        for variable, unit_path, port_key in cases:
            match = re.search(rf'^{variable}="([^"]+)"', script, re.MULTILINE)
            self.assertIsNotNone(match, variable)

            with self.subTest(variable=variable):
                if unit_path.is_relative_to(ROOT / "en-tts") and not HAS_EN_TTS:
                    self.skipTest("en-tts/ not present")
                self.assertEqual(urlsplit(match.group(1)).port, int(parse_unit(unit_path).environment[port_key]))


class FakeTtsHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _send(self, status: int, body: bytes, headers: dict[str, str], content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        server = self.server
        if self.path == "/healthz":
            server.health_requests += 1
            self._send(server.health_status, json.dumps(server.health_payload).encode(), {}, "application/json")
        elif self.path == "/":
            self._send(200, b"ok", {}, "text/plain")
        else:
            self._send(404, b'{"detail": "not found"}', {}, "application/json")

    def do_POST(self) -> None:
        server = self.server
        length = int(self.headers.get("Content-Length", "0"))
        server.speech_requests.append((self.path, json.loads(self.rfile.read(length))))
        self._send(server.speech_status, server.speech_body, server.speech_headers, "audio/L16")


class FakeTtsServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeTtsHandler)
        self.health_status = 200
        self.health_payload: dict[str, object] = {
            "status": "ok",
            "model_loaded": True,
            "reference_source_exists": True,
            "reference_wav_exists": True,
        }
        self.health_requests = 0
        self.speech_status = 200
        self.speech_body = b"\x01\x00" * 1600
        self.speech_headers = {"X-Sample-Rate-Hz": "16000"}
        self.speech_requests: list[tuple[str, object]] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class CheckTokkioEndpointsTtsTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.server = FakeTtsServer()
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def run_main(self, *args: str) -> tuple[int, list[dict[str, object]]]:
        output = io.StringIO()
        with redirect_stdout(output):
            returncode = endpoints.main(["--timeout", "5", *args])
        return returncode, json.loads(output.getvalue())

    def checks_by_name(self, checks: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        return {str(check["name"]): check for check in checks}

    def test_tts_check_passes_for_healthy_16k_stream_and_posts_prewarmed_phrase(self) -> None:
        returncode, checks = self.run_main("--tts-base-url", self.server.base_url + "/")

        self.assertEqual(returncode, 0, checks)
        by_name = self.checks_by_name(checks)
        self.assertTrue(by_name["tts_health"]["ok"])
        self.assertTrue(by_name["tts_speech"]["ok"])
        self.assertEqual(by_name["tts_speech"]["sample_rate_hz"], 16000)
        self.assertEqual(by_name["tts_speech"]["bytes"], 3200)
        self.assertEqual(
            self.server.speech_requests,
            [
                (
                    "/v1/audio/speech",
                    {"input": "確認しています", "voice": "kagawa", "response_format": "pcm", "stream": True},
                )
            ],
        )

    def test_tts_check_fails_when_health_status_is_not_ok(self) -> None:
        self.server.health_status = 503
        self.server.health_payload = {"status": "unconfigured"}

        returncode, checks = self.run_main("--tts-base-url", self.server.base_url)

        self.assertEqual(returncode, 1)
        self.assertFalse(self.checks_by_name(checks)["tts_health"]["ok"])

    def test_tts_health_requires_one_of_the_reference_files(self) -> None:
        self.server.health_payload = {
            "status": "ok",
            "reference_source_exists": False,
            "reference_wav_exists": False,
        }
        returncode, checks = self.run_main("--tts-base-url", self.server.base_url, "--tts-health-only")

        self.assertEqual(returncode, 1)
        self.assertFalse(self.checks_by_name(checks)["tts_health"]["ok"])

        # The server converts the source on first use, so a missing wav alone is fine.
        self.server.health_payload = {
            "status": "ok",
            "reference_source_exists": True,
            "reference_wav_exists": False,
        }
        returncode, checks = self.run_main("--tts-base-url", self.server.base_url, "--tts-health-only")

        self.assertEqual(returncode, 0, checks)

    def test_tts_speech_check_rejects_bad_responses(self) -> None:
        cases = {
            "http_500": (500, b'{"detail": "reference audio not found"}', {"X-Sample-Rate-Hz": "16000"}),
            "empty_body": (200, b"", {"X-Sample-Rate-Hz": "16000"}),
            "odd_length_body": (200, b"\x01\x00\x02", {"X-Sample-Rate-Hz": "16000"}),
            "wrong_sample_rate": (200, b"\x01\x00" * 10, {"X-Sample-Rate-Hz": "22050"}),
            "missing_sample_rate": (200, b"\x01\x00" * 10, {}),
        }
        for label, (status, body, headers) in cases.items():
            self.server.speech_status = status
            self.server.speech_body = body
            self.server.speech_headers = headers

            returncode, checks = self.run_main("--tts-base-url", self.server.base_url)

            with self.subTest(case=label):
                self.assertEqual(returncode, 1)
                by_name = self.checks_by_name(checks)
                self.assertTrue(by_name["tts_health"]["ok"])
                self.assertFalse(by_name["tts_speech"]["ok"])

    def test_tts_speech_check_accepts_a_configured_sample_rate(self) -> None:
        self.server.speech_headers = {"X-Sample-Rate-Hz": "22050"}

        returncode, checks = self.run_main(
            "--tts-base-url",
            self.server.base_url,
            "--tts-sample-rate-hz",
            "22050",
            "--tts-voice",
            "kagawa_en",
            "--tts-text",
            "Hello.",
        )

        self.assertEqual(returncode, 0, checks)
        self.assertEqual(self.server.speech_requests[0][1]["voice"], "kagawa_en")
        self.assertEqual(self.server.speech_requests[0][1]["input"], "Hello.")

    def test_tts_health_only_skips_synthesis(self) -> None:
        returncode, checks = self.run_main("--tts-base-url", self.server.base_url, "--tts-health-only")

        self.assertEqual(returncode, 0, checks)
        self.assertEqual(self.server.health_requests, 1)
        self.assertEqual(self.server.speech_requests, [])
        self.assertNotIn("tts_speech", self.checks_by_name(checks))

    def test_existing_checks_run_without_tts_and_set_the_exit_code(self) -> None:
        returncode, checks = self.run_main("--ui-url", self.server.base_url + "/")

        self.assertEqual(returncode, 0, checks)
        self.assertEqual([check["name"] for check in checks], ["ui"])
        self.assertEqual(self.server.health_requests, 0)
        self.assertEqual(self.server.speech_requests, [])

        returncode, checks = self.run_main("--ui-url", self.server.base_url + "/missing")

        self.assertEqual(returncode, 1)
        self.assertFalse(checks[0]["ok"])


INSTALL_SHIMS = {
    "sudo": '#!/usr/bin/env bash\necho "sudo $*" >> "${SHIM_LOG}"\nexit 0\n',
    "systemctl": (
        "#!/usr/bin/env bash\n"
        'echo "systemctl $*" >> "${SHIM_LOG}"\n'
        'if [[ "$1" == is-active ]]; then exit "${SHIM_IS_ACTIVE_RC:-3}"; fi\n'
        "exit 0\n"
    ),
    "ffmpeg": (
        "#!/usr/bin/env bash\n"
        'echo "ffmpeg $*" >> "${SHIM_LOG}"\n'
        'out="${*: -1}"\n'
        'mkdir -p "$(dirname "${out}")"\n'
        ': > "${out}"\n'
    ),
}
INSTALL_TOOLS = ("bash", "dirname", "sed", "grep", "head", "tail", "cat", "mkdir", "touch")


class IrodoriInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        tts_dir = self.root / "infra" / "tts"
        tts_dir.mkdir(parents=True)
        self.script = tts_dir / "install_irodori_tts_service.sh"
        shutil.copy2(INSTALL_SCRIPT, self.script)

        self.python = self.root / "Irodori-TTS" / ".venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        write_executable(self.python, "#!/bin/sh\nexit 0\n")
        self.reference_source = self.root / "Irodori-TTS" / "data" / "kagawa_voice.m4a"
        self.reference_source.parent.mkdir(parents=True)
        self.reference_source.write_bytes(b"m4a")
        self.data_root = self.root / "data"
        self.reference_wav = self.data_root / "reference" / "ref.wav"
        (tts_dir / "ace-irodori-tts.service").write_text(
            "[Service]\n"
            f"Environment=IRODORI_TTS_DATA_ROOT={self.data_root}\n"
            f"Environment=IRODORI_TTS_REFERENCE_SOURCE={self.reference_source}\n"
            f"Environment=IRODORI_TTS_REFERENCE_WAV={self.reference_wav}\n"
            f"ExecStart={self.python} {tts_dir / 'irodori_server.py'}\n",
            encoding="utf-8",
        )

        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for name, text in INSTALL_SHIMS.items():
            write_executable(self.bin_dir / name, text)
        self.tools_dir = self.root / "tools"
        self.tools_dir.mkdir()
        for name in INSTALL_TOOLS:
            location = shutil.which(name)
            if location:
                (self.tools_dir / name).symlink_to(location)
        self.log = self.root / "shim.log"

    def run_installer(self, *args: str, path: str | None = None, **extra_env: str) -> subprocess.CompletedProcess[str]:
        self.log.write_text("", encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if not key.startswith(("IRODORI_", "SHIM_"))}
        env.update(
            PATH=path or f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}",
            SHIM_LOG=str(self.log),
            USER=env.get("USER", "tester"),
        )
        env.update(extra_env)
        return subprocess.run(
            [BASH, str(self.script), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def test_installer_prepares_the_reference_named_in_the_unit_and_ignores_env_overrides(self) -> None:
        override_source = self.root / "other.m4a"
        override_source.write_bytes(b"other")
        override_wav = self.root / "override" / "ref.wav"

        result = self.run_installer(
            IRODORI_TTS_REFERENCE_SOURCE=str(override_source),
            IRODORI_TTS_REFERENCE_WAV=str(override_wav),
            IRODORI_TTS_DATA_ROOT=str(self.root / "override-data"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        ffmpeg_calls = [call for call in self.calls() if call.startswith("ffmpeg ")]
        self.assertEqual(len(ffmpeg_calls), 1)
        self.assertIn(f"-i {self.reference_source} ", ffmpeg_calls[0])
        self.assertTrue(ffmpeg_calls[0].endswith(str(self.reference_wav)))
        self.assertTrue(self.reference_wav.is_file())
        self.assertFalse(override_wav.exists())
        self.assertNotIn("override", "\n".join(self.calls()))

    def test_installer_reports_restart_for_an_active_unit_without_restarting_it(self) -> None:
        result = self.run_installer(SHIM_IS_ACTIVE_RC="0")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sudo systemctl daemon-reload", self.calls())
        self.assertFalse([call for call in self.calls() if "restart" in call])
        self.assertIn("sudo systemctl restart ace-irodori-tts.service", result.stdout + result.stderr)

    def test_installer_try_restarts_only_when_asked(self) -> None:
        result = self.run_installer("--restart")

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn("sudo systemctl try-restart ace-irodori-tts.service", calls)
        self.assertLess(
            calls.index("sudo systemctl daemon-reload"),
            calls.index("sudo systemctl try-restart ace-irodori-tts.service"),
        )

    def use_speechcpp_unit(self, *, with_voice: bool) -> tuple[Path, Path, Path]:
        speech_dir = self.root / "speech-cpp"
        speech_dir.mkdir()
        speech = speech_dir / "speech"
        # `speech voice MODEL REF OUT ...` writes OUT; every call is logged like the other shims.
        write_executable(
            speech,
            "#!/bin/sh\n"
            'echo "speech $*" >> "${SHIM_LOG}"\n'
            '[ "$1" = voice ] && : > "$4"\n'
            "exit 0\n",
        )
        model = speech_dir / "Irodori-TTS-859M-v4.1-F16.gguf"
        model.write_bytes(b"gguf")
        voice = self.data_root / "speech-cpp" / "kagawa.voice.gguf"
        if with_voice:
            voice.parent.mkdir(parents=True)
            voice.write_bytes(b"voice")
        unit = self.script.parent / "ace-irodori-tts.service"
        unit.write_text(
            unit.read_text(encoding="utf-8").replace(
                "[Service]\n",
                "[Service]\n"
                "Environment=IRODORI_TTS_BACKEND=speechcpp\n"
                f"Environment=IRODORI_TTS_SPEECHCPP_BIN={speech}\n"
                f"Environment=IRODORI_TTS_SPEECHCPP_MODEL={model}\n"
                f"Environment=IRODORI_TTS_SPEECHCPP_VOICE={voice}\n",
            ),
            encoding="utf-8",
        )
        return speech, model, voice

    def test_installer_builds_a_missing_speechcpp_voice_on_the_cpu_before_installing(self) -> None:
        speech, model, voice = self.use_speechcpp_unit(with_voice=False)

        result = self.run_installer()

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        (build,) = [call for call in calls if call.startswith("speech voice ")]
        self.assertEqual(build.split()[2:5], [str(model), str(self.reference_wav), str(voice)])
        self.assertIn("--device cpu", build)
        self.assertTrue(voice.is_file())
        self.assertLess(calls.index(build), next(i for i, c in enumerate(calls) if c.startswith("sudo install -m 0644")))

    def test_installer_keeps_an_existing_speechcpp_voice(self) -> None:
        self.use_speechcpp_unit(with_voice=True)

        result = self.run_installer()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse([call for call in self.calls() if call.startswith("speech ")])

    def test_installer_stops_before_sudo_when_speechcpp_files_are_missing(self) -> None:
        _speech, model, _voice = self.use_speechcpp_unit(with_voice=True)
        model.unlink()

        result = self.run_installer()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(model), result.stderr)
        self.assertFalse([call for call in self.calls() if call.startswith("sudo ")])

    def test_installer_rejects_unknown_arguments(self) -> None:
        result = self.run_installer("--bogus")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_installer_requires_the_unit_venv_python(self) -> None:
        self.python.unlink()

        result = self.run_installer()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[ERROR]", result.stderr)
        self.assertIn(str(self.python), result.stderr)
        self.assertEqual(self.calls(), [])

    def test_installer_requires_ffmpeg_before_changing_anything(self) -> None:
        (self.bin_dir / "ffmpeg").unlink()

        result = self.run_installer(path=f"{self.bin_dir}{os.pathsep}{self.tools_dir}")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[ERROR] ffmpeg", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_installer_does_not_need_ffmpeg_when_the_wav_exists(self) -> None:
        (self.bin_dir / "ffmpeg").unlink()
        self.reference_wav.parent.mkdir(parents=True)
        self.reference_wav.write_bytes(b"wav")

        result = self.run_installer(path=f"{self.bin_dir}{os.pathsep}{self.tools_dir}")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sudo systemctl daemon-reload", self.calls())


CONTROLLER_SOURCE_FILES = (
    "bot.py",
    "config.py",
    "local_rag.py",
    "tokkio_llm.py",
    "tokkio_irodori_tts.py",
    "tokkio_rag.py",
)
CONTROLLER_DEPLOYMENT = "ace-controller-ace-controller-deployment"
MANAGE_SHIMS = {
    "sudo": '#!/usr/bin/env bash\nexec "$@"\n',
    "sleep": "#!/usr/bin/env bash\nexit 0\n",
    "systemctl": r"""#!/usr/bin/env bash
echo "systemctl $*" >> "${SHIM_LOG}"
unit="${*: -1}"
case "$1" in
  is-active)
    if [[ " ${SHIM_ACTIVE_UNITS:-} " == *" ${unit} "* ]]; then
      [[ "$2" == --quiet ]] || echo active
      exit 0
    fi
    [[ "$2" == --quiet ]] || echo inactive
    exit 3
    ;;
  start)
    if [[ " ${SHIM_FAILING_STARTS:-} " == *" ${unit} "* ]]; then
      exit 5
    fi
    ;;
esac
exit 0
""",
    "curl": r"""#!/usr/bin/env bash
echo "curl $*" >> "${SHIM_LOG}"
url="${*: -1}"
for pattern in ${SHIM_CURL_FAIL:-}; do
  if [[ "${url}" == *"${pattern}"* ]]; then
    exit 22
  fi
done
exit 0
""",
    "kubectl": r"""#!/usr/bin/env bash
echo "kubectl $*" >> "${SHIM_LOG}"
case "$1" in
  get)
    if [[ "$2" == deployment,statefulset ]]; then
      printf 'Deployment\tace-controller-ace-controller-deployment\t1\n'
    elif [[ "$2" == pods && "$*" == *custom-columns* ]]; then
      echo ace-controller-ace-controller-deployment-0
    elif [[ "$2" == pods && "$*" == *--no-headers* ]]; then
      # SHIM_RUNNING_PODS are live; SHIM_TERMINATED_PODS are Failed/Succeeded
      # leftovers that a phase field selector filters out.
      [[ -z "${SHIM_RUNNING_PODS:-}" ]] || printf '%s\n' "${SHIM_RUNNING_PODS}"
      if [[ -n "${SHIM_TERMINATED_PODS:-}" && "$*" != *"--field-selector=status.phase!=Succeeded,status.phase!=Failed"* ]]; then
        printf '%s\n' "${SHIM_TERMINATED_PODS}"
      fi
    fi
    ;;
  exec)
    if [[ "$*" == *" sha256sum "* ]]; then
      target="${*: -1}"
      name="${target##*/}"
      if [[ "${name}" == config.yaml ]]; then
        local_file="${SHIM_SOURCE_DIR}/configs/${name}"
      else
        local_file="${SHIM_SOURCE_DIR}/src/${name}"
      fi
      hash="$(sha256sum "${local_file}")"
      echo "${hash%% *}  ${target}"
    fi
    ;;
esac
exit 0
""",
    # The first SHIM_ENDPOINT_FAILS endpoint checks fail; later ones exit with SHIM_ENDPOINT_RC.
    "python3": r"""#!/usr/bin/env bash
echo "python3 $*" >> "${SHIM_LOG}"
if [[ "$1" == *check_tokkio_endpoints.py ]]; then
  count_file="${SHIM_LOG}.endpoint-calls"
  count=$(( $(cat "${count_file}" 2>/dev/null || echo 0) + 1 ))
  echo "${count}" > "${count_file}"
  if (( count <= ${SHIM_ENDPOINT_FAILS:-0} )); then
    exit 1
  fi
  exit "${SHIM_ENDPOINT_RC:-0}"
fi
exit 0
""",
}


class ManageTokkioTtsLifecycleTests(unittest.TestCase):
    """Runs a sandbox copy of manage_tokkio.sh with PATH shims; nothing touches the real host."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        tokkio_dir = self.root / "infra" / "tokkio"
        tokkio_dir.mkdir(parents=True)
        self.script = tokkio_dir / "manage_tokkio.sh"
        shutil.copy2(MANAGE_SCRIPT, self.script)
        self.env_file = tokkio_dir / ".env"

        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for name, text in MANAGE_SHIMS.items():
            write_executable(self.bin_dir / name, text)

        self.source_dir = self.root / "controller-src"
        (self.source_dir / "src").mkdir(parents=True)
        (self.source_dir / "configs").mkdir()
        for name in CONTROLLER_SOURCE_FILES:
            (self.source_dir / "src" / name).write_text(f"# {name}\n", encoding="utf-8")
        (self.source_dir / "configs" / "config.yaml").write_text("Pipeline: {}\n", encoding="utf-8")

        self.workspace = self.root / "ws"
        generated = self.workspace / "controller" / "generated"
        generated.mkdir(parents=True)
        (generated / "app-workload-replicas.tsv").write_text(f"Deployment\t{CONTROLLER_DEPLOYMENT}\t1\n", encoding="utf-8")
        self.log = self.root / "shim.log"

    def run_manage(self, command: str, *env_lines: str, **shim_env: str) -> subprocess.CompletedProcess[str]:
        lines = [
            f"TOKKIO_WORKSPACE_DIR={self.workspace}",
            f"TOKKIO_CONTROLLER_SOURCE_DIR={self.source_dir}",
            "TOKKIO_APP_HOST_IPV4_ADDR=10.0.0.42",
            "TOKKIO_IRODORI_TTS_WAIT_ATTEMPTS=3",
            "TOKKIO_IRODORI_TTS_WAIT_SECONDS=0",
            "TOKKIO_QWEN_TTS_WAIT_ATTEMPTS=3",
            "TOKKIO_QWEN_TTS_WAIT_SECONDS=0",
            "TOKKIO_APP_WORKLOAD_WAIT_ATTEMPTS=1",
            "TOKKIO_APP_WORKLOAD_WAIT_SECONDS=0",
            "TOKKIO_ENDPOINT_WAIT_ATTEMPTS=3",
            "TOKKIO_ENDPOINT_WAIT_SECONDS=0",
            "TOKKIO_RAG_ENABLED=false",
            *env_lines,
        ]
        self.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log.write_text("", encoding="utf-8")
        Path(f"{self.log}.endpoint-calls").unlink(missing_ok=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith(("TOKKIO_", "SHIM_"))}
        env.update(
            PATH=f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}",
            SHIM_LOG=str(self.log),
            SHIM_SOURCE_DIR=str(self.source_dir),
        )
        env.update(shim_env)
        return subprocess.run(
            [BASH, str(self.script), command, "--env-file", str(self.env_file)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()

    def endpoint_check_calls(self) -> list[str]:
        return [call for call in self.calls() if "check_tokkio_endpoints.py" in call]

    def endpoint_check_call(self) -> str:
        # The full check from show_status; only it passes --kubectl.
        matches = [call for call in self.endpoint_check_calls() if "--kubectl" in call]
        self.assertEqual(len(matches), 1, self.calls())
        return matches[0]

    def tts_check_calls(self) -> list[str]:
        return [call for call in self.endpoint_check_calls() if "--tts-base-url" in call]

    def write_deploy_stub(self) -> None:
        write_executable(
            self.script.parent / "deploy_tokkio.sh",
            '#!/usr/bin/env bash\necho "deploy_tokkio.sh $*" >> "${SHIM_LOG}"\n',
        )

    def assert_workloads_restored(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertIn(f"Restoring deployment/{CONTROLLER_DEPLOYMENT} to 1 replica(s)", result.stdout)
        self.assertIn(f"kubectl scale deployment/{CONTROLLER_DEPLOYMENT} -n app --replicas=1", self.calls())
        self.assertIn("ace-controller source/config sync complete", result.stdout)
        self.assertIn("Verification URLs", result.stdout)

    def test_start_succeeds_and_checks_the_controller_facing_irodori_url(self) -> None:
        result = self.run_manage("start")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_workloads_restored(result)
        check = self.endpoint_check_call()
        self.assertIn("--tts-base-url http://10.0.0.42:8021", check)
        self.assertNotIn("--tts-health-only", check)

    def test_start_restores_workloads_when_irodori_health_times_out(self) -> None:
        result = self.run_manage("start", SHIM_CURL_FAIL=":8021/healthz")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Timed out waiting for Irodori TTS health", result.stderr)
        self.assert_workloads_restored(result)

    def assert_stop_reached_tts_and_services(self) -> None:
        calls = self.calls()
        self.assertIn("systemctl stop ace-irodori-tts.service", calls)
        self.assertIn("systemctl stop nginx coturn kubelet containerd", calls)

    def test_stop_ignores_terminated_leftover_pods(self) -> None:
        # Old ReplicaSet pods in Error / ContainerStatusUnknown stay in the namespace
        # forever; they must not make the scale-down wait time out.
        result = self.run_manage(
            "stop",
            SHIM_TERMINATED_PODS="a2f-a2f-deployment-old   0/1   Error   0   91d",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("All app workloads in namespace app are stopped.", result.stdout)
        self.assert_stop_reached_tts_and_services()

    def test_stop_still_stops_tts_and_services_when_pods_keep_running(self) -> None:
        result = self.run_manage(
            "stop",
            SHIM_RUNNING_PODS="ace-controller-ace-controller-deployment-0   1/1   Running   0   1h",
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Timed out waiting for pods in namespace app to stop.", result.stderr)
        self.assertIn("stop finished with failures", result.stderr)
        self.assert_stop_reached_tts_and_services()

    def test_restart_restores_workloads_when_irodori_health_times_out(self) -> None:
        result = self.run_manage("restart", SHIM_CURL_FAIL=":8021/healthz")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assert_workloads_restored(result)
        calls = self.calls()
        self.assertLess(
            calls.index("kubectl scale deployment --all -n app --replicas=0"),
            calls.index("systemctl start ace-irodori-tts.service"),
        )
        self.assertLess(
            calls.index("systemctl start ace-irodori-tts.service"),
            calls.index(f"kubectl scale deployment/{CONTROLLER_DEPLOYMENT} -n app --replicas=1"),
        )

    def test_start_skips_the_health_wait_when_the_tts_unit_does_not_start(self) -> None:
        result = self.run_manage("start", SHIM_FAILING_STARTS="ace-irodori-tts.service")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn("Waiting for Irodori TTS health", result.stdout)
        self.assertIn("ace-irodori-tts.service", result.stderr)
        self.assert_workloads_restored(result)

    def test_start_restores_workloads_when_required_rag_is_unavailable(self) -> None:
        result = self.run_manage(
            "start",
            "TOKKIO_RAG_ENABLED=true",
            "TOKKIO_RAG_PROVIDER=nvidia",
            "TOKKIO_RAG_MODE=always",
            "TOKKIO_RAG_WAIT_ATTEMPTS=1",
            "TOKKIO_RAG_WAIT_SECONDS=0",
            SHIM_CURL_FAIL=":8081/",
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Timed out waiting for RAG health", result.stderr)
        self.assert_workloads_restored(result)

    def test_status_finishes_and_fails_when_an_endpoint_check_fails(self) -> None:
        result = self.run_manage("status", SHIM_ENDPOINT_RC="1")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Verification URLs", result.stdout)

    def test_status_checks_endpoints_once_without_waiting(self) -> None:
        result = self.run_manage("status", SHIM_ENDPOINT_FAILS="1")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(self.endpoint_check_calls()), 1, self.calls())
        self.assertNotIn("Waiting for Tokkio endpoints", result.stdout)

    def test_start_and_restart_wait_for_endpoints_then_check_tts_once(self) -> None:
        for command in ("start", "restart"):
            result = self.run_manage(command, SHIM_ENDPOINT_FAILS="2")

            with self.subTest(command=command):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assert_workloads_restored(result)
                self.assertIn("Waiting for Tokkio endpoints (2/3)", result.stdout)
                checks = self.endpoint_check_calls()
                # Three wait attempts (ui/api/grafana only), then the full check from show_status.
                self.assertEqual(len(checks), 4, checks)
                for check in checks[:-1]:
                    self.assertIn("--api-url", check)
                    self.assertNotIn("--kubectl", check)
                    self.assertNotIn("--tts-base-url", check)
                self.assertEqual(self.tts_check_calls(), [checks[-1]])

    def test_start_and_restart_fail_when_endpoints_stay_down(self) -> None:
        for command in ("start", "restart"):
            result = self.run_manage(command, SHIM_ENDPOINT_RC="1")

            with self.subTest(command=command):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assert_workloads_restored(result)
                self.assertIn("Timed out waiting for Tokkio endpoints", result.stderr)
                errors = [line for line in result.stderr.splitlines() if line.startswith("[ERROR]")]
                self.assertEqual(len(errors), 1, result.stderr)
                self.assertIn(f"{command} finished with failures", errors[0])
                self.assertIn("[WARN]", errors[0])
                self.assertIn('"ok": false', errors[0])
                self.assertEqual(len(self.endpoint_check_calls()), 4, self.calls())
                # The synthesis request (billed with ElevenLabs) is not repeated per attempt.
                self.assertEqual(len(self.tts_check_calls()), 1, self.calls())

    def test_reapply_waits_for_endpoints_then_checks_tts_once(self) -> None:
        self.write_deploy_stub()

        result = self.run_manage("reapply", SHIM_ENDPOINT_FAILS="1")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"deploy_tokkio.sh install --env-file {self.env_file}", self.calls())
        self.assertIn("Waiting for Tokkio endpoints (1/3)", result.stdout)
        self.assertEqual(len(self.endpoint_check_calls()), 3, self.calls())
        self.assertEqual(len(self.tts_check_calls()), 1, self.calls())

    def test_reapply_reports_failed_endpoint_checks(self) -> None:
        self.write_deploy_stub()

        result = self.run_manage("reapply", SHIM_ENDPOINT_RC="1")

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Verification URLs", result.stdout)
        errors = [line for line in result.stderr.splitlines() if line.startswith("[ERROR]")]
        self.assertEqual(len(errors), 1, result.stderr)
        self.assertIn("reapply finished with failures", errors[0])
        self.assertIn('"ok": false', errors[0])
        self.assertEqual(len(self.tts_check_calls()), 1, self.calls())

    def test_status_passes_the_configured_tts_url_and_health_only_flag(self) -> None:
        result = self.run_manage(
            "status",
            "TOKKIO_IRODORI_TTS_BASE_URL=http://10.0.0.7:8022",
            "TOKKIO_TTS_SYNTHESIS_CHECK_ENABLED=false",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        check = self.endpoint_check_call()
        self.assertIn("--tts-base-url http://10.0.0.7:8022", check)
        self.assertIn("--tts-health-only", check)

    def test_status_skips_the_tts_endpoint_check_in_english_mode(self) -> None:
        result = self.run_manage(
            "status",
            "TOKKIO_TTS_PROCESSOR=QwenTTSService",
            "TOKKIO_QWEN_TTS_ENABLED=true",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("--tts-base-url", self.endpoint_check_call())

    def test_start_manages_english_tts_when_processor_is_qwen_and_flag_is_unset(self) -> None:
        result = self.run_manage(
            "start",
            "TOKKIO_TTS_PROCESSOR=QwenTTSService",
            "TOKKIO_IRODORI_TTS_ENABLED=false",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("systemctl start ace-indextts-tts.service", self.calls())
        self.assertIn("English TTS health check passed", result.stdout)

    def test_start_rejects_qwen_processor_with_english_tts_lifecycle_disabled(self) -> None:
        for command in ("start", "restart"):
            result = self.run_manage(
                command,
                "TOKKIO_TTS_PROCESSOR=QwenTTSService",
                "TOKKIO_QWEN_TTS_ENABLED=false",
            )

            with self.subTest(command=command):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("TOKKIO_QWEN_TTS_ENABLED", result.stderr)
                self.assertIn("TOKKIO_TTS_PROCESSOR=QwenTTSService", result.stderr)
                self.assertFalse([call for call in self.calls() if call.startswith(("systemctl start", "kubectl scale"))])

    def test_stop_still_works_with_qwen_processor_and_lifecycle_disabled(self) -> None:
        result = self.run_manage(
            "stop",
            "TOKKIO_TTS_PROCESSOR=QwenTTSService",
            "TOKKIO_QWEN_TTS_ENABLED=false",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("kubectl scale deployment --all -n app --replicas=0", self.calls())

    def test_status_warns_when_english_tts_holds_gpu1_in_japanese_mode(self) -> None:
        result = self.run_manage("status", SHIM_ACTIVE_UNITS="ace-indextts-tts.service")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ace-indextts-tts.service is active", result.stderr)
        self.assertIn("GPU1", result.stderr)

        result = self.run_manage("status")

        self.assertNotIn("ace-indextts-tts.service is active", result.stderr)


if __name__ == "__main__":
    unittest.main()
