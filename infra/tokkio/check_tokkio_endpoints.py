#!/usr/bin/env python3
"""Basic endpoint and pod checks for a Tokkio deployment."""

from __future__ import annotations

import argparse
import json
import ssl
import subprocess
import urllib.error
import urllib.request
from email.message import Message
from urllib.parse import urlsplit, urlunsplit

DEFAULT_TTS_VOICE = "kagawa"
# Prewarmed by ace-irodori-tts.service, so the check is served from the short cache
# instead of running a GPU synthesis next to live speech.
DEFAULT_TTS_TEXT = "確認しています"
DEFAULT_TTS_SAMPLE_RATE_HZ = 16000


def _fetch_url(url: str, insecure: bool, timeout: float) -> dict[str, object]:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "tokkio-check/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return {
                "ok": 200 <= response.status < 400,
                "status": response.status,
                "url": url,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "detail": str(exc), "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc), "url": url}


def _http_fallback_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        return None
    return urlunsplit(("http", parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def _should_retry_over_http(result: dict[str, object]) -> bool:
    detail = str(result.get("detail", ""))
    return "WRONG_VERSION_NUMBER" in detail


def check_url(name: str, url: str, insecure: bool, timeout: float) -> dict[str, object]:
    result = _fetch_url(url, insecure, timeout)
    result["name"] = name
    if result.get("ok"):
        return result

    fallback_url = _http_fallback_url(url)
    if not fallback_url or not _should_retry_over_http(result):
        return result

    fallback = _fetch_url(fallback_url, insecure, timeout)
    fallback["name"] = name
    fallback["requested_url"] = url
    fallback["protocol_mismatch"] = True
    if fallback.get("ok"):
        fallback["detail"] = "HTTPS endpoint responded as plain HTTP"
    return fallback


def check_kubectl(namespace: str, timeout: float) -> dict[str, object]:
    try:
        completed = subprocess.run(  # noqa: S603
            ["kubectl", "get", "pods", "-n", namespace],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "name": "kubectl_pods",
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "detail": (completed.stdout or completed.stderr).strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": "kubectl_pods", "ok": False, "detail": str(exc)}


def _request(
    url: str,
    insecure: bool,
    timeout: float,
    payload: dict[str, object] | None = None,
) -> tuple[int, Message, bytes]:
    context = ssl._create_unverified_context() if insecure else None
    headers = {"User-Agent": "tokkio-check/1.0"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _body_excerpt(body: bytes) -> str:
    return body[:200].decode("utf-8", "replace")


def check_tts_health(base_url: str, insecure: bool, timeout: float) -> dict[str, object]:
    url = base_url.rstrip("/") + "/healthz"
    result: dict[str, object] = {"name": "tts_health", "ok": False, "url": url}
    try:
        status, _headers, body = _request(url, insecure, timeout)
    except Exception as exc:  # noqa: BLE001
        result["detail"] = str(exc)
        return result

    result["status"] = status
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError:
        payload = None
    if status != 200 or not isinstance(payload, dict) or payload.get("status") != "ok":
        result["detail"] = f"expected HTTP 200 with status ok: {_body_excerpt(body)}"
        return result

    # Irodori converts the reference source on first use, so either file is enough.
    reference_keys = [key for key in ("reference_wav_exists", "reference_source_exists") if key in payload]
    if reference_keys and not any(payload[key] for key in reference_keys):
        result["detail"] = "reference audio is missing"
        return result

    result["ok"] = True
    return result


def check_tts_speech(
    base_url: str,
    voice: str,
    text: str,
    sample_rate_hz: int,
    insecure: bool,
    timeout: float,
) -> dict[str, object]:
    url = base_url.rstrip("/") + "/v1/audio/speech"
    result: dict[str, object] = {"name": "tts_speech", "ok": False, "url": url}
    payload = {"input": text, "voice": voice, "response_format": "pcm", "stream": True}
    try:
        status, headers, body = _request(url, insecure, timeout, payload)
    except Exception as exc:  # noqa: BLE001
        result["detail"] = str(exc)
        return result

    raw_rate = headers.get("X-Sample-Rate-Hz")
    result["status"] = status
    result["bytes"] = len(body)
    result["sample_rate_hz"] = int(raw_rate) if raw_rate and raw_rate.isdigit() else raw_rate
    if status != 200:
        result["detail"] = f"HTTP {status}: {_body_excerpt(body)}"
    elif not body or len(body) % 2:
        result["detail"] = "expected a non-empty, even-length s16le PCM body"
    elif result["sample_rate_hz"] != sample_rate_hz:
        result["detail"] = f"expected X-Sample-Rate-Hz {sample_rate_hz}, got {raw_rate}"
    else:
        result["ok"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokkio endpoint checks")
    parser.add_argument("--ui-url", help="Tokkio UI endpoint, for example https://<ip>:30111")
    parser.add_argument("--api-url", help="Tokkio API endpoint, for example http://<ip>:30888")
    parser.add_argument("--grafana-url", help="Grafana endpoint, for example http://<ip>:32300")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--kubectl", action="store_true", help="Also run kubectl get pods")
    parser.add_argument("--namespace", default="app", help="Kubernetes namespace for --kubectl")
    parser.add_argument(
        "--tts-base-url",
        help="TTS base URL, for example http://<ip>:8021; checks /healthz and a short /v1/audio/speech stream",
    )
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE, help="Voice for the TTS synthesis check")
    parser.add_argument("--tts-text", default=DEFAULT_TTS_TEXT, help="Text for the TTS synthesis check")
    parser.add_argument(
        "--tts-sample-rate-hz",
        type=int,
        default=DEFAULT_TTS_SAMPLE_RATE_HZ,
        help="Expected X-Sample-Rate-Hz of the TTS stream",
    )
    parser.add_argument(
        "--tts-health-only",
        action="store_true",
        help="Skip the TTS synthesis request and check /healthz only",
    )
    args = parser.parse_args(argv)

    checks: list[dict[str, object]] = []
    if args.ui_url:
        checks.append(check_url("ui", args.ui_url, args.insecure, args.timeout))
    if args.api_url:
        # API root returns 404 by design; /health is the liveness endpoint
        checks.append(check_url("api", args.api_url.rstrip("/") + "/health", args.insecure, args.timeout))
    if args.grafana_url:
        checks.append(check_url("grafana", args.grafana_url, args.insecure, args.timeout))
    if args.kubectl:
        checks.append(check_kubectl(args.namespace, args.timeout))
    if args.tts_base_url:
        checks.append(check_tts_health(args.tts_base_url, args.insecure, args.timeout))
        if not args.tts_health_only:
            checks.append(
                check_tts_speech(
                    args.tts_base_url,
                    args.tts_voice,
                    args.tts_text,
                    args.tts_sample_rate_hz,
                    args.insecure,
                    args.timeout,
                )
            )

    print(json.dumps(checks, ensure_ascii=True, indent=2))
    return 0 if all(check.get("ok") for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
